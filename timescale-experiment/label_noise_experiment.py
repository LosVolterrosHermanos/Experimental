#!/usr/bin/env python
"""
Label Noise Experiment for Momentum Strategy Comparison.

This script trains Mixture of Experts PLRF models using different optimizers
(Tanea with different momentum flavors, TarMSProp-SGD, Adam) with added label noise
to help distinguish between momentum strategies. It generates learning curves
comparison across optimizers with student-t distributed noise added to labels.
"""
import jax
import jax.numpy as jnp
import jax.random as random
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import NamedTuple, Callable, Dict, List, Union, Optional, Tuple
import numpy as np

from power_law_rf.optimizers import powerlaw_schedule, dana_optimizer, tanea_optimizer, TaneaOptimizerState

# Import MoE PLRF implementations from installed package
from power_law_rf.moe_plrf.moe_plrf import (
    TwoExpertPLRF,
    MoEPLRFTrainer,
    MixtureOfExpertsPLRF
)
from power_law_rf.moe_plrf.moe_plrf_ode import (
    ode_moe_dana_log_implicit,
    MoEODEInputs
)
from power_law_rf.ode import DanaHparams

class TaneaHparams(NamedTuple):
    g2: Callable[[Union[float, jnp.ndarray]], float]  # learning rate function
    g3: Callable[[Union[float, jnp.ndarray]], float]  # learning rate function
    delta: Callable[[Union[float, jnp.ndarray]], float]  # momentum function

import power_law_rf.deterministic_equivalent as theory

# Set random seed
key = random.PRNGKey(42)

# Model parameters
ALPHA = 1.0
BETA_LIST = [-0.3, 0.0, 0.8]
V = 2000  # Hidden dimension
D = 500   # Parameter dimension

# MoE parameters
M = 100  # Number of experts for general MoE
ZETA = 2.0  # Power-law exponent for expert selection

# Training parameters
STEPS = 5000
DT = 1e-3
G2_SCALE = 0.2
G3_OVER_G2 = 0.01
BATCH_SIZE = 100
TANEA_LR_SCALAR = 1E-2
TANEA_GLOBAL_EXPONENT = 0.0

# Label noise parameters
STUDENT_T_DOF = 6.0  # Degrees of freedom for student-t distribution
SIGMA = 0.0  # Scaling factor for the noise


def compute_tau_order_statistics(tau_vector):
    """Compute order statistics for tau vector in a jittable way.
    
    Args:
        tau_vector: A 1D array of non-negative tau values
        
    Returns:
        Tuple of (largest_order_stats, smallest_order_stats) where:
        - largest_order_stats: [largest, (1.1)^1-th largest, (1.1)^2-th largest, ...]
        - smallest_order_stats: [smallest, (1.1)^1-th smallest, (1.1)^2-th smallest, ...]
        where we take the (1.1)^k-th for k = 0, 1, 2, ..., up to n
    """
    n = len(tau_vector)
    if n == 0:
        return jnp.array([]), jnp.array([])
    
    # Sort in descending order for largest stats
    sorted_tau_desc = jnp.sort(tau_vector)[::-1]
    
    # Compute powers of 1.1 up to n, similar to evaluation times
    max_k = jnp.ceil(jnp.log(n) / jnp.log(1.1)).astype(jnp.int32)
    indices = jnp.int32(1.1 ** jnp.arange(max_k + 1)) - 1  # 0-indexed: [0, 0, 1, 2, 3, 4, ...]
    
    # Remove duplicates and clamp to valid range
    indices = jnp.unique(indices)
    indices = jnp.minimum(indices, n - 1)
    
    # Get largest order statistics (same as before)
    largest_order_stats = sorted_tau_desc[indices]
    
    # Get smallest order statistics using reversed indices
    # For smallest: indices from the end of the sorted array
    reversed_indices = (n - 1) - indices
    smallest_order_stats = sorted_tau_desc[reversed_indices]
    
    return largest_order_stats, smallest_order_stats


def extract_tau_statistics(opt_state):
    """Extract tau statistics from TaneaOptimizerState.
    
    Args:
        opt_state: TaneaOptimizerState containing tau tree
        
    Returns:
        Dictionary with tau statistics including both largest and smallest order statistics
    """
    if not isinstance(opt_state, TaneaOptimizerState):
        return {}
    
    # Flatten tau tree into a single vector
    tau_leaves = jax.tree_util.tree_leaves(opt_state.tau)
    tau_vector = jnp.concatenate([jnp.ravel(leaf) for leaf in tau_leaves])
    
    # Compute order statistics (now returns both largest and smallest)
    order_stats, reversed_order_stats = compute_tau_order_statistics(tau_vector)
    
    return {
        'tau_order_statistics': order_stats,
        'tau_reversed_order_statistics': reversed_order_stats,
        'tau_mean': jnp.mean(tau_vector),
        'tau_std': jnp.std(tau_vector),
        'tau_min': jnp.min(tau_vector),
        'tau_max': jnp.max(tau_vector)
    }


class TauTrackingLabelNoiseMoEPLRFTrainer(MoEPLRFTrainer):
    """Custom MoEPLRFTrainer that tracks tau statistics during training with label noise.
    
    Extends the base MoEPLRFTrainer to collect tau statistics from TaneaOptimizer
    at evaluation steps during training.
    """
    
    def train(self,
              key: random.PRNGKey,
              num_steps: int,
              batch_size: int,
              init_params: Optional[jnp.ndarray] = None,
              eval_freq: Optional[int] = None,
              track_per_expert_loss: bool = False,
              track_update_history: bool = False,
              track_tau_stats: bool = True) -> Dict:
        """Train the MoE model and return training metrics including tau statistics.
        
        Args:
            Same as parent class, plus:
            track_tau_stats: If True, collect tau statistics during eval steps
            
        Returns:
            Same as parent class, plus:
                - tau_statistics: Dictionary with tau stats at each eval step
        """
        if not track_tau_stats:
            return super().train(
                key=key,
                num_steps=num_steps,
                batch_size=batch_size,
                init_params=init_params,
                eval_freq=eval_freq,
                track_per_expert_loss=track_per_expert_loss,
                track_update_history=track_update_history
            )
        
        # Initialize parameters for all experts
        if init_params is None:
            init_params = jnp.zeros((self.model.d, self.model.m))

        params = init_params
        opt_state = self.optimizer.init(params)

        # Determine evaluation times
        if eval_freq is None:
            eval_times = jnp.unique(jnp.concatenate([
                jnp.array([0]),
                jnp.int32(1.1 ** jnp.arange(1, jnp.ceil(jnp.log(num_steps) / jnp.log(1.1)))),
                jnp.array([num_steps])
            ]))
        else:
            eval_times = jnp.arange(0, num_steps + 1, eval_freq)

        # Batch loss function for MoE
        @jax.jit
        def batch_loss_moe(params, X, y, expert_indices):
            """Compute mean squared error loss with expert routing."""
            # Create routing matrix
            R = self.model.create_routing_matrix(expert_indices, batch_size)

            # Compute predictions for all experts
            all_predictions = jnp.matmul(X, params)  # (batch_size, m)

            # Select predictions based on routing
            predictions = jnp.sum(all_predictions * R.T, axis=1)  # (batch_size,)

            # Compute loss
            return jnp.mean(optax.l2_loss(predictions, y))

        # Gradient computation for MoE
        @jax.jit
        def compute_moe_gradients(params, X, y, expert_indices):
            """Compute gradients for MoE model."""
            def loss_fn(params):
                return batch_loss_moe(params, X, y, expert_indices)
            
            return jax.grad(loss_fn)(params)

        # Training step
        @jax.jit
        def train_step(params, opt_state, key):
            """Single training step for MoE."""
            # Split key for data generation
            key_data, key_expert = random.split(key)
            
            # Generate batch with label noise
            X, y = self.model.generate_batch(key_data, batch_size)
            
            # Sample expert indices for this batch
            expert_indices = self.model.sample_expert_batch(key_expert, batch_size)
            
            # Compute gradients
            grads = compute_moe_gradients(params, X, y, expert_indices)
            
            # Update parameters
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            return params, opt_state

        # Training loop with progress bar
        losses = [self.model.population_risk(init_params)]
        timestamps = [0]
        
        # Initialize tau statistics tracking
        tau_statistics = {
            'timestamps': [0],
            'tau_order_statistics': [extract_tau_statistics(opt_state).get('tau_order_statistics', jnp.array([]))],
            'tau_reversed_order_statistics': [extract_tau_statistics(opt_state).get('tau_reversed_order_statistics', jnp.array([]))],
            'tau_mean': [extract_tau_statistics(opt_state).get('tau_mean', 0.0)],
            'tau_std': [extract_tau_statistics(opt_state).get('tau_std', 0.0)],
            'tau_min': [extract_tau_statistics(opt_state).get('tau_min', 0.0)],
            'tau_max': [extract_tau_statistics(opt_state).get('tau_max', 0.0)]
        }

        eval_idx = 1
        next_eval = eval_times[eval_idx] if eval_idx < len(eval_times) else num_steps + 1

        for step in tqdm(range(num_steps)):
            # Split key for this step
            key, subkey = random.split(key)

            # Perform training step
            params, opt_state = train_step(params, opt_state, subkey)

            # Evaluate if needed
            if step + 1 == next_eval:
                pop_risk = self.model.population_risk(params)
                losses.append(pop_risk)
                timestamps.append(step + 1)
                
                # Extract tau statistics
                tau_stats = extract_tau_statistics(opt_state)
                tau_statistics['timestamps'].append(step + 1)
                tau_statistics['tau_order_statistics'].append(tau_stats.get('tau_order_statistics', jnp.array([])))
                tau_statistics['tau_reversed_order_statistics'].append(tau_stats.get('tau_reversed_order_statistics', jnp.array([])))
                tau_statistics['tau_mean'].append(tau_stats.get('tau_mean', 0.0))
                tau_statistics['tau_std'].append(tau_stats.get('tau_std', 0.0))
                tau_statistics['tau_min'].append(tau_stats.get('tau_min', 0.0))
                tau_statistics['tau_max'].append(tau_stats.get('tau_max', 0.0))

                eval_idx += 1
                if eval_idx < len(eval_times):
                    next_eval = eval_times[eval_idx]
                else:
                    next_eval = num_steps + 1

        return {
            'timestamps': jnp.array(timestamps),
            'losses': jnp.array(losses),
            'tau_statistics': tau_statistics
        }


class LabelNoiseMixtureOfExpertsPLRF(MixtureOfExpertsPLRF):
    """Mixture of Experts PLRF with added student-t label noise."""
    
    def __init__(self,
                 alpha: float,
                 beta: float,
                 v: int,
                 d: int,
                 m: int,
                 zeta: float,
                 student_t_dof: float,
                 sigma: float,
                 key: random.PRNGKey):
        """Initialize the MoE PLRF model with label noise parameters.

        Args:
            alpha: Power law exponent for eigenvalue decay
            beta: Power law exponent for target coefficient decay
            v: Hidden dimension (number of random features)
            d: Embedded dimension (parameter dimension)
            m: Number of experts
            zeta: Power law exponent for expert selection (p(i) ∝ i^(-zeta))
            student_t_dof: Degrees of freedom for student-t distribution
            sigma: Scaling factor for the noise
            key: JAX random key
        """
        super().__init__(alpha, beta, v, d, m, zeta, key)
        self.student_t_dof = student_t_dof
        self.sigma = sigma
    
    def generate_batch(self, key: random.PRNGKey, batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate a batch of (X, y) training data with added student-t noise.

        Args:
            key: JAX random key
            batch_size: Number of samples to generate

        Returns:
            X: Input features of shape (batch_size, d)
            y: Target values of shape (batch_size,) with added noise
        """
        # Split key for data generation and noise
        key_data, key_noise = random.split(key)
        
        # Generate random features x ~ N(0, 1)
        x = random.normal(key_data, (batch_size, self.v))

        # Transform to get inputs and targets
        X = jnp.matmul(x, self.checkW)  # (batch_size, d)
        y_clean = jnp.matmul(x, self.checkb)  # (batch_size,)
        
        # Add student-t noise
        # Generate student-t random variables
        noise = random.t(key_noise, df=self.student_t_dof, shape=(batch_size,))
        # Scale by sigma
        y_noisy = y_clean + self.sigma * noise

        return X, y_noisy


def get_traceK(alpha, v):
    """Compute trace of K matrix for hyperparameter scaling."""
    return jnp.sum(jnp.arange(1, v + 1) ** (-alpha))


def get_tanea_hparams(alpha, beta, d, batch_size, g2_scale, g3_over_g2, traceK, tanea_lr_scalar, tanea_global_exponent):
    """Get Tanea hyperparameters."""
    g2 = lambda t: g2_scale * tanea_lr_scalar * (1 + t) ** tanea_global_exponent
    g3 = lambda t: g3_over_g2 * g2(t)
    delta = lambda t: 1.0 / (1 + t)
    return TaneaHparams(g2=g2, g3=g3, delta=delta)


def get_tarmsprop_sgd_hparams(alpha, beta, d, batch_size, g2_scale, traceK, tanea_lr_scalar, tanea_global_exponent):
    """Get TarMSProp-SGD hyperparameters."""
    g2 = lambda t: g2_scale * tanea_lr_scalar * (1 + t) ** tanea_global_exponent
    g3 = lambda t: 0.0  # No momentum term for SGD
    delta = lambda t: 1.0 / (1 + t)
    return TaneaHparams(g2=g2, g3=g3, delta=delta)


def get_adam_lr(alpha, beta, d, batch_size, g2_scale, traceK, tanea_lr_scalar, tanea_global_exponent):
    """Get Adam learning rate."""
    return g2_scale * tanea_lr_scalar * 0.717


# Compute traceK for hyperparameter scaling
traceK = get_traceK(ALPHA, V)

print("="*60)
print("Label Noise Experiment")
print("="*60)
print(f"Model parameters: α={ALPHA}, β={BETA_LIST}, V={V}, D={D}")
print(f"MoE parameters: M={M}, ζ={ZETA}")
print(f"Training parameters: STEPS={STEPS}, BATCH_SIZE={BATCH_SIZE}")
print(f"Label noise parameters: Student-t DOF={STUDENT_T_DOF}, σ={SIGMA}")
print("="*60)

# Main training experiment loop
moe_results = []

for beta in BETA_LIST:
    print(f"\nRunning MoE experiments for β = {beta}")

    # Create MoE model with label noise
    key, model_key = random.split(key)
    model = LabelNoiseMixtureOfExpertsPLRF(
        alpha=ALPHA,
        beta=beta,
        v=V,
        d=D,
        m=M,
        zeta=ZETA,
        student_t_dof=STUDENT_T_DOF,
        sigma=SIGMA,
        key=model_key
    )

    print(f"  Expert probabilities: {model.expert_probs}")
    print(f"  Optimal risk: {model.population_risk(model.optimal_params_per_expert()):.6f}")

    # Create hyperparameters
    tanea_hparams = get_tanea_hparams(ALPHA, beta, D, BATCH_SIZE, G2_SCALE, G3_OVER_G2, traceK, TANEA_LR_SCALAR, TANEA_GLOBAL_EXPONENT)
    tanea_opt = tanea_optimizer(tanea_hparams.g2, tanea_hparams.g3, tanea_hparams.delta)
    tanea_theory_opt = tanea_optimizer(tanea_hparams.g2, tanea_hparams.g3, tanea_hparams.delta, momentum_flavor="theory")
    tanea_always_on_opt = tanea_optimizer(tanea_hparams.g2, tanea_hparams.g3, tanea_hparams.delta, momentum_flavor="always-on")
    tanea_strong_clip_opt = tanea_optimizer(tanea_hparams.g2, tanea_hparams.g3, tanea_hparams.delta, momentum_flavor="strong-clip")
    tanea_first_moment_opt = tanea_optimizer(tanea_hparams.g2, tanea_hparams.g3, tanea_hparams.delta, tau_flavor="first-moment")
    tarmsprop_sgd_hparams = get_tarmsprop_sgd_hparams(ALPHA, beta, D, BATCH_SIZE, G2_SCALE, traceK, TANEA_LR_SCALAR, TANEA_GLOBAL_EXPONENT)
    tarmsprop_sgd_opt = tanea_optimizer(tarmsprop_sgd_hparams.g2, tarmsprop_sgd_hparams.g3, tarmsprop_sgd_hparams.delta)

    adam_opt = optax.adam(get_adam_lr(ALPHA, beta, D, BATCH_SIZE, G2_SCALE, traceK, TANEA_LR_SCALAR, TANEA_GLOBAL_EXPONENT),b1=0.0)

    # Tanea experiment (effective-clip)
    tanea_trainer = TauTrackingLabelNoiseMoEPLRFTrainer(model, tanea_opt)
    key, train_key = random.split(key)
    tanea_results = tanea_trainer.train(
        train_key,
        num_steps=STEPS,
        batch_size=BATCH_SIZE,  
        track_per_expert_loss=True,
        track_tau_stats=True
    )

    # Tanea theory experiment
    tanea_theory_trainer = TauTrackingLabelNoiseMoEPLRFTrainer(model, tanea_theory_opt)
    key, train_key = random.split(key)
    tanea_theory_results = tanea_theory_trainer.train(
        train_key,
        num_steps=STEPS,
        batch_size=BATCH_SIZE,  
        track_per_expert_loss=True,
        track_tau_stats=True
    )

    # Tanea always-on experiment
    tanea_always_on_trainer = TauTrackingLabelNoiseMoEPLRFTrainer(model, tanea_always_on_opt)
    key, train_key = random.split(key)
    tanea_always_on_results = tanea_always_on_trainer.train(
        train_key,
        num_steps=STEPS,
        batch_size=BATCH_SIZE,  
        track_per_expert_loss=True,
        track_tau_stats=True
    )

    # Tanea strong-clip experiment
    tanea_strong_clip_trainer = TauTrackingLabelNoiseMoEPLRFTrainer(model, tanea_strong_clip_opt)
    key, train_key = random.split(key)
    tanea_strong_clip_results = tanea_strong_clip_trainer.train(
        train_key,
        num_steps=STEPS,
        batch_size=BATCH_SIZE,  
        track_per_expert_loss=True,
        track_tau_stats=True
    )

    # Tanea first-moment experiment
    tanea_first_moment_trainer = TauTrackingLabelNoiseMoEPLRFTrainer(model, tanea_first_moment_opt)
    key, train_key = random.split(key)
    tanea_first_moment_results = tanea_first_moment_trainer.train(
        train_key,
        num_steps=STEPS,
        batch_size=BATCH_SIZE,  
        track_per_expert_loss=True,
        track_tau_stats=True
    )

    # Tarmsprop SGD experiment
    tarmsprop_sgd_trainer = MoEPLRFTrainer(model, tarmsprop_sgd_opt)
    key, train_key = random.split(key)
    tarmsprop_sgd_results = tarmsprop_sgd_trainer.train(
        train_key,
        num_steps=STEPS,
        batch_size=BATCH_SIZE,  
        track_per_expert_loss=True
    )

    # Adam experiment
    adam_trainer = MoEPLRFTrainer(model, adam_opt)
    key, train_key = random.split(key)
    adam_results = adam_trainer.train(
        train_key,
        num_steps=STEPS,
        batch_size=BATCH_SIZE,  
        track_per_expert_loss=True
    )

    moe_results.append({
        'beta': beta,
        'model': model,
        'tanea': tanea_results,
        'tanea_theory': tanea_theory_results,
        'tanea_always_on': tanea_always_on_results,
        'tanea_strong_clip': tanea_strong_clip_results,
        'tanea_first_moment': tanea_first_moment_results,
        'tarmsprop_sgd': tarmsprop_sgd_results,
        'adam': adam_results
    })


# Visualization: Learning curves comparison and per-expert loss plots
# Create a combined figure with learning curves on top and per-expert comparisons below
n_beta = len(moe_results)
n_tanea_optimizers = 5  # Number of Tanea-family optimizers
fig_combined, axes_combined = plt.subplots(1 + n_tanea_optimizers, n_beta, figsize=(6 * n_beta, 5 * (1 + n_tanea_optimizers)))
if n_beta == 1:
    axes_combined = axes_combined.reshape(-1, 1)

# Top row: Learning curves
axes_curves = axes_combined[0, :] if n_beta > 1 else [axes_combined[0, 0]]

for i, result in enumerate(moe_results):
    beta = result['beta']
    ax = axes_curves[i]
    
    # Plot each optimizer's learning curve
    optimizer_colors = {'tanea': 'red', 'tanea_theory': 'orange', 'tanea_always_on': 'purple', 'tanea_strong_clip': 'brown', 'tanea_first_moment': 'pink', 'tarmsprop_sgd': 'blue', 'adam': 'green'}
    
    for optimizer_name in ['tanea', 'tanea_theory', 'tanea_always_on', 'tanea_strong_clip', 'tanea_first_moment', 'tarmsprop_sgd', 'adam']:
        if optimizer_name in result and 'timestamps' in result[optimizer_name] and 'losses' in result[optimizer_name]:
            timestamps = result[optimizer_name]['timestamps']
            losses = result[optimizer_name]['losses']
            
            if len(timestamps) > 0 and len(losses) > 0:
                ax.loglog(timestamps, losses, 'o-', 
                         color=optimizer_colors.get(optimizer_name, 'black'), 
                         alpha=0.8, markersize=4, linewidth=2, 
                         label=optimizer_name.upper())
    
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Population Risk')
    ax.set_title(f'Learning Curves with Label Noise\nβ={beta}, M={M}, D={D}, ζ={ZETA}\nStudent-t DOF={STUDENT_T_DOF}, σ={SIGMA}')
    ax.grid(True, alpha=0.3)
    ax.legend()

# Per-expert loss comparison plots (rows 1-5)
tanea_optimizers = ['tanea', 'tanea_theory', 'tanea_always_on', 'tanea_strong_clip', 'tanea_first_moment']
tanea_labels = ['Tanea (Effective-Clip)', 'Tanea (Theory)', 'Tanea (Always-On)', 'Tanea (Strong-Clip)', 'Tanea (First-Moment)']

for i, result in enumerate(moe_results):
    beta = result['beta']
    
    # Get Adam per-expert losses for comparison
    adam_per_expert = result['adam'].get('per_expert_losses', {})
    adam_timestamps = result['adam'].get('timestamps', [])
    
    # Plot each Tanea optimizer vs Adam
    for tanea_idx, (tanea_opt, tanea_label) in enumerate(zip(tanea_optimizers, tanea_labels)):
        ax_expert = axes_combined[1 + tanea_idx, i] if n_beta > 1 else axes_combined[1 + tanea_idx, 0]
        
        if tanea_opt in result and 'per_expert_losses' in result[tanea_opt]:
            tanea_per_expert = result[tanea_opt]['per_expert_losses']
            tanea_timestamps = result[tanea_opt]['timestamps']
            
            # Plot Adam vs Tanea per-expert losses
            for expert_idx in range(min(len(adam_per_expert), len(tanea_per_expert))):
                if expert_idx in adam_per_expert and expert_idx in tanea_per_expert:
                    adam_losses = adam_per_expert[expert_idx]
                    tanea_losses = tanea_per_expert[expert_idx]
                    
                    # Only plot if we have data for both
                    if len(adam_losses) > 0 and len(tanea_losses) > 0:
                        # Use final losses for comparison
                        adam_final = adam_losses[-1]
                        tanea_final = tanea_losses[-1]
                        
                        # Plot point with expert index as label
                        ax_expert.scatter(adam_final, tanea_final, 
                                        alpha=0.7, s=50, 
                                        label=f'Expert {expert_idx}' if expert_idx < 10 else None)
            
            # Add diagonal line for reference (equal performance)
            if len(adam_per_expert) > 0 and len(tanea_per_expert) > 0:
                # Get range for diagonal line
                all_adam_finals = [adam_per_expert[j][-1] for j in adam_per_expert.keys() if len(adam_per_expert[j]) > 0]
                all_tanea_finals = [tanea_per_expert[j][-1] for j in tanea_per_expert.keys() if len(tanea_per_expert[j]) > 0]
                
                if all_adam_finals and all_tanea_finals:
                    min_loss = min(min(all_adam_finals), min(all_tanea_finals))
                    max_loss = max(max(all_adam_finals), max(all_tanea_finals))
                    
                    ax_expert.plot([min_loss, max_loss], [min_loss, max_loss], 
                                 'k--', alpha=0.5, label='Equal Performance')
            
            ax_expert.set_xlabel('Adam Final Loss')
            ax_expert.set_ylabel(f'{tanea_label} Final Loss')
            ax_expert.set_title(f'{tanea_label} vs Adam Per-Expert Losses\nβ={beta}')
            ax_expert.grid(True, alpha=0.3)
            ax_expert.set_xscale('log')
            ax_expert.set_yscale('log')
            if len(adam_per_expert) <= 10:  # Only show legend if not too many experts
                ax_expert.legend(fontsize=8)

plt.tight_layout()

# Save combined figure
beta_str = "_".join([f"beta{beta}" for beta in BETA_LIST])
combined_filename = f"label_noise_combined_analysis_M{M}_D{D}_zeta{ZETA}_dof{STUDENT_T_DOF}_sigma{SIGMA}_{beta_str}_steps{STEPS}.pdf"
combined_filepath = f"./results/{combined_filename}"
plt.savefig(combined_filepath, dpi=300, bbox_inches='tight')
print(f"Combined analysis figure saved to: {combined_filepath}")

plt.show()

# Visualization: Tau order statistics evolution
fig, axes = plt.subplots(2, len(moe_results), figsize=(6 * len(moe_results), 10))
if len(moe_results) == 1:
    axes = axes.reshape(2, 1)

for i, result in enumerate(moe_results):
    beta = result['beta']
    
    # Plot second-moment tau statistics (top row)
    if 'tau_statistics' in result['tanea']:
        tau_times = result['tanea']['tau_statistics']['timestamps']
        tau_order_stats = result['tanea']['tau_statistics']['tau_order_statistics']
        
        # Check if reversed order statistics are available
        tau_reversed_order_stats = result['tanea']['tau_statistics'].get('tau_reversed_order_statistics', None)
        
        if len(tau_order_stats) > 0:
            ax = axes[0, i]  # Top row for second-moment tau
            
            # Create color map for time evolution
            n_timestamps = len(tau_times)
            colors = plt.cm.plasma(np.linspace(0, 0.8, n_timestamps))
            
            # Find overall max and min for y-axis range (include both regular and reversed stats)
            all_order_stats = []
            for order_stats in tau_order_stats:
                if len(order_stats) > 0:
                    all_order_stats.extend(order_stats)
            
            # Also include reversed order statistics if available
            if tau_reversed_order_stats:
                for order_stats in tau_reversed_order_stats:
                    if len(order_stats) > 0:
                        all_order_stats.extend(order_stats)
            
            if all_order_stats:
                max_tau = max(all_order_stats)
                min_tau_plot = max_tau * 1e-5  # 5 orders of magnitude lower
                
                # Plot largest order statistics for each timestamp
                for t_idx, (timestamp, order_stats) in enumerate(zip(tau_times, tau_order_stats)):
                    if len(order_stats) > 0:
                        # k values: 0, 1, 2, ..., max_k where (1.1)^k corresponds to order stat index
                        k_values = np.arange(len(order_stats))
                        
                        # Filter order stats to only show those within our range
                        valid_mask = order_stats >= min_tau_plot
                        if np.any(valid_mask):
                            filtered_k = 1.1**(k_values[valid_mask])
                            filtered_stats = order_stats[valid_mask]
                            
                            ax.scatter(filtered_k, filtered_stats, 
                                     color=colors[t_idx], alpha=0.7, s=20)
                
                # Plot smallest order statistics if available (same color scheme)
                if tau_reversed_order_stats:
                    for t_idx, (timestamp, reversed_order_stats) in enumerate(zip(tau_times, tau_reversed_order_stats)):
                        if len(reversed_order_stats) > 0:
                            # k values: 0, 1, 2, ..., max_k where (1.1)^k corresponds to order stat index
                            k_values = np.arange(len(reversed_order_stats))
                            
                            # Filter order stats to only show those within our range
                            valid_mask = reversed_order_stats >= min_tau_plot
                            if np.any(valid_mask):
                                filtered_k = 1.1**(k_values[valid_mask])
                                filtered_stats = reversed_order_stats[valid_mask]
                                
                                ax.scatter(filtered_k, filtered_stats, 
                                         color=colors[t_idx], alpha=0.7, s=20)
                
                # Set y-axis limits
                ax.set_ylim(min_tau_plot, max_tau * 1.1)
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('k (order statistic index)')
            ax.set_ylabel('τ_k')
            ax.set_title(f'Second-Moment Tau Order Statistics Evolution with Label Noise\nβ={beta}, M={M}, D={D}, ζ={ZETA}\nStudent-t DOF={STUDENT_T_DOF}, σ={SIGMA}')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar to show time evolution
            sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=0.8))
            sm.set_array([])
            
            # Map timestamp indices to [0, 0.8] range
            if n_timestamps > 1:
                time_values = np.linspace(0, 0.8, n_timestamps)
                actual_times = np.array(tau_times)
                cbar = plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.04)
                
                # Set colorbar ticks to show actual iteration numbers
                tick_positions = np.linspace(0, 0.8, min(5, n_timestamps))
                tick_labels = []
                for pos in tick_positions:
                    # Find closest timestamp index
                    idx = int(pos / 0.8 * (n_timestamps - 1))
                    tick_labels.append(f'{actual_times[idx]:.0f}')
                
                cbar.set_ticks(tick_positions)
                cbar.set_ticklabels(tick_labels)
                cbar.set_label('Training Iteration')
    
    # Plot first-moment tau statistics (bottom row)
    if 'tau_statistics' in result['tanea_first_moment']:
        tau_times = result['tanea_first_moment']['tau_statistics']['timestamps']
        tau_order_stats = result['tanea_first_moment']['tau_statistics']['tau_order_statistics']
        
        # Check if reversed order statistics are available
        tau_reversed_order_stats = result['tanea_first_moment']['tau_statistics'].get('tau_reversed_order_statistics', None)
        
        if len(tau_order_stats) > 0:
            ax = axes[1, i]  # Bottom row for first-moment tau
            
            # Create color map for time evolution
            n_timestamps = len(tau_times)
            colors = plt.cm.plasma(np.linspace(0, 0.8, n_timestamps))
            
            # Find overall max and min for y-axis range (include both regular and reversed stats)
            all_order_stats = []
            for order_stats in tau_order_stats:
                if len(order_stats) > 0:
                    all_order_stats.extend(order_stats)
            
            # Also include reversed order statistics if available
            if tau_reversed_order_stats:
                for order_stats in tau_reversed_order_stats:
                    if len(order_stats) > 0:
                        all_order_stats.extend(order_stats)
            
            if all_order_stats:
                max_tau = max(all_order_stats)
                min_tau_plot = max_tau * 1e-5  # 5 orders of magnitude lower
                
                # Plot largest order statistics for each timestamp
                for t_idx, (timestamp, order_stats) in enumerate(zip(tau_times, tau_order_stats)):
                    if len(order_stats) > 0:
                        # k values: 0, 1, 2, ..., max_k where (1.1)^k corresponds to order stat index
                        k_values = np.arange(len(order_stats))
                        
                        # Filter order stats to only show those within our range
                        valid_mask = order_stats >= min_tau_plot
                        if np.any(valid_mask):
                            filtered_k = 1.1**(k_values[valid_mask])
                            filtered_stats = order_stats[valid_mask]
                            
                            ax.scatter(filtered_k, filtered_stats, 
                                     color=colors[t_idx], alpha=0.7, s=20)
                
                # Plot smallest order statistics if available (same color scheme)
                if tau_reversed_order_stats:
                    for t_idx, (timestamp, reversed_order_stats) in enumerate(zip(tau_times, tau_reversed_order_stats)):
                        if len(reversed_order_stats) > 0:
                            # k values: 0, 1, 2, ..., max_k where (1.1)^k corresponds to order stat index
                            k_values = np.arange(len(reversed_order_stats))
                            
                            # Filter order stats to only show those within our range
                            valid_mask = reversed_order_stats >= min_tau_plot
                            if np.any(valid_mask):
                                filtered_k = 1.1**(k_values[valid_mask])
                                filtered_stats = reversed_order_stats[valid_mask]
                                
                                ax.scatter(filtered_k, filtered_stats, 
                                         color=colors[t_idx], alpha=0.7, s=20)
                
                # Set y-axis limits
                ax.set_ylim(min_tau_plot, max_tau * 1.1)
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('k (order statistic index)')
            ax.set_ylabel('τ_k')
            ax.set_title(f'First-Moment Tau Order Statistics Evolution with Label Noise\nβ={beta}, M={M}, D={D}, ζ={ZETA}\nStudent-t DOF={STUDENT_T_DOF}, σ={SIGMA}')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar to show time evolution
            sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=0.8))
            sm.set_array([])
            
            # Map timestamp indices to [0, 0.8] range
            if n_timestamps > 1:
                time_values = np.linspace(0, 0.8, n_timestamps)
                actual_times = np.array(tau_times)
                cbar = plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.04)
                
                # Set colorbar ticks to show actual iteration numbers
                tick_positions = np.linspace(0, 0.8, min(5, n_timestamps))
                tick_labels = []
                for pos in tick_positions:
                    # Find closest timestamp index
                    idx = int(pos / 0.8 * (n_timestamps - 1))
                    tick_labels.append(f'{actual_times[idx]:.0f}')
                
                cbar.set_ticks(tick_positions)
                cbar.set_ticklabels(tick_labels)
                cbar.set_label('Training Iteration')

plt.tight_layout()

# Save the tau figure with descriptive name
beta_str = "_".join([f"beta{beta}" for beta in BETA_LIST])
tau_filename = f"label_noise_tau_order_stats_M{M}_D{D}_zeta{ZETA}_dof{STUDENT_T_DOF}_sigma{SIGMA}_{beta_str}_steps{STEPS}.pdf"
tau_filepath = f"./results/{tau_filename}"
plt.savefig(tau_filepath, dpi=300, bbox_inches='tight')
print(f"Tau statistics figure saved to: {tau_filepath}")

plt.show()

# Summary statistics output
print("\n" + "="*60)
print("Label Noise Experiment Summary")
print("="*60)

for result in moe_results:
    beta = result['beta']
    print(f"\nβ = {beta}")
    
    for optimizer_name in ['tanea', 'tanea_theory', 'tanea_always_on', 'tanea_strong_clip', 'tanea_first_moment', 'tarmsprop_sgd', 'adam']:
        if optimizer_name in result and 'timestamps' in result[optimizer_name] and 'losses' in result[optimizer_name]:
            timestamps = result[optimizer_name]['timestamps']
            losses = result[optimizer_name]['losses']
            
            if len(timestamps) > 0 and len(losses) > 0:
                initial_loss = losses[0]
                final_loss = losses[-1]
                min_loss = jnp.min(losses)
                
                print(f"  {optimizer_name.upper()}:")
                print(f"    Initial loss: {initial_loss:.6f}")
                print(f"    Final loss: {final_loss:.6f}")
                print(f"    Min loss: {min_loss:.6f}")
                print(f"    Total improvement: {initial_loss - final_loss:.6f}")
    
    # Add tau statistics summary for tanea optimizers
    print(f"\n  Tau Statistics Summary:")
    for optimizer_name in ['tanea', 'tanea_theory', 'tanea_always_on', 'tanea_strong_clip', 'tanea_first_moment']:
        if 'tau_statistics' in result[optimizer_name]:
            tau_stats = result[optimizer_name]['tau_statistics']
            if len(tau_stats['timestamps']) > 0:
                final_mean = tau_stats['tau_mean'][-1]
                final_std = tau_stats['tau_std'][-1]
                final_max = tau_stats['tau_max'][-1]
                final_order_stats = tau_stats['tau_order_statistics'][-1]
                
                print(f"    {optimizer_name.upper()}:")
                print(f"      Final tau mean: {final_mean:.6f}")
                print(f"      Final tau std: {final_std:.6f}")
                print(f"      Final tau max: {final_max:.6f}")
                print(f"      Largest order statistics (first 5): {final_order_stats[:5] if len(final_order_stats) > 0 else []}")
                
                # Also show smallest order statistics if available
                if 'tau_reversed_order_statistics' in tau_stats and len(tau_stats['tau_reversed_order_statistics']) > 0:
                    final_reversed_order_stats = tau_stats['tau_reversed_order_statistics'][-1]
                    print(f"      Smallest order statistics (first 5): {final_reversed_order_stats[:5] if len(final_reversed_order_stats) > 0 else []}")

print("\n" + "="*60)
print("Experiment completed!")
print("="*60) 