#!/usr/bin/env python
"""
MoE PLRF Training with Tau Statistics Collection and Visualization.

This script trains Mixture of Experts PLRF models using different optimizers
(Tanea, TarMSProp-SGD, Adam) and collects tau statistics from TaneaOptimizer.
It generates two main plots:
1. Tau order statistics evolution over training
2. Learning curves comparison across optimizers
"""
import jax
import jax.numpy as jnp
import jax.random as random
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import NamedTuple, Callable, Dict, List, Union, Optional
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
BETA_LIST = [0.5, 0.8]
V = 2000  # Hidden dimension
D = 500   # Parameter dimension

# MoE parameters
M = 100  # Number of experts for general MoE
ZETA = 1.0  # Power-law exponent for expert selection

# Training parameters
STEPS = 1000
DT = 1e-3
G2_SCALE = 0.2
G3_OVER_G2 = 0.1
BATCH_SIZE = 100
TANEA_LR_SCALAR = 1E-2
TANEA_GLOBAL_EXPONENT = 0.0


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


class TauTrackingMoEPLRFTrainer(MoEPLRFTrainer):
    """Custom MoEPLRFTrainer that tracks tau statistics during training.
    
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
            """Compute gradients with proper expert routing."""
            # Create routing matrix
            R = self.model.create_routing_matrix(expert_indices, batch_size)

            # Count samples per expert
            samples_per_expert = jnp.sum(R, axis=1)  # (m,)

            # Function to compute loss for gradient
            def loss_fn(params):
                all_predictions = jnp.matmul(X, params)  # (batch_size, m)
                predictions = jnp.sum(all_predictions * R.T, axis=1)
                return jnp.mean(optax.l2_loss(predictions, y))

            # Get gradients
            grads = jax.grad(loss_fn)(params)

            return grads, samples_per_expert

        # Training step
        @jax.jit
        def train_step(params, opt_state, key):
            """Single MoE SGD step."""
            # Split keys
            key_data, key_expert = random.split(key)

            # Generate batch and sample experts
            X, y = self.model.generate_batch(key_data, batch_size)
            expert_indices = self.model.sample_expert_batch(key_expert, batch_size)

            # Compute gradients with routing
            grads, samples_per_expert = compute_moe_gradients(params, X, y, expert_indices)

            # Update parameters
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            # Compute batch loss for monitoring
            batch_loss_val = batch_loss_moe(params, X, y, expert_indices)

            return params, opt_state, batch_loss_val, samples_per_expert

        # Training loop
        losses = [self.model.population_risk(init_params)]
        timestamps = [0]
        expert_update_counts = jnp.zeros(self.model.m)
        expert_sample_counts = jnp.zeros(self.model.m)

        # Optionally track per-expert losses
        if track_per_expert_loss:
            per_expert_losses = {i: [super(MixtureOfExpertsPLRF, self.model).population_risk(init_params[:, i])]
                                for i in range(self.model.m)}

        # Optionally track update history
        if track_update_history:
            update_history = {
                'timestamps': [0],
                'update_counts': [expert_update_counts.copy()],
                'sample_counts': [expert_sample_counts.copy()]
            }

        # Storage for tau statistics
        tau_statistics = {
            'timestamps': [],
            'tau_order_statistics': [],
            'tau_reversed_order_statistics': [],
            'tau_mean': [],
            'tau_std': [],
            'tau_min': [],
            'tau_max': []
        }
        
        # Initial tau statistics
        initial_tau_stats = extract_tau_statistics(opt_state)
        if initial_tau_stats:
            tau_statistics['timestamps'].append(0)
            tau_statistics['tau_order_statistics'].append(initial_tau_stats['tau_order_statistics'])
            tau_statistics['tau_reversed_order_statistics'].append(initial_tau_stats['tau_reversed_order_statistics'])
            tau_statistics['tau_mean'].append(initial_tau_stats['tau_mean'])
            tau_statistics['tau_std'].append(initial_tau_stats['tau_std'])
            tau_statistics['tau_min'].append(initial_tau_stats['tau_min'])
            tau_statistics['tau_max'].append(initial_tau_stats['tau_max'])

        eval_idx = 1
        next_eval = eval_times[eval_idx] if eval_idx < len(eval_times) else num_steps + 1

        for step in tqdm(range(num_steps)):
            # Split key
            key, subkey = random.split(key)

            # Perform training step
            params, opt_state, batch_loss_val, samples_per_expert = train_step(params, opt_state, subkey)

            # Update expert counts
            expert_update_counts += (samples_per_expert > 0).astype(jnp.float32)
            expert_sample_counts += samples_per_expert

            # Evaluate if needed
            if step + 1 == next_eval:
                pop_risk = self.model.population_risk(params)
                losses.append(pop_risk)
                timestamps.append(step + 1)

                if track_per_expert_loss:
                    for i in range(self.model.m):
                        expert_risk = super(MixtureOfExpertsPLRF, self.model).population_risk(params[:, i])
                        per_expert_losses[i].append(expert_risk)

                if track_update_history:
                    update_history['timestamps'].append(step + 1)
                    update_history['update_counts'].append(expert_update_counts.copy())
                    update_history['sample_counts'].append(expert_sample_counts.copy())

                # Collect tau statistics
                tau_stats = extract_tau_statistics(opt_state)
                if tau_stats:
                    tau_statistics['timestamps'].append(step + 1)
                    tau_statistics['tau_order_statistics'].append(tau_stats['tau_order_statistics'])
                    tau_statistics['tau_reversed_order_statistics'].append(tau_stats['tau_reversed_order_statistics'])
                    tau_statistics['tau_mean'].append(tau_stats['tau_mean'])
                    tau_statistics['tau_std'].append(tau_stats['tau_std'])
                    tau_statistics['tau_min'].append(tau_stats['tau_min'])
                    tau_statistics['tau_max'].append(tau_stats['tau_max'])

                eval_idx += 1
                if eval_idx < len(eval_times):
                    next_eval = eval_times[eval_idx]
                else:
                    next_eval = num_steps + 1

        # Prepare results
        results = {
            'timestamps': jnp.array(timestamps),
            'losses': jnp.array(losses),
            'expert_update_counts': expert_update_counts,
            'expert_sample_counts': expert_sample_counts,
            'final_params': params
        }

        if track_per_expert_loss:
            results['per_expert_losses'] = {i: jnp.array(per_expert_losses[i]) for i in range(self.model.m)}

        if track_update_history:
            results['update_history'] = {
                'timestamps': jnp.array(update_history['timestamps']),
                'update_counts': jnp.array(update_history['update_counts']),
                'sample_counts': jnp.array(update_history['sample_counts'])
            }

        # Convert tau statistics lists to arrays
        for key in tau_statistics:
            if key not in ['tau_order_statistics', 'tau_reversed_order_statistics']:
                tau_statistics[key] = jnp.array(tau_statistics[key])

        results['tau_statistics'] = tau_statistics
        
        return results


def get_traceK(alpha, v):
  x_grid = jnp.arange(1, v+1).reshape(1, v)
  population_eigs = x_grid ** -alpha
  population_trace = jnp.sum(population_eigs**2)
  return population_trace

def get_tanea_hparams(alpha, beta, d, batch_size, g2_scale, g3_over_g2, traceK, tanea_lr_scalar, tanea_global_exponent):
  kappa_b = jnp.log(batch_size) / jnp.log(d)  # exponent for batch wrt d
  learning_rate = g2_scale * jnp.minimum(1.0, jnp.float32(batch_size) / traceK)
  tanea_params = TaneaHparams(
    g2=powerlaw_schedule(tanea_lr_scalar*learning_rate, 0.0, -tanea_global_exponent, 1.0),
    g3=powerlaw_schedule(tanea_lr_scalar*learning_rate*g3_over_g2, 0.0, -tanea_global_exponent-(1.0 - kappa_b) / (2 * alpha), 1.0),
    delta=powerlaw_schedule(1.0, 0.0, -1.0, 4.0+2*(alpha+beta)/(2*alpha))
  )
  return tanea_params

def get_tarmsprop_sgd_hparams(alpha, beta, d, batch_size, g2_scale, traceK, tanea_lr_scalar, tanea_global_exponent):
  kappa_b = jnp.log(batch_size) / jnp.log(d)  # exponent for batch wrt d
  learning_rate = g2_scale * jnp.minimum(1.0, jnp.float32(batch_size) / traceK)
  tanea_params = TaneaHparams(
    g2=powerlaw_schedule(tanea_lr_scalar*learning_rate, 0.0, -tanea_global_exponent, 1.0),
    g3=powerlaw_schedule(0.0, 0.0, -tanea_global_exponent-(1.0 - kappa_b) / (2 * alpha), 1.0),
    delta=powerlaw_schedule(1.0, 0.0, -1.0, 4.0+2*(alpha+beta)/(2*alpha))
  )
  return tanea_params

def get_adam_lr(alpha, beta, d, batch_size, g2_scale, traceK, tanea_lr_scalar, tanea_global_exponent):
   return 0.5*g2_scale * jnp.minimum(1.0, jnp.float32(batch_size) / traceK)*tanea_lr_scalar

traceK = get_traceK(ALPHA, V)

#@title Part 4: General MoE with Power-Law Expert Selection and Tau Statistics
"""
Test the general MoE model with power-law expert selection using Tanea optimizer 
and collect tau statistics during training.
"""

print("\n" + "="*60)
print(f"General MoE PLRF: {M} Experts with Power-Law Selection + Tau Stats")
print("="*60)

# Main training experiment loop
moe_results = []

for beta in BETA_LIST:
    print(f"\nRunning MoE experiments for β = {beta}")

    # Create MoE model
    key, model_key = random.split(key)
    model = MixtureOfExpertsPLRF(
        alpha=ALPHA,
        beta=beta,
        v=V,
        d=D,
        m=M,
        zeta=ZETA,
        key=model_key
    )

    print(f"  Expert probabilities: {model.expert_probs}")
    print(f"  Optimal risk: {model.population_risk(model.optimal_params_per_expert()):.6f}")

    # Create hyperparameters
    tanea_hparams = get_tanea_hparams(ALPHA, beta, D, BATCH_SIZE, G2_SCALE, G3_OVER_G2, traceK, TANEA_LR_SCALAR, TANEA_GLOBAL_EXPONENT)
    tanea_opt = tanea_optimizer(tanea_hparams.g2, tanea_hparams.g3, tanea_hparams.delta)
    tarmsprop_sgd_hparams = get_tarmsprop_sgd_hparams(ALPHA, beta, D, BATCH_SIZE, G2_SCALE, traceK, TANEA_LR_SCALAR, TANEA_GLOBAL_EXPONENT)
    tarmsprop_sgd_opt = tanea_optimizer(tarmsprop_sgd_hparams.g2, tarmsprop_sgd_hparams.g3, tarmsprop_sgd_hparams.delta)

    adam_opt = optax.adam(get_adam_lr(ALPHA, beta, D, BATCH_SIZE, G2_SCALE, traceK, TANEA_LR_SCALAR, TANEA_GLOBAL_EXPONENT),b1=0.0)

    # Tanea experiment with tau statistics
    tanea_trainer = TauTrackingMoEPLRFTrainer(model, tanea_opt)
    key, train_key = random.split(key)
    tanea_results = tanea_trainer.train(
        train_key,
        num_steps=STEPS,
        batch_size=BATCH_SIZE,  
        track_per_expert_loss=True,
        track_tau_stats=True
    )

    # Tarmsprop SGD experiment with tau statistics
    tarmsprop_sgd_trainer = TauTrackingMoEPLRFTrainer(model, tarmsprop_sgd_opt)
    key, train_key = random.split(key)
    tarmsprop_sgd_results = tarmsprop_sgd_trainer.train(
        train_key,
        num_steps=STEPS,
        batch_size=BATCH_SIZE,  
        track_per_expert_loss=True,
        track_tau_stats=True
    )

    # Adam experiment (no tau statistics since it's not Tanea)
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
        'tarmsprop_sgd': tarmsprop_sgd_results,
        'adam': adam_results
    })


# Visualization: Tau order statistics evolution
fig, axes = plt.subplots(1, len(moe_results), figsize=(6 * len(moe_results), 5))
if len(moe_results) == 1:
    axes = [axes]

for i, result in enumerate(moe_results):
    beta = result['beta']
    
    if 'tau_statistics' in result['tanea']:
        tau_times = result['tanea']['tau_statistics']['timestamps']
        tau_order_stats = result['tanea']['tau_statistics']['tau_order_statistics']
        
        # Check if reversed order statistics are available
        tau_reversed_order_stats = result['tanea']['tau_statistics'].get('tau_reversed_order_statistics', None)
        
        if len(tau_order_stats) > 0:
            ax = axes[i]
            
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
            ax.set_title(f'Tanea Tau Order Statistics Evolution\nβ={beta}, M={M}, D={D}, ζ={ZETA}')
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

# Save the figure with descriptive name
beta_str = "_".join([f"beta{beta}" for beta in BETA_LIST])
filename = f"tanea_tau_order_stats_M{M}_D{D}_zeta{ZETA}_{beta_str}_steps{STEPS}.pdf"
filepath = f"./results/{filename}"
plt.savefig(filepath, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {filepath}")

plt.show()

# Visualization: Learning curves comparison
fig_curves, axes_curves = plt.subplots(1, len(moe_results), figsize=(6 * len(moe_results), 5))
if len(moe_results) == 1:
    axes_curves = [axes_curves]

for i, result in enumerate(moe_results):
    beta = result['beta']
    ax = axes_curves[i]
    
    # Plot each optimizer's learning curve
    optimizer_colors = {'tanea': 'red', 'tarmsprop_sgd': 'blue', 'adam': 'green'}
    
    for optimizer_name in ['tanea', 'tarmsprop_sgd', 'adam']:
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
    ax.set_title(f'Learning Curves\nβ={beta}, M={M}, D={D}, ζ={ZETA}')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()

# Save learning curves figure
curves_filename = f"learning_curves_M{M}_D{D}_zeta{ZETA}_{beta_str}_steps{STEPS}.pdf"
curves_filepath = f"./results/{curves_filename}"
plt.savefig(curves_filepath, dpi=300, bbox_inches='tight')
print(f"Learning curves figure saved to: {curves_filepath}")

plt.show()

# Summary statistics output
print("\n" + "="*60)
print("Tau Statistics Summary")
print("="*60)

for result in moe_results:
    beta = result['beta']
    print(f"\nβ = {beta}")
    
    for optimizer_name in ['tanea', 'tarmsprop_sgd']:
        if 'tau_statistics' in result[optimizer_name]:
            tau_stats = result[optimizer_name]['tau_statistics']
            if len(tau_stats['timestamps']) > 0:
                final_mean = tau_stats['tau_mean'][-1]
                final_std = tau_stats['tau_std'][-1]
                final_max = tau_stats['tau_max'][-1]
                final_order_stats = tau_stats['tau_order_statistics'][-1]
                
                print(f"  {optimizer_name.upper()}:")
                print(f"    Final tau mean: {final_mean:.6f}")
                print(f"    Final tau std: {final_std:.6f}")
                print(f"    Final tau max: {final_max:.6f}")
                print(f"    Largest order statistics (first 5): {final_order_stats[:5] if len(final_order_stats) > 0 else []}")
                
                # Also show smallest order statistics if available
                if 'tau_reversed_order_statistics' in tau_stats and len(tau_stats['tau_reversed_order_statistics']) > 0:
                    final_reversed_order_stats = tau_stats['tau_reversed_order_statistics'][-1]
                    if len(final_reversed_order_stats) > 0:
                        print(f"    Smallest order statistics (first 5): {final_reversed_order_stats[:5]}")