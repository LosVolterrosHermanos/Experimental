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
import argparse
import os

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


def parse_args():
    """Parse command line arguments for the label noise experiment."""
    parser = argparse.ArgumentParser(description="Label Noise Experiment for Momentum Strategy Comparison")
    
    # Model parameters
    parser.add_argument("--alpha", type=float, default=1.0, help="Power law exponent for eigenvalue decay")
    parser.add_argument("--beta", type=str, default="-0.3,0.0,0.8", help="Comma-separated list of beta values (power law exponent for target coefficient decay)")
    parser.add_argument("--v", type=int, default=2000, help="Hidden dimension (number of random features)")
    parser.add_argument("--d", type=int, default=500, help="Embedded dimension (parameter dimension)")
    
    # MoE parameters
    parser.add_argument("--m", type=int, default=100, help="Number of experts")
    parser.add_argument("--zeta", type=float, default=0.5, help="Power-law exponent for expert selection (p(i) ∝ i^(-zeta))")
    
    # Training parameters
    parser.add_argument("--steps", type=int, default=125000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=100, help="Training batch size")
    parser.add_argument("--g2_scale", type=float, default=0.2, help="Base learning rate scale")
    parser.add_argument("--g3_over_g2", type=float, default=0.01, help="G3 to G2 ratio for momentum")
    parser.add_argument("--tanea_lr_scalar", type=float, default=1e-2, help="Tanea learning rate scalar")
    parser.add_argument("--tanea_global_exponent", type=float, default=0.0, help="Tanea global time exponent")
    
    # Label noise parameters
    parser.add_argument("--student_t_dof", type=float, default=3.0, help="Degrees of freedom for student-t distribution")
    parser.add_argument("--sigma", type=float, default=0.1, help="Scaling factor for the noise")
    
    # Optimizer enable/disable flags
    parser.add_argument("--enable_tanea", action="store_true", default=True, help="Enable Tanea (effective-clip) optimizer")
    parser.add_argument("--disable_tanea", action="store_true", help="Disable Tanea (effective-clip) optimizer")
    parser.add_argument("--enable_tanea_theory", action="store_true", help="Enable Tanea (theory) optimizer")
    parser.add_argument("--enable_tanea_always_on", action="store_true", default=True, help="Enable Tanea (always-on) optimizer")
    parser.add_argument("--disable_tanea_always_on", action="store_true", help="Disable Tanea (always-on) optimizer")
    parser.add_argument("--enable_tanea_strong_clip", action="store_true", help="Enable Tanea (strong-clip) optimizer")
    parser.add_argument("--enable_tanea_first_moment", action="store_true", default=True, help="Enable Tanea (first-moment) optimizer")
    parser.add_argument("--disable_tanea_first_moment", action="store_true", help="Disable Tanea (first-moment) optimizer")
    parser.add_argument("--enable_tanea_g3zero", action="store_true", default=True, help="Enable Tanea G3=0 (formerly TarMSProp-SGD) optimizer")
    parser.add_argument("--disable_tanea_g3zero", action="store_true", help="Disable Tanea G3=0 optimizer")
    parser.add_argument("--enable_rmsprop_dana", action="store_true", default=True, help="Enable RMSprop+Dana optimizer")
    parser.add_argument("--disable_rmsprop_dana", action="store_true", help="Disable RMSprop+Dana optimizer")
    parser.add_argument("--enable_adam", action="store_true", default=True, help="Enable Adam optimizer")
    parser.add_argument("--disable_adam", action="store_true", help="Disable Adam optimizer")
    
    # Output parameters
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to store results")
    parser.add_argument("--output_prefix", type=str, default="label_noise", help="Prefix for output files")
    parser.add_argument("--no_plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    
    return parser.parse_args()


# Set random seed (will be overridden by command line args)
key = random.PRNGKey(42)


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
        
        # Initialize per-expert loss tracking if requested
        per_expert_losses = None
        if track_per_expert_loss:
            per_expert_losses = {i: [super(MixtureOfExpertsPLRF, self.model).population_risk(init_params[:, i])]
                               for i in range(self.model.m)}

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
                
                # Track per-expert losses if requested
                if track_per_expert_loss and per_expert_losses is not None:
                    for i in range(self.model.m):
                        expert_risk = super(MixtureOfExpertsPLRF, self.model).population_risk(params[:, i])
                        per_expert_losses[i].append(expert_risk)

                eval_idx += 1
                if eval_idx < len(eval_times):
                    next_eval = eval_times[eval_idx]
                else:
                    next_eval = num_steps + 1

        # Prepare results
        results = {
            'timestamps': jnp.array(timestamps),
            'losses': jnp.array(losses),
            'tau_statistics': tau_statistics
        }
        
        # Add per-expert losses if tracked
        if track_per_expert_loss and per_expert_losses is not None:
            results['per_expert_losses'] = {i: jnp.array(per_expert_losses[i]) for i in range(self.model.m)}
        
        return results


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
    x_grid = jnp.arange(1, v+1).reshape(1, v)
    population_eigs = x_grid ** -alpha
    population_trace = jnp.sum(population_eigs**2)
    return population_trace


def get_tanea_hparams(alpha, beta, d, batch_size, g2_scale, g3_over_g2, traceK, tanea_lr_scalar, tanea_global_exponent):
    """Get Tanea hyperparameters."""
    kappa_b = jnp.log(batch_size) / jnp.log(d)  # exponent for batch wrt d
    learning_rate = g2_scale * jnp.minimum(1.0, jnp.float32(batch_size) / traceK)
    tanea_params = TaneaHparams(
        g2=powerlaw_schedule(tanea_lr_scalar*learning_rate, 0.0, -tanea_global_exponent, 1.0),
        g3=powerlaw_schedule(tanea_lr_scalar*learning_rate*g3_over_g2, 0.0, -tanea_global_exponent-(1.0 - kappa_b) / (2 * alpha), 1.0),
        delta=powerlaw_schedule(1.0, 0.0, -1.0, 4.0+2*(alpha+beta)/(2*alpha))
    )
    return tanea_params


def get_tarmsprop_sgd_hparams(alpha, beta, d, batch_size, g2_scale, traceK, tanea_lr_scalar, tanea_global_exponent):
    """Get TarMSProp-SGD hyperparameters."""
    kappa_b = jnp.log(batch_size) / jnp.log(d)  # exponent for batch wrt d
    learning_rate = g2_scale * jnp.minimum(1.0, jnp.float32(batch_size) / traceK)
    tanea_params = TaneaHparams(
        g2=powerlaw_schedule(tanea_lr_scalar*learning_rate, 0.0, -tanea_global_exponent, 1.0),
        g3=powerlaw_schedule(0.0, 0.0, -tanea_global_exponent-(1.0 - kappa_b) / (2 * alpha), 1.0),
        delta=powerlaw_schedule(1.0, 0.0, -1.0, 4.0+2*(alpha+beta)/(2*alpha))
    )
    return tanea_params


def get_adam_lr(alpha, beta, d, batch_size, g2_scale, traceK, tanea_lr_scalar, tanea_global_exponent):
    """Get Adam learning rate."""
    return 0.5*g2_scale * jnp.minimum(1.0, jnp.float32(batch_size) / traceK)*tanea_lr_scalar


def get_rmsprop_dana_optimizer(alpha, beta, d, batch_size, g2_scale, g3_over_g2, traceK, tanea_lr_scalar, tanea_global_exponent):
    """Get RMSprop+Dana optimizer with hyperparameters based on Tanea/Adam settings."""
    # Get Adam LR for Dana g2
    adam_lr = get_adam_lr(alpha, beta, d, batch_size, g2_scale, traceK, tanea_lr_scalar, tanea_global_exponent)
    
    # # Get Tanea hyperparameters for kappa (delta parameter)
    # tanea_hparams = get_tanea_hparams(alpha, beta, d, batch_size, g2_scale, g3_over_g2, traceK, tanea_lr_scalar, tanea_global_exponent)
    
    # RMSprop decay (same as Adam beta2)
    rms_decay = 0.999
    
    # Dana parameters
    dana_g2 = adam_lr
    dana_g3 = g3_over_g2 * adam_lr

    kappa_b = jnp.log(batch_size) / jnp.log(d)  # exponent for batch wrt d
    
    # Create Dana optimizer schedules
    g1 = powerlaw_schedule(1.0, 0.0, 0.0, 1)
    g2 = powerlaw_schedule(dana_g2, 0.0, 0.0, 1)
    g3 = powerlaw_schedule(dana_g3, 0.0, -1.0*(1.0 - kappa_b) / (2 * alpha), 1)
    Delta = powerlaw_schedule(1.0, 0.0, -1.0, 4.0+2*(alpha+beta)/(2*alpha))
    
    # Create Dana optimizer
    dana_opt = dana_optimizer(g1=g1, g2=g2, g3=g3, Delta=Delta)
    
    # Chain RMSProp and Dana optimizers
    optimizer = optax.chain(
        optax.scale_by_rms(decay=rms_decay, eps=1e-8, bias_correction=True),
        dana_opt
    )
    
    return optimizer


def main():
    """Main function to run the label noise experiment."""
    args = parse_args()
    
    # Override global random seed
    global key
    key = random.PRNGKey(args.random_seed)
    
    # Parse beta list from string
    beta_list = [float(b.strip()) for b in args.beta.split(',')]
    
    # Process optimizer flags (disable flags override enable flags)
    enable_tanea = args.enable_tanea and not args.disable_tanea
    enable_tanea_theory = args.enable_tanea_theory
    enable_tanea_always_on = args.enable_tanea_always_on and not args.disable_tanea_always_on
    enable_tanea_strong_clip = args.enable_tanea_strong_clip
    enable_tanea_first_moment = args.enable_tanea_first_moment and not args.disable_tanea_first_moment
    enable_tanea_g3zero = args.enable_tanea_g3zero and not args.disable_tanea_g3zero
    enable_rmsprop_dana = args.enable_rmsprop_dana and not args.disable_rmsprop_dana
    enable_adam = args.enable_adam and not args.disable_adam
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Compute traceK for hyperparameter scaling
    traceK = get_traceK(args.alpha, args.v)

    print("="*60)
    print("Label Noise Experiment")
    print("="*60)
    print(f"Model parameters: α={args.alpha}, β={beta_list}, V={args.v}, D={args.d}")
    print(f"MoE parameters: M={args.m}, ζ={args.zeta}")
    print(f"Training parameters: STEPS={args.steps}, BATCH_SIZE={args.batch_size}")
    print(f"Label noise parameters: Student-t DOF={args.student_t_dof}, σ={args.sigma}")
    print(f"Enabled optimizers: Tanea={enable_tanea}, TaneaTheory={enable_tanea_theory}, TaneaAlwaysOn={enable_tanea_always_on}")
    print(f"                   TaneaStrongClip={enable_tanea_strong_clip}, TaneaFirstMoment={enable_tanea_first_moment}")
    print(f"                   TaneaG3Zero={enable_tanea_g3zero}, RMSpropDana={enable_rmsprop_dana}, Adam={enable_adam}")
    print(f"Results directory: {args.results_dir}")
    print("="*60)

    # Main training experiment loop
    moe_results = []

    for beta in beta_list:
        print(f"\nRunning MoE experiments for β = {beta}")

        # Create MoE model with label noise
        key, model_key = random.split(key)
        model = LabelNoiseMixtureOfExpertsPLRF(
            alpha=args.alpha,
            beta=beta,
            v=args.v,
            d=args.d,
            m=args.m,
            zeta=args.zeta,
            student_t_dof=args.student_t_dof,
            sigma=args.sigma,
            key=model_key
        )

        print(f"  Expert probabilities: {model.expert_probs}")
        print(f"  Optimal risk: {model.population_risk(model.optimal_params_per_expert()):.6f}")

        # Create hyperparameters
        optimizers_dict = {}
        
        if enable_tanea:
            tanea_hparams = get_tanea_hparams(args.alpha, beta, args.d, args.batch_size, args.g2_scale, args.g3_over_g2, traceK, args.tanea_lr_scalar, args.tanea_global_exponent)
            optimizers_dict['tanea'] = tanea_optimizer(tanea_hparams.g2, tanea_hparams.g3, tanea_hparams.delta)
        
        if enable_tanea_theory:
            tanea_hparams = get_tanea_hparams(args.alpha, beta, args.d, args.batch_size, args.g2_scale, args.g3_over_g2, traceK, args.tanea_lr_scalar, args.tanea_global_exponent)
            optimizers_dict['tanea_theory'] = tanea_optimizer(tanea_hparams.g2, tanea_hparams.g3, tanea_hparams.delta, momentum_flavor="theory")
        
        if enable_tanea_always_on:
            tanea_hparams = get_tanea_hparams(args.alpha, beta, args.d, args.batch_size, args.g2_scale, args.g3_over_g2, traceK, args.tanea_lr_scalar, args.tanea_global_exponent)
            optimizers_dict['tanea_always_on'] = tanea_optimizer(tanea_hparams.g2, tanea_hparams.g3, tanea_hparams.delta, momentum_flavor="always-on")
        
        if enable_tanea_strong_clip:
            tanea_hparams = get_tanea_hparams(args.alpha, beta, args.d, args.batch_size, args.g2_scale, args.g3_over_g2, traceK, args.tanea_lr_scalar, args.tanea_global_exponent)
            optimizers_dict['tanea_strong_clip'] = tanea_optimizer(tanea_hparams.g2, tanea_hparams.g3, tanea_hparams.delta, momentum_flavor="strong-clip")
        
        if enable_tanea_first_moment:
            tanea_hparams = get_tanea_hparams(args.alpha, beta, args.d, args.batch_size, args.g2_scale, args.g3_over_g2, traceK, args.tanea_lr_scalar, args.tanea_global_exponent)
            optimizers_dict['tanea_first_moment'] = tanea_optimizer(tanea_hparams.g2, tanea_hparams.g3, tanea_hparams.delta, tau_flavor="first-moment")
        
        if enable_tanea_g3zero:
            tanea_g3zero_hparams = get_tarmsprop_sgd_hparams(args.alpha, beta, args.d, args.batch_size, args.g2_scale, traceK, args.tanea_lr_scalar, args.tanea_global_exponent)
            optimizers_dict['tanea_g3zero'] = tanea_optimizer(tanea_g3zero_hparams.g2, tanea_g3zero_hparams.g3, tanea_g3zero_hparams.delta)
        
        if enable_rmsprop_dana:
            optimizers_dict['rmsprop_dana'] = get_rmsprop_dana_optimizer(args.alpha, beta, args.d, args.batch_size, args.g2_scale, args.g3_over_g2, traceK, args.tanea_lr_scalar, args.tanea_global_exponent)
        
        if enable_adam:
            optimizers_dict['adam'] = optax.adam(get_adam_lr(args.alpha, beta, args.d, args.batch_size, args.g2_scale, traceK, args.tanea_lr_scalar, args.tanea_global_exponent), b1=0.0)

        # Run training experiments for enabled optimizers
        results_dict = {'beta': beta, 'model': model}
        
        for opt_name, optimizer in optimizers_dict.items():
            print(f"  Running {opt_name} experiment...")
            
            # Use TauTrackingLabelNoiseMoEPLRFTrainer for Tanea-family optimizers
            if opt_name.startswith('tanea'):
                trainer = TauTrackingLabelNoiseMoEPLRFTrainer(model, optimizer)
                key, train_key = random.split(key)
                results = trainer.train(
                    train_key,
                    num_steps=args.steps,
                    batch_size=args.batch_size,  
                    track_per_expert_loss=True,
                    track_tau_stats=True
                )
            else:
                # Use regular MoEPLRFTrainer for non-Tanea optimizers (Adam, RMSprop+Dana)
                trainer = MoEPLRFTrainer(model, optimizer)
                key, train_key = random.split(key)
                results = trainer.train(
                    train_key,
                    num_steps=args.steps,
                    batch_size=args.batch_size,  
                    track_per_expert_loss=True
                )
            
            results_dict[opt_name] = results

        moe_results.append(results_dict)

    # Skip visualization if --no_plots flag is set
    if not args.no_plots:
        # Visualization: Learning curves comparison and per-expert loss plots
        # Create a combined figure with learning curves on top and per-expert comparisons below
        n_beta = len(moe_results)

        # Count enabled Tanea optimizers for per-expert plots
        enabled_tanea_opts = []
        if enable_tanea:
            enabled_tanea_opts.append(('tanea', 'Tanea (Effective-Clip)'))
        if enable_tanea_theory:
            enabled_tanea_opts.append(('tanea_theory', 'Tanea (Theory)'))
        if enable_tanea_always_on:
            enabled_tanea_opts.append(('tanea_always_on', 'Tanea (Always-On)'))
        if enable_tanea_strong_clip:
            enabled_tanea_opts.append(('tanea_strong_clip', 'Tanea (Strong-Clip)'))
        if enable_tanea_first_moment:
            enabled_tanea_opts.append(('tanea_first_moment', 'Tanea (First-Moment)'))
        if enable_rmsprop_dana:
            enabled_tanea_opts.append(('rmsprop_dana', 'RMSprop+Dana'))

        n_tanea_optimizers = len(enabled_tanea_opts)
        fig_combined, axes_combined = plt.subplots(1 + n_tanea_optimizers, n_beta, figsize=(6 * n_beta, 5 * (1 + n_tanea_optimizers)))
        if n_beta == 1:
            axes_combined = axes_combined.reshape(-1, 1)

        # Top row: Learning curves
        axes_curves = axes_combined[0, :] if n_beta > 1 else [axes_combined[0, 0]]

        for i, result in enumerate(moe_results):
            beta = result['beta']
            ax = axes_curves[i]
            
            # Plot each optimizer's learning curve
            optimizer_colors = {
                'tanea': 'red', 
                'tanea_theory': 'orange', 
                'tanea_always_on': 'purple', 
                'tanea_strong_clip': 'brown', 
                'tanea_first_moment': 'pink', 
                'tanea_g3zero': 'blue',  # Renamed from tarmsprop_sgd
                'rmsprop_dana': 'darkred',  # New optimizer
                'adam': 'green'
            }
            
            # Get all available optimizers in this result (excluding 'beta' and 'model')
            available_optimizers = [k for k in result.keys() if k not in ['beta', 'model']]
            
            for optimizer_name in available_optimizers:
                if optimizer_name in result and 'timestamps' in result[optimizer_name] and 'losses' in result[optimizer_name]:
                    timestamps = result[optimizer_name]['timestamps']
                    losses = result[optimizer_name]['losses']
                    
                    if len(timestamps) > 0 and len(losses) > 0:
                        # Create display name
                        display_name = optimizer_name.upper().replace('_', ' ')
                        if optimizer_name == 'tanea_g3zero':
                            display_name = 'TANEA G3=0'
                        elif optimizer_name == 'rmsprop_dana':
                            display_name = 'RMSPROP+DANA'
                        
                        ax.loglog(timestamps, losses, 'o-', 
                                 color=optimizer_colors.get(optimizer_name, 'black'), 
                                 alpha=0.8, markersize=4, linewidth=2, 
                                 label=display_name)
            
            ax.set_xlabel('Training Iteration')
            ax.set_ylabel('Population Risk')
            ax.set_title(f'Learning Curves with Label Noise\nβ={beta}, M={args.m}, D={args.d}, ζ={args.zeta}\nStudent-t DOF={args.student_t_dof}, σ={args.sigma}')
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Per-expert loss comparison plots (rows 1-N for enabled Tanea optimizers)
        for i, result in enumerate(moe_results):
            beta = result['beta']
            model = result['model']
            
            # Get Adam per-expert losses for comparison
            adam_per_expert = result.get('adam', {}).get('per_expert_losses', {})
            adam_timestamps = result.get('adam', {}).get('timestamps', [])
            
            # Get Tanea G3=0 (formerly TarMSProp-SGD) per-expert losses for grey dots
            tanea_g3zero_per_expert = result.get('tanea_g3zero', {}).get('per_expert_losses', {})
            
            # Get expert selection probabilities for color mapping
            expert_probs = model.expert_probs  # This should be the probability vector p(i) ∝ i^(-zeta)
            log_expert_probs = np.log(expert_probs)
            
            # Normalize log probabilities to [0, 1] for colormap
            if len(log_expert_probs) > 1:
                log_prob_min = np.min(log_expert_probs)
                log_prob_max = np.max(log_expert_probs)
                log_prob_normalized = (log_expert_probs - log_prob_min) / (log_prob_max - log_prob_min)
            else:
                log_prob_normalized = np.array([0.5])  # Single expert case
            
            # Debug: Print what we have
            print(f"Beta {beta}: Adam has per_expert_losses: {adam_per_expert is not None and len(adam_per_expert) > 0}")
            print(f"  Tanea G3=0 has per_expert_losses: {tanea_g3zero_per_expert is not None and len(tanea_g3zero_per_expert) > 0}")
            print(f"  Expert probabilities shape: {expert_probs.shape}, log prob range: [{np.min(log_expert_probs):.3f}, {np.max(log_expert_probs):.3f}]")
            if adam_per_expert:
                print(f"  Adam per-expert keys: {list(adam_per_expert.keys())[:5]}...")  # First 5 keys
                if len(adam_per_expert) > 0:
                    first_key = list(adam_per_expert.keys())[0]
                    print(f"  Adam expert {first_key} has {len(adam_per_expert[first_key])} loss values")
            
            # Plot each enabled Tanea optimizer vs Adam
            for tanea_idx, (tanea_opt, tanea_label) in enumerate(enabled_tanea_opts):
                ax_expert = axes_combined[1 + tanea_idx, i] if n_beta > 1 else axes_combined[1 + tanea_idx, 0]
                
                if tanea_opt in result and 'per_expert_losses' in result[tanea_opt]:
                    tanea_per_expert = result[tanea_opt]['per_expert_losses']
                    tanea_timestamps = result[tanea_opt]['timestamps']
                    
                    # Debug: Print what we have for this tanea optimizer
                    print(f"  {tanea_opt} has per_expert_losses: {tanea_per_expert is not None and len(tanea_per_expert) > 0}")
                    if tanea_per_expert:
                        print(f"    {tanea_opt} per-expert keys: {list(tanea_per_expert.keys())[:5]}...")  # First 5 keys
                        if len(tanea_per_expert) > 0:
                            first_key = list(tanea_per_expert.keys())[0]
                            print(f"    {tanea_opt} expert {first_key} has {len(tanea_per_expert[first_key])} loss values")
                    
                    # First plot Tanea G3=0 vs Adam in grey for all experts (background)
                    if len(tanea_g3zero_per_expert) > 0:
                        for expert_idx in range(min(len(adam_per_expert), len(tanea_g3zero_per_expert))):
                            if expert_idx in adam_per_expert and expert_idx in tanea_g3zero_per_expert:
                                adam_losses = adam_per_expert[expert_idx]
                                g3zero_losses = tanea_g3zero_per_expert[expert_idx]
                                
                                if len(adam_losses) > 0 and len(g3zero_losses) > 0:
                                    adam_final = adam_losses[-1]
                                    g3zero_final = g3zero_losses[-1]
                                    
                                    # Plot Tanea G3=0 points in grey
                                    ax_expert.scatter(adam_final, g3zero_final, 
                                                    color='grey', alpha=0.4, s=30, 
                                                    marker='s', edgecolors='none')
                    
                    # Plot Adam vs Tanea per-expert losses with plasma colors
                    points_plotted = 0
                    for expert_idx in range(min(len(adam_per_expert), len(tanea_per_expert))):
                        if expert_idx in adam_per_expert and expert_idx in tanea_per_expert:
                            adam_losses = adam_per_expert[expert_idx]
                            tanea_losses = tanea_per_expert[expert_idx]
                            
                            # Only plot if we have data for both
                            if len(adam_losses) > 0 and len(tanea_losses) > 0:
                                # Use final losses for comparison
                                adam_final = adam_losses[-1]
                                tanea_final = tanea_losses[-1]
                                
                                # Get color based on log probability
                                color_val = log_prob_normalized[expert_idx] if expert_idx < len(log_prob_normalized) else 0.5
                                color = plt.cm.plasma(color_val)
                                
                                # Plot point with color based on expert selection probability
                                ax_expert.scatter(adam_final, tanea_final, 
                                                color=color, alpha=0.8, s=60, 
                                                edgecolors='black', linewidth=0.5)
                                points_plotted += 1
                    
                    print(f"    {tanea_opt}: Plotted {points_plotted} points")
                    
                    # Add diagonal line for reference (equal performance)
                    if len(adam_per_expert) > 0 and len(tanea_per_expert) > 0:
                        # Get range for diagonal line
                        all_adam_finals = [adam_per_expert[j][-1] for j in adam_per_expert.keys() if len(adam_per_expert[j]) > 0]
                        all_tanea_finals = [tanea_per_expert[j][-1] for j in tanea_per_expert.keys() if len(tanea_per_expert[j]) > 0]
                        
                        # Also include Tanea G3=0 for range calculation
                        if len(tanea_g3zero_per_expert) > 0:
                            all_g3zero_finals = [tanea_g3zero_per_expert[j][-1] for j in tanea_g3zero_per_expert.keys() if len(tanea_g3zero_per_expert[j]) > 0]
                            all_tanea_finals.extend(all_g3zero_finals)
                        
                        if all_adam_finals and all_tanea_finals:
                            min_loss = min(min(all_adam_finals), min(all_tanea_finals))
                            max_loss = max(max(all_adam_finals), max(all_tanea_finals))
                            
                            ax_expert.plot([min_loss, max_loss], [min_loss, max_loss], 
                                         'k--', alpha=0.5, linewidth=1, label='Equal Performance')
                    
                    # Add colorbar for plasma mapping (only for the first plot in each row)
                    if i == 0 and points_plotted > 0:
                        sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=1))
                        sm.set_array([])
                        cbar = plt.colorbar(sm, ax=ax_expert, fraction=0.05, pad=0.04)
                        cbar.set_label('Log Expert Selection Probability\n(normalized)', fontsize=8)
                        
                        # Set colorbar ticks
                        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
                        # Map back to actual log probability values
                        actual_log_probs = log_prob_min + np.array([0, 0.25, 0.5, 0.75, 1.0]) * (log_prob_max - log_prob_min)
                        cbar.set_ticklabels([f'{val:.1f}' for val in actual_log_probs])
                    
                    # Add legend with custom elements
                    legend_elements = [
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
                                  markersize=8, alpha=0.8, markeredgecolor='black', linewidth=0,
                                  label=f'{tanea_label}'),
                        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='grey', 
                                  markersize=6, alpha=0.4, linewidth=0,
                                  label='Tanea G3=0'),
                        plt.Line2D([0], [0], color='black', linestyle='--', alpha=0.5,
                                  label='Equal Performance')
                    ]
                    ax_expert.legend(handles=legend_elements, fontsize=8, loc='upper left')
                    
                    ax_expert.set_xlabel('Adam Final Loss')
                    ax_expert.set_ylabel(f'{tanea_label} Final Loss')
                    ax_expert.set_title(f'{tanea_label} vs Adam Per-Expert Losses\nβ={beta}')
                    ax_expert.grid(True, alpha=0.3)
                    ax_expert.set_xscale('log')
                    ax_expert.set_yscale('log')
                else:
                    # Debug: Print why we're not plotting
                    print(f"  {tanea_opt}: Not plotting - tanea_opt in result: {tanea_opt in result}")
                    if tanea_opt in result:
                        print(f"    per_expert_losses in result[{tanea_opt}]: {'per_expert_losses' in result[tanea_opt]}")
                    
                    # Show empty plot message
                    ax_expert.text(0.5, 0.5, f'No per-expert data\nfor {tanea_label}', 
                                 transform=ax_expert.transAxes, ha='center', va='center',
                                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                    ax_expert.set_xlabel('Adam Final Loss')
                    ax_expert.set_ylabel(f'{tanea_label} Final Loss')
                    ax_expert.set_title(f'{tanea_label} vs Adam Per-Expert Losses\nβ={beta}')

        plt.tight_layout()

        # Save combined figure
        beta_str = "_".join([f"beta{beta}" for beta in beta_list])
        combined_filename = f"{args.output_prefix}_combined_analysis_M{args.m}_D{args.d}_zeta{args.zeta}_dof{args.student_t_dof}_sigma{args.sigma}_{beta_str}_steps{args.steps}.pdf"
        combined_filepath = os.path.join(args.results_dir, combined_filename)
        plt.savefig(combined_filepath, dpi=300, bbox_inches='tight')
        print(f"Combined analysis figure saved to: {combined_filepath}")

        plt.show()
    else:
        print("Skipping plots due to --no_plots flag")

    # Summary statistics output
    print("\n" + "="*60)
    print("Label Noise Experiment Summary")
    print("="*60)

    for result in moe_results:
        beta = result['beta']
        print(f"\nβ = {beta}")
        
        # Get all available optimizers in this result (excluding 'beta' and 'model')
        available_optimizers = [k for k in result.keys() if k not in ['beta', 'model']]
        
        for optimizer_name in available_optimizers:
            if optimizer_name in result and 'timestamps' in result[optimizer_name] and 'losses' in result[optimizer_name]:
                timestamps = result[optimizer_name]['timestamps']
                losses = result[optimizer_name]['losses']
                
                if len(timestamps) > 0 and len(losses) > 0:
                    initial_loss = losses[0]
                    final_loss = losses[-1]
                    min_loss = jnp.min(losses)
                    
                    # Create display name
                    display_name = optimizer_name.upper().replace('_', ' ')
                    if optimizer_name == 'tanea_g3zero':
                        display_name = 'TANEA G3=0'
                    elif optimizer_name == 'rmsprop_dana':
                        display_name = 'RMSPROP+DANA'
                    
                    print(f"  {display_name}:")
                    print(f"    Initial loss: {initial_loss:.6f}")
                    print(f"    Final loss: {final_loss:.6f}")
                    print(f"    Min loss: {min_loss:.6f}")
                    print(f"    Total improvement: {initial_loss - final_loss:.6f}")
        
        # Add tau statistics summary for tanea optimizers
        print(f"\n  Tau Statistics Summary:")
        tanea_opts_with_tau = [k for k in available_optimizers if k.startswith('tanea')]
        for optimizer_name in tanea_opts_with_tau:
            if optimizer_name in result and 'tau_statistics' in result[optimizer_name]:
                tau_stats = result[optimizer_name]['tau_statistics']
                if len(tau_stats['timestamps']) > 0:
                    final_mean = tau_stats['tau_mean'][-1]
                    final_std = tau_stats['tau_std'][-1]
                    final_max = tau_stats['tau_max'][-1]
                    final_order_stats = tau_stats['tau_order_statistics'][-1]
                    
                    # Create display name
                    display_name = optimizer_name.upper().replace('_', ' ')
                    if optimizer_name == 'tanea_g3zero':
                        display_name = 'TANEA G3=0'
                    
                    print(f"    {display_name}:")
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


if __name__ == "__main__":
    main()
