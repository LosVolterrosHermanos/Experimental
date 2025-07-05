#!/usr/bin/env python
"""
Annealing Experiment for Tanea Learning Rate Schedules.

This script trains Mixture of Experts PLRF models comparing different learning rate
schedules for Tanea against Adam and RMSprop+Dana. It tests three different schedules:
1. Original (no additional scheduling)
2. Schedule-MK1: Cosine decay for both g2 and g3
3. Schedule-MK2: Cosine decay for g2, linear decay to 0 for g3
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
    """Parse command line arguments for the annealing experiment."""
    parser = argparse.ArgumentParser(description="Annealing Experiment for Tanea Learning Rate Schedules")
    
    # Model parameters
    parser.add_argument("--alpha", type=float, default=1.0, help="Power law exponent for eigenvalue decay")
    parser.add_argument("--beta", type=float, default=0.0, help="Power law exponent for target coefficient decay")
    parser.add_argument("--v", type=int, default=2000, help="Hidden dimension (number of random features)")
    parser.add_argument("--d", type=int, default=500, help="Embedded dimension (parameter dimension)")
    
    # MoE parameters
    parser.add_argument("--m", type=int, default=100, help="Number of experts")
    parser.add_argument("--zeta", type=float, default=0.5, help="Power-law exponent for expert selection (p(i) ∝ i^(-zeta))")
    
    # Training parameters
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=100, help="Training batch size")
    parser.add_argument("--g2_scale", type=float, default=0.2, help="Base learning rate scale")
    parser.add_argument("--g3_over_g2", type=float, default=0.05, help="G3 to G2 ratio for momentum")
    parser.add_argument("--tanea_lr_scalar", type=float, default=1e-2, help="Tanea learning rate scalar")
    parser.add_argument("--tanea_global_exponent", type=float, default=0.0, help="Tanea global time exponent")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2 parameter (also used for RMSprop decay in RMSprop+Dana)")
  
    # Label noise parameters
    parser.add_argument("--student_t_dof", type=float, default=3.0, help="Degrees of freedom for student-t distribution")
    parser.add_argument("--sigma", type=float, default=0.1, help="Scaling factor for the noise")
    
    # Schedule parameters
    parser.add_argument("--cosine_decay_steps", type=int, default=None, help="Number of steps for cosine decay (default: training steps)")
    parser.add_argument("--linear_decay_steps", type=int, default=None, help="Number of steps for linear decay (default: training steps)")
    
    # Output parameters
    parser.add_argument("--results_dir", type=str, default="annealing-results", help="Directory to store results")
    parser.add_argument("--output_prefix", type=str, default="annealing", help="Prefix for output files")
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
    """Custom MoEPLRFTrainer that tracks tau statistics during training for label noise models."""
    
    def train(self,
              key: random.PRNGKey,
              num_steps: int,
              batch_size: int,
              init_params: Optional[jnp.ndarray] = None,
              eval_freq: Optional[int] = None,
              track_per_expert_loss: bool = False,
              track_update_history: bool = False,
              track_tau_stats: bool = True) -> Dict:
        """Train the MoE model and return training metrics including tau statistics."""
        
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
            # Use vectorized per-expert risk computation and store as arrays
            initial_per_expert_risks = self.model.per_expert_population_risk(init_params)
            per_expert_losses = [initial_per_expert_risks]

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
                    # Use vectorized per-expert risk computation
                    current_per_expert_risks = self.model.per_expert_population_risk(params)
                    per_expert_losses.append(current_per_expert_risks)

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
            # Convert list of arrays to a single array of shape (n_eval_times, m)
            results['per_expert_losses'] = jnp.array(per_expert_losses)
        
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
        """Initialize the Label Noise MoE PLRF model."""
        # Initialize base MoE model
        super().__init__(alpha, beta, v, d, m, zeta, key)
        
        self.student_t_dof = student_t_dof
        self.sigma = sigma
    
    def generate_batch(self, key: random.PRNGKey, batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate a batch of (X, y) training data with student-t label noise."""
        # Split key for base data and noise
        key_data, key_noise = random.split(key)
        
        # Generate base batch
        X, y_clean = super().generate_batch(key_data, batch_size)
        
        # Add student-t distributed label noise
        # Student-t distribution with specified degrees of freedom and scaling
        noise = random.t(key_noise, self.student_t_dof, shape=(batch_size,)) * self.sigma
        y_noisy = y_clean + noise
        
        return X, y_noisy


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

def get_adam_lr(alpha, beta, d, batch_size, g2_scale, traceK, tanea_lr_scalar, tanea_global_exponent):
    """Get Adam learning rate."""
    return 0.5*g2_scale * jnp.minimum(1.0, jnp.float32(batch_size) / traceK)*tanea_lr_scalar

def get_rmsprop_dana_optimizer(alpha, beta, d, batch_size, g2_scale, g3_over_g2, traceK, tanea_lr_scalar, tanea_global_exponent, adam_beta2):
    """Get RMSprop+Dana optimizer with hyperparameters based on Tanea/Adam settings."""
    # Get Adam LR for Dana g2
    adam_lr = get_adam_lr(alpha, beta, d, batch_size, g2_scale, traceK, tanea_lr_scalar, tanea_global_exponent)
    
    # RMSprop decay (same as Adam beta2)
    rms_decay = adam_beta2
    
    # Dana parameters
    dana_g2 = adam_lr
    dana_g3 = g3_over_g2 * adam_lr

    # Create power law schedules for Dana parameters
    kappa_b = jnp.log(batch_size) / jnp.log(d)  # exponent for batch wrt d
    g1 = powerlaw_schedule(1.0, 0.0, 0.0, 1)
    g2 = powerlaw_schedule(dana_g2, 0.0, -1.0*(1.0 - kappa_b) / (2 * alpha), 1)
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
    """Main function to run the annealing experiment."""
    args = parse_args()
    
    # Set random seed
    global key
    key = random.PRNGKey(args.random_seed)
    
    # Use single beta value
    beta = args.beta
    
    # Set default decay steps if not provided
    cosine_decay_steps = args.cosine_decay_steps or args.steps
    linear_decay_steps = args.linear_decay_steps or args.steps
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Compute traceK for hyperparameter scaling
    traceK = get_traceK(args.alpha, args.v)

    print("="*60)
    print("Annealing Experiment for Tanea Learning Rate Schedules")
    print("="*60)
    print(f"Model parameters: α={args.alpha}, β={beta}, V={args.v}, D={args.d}")
    print(f"MoE parameters: M={args.m}, ζ={args.zeta}")
    print(f"Training parameters: STEPS={args.steps}, BATCH_SIZE={args.batch_size}")
    print(f"Label noise parameters: Student-t DOF={args.student_t_dof}, σ={args.sigma}")
    print(f"Schedule parameters: Cosine decay steps={cosine_decay_steps}, Linear decay steps={linear_decay_steps}")
    print(f"Optimizers: Tanea (MK2), Adam, RMSprop+Dana")
    print(f"Results directory: {args.results_dir}")
    print("="*60)

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

    # Create optimizers dictionary
    optimizers_dict = {}
    
    # Get base Tanea hyperparameters
    base_tanea_hparams = get_tanea_hparams(args.alpha, beta, args.d, args.batch_size, args.g2_scale, args.g3_over_g2, traceK, args.tanea_lr_scalar, args.tanea_global_exponent)
    
    # 1. Original Tanea (no additional scheduling)
    optimizers_dict['tanea_original'] = tanea_optimizer(base_tanea_hparams.g2, base_tanea_hparams.g3, base_tanea_hparams.delta, momentum_flavor="mk2")
    
    # 2. Schedule-MK1: Cosine decay for both g2 and g3
    cosdecay = optax.cosine_decay_schedule(1.0,cosine_decay_steps)
    g2_mk1 = lambda t : base_tanea_hparams.g2(t) * cosdecay(t)
    g3_mk1 = lambda t : base_tanea_hparams.g3(t) * cosdecay(t)
    optimizers_dict['tanea_schedule_mk1'] = tanea_optimizer(g2_mk1, g3_mk1, base_tanea_hparams.delta, momentum_flavor="mk2")
    
    # 3. Schedule-MK2: Cosine decay for g2, linear decay to 0 for g3
    log_decay = lambda t : 1.0/(1.0+jnp.log(1.0 + t))
    #lindecay = optax.linear_schedule(1.0, 0.0, linear_decay_steps)
    g2_mk2 = lambda t : base_tanea_hparams.g2(t) * (log_decay(t))
    g3_mk2 = lambda t : base_tanea_hparams.g3(t) * (log_decay(t)**2)
    optimizers_dict['tanea_schedule_mk2'] = tanea_optimizer(g2_mk2, g3_mk2, base_tanea_hparams.delta, momentum_flavor="mk2")
    
    # 4. Adam
    optimizers_dict['adam'] = optax.adam(get_adam_lr(args.alpha, beta, args.d, args.batch_size, args.g2_scale, traceK, args.tanea_lr_scalar, args.tanea_global_exponent), b1=0.0, b2=args.adam_beta2)
    
    # 5. RMSprop+Dana
    optimizers_dict['rmsprop_dana'] = get_rmsprop_dana_optimizer(args.alpha, beta, args.d, args.batch_size, args.g2_scale, args.g3_over_g2, traceK, args.tanea_lr_scalar, args.tanea_global_exponent, args.adam_beta2)

    # Run training experiments for all optimizers
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
        print(f"    Final loss: {results['losses'][-1]:.6f}")

    # Save results
    import pickle
    timestamp = os.times().elapsed
    results_filename = f"{args.results_dir}/{args.output_prefix}_results_{int(timestamp)}.pkl"
    
    results_data = {
        'results': results_dict,  # Single result dict instead of list
        'args': vars(args),
        'cosine_decay_steps': cosine_decay_steps,
        'linear_decay_steps': linear_decay_steps
    }
    
    with open(results_filename, 'wb') as f:
        pickle.dump(results_data, f)
    
    print(f"\nResults saved to {results_filename}")

    # Generate plots if requested
    if not args.no_plots:
        print("\nGenerating combined visualization...")
        
        # Create a comprehensive figure with learning curves and per-expert comparisons
        fig = plt.figure(figsize=(16, 12))
        
        # Define time points: 25%, 50%, final of training (chronological order)
        time_points = [
            ('25pct', int(args.steps * 0.25), 'blue'),
            ('50pct', int(args.steps * 0.5), 'orange'), 
            ('final', args.steps, 'red')
        ]
        
        # Colors for time points (chronological: earliest=blue, latest=red)
        time_colors = {
            '25pct': 'blue',
            '50pct': 'orange', 
            'final': 'red'
        }
        
        # Top subplot: Learning curves
        ax_curves = plt.subplot(3, 4, (1, 4))  # Top row, spans 4 columns
        
        # Plot each optimizer's learning curve
        optimizer_colors = {
            'tanea_original': 'red', 
            'tanea_schedule_mk1': 'orange',
            'tanea_schedule_mk2': 'purple',
            'rmsprop_dana': 'darkred',
            'adam': 'green'
        }
        
        optimizer_labels = {
            'tanea_original': 'TANEA (Original)',
            'tanea_schedule_mk1': 'TANEA (Schedule-MK1)',
            'tanea_schedule_mk2': 'TANEA (Schedule-MK2)',
            'rmsprop_dana': f'RMSPROP+DANA (β₂={args.adam_beta2})',
            'adam': f'ADAM (β₁=0.0, β₂={args.adam_beta2})'
        }
        
        # Get all available optimizers in this result (excluding 'beta' and 'model')
        available_optimizers = [k for k in results_dict.keys() if k not in ['beta', 'model']]
        
        for optimizer_name in available_optimizers:
            if optimizer_name in results_dict and 'timestamps' in results_dict[optimizer_name] and 'losses' in results_dict[optimizer_name]:
                timestamps = results_dict[optimizer_name]['timestamps']
                losses = results_dict[optimizer_name]['losses']
                
                if len(timestamps) > 0 and len(losses) > 0:
                    # Create display name
                    display_name = optimizer_labels.get(optimizer_name, optimizer_name.upper())
                    
                    ax_curves.loglog(timestamps, losses, 'o-', 
                             color=optimizer_colors.get(optimizer_name, 'black'), 
                             alpha=0.8, markersize=4, linewidth=2, 
                             label=display_name)
        
        ax_curves.set_xlabel('Training Iteration')
        ax_curves.set_ylabel('Population Risk')
        ax_curves.set_title(f'Learning Curves - Annealing Experiment\nβ={beta}, M={args.m}, D={args.d}, ζ={args.zeta}, Student-t DOF={args.student_t_dof}, σ={args.sigma}')
        ax_curves.grid(True, alpha=0.3)
        ax_curves.legend(loc='upper right')

        # Per-expert loss visualization combining all time points
        print("  Adding per-expert loss comparisons...")
        
        # Get Adam and comparison optimizer per-expert losses
        adam_per_expert_array = results_dict.get('adam', {}).get('per_expert_losses', None)
        rmsprop_dana_per_expert_array = results_dict.get('rmsprop_dana', {}).get('per_expert_losses', None)
        
        # Get Tanea variants for comparison
        tanea_variants = ['tanea_original', 'tanea_schedule_mk1', 'tanea_schedule_mk2']
        
        if adam_per_expert_array is not None and 'timestamps' in results_dict['adam']:
            adam_timestamps = results_dict['adam']['timestamps']
            
            # Create subplot positions for 2x4 grid (excluding top row)
            subplot_positions = [
                (3, 4, 5), (3, 4, 6), (3, 4, 7), (3, 4, 8),  # Second row
                (3, 4, 9), (3, 4, 10), (3, 4, 11), (3, 4, 12)  # Third row  
            ]
            
            plot_idx = 0
            
            # Plot Tanea variants vs Adam
            for tanea_variant in tanea_variants:
                if tanea_variant in results_dict and 'per_expert_losses' in results_dict[tanea_variant] and plot_idx < len(subplot_positions):
                    tanea_per_expert_array = results_dict[tanea_variant]['per_expert_losses']
                    
                    if tanea_per_expert_array is not None:
                        ax_expert = plt.subplot(*subplot_positions[plot_idx])
                        
                        # Plot all time points on the same axes (earliest to latest)
                        for time_label, target_step, time_color in time_points:
                            # Find closest timestamp index
                            closest_idx = np.argmin(np.abs(adam_timestamps - target_step))
                            actual_step = adam_timestamps[closest_idx]
                            
                            if closest_idx < len(tanea_per_expert_array):
                                # Get losses at this time point
                                adam_losses_at_time = adam_per_expert_array[closest_idx, :]
                                tanea_losses_at_time = tanea_per_expert_array[closest_idx, :]
                                
                                # Plot all experts with time-based color
                                ax_expert.scatter(adam_losses_at_time, tanea_losses_at_time, 
                                                c=time_color, alpha=0.7, s=40, 
                                                label=f'{time_label} (Step {actual_step})',
                                                edgecolors='black', linewidth=0.3)
                        
                        # Add diagonal line for reference (equal performance)
                        # Get overall range across all time points
                        all_adam_losses = []
                        all_tanea_losses = []
                        for time_label, target_step, _ in time_points:
                            closest_idx = np.argmin(np.abs(adam_timestamps - target_step))
                            if closest_idx < len(tanea_per_expert_array):
                                all_adam_losses.extend(adam_per_expert_array[closest_idx, :])
                                all_tanea_losses.extend(tanea_per_expert_array[closest_idx, :])
                        
                        if all_adam_losses and all_tanea_losses:
                            min_loss = min(min(all_adam_losses), min(all_tanea_losses))
                            max_loss = max(max(all_adam_losses), max(all_tanea_losses))
                            
                            ax_expert.plot([min_loss, max_loss], [min_loss, max_loss], 
                                         'k--', alpha=0.5, linewidth=1, label='Equal Performance')
                        
                        ax_expert.set_xlabel('Adam Loss')
                        ax_expert.set_ylabel(f'{optimizer_labels[tanea_variant]} Loss')
                        ax_expert.set_title(f'{optimizer_labels[tanea_variant]} vs Adam\n(Time Evolution)')
                        ax_expert.grid(True, alpha=0.3)
                        ax_expert.legend(fontsize=8)
                        
                        plot_idx += 1
            
            # Plot RMSprop+Dana vs Adam
            if rmsprop_dana_per_expert_array is not None and plot_idx < len(subplot_positions):
                ax_expert = plt.subplot(*subplot_positions[plot_idx])
                
                # Plot all time points on the same axes (earliest to latest)
                for time_label, target_step, time_color in time_points:
                    # Find closest timestamp index
                    closest_idx = np.argmin(np.abs(adam_timestamps - target_step))
                    actual_step = adam_timestamps[closest_idx]
                    
                    if closest_idx < len(rmsprop_dana_per_expert_array):
                        # Get losses at this time point
                        adam_losses_at_time = adam_per_expert_array[closest_idx, :]
                        rmsprop_losses_at_time = rmsprop_dana_per_expert_array[closest_idx, :]
                        
                        # Plot all experts with time-based color
                        ax_expert.scatter(adam_losses_at_time, rmsprop_losses_at_time, 
                                        c=time_color, alpha=0.7, s=40, 
                                        label=f'{time_label} (Step {actual_step})',
                                        edgecolors='black', linewidth=0.3)
                
                # Add diagonal line for reference (equal performance)
                # Get overall range across all time points
                all_adam_losses = []
                all_rmsprop_losses = []
                for time_label, target_step, _ in time_points:
                    closest_idx = np.argmin(np.abs(adam_timestamps - target_step))
                    if closest_idx < len(rmsprop_dana_per_expert_array):
                        all_adam_losses.extend(adam_per_expert_array[closest_idx, :])
                        all_rmsprop_losses.extend(rmsprop_dana_per_expert_array[closest_idx, :])
                
                if all_adam_losses and all_rmsprop_losses:
                    min_loss = min(min(all_adam_losses), min(all_rmsprop_losses))
                    max_loss = max(max(all_adam_losses), max(all_rmsprop_losses))
                    
                    ax_expert.plot([min_loss, max_loss], [min_loss, max_loss], 
                                 'k--', alpha=0.5, linewidth=1, label='Equal Performance')
                
                ax_expert.set_xlabel('Adam Loss')
                ax_expert.set_ylabel('RMSprop+Dana Loss')
                ax_expert.set_title('RMSprop+Dana vs Adam\n(Time Evolution)')
                ax_expert.grid(True, alpha=0.3)
                ax_expert.legend(fontsize=8)
        
        plt.tight_layout()

        # Save the combined figure
        combined_plots_filename = f"{args.results_dir}/{args.output_prefix}_combined_{int(timestamp)}.pdf"
        plt.savefig(combined_plots_filename, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to {combined_plots_filename}")

        plt.show()

    print("\nAnnealing experiment completed!")


if __name__ == "__main__":
    main()