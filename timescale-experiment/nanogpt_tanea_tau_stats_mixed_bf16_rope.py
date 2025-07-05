#!/usr/bin/env python
"""
NanoGPT training with Tanea optimizer using mixed precision (bfloat16 matmuls, float32 everything else) and RoPE.
Based on nanogpt_tanea_tau_stats_pure_bf16_rope.py but with mixed precision for stability.
"""

import os
import signal
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.stats as stats
import argparse
import logging
from typing import Dict, List, Any
from tqdm import tqdm

# Import from the gpt2 directory
import sys
sys.path.append('../dana-nonquadratic-tests/gpt2')
from nanogpt_minimal import count_params
from nanogpt_rope_mixed_precision import GPTWithRoPE, ModelConfig
from fineweb_dataset import FineWebDataset, create_fineweb_datasets

import jax
# Enable bfloat16 for matrix multiplications only
jax.config.update('jax_default_matmul_precision', 'bfloat16')

import jax.numpy as jnp
from power_law_rf.optimizers import powerlaw_schedule, tanea_optimizer, TaneaOptimizerState
import optax
from flax.core import FrozenDict
from flax.training.train_state import TrainState

LOG_STEPS_BASE = 1.1
INIT_STD = 0.02

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        opt_state: Optimizer state (may be from optax.chain)
        
    Returns:
        Dictionary with tau statistics including both largest and smallest order statistics
    """
    # Handle optax.chain optimizer - extract the Tanea state
    tanea_state = opt_state
    if hasattr(opt_state, '__len__') and len(opt_state) > 1:
        # optax.chain creates a tuple: (clip_state, tanea_state, ...)
        tanea_state = opt_state[1]
    
    if not isinstance(tanea_state, TaneaOptimizerState):
        return {}
    
    # Flatten tau tree into a single vector
    tau_leaves = jax.tree_util.tree_leaves(tanea_state.tau)
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

# Model configuration and helper functions are now imported from the separate module

@jax.jit
def train_step(state: TrainState, x: jnp.ndarray, y: jnp.ndarray):
    def loss_fn(params: FrozenDict) -> jnp.ndarray:
        logits = state.apply_fn(params, x, False)
        # Loss computation in float32
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        return loss

    loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return loss, new_state

# FineWebDataset is now imported from the separate module

def parse_args():
    parser = argparse.ArgumentParser(description="Train nanogpt with Tanea optimizer using mixed precision (bfloat16 matmuls) and RoPE")
    parser.add_argument(
        "--train_steps", type=int, default=10000,
        help="Number of training steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--seq_len", type=int, default=1024,
        help="Sequence length for training"
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=64,
        help="Validation batch size"
    )
    parser.add_argument(
        "--val_max_tokens", type=int, default=None,
        help="Maximum tokens to load for validation"
    )
    parser.add_argument(
        "--val_steps", type=int, default=20,
        help="Number of validation steps"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=2.0,
        help="Gradient clipping value"
    )
    parser.add_argument(
        "--init_std", type=float, default=0.02,
        help="Weight initialization standard deviation"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Directory to store results"
    )
    # Add Tanea hyperparameters
    parser.add_argument(
        "--tanea_g2", type=float, default=1E-4,
        help="Tanea G2 parameter"
    )
    parser.add_argument(
        "--tanea_g3", type=float, default=1E-5,
        help="Tanea G3 parameter"
    )
    parser.add_argument(
        "--tanea_delta", type=float, default=8.0,
        help="Tanea Delta parameter"
    )
    parser.add_argument(
        "--tanea_kappa", type=float, default=1.0,
        help="Tanea Kappa parameter"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0,
        help="Weight decay parameter"
    )
    parser.add_argument(
        "--power_weight_decay", type=float, default=1.0,
        help="Power of weight decay parameter"
    )
    parser.add_argument(
        "--weight_decay_ts", type=float, default=1.0,
        help="Timescale of weight decay parameter"
    )
    parser.add_argument(
        "--momentum_flavor", type=str, default="effective-clip",
        choices=["effective-clip", "theory", "always-on", "strong-clip", "mk2"],
        help="Tanea momentum flavor"
    )
    # RoPE specific parameters
    parser.add_argument(
        "--rope_base", type=float, default=10000.0,
        help="Base frequency for RoPE"
    )
    return parser.parse_args()

def evaluate_validation_loss(state, val_dataset, config, val_steps=20):
    """Evaluate validation loss"""
    total_loss = 0.0
    steps_taken = 0
    
    # Create a fresh iterator each time we evaluate
    val_iterator = val_dataset.iterate_once(config["val_batch_size"], config["seq_len"])
    
    for x, y, w in val_iterator:
        if steps_taken >= val_steps:
            break
            
        loss, _ = train_step(state, x, y)  # Don't update state for validation
        total_loss += loss
        steps_taken += 1
    
    if steps_taken == 0:
        return float('inf')  # Return inf if no validation data
    
    return total_loss / steps_taken

def main():
    """
    Train NanoGPT with Tanea optimizer and collect tau statistics using mixed precision and RoPE.
    """
    args = parse_args()
    
    # Override INIT_STD if provided
    global INIT_STD
    INIT_STD = args.init_std
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Configuration dictionary
    val_max_tokens = args.val_max_tokens
    if val_max_tokens is None:
        # Default to enough tokens for validation batches
        val_max_tokens = args.val_batch_size * args.seq_len * (args.val_steps+1)
    
    config = {
        "train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "val_batch_size": args.val_batch_size,
        "val_max_tokens": val_max_tokens,
        "val_steps": args.val_steps,
        "grad_clip": args.grad_clip,
        "init_std": args.init_std,
        "results_dir": args.results_dir,
        "tanea_g2": args.tanea_g2,
        "tanea_g3": args.tanea_g3,
        "tanea_delta": args.tanea_delta,
        "tanea_kappa": args.tanea_kappa,
        "weight_decay": args.weight_decay,
        "power_weight_decay": args.power_weight_decay,
        "weight_decay_ts": args.weight_decay_ts,
        "momentum_flavor": args.momentum_flavor,
        "rope_base": args.rope_base,
        "precision": "mixed_bfloat16_rope"
    }
    
    # Create LOG_STEPS
    LOG_STEPS = jnp.unique(jnp.concatenate([
        jnp.array([0]),
        jnp.int32(LOG_STEPS_BASE**jnp.arange(1, jnp.ceil(jnp.log(config["train_steps"])/jnp.log(LOG_STEPS_BASE)))),
        jnp.array([config["train_steps"]])
    ]))
    
    # Initialize Tanea optimizer
    g2 = powerlaw_schedule(config["tanea_g2"], 0.0, 0.0, 1)
    g3 = powerlaw_schedule(config["tanea_g3"], 0.0, -1.0*config["tanea_kappa"], 1)
    delta = powerlaw_schedule(1.0, 0.0, -1.0, config["tanea_delta"])
    wdscheduler = powerlaw_schedule(1.0*config["weight_decay"], 0.0, -1.0*config["power_weight_decay"], config["weight_decay_ts"])
    tanea = tanea_optimizer(g2=g2, g3=g3, Delta=delta, wd=wdscheduler, momentum_flavor=config["momentum_flavor"])

    tanea = optax.chain(
        optax.clip_by_global_norm(config['grad_clip']),
        tanea
    )
    optimizer = tanea
    
    # Initialize model with mixed precision
    key = jax.random.PRNGKey(0)
    model_config = ModelConfig(rope_base=config["rope_base"])
    model = GPTWithRoPE(model_config, mixed_precision=True, init_std=config["init_std"])
    params = model.init(key)
    num_params = count_params(params)
    
    logger.info(f"Model initialized with {num_params:,} parameters")
    logger.info("Using mixed precision (bfloat16 matmuls, float32 everything else) with RoPE")
    logger.info(f"Optimizer: Tanea (momentum_flavor={config['momentum_flavor']}) with grad_clip={config['grad_clip']}")
    logger.info(f"Tanea params: g2={config['tanea_g2']}, g3={config['tanea_g3']}, delta={config['tanea_delta']}, kappa={config['tanea_kappa']}")
    
    # Initialize train state
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer)
    
    # Initialize datasets using the new utility function
    data_root = os.path.expanduser("../dana-nonquadratic-tests/gpt2/fineweb-edu/sample/10BT")
    train_dataset, val_dataset = create_fineweb_datasets(
        data_root, 
        val_max_tokens=config["val_max_tokens"],
        val_files_count=1
    )
    
    # Create training iterator
    train_iterator = train_dataset.iterate_once(config["batch_size"], config["seq_len"])
    
    # Storage for losses and metrics
    metrics_history = {
        'step': [],
        'train_loss': [],
        'val_loss': [],
        'tokens_processed': [],
        'time_elapsed': []
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
    initial_tau_stats = extract_tau_statistics(state.opt_state)
    if initial_tau_stats:
        tau_statistics['timestamps'].append(0)
        tau_statistics['tau_order_statistics'].append(initial_tau_stats['tau_order_statistics'])
        tau_statistics['tau_reversed_order_statistics'].append(initial_tau_stats['tau_reversed_order_statistics'])
        tau_statistics['tau_mean'].append(initial_tau_stats['tau_mean'])
        tau_statistics['tau_std'].append(initial_tau_stats['tau_std'])
        tau_statistics['tau_min'].append(initial_tau_stats['tau_min'])
        tau_statistics['tau_max'].append(initial_tau_stats['tau_max'])
    
    # Training loop with loss logging
    pbar = tqdm(range(config["train_steps"]), desc="Training")
    start_time = time.time()
    
    for step in pbar:
        # Get next batch
        x, y, w = next(train_iterator)
        
        # Forward and backward pass
        loss, state = train_step(state, x, y)
        
        # Update progress bar
        pbar.set_postfix(loss=f"{loss:.4f}")
        
        # Log metrics at specified steps
        if step in LOG_STEPS:
            # Evaluate validation loss
            val_loss = evaluate_validation_loss(state, val_dataset, config, config["val_steps"])
            
            total_tokens = step * config["batch_size"] * config["seq_len"]
            metrics_history['step'].append(step)
            metrics_history['train_loss'].append(float(loss))
            metrics_history['val_loss'].append(float(val_loss))
            metrics_history['tokens_processed'].append(total_tokens)
            metrics_history['time_elapsed'].append(time.time() - start_time)
            
            # Collect tau statistics
            tau_stats = extract_tau_statistics(state.opt_state)
            if tau_stats:
                tau_statistics['timestamps'].append(step)
                tau_statistics['tau_order_statistics'].append(tau_stats['tau_order_statistics'])
                tau_statistics['tau_reversed_order_statistics'].append(tau_stats['tau_reversed_order_statistics'])
                tau_statistics['tau_mean'].append(tau_stats['tau_mean'])
                tau_statistics['tau_std'].append(tau_stats['tau_std'])
                tau_statistics['tau_min'].append(tau_stats['tau_min'])
                tau_statistics['tau_max'].append(tau_stats['tau_max'])
            
            # Print detailed metrics
            elapsed = time.time() - start_time
            average_tokens_per_second = total_tokens / elapsed
            tqdm.write(f"\nStep: {step}/{config['train_steps']} ({100.0 * step / config['train_steps']:.1f}%)")
            tqdm.write(f"  Train Loss: {loss:.6f}")
            tqdm.write(f"  Val Loss: {val_loss:.6f}")
            tqdm.write(f"  Time: {elapsed:.2f}s ({elapsed/60:.2f}m)")
            tqdm.write(f"  Tokens: {total_tokens:,} ({average_tokens_per_second:.1f} tokens/s)")
            if tau_stats:
                tqdm.write(f"  Tau Mean: {tau_stats['tau_mean']:.6f}, Tau Max: {tau_stats['tau_max']:.6f}")
            tqdm.write(f"  G2: {config['tanea_g2']}, G3: {config['tanea_g3']}, Delta: {config['tanea_delta']}")
            tqdm.write(f"  Momentum Flavor: {config['momentum_flavor']}")
            tqdm.write(f"  Precision: mixed bfloat16 + RoPE\n")
    
    # Convert tau statistics lists to arrays
    for key in tau_statistics:
        if key not in ['tau_order_statistics', 'tau_reversed_order_statistics']:
            tau_statistics[key] = jnp.array(tau_statistics[key])
    
    # Save results
    results_data = {
        'metrics': metrics_history,
        'tau_statistics': tau_statistics,
        'config': config,
        'num_params': num_params,
        'precision': 'mixed_bfloat16_rope'
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_filename = (
        f"{config['results_dir']}/nanogpt_tanea_results_mixed_bf16_rope_{timestamp}_"
        f"steps_{config['train_steps']}_bs_{config['batch_size']}_"
        f"seq_{config['seq_len']}_"
        f"g2_{config['tanea_g2']}_g3_{config['tanea_g3']}_delta_{config['tanea_delta']}_"
        f"flavor_{config['momentum_flavor']}.pkl"
    )
    
    with open(results_filename, 'wb') as f:
        pickle.dump(results_data, f)
    
    print(f"Results saved to {results_filename}")
    
    # Save checkpoint of weights
    checkpoint_dir = "weight-checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_filename = (
        f"{checkpoint_dir}/nanogpt_tanea_checkpoint_mixed_bf16_rope_{timestamp}_"
        f"steps_{config['train_steps']}_bs_{config['batch_size']}_"
        f"seq_{config['seq_len']}_"
        f"g2_{config['tanea_g2']}_g3_{config['tanea_g3']}_delta_{config['tanea_delta']}_"
        f"flavor_{config['momentum_flavor']}.pkl"
    )
    
    checkpoint_data = {
        'params': state.params,
        'config': config,
        'num_params': num_params,
        'precision': 'mixed_bfloat16_rope',
        'final_train_loss': float(loss),
        'final_val_loss': float(val_loss) if 'val_loss' in locals() else None
    }
    
    with open(checkpoint_filename, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    print(f"Checkpoint saved to {checkpoint_filename}")
    
    return results_data

if __name__ == "__main__":
    main()