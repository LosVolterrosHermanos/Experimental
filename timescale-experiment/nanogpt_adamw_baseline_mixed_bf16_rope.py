#!/usr/bin/env python
"""
NanoGPT training with AdamW optimizer using mixed precision (bfloat16 matmuls, float32 everything else) and RoPE.
Based on nanogpt_adamw_baseline_pure_bf16_rope.py but with mixed precision for improved stability.
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
import optax
from flax.core import FrozenDict
from flax.training.train_state import TrainState

LOG_STEPS_BASE = 1.1
INIT_STD = 0.02

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@jax.jit
def train_step(state: TrainState, x: jnp.ndarray, y: jnp.ndarray):
    def loss_fn(params: FrozenDict) -> jnp.ndarray:
        logits = state.apply_fn(params, x, False)
        # Loss computation in float32 for numerical stability
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        return loss

    loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return loss, new_state

def parse_args():
    parser = argparse.ArgumentParser(description="Train nanogpt with AdamW optimizer using mixed precision (bfloat16 matmuls) and RoPE")
    parser.add_argument(
        "--train_steps", type=int, default=96000,
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
        "--val_batch_size", type=int, default=32,
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
    # Add AdamW hyperparameters
    parser.add_argument(
        "--lr", type=float, default=16E-5,
        help="Learning rate for AdamW optimizer"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.9,
        help="Beta1 parameter for AdamW"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.95,
        help="Beta2 parameter for AdamW"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1E-3,
        help="Weight decay parameter for AdamW"
    )
    # RoPE specific parameters
    parser.add_argument(
        "--rope_base", type=float, default=10000.0,
        help="Base frequency for RoPE"
    )
    # Attention implementation parameters
    parser.add_argument(
        "--attention_implementation", type=str, default="naive",
        choices=["naive", "xla", "cudnn"],
        help="Attention implementation to use: naive (manual), xla (JAX XLA), or cudnn (cuDNN)"
    )
    # Validation parameters
    parser.add_argument(
        "--disable_validation", action="store_true",
        help="Disable validation loss computation for faster training"
    )
    # WSD scheduler parameters
    parser.add_argument(
        "--enable_wsd", action="store_true",
        help="Enable WSD (Warmup-Stable-Decay) schedule using optax.chain"
    )
    parser.add_argument(
        "--warmup_fraction", type=float, default=0.02,
        help="Fraction of training steps for warmup phase (default: 0.1)"
    )
    parser.add_argument(
        "--decay_fraction", type=float, default=0.2,
        help="Final decay fraction for WSD schedule (default: 0.0)"
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
    Train NanoGPT with AdamW optimizer using mixed precision (bfloat16 matmuls, float32 everything else) and RoPE.
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
        val_max_tokens = args.val_batch_size * args.seq_len * (args.val_steps + 1)
    
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
        "lr": args.lr,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "weight_decay": args.weight_decay,
        "rope_base": args.rope_base,
        "attention_implementation": args.attention_implementation,
        "disable_validation": args.disable_validation,
        "enable_wsd": args.enable_wsd,
        "warmup_fraction": args.warmup_fraction,
        "decay_fraction": args.decay_fraction,
        "precision": "mixed_bfloat16_rope"
    }
    
    # Create LOG_STEPS
    LOG_STEPS = jnp.unique(jnp.concatenate([
        jnp.array([0]),
        jnp.int32(LOG_STEPS_BASE**jnp.arange(1, jnp.ceil(jnp.log(config["train_steps"])/jnp.log(LOG_STEPS_BASE)))),
        jnp.array([config["train_steps"]])
    ]))
    
    # Initialize AdamW optimizer with optional WSD schedule
    if config["enable_wsd"]:
        # Create WSD (Warmup-Stable-Decay) schedule using linear_onecycle_schedule
        wsd_schedule = lambda t : jnp.minimum(jnp.minimum( t/(config["train_steps"]*config["warmup_fraction"]), (1.0 - (t/(config["train_steps"])))/(1.0 - config["decay_fraction"])),1.0)
        # wsd_schedule = optax.schedules.linear_onecycle_schedule(
        #     config["train_steps"],
        #     1.0,
        #     pct_start=config["warmup_fraction"],
        #     pct_final=config["decay_fraction"],
        #     div_factor=1.0,
        #     final_div_factor=10000.0
        # )
        
        optimizer = optax.chain(
            optax.clip_by_global_norm(config['grad_clip']),
            optax.adamw(
                learning_rate=config['lr'],
                b1=config['beta1'],
                b2=config['beta2'],
                weight_decay=config['weight_decay']
            ),
            optax.scale_by_schedule(wsd_schedule)
        )
    else:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config['grad_clip']),
            optax.adamw(
                learning_rate=config['lr'],
                b1=config['beta1'],
                b2=config['beta2'],
                weight_decay=config['weight_decay']
            )
        )
    
    # Initialize model with mixed precision
    key = jax.random.PRNGKey(0)
    model_config = ModelConfig(
        rope_base=config["rope_base"],
        attention_implementation=config["attention_implementation"]
    )
    model = GPTWithRoPE(model_config, mixed_precision=True, init_std=config["init_std"])
    params = model.init(key)
    num_params = count_params(params)
    
    logger.info(f"Model initialized with {num_params:,} parameters")
    logger.info("Using mixed precision (bfloat16 matmuls, float32 everything else) with RoPE positional embedding")
    logger.info(f"Attention implementation: {config['attention_implementation']}")
    logger.info(f"Validation: {'disabled' if config['disable_validation'] else 'enabled'}")
    
    # Initialize train state
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer)
    
    # Initialize datasets using the new utility function
    data_root = os.path.expanduser("../dana-nonquadratic-tests/gpt2/fineweb-edu/sample/10BT")
    if config["disable_validation"]:
        # Only create training dataset - use create_fineweb_datasets but ignore validation
        train_dataset, _ = create_fineweb_datasets(
            data_root, 
            val_max_tokens=config["val_max_tokens"],
            val_files_count=1
        )
        val_dataset = None
    else:
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
            # Evaluate validation loss (if enabled)
            if config["disable_validation"]:
                val_loss = float('nan')  # Use NaN to indicate disabled validation
            else:
                val_loss = evaluate_validation_loss(state, val_dataset, config, config["val_steps"])
            
            total_tokens = step * config["batch_size"] * config["seq_len"]
            metrics_history['step'].append(step)
            metrics_history['train_loss'].append(float(loss))
            metrics_history['val_loss'].append(float(val_loss))
            metrics_history['tokens_processed'].append(total_tokens)
            metrics_history['time_elapsed'].append(time.time() - start_time)
            
            # Print detailed metrics
            elapsed = time.time() - start_time
            average_tokens_per_second = total_tokens / elapsed
            logger.info(f"\nStep: {step}/{config['train_steps']} ({100.0 * step / config['train_steps']:.1f}%)")
            logger.info(f"  Train Loss: {loss:.6f}")
            if config["disable_validation"]:
                logger.info(f"  Val Loss: disabled")
            else:
                logger.info(f"  Val Loss: {val_loss:.6f}")
            logger.info(f"  Time: {elapsed:.2f}s ({elapsed/60:.2f}m)")
            logger.info(f"  Tokens: {total_tokens:,} ({average_tokens_per_second:.1f} tokens/s)")
            logger.info(f"  LR: {config['lr']}, Beta1: {config['beta1']}, Beta2: {config['beta2']}, WD: {config['weight_decay']}")
            logger.info(f"  Attention Implementation: {config['attention_implementation']}")
            if config["enable_wsd"]:
                logger.info(f"  WSD Schedule: enabled (warmup fraction: {config['warmup_fraction']}, decay fraction: {config['decay_fraction']})")
            else:
                logger.info(f"  WSD Schedule: disabled")
            logger.info(f"  Precision: mixed bfloat16 + RoPE\n")
    
    # Save results
    results_data = {
        'metrics': metrics_history,
        'config': config,
        'num_params': num_params,
        'optimizer_type': 'adamw',
        'precision': 'mixed_bfloat16_rope'
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    wsd_suffix = ""
    if config["enable_wsd"]:
        wsd_suffix = f"_wsd_{config['warmup_fraction']}_{config['decay_fraction']}"
    
    results_filename = (
        f"{config['results_dir']}/nanogpt_adamw_baseline_mixed_bf16_rope_{timestamp}_"
        f"steps_{config['train_steps']}_bs_{config['batch_size']}_"
        f"seq_{config['seq_len']}_"
        f"lr_{config['lr']}_beta1_{config['beta1']}_beta2_{config['beta2']}_wd_{config['weight_decay']}_"
        f"attn_{config['attention_implementation']}{wsd_suffix}.pkl"
    )
    
    with open(results_filename, 'wb') as f:
        pickle.dump(results_data, f)
    
    print(f"Results saved to {results_filename}")
    
    return results_data

if __name__ == "__main__":
    main()