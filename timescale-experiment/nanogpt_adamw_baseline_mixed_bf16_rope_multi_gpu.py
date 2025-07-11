#!/usr/bin/env python
"""
NanoGPT training with AdamW optimizer using mixed precision (bfloat16 matmuls, float32 everything else) and RoPE.
Multi-GPU data parallel version for 4 GPU systems.
Based on nanogpt_adamw_baseline_mixed_bf16_rope.py with multi-GPU support from nanodo patterns.
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
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import optax
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from flax import linen as nn

LOG_STEPS_BASE = 1.1
INIT_STD = 0.02

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='train_adamw_multi_gpu.log',  # Separate log file
    filemode='w'
)
logger = logging.getLogger(__name__)

def _init_train_state_sharded(config, model, key, mesh):
    """Creates a sharded training state for multi-GPU training."""
    inputs = jax.ShapeDtypeStruct(shape=(1, config["seq_len"]), dtype=jnp.int32)
    
    def init(rng, inputs):
        params = model.init(rng)
        
        # Initialize AdamW optimizer with optional WSD schedule
        if config["enable_wsd"]:
            # Create WSD (Warmup-Stable-Decay) schedule using linear_onecycle_schedule
            wsd_schedule = optax.schedules.linear_onecycle_schedule(
                config["train_steps"],
                config["lr"],
                pct_start=config["warmup_fraction"],
                pct_final=config["decay_fraction"],
                div_factor=1.0,
                final_div_factor=10000.0
            )
            
            optimizer = optax.chain(
                optax.clip_by_global_norm(config["grad_clip"]),
                optax.adamw(
                    learning_rate=wsd_schedule,
                    b1=config["beta1"],
                    b2=config["beta2"],
                    weight_decay=config["weight_decay"]
                )
            )
        else:
            optimizer = optax.chain(
                optax.clip_by_global_norm(config["grad_clip"]),
                optax.adamw(
                    learning_rate=config["lr"],
                    b1=config["beta1"],
                    b2=config["beta2"],
                    weight_decay=config["weight_decay"]
                )
            )
        
        return TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer)
    
    params_shape = jax.eval_shape(init, key, inputs)
    shardings = nn.get_sharding(params_shape, mesh)
    state = jax.jit(init, out_shardings=shardings)(key, inputs)
    return shardings, state

@jax.jit
def train_step_sharded(state: TrainState, x: jnp.ndarray, y: jnp.ndarray, mesh: Mesh):
    """Sharded training step for multi-GPU data parallelism."""
    # Add sharding constraints for input data
    x = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P("data")))
    y = jax.lax.with_sharding_constraint(y, NamedSharding(mesh, P("data")))
    
    def loss_fn(params: FrozenDict) -> jnp.ndarray:
        logits = state.apply_fn(params, x, False)
        # Loss computation in float32 for numerical stability
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        return loss

    loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return loss, new_state

def parse_args():
    parser = argparse.ArgumentParser(description="Train nanogpt with AdamW optimizer using mixed precision (bfloat16 matmuls) and RoPE on multiple GPUs")
    parser.add_argument(
        "--train_steps", type=int, default=10000,
        help="Number of training steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Total training batch size (will be divided across GPUs)"
    )
    parser.add_argument(
        "--seq_len", type=int, default=1024,
        help="Sequence length for training"
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=64,
        help="Total validation batch size (will be divided across GPUs)"
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
        "--lr", type=float, default=3e-4,
        help="Learning rate for AdamW optimizer"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.9,
        help="Beta1 parameter for AdamW"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999,
        help="Beta2 parameter for AdamW"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01,
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
        "--warmup_fraction", type=float, default=0.1,
        help="Fraction of training steps for warmup phase (default: 0.1)"
    )
    parser.add_argument(
        "--decay_fraction", type=float, default=0.0,
        help="Final decay fraction for WSD schedule (default: 0.0)"
    )
    # Checkpoint parameters
    parser.add_argument(
        "--disable_checkpoint", action="store_true",
        help="Disable saving model weights checkpoint"
    )
    return parser.parse_args()

def evaluate_validation_loss(state, val_dataset, config, mesh, val_steps=20):
    """Evaluate validation loss with multi-GPU support"""
    total_loss = 0.0
    steps_taken = 0
    
    # Calculate per-device batch size
    per_device_val_batch_size = config["val_batch_size"] // jax.device_count()
    
    # Create a fresh iterator each time we evaluate
    val_iterator = val_dataset.iterate_once(per_device_val_batch_size, config["seq_len"])
    
    for x, y, w in val_iterator:
        if steps_taken >= val_steps:
            break
            
        loss, _ = train_step_sharded(state, x, y, mesh)  # Don't update state for validation
        total_loss += loss
        steps_taken += 1
    
    if steps_taken == 0:
        return float('inf')  # Return inf if no validation data
    
    return total_loss / steps_taken

def main():
    """
    Train NanoGPT with AdamW optimizer using mixed precision, RoPE, and multi-GPU data parallelism.
    """
    args = parse_args()
    
    # Log JAX device information
    logger.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
    logger.info('JAX local devices: %r', jax.local_devices())
    logger.info('Total devices available: %d', jax.device_count())
    
    # Validate batch size is divisible by device count
    if args.batch_size % jax.device_count() != 0:
        raise ValueError(f"Batch size ({args.batch_size}) must be divisible by the number of devices ({jax.device_count()})")
    
    if args.val_batch_size % jax.device_count() != 0:
        raise ValueError(f"Validation batch size ({args.val_batch_size}) must be divisible by the number of devices ({jax.device_count()})")
    
    # Calculate per-device batch sizes
    per_device_batch_size = args.batch_size // jax.device_count()
    per_device_val_batch_size = args.val_batch_size // jax.device_count()
    
    logger.info(f"Total batch size: {args.batch_size}, per-device batch size: {per_device_batch_size}")
    logger.info(f"Total validation batch size: {args.val_batch_size}, per-device validation batch size: {per_device_val_batch_size}")
    
    # Create device mesh for data parallelism
    mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), ("data",))
    logger.info(f"Created device mesh: {mesh}")
    
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
        "per_device_batch_size": per_device_batch_size,
        "seq_len": args.seq_len,
        "val_batch_size": args.val_batch_size,
        "per_device_val_batch_size": per_device_val_batch_size,
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
        "disable_checkpoint": args.disable_checkpoint,
        "enable_wsd": args.enable_wsd,
        "warmup_fraction": args.warmup_fraction,
        "decay_fraction": args.decay_fraction,
        "precision": "mixed_bfloat16_rope",
        "num_devices": jax.device_count()
    }
    
    # Create LOG_STEPS
    LOG_STEPS = jnp.unique(jnp.concatenate([
        jnp.array([0]),
        jnp.int32(LOG_STEPS_BASE**jnp.arange(1, jnp.ceil(jnp.log(config["train_steps"])/jnp.log(LOG_STEPS_BASE)))),
        jnp.array([config["train_steps"]])
    ]))
    
    # Initialize model with mixed precision
    key = jax.random.PRNGKey(0)
    model_config = ModelConfig(
        rope_base=config["rope_base"],
        attention_implementation=config["attention_implementation"]
    )
    model = GPTWithRoPE(model_config, mixed_precision=True, init_std=config["init_std"])
    
    # Initialize sharded train state
    shardings, state = _init_train_state_sharded(config, model, key, mesh)
    num_params = count_params(state.params)
    
    logger.info(f"Model initialized with {num_params:,} parameters")
    logger.info("Using mixed precision (bfloat16 matmuls, float32 everything else) with RoPE")
    logger.info(f"Multi-GPU data parallelism enabled with {jax.device_count()} devices")
    logger.info(f"Attention implementation: {config['attention_implementation']}")
    logger.info(f"Validation: {'disabled' if config['disable_validation'] else 'enabled'}")
    logger.info(f"Optimizer: AdamW (lr={config['lr']}, β1={config['beta1']}, β2={config['beta2']}, wd={config['weight_decay']})")
    logger.info(f"Gradient clipping: {config['grad_clip']}")
    
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
    
    # Create training iterator with per-device batch size
    train_iterator = train_dataset.iterate_once(config["per_device_batch_size"], config["seq_len"])
    
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
        
        # Forward and backward pass with sharding
        loss, state = train_step_sharded(state, x, y, mesh)
        
        # Update progress bar
        pbar.set_postfix(loss=f"{loss:.4f}")
        
        # Log metrics at specified steps
        if step in LOG_STEPS:
            # Evaluate validation loss (if enabled)
            if config["disable_validation"]:
                val_loss = float('nan')  # Use NaN to indicate disabled validation
            else:
                val_loss = evaluate_validation_loss(state, val_dataset, config, mesh, config["val_steps"])
            
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
            logger.info(f"  Multi-GPU throughput: {average_tokens_per_second/jax.device_count():.1f} tokens/s per device")
            logger.info(f"  Optimizer: AdamW (lr={config['lr']}, β1={config['beta1']}, β2={config['beta2']}, wd={config['weight_decay']})")
            logger.info(f"  Attention Implementation: {config['attention_implementation']}")
            logger.info(f"  Gradient Clipping: {config['grad_clip']}")
            if config["enable_wsd"]:
                logger.info(f"  WSD Schedule: enabled (warmup fraction: {config['warmup_fraction']}, decay fraction: {config['decay_fraction']})")
            else:
                logger.info(f"  WSD Schedule: disabled")
            logger.info(f"  Precision: mixed bfloat16 + RoPE, {jax.device_count()}-GPU data parallel\n")
    
    # Save results
    results_data = {
        'metrics': metrics_history,
        'config': config,
        'num_params': num_params,
        'optimizer_type': 'adamw',
        'precision': 'mixed_bfloat16_rope',
        'multi_gpu': True,
        'num_devices': jax.device_count()
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    wsd_suffix = ""
    if config["enable_wsd"]:
        wsd_suffix = f"_wsd_{config['warmup_fraction']}_{config['decay_fraction']}"
    
    results_filename = (
        f"{config['results_dir']}/nanogpt_adamw_baseline_mixed_bf16_rope_multi_gpu_{timestamp}_"
        f"steps_{config['train_steps']}_bs_{config['batch_size']}_"
        f"seq_{config['seq_len']}_devices_{jax.device_count()}_"
        f"lr_{config['lr']}_beta1_{config['beta1']}_beta2_{config['beta2']}_wd_{config['weight_decay']}_"
        f"attn_{config['attention_implementation']}{wsd_suffix}.pkl"
    )
    
    with open(results_filename, 'wb') as f:
        pickle.dump(results_data, f)
    
    print(f"Results saved to {results_filename}")
    
    # Save checkpoint of weights (if enabled)
    if not config["disable_checkpoint"]:
        checkpoint_dir = "weight-checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_filename = (
            f"{checkpoint_dir}/nanogpt_adamw_checkpoint_mixed_bf16_rope_multi_gpu_{timestamp}_"
            f"steps_{config['train_steps']}_bs_{config['batch_size']}_"
            f"seq_{config['seq_len']}_devices_{jax.device_count()}_"
            f"lr_{config['lr']}_beta1_{config['beta1']}_beta2_{config['beta2']}_wd_{config['weight_decay']}_"
            f"attn_{config['attention_implementation']}{wsd_suffix}.pkl"
        )
        
        checkpoint_data = {
            'params': state.params,
            'config': config,
            'num_params': num_params,
            'optimizer_type': 'adamw',
            'precision': 'mixed_bfloat16_rope',
            'multi_gpu': True,
            'num_devices': jax.device_count(),
            'final_train_loss': float(loss),
            'final_val_loss': float(val_loss) if 'val_loss' in locals() and not config["disable_validation"] else None
        }
        
        with open(checkpoint_filename, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        print(f"Checkpoint saved to {checkpoint_filename}")
    else:
        print("Checkpoint saving disabled")
    
    return results_data

if __name__ == "__main__":
    main()