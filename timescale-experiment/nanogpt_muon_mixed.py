#!/usr/bin/env python
"""
NanoGPT training with Muon optimizer for transformer layers and AdamW for embedding/readout layers.
Uses mixed precision (bfloat16 matmuls, float32 everything else) and RoPE.
Multi-GPU data parallel version for 4 GPU systems.
Based on nanogpt_adamw_baseline.py with Muon integration.
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
import functools
from typing import Dict, List, Any
from tqdm import tqdm

#recommended flags for faster training
# os.environ['XLA_FLAGS'] = (
#     '--xla_gpu_enable_triton_softmax_fusion=true '
#     '--xla_gpu_triton_gemm_any=True '
#     '--xla_gpu_enable_async_collectives=true '
#     '--xla_gpu_enable_latency_hiding_scheduler=true '
#     '--xla_gpu_enable_highest_priority_async_stream=true '
# )

# Import from the gpt2 directory
import sys
sys.path.append('../dana-nonquadratic-tests/gpt2')
from nanogpt_minimal import count_params
from nanogpt_rope_mixed_precision_v4 import GPTWithRoPE, ModelConfig, get_model_config
from fineweb_dataset import FineWebDataset, create_fineweb_datasets

# Import Muon optimizer
sys.path.append('../muon')
from _muon import muon, scale_by_muon

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

LOG_STEPS_BASE = 1.01
INIT_STD = 0.02
SCAN_BLOCK_SIZE = 10

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='train_muon_mixed_multi_gpu.log',  # Separate log file
    filemode='w'
)
logger = logging.getLogger(__name__)

def create_param_labels(params):
    """Create parameter labels for partitioning between Muon and AdamW optimizers.
    
    Args:
        params: Parameter tree from the model
        
    Returns:
        Label tree with 'muon' for 2D weight matrices and 'adamw' for 1D parameters and embeddings
    """
    def label_fn(path, param):
        path_str = '.'.join(str(p) for p in path)
        # Use AdamW for embedding and readout layers
        if 'wte' in path_str or 'wpe' in path_str or 'ln_f' in path_str or 'head' in path_str:
            return 'adamw'
        # Use Muon only for 2D weight matrices (kernel parameters)
        # Use AdamW for 1D parameters (biases, layer norm scales/biases)
        elif param.ndim == 2:
            return 'muon'
        else:
            return 'adamw'
    
    return jax.tree_util.tree_map_with_path(label_fn, params)

def _init_train_state_sharded(config, model, key, mesh):
    """Creates a sharded training state for multi-GPU training."""
    inputs = jax.ShapeDtypeStruct(shape=(1, config["seq_len"]), dtype=jnp.int32)
    
    def init(rng, inputs):
        params = model.init(rng)
        
        # Create parameter labels for partitioning
        param_labels = create_param_labels(params)
        
        # Create Muon optimizer for transformer layers
        muon_transforms = []
        muon_transforms.append(optax.clip_by_global_norm(config["grad_clip"]))
        muon_transforms.append(scale_by_muon(
            ns_coeffs=config["muon_ns_coeffs"],
            ns_steps=config["muon_ns_steps"],
            beta=config["muon_beta"],
            eps=config["muon_eps"],
            mu_dtype=jnp.float32,
            nesterov=config["muon_nesterov"],
            adaptive=config["muon_adaptive"]
        ))
        muon_transforms.append(optax.add_decayed_weights(config["muon_weight_decay"]))
        muon_transforms.append(optax.scale_by_learning_rate(config["muon_lr"]))
        
        # Create AdamW optimizer for embedding/readout layers  
        adamw_transforms = []
        adamw_transforms.append(optax.clip_by_global_norm(config["grad_clip"]))
        adamw_transforms.append(optax.adamw(
            learning_rate=config["adamw_lr"],
            b1=config["adamw_beta1"],
            b2=config["adamw_beta2"],
            weight_decay=config["adamw_weight_decay"],
            mu_dtype=jnp.float32
        ))
        
        # Create base optimizer using partition
        base_optimizer = optax.partition(
            transforms={
                'muon': optax.chain(*muon_transforms),
                'adamw': optax.chain(*adamw_transforms)
            },
            param_labels=param_labels
        )
        
        # Apply WSD schedule if enabled
        if config["enable_wsd"]:
            # Create WSD (Warmup-Stable-Decay) schedule
            if config["warmup_fraction"] == 0.0 and config["decay_fraction"] == 1.0:
                wsd_schedule = lambda t : 1.0
            elif config["warmup_fraction"] == 0.0 and config["decay_fraction"] < 1.0:
                wsd_schedule = lambda t : jnp.minimum( (1.0 - (t/(config["train_steps"])))/(1.0 - config["decay_fraction"]),1.0)
            elif config["warmup_fraction"] > 0.0 and config["decay_fraction"] == 1.0:
                wsd_schedule = lambda t : jnp.minimum( t/(config["train_steps"]*config["warmup_fraction"]),1.0)
            else:
                wsd_schedule = lambda t : jnp.minimum(jnp.minimum( t/(config["train_steps"]*config["warmup_fraction"]), (1.0 - (t/(config["train_steps"])))/(1.0 - config["decay_fraction"])),1.0)
            
            base_optimizer = optax.chain(
                base_optimizer,
                optax.scale_by_schedule(wsd_schedule)
            )

        # Wrap with MultiSteps for gradient accumulation
        if config["grad_accumulation_steps"] > 1:
            optimizer = optax.MultiSteps(base_optimizer, every_k_schedule=config["grad_accumulation_steps"])
        else:
            optimizer = base_optimizer
        
        return TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer)
    
    params_shape = jax.eval_shape(init, key, inputs)
    shardings = nn.get_sharding(params_shape, mesh)
    state = jax.jit(init, out_shardings=shardings)(key, inputs)
    return shardings, state

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

def eval_step_sharded(state: TrainState, x: jnp.ndarray, y: jnp.ndarray, mesh: Mesh):
    """Sharded evaluation step for multi-GPU data parallelism (no gradient computation)."""
    # Add sharding constraints for input data
    x = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P("data")))
    y = jax.lax.with_sharding_constraint(y, NamedSharding(mesh, P("data")))
    
    # Forward pass only - no gradients
    logits = state.apply_fn(state.params, x, False)
    # Loss computation in float32 for numerical stability
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    return loss

def create_train_block_fn(mesh):
    """Create a JIT-compiled train block function with mesh frozen."""
    
    # Create the train step function with mesh frozen
    train_step_fn = jax.jit(functools.partial(train_step_sharded, mesh=mesh))
    
    @jax.jit
    def train_block(state, data_batch):
        """Train for multiple steps using jax.lax.scan without host-side control flow."""
        def train_step_scan(carry_state, batch_data):
            x, y, w = batch_data
            loss, new_state = train_step_fn(carry_state, x, y)
            return new_state, loss
        
        # Run scan over the data batch
        final_state, losses = jax.lax.scan(
            train_step_scan,
            state,
            data_batch  # Shape: (scan_batch_size, batch_size, seq_len)
        )
        return final_state, losses
    
    return train_block

def create_eval_block_fn(mesh):
    """Create a JIT-compiled evaluation block function with mesh frozen."""
    
    # Create the eval step function with mesh frozen
    eval_step_fn = jax.jit(functools.partial(eval_step_sharded, mesh=mesh))
    
    @jax.jit
    def eval_block(state, data_batch):
        """Evaluate multiple steps using jax.lax.scan without host-side control flow."""
        def eval_step_scan(carry_state, batch_data):
            x, y, w = batch_data
            loss = eval_step_fn(carry_state, x, y)
            return carry_state, loss  # State doesn't change during evaluation
        
        # Run scan over the data batch
        _, losses = jax.lax.scan(
            eval_step_scan,
            state,
            data_batch  # Shape: (val_steps, batch_size, seq_len)
        )
        return losses
    
    return eval_block

def parse_args():
    parser = argparse.ArgumentParser(description="Train nanogpt with Muon+AdamW mixed optimizer using mixed precision (bfloat16 matmuls) and RoPE on multiple GPUs")
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
    # Muon hyperparameters
    parser.add_argument(
        "--muon_lr", type=float, default=3e-4,
        help="Learning rate for Muon optimizer (transformer layers)"
    )
    parser.add_argument(
        "--muon_beta", type=float, default=0.95,
        help="Beta parameter for Muon momentum"
    )
    parser.add_argument(
        "--muon_eps", type=float, default=1e-8,
        help="Epsilon for Muon optimizer"
    )
    parser.add_argument(
        "--muon_weight_decay", type=float, default=0.01,
        help="Weight decay for Muon optimizer"
    )
    parser.add_argument(
        "--muon_ns_steps", type=int, default=5,
        help="Newton-Schulz iteration steps for Muon"
    )
    parser.add_argument(
        "--muon_nesterov", action="store_true",
        help="Enable Nesterov momentum for Muon"
    )
    parser.add_argument(
        "--muon_adaptive", action="store_true",
        help="Enable adaptive scaling for Muon"
    )
    # AdamW hyperparameters for embeddings
    parser.add_argument(
        "--adamw_lr", type=float, default=3e-4,
        help="Learning rate for AdamW optimizer (embedding/readout layers)"
    )
    parser.add_argument(
        "--adamw_beta1", type=float, default=0.9,
        help="Beta1 parameter for AdamW"
    )
    parser.add_argument(
        "--adamw_beta2", type=float, default=0.95,
        help="Beta2 parameter for AdamW"
    )
    parser.add_argument(
        "--adamw_weight_decay", type=float, default=0.01,
        help="Weight decay parameter for AdamW"
    )
    parser.add_argument(
        "--data_root", type=str, default="../dana-nonquadratic-tests/gpt2/fineweb-edu/sample/10BT",
        help="Root directory for training data"
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
        "--decay_fraction", type=float, default=1.0,
        help="Final decay fraction for WSD schedule (default: 0.0)"
    )
    # Checkpoint parameters
    parser.add_argument(
        "--disable_checkpoint", action="store_true",
        help="Disable saving model weights checkpoint"
    )
    parser.add_argument(
        "--model_size", type=str, default="GPT2-nano",
        choices=["GPT2-nano", "GPT2-medium", "GPT2-large", "GPT2-jumbo"],
        help="Model size to use"
    )
    parser.add_argument(
        "--grad_accumulation_steps", type=int, default=1,
        help="Number of gradient accumulation steps (default: 1, no accumulation)"
    )
    return parser.parse_args()

def evaluate_validation_loss(state, val_dataset, config, eval_block_fn, val_steps=20):
    """Evaluate validation loss with multi-GPU support using scan blocks"""
    
    # Create a fresh iterator each time we evaluate validation loss, which will be sharded across devices
    val_iterator = val_dataset.iterate_once(config["val_batch_size"], config["seq_len"])
    
    # Collect validation data for the entire block
    val_data = []
    steps_collected = 0
    
    for x, y, w in val_iterator:
        if steps_collected >= val_steps:
            break
        val_data.append((x, y, w))
        steps_collected += 1
    
    if steps_collected == 0:
        return float('inf')  # Return inf if no validation data
    
    # Stack the data into arrays for scan
    # Shape: (steps_collected, batch_size, seq_len)
    val_x = jnp.stack([item[0] for item in val_data])
    val_y = jnp.stack([item[1] for item in val_data])
    val_w = jnp.stack([item[2] for item in val_data])
    val_data_batch = (val_x, val_y, val_w)
    
    # Run the evaluation scan block
    val_losses = eval_block_fn(state, val_data_batch)
    
    # Return average validation loss
    return jnp.mean(val_losses)

def main():
    """
    Train NanoGPT with Muon+AdamW mixed optimizer using mixed precision, RoPE, and multi-GPU data parallelism.
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
    
    # Create JIT-compiled train step function with mesh frozen
    train_step_fn = jax.jit(functools.partial(train_step_sharded, mesh=mesh))
    # Create JIT-compiled eval step function with mesh frozen
    eval_step_fn = jax.jit(functools.partial(eval_step_sharded, mesh=mesh))
    
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
    
    # Default Newton-Schulz coefficients for Muon
    muon_ns_coeffs = (3.4445, -4.7750, 2.0315)
    
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
        "muon_lr": args.muon_lr,
        "muon_beta": args.muon_beta,
        "muon_eps": args.muon_eps,
        "muon_weight_decay": args.muon_weight_decay,
        "muon_ns_coeffs": muon_ns_coeffs,
        "muon_ns_steps": args.muon_ns_steps,
        "muon_nesterov": args.muon_nesterov,
        "muon_adaptive": args.muon_adaptive,
        "adamw_lr": args.adamw_lr,
        "adamw_beta1": args.adamw_beta1,
        "adamw_beta2": args.adamw_beta2,
        "adamw_weight_decay": args.adamw_weight_decay,
        "rope_base": args.rope_base,
        "attention_implementation": args.attention_implementation,
        "disable_validation": args.disable_validation,
        "disable_checkpoint": args.disable_checkpoint,
        "enable_wsd": args.enable_wsd,
        "warmup_fraction": args.warmup_fraction,
        "decay_fraction": args.decay_fraction,
        "model_size": args.model_size,
        "grad_accumulation_steps": args.grad_accumulation_steps,
        "precision": "mixed_bfloat16_rope",
        "num_devices": jax.device_count()
    }
    
    # Calculate number of blocks and create block-based logging steps
    total_blocks = (config["train_steps"] + SCAN_BLOCK_SIZE - 1) // SCAN_BLOCK_SIZE
    
    # Create LOG_STEPS based on blocks (then convert back to step numbers)
    LOG_BLOCKS = jnp.unique(jnp.concatenate([
        jnp.array([0]),
        jnp.int32(LOG_STEPS_BASE**jnp.arange(1, jnp.ceil(jnp.log(total_blocks)/jnp.log(LOG_STEPS_BASE)))),
        jnp.array([total_blocks])
    ]))
    # Convert block numbers to step numbers (end of each block)
    LOG_STEPS = jnp.minimum(LOG_BLOCKS * SCAN_BLOCK_SIZE, config["train_steps"])
    
    # Initialize model with mixed precision
    key = jax.random.PRNGKey(0)
    model_config = get_model_config(config["model_size"])
    model_config.rope_base = config["rope_base"]
    model_config.attention_implementation = config["attention_implementation"]
    model = GPTWithRoPE(model_config, init_std=config["init_std"])
    
    # Initialize sharded train state
    shardings, state = _init_train_state_sharded(config, model, key, mesh)
    num_params = count_params(state.params)
    
    # Create JIT-compiled train block function with mesh frozen
    train_block_fn = create_train_block_fn(mesh)
    # Create JIT-compiled eval block function with mesh frozen
    eval_block_fn = create_eval_block_fn(mesh)
    
    logger.info(f"Model initialized with {num_params:,} parameters")
    logger.info("Using mixed precision (bfloat16 model, float32 optimizer) with RoPE")
    logger.info(f"Multi-GPU data parallelism enabled with {jax.device_count()} devices")
    logger.info(f"Attention implementation: {config['attention_implementation']}")
    logger.info(f"Validation: {'disabled' if config['disable_validation'] else 'enabled'}")
    logger.info(f"Optimizer: Muon (transformer layers) + AdamW (embedding/readout layers)")
    logger.info(f"Muon params: lr={config['muon_lr']}, β={config['muon_beta']}, wd={config['muon_weight_decay']}, ns_steps={config['muon_ns_steps']}")
    logger.info(f"AdamW params: lr={config['adamw_lr']}, β1={config['adamw_beta1']}, β2={config['adamw_beta2']}, wd={config['adamw_weight_decay']}")
    logger.info(f"Gradient clipping: {config['grad_clip']}")
    
    # Initialize datasets using the new utility function
    data_root = os.path.expanduser(args.data_root)
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
    
    # Create training iterator with full batch size, which JAX will automatically shard across devices
    train_iterator = train_dataset.iterate_once(config["batch_size"], config["seq_len"])
    
    # Storage for losses and metrics
    metrics_history = {
        'step': [],
        'train_loss': [],
        'val_loss': [],
        'tokens_processed': [],
        'time_elapsed': []
    }
    
    # Training loop with scan blocks
    pbar = tqdm(range(total_blocks), desc="Training Blocks")
    start_time = time.time()
    current_step = 0
    final_train_loss = 0.0  # Initialize final loss tracking
    
    for block_idx in pbar:
        # Calculate how many steps to process in this block
        remaining_steps = config["train_steps"] - current_step
        block_size = min(SCAN_BLOCK_SIZE, remaining_steps)
        
        if block_size <= 0:
            break
        
        # Collect data for the entire block
        block_data = []
        for _ in range(block_size):
            x, y, w = next(train_iterator)
            block_data.append((x, y, w))
        
        # Stack the data into arrays for scan
        # Shape: (block_size, batch_size, seq_len)
        block_x = jnp.stack([item[0] for item in block_data])
        block_y = jnp.stack([item[1] for item in block_data])
        block_w = jnp.stack([item[2] for item in block_data])
        data_batch = (block_x, block_y, block_w)
        
        # Run the scan block
        state, block_losses = train_block_fn(state, data_batch)
        
        # Calculate average loss for this block and update progress bar
        avg_block_loss = jnp.mean(block_losses)
        final_train_loss = float(avg_block_loss)  # Update final loss
        current_step += block_size
        pbar.set_postfix(loss=f"{avg_block_loss:.4f}", step=f"{current_step}/{config['train_steps']}")
        
        # Log metrics at specified steps (check at end of each block)
        if current_step in LOG_STEPS:
            jax.block_until_ready(avg_block_loss)
            # Evaluate validation loss (if enabled)
            if config["disable_validation"]:
                val_loss = float('nan')  # Use NaN to indicate disabled validation
            else:
                val_loss = evaluate_validation_loss(state, val_dataset, config, eval_block_fn, config["val_steps"])
            
            total_tokens = current_step * config["batch_size"] * config["seq_len"]
            metrics_history['step'].append(current_step)
            metrics_history['train_loss'].append(float(avg_block_loss))
            metrics_history['val_loss'].append(float(val_loss))
            metrics_history['tokens_processed'].append(total_tokens)
            metrics_history['time_elapsed'].append(time.time() - start_time)
            
            # Print detailed metrics
            elapsed = time.time() - start_time
            average_tokens_per_second = total_tokens / elapsed
            effective_batch_size = config["batch_size"] * config["grad_accumulation_steps"]
            logger.info(f"\nStep: {current_step}/{config['train_steps']} ({100.0 * current_step / config['train_steps']:.1f}%)")
            logger.info(f"  Train Loss: {avg_block_loss:.6f}")
            if config["disable_validation"]:
                logger.info(f"  Val Loss: disabled")
            else:
                logger.info(f"  Val Loss: {val_loss:.6f}")
            logger.info(f"  Time: {elapsed:.2f}s ({elapsed/60:.2f}m)")
            logger.info(f"  Tokens: {total_tokens:,} ({average_tokens_per_second:.1f} tokens/s)")
            logger.info(f"  Multi-GPU throughput: {average_tokens_per_second/jax.device_count():.1f} tokens/s per device")
            if config["grad_accumulation_steps"] > 1:
                logger.info(f"  Batch Size: {config['batch_size']} per step, {effective_batch_size} effective (grad accumulation: {config['grad_accumulation_steps']})")
            else:
                logger.info(f"  Batch Size: {config['batch_size']}")
            logger.info(f"  Optimizer: Muon (transformer) + AdamW (embedding/readout)")
            logger.info(f"  Muon: lr={config['muon_lr']}, β={config['muon_beta']}, wd={config['muon_weight_decay']}")
            logger.info(f"  AdamW: lr={config['adamw_lr']}, β1={config['adamw_beta1']}, β2={config['adamw_beta2']}, wd={config['adamw_weight_decay']}")
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
        'optimizer_type': 'muon_adamw_mixed',
        'precision': 'mixed_bfloat16_rope',
        'multi_gpu': True,
        'num_devices': jax.device_count()
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    wsd_suffix = ""
    if config["enable_wsd"]:
        wsd_suffix = f"_wsd_{config['warmup_fraction']}_{config['decay_fraction']}"
    
    results_filename = (
        f"{config['results_dir']}/nanogpt_muon_mixed_baseline_mixed_bf16_rope_multi_gpu_{timestamp}_"
        f"steps_{config['train_steps']}_bs_{config['batch_size']}_"
        f"seq_{config['seq_len']}_devices_{jax.device_count()}_"
        f"muon_lr_{config['muon_lr']}_adamw_lr_{config['adamw_lr']}_"
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
            f"{checkpoint_dir}/nanogpt_muon_mixed_checkpoint_mixed_bf16_rope_multi_gpu_{timestamp}_"
            f"steps_{config['train_steps']}_bs_{config['batch_size']}_"
            f"seq_{config['seq_len']}_devices_{jax.device_count()}_"
            f"muon_lr_{config['muon_lr']}_adamw_lr_{config['adamw_lr']}_"
            f"attn_{config['attention_implementation']}{wsd_suffix}.pkl"
        )
        
        checkpoint_data = {
            'params': state.params,
            'config': config,
            'num_params': num_params,
            'optimizer_type': 'muon_adamw_mixed',
            'precision': 'mixed_bfloat16_rope',
            'multi_gpu': True,
            'num_devices': jax.device_count(),
            'final_train_loss': final_train_loss,
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