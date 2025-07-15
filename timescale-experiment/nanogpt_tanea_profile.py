#!/usr/bin/env python
"""
NanoGPT training with Tanea optimizer using mixed precision (bfloat16 matmuls, float32 everything else) and RoPE.
Multi-GPU data parallel version for 4 GPU systems.
Based on nanogpt_tanea_tau_stats_mixed_bf16_rope.py with multi-GPU support from nanodo patterns.
"""

import os
import signal
import time
import numpy as np
import pickle
import argparse
import logging
import functools
from typing import Dict, List, Any
from tqdm import tqdm

# Import from the gpt2 directory
import sys
sys.path.append('../dana-nonquadratic-tests/gpt2')
from nanogpt_minimal import count_params
from nanogpt_rope_mixed_precision_v2 import GPTWithRoPE, ModelConfig, get_model_config
from fineweb_dataset import FineWebDataset, create_fineweb_datasets

import jax
import jax.profiler
# Enable bfloat16 for matrix multiplications only
jax.config.update('jax_default_matmul_precision', 'bfloat16')

import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from power_law_rf.optimizers import powerlaw_schedule, tanea_optimizer
import optax
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from flax import linen as nn

LOG_STEPS_BASE = 1.01
INIT_STD = 0.02

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='train_multi_gpu.log',  # Separate log file
    filemode='w'
)
logger = logging.getLogger(__name__)



def _init_train_state_sharded(config, model, key, mesh):
    """Creates a sharded training state for multi-GPU training."""
    inputs = jax.ShapeDtypeStruct(shape=(1, config["seq_len"]), dtype=jnp.int32)
    
    def init(rng, inputs):
        params = model.init(rng)
        
        # Initialize Tanea optimizer
        g2 = powerlaw_schedule(config["tanea_g2"], 0.0, 0.0, 1)
        g3 = powerlaw_schedule(config["tanea_g3"], 0.0, -1.0*config["tanea_kappa"], 1)
        delta = powerlaw_schedule(1.0, 0.0, -1.0, config["tanea_delta"])
        wdscheduler = powerlaw_schedule(1.0*config["weight_decay"], 0.0, -1.0*config["power_weight_decay"], config["weight_decay_ts"])
        tanea = tanea_optimizer(g2=g2, g3=g3, Delta=delta, wd=wdscheduler, momentum_flavor=config["momentum_flavor"], clipsnr=config["clipsnr"])

        # Create optimizer chain with optional WSD schedule
        if config["enable_wsd"]:
            # Create WSD (Warmup-Stable-Decay) schedule
            if config["warmup_fraction"] == 0.0 and config["decay_fraction"] == 0.0:
                wsd_schedule = lambda t : 1.0
            elif config["warmup_fraction"] == 0.0 and config["decay_fraction"] > 0.0:
                wsd_schedule = lambda t : jnp.minimum( (1.0 - (t/(config["train_steps"])))/(1.0 - config["decay_fraction"]),1.0)
            elif config["warmup_fraction"] > 0.0 and config["decay_fraction"] == 0.0:
                wsd_schedule = lambda t : jnp.minimum( t/(config["train_steps"]*config["warmup_fraction"]),1.0)
            else:
                wsd_schedule = lambda t : jnp.minimum(jnp.minimum( t/(config["train_steps"]*config["warmup_fraction"]), (1.0 - (t/(config["train_steps"])))/(1.0 - config["decay_fraction"])),1.0)

            optimizer = optax.chain(
                optax.clip_by_global_norm(config["grad_clip"]),
                tanea,
                optax.scale_by_schedule(wsd_schedule)
            )
        else:
            optimizer = optax.chain(
                optax.clip_by_global_norm(config["grad_clip"]),
                tanea
            )
        
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
        # Loss computation in float32
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        return loss

    loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return loss, new_state

def parse_args():
    parser = argparse.ArgumentParser(description="Train nanogpt with Tanea optimizer using mixed precision (bfloat16 matmuls) and RoPE on multiple GPUs")
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
        "--data_root", type=str, default="../dana-nonquadratic-tests/gpt2/fineweb-edu/sample/10BT",
        help="Root directory for training data"
    )
    parser.add_argument(
        "--momentum_flavor", type=str, default="effective-clip",
        choices=["effective-clip", "theory", "always-on", "always-on-mk2", "strong-clip", "mk2", "mk3"],
        help="Tanea momentum flavor"
    )
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
    # Gradient clipping parameters
    parser.add_argument(
        "--grad_clip", type=float, default=2.0,
        help="Gradient clipping threshold (default: 2.0, set to 0 to disable)"
    )
    # Clipsnr parameter for Tanea optimizer
    parser.add_argument(
        "--clipsnr", type=float, default=2.0,
        help="Clipping factor for signal-to-noise ratio in Tanea optimizer (default: 2.0)"
    )
    parser.add_argument(
        "--model_size", type=str, default="GPT2-nano",
        choices=["GPT2-nano", "GPT2-medium", "GPT2-large", "GPT2-jumbo"],
        help="Model size to use"
    )
    # Profiling parameters
    parser.add_argument(
        "--enable_profiler", action="store_true",
        help="Enable JAX profiler around the main training loop"
    )
    parser.add_argument(
        "--profiler_dir", type=str, default="/tmp/nanogpt_profile",
        help="Directory to save profiler traces"
    )
    return parser.parse_args()


def main():
    """
    Train NanoGPT with Tanea optimizer using mixed precision, RoPE, and multi-GPU data parallelism.
    """
    args = parse_args()
    
    # Log JAX device information
    logger.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
    logger.info('JAX local devices: %r', jax.local_devices())
    logger.info('Total devices available: %d', jax.device_count())
    
    # Validate batch size is divisible by device count
    if args.batch_size % jax.device_count() != 0:
        raise ValueError(f"Batch size ({args.batch_size}) must be divisible by the number of devices ({jax.device_count()})")
    
    # Calculate per-device batch sizes
    per_device_batch_size = args.batch_size // jax.device_count()
    
    logger.info(f"Total batch size: {args.batch_size}, per-device batch size: {per_device_batch_size}")
    
    # Create device mesh for data parallelism
    mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), ("data",))
    logger.info(f"Created device mesh: {mesh}")
    
    # Create JIT-compiled train step function with mesh frozen
    train_step_fn = jax.jit(functools.partial(train_step_sharded, mesh=mesh))
    
    # Override INIT_STD if provided
    global INIT_STD
    INIT_STD = args.init_std
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Configuration dictionary
    config = {
        "train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "per_device_batch_size": per_device_batch_size,
        "seq_len": args.seq_len,
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
        "enable_wsd": args.enable_wsd,
        "warmup_fraction": args.warmup_fraction,
        "decay_fraction": args.decay_fraction,
        "rope_base": args.rope_base,
        "attention_implementation": args.attention_implementation,
        "grad_clip": args.grad_clip,
        "clipsnr": args.clipsnr,
        "model_size": args.model_size,
        "precision": "mixed_bfloat16_rope",
        "num_devices": jax.device_count(),
        "enable_profiler": args.enable_profiler,
        "profiler_dir": args.profiler_dir
    }
    
    # Create LOG_STEPS
    LOG_STEPS = jnp.unique(jnp.concatenate([
        jnp.array([0]),
        jnp.int32(LOG_STEPS_BASE**jnp.arange(1, jnp.ceil(jnp.log(config["train_steps"])/jnp.log(LOG_STEPS_BASE)))),
        jnp.array([config["train_steps"]])
    ]))
    
    # Initialize model with mixed precision
    key = jax.random.PRNGKey(0)
    model_config = get_model_config(config["model_size"])
    model_config.rope_base = config["rope_base"]
    model_config.attention_implementation = config["attention_implementation"]
    model = GPTWithRoPE(model_config, init_std=config["init_std"])
    
    # Initialize sharded train state
    shardings, state = _init_train_state_sharded(config, model, key, mesh)
    num_params = count_params(state.params)
    
    logger.info(f"Model initialized with {num_params:,} parameters")
    logger.info("Using mixed precision (bfloat16 matmuls, float32 everything else) with RoPE")
    logger.info(f"Multi-GPU data parallelism enabled with {jax.device_count()} devices")
    logger.info(f"Attention implementation: {config['attention_implementation']}")
    logger.info(f"Gradient clipping: {config['grad_clip']}")
    logger.info(f"Optimizer: Tanea (momentum_flavor={config['momentum_flavor']})")
    logger.info(f"Tanea params: g2={config['tanea_g2']}, g3={config['tanea_g3']}, delta={config['tanea_delta']}, kappa={config['tanea_kappa']}")
    logger.info(f"Profiler: {'enabled' if config['enable_profiler'] else 'disabled'}")
    if config['enable_profiler']:
        logger.info(f"Profiler directory: {config['profiler_dir']}")
    
    # Initialize training dataset
    data_root = os.path.expanduser(args.data_root)
    import glob
    
    # Find all parquet files for training only
    parquet_files = sorted(glob.glob(os.path.join(data_root, "*.parquet")))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_root}")
    
    logger.info(f"Found {len(parquet_files)} parquet files in {data_root}")
    logger.info(f"Using all {len(parquet_files)} files for training (no validation)")
    
    # Create training dataset using all files
    train_dataset = FineWebDataset(parquet_files)
    
    # Create training iterator with full batch size, which JAX will automatically shard across devices
    train_iterator = train_dataset.iterate_once(config["batch_size"], config["seq_len"])
    
    
    # Training loop
    pbar = tqdm(range(config["train_steps"]), desc="Training")
    start_time = time.time()
    tokens_processed = 0
    
    # Start profiler if enabled
    if config["enable_profiler"]:
        os.makedirs(config["profiler_dir"], exist_ok=True)
        jax.profiler.start_trace(config["profiler_dir"])
        logger.info(f"Started JAX profiler, saving to {config['profiler_dir']}")

    step_tokens = config["batch_size"] * config["seq_len"]  # Tokens per iteration
    
    try:
        for step in pbar:
            # Get next batch
            x, y, w = next(train_iterator)
            
            # Forward and backward pass with sharding
            loss, state = train_step_fn(state, x, y)
            
            # Calculate tokens processed
            tokens_processed += step_tokens
            
            # Get iteration rate from tqdm and convert to tokens/sec
            if hasattr(pbar, 'format_dict') and pbar.format_dict.get('rate'):
                iterations_per_sec = pbar.format_dict['rate']
                tokens_per_sec = iterations_per_sec * step_tokens
            else:
                tokens_per_sec = 0
            
            # Update progress bar with loss and tokens/sec
            pbar.set_postfix(loss=f"{loss:.4f}", **{"token/s": f"{tokens_per_sec:,.0f}"})
    finally:
        # Stop profiler if enabled
        if config["enable_profiler"]:
            jax.profiler.stop_trace()
            logger.info(f"Stopped JAX profiler")

    end_time = time.time()
    total_duration = end_time - start_time
    overall_tokens_per_sec = tokens_processed / total_duration if total_duration > 0 else 0
    
    logger.info(f"Training completed in {total_duration:.2f} seconds")
    logger.info(f"Total tokens processed: {tokens_processed:,}")
    logger.info(f"Overall throughput: {overall_tokens_per_sec:.0f} tokens/sec")
    return None

if __name__ == "__main__":
    main()