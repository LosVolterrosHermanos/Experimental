#!/usr/bin/env python
"""
NanoGPT training with Tanea optimizer, FSDP sharding, Hydra configuration, and picodo-style training loop.
Integrates:
1. FSDP sharding across feature dimensions (like picodo)
2. Hydra + OmegaConf configuration system (like picodo)
3. Async logging training loop without scan blocks (like picodo)
4. Same model implementation as original nanogpt_tanea.py (Flax Linen)
"""

import os
import time
import numpy as np
import pickle
import logging
import functools
from typing import Dict, List, Any
from tqdm import tqdm

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
import wandb

# Import from the gpt2 directory
import sys
sys.path.append('../dana-nonquadratic-tests/gpt2')
from nanogpt_minimal import count_params
from nanogpt_rope_mixed_precision_v3 import ModelConfig, get_model_config
from nanogpt_fsdp import FSDBGPTWithRoPE, create_fsdp_model_config
from fineweb_dataset import FineWebDataset, create_fineweb_datasets

import jax
# Enable bfloat16 for matrix multiplications only
jax.config.update('jax_default_matmul_precision', 'bfloat16')

import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from power_law_rf.optimizers import powerlaw_schedule, tanea_optimizer, TaneaOptimizerState
import optax
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from flax import linen as nn

LOG_STEPS_BASE = 1.01
TAU_ORDER_STATS_BASE = 2.0

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_tau_order_statistics(tau_vector):
    """Compute order statistics for tau vector in a jittable way."""
    n = len(tau_vector)
    if n == 0:
        return np.array([]), np.array([])
    
    # Sort in ascending order
    sorted_tau_asc = jnp.sort(tau_vector)
    
    # Compute powers of 1.1 up to n, similar to evaluation times
    max_k = jnp.ceil(jnp.log(n) / jnp.log(1.1)).astype(jnp.int32)
    indices = jnp.int32(1.1 ** jnp.arange(max_k + 1)) - 1  # 0-indexed
    
    # Remove duplicates and clamp to valid range
    indices = jnp.unique(indices)
    indices = jnp.minimum(indices, n - 1)
    
    # Get smallest order statistics 
    smallest_order_stats = sorted_tau_asc[indices]
    
    # Get largest order statistics using reversed indices
    reversed_indices = (n - 1) - indices
    largest_order_stats = sorted_tau_asc[reversed_indices]
    
    return largest_order_stats, smallest_order_stats


def extract_tau_statistics(opt_state):
    """Extract tau statistics from TaneaOptimizerState."""
    # Handle optax.chain optimizer - extract the Tanea state
    tanea_state = opt_state
    if hasattr(opt_state, '__len__') and len(opt_state) > 1:
        # optax.chain creates a tuple: (clip_state, tanea_state, ...)
        tanea_state = opt_state[1]
    
    if not isinstance(tanea_state, TaneaOptimizerState):
        return {}
    
    def compute_tau_order_stats_wrapper(x):
        if x is None:
            return None
        else:
            u,v = compute_tau_order_statistics(jnp.ravel(x))
            return np.array(u), np.array(v)

    # Flatten tau tree into a single vector
    tau_stats = jax.tree.map(compute_tau_order_stats_wrapper, tanea_state.tau)
    
    return tau_stats




def create_sharded_train_state(config: DictConfig, model, key, mesh):
    """Create sharded training state with FSDP."""
    inputs = jax.ShapeDtypeStruct(shape=(1, config.model.block_size), dtype=jnp.int32)
    
    def init(rng, inputs):
        # Initialize model parameters
        variables = model.init(rng, inputs)
        params = variables['params']
        
        # Initialize Tanea optimizer
        g2 = powerlaw_schedule(config.tanea.g2, 0.0, 0.0, 1)
        g3 = powerlaw_schedule(config.tanea.g3, 0.0, -1.0*config.tanea.kappa, 1)
        delta = powerlaw_schedule(1.0, 0.0, -1.0, config.tanea.delta)
        wdscheduler = powerlaw_schedule(1.0*config.tanea.weight_decay, 0.0, -1.0*config.tanea.power_weight_decay, config.tanea.weight_decay_ts)
        tanea = tanea_optimizer(g2=g2, g3=g3, Delta=delta, wd=wdscheduler, 
                                momentum_flavor=config.tanea.momentum_flavor, clipsnr=config.tanea.clipsnr,
                                y_dtype=jnp.float32)

        # Create optimizer chain
        optimizer_chain = [optax.clip_by_global_norm(config.tanea.grad_clip), tanea]
        
        # Add WSD schedule if enabled
        if config.tanea.enable_wsd:
            num_train_steps = config.training.num_tokens_train // (config.training.batch_size_train * config.model.block_size)
            if config.tanea.warmup_fraction == 0.0 and config.tanea.decay_fraction == 1.0:
                wsd_schedule = lambda t : 1.0
            elif config.tanea.warmup_fraction == 0.0 and config.tanea.decay_fraction < 1.0:
                wsd_schedule = lambda t : jnp.minimum( (1.0 - (t/num_train_steps))/(1.0 - config.tanea.decay_fraction),1.0)
            elif config.tanea.warmup_fraction > 0.0 and config.tanea.decay_fraction == 1.0:
                wsd_schedule = lambda t : jnp.minimum( t/(num_train_steps*config.tanea.warmup_fraction),1.0)
            else:
                wsd_schedule = lambda t : jnp.minimum(jnp.minimum( t/(num_train_steps*config.tanea.warmup_fraction), (1.0 - (t/num_train_steps))/(1.0 - config.tanea.decay_fraction)),1.0)
            
            optimizer_chain.append(optax.scale_by_schedule(wsd_schedule))
        
        optimizer = optax.chain(*optimizer_chain)
        
        # Wrap with gradient accumulation if needed
        if config.training.grad_accumulation_steps > 1:
            optimizer = optax.MultiSteps(optimizer, every_k_schedule=config.training.grad_accumulation_steps)
        
        return TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer
        )
    
    # Create sharded initialization
    params_shape = jax.eval_shape(init, key, inputs)
    shardings = nn.get_sharding(params_shape, mesh)
    state = jax.jit(init, out_shardings=shardings)(key, inputs)
    return shardings, state


def loss_fn(params, apply_fn, batch):
    """Compute loss function."""
    x, y, weights = batch
    logits = apply_fn({'params': params}, x, deterministic=True)
    # Compute cross-entropy loss
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    # Apply sample weights and return mean
    weighted_loss = jnp.sum(losses * weights) / jnp.sum(weights)
    return weighted_loss


@functools.partial(jax.jit, static_argnames=['apply_fn'])
def train_step(state, apply_fn, batch):
    """Single training step with gradient computation."""
    loss, grads = jax.value_and_grad(loss_fn)(state.params, apply_fn, batch)
    new_state = state.apply_gradients(grads=grads)
    return loss, new_state


@functools.partial(jax.jit, static_argnames=['apply_fn'])  
def eval_step(params, apply_fn, batch):
    """Single evaluation step."""
    return loss_fn(params, apply_fn, batch)


def get_in_out(batch):
    """Extract input and output sequences from batch (compatible with fineweb dataset)."""
    x, y, weights = batch
    return x, y, weights


def train_and_evaluate(config: DictConfig):
    """Main training and evaluation function using picodo-style training loop."""
    
    # Setup logging
    logger.info(f"Starting training with config: {OmegaConf.to_yaml(config)}")
    
    # JAX setup
    logger.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
    logger.info('JAX local devices: %r', jax.local_devices())
    logger.info('Total devices available: %d', jax.device_count())
    
    # Create device mesh for FSDP
    mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), ('data',))
    logger.info(f"Created device mesh for FSDP: {mesh}")
    
    # Validate batch sizes
    if config.training.batch_size_train % jax.device_count() != 0:
        raise ValueError(f"Training batch size ({config.training.batch_size_train}) must be divisible by device count ({jax.device_count()})")
    if config.training.batch_size_valid % jax.device_count() != 0:
        raise ValueError(f"Validation batch size ({config.training.batch_size_valid}) must be divisible by device count ({jax.device_count()})")
    
    # Create datasets
    data_root = os.path.expanduser(config.data.root_path)
    train_dataset, val_dataset = create_fineweb_datasets(
        data_root,
        val_max_tokens=config.training.num_tokens_valid,
        val_files_count=1
    )
    
    # Create data iterators
    train_iterator = train_dataset.iterate_once(config.training.batch_size_train, config.model.block_size)
    
    # Calculate training steps
    num_train_steps = config.training.num_tokens_train // (config.training.batch_size_train * config.model.block_size)
    num_valid_steps = config.training.num_tokens_valid // (config.training.batch_size_valid * config.model.block_size)
    
    logger.info(f"Training steps: {num_train_steps}")
    logger.info(f"Validation steps: {num_valid_steps}")
    
    # Initialize model
    base_config = get_model_config(config.model.size)
    base_config.rope_base = config.model.rope_base
    base_config.attention_implementation = config.model.attention_implementation
    
    # Create FSDP model config
    model_config = create_fsdp_model_config(
        base_config, 
        fsdp_enabled=config.model.fsdp_enabled,
        dtype=config.model.get('dtype', None)
    )
    
    # Create FSDP model
    model = FSDBGPTWithRoPE(
        config=model_config,
        init_std=config.model.init_std,
        fsdp_enabled=config.model.fsdp_enabled
    )
    
    # Initialize sharded state
    key = jax.random.PRNGKey(config.seed)
    with mesh:
        shardings, state = create_sharded_train_state(config, model, key, mesh)
    
    num_params = count_params(state.params)
    logger.info(f"Model initialized with {num_params:,} parameters")
    logger.info(f"FSDP enabled: {config.model.fsdp_enabled}")
    
    # Create JIT-compiled functions
    train_step_fn = functools.partial(train_step, apply_fn=state.apply_fn)
    eval_step_fn = functools.partial(eval_step, apply_fn=state.apply_fn)
    
    # Setup wandb
    if config.wandb.project is not None:
        wandb.init(
            project=config.wandb.project,
            config=OmegaConf.to_container(config, resolve=True),
            mode=config.wandb.mode
        )
    
    # Create logging steps
    LOG_STEPS = jnp.unique(jnp.concatenate([
        jnp.array([0]),
        jnp.int32(LOG_STEPS_BASE**jnp.arange(1, jnp.ceil(jnp.log(num_train_steps)/jnp.log(LOG_STEPS_BASE)))),
        jnp.array([num_train_steps])
    ]))
    
    TAU_ORDER_STATS_STEPS = jnp.unique(jnp.concatenate([
        jnp.array([0]),
        jnp.int32(TAU_ORDER_STATS_BASE**jnp.arange(1, jnp.ceil(jnp.log(num_train_steps)/jnp.log(TAU_ORDER_STATS_BASE)))),
        jnp.array([num_train_steps])
    ]))
    
    # Training loop (picodo style - no scan blocks, async logging)
    with mesh:
        pbar = tqdm(range(num_train_steps), desc="Training")
        start_time = time.time()
        
        # Storage for metrics
        metrics_history = {'step': [], 'train_loss': [], 'val_loss': [], 'tokens_processed': [], 'time_elapsed': []}
        tau_statistics = {'timestamps': [], 'tau_statistics': []}
        
        # Async logging variables
        pending_train_metrics = None
        pending_eval_metrics = None
        
        for step in pbar:
            # Training step
            batch = next(train_iterator)
            # Add sharding constraints for input data
            x, y, w = get_in_out(batch)
            x = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P("data")))
            y = jax.lax.with_sharding_constraint(y, NamedSharding(mesh, P("data")))
            w = jax.lax.with_sharding_constraint(w, NamedSharding(mesh, P("data")))
            sharded_batch = (x, y, w)
            
            loss, state = train_step_fn(state, sharded_batch)
            
            train_metrics = {
                'train_loss': float(loss),
                'train_tokens_seen': (step + 1) * config.training.batch_size_train * config.model.block_size
            }
            
            # Update progress bar
            pbar.set_postfix(loss=f"{loss:.4f}")
            
            # Async logging - process previous metrics while dispatching current step
            if pending_train_metrics is not None:
                if config.wandb.project is not None:
                    wandb.log(pending_train_metrics, step-1)
            pending_train_metrics = train_metrics
            
            if pending_eval_metrics is not None:
                if config.wandb.project is not None:
                    wandb.log(pending_eval_metrics, step-1)
                pending_eval_metrics = None
            
            # Evaluation step (non-blocking)
            if ((step + 1) % config.training.eval_every_steps == 0) or ((step + 1) == num_train_steps):
                val_iterator = val_dataset.iterate_once(config.training.batch_size_valid, config.model.block_size)
                val_losses = []
                for val_step in range(min(config.training.val_steps, num_valid_steps)):
                    val_batch = next(val_iterator)
                    val_x, val_y, val_w = get_in_out(val_batch)
                    val_x = jax.lax.with_sharding_constraint(val_x, NamedSharding(mesh, P("data")))
                    val_y = jax.lax.with_sharding_constraint(val_y, NamedSharding(mesh, P("data")))
                    val_w = jax.lax.with_sharding_constraint(val_w, NamedSharding(mesh, P("data")))
                    val_sharded_batch = (val_x, val_y, val_w)
                    
                    val_loss = eval_step_fn(state.params, val_sharded_batch)
                    val_losses.append(val_loss)
                
                pending_eval_metrics = {'eval_loss': float(jnp.mean(jnp.array(val_losses)))}
            
            # Detailed logging at specified steps
            if (step + 1) in LOG_STEPS:
                jax.block_until_ready(loss)
                elapsed = time.time() - start_time
                total_tokens = (step + 1) * config.training.batch_size_train * config.model.block_size
                
                metrics_history['step'].append(step + 1)
                metrics_history['train_loss'].append(float(loss))
                metrics_history['val_loss'].append(pending_eval_metrics['eval_loss'] if pending_eval_metrics else float('nan'))
                metrics_history['tokens_processed'].append(total_tokens)
                metrics_history['time_elapsed'].append(elapsed)
                
                logger.info(f"\nStep: {step + 1}/{num_train_steps} ({100.0 * (step + 1) / num_train_steps:.1f}%)")
                logger.info(f"  Train Loss: {loss:.6f}")
                logger.info(f"  Val Loss: {pending_eval_metrics['eval_loss'] if pending_eval_metrics else 'N/A':.6f}")
                logger.info(f"  Time: {elapsed:.2f}s ({elapsed/60:.2f}m)")
                logger.info(f"  Tokens: {total_tokens:,} ({total_tokens/elapsed:.1f} tokens/s)")
                logger.info(f"  Multi-GPU throughput: {total_tokens/elapsed/jax.device_count():.1f} tokens/s per device")
                logger.info(f"  FSDP enabled: {config.model.fsdp_enabled}")
                logger.info(f"  Tanea params: g2={config.tanea.g2}, g3={config.tanea.g3}, delta={config.tanea.delta}\n")
            
            # Tau statistics collection
            if (step + 1) in TAU_ORDER_STATS_STEPS:
                tau_stats = extract_tau_statistics(state.opt_state)
                if tau_stats:
                    tau_statistics['timestamps'].append(step + 1)
                    tau_statistics['tau_statistics'].append(tau_stats)
        
        # Final logging
        if config.wandb.project is not None:
            wandb.log(pending_train_metrics, step)
            if pending_eval_metrics:
                wandb.log(pending_eval_metrics, step)
    
    # Save results
    results_data = {
        'metrics': metrics_history,
        'tau_statistics': tau_statistics,
        'config': OmegaConf.to_container(config, resolve=True),
        'num_params': num_params,
        'fsdp_enabled': config.model.fsdp_enabled,
        'num_devices': jax.device_count()
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_filename = f"results/nanogpt_tanea_fsdp_{timestamp}.pkl"
    os.makedirs("results", exist_ok=True)
    
    with open(results_filename, 'wb') as f:
        pickle.dump(results_data, f)
    
    logger.info(f"Results saved to {results_filename}")
    return results_data


@hydra.main(version_base=None, config_path='configs', config_name='nanogpt_tanea_fsdp')
def main(config: DictConfig):
    """Main entry point using Hydra configuration."""
    train_and_evaluate(config)


if __name__ == '__main__':
    main()