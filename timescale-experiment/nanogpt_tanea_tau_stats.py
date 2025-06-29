#!/usr/bin/env python
"""
NanoGPT training with Tanea optimizer and tau statistics tracking.
Based on nanogpt_log_losses_dana.py but using Tanea optimizer instead of DANA.
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
import tiktoken
import glob
import pandas as pd
from typing import Dict, List, Any
from tqdm import tqdm
from dataclasses import dataclass

# Import from the gpt2 directory
import sys
sys.path.append('../dana-nonquadratic-tests/gpt2')
from nanogpt_minimal import ModelConfig, TextDataset, init_train_state, train_step, count_params, GPT

import jax
#jax.config.update('jax_default_dtype_bits', '16')
#jax.config.update('jax_default_matmul_precision', 'bfloat16')
import jax.numpy as jnp
from power_law_rf.optimizers import powerlaw_schedule, tanea_optimizer, TaneaOptimizerState
import optax
from flax import linen as nn
from flax.core import FrozenDict
from flax.training.train_state import TrainState

LOG_STEPS_BASE = 1.1

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_tau_order_statistics(tau_vector):
    """Compute order statistics for tau vector in a jittable way.
    
    Args:
        tau_vector: A 1D array of non-negative tau values
        
    Returns:
        Array of order statistics: [largest, (1.1)^1-th largest, (1.1)^2-th largest, ...]
        where we take the (1.1)^k-th largest for k = 0, 1, 2, ..., up to n
    """
    n = len(tau_vector)
    if n == 0:
        return jnp.array([])
    
    # Sort in descending order
    sorted_tau = jnp.sort(tau_vector)[::-1]
    
    # Compute powers of 1.1 up to n, similar to evaluation times
    max_k = jnp.ceil(jnp.log(n) / jnp.log(1.1)).astype(jnp.int32)
    indices = jnp.int32(1.1 ** jnp.arange(max_k + 1)) - 1  # 0-indexed: [0, 0, 1, 2, 3, 4, ...]
    
    # Remove duplicates and clamp to valid range
    indices = jnp.unique(indices)
    indices = jnp.minimum(indices, n - 1)
    
    return sorted_tau[indices]

def extract_tau_statistics(opt_state):
    """Extract tau statistics from TaneaOptimizerState.
    
    Args:
        opt_state: Optimizer state (may be from optax.chain)
        
    Returns:
        Dictionary with tau statistics
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
    
    # Compute order statistics
    order_stats = compute_tau_order_statistics(tau_vector)
    
    return {
        'tau_order_statistics': order_stats,
        'tau_mean': jnp.mean(tau_vector),
        'tau_std': jnp.std(tau_vector),
        'tau_min': jnp.min(tau_vector),
        'tau_max': jnp.max(tau_vector)
    }

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    n_layer: int = 12
    dropout_rate: float = 0.1

class FineWebDataset:
    """Dataset class that reads parquet files one at a time and tokenizes on-the-fly, similar to TextDataset"""
    def __init__(self, parquet_files, max_tokens=None, is_validation=False):
        self.parquet_files = parquet_files
        self.max_tokens = max_tokens
        self.is_validation = is_validation
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc._special_tokens['<|endoftext|>']
        
        logger.info(f"FineWebDataset initialized with {len(parquet_files)} parquet files")
        
        # For validation, load all data into memory for reuse
        if is_validation:
            logger.info("Loading all validation data into memory for reuse")
            self._all_tokens = None
            self._load_all_validation_data()
    
    def _load_all_validation_data(self):
        """Load all validation data into memory efficiently"""
        import numpy as np
        
        # Use numpy arrays for memory efficiency
        token_chunks = []
        total_tokens = 0
        
        for file_idx, parquet_file in enumerate(self.parquet_files):
            logger.info(f"Loading validation file {file_idx+1}/{len(self.parquet_files)}: {os.path.basename(parquet_file)}")
            
            df = pd.read_parquet(parquet_file)
            
            for text in df['text']:
                # Tokenize text
                tokens = [self.eot]  # Start with end-of-text token
                tokens.extend(self.enc.encode_ordinary(text))
                
                # Convert to numpy array with efficient dtype
                token_array = np.array(tokens, dtype=np.int32)
                token_chunks.append(token_array)
                total_tokens += len(token_array)
                
                # Check max_tokens limit
                if self.max_tokens and total_tokens >= self.max_tokens:
                    logger.info(f"Reached validation max tokens limit: {self.max_tokens}")
                    # Concatenate what we have so far
                    self._all_tokens = np.concatenate(token_chunks, dtype=np.int32)
                    # Trim to exact limit
                    if len(self._all_tokens) > self.max_tokens:
                        self._all_tokens = self._all_tokens[:self.max_tokens]
                    del df
                    return
            
            del df
        
        # Concatenate all token chunks into single array
        if token_chunks:
            self._all_tokens = np.concatenate(token_chunks, dtype=np.int32)
        else:
            self._all_tokens = np.array([], dtype=np.int32)
        
        logger.info(f"Loaded {len(self._all_tokens):,} validation tokens into memory ({self._all_tokens.nbytes / 1024 / 1024:.2f} MB)")
        
    def iterate_once(self, batch_size, seq_len):
        """Iterator that yields batches of (x, y, w) similar to TextDataset"""
        if self.is_validation:
            # For validation, use preloaded data
            return self._iterate_validation_data(batch_size, seq_len)
        else:
            # For training, process files one at a time
            return self._iterate_training_data(batch_size, seq_len)
    
    def _iterate_validation_data(self, batch_size, seq_len):
        """Iterate through preloaded validation data"""
        import numpy as np
        
        if not hasattr(self, '_all_tokens') or len(self._all_tokens) == 0:
            logger.warning("No validation data available")
            return
        
        # Create batches from preloaded tokens
        n = len(self._all_tokens)
        num_batches = n // (batch_size * seq_len)
        
        if num_batches == 0:
            logger.warning("Validation set too small for batch size and sequence length")
            return
        
        # Trim to ensure even division into batches
        tokens = self._all_tokens[:num_batches * batch_size * seq_len]
        
        # Reshape into batches
        token_array = np.array(tokens).reshape(batch_size, -1)
        
        # Create input/target batches
        for i in range(0, token_array.shape[1] - seq_len, seq_len):
            x = token_array[:, i:i+seq_len]
            y = token_array[:, i+1:i+seq_len+1]
            w = np.ones_like(x, dtype=np.uint8)  # All tokens are valid
            
            yield jnp.array(x), jnp.array(y), jnp.array(w)
    
    def _iterate_training_data(self, batch_size, seq_len):
        """Process training files one at a time"""
        import numpy as np
        
        current_tokens = np.array([], dtype=np.int32)
        tokens_yielded = 0
        batch_size_tokens = batch_size * seq_len
        
        # Process one parquet file at a time (like the original)
        for file_idx, parquet_file in enumerate(self.parquet_files):
            logger.info(f"Processing parquet file {file_idx+1}/{len(self.parquet_files)}: {os.path.basename(parquet_file)}")
            
            # Load the current parquet file
            df = pd.read_parquet(parquet_file)
            logger.info(f"Loaded {len(df)} text documents from {os.path.basename(parquet_file)}")
            
            # Process each text document in the file
            for text in df['text']:
                # Tokenize text on-the-fly
                tokens = [self.eot]  # Start with end-of-text token
                tokens.extend(self.enc.encode_ordinary(text))
                
                # Convert to numpy array for efficiency
                new_tokens = np.array(tokens, dtype=np.int32)
                current_tokens = np.concatenate([current_tokens, new_tokens])
                
                # Yield batches when we have enough tokens
                while len(current_tokens) >= batch_size_tokens + 1:
                    # Extract batch
                    batch_tokens = current_tokens[:batch_size_tokens + 1]
                    current_tokens = current_tokens[batch_size_tokens:]
                    
                    # Convert to JAX arrays and reshape
                    x = jnp.array(batch_tokens[:-1]).reshape(batch_size, seq_len)
                    y = jnp.array(batch_tokens[1:]).reshape(batch_size, seq_len)
                    w = jnp.ones_like(x)  # Dummy weights
                    
                    tokens_yielded += batch_size_tokens
                    
                    # Check max_tokens limit
                    if self.max_tokens and tokens_yielded >= self.max_tokens:
                        return
                        
                    yield x, y, w
                
                # Check max_tokens limit after processing each text
                if self.max_tokens and tokens_yielded >= self.max_tokens:
                    return
            
            # Free memory after processing each file
            del df

def parse_args():
    parser = argparse.ArgumentParser(description="Train nanogpt with Tanea optimizer and tau statistics")
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
    Train NanoGPT with Tanea optimizer and collect tau statistics.
    """
    args = parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Configuration dictionary
    val_max_tokens = args.val_max_tokens
    if val_max_tokens is None:
        # Default to enough tokens for 30 validation batches
        val_max_tokens = args.val_batch_size * args.seq_len * 30
    
    config = {
        "train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "val_batch_size": args.val_batch_size,
        "val_max_tokens": val_max_tokens,
        "grad_clip": args.grad_clip,
        "init_std": args.init_std,
        "results_dir": args.results_dir,
        "tanea_g2": args.tanea_g2,
        "tanea_g3": args.tanea_g3,
        "tanea_delta": args.tanea_delta,
        "tanea_kappa": args.tanea_kappa,
        "weight_decay": args.weight_decay,
        "power_weight_decay": args.power_weight_decay,
        "weight_decay_ts": args.weight_decay_ts
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
    tanea = tanea_optimizer(g2=g2, g3=g3, Delta=delta, wd=wdscheduler)

    tanea = optax.chain(
        optax.clip_by_global_norm(config['grad_clip']),
        tanea
    )
    optimizer = tanea
    
    # Initialize model
    key = jax.random.PRNGKey(0)
    model = GPT(ModelConfig())
    params = model.init(key)
    num_params = count_params(params)
    
    logger.info(f"Model initialized with {num_params:,} parameters")
    
    # Initialize train state
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer)
    
    # Get parquet files and split for train/val
    data_root = os.path.expanduser("../dana-nonquadratic-tests/gpt2/fineweb-edu/sample/10BT")
    
    parquet_files = sorted(glob.glob(os.path.join(data_root, "*_00000.parquet")))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_root}")
        
    logger.info(f"Found {len(parquet_files)} parquet files")
    
    # Reserve the first file for validation, rest for training
    val_files = parquet_files[:1]
    train_files = parquet_files[1:]
    
    logger.info(f"Using {len(train_files)} files for training")
    logger.info(f"Using {len(val_files)} files for validation")
    
    # Initialize datasets
    train_dataset = FineWebDataset(train_files)
    val_dataset = FineWebDataset(val_files, max_tokens=config["val_max_tokens"], is_validation=True)
    
    logger.info(f"Validation dataset limited to {config['val_max_tokens']:,} tokens")
    
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
            val_loss = evaluate_validation_loss(state, val_dataset, config)
            
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
            tqdm.write(f"  G2: {config['tanea_g2']}, G3: {config['tanea_g3']}, Delta: {config['tanea_delta']}\n")
    
    # Convert tau statistics lists to arrays
    for key in tau_statistics:
        if key != 'tau_order_statistics':
            tau_statistics[key] = jnp.array(tau_statistics[key])
    
    # Save results
    results_data = {
        'metrics': metrics_history,
        'tau_statistics': tau_statistics,
        'config': config,
        'num_params': num_params
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_filename = (
        f"{config['results_dir']}/nanogpt_tanea_results_{timestamp}_"
        f"steps_{config['train_steps']}_bs_{config['batch_size']}_"
        f"seq_{config['seq_len']}_"
        f"g2_{config['tanea_g2']}_g3_{config['tanea_g3']}_delta_{config['tanea_delta']}.pkl"
    )
    
    with open(results_filename, 'wb') as f:
        pickle.dump(results_data, f)
    
    print(f"Results saved to {results_filename}")
    
    return results_data

if __name__ == "__main__":
    main()