#!/usr/bin/env python
"""
Modified version of nanogpt_log_losses_adana.py to use fineweb-edu dataset
with validation evaluation and plotting.
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

# Try to import directly
from nanogpt_minimal import ModelConfig, TextDataset, init_train_state, train_step, count_params, GPT
import jax
import jax.numpy as jnp
import optimizers
import optax
from flax import linen as nn
from flax.core import FrozenDict
from flax.training.train_state import TrainState

LOG_STEPS_BASE = 1.01

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    parser = argparse.ArgumentParser(description="Train nanogpt with fineweb-edu dataset")
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
        help="Validation batch size (comparable to Karpathy's)"
    )
    parser.add_argument(
        "--val_max_tokens", type=int, default=None,
        help="Maximum tokens to load for validation (default: val_batch_size * seq_len * 30)"
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
    # Add Dana hyperparameters
    parser.add_argument(
        "--dana_g2", type=float, default=1.0,
        help="DANA G2 parameter"
    )
    parser.add_argument(
        "--dana_delta", type=float, default=8.0,
        help="DANA Delta parameter"
    )
    parser.add_argument(
        "--dana_g3_iv", type=float, default=0.2,
        help="DANA G3 initial value"
    )
    parser.add_argument(
        "--dana_g3_sv", type=float, default=0.0,
        help="DANA G3 saturation value"
    )
    parser.add_argument(
        "--dana_g3_p", type=float, default=-0.8,
        help="DANA G3 power"
    )
    parser.add_argument(
        "--dana_g3_ts", type=float, default=1.0,
        help="DANA G3 time scale"
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

def create_learning_curve_plot(step_numbers, train_losses, val_losses, num_params, config):
    """Create a log-log plot of the learning curve with both training and validation losses."""
    # Convert to numpy arrays
    steps = np.array(step_numbers)
    train_loss_values = np.array(train_losses)
    val_loss_values = np.array(val_losses)
    
    # Calculate tokens processed
    tokens_per_step = config["batch_size"] * config["seq_len"]
    tokens = steps * tokens_per_step
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot both training and validation curves
    ax.loglog(tokens, train_loss_values, 'o-', color='blue', label='Training Loss', markersize=4)
    ax.loglog(tokens, val_loss_values, 's-', color='red', label='Validation Loss', markersize=4)
    
    # Fit power law to training data if we have enough points
    if len(tokens) > 4 and tokens[-1] >= 1e6:
        # Find index where tokens first exceeds 1M
        start_idx = np.where(tokens >= 1e6)[0][0]
        # Linear fit in log-log space from 1M tokens onwards
        log_tokens = np.log(tokens + 1)
        log_train_loss = np.log(train_loss_values)
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_tokens[start_idx:], log_train_loss[start_idx:])
        # Power law parameters: loss = A * tokens^beta
        A = np.exp(intercept)
        beta = slope
        fit_loss = A * ((tokens + 1) ** beta)
        ax.loglog(tokens, fit_loss, '--', color='green', label=f'Training Fit: {A:.4f} t^({beta:.4f})', alpha=0.7)
    
    # Set axis labels and title
    ax.set_xlabel('Training Tokens')
    ax.set_ylabel('Loss')
    ax.set_ylim(min(min(train_loss_values), min(val_loss_values)) * 0.9, 12)
    ax.set_title(f'NanoGPT Language Model Learning Curve (FineWeb-Edu)\n{num_params:,} parameters')
    
    # Convert x-axis to better format
    def format_tokens(x, pos):
        if x >= 1e6:
            return f'{x/1e6:.1f}M'
        elif x >= 1e3:
            return f'{x/1e3:.1f}K'
        else:
            return f'{x:.0f}'
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_tokens))
    
    # Add grid
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend()
    
    # Generate a more descriptive filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = (
        f"{config['results_dir']}/nanogpt_dana_fineweb_learning_curve_{timestamp}_"
        f"steps_{config['train_steps']}_bs_{config['batch_size']}_"
        f"seq_{config['seq_len']}_"
        f"g2_{config['dana_g2']}_delta_{config['dana_delta']}_g3iv_{config['dana_g3_iv']}_"
        f"g3sv_{config['dana_g3_sv']}_g3p_{config['dana_g3_p']}_g3ts_{config['dana_g3_ts']}.pdf"
    )
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Learning curve plot saved as {output_filename}")

def modify_nanogpt_for_fineweb():
    """
    Train NanoGPT with fineweb-edu dataset and validation evaluation.
    """
    args = parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Configuration dictionary
    # Set default val_max_tokens if not specified
    val_max_tokens = args.val_max_tokens
    if val_max_tokens is None:
        # Default to enough tokens for 30 validation batches (with some buffer)
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
        "dana_g2": args.dana_g2,
        "dana_delta": args.dana_delta,
        "dana_g3_iv": args.dana_g3_iv,
        "dana_g3_sv": args.dana_g3_sv,
        "dana_g3_p": args.dana_g3_p,
        "dana_g3_ts": args.dana_g3_ts
    }
    
    # Create LOG_STEPS
    LOG_STEPS = jnp.unique(jnp.concatenate([
        jnp.array([0]),
        jnp.int32(LOG_STEPS_BASE**jnp.arange(1, jnp.ceil(jnp.log(config["train_steps"])/jnp.log(LOG_STEPS_BASE)))),
        jnp.array([config["train_steps"]])
    ]))
    
    # Initialize DANA optimizer
    g1 = optimizers.powerlaw_schedule(1.0, 0.0, 0.0, 1)
    g2 = optimizers.powerlaw_schedule(config["dana_g2"], 0.0, 0.0, 1)
    g3 = optimizers.powerlaw_schedule(config["dana_g3_iv"], 0.0, config["dana_g3_p"], 1)
    Delta = optimizers.powerlaw_schedule(1.0, 0.0, -1.0, config["dana_delta"])
    dana = optimizers.dana_optimizer(g1=g1, g2=g2, g3=g3, Delta=Delta)
    
    dana = optax.chain(
        optax.clip_by_global_norm(config['grad_clip']),
        dana
    )
    optimizer = dana
    
    # Initialize model
    key = jax.random.PRNGKey(0)
    model = GPT(ModelConfig())
    params = model.init(key)
    num_params = count_params(params)
    
    # Initialize train state
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer)
    
    # Get parquet files and split for train/val
    data_root = os.path.expanduser("./fineweb-edu/sample/10BT")
    #data_root = os.path.expanduser("~/fineweb-edu/sample/10BT")

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
            
            # Print detailed metrics
            elapsed = time.time() - start_time
            average_tokens_per_second = total_tokens / elapsed
            tqdm.write(f"\nStep: {step}/{config['train_steps']} ({100.0 * step / config['train_steps']:.1f}%)")
            tqdm.write(f"  Train Loss: {loss:.6f}")
            tqdm.write(f"  Val Loss: {val_loss:.6f}")
            tqdm.write(f"  Time: {elapsed:.2f}s ({elapsed/60:.2f}m)")
            tqdm.write(f"  Tokens: {total_tokens:,} ({average_tokens_per_second:.1f} tokens/s)")
            tqdm.write(f"  G2: {config['dana_g2']}, G3_iv: {config['dana_g3_iv']}, g3p: {config['dana_g3_p']}\n")
    
    # Create CHECKPOINTS directory
    os.makedirs("CHECKPOINTS", exist_ok=True)
    
    # Save model parameters
    checkpoint_path = f"CHECKPOINTS/model_dana_fineweb_step_{config['train_steps']}.pkl"
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(state.params, f)
    print(f"Saved checkpoint to {checkpoint_path}")
    
    # Create log-log plot of losses
    create_learning_curve_plot(
        metrics_history['step'],
        metrics_history['train_loss'],
        metrics_history['val_loss'],
        num_params,
        config
    )
    
    # Save metrics history with configuration
    metrics_data = {
        'metrics': metrics_history,
        'config': config,
        'num_params': num_params
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    metrics_filename = (
        f"{config['results_dir']}/nanogpt_dana_fineweb_metrics_{timestamp}_"
        f"steps_{config['train_steps']}_bs_{config['batch_size']}_"
        f"seq_{config['seq_len']}_"
        f"g2_{config['dana_g2']}_delta_{config['dana_delta']}_g3iv_{config['dana_g3_iv']}_"
        f"g3sv_{config['dana_g3_sv']}_g3p_{config['dana_g3_p']}_g3ts_{config['dana_g3_ts']}.pkl"
    )
    
    with open(metrics_filename, 'wb') as f:
        pickle.dump(metrics_data, f)
    
    print(f"Saved metrics data to {metrics_filename}")
    
    return metrics_history, config, num_params

if __name__ == "__main__":
    modify_nanogpt_for_fineweb()