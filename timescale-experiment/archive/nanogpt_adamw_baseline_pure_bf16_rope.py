#!/usr/bin/env python
"""
NanoGPT training with AdamW optimizer using pure bfloat16 precision and RoPE.
Based on nanogpt_adamw_baseline_pure_bf16.py but with Rotary Position Embedding instead of learned positional embedding.
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
from nanogpt_minimal import ModelConfig, TextDataset, count_params

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.core import FrozenDict
from flax.training.train_state import TrainState

jax.config.update('jax_default_matmul_precision', 'bfloat16')

LOG_STEPS_BASE = 1.1
INIT_STD = 0.02

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

def create_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0, dtype=jnp.bfloat16):
    """Create RoPE cache for rotary position embeddings."""
    # Create frequency tensor
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    
    # Create position tensor
    position = jnp.arange(seq_len, dtype=jnp.float32)
    
    # Create frequency matrix
    freqs = jnp.outer(position, inv_freq)
    
    # Create cos and sin caches
    cos_cache = jnp.cos(freqs).astype(dtype)
    sin_cache = jnp.sin(freqs).astype(dtype)
    
    return cos_cache, sin_cache

def apply_rope(x, cos_cache, sin_cache):
    """Apply rotary position embedding to query or key tensors.
    
    Args:
        x: Input tensor of shape (batch, heads, seq_len, head_dim)
        cos_cache: Cosine cache of shape (seq_len, head_dim//2)
        sin_cache: Sine cache of shape (seq_len, head_dim//2)
    
    Returns:
        Tensor with RoPE applied
    """
    batch, heads, seq_len, head_dim = x.shape
    
    # Split x into even and odd indices
    x_even = x[..., ::2]  # (batch, heads, seq_len, head_dim//2)
    x_odd = x[..., 1::2]  # (batch, heads, seq_len, head_dim//2)
    
    # Get the appropriate cos and sin values for the sequence length
    cos = cos_cache[:seq_len, :]  # (seq_len, head_dim//2)
    sin = sin_cache[:seq_len, :]  # (seq_len, head_dim//2)
    
    # Apply rotation
    # Reshape cos and sin to broadcast properly
    cos = cos[None, None, :, :]  # (1, 1, seq_len, head_dim//2)
    sin = sin[None, None, :, :]  # (1, 1, seq_len, head_dim//2)
    
    # Rotary transformation: [x_even, x_odd] * [[cos, -sin], [sin, cos]]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    
    # Interleave even and odd back together
    out = jnp.zeros_like(x)
    out = out.at[..., ::2].set(out_even)
    out = out.at[..., 1::2].set(out_odd)
    
    return out

class CausalSelfAttention(nn.Module):
    config: ModelConfig

    def setup(self):
        # Initialize dense layers with smaller weights using kernel_init
        self.q_proj = nn.Dense(
            self.config.n_embd,
            kernel_init=nn.initializers.normal(stddev=INIT_STD),
            dtype=jnp.bfloat16
        )
        self.k_proj = nn.Dense(
            self.config.n_embd,
            kernel_init=nn.initializers.normal(stddev=INIT_STD),
            dtype=jnp.bfloat16
        )
        self.v_proj = nn.Dense(
            self.config.n_embd,
            kernel_init=nn.initializers.normal(stddev=INIT_STD),
            dtype=jnp.bfloat16
        )
        self.out_proj = nn.Dense(
            self.config.n_embd,
            kernel_init=nn.initializers.normal(stddev=INIT_STD),
            dtype=jnp.bfloat16
        )
        
        # Create RoPE cache
        head_dim = self.config.n_embd // self.config.n_head
        self.cos_cache, self.sin_cache = create_rope_cache(
            self.config.block_size, head_dim, dtype=jnp.bfloat16
        )

    @nn.compact
    def __call__(self, x, deterministic=True):
        assert len(x.shape) == 3
        B, T, C = x.shape  # batch size, sequence length, embedding dimensionality

        # Keep everything in bfloat16
        x = x.astype(jnp.bfloat16)

        # calculate query, key, values for all heads in batch
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)  # (B, T, C)
        v = self.v_proj(x)  # (B, T, C)

        # reshape to separate heads
        head_dim = C // self.config.n_head
        q = jnp.reshape(q, (B, T, self.config.n_head, head_dim))
        k = jnp.reshape(k, (B, T, self.config.n_head, head_dim))
        v = jnp.reshape(v, (B, T, self.config.n_head, head_dim))

        # transpose to get (B, nh, T, hs)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Apply RoPE to queries and keys
        q = apply_rope(q, self.cos_cache, self.sin_cache)
        k = apply_rope(k, self.cos_cache, self.sin_cache)

        # causal self-attention in bfloat16
        att = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * (jnp.bfloat16(1.0 / jnp.sqrt(head_dim)))
        
        # create causal mask
        mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bfloat16))[None, None, :, :]  # (1, 1, T, T)
        att = jnp.where(mask, att, jnp.bfloat16(-1e4))  # Use -1e4 instead of -inf for bfloat16
        att = jax.nn.softmax(att, axis=-1)
        
        y = jnp.matmul(att, v)  # (B, nh, T, hs)

        # re-assemble all head outputs side by side
        y = jnp.transpose(y, (0, 2, 1, 3))  # (B, T, nh, hs)
        y = jnp.reshape(y, (B, T, C))  # (B, T, C)

        # output projection
        y = self.out_proj(y)
        return y

class MLP(nn.Module):
    config: ModelConfig

    def setup(self):
        # Initialize with smaller weights using kernel_init
        self.fc1 = nn.Dense(
            self.config.n_embd*4,
            kernel_init=nn.initializers.normal(stddev=INIT_STD),
            dtype=jnp.bfloat16
        )
        self.fc2 = nn.Dense(
            self.config.n_embd,
            kernel_init=nn.initializers.normal(stddev=INIT_STD),
            dtype=jnp.bfloat16
        )

    @nn.compact
    def __call__(self, x, deterministic=True):
        x = x.astype(jnp.bfloat16)
        x = self.fc1(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=deterministic)
        x = self.fc2(x)
        x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=deterministic)
        return x

class Block(nn.Module):
  config: ModelConfig

  @nn.checkpoint  # Add gradient checkpointing to save memory
  @nn.compact
  def __call__(self, x):
    x = x.astype(jnp.bfloat16)
    x = nn.LayerNorm(dtype=jnp.bfloat16)(x)
    x = x + CausalSelfAttention(self.config)(x)
    x = nn.LayerNorm(dtype=jnp.bfloat16)(x)
    x = x + MLP(self.config)(x)
    return x

class GPT(nn.Module):
  config: ModelConfig
  def setup(self):
    # Only token embedding - no positional embedding since we use RoPE
    self.wte = nn.Embed(self.config.vocab_size, self.config.n_embd, dtype=jnp.bfloat16)
    self.ln_f = nn.LayerNorm(dtype=jnp.bfloat16)
    self.head = nn.Dense(self.config.vocab_size, 
                         kernel_init=nn.initializers.normal(stddev=INIT_STD*0.5),
                         dtype=jnp.bfloat16)

  @nn.compact
  def __call__(self, x, deterministic=False):
    
    B, T = x.shape
    assert T <= self.config.block_size

    # Only token embedding - RoPE handles position information
    tok_emb = self.wte(x)
    x = tok_emb.astype(jnp.bfloat16)

    for _ in range(self.config.n_layer):
      x = Block(self.config)(x)
    x = self.ln_f(x)
    
    # Final projection - convert to float32 only for loss computation
    logits = self.head(x).astype(jnp.float32)
    return logits
  
  def init(self, rng):
    tokens = jnp.zeros((1, self.config.block_size), dtype=jnp.uint16)
    params = super().init(rng, tokens, True)
    return params

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
    parser = argparse.ArgumentParser(description="Train nanogpt with AdamW optimizer using pure bfloat16 precision and RoPE")
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
        "--beta2", type=float, default=0.95,
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
    Train NanoGPT with AdamW optimizer using pure bfloat16 precision and RoPE.
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
        "lr": args.lr,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "weight_decay": args.weight_decay,
        "rope_base": args.rope_base,
        "precision": "pure_bfloat16_rope"
    }
    
    # Create LOG_STEPS
    LOG_STEPS = jnp.unique(jnp.concatenate([
        jnp.array([0]),
        jnp.int32(LOG_STEPS_BASE**jnp.arange(1, jnp.ceil(jnp.log(config["train_steps"])/jnp.log(LOG_STEPS_BASE)))),
        jnp.array([config["train_steps"]])
    ]))
    
    # Initialize AdamW optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(config['grad_clip']),
        optax.adamw(
            learning_rate=config['lr'],
            b1=config['beta1'],
            b2=config['beta2'],
            weight_decay=config['weight_decay']
        )
    )
    
    # Initialize model
    key = jax.random.PRNGKey(0)
    model = GPT(ModelConfig())
    params = model.init(key)
    num_params = count_params(params)
    
    logger.info(f"Model initialized with {num_params:,} parameters")
    logger.info("Using pure bfloat16 precision with RoPE positional embedding")
    
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
            tqdm.write(f"  LR: {config['lr']}, Beta1: {config['beta1']}, Beta2: {config['beta2']}, WD: {config['weight_decay']}")
            tqdm.write(f"  Precision: pure bfloat16 + RoPE\n")
    
    # Save results
    results_data = {
        'metrics': metrics_history,
        'config': config,
        'num_params': num_params,
        'optimizer_type': 'adamw',
        'precision': 'pure_bfloat16_rope'
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_filename = (
        f"{config['results_dir']}/nanogpt_adamw_baseline_pure_bf16_rope_{timestamp}_"
        f"steps_{config['train_steps']}_bs_{config['batch_size']}_"
        f"seq_{config['seq_len']}_"
        f"lr_{config['lr']}_beta1_{config['beta1']}_beta2_{config['beta2']}_wd_{config['weight_decay']}.pkl"
    )
    
    with open(results_filename, 'wb') as f:
        pickle.dump(results_data, f)
    
    print(f"Results saved to {results_filename}")
    
    return results_data

if __name__ == "__main__":
    main()