#!/usr/bin/env python
"""
NanoGPT implementation with RoPE (Rotary Position Embedding) and mixed precision support.

This implementation combines token embeddings with rotary position embeddings,
supporting both pure precision modes and mixed precision training.

References:
- RoPE: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
  https://arxiv.org/abs/2104.09864
- Mixed Precision Training: "Mixed Precision Training" 
  https://arxiv.org/abs/1710.03740

The mixed precision mode uses bfloat16 for matrix multiplications and float32 
for other operations to maintain numerical stability while accelerating training.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for NanoGPT model with RoPE.
    
    Args:
        vocab_size: Size of the vocabulary
        n_head: Number of attention heads
        n_embd: Embedding dimension
        block_size: Maximum sequence length
        n_layer: Number of transformer layers
        dropout_rate: Dropout probability
        rope_base: Base frequency for RoPE (default 10000.0 as in the paper)
    """
    vocab_size: int = 50257
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    n_layer: int = 12
    dropout_rate: float = 0.1
    rope_base: float = 10000.0


def create_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0, dtype=jnp.float32):
    """Create RoPE cache for rotary position embeddings.
    
    Implements the rotary position embedding as described in:
    "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    https://arxiv.org/abs/2104.09864
    
    Args:
        seq_len: Maximum sequence length
        head_dim: Dimension of each attention head
        base: Base frequency for the sinusoidal embeddings (default 10000.0)
        dtype: Data type for the cache
    
    Returns:
        Tuple of (cos_cache, sin_cache) with shape (seq_len, head_dim//2)
    """
    # Create frequency tensor following the RoPE paper
    # θ_i = base^(-2i/d) for i ∈ [0, 1, ..., d/2-1]
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    
    # Create position tensor
    position = jnp.arange(seq_len, dtype=jnp.float32)
    
    # Create frequency matrix: outer product of positions and frequencies
    freqs = jnp.outer(position, inv_freq)  # (seq_len, head_dim//2)
    
    # Create cos and sin caches
    cos_cache = jnp.cos(freqs).astype(dtype)
    sin_cache = jnp.sin(freqs).astype(dtype)
    
    return cos_cache, sin_cache


def apply_rope(x, cos_cache, sin_cache):
    """Apply rotary position embedding to query or key tensors.
    
    Implements the rotary transformation as described in the RoPE paper:
    For a vector [x_0, x_1, x_2, x_3, ...], we apply rotation to pairs:
    [x_0, x_1] -> [x_0*cos - x_1*sin, x_0*sin + x_1*cos]
    [x_2, x_3] -> [x_2*cos - x_3*sin, x_2*sin + x_3*cos]
    
    Args:
        x: Input tensor of shape (batch, heads, seq_len, head_dim)
        cos_cache: Cosine cache of shape (seq_len, head_dim//2)
        sin_cache: Sine cache of shape (seq_len, head_dim//2)
    
    Returns:
        Tensor with RoPE applied, same shape as input
    """
    batch, heads, seq_len, head_dim = x.shape
    
    # Split x into even and odd indices (pairs for rotation)
    x_even = x[..., ::2]   # (batch, heads, seq_len, head_dim//2)
    x_odd = x[..., 1::2]   # (batch, heads, seq_len, head_dim//2)
    
    # Get the appropriate cos and sin values for the sequence length
    cos = cos_cache[:seq_len, :]  # (seq_len, head_dim//2)
    sin = sin_cache[:seq_len, :]  # (seq_len, head_dim//2)
    
    # Reshape cos and sin to broadcast properly
    cos = cos[None, None, :, :]  # (1, 1, seq_len, head_dim//2)
    sin = sin[None, None, :, :]  # (1, 1, seq_len, head_dim//2)
    
    # Apply rotary transformation
    # [x_even, x_odd] * [[cos, -sin], [sin, cos]]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    
    # Interleave even and odd back together
    out = jnp.zeros_like(x)
    out = out.at[..., ::2].set(out_even)
    out = out.at[..., 1::2].set(out_odd)
    
    return out


class CausalSelfAttention(nn.Module):
    """Causal self-attention with RoPE position embeddings.
    
    Supports both pure precision and mixed precision modes:
    - Pure mode: All computations in specified dtype
    - Mixed mode: Matrix multiplications in bfloat16, other ops in float32
    """
    config: ModelConfig
    mixed_precision: bool = True
    init_std: float = 0.02

    def setup(self):
        # Determine parameter dtype based on precision mode
        param_dtype = jnp.float32 if self.mixed_precision else jnp.bfloat16
        
        # Initialize projection layers
        self.q_proj = nn.Dense(
            self.config.n_embd,
            kernel_init=nn.initializers.normal(stddev=self.init_std),
            dtype=param_dtype
        )
        self.k_proj = nn.Dense(
            self.config.n_embd,
            kernel_init=nn.initializers.normal(stddev=self.init_std),
            dtype=param_dtype
        )
        self.v_proj = nn.Dense(
            self.config.n_embd,
            kernel_init=nn.initializers.normal(stddev=self.init_std),
            dtype=param_dtype
        )
        self.out_proj = nn.Dense(
            self.config.n_embd,
            kernel_init=nn.initializers.normal(stddev=self.init_std),
            dtype=param_dtype
        )
        
        # Create RoPE cache (always in float32 for numerical stability)
        head_dim = self.config.n_embd // self.config.n_head
        self.cos_cache, self.sin_cache = create_rope_cache(
            self.config.block_size, head_dim, self.config.rope_base, dtype=jnp.float32
        )

    @nn.compact
    def __call__(self, x, deterministic=True):
        assert len(x.shape) == 3
        B, T, C = x.shape  # batch size, sequence length, embedding dimensionality

        if self.mixed_precision:
            # Mixed precision: keep input in float32
            x = x.astype(jnp.float32)
            
            # Cast to bfloat16 for matrix operations, then back to float32
            x_bf16 = x.astype(jnp.bfloat16)
            q = self.q_proj(x_bf16).astype(jnp.float32)
            k = self.k_proj(x_bf16).astype(jnp.float32)
            v = self.v_proj(x_bf16).astype(jnp.float32)
        else:
            # Pure precision mode
            target_dtype = jnp.bfloat16
            x = x.astype(target_dtype)
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

        # Reshape to separate heads
        head_dim = C // self.config.n_head
        q = jnp.reshape(q, (B, T, self.config.n_head, head_dim))
        k = jnp.reshape(k, (B, T, self.config.n_head, head_dim))
        v = jnp.reshape(v, (B, T, self.config.n_head, head_dim))

        # Transpose to get (B, nh, T, hs)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Apply RoPE to queries and keys
        if self.mixed_precision:
            # Apply RoPE in float32
            q = apply_rope(q, self.cos_cache, self.sin_cache)
            k = apply_rope(k, self.cos_cache, self.sin_cache)
        else:
            # Apply RoPE in target dtype
            cos_cache = self.cos_cache.astype(jnp.bfloat16)
            sin_cache = self.sin_cache.astype(jnp.bfloat16)
            q = apply_rope(q, cos_cache, sin_cache)
            k = apply_rope(k, cos_cache, sin_cache)

        # Attention computation
        if self.mixed_precision:
            # Mixed precision: matmul in bfloat16, other ops in float32
            q_bf16 = q.astype(jnp.bfloat16)
            k_bf16 = k.astype(jnp.bfloat16)
            v_bf16 = v.astype(jnp.bfloat16)
            
            att = jnp.matmul(q_bf16, jnp.transpose(k_bf16, (0, 1, 3, 2))) * (1.0 / jnp.sqrt(head_dim))
            att = att.astype(jnp.float32)  # Cast back to float32 for numerical ops
            
            # Create causal mask in float32
            mask = jnp.tril(jnp.ones((T, T), dtype=jnp.float32))[None, None, :, :]
            att = jnp.where(mask, att, float('-inf'))
            att = jax.nn.softmax(att, axis=-1)
            
            # Second matmul: cast to bfloat16 for matmul, then back to float32
            att_bf16 = att.astype(jnp.bfloat16)
            y = jnp.matmul(att_bf16, v_bf16).astype(jnp.float32)
        else:
            # Pure precision mode
            att = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * (jnp.bfloat16(1.0 / jnp.sqrt(head_dim)))
            
            # Create causal mask
            mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bfloat16))[None, None, :, :]
            att = jnp.where(mask, att, jnp.bfloat16(-1e4))  # Use -1e4 instead of -inf for bfloat16
            att = jax.nn.softmax(att, axis=-1)
            
            y = jnp.matmul(att, v)

        # Re-assemble all head outputs side by side
        y = jnp.transpose(y, (0, 2, 1, 3))  # (B, T, nh, hs)
        y = jnp.reshape(y, (B, T, C))  # (B, T, C)

        # Output projection
        if self.mixed_precision:
            y_bf16 = y.astype(jnp.bfloat16)
            y = self.out_proj(y_bf16).astype(jnp.float32)
        else:
            y = self.out_proj(y)
            
        return y


class MLP(nn.Module):
    """Multi-layer perceptron with GELU activation and dropout.
    
    Supports both pure precision and mixed precision modes.
    """
    config: ModelConfig
    mixed_precision: bool = True
    init_std: float = 0.02

    def setup(self):
        # Determine parameter dtype based on precision mode
        param_dtype = jnp.float32 if self.mixed_precision else jnp.bfloat16
        
        self.fc1 = nn.Dense(
            self.config.n_embd * 4,
            kernel_init=nn.initializers.normal(stddev=self.init_std),
            dtype=param_dtype
        )
        self.fc2 = nn.Dense(
            self.config.n_embd,
            kernel_init=nn.initializers.normal(stddev=self.init_std),
            dtype=param_dtype
        )

    @nn.compact
    def __call__(self, x, deterministic=True):
        if self.mixed_precision:
            # Mixed precision: matmul in bfloat16, activations in float32
            x = x.astype(jnp.float32)
            
            x_bf16 = x.astype(jnp.bfloat16)
            x = self.fc1(x_bf16).astype(jnp.float32)
            x = nn.gelu(x, approximate=True)  # GELU in float32
            x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=deterministic)
            
            x_bf16 = x.astype(jnp.bfloat16)
            x = self.fc2(x_bf16).astype(jnp.float32)
            x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=deterministic)
        else:
            # Pure precision mode
            x = x.astype(jnp.bfloat16)
            x = self.fc1(x)
            x = nn.gelu(x, approximate=True)
            x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=deterministic)
            x = self.fc2(x)
            x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=deterministic)
            
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm and residual connections.
    
    Supports both pure precision and mixed precision modes.
    """
    config: ModelConfig
    mixed_precision: bool = True
    init_std: float = 0.02

    @nn.checkpoint  # Add gradient checkpointing to save memory
    @nn.compact
    def __call__(self, x):
        # Determine dtype based on precision mode
        norm_dtype = jnp.float32 if self.mixed_precision else jnp.bfloat16
        
        if self.mixed_precision:
            x = x.astype(jnp.float32)
        else:
            x = x.astype(jnp.bfloat16)
        
        # Pre-norm architecture
        x = nn.LayerNorm(dtype=norm_dtype)(x)
        x = x + CausalSelfAttention(
            self.config, 
            mixed_precision=self.mixed_precision, 
            init_std=self.init_std
        )(x)
        
        x = nn.LayerNorm(dtype=norm_dtype)(x)
        x = x + MLP(
            self.config, 
            mixed_precision=self.mixed_precision, 
            init_std=self.init_std
        )(x)
        
        return x


class GPTWithRoPE(nn.Module):
    """GPT model with RoPE position embeddings.
    
    This implementation replaces learned positional embeddings with 
    Rotary Position Embeddings (RoPE) as described in the RoFormer paper.
    
    Supports both pure precision and mixed precision training modes.
    """
    config: ModelConfig
    mixed_precision: bool = True
    init_std: float = 0.02

    def setup(self):
        # Determine parameter dtype based on precision mode
        param_dtype = jnp.float32 if self.mixed_precision else jnp.bfloat16
        
        # Token embeddings (no positional embeddings - RoPE handles positions)
        self.wte = nn.Embed(
            self.config.vocab_size, 
            self.config.n_embd, 
            dtype=param_dtype
        )
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(dtype=param_dtype)
        self.head = nn.Dense(
            self.config.vocab_size,
            kernel_init=nn.initializers.normal(stddev=self.init_std * 0.5),
            dtype=param_dtype
        )

    @nn.compact
    def __call__(self, x, deterministic=False):
        """Forward pass through the GPT model.
        
        Args:
            x: Input token indices of shape (batch_size, seq_len)
            deterministic: Whether to use deterministic mode (no dropout)
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        B, T = x.shape
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"

        # Token embedding (no positional embedding - RoPE handles positions)
        tok_emb = self.wte(x)
        
        if self.mixed_precision:
            x = tok_emb.astype(jnp.float32)
        else:
            x = tok_emb.astype(jnp.bfloat16)

        # Apply transformer blocks
        for _ in range(self.config.n_layer):
            x = TransformerBlock(
                self.config, 
                mixed_precision=self.mixed_precision, 
                init_std=self.init_std
            )(x)
            
        # Final layer norm
        x = self.ln_f(x)
        
        # Output projection
        if self.mixed_precision:
            # Final projection: cast to bfloat16 for matmul, keep result as float32
            x_bf16 = x.astype(jnp.bfloat16)
            logits = self.head(x_bf16).astype(jnp.float32)
        else:
            # Convert to float32 only for loss computation
            logits = self.head(x).astype(jnp.float32)
            
        return logits

    def init(self, rng):
        """Initialize model parameters."""
        tokens = jnp.zeros((1, self.config.block_size), dtype=jnp.uint16)
        params = super().init(rng, tokens, True)
        return params


def count_params(params):
    """Count the total number of parameters in the model."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))