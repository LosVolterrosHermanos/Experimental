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

# Conditional import for kvax
try:
    from kvax.ops import flash_attention, create_attention_mask
    from kvax.utils import PADDING_SEGMENT_ID, attention_specs
    KVAX_AVAILABLE = True
except ImportError:
    KVAX_AVAILABLE = False


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
        attention_implementation: Attention implementation to use ('naive', 'xla', 'cudnn', 'kvax')
    """
    vocab_size: int = 50304
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    n_layer: int = 12
    dropout_rate: float = 0.1
    rope_base: float = 10000.0
    attention_implementation: str = 'naive'


# GPT-2 model size configurations
GPT2_CONFIGS = {
    'GPT2-nano': ModelConfig(
        vocab_size=50304,
        n_head=12,
        n_embd=768,
        block_size=1024,
        n_layer=12,
        dropout_rate=0.1,
        rope_base=10000.0,
        attention_implementation='naive'
    ),
    'GPT2-medium': ModelConfig(
        vocab_size=50304,
        n_head=16,
        n_embd=1024,
        block_size=1024,
        n_layer=24,
        dropout_rate=0.1,
        rope_base=10000.0,
        attention_implementation='naive'
    ),
    'GPT2-large': ModelConfig(
        vocab_size=50304,
        n_head=20,
        n_embd=1280,
        block_size=1024,
        n_layer=36,
        dropout_rate=0.1,
        rope_base=10000.0,
        attention_implementation='naive'
    ),
    'GPT2-jumbo': ModelConfig(
        vocab_size=50304,
        n_head=25,
        n_embd=1600,
        block_size=1024,
        n_layer=48,
        dropout_rate=0.1,
        rope_base=10000.0,
        attention_implementation='naive'
    )
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get model configuration by name.
    
    Args:
        model_name: Name of the model ('GPT2-nano', 'GPT2-medium', 'GPT2-large', 'GPT2-jumbo')
        
    Returns:
        ModelConfig: Configuration for the specified model
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in GPT2_CONFIGS:
        available_models = ', '.join(GPT2_CONFIGS.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available_models}")
    return GPT2_CONFIGS[model_name]


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
        x: Input tensor of shape (batch, seq_len, heads, head_dim)
        cos_cache: Cosine cache of shape (seq_len, head_dim//2)
        sin_cache: Sine cache of shape (seq_len, head_dim//2)
    
    Returns:
        Tensor with RoPE applied, same shape as input
    """
    batch, seq_len, heads, head_dim = x.shape
    
    # Split x into even and odd indices (pairs for rotation)
    x_even = x[..., ::2]   # (batch, seq_len, heads, head_dim//2)
    x_odd = x[..., 1::2]   # (batch, seq_len, heads, head_dim//2)
    
    # Get the appropriate cos and sin values for the sequence length
    cos = cos_cache[:seq_len, :]  # (seq_len, head_dim//2)
    sin = sin_cache[:seq_len, :]  # (seq_len, head_dim//2)
    
    # Reshape cos and sin to broadcast properly
    cos = cos[None, :, None, :]  # (1, seq_len, 1, head_dim//2)
    sin = sin[None, :, None, :]  # (1, seq_len, 1, head_dim//2)
    
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
    
    Uses mixed precision mode: Matrix multiplications in bfloat16, other ops in float32
    """
    config: ModelConfig
    init_std: float = 0.02

    def setup(self):
        # Parameters stored in bfloat16 for efficiency
        param_dtype = jnp.bfloat16
        
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

        # Mixed precision: keep computations in bfloat16
        #x = x.astype(jnp.bfloat16)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to separate heads
        head_dim = C // self.config.n_head
        q = jnp.reshape(q, (B, T, self.config.n_head, head_dim))
        k = jnp.reshape(k, (B, T, self.config.n_head, head_dim))
        v = jnp.reshape(v, (B, T, self.config.n_head, head_dim))

        # Apply RoPE to queries and keys (cast to float32 for RoPE, then back)
        # Cast to float32 for RoPE computation, then back to bfloat16
        q_f32 = q.astype(jnp.float32)
        k_f32 = k.astype(jnp.float32)
        q = apply_rope(q_f32, self.cos_cache, self.sin_cache).astype(jnp.bfloat16)
        k = apply_rope(k_f32, self.cos_cache, self.sin_cache).astype(jnp.bfloat16)

        # Attention computation
        if self.config.attention_implementation == 'kvax':
            if not KVAX_AVAILABLE:
                raise ImportError("kvax is not installed. Install with: pip install kvax")
            
            # Use kvax flash attention with proper attention specs
            # Create segment IDs and positions for kvax
            positions = jnp.arange(T)[None, :].repeat(B, axis=0)  # (B, T)
            segment_ids = jnp.zeros((B, T), dtype=jnp.int32)  # All tokens in same segment
            
            # Reshape for kvax (expects BNTH format)
            q_kvax = jnp.transpose(q, (0, 2, 1, 3))  # (B, N, T, H)
            k_kvax = jnp.transpose(k, (0, 2, 1, 3))  # (B, N, T, H)
            v_kvax = jnp.transpose(v, (0, 2, 1, 3))  # (B, N, T, H)
            
            # Set attention specs and apply kvax flash attention
            with attention_specs(
                query_specs=("data", None, None, None),  # No sharding for single GPU
                kv_specs=("data", None, None, None),
            ):
                # Create attention mask for causal attention
                attention_mask = create_attention_mask(
                    positions, segment_ids, positions, segment_ids
                )
                
                # Apply kvax flash attention
                y_kvax = flash_attention(
                    query=q_kvax,
                    key=k_kvax,
                    value=v_kvax,
                    query_positions=positions,
                    query_segment_ids=segment_ids,
                    kv_positions=positions,
                    kv_segment_ids=segment_ids,
                    mask=attention_mask
                )
            
            # Reshape back to BTNH format
            y = jnp.transpose(y_kvax, (0, 2, 1, 3))  # (B, T, N, H)
            
        elif self.config.attention_implementation in ['cudnn', 'xla']:
            # Use jax.nn.dot_product_attention with specified implementation
            # Keep attention computation in bfloat16 and in BTNH format
            y = jax.nn.dot_product_attention(
                q, k, v,
                is_causal=True,
                implementation=self.config.attention_implementation
            )
        else:
            # Fallback to einsum implementation (no transposes needed)
            # Direct computation in BTNH format using einsum
            scale = jnp.bfloat16(1.0 / jnp.sqrt(head_dim))
            
            # Attention scores: (B,T,N,H) × (B,S,N,H) -> (B,N,T,S)
            att = jnp.einsum('btnh,bsnh->bnts', q, k) * scale
            
            # Cast to float32 only for softmax (numerically sensitive)
            att_f32 = att.astype(jnp.float32)
            mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))[None, None, :, :]
            att_f32 = jnp.where(mask, att_f32, float('-inf'))
            att_f32 = jax.nn.softmax(att_f32, axis=-1)
            
            # Value multiplication: (B,N,T,S) × (B,S,N,H) -> (B,T,N,H)
            att = att_f32.astype(jnp.bfloat16)
            y = jnp.einsum('bnts,bsnh->btnh', att, v)

        # Re-assemble all head outputs side by side
        y = jnp.reshape(y, (B, T, C))  # (B, T, C)

        # Output projection (keep in bfloat16)
        y = self.out_proj(y)
            
        return y


class MLP(nn.Module):
    """Multi-layer perceptron with GELU activation and dropout.
    
    Uses mixed precision mode.
    """
    config: ModelConfig
    init_std: float = 0.02

    def setup(self):
        # Parameters stored in bfloat16 for efficiency
        param_dtype = jnp.bfloat16
        
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
        # Mixed precision: keep computations in bfloat16
        #x = x.astype(jnp.bfloat16)
        x = self.fc1(x)
        x = nn.gelu(x, approximate=True)  # GELU can work in bfloat16
        x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=deterministic)
        x = self.fc2(x)
        x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=deterministic)
            
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm and residual connections.
    
    Uses mixed precision mode.
    """
    config: ModelConfig
    init_std: float = 0.02

    @nn.checkpoint  # Add gradient checkpointing to save memory
    @nn.compact
    def __call__(self, x):
        # LayerNorm needs float32 for numerical stability
        norm_dtype = jnp.float32
        
        # Ensure input is in bfloat16
        #x = x.astype(jnp.bfloat16)
        
        # Pre-norm architecture - cast to float32 for LayerNorm, then back
        x_norm = nn.LayerNorm(dtype=norm_dtype)(x.astype(jnp.float32)).astype(jnp.bfloat16)
            
        x = x + CausalSelfAttention(
            self.config, 
            init_std=self.init_std
        )(x_norm)
        
        # Second LayerNorm
        x_norm = nn.LayerNorm(dtype=norm_dtype)(x.astype(jnp.float32)).astype(jnp.bfloat16)
            
        x = x + MLP(
            self.config, 
            init_std=self.init_std
        )(x_norm)
        
        return x


class GPTWithRoPE(nn.Module):
    """GPT model with RoPE position embeddings.
    
    This implementation replaces learned positional embeddings with 
    Rotary Position Embeddings (RoPE) as described in the RoFormer paper.
    
    Uses mixed precision training mode.
    """
    config: ModelConfig
    init_std: float = 0.02

    def setup(self):
        # Parameters stored in bfloat16 for efficiency
        param_dtype = jnp.bfloat16
        
        # Token embeddings (no positional embeddings - RoPE handles positions)
        self.wte = nn.Embed(
            self.config.vocab_size, 
            self.config.n_embd, 
            dtype=param_dtype
        )
        
        # Final layer norm and output projection
        # LayerNorm needs float32 for numerical stability in mixed precision
        norm_dtype = jnp.float32
        self.ln_f = nn.LayerNorm(dtype=norm_dtype)
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
        x = self.wte(x)
        assert x.dtype == jnp.bfloat16, f"Embedding output should be bfloat16, got {x.dtype}"

        # Apply transformer blocks
        for _ in range(self.config.n_layer):
            x = TransformerBlock(
                self.config, 
                init_std=self.init_std
            )(x)
            
        # Final layer norm
        # Cast to float32 for LayerNorm, then back to bfloat16
        x = self.ln_f(x.astype(jnp.float32)).astype(jnp.bfloat16)
        
        # Output projection - convert to float32 for loss computation
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