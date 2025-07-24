#!/usr/bin/env python
"""
NanoGPT model implementations with FSDP (Fully Sharded Data Parallel) support.

This module provides FSDP-aware versions of the GPT model components, building on
the base implementations in nanogpt_rope_mixed_precision_v3.py while adding
sharding capabilities across the feature dimension.

The FSDP sharding strategy follows the picodo pattern:
- Embeddings: Shard along feature dimension [V, D] -> [V, D/N]  
- Dense layers: Shard along input or output features based on layer type
- Parameters and optimizer state are distributed across devices
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import PartitionSpec as P
from typing import Optional

from nanogpt_rope_mixed_precision_v3 import (
    ModelConfig, apply_rope, create_rope_cache
)


def create_fsdp_init_fn(layer_type: str, fsdp_enabled: bool, init_std: float = 0.02):
    """Create FSDP-aware initialization function for Flax Linen layers.
    
    Args:
        layer_type: Type of layer ('embedding', 'dense_in', 'dense_out')
        fsdp_enabled: Whether FSDP is enabled
        init_std: Standard deviation for initialization
        
    Returns:
        Initialization function that applies appropriate sharding constraints
    """
    if fsdp_enabled:
        def init_fn(key, shape, dtype=jnp.float32):
            # Create base initialization
            if layer_type == "embedding":
                # Variance scaling for embeddings
                init_val = jax.nn.initializers.variance_scaling(
                    1.0, 'fan_in', 'normal', out_axis=0
                )(key, shape, dtype)
            else:
                # Xavier uniform for dense layers
                init_val = jax.nn.initializers.xavier_uniform()(key, shape, dtype)
            
            # Apply FSDP partitioning based on layer type
            if layer_type == "embedding":  # [V, D] -> shard along D
                pspec = P(None, "data")
            elif layer_type == "dense_in":  # [D, ...] -> shard along input D
                pspec = P("data", *[None] * (len(shape) - 1))
            elif layer_type == "dense_out":  # [..., D] -> shard along output D
                pspec = P(*[None] * (len(shape) - 1), "data")
            else:
                pspec = P(*[None] * len(shape))
            
            # Apply sharding constraint
            return jax.lax.with_sharding_constraint(init_val, pspec)
        
        return init_fn
    else:
        # Standard initialization without sharding
        if layer_type == "embedding":
            return jax.nn.initializers.variance_scaling(1.0, 'fan_in', 'normal', out_axis=0)
        else:
            return jax.nn.initializers.xavier_uniform()


class FSDBGPTWithRoPE(nn.Module):
    """GPT model with RoPE and FSDP sharding support.
    
    This is the main model class that implements FSDP sharding across feature dimensions.
    The model architecture follows the same structure as GPTWithRoPE but with parameters
    distributed across devices for memory efficiency.
    
    Args:
        config: Model configuration
        init_std: Standard deviation for weight initialization
        fsdp_enabled: Whether to enable FSDP sharding
    """
    
    config: ModelConfig
    init_std: float = 0.02
    fsdp_enabled: bool = True
    
    def setup(self):
        # Determine dtype from config
        dtype = jnp.bfloat16 if (hasattr(self.config, 'dtype') and 
                                self.config.dtype == 'bfloat16') else jnp.float32
        
        # Token embeddings with FSDP sharding
        self.wte = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.n_embd,
            embedding_init=create_fsdp_init_fn('embedding', self.fsdp_enabled, self.init_std),
            dtype=dtype
        )
        
        # Transformer blocks
        self.h = [FSDBTransformerBlock(self.config, self.init_std, self.fsdp_enabled) 
                  for _ in range(self.config.n_layer)]
        
        # Final layer norm (not sharded - small parameters)
        self.ln_f = nn.LayerNorm(dtype=dtype)
    
    def __call__(self, idx, deterministic=True):
        """Forward pass through the model.
        
        Args:
            idx: Input token indices [B, T]
            deterministic: Whether to use deterministic mode (no dropout)
            
        Returns:
            logits: Output logits [B, T, vocab_size]
        """
        B, T = idx.shape
        
        # Token embeddings
        x = self.wte(idx)  # (B, T, C)
        
        # Apply transformer blocks
        for block in self.h:
            x = block(x, deterministic=deterministic)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language model head (tied weights with embedding)
        # Cast to float32 for numerical stability in final projection
        logits = self.wte.attend(x.astype(jnp.float32))  # (B, T, vocab_size)
        
        return logits


class FSDBTransformerBlock(nn.Module):
    """Transformer block with FSDP sharding.
    
    Implements the standard transformer block with pre-layer normalization
    and FSDP sharding for the attention and MLP components.
    """
    
    config: ModelConfig
    init_std: float = 0.02
    fsdp_enabled: bool = True
    
    def setup(self):
        dtype = jnp.bfloat16 if (hasattr(self.config, 'dtype') and 
                                self.config.dtype == 'bfloat16') else jnp.float32
        
        self.ln_1 = nn.LayerNorm(dtype=dtype)
        self.attn = FSDBCausalSelfAttention(self.config, self.init_std, self.fsdp_enabled)
        self.ln_2 = nn.LayerNorm(dtype=dtype)
        self.mlp = FSDBMLP(self.config, self.init_std, self.fsdp_enabled)
    
    def __call__(self, x, deterministic=True):
        """Forward pass through transformer block.
        
        Args:
            x: Input tensor [B, T, C]
            deterministic: Whether to use deterministic mode
            
        Returns:
            Output tensor [B, T, C]
        """
        # Pre-norm attention
        x = x + self.attn(self.ln_1(x), deterministic=deterministic)
        # Pre-norm MLP
        x = x + self.mlp(self.ln_2(x), deterministic=deterministic)
        return x


class FSDBCausalSelfAttention(nn.Module):
    """Causal self-attention with FSDP sharding and RoPE.
    
    Implements multi-head causal self-attention with:
    - FSDP sharding on input/output projections
    - Rotary Position Embedding (RoPE)
    - Mixed precision support
    - Multiple attention implementations (naive, XLA, cuDNN)
    """
    
    config: ModelConfig
    init_std: float = 0.02
    fsdp_enabled: bool = True
    
    def setup(self):
        param_dtype = jnp.bfloat16 if (hasattr(self.config, 'dtype') and 
                                      self.config.dtype == 'bfloat16') else jnp.float32
        
        # Q, K, V projections with FSDP sharding on input dimension
        self.q_proj = nn.Dense(
            self.config.n_embd,
            kernel_init=create_fsdp_init_fn('dense_in', self.fsdp_enabled, self.init_std),
            dtype=param_dtype,
            use_bias=False
        )
        self.k_proj = nn.Dense(
            self.config.n_embd,
            kernel_init=create_fsdp_init_fn('dense_in', self.fsdp_enabled, self.init_std),
            dtype=param_dtype,
            use_bias=False
        )
        self.v_proj = nn.Dense(
            self.config.n_embd,
            kernel_init=create_fsdp_init_fn('dense_in', self.fsdp_enabled, self.init_std),
            dtype=param_dtype,
            use_bias=False
        )
        
        # Output projection with FSDP sharding on output dimension
        self.out_proj = nn.Dense(
            self.config.n_embd,
            kernel_init=create_fsdp_init_fn('dense_out', self.fsdp_enabled, self.init_std),
            dtype=param_dtype,
            use_bias=False
        )
        
        # Create RoPE cache
        head_dim = self.config.n_embd // self.config.n_head
        self.cos_cache, self.sin_cache = create_rope_cache(
            self.config.block_size, head_dim, self.config.rope_base, dtype=jnp.float32
        )
    
    def __call__(self, x, deterministic=True):
        """Forward pass through attention layer.
        
        Args:
            x: Input tensor [B, T, C]
            deterministic: Whether to use deterministic mode
            
        Returns:
            Output tensor [B, T, C]
        """
        B, T, C = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x) 
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        head_dim = C // self.config.n_head
        q = q.reshape(B, T, self.config.n_head, head_dim)
        k = k.reshape(B, T, self.config.n_head, head_dim)
        v = v.reshape(B, T, self.config.n_head, head_dim)
        
        # Apply RoPE to queries and keys
        q_f32 = q.astype(jnp.float32)
        k_f32 = k.astype(jnp.float32)
        q = apply_rope(q_f32, self.cos_cache, self.sin_cache).astype(q.dtype)
        k = apply_rope(k_f32, self.cos_cache, self.sin_cache).astype(k.dtype)
        
        # Attention computation
        if self.config.attention_implementation in ['cudnn', 'xla']:
            # Use optimized JAX attention
            y = jax.nn.dot_product_attention(
                q, k, v,
                is_causal=True,
                implementation=self.config.attention_implementation
            )
        else:
            # Manual attention computation
            scale = 1.0 / jnp.sqrt(head_dim)
            att = jnp.einsum('btnh,bsnh->bnts', q, k) * scale
            
            # Apply causal mask and softmax in float32 for numerical stability
            att_f32 = att.astype(jnp.float32)
            mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))[None, None, :, :]
            att_f32 = jnp.where(mask, att_f32, float('-inf'))
            att_f32 = jax.nn.softmax(att_f32, axis=-1)
            
            # Apply attention to values
            att = att_f32.astype(v.dtype)
            y = jnp.einsum('bnts,bsnh->btnh', att, v)
        
        # Reshape and project output
        y = y.reshape(B, T, C)
        y = self.out_proj(y)
        
        return y


class FSDBMLP(nn.Module):
    """Multi-layer perceptron with FSDP sharding.
    
    Implements the standard transformer MLP with:
    - FSDP sharding on input and output projections
    - GELU activation
    - Mixed precision support
    """
    
    config: ModelConfig
    init_std: float = 0.02
    fsdp_enabled: bool = True
    
    def setup(self):
        param_dtype = jnp.bfloat16 if (hasattr(self.config, 'dtype') and 
                                      self.config.dtype == 'bfloat16') else jnp.float32
        
        # First linear layer: input dimension sharded
        self.fc1 = nn.Dense(
            4 * self.config.n_embd,
            kernel_init=create_fsdp_init_fn('dense_in', self.fsdp_enabled, self.init_std),
            dtype=param_dtype,
            use_bias=False
        )
        
        # Second linear layer: output dimension sharded
        self.fc2 = nn.Dense(
            self.config.n_embd,
            kernel_init=create_fsdp_init_fn('dense_out', self.fsdp_enabled, self.init_std),
            dtype=param_dtype,
            use_bias=False
        )
    
    def __call__(self, x, deterministic=True):
        """Forward pass through MLP.
        
        Args:
            x: Input tensor [B, T, C]
            deterministic: Whether to use deterministic mode
            
        Returns:
            Output tensor [B, T, C]
        """
        x = self.fc1(x)
        x = jax.nn.gelu(x)
        x = self.fc2(x)
        return x


def create_fsdp_model_config(base_config: ModelConfig, fsdp_enabled: bool = True, 
                            dtype: Optional[str] = None) -> ModelConfig:
    """Create a model config with FSDP settings.
    
    Args:
        base_config: Base model configuration
        fsdp_enabled: Whether to enable FSDP
        dtype: Model dtype ('bfloat16' or None for float32)
        
    Returns:
        Updated model configuration
    """
    # Create a copy of the base config
    config = ModelConfig(
        vocab_size=base_config.vocab_size,
        n_head=base_config.n_head,
        n_embd=base_config.n_embd,
        block_size=base_config.block_size,
        n_layer=base_config.n_layer,
        dropout_rate=base_config.dropout_rate,
        rope_base=base_config.rope_base,
        attention_implementation=base_config.attention_implementation
    )
    
    # Add FSDP-specific attributes
    config.fsdp_enabled = fsdp_enabled
    if dtype is not None:
        config.dtype = dtype
    
    return config