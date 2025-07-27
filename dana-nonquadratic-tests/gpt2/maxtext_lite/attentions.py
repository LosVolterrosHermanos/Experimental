#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Attention Layers - Simplified version for MaxText Lite."""

import functools
from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn

from .common_types import Array, DType


def apply_mask_to_logits(logits: Array, mask: Array):
  """Apply a floating-point mask to logits."""
  return jnp.where(mask, logits, -1e30)


class SimplifiedAttentionOp(nn.Module):
  """Simplified attention operation that can fall back to different implementations."""
  
  config: Any
  mesh: Optional[Any] = None
  attention_kernel: str = 'flash'
  max_target_length: int = 1024
  float32_qk_product: bool = False
  float32_logits: bool = True
  quant: Optional[Any] = None
  kv_quant: Optional[Any] = None
  num_query_heads: int = 12
  num_kv_heads: int = 12
  dtype: DType = jnp.float32
  
  @nn.compact
  def __call__(self, query, key, value, decoder_segment_ids=None, model_mode='train'):
    """Apply attention operation."""
    B, T, H, D = query.shape
    
    # Try to use JAX's built-in attention if available
    if self.attention_kernel in ['flash', 'cudnn_flash_te']:
      try:
        # Use JAX's dot_product_attention which can dispatch to Flash Attention
        result = jax.nn.dot_product_attention(
            query=query,
            key=key,
            value=value,
            is_causal=True,
            implementation=self.attention_kernel if self.attention_kernel != 'flash' else None
        )
        return result
      except Exception:
        # Fall back to manual implementation
        pass
    
    # Manual attention implementation
    scale = 1.0 / jnp.sqrt(D)
    
    # Compute attention scores
    if self.float32_qk_product:
      scores = jnp.einsum('btnh,bsnh->bnts', query.astype(jnp.float32), key.astype(jnp.float32))
      scores = scores * scale
    else:
      scores = jnp.einsum('btnh,bsnh->bnts', query, key) * scale
    
    # Apply causal mask
    mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))[None, None, :, :]
    scores = apply_mask_to_logits(scores, mask)
    
    # Apply softmax
    if self.float32_logits:
      attn_weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)
      attn_weights = attn_weights.astype(self.dtype)
    else:
      attn_weights = jax.nn.softmax(scores, axis=-1)
    
    # Apply attention to values
    result = jnp.einsum('bnts,bsnh->btnh', attn_weights, value)
    
    return result


def attention_op_as_linen(
    config,
    mesh=None,
    attention_kernel='flash',
    max_target_length=1024,
    float32_qk_product=False,
    float32_logits=True,
    quant=None,
    kv_quant=None,
    num_query_heads=12,
    num_kv_heads=12,
    dtype=jnp.float32,
):
  """Create attention operation as Linen module."""
  
  return SimplifiedAttentionOp(
      config=config,
      mesh=mesh,
      attention_kernel=attention_kernel,
      max_target_length=max_target_length,
      float32_qk_product=float32_qk_product,
      float32_logits=float32_logits,
      quant=quant,
      kv_quant=kv_quant,
      num_query_heads=num_query_heads,
      num_kv_heads=num_kv_heads,
      dtype=dtype,
  )