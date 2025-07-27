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

"""Embeddings - Simplified version for MaxText Lite."""

import jax.numpy as jnp
from flax import linen as nn

from .common_types import Array


class SimplifiedRotaryEmbedding(nn.Module):
  """Simplified rotary embedding implementation."""
  
  min_timescale: float = 1.0
  max_timescale: float = 10000.0
  embedding_dims: int = 128
  
  def setup(self):
    # Create frequency tensor
    fraction = jnp.arange(0, self.embedding_dims, 2, dtype=jnp.float32) / self.embedding_dims
    timescale = self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction
    self.inv_timescale = 1.0 / timescale
  
  def __call__(self, length: int) -> tuple[Array, Array]:
    """Generate cos and sin for rotary embedding."""
    position = jnp.arange(length, dtype=jnp.float32)
    sinusoid_inp = jnp.outer(position, self.inv_timescale)
    cos = jnp.cos(sinusoid_inp)
    sin = jnp.sin(sinusoid_inp)
    return cos, sin


def llama_rotary_embedding_as_linen(
    min_timescale=1.0,
    max_timescale=10000.0,
    embedding_dims=128,
    name=None,
):
  """Create LLaMA-style rotary embedding."""
  return SimplifiedRotaryEmbedding(
      min_timescale=min_timescale,
      max_timescale=max_timescale,
      embedding_dims=embedding_dims,
      name=name,
  )


def rotary_embedding_as_linen(
    min_timescale=1.0,
    max_timescale=10000.0,
    embedding_dims=128,
    name=None,
):
  """Create rotary embedding."""
  return SimplifiedRotaryEmbedding(
      min_timescale=min_timescale,
      max_timescale=max_timescale,
      embedding_dims=embedding_dims,
      name=name,
  )