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

"""Linear Layers - Simplified version for MaxText Lite."""

import functools
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax
from flax import linen as nn

from .common_types import DType, Array
from .initializers import NdInitializer, nd_dense_init, default_bias_init


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int, ...]:
  return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def _canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)


def _compute_dot_general_simple(inputs, kernel, axis, contract_ind, matmul_precision):
  """Simplified dot_general without quantization."""
  dot_general = lax.dot_general
  matmul_precision = lax.Precision(matmul_precision)
  return dot_general(inputs, kernel, ((axis, contract_ind), ((), ())), precision=matmul_precision)


class SimpleDenseGeneral(nn.Module):
  """Simplified dense general layer without advanced features."""
  
  features: Union[Iterable[int], int]
  axis: Union[Iterable[int], int] = -1
  dtype: DType = jnp.float32
  weight_dtype: DType = jnp.float32
  kernel_init: Callable = nn.initializers.lecun_normal()
  bias_init: Callable = nn.initializers.zeros
  use_bias: bool = True
  matmul_precision: str = 'default'
  
  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Apply dense transformation."""
    # Normalize inputs
    inputs = jnp.asarray(inputs, self.dtype)
    axis = self.axis
    
    if isinstance(axis, int):
      axis = (axis,)
    axis = _normalize_axes(axis, inputs.ndim)
    
    if isinstance(self.features, int):
      features = (self.features,)
    else:
      features = tuple(self.features)
    
    # Compute kernel shape
    kernel_shape = tuple(inputs.shape[ax] for ax in axis) + features
    kernel = self.param('kernel', self.kernel_init, kernel_shape, self.weight_dtype)
    kernel = jnp.asarray(kernel, self.dtype)
    
    # Contract over axis dimensions
    contract_ind = tuple(range(len(axis)))
    
    # Compute dot product
    y = _compute_dot_general_simple(
        inputs, kernel, axis, contract_ind, self.matmul_precision
    )
    
    # Add bias if enabled
    if self.use_bias:
      bias = self.param('bias', self.bias_init, features, self.weight_dtype)
      bias = jnp.asarray(bias, self.dtype)
      y = y + bias
    
    return y


def dense_general(
    inputs_shape,
    out_features_shape,
    axis=-1,
    kernel_init=None,
    kernel_axes=None,
    dtype=jnp.float32,
    weight_dtype=jnp.float32,
    name=None,
    quant=None,  # Ignored in simplified version
    use_bias=True,
    matmul_precision='default',
):
  """Create a dense general layer - simplified version."""
  
  if kernel_init is None:
    kernel_init = nn.initializers.lecun_normal()
  
  return SimpleDenseGeneral(
      features=out_features_shape,
      axis=axis,
      dtype=dtype,
      weight_dtype=weight_dtype,
      kernel_init=kernel_init,
      use_bias=use_bias,
      matmul_precision=matmul_precision,
      name=name,
  )