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

"""Quantizations - Simplified version for MaxText Lite."""

from typing import Optional, Any


class AqtQuantization:
  """Simplified quantization class that acts as a no-op."""
  
  def __init__(self, quant_dg: int = 8, quant_mode: str = 'int8', **kwargs):
    self.quant_dg = quant_dg
    self.quant_mode = quant_mode
    self.kwargs = kwargs
  
  def dot_general_cls(self, mesh_axes=None):
    """Return a no-op dot_general class."""
    return lambda: lambda *args, **kwargs: None  # Placeholder
  
  def __bool__(self):
    """Make this falsy so quantization is skipped."""
    return False