# Copyright 2025 The Protoscribe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Miscellaneous utilities."""

import enum

import jax.numpy as jnp


@enum.unique
class RunType(enum.Enum):
  """Type of the run."""

  TRAIN = "train"
  EVAL = "eval"
  PREDICT = "predict"


def shift_right(x: jnp.ndarray, axis: int = 1):
  """Shifts the input to the right by padding on axis.

  Args:
    x: Input array.
    axis: Axis along which to pad.

  Returns:
    Array shifted to the right.
  """
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  padded = jnp.pad(
      x, pad_widths, mode="constant", constant_values=x.dtype.type(0)
  )
  return padded[:, :-1]


def nonzero_sequence_mask(x: jnp.ndarray) -> jnp.ndarray:
  """Given a three dimensional sequence returns the corresponding mask.

  Checks that the last dimension is not all zeros.

  Args:
    x: An array (B, L, D), with the first batch and the second length dimension.

  Returns:
    An integer array (B, L) where the non-zero elements are marked as 1.
  """
  mask = 1 - jnp.all(x == jnp.zeros_like(x[0, 0, :], dtype=x.dtype), axis=-1)
  return mask.astype(jnp.int32)
