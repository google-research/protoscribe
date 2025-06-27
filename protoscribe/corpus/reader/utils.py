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

"""Miscellaneous utils."""

import tensorflow as tf


def pad_or_trim_sequence(
    inputs: tf.Tensor, max_sequence_length: int
) -> tf.Tensor:
  """Pads the 1-D or 2-D inputs as necessary.

  The first dimension is the temporal dimension.

  Args:
    inputs: Input tensor with dimension (L,) or (L, D), where L is the length
      of the sequence.
    max_sequence_length: Maximum sequence to pad to.

  Returns:
    Padded tensor.
  """
  pad_amount = max_sequence_length - tf.shape(inputs)[0]
  if pad_amount == 0:
    return inputs
  elif pad_amount < 0:
    return inputs[:pad_amount, ...]
  else:
    rank = len(tf.shape(inputs))  # For some reason `tf.rank` doesn't work here.
    if rank == 2:
      paddings = [[0, pad_amount], [0, 0]]
    else:  # Assume 1-D.
      paddings = [[0, pad_amount]]

    return tf.pad(inputs, paddings, mode="CONSTANT", constant_values=0)
