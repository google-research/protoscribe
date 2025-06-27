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

"""Basic Tensorflow utilities.

This is a very partial clone of `lingvo.core.py_utils` with modifications.
"""

import threading
from typing import Any

import tensorflow as tf

_global_variable_scope = None
_GLOBAL_STEP_TF2 = None


def _get_global_variable_scope():
  """Gets the global variable scope (as if no variable_scope has been set).

  Returns:
    The VariableScope corresponding to as if no tf.variable_scope is in effect.
  """
  if not _global_variable_scope:
    # Each thread gets its own default global variable scope, and we take
    # advantage of that in order to get a top-level scope. This avoids the
    # need to call tf.get_variable_scope() at the module level, which allows
    # this module to be imported without modifying global state (i.e. creating
    # the default graph). It is important to not mutate the global state at
    # module load time, because it let's us flip flags after import that affect
    # core TensorFlow behavior.
    def initialize():
      global _global_variable_scope
      _global_variable_scope = tf.compat.v1.get_variable_scope()

    t = threading.Thread(target=initialize)
    t.start()
    t.join()
  return _global_variable_scope


def get_or_create_global_step_var() -> tf.Variable:
  """Create or reuse the global step variable explicitly.

  This avoids using `get_or_create_global_step` because it pins the global step
  to cpu directly.

  Returns:
    The global step tf.Variable.
  """
  with tf.compat.v1.variable_scope(
      _get_global_variable_scope(),
      use_resource=True
  ):
    global _GLOBAL_STEP_TF2
    if _GLOBAL_STEP_TF2 is None:
      _GLOBAL_STEP_TF2 = tf.Variable(0, name="global_step", dtype=tf.int64)
    return _GLOBAL_STEP_TF2


def get_global_step() -> tf.Variable:
  """Return the global_step in the current graph."""
  assert _GLOBAL_STEP_TF2 is not None
  return _GLOBAL_STEP_TF2


def get_shape(
    tensor: Any,  # anything that can be converted to a tf.Tensor
    ndims: int | None = None,
    optimize_for_reshape: bool = False,
) -> list[int | tf.Tensor] | tf.Tensor:
  """Returns tensor's shape as a list which can be unpacked, unlike tf.shape.

  If the tensor is unranked, and ndims is None, returns the shape as a Tensor.
  Otherwise, returns a list of values. Each element in the list is an int (when
  the corresponding dimension is static), or a scalar tf.Tensor (when the
  corresponding dimension is dynamic).

  Args:
    tensor: The input tensor.
    ndims: If not None, returns the shapes for the first `ndims` dimensions.
    optimize_for_reshape: If true, the output for the first dynamic dimension
      will be set to -1 instead of a tf.Tensor with the dynamic value. This way
      if all other dimensions are static, the result can be used in tf.reshape
      without tf.shape + tf.strided_slice + tf.pack.
  """
  tensor = tf.convert_to_tensor(tensor)
  dynamic_shape = tf.shape(tensor)

  # Early exit for unranked tensor.
  if tensor.shape.ndims is None:
    if ndims is None:
      return dynamic_shape
    else:
      return [dynamic_shape[x] for x in range(ndims)]

  # Ranked tensor.
  if ndims is None:
    ndims = tensor.shape.ndims
  else:
    ndims = min(ndims, tensor.shape.ndims)

  # Return mixture of static and dynamic dims.
  static_shape = tensor.shape.as_list()
  shapes = []
  for x in range(ndims):
    if static_shape[x] is not None:
      shapes.append(static_shape[x])
    elif optimize_for_reshape:
      optimize_for_reshape = False  # only replace the first occurrence
      shapes.append(-1)
    else:
      shapes.append(dynamic_shape[x])
  return shapes
