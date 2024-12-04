# Copyright 2024 The Protoscribe Authors.
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

"""Class for relative position biases supporting higher rank.

T5 uses a form of relative attention which biases the attention matrix, so each
head effectively attends to things at different scales, irrespective of the
contents of keys and queries.

This is similar to the T5 version but allows for higher rank features with
orthogonal relative positions (such as separate relative position embeddings
for spatial and temporal distance).
"""
from typing import Any, Callable, Mapping, Tuple, Union

from absl import logging  # pylint: disable=unused-import
from flax import linen as nn
from flax.linen import partitioning
from jax import lax
import jax.numpy as jnp
import numpy as np

from flaxformer import activation_partitioning

Array = Any


def infer_shape(size: int, dims: Tuple[int, ...]) -> Tuple[int, ...]:
  """Infers the shape of a feature in the sequence.

  The `size` argument is the total number of sequence positions that correspond
  to a feature. For example, a feature that has 2 images encoded as
  14x14 image patches might have `size=2x14x14=392`. Size is usually derived
  from feature `bounds=[start, end)`, as populated by the
  `multimodal_feature.py` library.

  The `dims` argument is manually set by the user and represents the internal
  shape of one instance of a feature, such as `(0, 14, 14, 0)` for a 14x14
  spatial grid with 0-width on the text axis and 0-width on the temporal axis
  (assuming the axes are `(text, y, x, time)`).

  It's possible to set one of the dims equal to -1, in which case that dim
  is inferred as `size/prod(remaining_dims)`. This is a convenient way to
  handle features that have variable length such as text, e.g.
  `dims=(-1, 0, 0, 0)`.

  Note that 0s are allowed in the `dims`, since some features do not have a
  meaningful position along every axis. For example, `text_tokens` do not
  have a spatial position. If 0s are provided to this function, they are treated
  the same as 1s.

  Examples:
    size=12, dims=(3, 4) -> (3, 4)
    size=24, dims=(3, 4) -> (3, 4)
    size=12, dims=(3, 2, -1) -> (3, 2, 2)
    size=5, dims=(-1, 1) -> (5, 1)
    size=5, dims=(-1, 0) -> (5, 0)

  Args:
    size: int, total number of elements
    dims: int, tuple of dims with at most one -1

  Returns:
    dims, with a single -1 replaced by a positive integer, if a -1 was provided
  """
  positive_dims = [d for d in dims if d > 0]
  if len(positive_dims) == len(dims):
    if np.prod(dims) != size:
      raise ValueError(f'size={size} does not match shape={dims}')
    return dims
  negative_dims = [d for d in dims if d < 0]
  if len(negative_dims) > 1:
    raise ValueError(f'at most one dim in {dims} may be negative')
  if size < 1:
    raise ValueError(f'size={size} must be positive for dim inference')
  factor = np.prod(positive_dims, dtype=np.int32)
  if size % factor != 0:
    raise ValueError(f'{size} is not a multiple of {factor}')
  inferred_d = size // factor
  dims = tuple(d if d >= 0 else inferred_d for d in dims)
  return dims


def relpos_for_shape(
    shape: Tuple[int, ...], dtype: Any, computation_module: Any
) -> Tuple[Array]:
  """Computes the relative positions for each axis in `shape`.

  For shapes with rank-1, this returns a single Array of shape
  [seqlen, seqlen]. The relative positions are similar to the ones T5 uses for
  text tokens:

    [[ 0,  1,  2,  3, ...],
     [-1,  0,  1,  2, ...],
     [-2, -1,  0,  1, ...],
     ...]

  For shapes with rank-2, this returns *two* Arrays of shape [seqlen, seqlen],
  where `seqlen = np.prod(shape)` is the flattened sequence length. The
  flattened sequence is assumed to be row-major order, which is the same used
  by numpy and TF when flattening an Array

    tuple(
      [[0, 1, 0, 1],
       [-1, 0, -1, 0]],  # "outer" axis
      [[0, 0, 1, 1],
       [-1, -1, 0, 0]],  # "inner" axis
    )

  Args:
    shape: tuple of ints, internal shape of the feature
    dtype: dtype of return value
    computation_module: jnp or np

  Returns:
    a relpos array for each axis of `shape`
  """
  shape = tuple((d or 1) for d in shape)  # 0 is treated as 1 here
  feature_len = np.prod(shape, dtype=computation_module.int32)
  relpos_arrays = []
  for i, d in enumerate(shape):
    x = computation_module.arange(d, dtype=dtype)
    tiling = []
    for j, d2 in enumerate(shape):
      if j < i:
        x = x[..., None]
        tiling.append(d2)
      elif j == i:
        tiling.append(1)
      else:
        x = x[None, ...]
        tiling.append(d2)
    tiling = reversed(tiling)
    x = computation_module.tile(x, tiling)
    x = computation_module.reshape(x, [feature_len])
    x = x[None, :] - x[:, None]
    relpos_arrays.append(x)
  return tuple(relpos_arrays)


def relpos_nd(
    feature_bounds: Mapping[str, Tuple[int, int]],
    feature_shapes: Mapping[str, Tuple[int, ...]],
    computation_module: Any,
) -> Tuple[Array, ...]:
  """Returns relative positions for the provided sequence and shapes.

  This function combines the relative positions for multiple features into
  global relative position Arrays per axis, to be used for self-attention bias.

  Example:
    `text_tokens` comes first in the sequence (bounds=[0, 3], shape=[3]) and has
    the following relpos:
      [[ 0,  1,  2],
       [-1,  0,  1],
       [-2, -1,  0]]  # temporal

      [[ 0,  0,  0],
       [ 0,  0,  0],
       [ 0,  0,  0]]  # x-axis

      [[ 0,  0,  0],
       [ 0,  0,  0],
       [ 0,  0,  0]]  # y-axis

    And `image_dense` comes next and has 2x2 elements
    (bounds=[3, 7], shape=[2x2]) with the following relpos:
      [[ 0,  0,  0,  0],
       [ 0,  0,  0,  0],
       [ 0,  0,  0,  0],
       [ 0,  0,  0,  0]]  # temporal

      [[ 0,  1,  0,  1],
       [-1,  0, -1,  0],
       [ 0,  1,  0,  1],
       [-1,  0, -1,  0]]  # x-axis

      [[ 0,  0,  1,  1],
       [ 0,  0,  1,  1],
       [-1, -1,  0,  0],
       [-1, -1,  0,  0]]  # x-axis

    The global relpos arrays returned by self-attention will be
      [[  0,  1,  2,  0,  0,  0,  0],
       [ -1,  0,  1,  0,  0,  0,  0],
       [ -2, -1,  0,  0,  0,  0,  0],
       [  0,  0,  0,  0,  0,  0,  0],
       [  0,  0,  0,  0,  0,  0,  0],
       [  0,  0,  0,  0,  0,  0,  0],
       [  0,  0,  0,  0,  0,  0,  0]]  # temporal

      [[  0,  0,  0,  0,  0,  0,  0],
       [  0,  0,  0,  0,  0,  0,  0],
       [  0,  0,  0,  0,  0,  0,  0],
       [  0,  0,  0,  0,  1,  0,  1],
       [  0,  0,  0, -1,  0, -1,  0],
       [  0,  0,  0,  0,  1,  0,  1],
       [  0,  0,  0, -1,  0, -1,  0]]  # x-axis

      [[  0,  0,  0,  0,  0,  0,  0],
       [  0,  0,  0,  0,  0,  0,  0],
       [  0,  0,  0,  0,  0,  0,  0],
       [  0,  0,  0,  0,  0,  1,  1],
       [  0,  0,  0,  0,  0,  1,  1],
       [  0,  0,  0, -1, -1,  0,  0],
       [  0,  0,  0, -1, -1,  0,  0]]  # y-axis

  Args:
    feature_bounds: map of feature name to [start, end) offsets in `relpos`
    feature_shapes: map of feature name to shape
    computation_module: jnp or np

  Returns:
    a sequence of relpos arrays, one per axis
  """
  seqlen = max(end for (_, end) in feature_bounds.values())
  rank = len(next(iter(feature_shapes.values())))
  for shape in feature_shapes.values():
    if rank != len(shape):
      raise ValueError('shape mismatch for feature shapes=%s' % feature_shapes)
  relpos = [
      computation_module.full(
          [seqlen, seqlen], 0, dtype=computation_module.int32
      )
      for _ in range(rank)
  ]
  for name, bounds in feature_bounds.items():
    if name not in feature_shapes:
      raise ValueError(f'feature_shapes["{name}"] was not set')
    incomplete_shape = feature_shapes[name]
    feature_size = bounds[1] - bounds[0]
    shape = infer_shape(feature_size, incomplete_shape)
    feature_len = np.prod([(d or 1) for d in shape])
    feature_count = feature_size // feature_len
    logging.info('feature_bounds[%s]=%s', name, bounds)
    logging.info('shape[%s]=%s', name, shape)
    local_relpos = list(
        relpos_for_shape(shape, computation_module.int32, computation_module)
    )
    mask = computation_module.full(
        [feature_len, feature_len], 1, dtype=computation_module.int32
    )
    def _pad(v, b=bounds, fc=feature_count):
      v = computation_module.tile(v, (fc, fc))
      pad_amt = (b[0], seqlen - b[1])
      return computation_module.pad(v, (pad_amt, pad_amt))
    mask = _pad(mask)
    for i, local_rp in enumerate(local_relpos):
      local_rp = _pad(local_rp) * mask
      relpos[i] *= (1 - mask)
      relpos[i] += local_rp
  return tuple(relpos)


def relpos_weights(
    feature_bounds: Mapping[str, Tuple[int, int]],
    feature_shapes: Mapping[str, Tuple[int, ...]],
    dtype: Any,
    computation_module: Any,
    feature_weights: Union[None, Mapping[str, Tuple[int, ...]]] = None,
) -> Tuple[Array, ...]:
  """Returns averaging weights to use for relpos biases.

  Relpos weights allows for backwards compatibility with pretrained
  1D-RelposBias models. For text tokens, the weight will be 1.0 on the text axis
  and 0.0 on the two spatial axes, making the biases match what was used for
  text in 1D-RelposBias models. For image tokens, the weight will be 0.5 on the
  spatial axes (since there are two of them) and 0.0 on the text axis,
  which means that statistics the same as 1D-RelposBias, even when upgrading to
  2D+ RelposBias (basically to avoid shocking the model).

  Example:
    `text_tokens` comes first in the sequence and has len=3. `image_dense` comes
    next and has 2x2 elements with the following relpos.

    The returned weights will be:
      [[  1,  1,  1,  0,  0,  0,  0],
       [  1,  1,  1,  0,  0,  0,  0],
       [  1,  1,  1,  0,  0,  0,  0],
       [  0,  0,  0,  0,  0,  0,  0],
       [  0,  0,  0,  0,  0,  0,  0],
       [  0,  0,  0,  0,  0,  0,  0],
       [  0,  0,  0,  0,  0,  0,  0]]  # text

      [[  0,  0,  0,  0,  0,  0,  0],
       [  0,  0,  0,  0,  0,  0,  0],
       [  0,  0,  0,  0,  0,  0,  0],
       [  0,  0,  0, .5, .5, .5, .5],
       [  0,  0,  0, .5, .5, .5, .5],
       [  0,  0,  0, .5, .5, .5, .5],
       [  0,  0,  0, .5, .5, .5, .5]]  # x-axis

      [[  0,  0,  0,  0,  0,  0,  0],
       [  0,  0,  0,  0,  0,  0,  0],
       [  0,  0,  0,  0,  0,  0,  0],
       [  0,  0,  0, .5, .5, .5, .5],
       [  0,  0,  0, .5, .5, .5, .5],
       [  0,  0,  0, .5, .5, .5, .5],
       [  0,  0,  0, .5, .5, .5, .5]]  # y-axis

  Args:
    feature_bounds: map of feature name to [start, end) offsets in `relpos`
    feature_shapes: map of feature name to shape
    dtype: data type of the bias values (e.g. np.float32)
    computation_module: jnp or np
    feature_weights: map of feature name to rel pos weights to assign feature
      weights when needed. If not set, the weights will be uniformly normalized
      to all axis the feature involves.

  Returns:
    weights to use for multiplying by relpos embeddings, one per axis
  """
  seqlen = max(end for (_, end) in feature_bounds.values())
  rank = len(next(iter(feature_shapes.values())))
  for shape in feature_shapes.values():
    if rank != len(shape):
      raise ValueError('shape mismatch for feature shapes=%s' % feature_shapes)
  weights = [
      computation_module.full([seqlen, seqlen], 0, dtype=dtype)
      for _ in range(rank)
  ]
  feature_weights = feature_weights or {}
  for name, bounds in feature_bounds.items():
    if name not in feature_shapes:
      raise ValueError(f'feature_shapes["{name}"] was not set')
    incomplete_shape = feature_shapes[name]
    feature_size = bounds[1] - bounds[0]
    shape = infer_shape(feature_size, incomplete_shape)
    feature_len = np.prod([(d or 1) for d in shape])
    feature_count = feature_size // feature_len
    num_axes = len([d for d in shape if d > 0])
    if num_axes == 0:
      raise ValueError(f'feature={name} did not use any axes shape={shape}')
    for axis in range(rank):
      if name not in feature_weights:
        weight = 1.0 / num_axes
      else:
        weight = feature_weights[name][axis]

      def _pad(v, b=bounds, fc=feature_count):
        v = computation_module.tile(v, (fc, fc))
        pad_amt = (b[0], seqlen - b[1])
        return computation_module.pad(v, (pad_amt, pad_amt))

      mask = computation_module.full(
          [feature_len, feature_len], weight, dtype=dtype
      )
      mask = _pad(mask)
      if shape[axis] > 0:
        weights[axis] += mask
  return tuple(weights)


class RelativePositionBiasesND(nn.Module):
  """Adds T5-style relative positional embeddings to the attention logits.

  This module provides a way for users to inject shape information about
  higher rank features into the Relpos biases in a way that allows the
  relative positions to be modeled separately per axis. See
  http://docs/document/d/1vZd_dJSaGHCW_x7Bg7rZXfaaBEaINKqZC6C5VurgseY?resourcekey=0-qBTRE5jIa8IlOvh1Psshaw#heading=h.9pnl42entj4j

  Examples w/ 4 axes:
    1D feature with "text" dimension only:
      feature_shapes['text_tokens'] = (256, 0, 0, 0)

    The same feature but the text feature length is inferred:
       feature_shapes['text_tokens'] = (-1, 0, 0, 0)

    2D spatial feature such as image patches:
      feature_shapes['image_dense'] = (0, 16, 16, 1)

    3D temporospatial feature such as video with inferred frame count:
      feature_shapes['video_dense'] = (0, 16, 16, -1)

  Shape dims can be positive, negative, or 0:
    `dim > 0` means that the feature has relative positions on the
      corresponding axis, e.g. `feature_shapes['image_dense'][1]=16` means that
      the `image_dense` feature has 16 positions on axis 1

    `dim < 0` also means that the feature has relative positions, but that the
      exact length should be inferred. This is provided for convenience, since
      one dim can typically be inferred from the TASK_FEATURE_LENGTHS.

    `dim = 0` means that the feature does not have relative positions on
      the corresponding axis, such as text tokens on a spatial axis

  Using a convention of (text, x, y, time) allows the relpos
  biases to be portable across tasks and modalities, so it's recommended to
  use this convention.

  Attributes:
    num_buckets: Number of buckets to bucket distances between key and query
      positions into.
    max_distance: Maximum distance before everything is lumped into the last
      distance bucket.
    num_heads: Number of heads in the attention layer. Each head will get a
      different relative position weighting.
    shape_dim_names: N names, one for each axis in this Relpos scheme. Must be
      unique. All shapes in `feature_shapes` must have N elements.
    dtype: Type of arrays through this module.
    embedding_init: initializer for relative embedding table.
    head_axis_name: Axis to partition the relpos bias heads on. Setting this
      field trades training performance for unbounded parallelism in mixed
      models.
    feature_shapes: map of feature name to shape, each shape having N dims
  """
  num_buckets: int
  max_distance: int
  num_heads: int
  dtype: Any
  feature_shapes: Mapping[str, Tuple[int, ...]]
  shape_dim_names: Tuple[str, ...]
  embedding_init: Callable[..., Array] = nn.linear.default_embed_init
  head_axis_name: str = 'heads'
  on_device_computation: bool = False
  feature_weights: Union[None, Mapping[str, Tuple[int, ...]]] = None

  @staticmethod
  def _relative_position_bucket(
      relative_position,
      computation_module,
      bidirectional=True,
      num_buckets=32,
      max_distance=128,
  ):
    """Translate relative position to a bucket number for relative attention.

    The relative position is defined as memory_position - query_position, i.e.
    the distance in tokens from the attending position to the attended-to
    position.  If bidirectional=False, then positive relative positions are
    invalid.
    We use smaller buckets for small absolute relative_position and larger
    buckets for larger absolute relative_positions.  All relative
    positions >=max_distance  map to the same bucket.  All relative
    positions <=-max_distance map to the same bucket.  This should allow for
    more graceful generalization to longer sequences than the model has been
    trained on.

    Args:
      relative_position: an int32 array
      computation_module: jnp or np
      bidirectional: a boolean - whether the attention is bidirectional
      num_buckets: an integer
      max_distance: an integer

    Returns:
      a Tensor with the same shape as relative_position, containing int32
        values in the range [0, num_buckets)
    """
    ret = 0
    n = -relative_position
    if bidirectional:
      num_buckets //= 2
      ret += (n < 0).astype(computation_module.int32) * num_buckets
      n = computation_module.abs(n)
    else:
      n = computation_module.maximum(n, 0)
    # now n is in the range [0, inf)
    max_exact = num_buckets // 2
    is_small = (n < max_exact)
    val_if_large = max_exact + (
        computation_module.log(
            n.astype(computation_module.float32) / max_exact
            + computation_module.finfo(computation_module.float32).eps
        )
        / computation_module.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).astype(computation_module.int32)
    val_if_large = computation_module.minimum(val_if_large, num_buckets - 1)
    ret += computation_module.where(is_small, n, val_if_large)
    return ret

  @nn.compact
  def __call__(self,
               qlen,
               klen,
               sequence_metadata,
               bidirectional=True,
               decode=False):
    """Produce relative position embedding attention biases.

    Args:
      qlen: attention query length.
      klen: attention key length.
      sequence_metadata: a multimodal_feature.SequenceMetadata object with
        feature bounds
      bidirectional: whether to allow positive memory-query relative position
        embeddings.
      decode: whether to cache relative position bias during autoregressive
        decoding.

    Returns:
      output: `(1, num_heads, q_len, k_len)` attention bias
    """
    # bidirectional embeddings don't make sense when decoding (and break cache).
    if decode and bidirectional:
      raise ValueError(
          'bidirectional RelativePositionBiases are not supported when '
          'decode=True.'
      )

    # We only cache the bias if the model was already initialized, i.e. if this
    # module is called with model.apply and decode = True. We raise an error if
    # called with model.init and decode = True, since this can cache incorrect
    # positional embeddings produced by random parameters.
    is_initialized = self.has_variable('params', 'rel_embedding')
    if decode and not is_initialized:
      raise ValueError(
          'decode-mode cannot be enabled during init. use model.apply to '
          'initialize the decoding cache.')

    # Return pre-computed relative position bias in cache during decode steps.
    if decode and self.has_variable('cache', 'cached_bias'):
      cached_bias = self.get_variable('cache', 'cached_bias')
      expected_bias_shape = (1, self.num_heads, qlen, klen)
      if cached_bias.shape != expected_bias_shape:
        raise ValueError(f'The cached relative position attention bias was '
                         f'expected to have shape {expected_bias_shape} but '
                         f'instead has the shape {cached_bias.shape}.')
      return cached_bias

    computation_module = jnp if self.on_device_computation else np

    # rp_x, rp_y, and rp_t all have the same shape (qlen, klen), but they
    # respect the unflattened shape of higher rank features along each
    # respective axis. See `relpos_nd` for more details.
    relpos_per_axis = relpos_nd(
        feature_bounds=sequence_metadata.feature_name_to_bounds_map,
        feature_shapes=self.feature_shapes,
        computation_module=computation_module)
    relpos_weights_per_axis = relpos_weights(
        feature_bounds=sequence_metadata.feature_name_to_bounds_map,
        feature_shapes=self.feature_shapes,
        dtype=self.dtype,
        computation_module=computation_module,
        feature_weights=self.feature_weights,
    )
    values = 0.
    for name, rp, rp_weight in zip(
        self.shape_dim_names, relpos_per_axis, relpos_weights_per_axis):
      rp_bucket = self._relative_position_bucket(
          rp,
          computation_module,
          bidirectional=bidirectional,
          num_buckets=self.num_buckets,
          max_distance=self.max_distance)
      rp_bucket = activation_partitioning.with_sharding_migration(
          rp_bucket, None, logical_axis_names=('length_sharded', 'length')
      )
      relative_attention_bias = partitioning.param_with_axes(
          f'rel_embedding{name}',
          self.embedding_init, (self.num_heads, self.num_buckets),
          jnp.float32,
          axes=(self.head_axis_name, 'relpos_buckets'))
      relative_attention_bias = jnp.asarray(relative_attention_bias, self.dtype)
      # Instead of using a slow gather, we create a leading-dimension one-hot
      # array from rp_bucket and use it to perform the gather-equivalent via a
      # contraction, i.e.:
      # (num_head, num_buckets) x (num_buckets one-hot, qlen, klen).
      # This is equivalent to relative_attention_bias[:, rp_bucket]
      bcast_iota = lax.broadcasted_iota(jnp.int32, (self.num_buckets, 1, 1), 0)
      rp_bucket_one_hot = jnp.array(
          rp_bucket[jnp.newaxis, ...] == bcast_iota, dtype=self.dtype)
      rp_bucket_one_hot = activation_partitioning.with_sharding_migration(
          rp_bucket_one_hot,
          None,
          logical_axis_names=('relpos_buckets', 'length_sharded', 'length'),
      )
      # --> shape (qlen, klen, num_heads)
      rp_bias = lax.dot_general(
          relative_attention_bias,
          rp_bucket_one_hot,
          (((1,), (0,)), ((), ())),  # rhs, lhs contracting dims
      )  # no batched dims
      rp_bias = activation_partitioning.with_sharding_migration(
          rp_bias,
          None,
          logical_axis_names=(self.head_axis_name, 'length_sharded', 'length'),
      )
      values += rp_weight[jnp.newaxis, ...] * rp_bias
    # Add a singleton batch dimension.
    # --> shape (1, num_heads, qlen, klen)
    out = values[jnp.newaxis, ...]

    # Store computed relative position bias in cache after first calculation.
    if decode:
      _ = self.variable('cache', 'cached_bias', lambda: out)

    return out
