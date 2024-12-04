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

"""RelativePositionBiases for multimodal encoders.

In a mixed-modality sequence, relative position biases break down. For
example, the distance between image 2 and token 3 is not meaningful,
especially when packing is used.

This module disables sequence-level biases for pairs of sequence positions
that span multiple modalities. It also introduces a set of attention biases
for each pair of input modalities.
"""
from typing import Any, Callable, Optional, Sequence

from flax import linen as nn
from jax import lax
import jax.numpy as jnp

# Type aliases and stubs. Jax doesn't have a unified array type given XLA/etc.
Array = Any


class MultimodalRelativePositionBiases(nn.Module):
  """Relative position biases for a multimodal context.

  This module is meant to be used with
  flaxformer.components.relative_position_biases

  Attributes:
    num_heads: number of attention heads
    dtype: dtype to use
    embedding_init: how to initialize the biases
    max_num_modalities: number of modalities to support
    bias_free_modality_ids: modalities which will have their input relative
      position biases masked, to rely only on alternative position encodings.
  """
  num_heads: int
  dtype: Any
  embedding_init: Callable[..., Array] = nn.linear.default_embed_init
  # WARNING: increasing `max_num_modalities` invalidates previous checkpoints
  max_num_modalities: int = 16
  bias_free_modality_ids: Optional[Sequence[int]] = None

  @nn.compact
  def __call__(self, encoder_bias: Array, modality_segment_ids: Array):
    """Modifies relative position biases for multimodality.

    Args:
      encoder_bias: Array of shape (1, num_heads, q_len, k_len), usually the
        sequence-level biases computed by `relative_position_biases.py`
      modality_segment_ids: Array of shape [q_len] containing the modality
        of each sequence position

    Returns:
      output: `(1, num_heads, q_len, k_len)` attention bias
    """
    # TODO: Assert that `ids < max_num_modalities`
    num_buckets = self.max_num_modalities ** 2
    bias_params = nn.partitioning.param_with_axes(
        'rel_embedding',
        self.embedding_init,
        (self.num_heads, num_buckets),
        jnp.float32,
        axes=('heads', 'relpos_buckets'))
    bias_params = jnp.asarray(bias_params, self.dtype)
    q_subsegments = jnp.expand_dims(modality_segment_ids, 0)
    k_subsegments = jnp.expand_dims(modality_segment_ids, 1)
    # Prevent the sequence-level bias from being used across modalities.
    rp_cross_modal_mask = jnp.equal(q_subsegments, k_subsegments)
    rp_cross_modal_mask = jnp.asarray(rp_cross_modal_mask, encoder_bias.dtype)
    encoder_bias *= rp_cross_modal_mask[jnp.newaxis, jnp.newaxis, ...]
    if self.bias_free_modality_ids:
      # Mask modality ids that are bias free.
      ids_to_mask = jnp.expand_dims(
          jnp.asarray(self.bias_free_modality_ids, encoder_bias.dtype), 1)
      mask_1d = jnp.any(jnp.equal(q_subsegments, ids_to_mask), axis=0)
      mask_2d = jnp.logical_and(
          jnp.expand_dims(mask_1d, 0), jnp.expand_dims(mask_1d, 1))
      mask_2d = 1.0 - jnp.asarray(mask_2d, encoder_bias.dtype)
      encoder_bias *= mask_2d[jnp.newaxis, jnp.newaxis, ...]
    # Compute the modality-level bias.
    rp_bucket = q_subsegments + self.max_num_modalities * k_subsegments
    # Instead of using a slow gather, we create a leading-dimension one-hot
    # array from rp_bucket and use it to perform the gather-equivalent via a
    # contraction, i.e.:
    # (num_heads, qlen, klen) =
    #   (num_heads, num_buckets) x (num_buckets, qlen, klen)
    # This is equivalent to bias_params[:, rp_bucket]
    bcast_iota = lax.broadcasted_iota(jnp.int32, (num_buckets, 1, 1), 0)
    rp_bucket_one_hot = jnp.array(
        rp_bucket[jnp.newaxis, ...] == bcast_iota, dtype=self.dtype)
    multimodal_encoder_bias = lax.dot_general(
        bias_params,
        rp_bucket_one_hot,
        (
            ((1,), (0,)),  # lhs, rhs contracting dims
            ((), ())))  # no batched dims
    # Add a batch dimension
    multimodal_encoder_bias = multimodal_encoder_bias[jnp.newaxis, ...]
    return multimodal_encoder_bias + encoder_bias
