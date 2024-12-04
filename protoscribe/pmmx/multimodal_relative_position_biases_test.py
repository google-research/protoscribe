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

"""Tests for multimodal_relative_position_biases."""

from absl import logging  # pylint: disable=unused-import
from absl.testing import absltest
from absl.testing import parameterized
import flax.linen as nn
import jax
from jax import random
import jax.numpy as jnp
# import numpy as np
from protoscribe.pmmx import multimodal_relative_position_biases

from flaxformer.components import relative_position_biases


class MultimodalRelativePositionBiasesTest(parameterized.TestCase,
                                           absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.num_heads = 3
    self.relative_attention = relative_position_biases.RelativePositionBiases(
        num_buckets=12,
        max_distance=10,
        num_heads=3,
        dtype=jnp.float32,
    )
    self.features = {
        'input_tokens': jnp.full([2, 2], 1),
        'input_vectors': jnp.full([2, 3, 1], 2),
    }
    self.modality_segment_ids = jnp.array([0, 0, 1, 1, 1], dtype=jnp.int32)
    self.query_len = 5
    self.key_len = 5

  @parameterized.named_parameters(
      ('no_bias_masking', ()), ('bias_masking', (0,)))
  def test_params(self, bias_free_modality_ids):
    """Tests that bidirectional relative position biases have expected params."""
    outputs, _ = self.relative_attention.init_with_output(
        random.PRNGKey(0), self.query_len, self.key_len, bidirectional=True)
    multimodal_relative_attention = (
        multimodal_relative_position_biases.MultimodalRelativePositionBiases(
            num_heads=3,
            bias_free_modality_ids=bias_free_modality_ids,
            dtype=jnp.float32,
        ))
    params = multimodal_relative_attention.init(
        random.PRNGKey(0), outputs, self.modality_segment_ids)
    param_shapes = jax.tree.map(lambda x: x.shape, params)
    self.assertEqual(param_shapes, {
        'params': {
            'rel_embedding': (3, 256),
        },
        'params_axes': {
            'rel_embedding_axes': nn.partitioning.AxisMetadata(
                names=('heads', 'relpos_buckets')),
        },
    })

  def test_regression_values(self):
    """Tests that bidirectional relative position biases match expected values.

    See top docstring note on matching P5X behavior for these regression tests.
    """
    outputs, _ = self.relative_attention.init_with_output(
        random.PRNGKey(0), self.query_len, self.key_len, bidirectional=True)
    self.assertEqual(outputs.shape,
                     (1, self.num_heads, self.query_len, self.key_len))
    multimodal_relative_attention = (
        multimodal_relative_position_biases.MultimodalRelativePositionBiases(
            num_heads=3,
            dtype=jnp.float32,
        ))
    outputs, _ = multimodal_relative_attention.init_with_output(
        random.PRNGKey(0), outputs, self.modality_segment_ids)
    self.assertEqual(outputs.shape,
                     (1, self.num_heads, self.query_len, self.key_len))
    # TODO: Somehow show below the multimodal attention bias scheme,
    # with intra-modality biases set to the sequence-level biases, and
    # cross-modality biases set to a flat value, independent of sequence
    # position.
    #
    # Additional Flax maintainer note: please think about using a property
    # test rather than hard-coding the values.

  def test_bias_masking(self):
    outputs, _ = self.relative_attention.init_with_output(
        random.PRNGKey(0), self.query_len, self.key_len, bidirectional=True)
    self.assertEqual(outputs.shape,
                     (1, self.num_heads, self.query_len, self.key_len))
    multimodal_relative_attention = (
        multimodal_relative_position_biases.MultimodalRelativePositionBiases(
            num_heads=3,
            dtype=jnp.float32,
            bias_free_modality_ids=(1,),
        ))
    outputs, _ = multimodal_relative_attention.init_with_output(
        random.PRNGKey(0), outputs, self.modality_segment_ids)
    self.assertEqual(outputs.shape,
                     (1, self.num_heads, self.query_len, self.key_len))


if __name__ == '__main__':
  absltest.main()
