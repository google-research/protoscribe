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

"""Tests for `pmmx_architecture` library."""

from absl import logging  # pylint: disable=unused-import
from absl.testing import absltest
import gin
from jax import random
import numpy as np
from protoscribe.pmmx import multimodal_feature
from protoscribe.pmmx import pmmx_architecture
from protoscribe.pmmx.utils import t5_save_format

from flaxformer import testing_utils

expected_files = testing_utils.ExpectedJsonFiles(
    'protoscribe/pmmx/testdata')
check_params = expected_files.check_params_shapes_only

_INPUT_TOKENS = np.array(
    [
        # Batch 1.
        [101, 183, 20, 75],
        # Batch 2.
        [101, 392, 19, 7],
    ],
    dtype=np.int32)

_INPUT_VECTORS = np.array(
    [
        [[1, 2], [3, 4], [5, 6]],
        [[-1, -2], [-3, -4], [0, 0]],
    ],
    dtype=np.float32)


class PmmxArchitectureMultimodalTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    gin.add_config_file_search_path('protoscribe/pmmx/configs')
    gin.parse_config_file('models/p1_t5_1_1_testing.gin')

  def tearDown(self):
    super().tearDown()
    gin.clear_config()

  def test_encoder_shapes(self):
    """Tests if the encoder has correct shapes."""
    transformer = gin.get_configurable(
        pmmx_architecture.MultimodalEncoderDecoder)()
    encoder_batch = {
        'text_tokens': _INPUT_TOKENS,
        'image_v2_dense': _INPUT_VECTORS
    }
    (output, mask, segment_ids), variables = transformer.init_with_output(
        random.PRNGKey(0), encoder_batch, method=transformer.encode,
        enable_dropout=False)
    params = variables['params']
    reformatted = transformer.apply({},
                                    params,
                                    method=transformer.to_save_format)
    check_params(reformatted, 'encoder_shapes.json')
    self.assertEqual(output.shape, (2, 7, 7))
    self.assertEqual(mask.shape, (2, 7))
    self.assertIsNone(segment_ids)

    # Convert back to Flax module structure format and test again.
    params2 = t5_save_format.load(transformer, reformatted)
    encoder_batch = {
        'text_tokens': _INPUT_TOKENS,
        'image_v2_dense': _INPUT_VECTORS
    }
    (output2, mask2, segment_ids2) = transformer.apply(
        {'params': params2}, encoder_batch, method=transformer.encode,
        enable_dropout=False)
    np.testing.assert_allclose(output, output2, rtol=1e-8)
    np.testing.assert_allclose(mask, mask2, rtol=1e-8)
    self.assertIsNone(segment_ids2)

  def test_sow_intermediates(self):
    """Tests intermediate tracking using `Module.sow` in the EncoderDecoder."""
    transformer = gin.get_configurable(
        pmmx_architecture.MultimodalEncoderDecoder
    )()
    encoder_batch = {
        'text_tokens': _INPUT_TOKENS,
        'image_v2_dense': _INPUT_VECTORS
    }
    _, variables = transformer.init_with_output(
        random.PRNGKey(0),
        encoder_batch,
        method=transformer.encode,
        enable_dropout=False,
        capture_intermediates=True,
    )

    # Note: the 'intermediates' collection must be set to mutable in order to
    # get the tracked values back in `modified_variables`.
    # Check the shape of tracked intermediates.
    intermediates = variables['intermediates']

    final_encoder_outputs = intermediates['encoder']['final_encoder_outputs']
    self.assertLen(final_encoder_outputs, 1)
    self.assertEqual(final_encoder_outputs[0].shape, (2, 7, 7))

  def test_entire_transformer_shared_embeds(self):
    transformer = gin.get_configurable(
        pmmx_architecture.MultimodalEncoderDecoder)()

    encoder_input_tokens = np.zeros((16, 8), dtype=np.float32)
    encoder_input_vectors = np.zeros((16, 5, 3), dtype=np.float32)
    decoder_input_tokens = np.zeros((16, 8), dtype=np.float32)
    decoder_target_tokens = np.zeros((16, 8), dtype=np.float32)

    batch = {
        'text_tokens': encoder_input_tokens,
        'image_v2_dense': encoder_input_vectors,
        'decoder_target_tokens': decoder_target_tokens,
        'decoder_input_tokens': decoder_input_tokens
    }

    output, variables = transformer.init_with_output(
        random.PRNGKey(0), batch, enable_dropout=False)
    params = variables['params']
    reformatted = transformer.apply({}, params,
                                    method=transformer.to_save_format)
    check_params(
        reformatted, 'encoder_decoder_shared_embedding_shapes.json')
    self.assertEqual(output.shape, (16, 8, 32128))

    # Convert back to Flax module structure format and test again.
    params2 = t5_save_format.load(transformer, reformatted)
    output2 = transformer.apply({'params': params2},
                                batch,
                                enable_dropout=False)
    np.testing.assert_allclose(output, output2, rtol=1e-8)

  def test_entire_transformer_shared_embeds_extra_features(self):
    gin.parse_config_file('models/p1_t5_1_1_testing_extra_features.gin')
    transformer = gin.get_configurable(
        pmmx_architecture.MultimodalEncoderDecoder)()

    encoder_input_tokens = np.zeros((16, 8), dtype=np.float32)
    encoder_input_vectors = np.zeros((16, 5, 3), dtype=np.float32)
    decoder_input_tokens = np.zeros((16, 8), dtype=np.float32)
    decoder_target_tokens = np.zeros((16, 8), dtype=np.float32)
    extra = np.zeros((16, 8), dtype=np.float32)
    extra_input_vectors = np.zeros((16, 8, 2), dtype=np.float32)

    batch = {
        'text_tokens': encoder_input_tokens,
        'new_tokens': encoder_input_tokens,
        'image_v2_dense': encoder_input_vectors,
        'decoder_target_tokens': decoder_target_tokens,
        'decoder_input_tokens': decoder_input_tokens,
        'extra1': extra,
        'extra2': extra,
        'extra3': extra_input_vectors,
    }

    output, variables = transformer.init_with_output(
        random.PRNGKey(0), batch, enable_dropout=False)
    params = variables['params']
    reformatted = transformer.apply({}, params,
                                    method=transformer.to_save_format)
    check_params(reformatted,
                 'encoder_decoder_shared_embedding_shapes_extra_features.json')
    self.assertEqual(output.shape, (16, 8, 32128))

    # Convert back to Flax module structure format and test again.
    params2 = t5_save_format.load(transformer, reformatted)
    output2 = transformer.apply({'params': params2},
                                batch,
                                enable_dropout=False)
    np.testing.assert_allclose(output, output2, rtol=1e-8)

  def test_entire_transformer_shared_embeds_sub_features(self):
    gin.parse_config_file(
        'models/p1_t5_1_1_testing_shared_embeds_sub_features.gin')
    transformer = gin.get_configurable(
        pmmx_architecture.MultimodalEncoderDecoder)()

    encoder_input_tokens = np.zeros((16, 8), dtype=np.float32)
    encoder_input_frames = np.zeros((16, 5, 3), dtype=np.float32)
    encoder_input_token_timestamps = np.zeros((16, 8, 1), dtype=np.float32)
    encoder_input_frame_timestamps = np.zeros((16, 5, 1), dtype=np.float32)

    decoder_input_tokens = np.zeros((16, 8), dtype=np.float32)
    decoder_target_tokens = np.zeros((16, 8), dtype=np.float32)

    batch = {
        'text_tokens': encoder_input_tokens,
        'frame_dense': encoder_input_frames,
        'text_timestamps': encoder_input_token_timestamps,
        'frame_timestamps': encoder_input_frame_timestamps,
        'decoder_target_tokens': decoder_target_tokens,
        'decoder_input_tokens': decoder_input_tokens,
    }

    output, variables = transformer.init_with_output(
        random.PRNGKey(0), batch, enable_dropout=False)
    params = variables['params']
    reformatted = transformer.apply({}, params,
                                    method=transformer.to_save_format)
    check_params(reformatted,
                 'encoder_decoder_shared_embedding_shapes_sub_features.json')
    self.assertEqual(output.shape, (16, 8, 32128))

    # Convert back to Flax module structure format and test again.
    params2 = t5_save_format.load(transformer, reformatted)
    output2 = transformer.apply({'params': params2},
                                batch,
                                enable_dropout=False)
    np.testing.assert_allclose(output, output2, rtol=1e-8)

  def test_transformer_decode_shapes(self):
    """Tests if the decoding parameters have the expected shapes."""
    transformer = gin.get_configurable(
        pmmx_architecture.MultimodalEncoderDecoder)()
    decoder_input_tokens = np.array(
        [
            # Batch 1.
            [183, 20, 75],
            # Batch 2.
            [392, 19, 7],
        ],
        dtype=np.int32)
    decoder_batch = {
        'decoder_input_tokens': decoder_input_tokens,
        'decoder_target_tokens': decoder_input_tokens  # for mask generation
    }
    encoder_batch = {
        'text_tokens': np.zeros([2, 3], dtype=np.int32),
        'image_v2_dense': np.zeros([2, 3, 2], dtype=np.float32),
    }
    feature_spec = (
        ('text_tokens', 'text_tokens'), ('image_v2_dense', 'image_v2_dense'))
    modality_spec = ('text_tokens', 'image_v2_dense')
    encoder_features = multimodal_feature.linearize_encoder_features(
        encoder_batch, feature_spec, modality_spec)
    encoder_mask = multimodal_feature.attention_mask_for_zeros(
        [pf.values for pf in encoder_features])
    encoder_segment_ids = None
    output, variables = transformer.init_with_output(
        random.PRNGKey(0),
        encoded=None,
        encoder_mask=encoder_mask,
        encoder_segment_ids=encoder_segment_ids,
        decoder_batch=decoder_batch,
        enable_dropout=False,
        method=transformer.decode)
    params = variables['params']
    reformatted = transformer.apply({},
                                    params,
                                    method=transformer.to_save_format)
    check_params(reformatted, 'decoder_shapes_per_layer.json')
    self.assertEqual(output.shape, (2, 3, 32128))

    # Convert back to Flax module structure format and test again.
    params2 = t5_save_format.load(transformer, reformatted)
    output2 = transformer.apply(
        {'params': params2},
        encoded=None,
        encoder_mask=encoder_mask,
        encoder_segment_ids=encoder_segment_ids,
        decoder_batch=decoder_batch,
        enable_dropout=False,
        method=transformer.decode,
    )
    np.testing.assert_allclose(output, output2, rtol=1e-8)

  def test_encoder_example_packing(self):
    transformer = gin.get_configurable(
        pmmx_architecture.MultimodalEncoderDecoder)()
    encoder_input_tokens = np.array(
        [
            # Batch 1.
            [101, 183, 20, 75],
            # Batch 2 (last id is pad)
            [101, 392, 19, 0],
        ],
        dtype=np.int32)
    encoder_input_vectors = np.array(
        [
            # Batch 1.
            [[1, 2], [3, 4]],
            # Batch 2 (last vector is pad).
            [[5, 6], [0, 0]],
        ],
        dtype=np.float32)
    batch = {
        'text_tokens': encoder_input_tokens,
        'image_v2_dense': encoder_input_vectors,
    }
    (output, _, _), variables = transformer.init_with_output(
        random.PRNGKey(0),
        batch,
        enable_dropout=False,
        method=transformer.encode,
    )

    encoder_input_tokens_packed = np.array([[101, 183, 20, 75, 101, 392, 19]],
                                           dtype=np.int32)
    encoder_input_vectors_packed = np.array([[[1, 2], [3, 4], [5, 6]]],
                                            dtype=np.float32)
    encoder_token_segment_ids = np.array([[0, 0, 0, 0, 1, 1, 1]],
                                         dtype=np.int32)
    encoder_vector_segment_ids = np.array([[0, 0, 1]], dtype=np.int32)
    encoder_token_positions = np.array([[0, 1, 2, 3, 0, 1, 2]], dtype=np.int32)
    encoder_vector_positions = np.array([[0, 1, 0]], dtype=np.int32)

    batch = {
        'text_tokens': encoder_input_tokens_packed,
        'image_v2_dense': encoder_input_vectors_packed,
        'text_tokens_positions': encoder_token_positions,
        'image_v2_dense_positions': encoder_vector_positions,
        'text_tokens_segment_ids': encoder_token_segment_ids,
        'image_v2_dense_segment_ids': encoder_vector_segment_ids,
    }

    (output_packed, _, _) = transformer.apply(
        variables,
        batch,
        enable_dropout=False,
        method=transformer.encode,
    )

    # Check the first batch (4 tokens, 2 vectors).
    np.testing.assert_allclose(
        output[0, 0:4, :], output_packed[0, 0:4, :], rtol=1e-3)
    np.testing.assert_allclose(
        output[0, 4:6, :], output_packed[0, 7:9, :], rtol=1e-3)

    # Check the second batch (3 tokens, 1 vector).
    np.testing.assert_allclose(
        output[1, 0:3, :], output_packed[0, 4:7, :], rtol=1e-3)
    np.testing.assert_allclose(
        output[1, 4:5, :], output_packed[0, 9:10, :], rtol=1e-3)

  def test_sinusoidal_embeddings(self):
    embed = pmmx_architecture.SinusoidalEmbed(features=4)
    position_ids = np.array([[0, 1, 2, 0], [0, 0, 1, 2]], dtype=np.int32)
    position_embeds = embed.apply({}, position_ids)
    np.testing.assert_allclose(
        position_embeds,
        [[[0., 0., 1., 1.],
          [.8415, .0001, .5403, 1.],
          [.9093, .0002, -.4161, 1.],
          [0., 0., 1., 1.]],
         [[0., 0., 1., 1.],
          [0., 0., 1., 1.],
          [.8415, .0001, .5403, 1.],
          [.9093, .0002, -.4161, 1.]]],
        atol=5e-5)

  def test_encoder_outputs_as_dict(self):
    """Test that encode outputs dicts when (outputs_as_dict == True)."""
    gin.parse_config_file(
        'models/p1_t5_1_1_testing_multimodal_encoder_outputs_as_dict.gin')
    transformer = gin.get_configurable(
        pmmx_architecture.MultimodalEncoderDecoder)()
    encoder_batch = {
        'text_tokens': _INPUT_TOKENS,
        'image_v2_dense': _INPUT_VECTORS
    }
    (outputs, masks, segment_ids), _ = transformer.init_with_output(
        random.PRNGKey(0), encoder_batch,
        method=transformer.encode, enable_dropout=False)

    # Assert the output types.
    self.assertIsInstance(outputs, dict)
    self.assertIsInstance(masks, dict)
    self.assertIsNone(segment_ids)

    # Assert the output shapes.
    self.assertEqual(outputs['text_tokens'].shape, (2, 4, 13))
    self.assertEqual(outputs['image_v2_dense'].shape, (2, 3, 13))
    self.assertEqual(masks['text_tokens'].shape, (2, 4))
    self.assertEqual(masks['image_v2_dense'].shape, (2, 3))


if __name__ == '__main__':
  absltest.main()
