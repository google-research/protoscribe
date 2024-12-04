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

"""Tests for multimodal_feature.py."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from protoscribe.pmmx import multimodal_feature

_TEXT_FEATURE = multimodal_feature.MultimodalFeature(
    name='text_tokens',
    values=np.array([[0, 1, 2, 3, 4]]),
    modality_name='text_tokens',
    modality_id=0,
    positions=None,
    segment_ids=None)

_IMAGE_FEATURE = multimodal_feature.MultimodalFeature(
    name='image_dense',
    values=np.array([[[2, 3], [0, 0], [.1, 0.]]], float),
    modality_name='image_dense',
    modality_id=1,
    positions=None,
    segment_ids=None)


class MultimodalFeatureTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          batch={
              'image_dense': 3,
              'text_tokens': 2,
              'title_text_tokens': 1,
          },
          feature_spec=(
              ('title_text_tokens', 'text_tokens'),
              ('text_tokens', 'text_tokens'),
              ('image_dense', 'image_dense')
          ),
          modality_spec=('image_dense', 'text_tokens'),
          expect_encoder_features=[
              multimodal_feature.MultimodalFeature(
                  'title_text_tokens', np.array(1), 'text_tokens', 1,
                  None, None),
              multimodal_feature.MultimodalFeature(
                  'text_tokens', np.array(2), 'text_tokens', 1,
                  None, None),
              multimodal_feature.MultimodalFeature(
                  'image_dense', np.array(3), 'image_dense', 0,
                  None, None),
          ],
          expect_embedder_names=(
              'embedder', 'embedder', 'image_dense_embedder'
          ),
      ),
      dict(
          batch={
              'text_tokens': 2,
              'text_tokens_positions': 6,
              'text_tokens_segment_ids': 8,
          },
          feature_spec=(
              ('title_text_tokens', 'text_tokens'),
              ('text_tokens', 'text_tokens'),
              ('image_dense', 'image_dense')
          ),
          modality_spec=('image_dense', 'text_tokens'),
          expect_encoder_features=[
              multimodal_feature.MultimodalFeature(
                  'text_tokens', np.array(2), 'text_tokens', 1,
                  np.array(6), np.array(8)),
          ],
          expect_embedder_names=('embedder',),
      ),
  )
  def test_linearize_encoder_features(self, batch, feature_spec, modality_spec,
                                      expect_encoder_features,
                                      expect_embedder_names):
    encoder_features = multimodal_feature.linearize_encoder_features(
        batch, feature_spec, modality_spec)
    self.assertEqual(encoder_features, expect_encoder_features)
    for (f, e) in zip(encoder_features, expect_embedder_names):
      self.assertEqual(f.embedder_name, e)

  @parameterized.parameters(
      dict(
          inputs=[
              np.array([[1, 2, 3, 0, 0], [0, 5, 4, 0, 1]]),
              np.array([[[1, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, .001, 0]]],
                       dtype=float)
          ],
          expect=np.array([[True, True, True, False, False, True, False],
                           [False, True, True, False, True, False, True]])
      ),
  )
  def test_attention_mask_for_zeros(self, inputs, expect):
    actual = multimodal_feature.attention_mask_for_zeros(inputs)
    np.testing.assert_allclose(actual, expect)

  @parameterized.parameters(
      dict(
          feature=multimodal_feature.MultimodalFeature(
              name='blah_text_tokens',
              values=np.array(1),
              modality_name='text_tokens',
              modality_id=0,
              positions=np.array(3),
              segment_ids=np.array(4)),
          expect_embedder_name='embedder'
      ),
      dict(
          feature=multimodal_feature.MultimodalFeature(
              name='image_dense',
              values=np.array(1),
              modality_name='image_dense',
              modality_id=1,
              positions=np.array(3),
              segment_ids=np.array(4)),
          expect_embedder_name='image_dense_embedder'
      ),
  )
  def test_multimodal_feature(self, feature, expect_embedder_name):
    self.assertEqual(feature.embedder_name, expect_embedder_name)

  @parameterized.parameters(
      dict(
          feature=_TEXT_FEATURE,
          expect=np.array([0, 0, 0, 0, 0])
      ),
      dict(
          feature=_IMAGE_FEATURE,
          expect=np.array([1, 1, 1])
      ),
  )
  def test_make_modality_segment_ids(self, feature, expect):
    np.testing.assert_allclose(
        feature.make_modality_segment_ids(np.int32), expect)

  @parameterized.parameters(
      dict(
          features=[_TEXT_FEATURE],
          expect=multimodal_feature.SequenceMetadata(
              modality_segment_ids=np.array([0, 0, 0, 0, 0]),
              feature_name_to_segment_id_map={'text_tokens': 0},
              feature_name_to_bounds_map={'text_tokens': (0, 5)}),
      ),
      dict(
          features=[_IMAGE_FEATURE],
          expect=multimodal_feature.SequenceMetadata(
              modality_segment_ids=np.array([1, 1, 1]),
              feature_name_to_segment_id_map={'image_dense': 1},
              feature_name_to_bounds_map={'image_dense': (0, 3)}),
      ),
      dict(
          features=[_TEXT_FEATURE, _IMAGE_FEATURE],
          expect=multimodal_feature.SequenceMetadata(
              modality_segment_ids=np.array([0, 0, 0, 0, 0, 1, 1, 1]),
              feature_name_to_segment_id_map={
                  'text_tokens': 0,
                  'image_dense': 1
              },
              feature_name_to_bounds_map={
                  'text_tokens': (0, 5),
                  'image_dense': (5, 8),
              }),
      ),
      dict(
          features=[_IMAGE_FEATURE, _TEXT_FEATURE],
          expect=multimodal_feature.SequenceMetadata(
              modality_segment_ids=np.array([1, 1, 1, 0, 0, 0, 0, 0]),
              feature_name_to_segment_id_map={
                  'text_tokens': 0,
                  'image_dense': 1
              },
              feature_name_to_bounds_map={
                  'text_tokens': (3, 8),
                  'image_dense': (0, 3),
              }),
      ),
  )
  def test_make_sequence_metadata(
      self, features, expect: multimodal_feature.SequenceMetadata):
    actual = multimodal_feature.make_sequence_metadata(
        features, np.int32  # pytype: disable=wrong-arg-types # np-dtype
    )
    for field in dataclasses.fields(multimodal_feature.SequenceMetadata):
      actual_x = getattr(actual, field.name)
      expect_x = getattr(expect, field.name)
      if isinstance(actual_x, np.ndarray) or isinstance(expect_x, np.ndarray):
        np.testing.assert_allclose(actual_x, expect_x)
      else:
        self.assertEqual(actual_x, expect_x)

  @parameterized.parameters(
      dict(
          batch=[
              _TEXT_FEATURE, _IMAGE_FEATURE
          ],
          expects=[
              np.array([[[[0, 0, 0, 0, 0, 0, 0, 0],  # text_tokens=0
                          [0, 1, 1, 1, 1, 1, 0, 1],
                          [0, 1, 1, 1, 1, 1, 0, 1],
                          [0, 1, 1, 1, 1, 1, 0, 1],
                          [0, 1, 1, 1, 1, 1, 0, 1]]]], np.int32),
              np.array([[[[0, 1, 1, 1, 1, 1, 0, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0],  # image_dense=[0, 0]
                          [0, 1, 1, 1, 1, 1, 0, 1]]]], np.int32),
          ]
      )
  )
  def test_make_attention_mask(self, batch, expects):
    for feature, expect in zip(batch, expects):
      actual = feature.make_attention_mask(batch, np.int32)
      np.testing.assert_allclose(actual, expect)

  @parameterized.parameters(
      dict(
          encoder_features=[_TEXT_FEATURE, _IMAGE_FEATURE],
          expected=np.array(
              [[[
                  [0, 0, 0, 0, 0, 0, 0, 0],  # text_tokens=0
                  [0, 1, 1, 1, 1, 1, 0, 1],
                  [0, 1, 1, 1, 1, 1, 0, 1],
                  [0, 1, 1, 1, 1, 1, 0, 1],
                  [0, 1, 1, 1, 1, 1, 0, 1],
                  [0, 1, 1, 1, 1, 1, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0],  # image_dense=[0, 0]
                  [0, 1, 1, 1, 1, 1, 0, 1]
              ]]],
              np.int32)))
  def test_make_encoder_mask_fn(self, encoder_features, expected):
    encoder_mask, encoder_segment_ids = multimodal_feature.make_encoder_mask(
        encoder_features, np.int32  # pytype: disable=wrong-arg-types # np-dtype
    )
    np.testing.assert_allclose(encoder_mask, expected)
    self.assertIsNone(encoder_segment_ids)


if __name__ == '__main__':
  absltest.main()
