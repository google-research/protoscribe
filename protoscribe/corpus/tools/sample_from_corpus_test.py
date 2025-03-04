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

import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from protoscribe.corpus.tools import sample_from_corpus as lib
from protoscribe.sketches.utils import stroke_tokenizer as tokenizer_lib
from protoscribe.sketches.utils import stroke_utils as stroke_lib

import glob
import os

# Accounting document in sketch-5 format that includes BOS and EOS vectors.
_DOC_NORMALIZED = np.array(
    [
        [0., 0., 1., 0., 0.],
        [0.4201021, 0.5, 1., 0., 0.],
        [0.43676072, 0.6175065, 0., 1., 0.],
        [0.780187, 0.5761741, 1., 0., 0.],
        [0.44013697, 0.533266, 1., 0., 0.],
        [0.42301813, 0.51763576, 1., 0., 0.],
        [0.41337067, 0.49660704, 1., 0., 0.],
        [0.40538824, 0.49817938, 1., 0., 0.],
        [0.41127518, 0.50275344, 1., 0., 0.],
        [0.4087121, 0.50840235, 1., 0., 0.],
        [0.43643135, 0.47344077, 1., 0., 0.],
        [0.42362323, 0.49206588, 1., 0., 0.],
        [0.4221357, 0.48033512, 1., 0., 0.],
        [0.4176514, 0.49348798, 1., 0., 0.],
        [0.41829592, 0.50508344, 0., 1., 0.],
        [0.4828203, 0.4504948, 1., 0., 0.],
        [0.43003374, 0.5016584, 1., 0., 0.],
        [0.43130848, 0.4974074, 1., 0., 0.],
        [0.4427408, 0.48692864, 1., 0., 0.],
        [0.43054867, 0.4910191, 1., 0., 0.],
        [0.4132781, 0.5071517, 1., 0., 0.],
        [0.41441515, 0.50949436, 1., 0., 0.],
        [0.41038048, 0.52534735, 1., 0., 0.],
        [0.41752526, 0.5138275, 1., 0., 0.],
        [0.4139719, 0.49438402, 1., 0., 0.],
        [0.40270454, 0.474593, 0., 1., 0.],
        [0.4288707, 0.5145928, 1., 0., 0.],
        [0.35401392, 0.55719376, 0., 1., 0.],
        [0., 0., 0., 0., 1.],
    ],
    dtype=np.float32
)

# Indices of the points where the strokes end in the sketch-3 format document.
_EXPECTED_LIFT_POINT_INDICES = [1, 13, 24, 26]

# Mock a very basic codebook with a fixed number of clusters.
_NUM_CLUSTERS = 256
_RANDOM_SEED = 42
_RANDOM_CODEBOOK = np.random.default_rng(seed=_RANDOM_SEED).standard_normal(
    size=(_NUM_CLUSTERS, 2), dtype=np.float32
)

# Expected number of strokes for the document defined above.
_EXPECTED_NUM_STROKES = 4

_MAX_TOKEN_SEQUENCE_LENGTH = 64


class SampleFromCorpusTest(parameterized.TestCase):

  def _make_random_tokenizer(self) -> tokenizer_lib.StrokeTokenizer:
    """Makes tokenizer with random codebook."""
    codebook_path = os.path.join(
        absltest.get_default_test_tmpdir(), "mock_random_codebook.npy"
    )
    with open(codebook_path, mode="wb") as f:
      np.save(f, _RANDOM_CODEBOOK)
    return tokenizer_lib.StrokeTokenizer(
        codebook_path, _MAX_TOKEN_SEQUENCE_LENGTH
    )

  def _make_identity_tokenizer(self) -> tokenizer_lib.StrokeTokenizer:
    """Makes tokenizer with centroids coinciding with the inputs."""
    codebook = _DOC_NORMALIZED[1:-1, :2]
    codebook_path = os.path.join(
        absltest.get_default_test_tmpdir(), "mock_identity_codebook.npy"
    )
    with open(codebook_path, mode="wb") as f:
      np.save(f, codebook)
    return tokenizer_lib.StrokeTokenizer(
        codebook_path, _MAX_TOKEN_SEQUENCE_LENGTH
    )

  def setUp(self):
    super().setUp()

    self._random_tokenizer = self._make_random_tokenizer()
    self._identity_tokenizer = self._make_identity_tokenizer()

  @parameterized.named_parameters(
      ("slow-tokenizer", lib.TokenizerMode.SLOW),
      ("jax-tokenizer", lib.TokenizerMode.JAX),
      ("tf-tokenizer", lib.TokenizerMode.TF),
  )
  def test_tokenize_and_reconstruct_with_random_codebook(
      self, tokenizer_mode: lib.TokenizerMode
  ) -> None:
    glyph_affiliations = np.ones((_DOC_NORMALIZED.shape[0],))
    doc_stroke3 = lib.tokenize_and_reconstruct(
        _DOC_NORMALIZED,
        glyph_affiliations=glyph_affiliations,
        tokenizer=self._random_tokenizer,
        tokenizer_mode=tokenizer_mode
    )
    # The sketch-3 array shape should exclude the BOS and EOS vectors.
    self.assertEqual(doc_stroke3.shape, (_DOC_NORMALIZED.shape[0] - 2, 3))

    # Count the number of times the pen is stylus is lifted from the paper. This
    # corresponds to the number of strokes.
    lift_point_indices = np.where(doc_stroke3[:, 2] == 1.)
    lift_point_indices = lift_point_indices[0].tolist()
    self.assertLen(lift_point_indices, _EXPECTED_NUM_STROKES)
    self.assertEqual(_EXPECTED_LIFT_POINT_INDICES, lift_point_indices)

  @parameterized.named_parameters(
      ("slow-tokenizer", lib.TokenizerMode.SLOW),
      ("jax-tokenizer", lib.TokenizerMode.JAX),
      ("tf-tokenizer", lib.TokenizerMode.TF),
  )
  def test_tokenize_and_reconstruct_with_identity_codebook(
      self, tokenizer_mode: lib.TokenizerMode
  ) -> None:
    reference_stroke3 = stroke_lib.stroke5_to_stroke3(_DOC_NORMALIZED)
    glyph_affiliations = np.ones((_DOC_NORMALIZED.shape[0],))
    doc_stroke3 = lib.tokenize_and_reconstruct(
        _DOC_NORMALIZED,
        glyph_affiliations=glyph_affiliations,
        tokenizer=self._identity_tokenizer,
        tokenizer_mode=tokenizer_mode
    )
    codebook = _DOC_NORMALIZED[1:-1, :]
    self.assertEqual(doc_stroke3.shape[0], codebook.shape[0])
    self.assertEqual(reference_stroke3.shape, doc_stroke3.shape)
    self.assertEqual(doc_stroke3.tolist(), reference_stroke3.tolist())


if __name__ == "__main__":
  absltest.main()
