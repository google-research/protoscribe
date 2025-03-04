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

"""Tests for stroke tokenizer."""

import os
from typing import Tuple

from absl import flags
from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from protoscribe.sketches.utils import stroke_tokenizer as lib
from protoscribe.sketches.utils import stroke_utils
import tensorflow as tf

FLAGS = flags.FLAGS

_STROKE_LENGTH = 500
_STROKE_DIM = 3
_GLYPH_ID_MIN = 5
_GLYPH_ID_MAX = 100


class StrokeTokenizerTest(absltest.TestCase):

  def _get_tokenizer(self, max_seq_length: int) -> lib.StrokeTokenizer:
    vocab_file = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/sketches/utils",
        "testdata/stroke_vocab1024.npy")
    return lib.StrokeTokenizer(vocab_file, max_seq_length)

  def _get_strokes_and_glyph_ids(self) -> Tuple[
      np.ndarray, np.ndarray, np.ndarray
  ]:
    strokes3 = np.random.rand(_STROKE_LENGTH, _STROKE_DIM)
    strokes3[:, -1] = np.random.randint(0, high=2, size=_STROKE_LENGTH)
    pen_up = np.where(strokes3[:, 2] == 1)[0]
    num_separators = pen_up.shape[0]
    glyph_ids = np.random.randint(
        _GLYPH_ID_MIN, high=_GLYPH_ID_MAX, size=_STROKE_LENGTH
    )
    return strokes3, num_separators, glyph_ids

  def test_encode_long_max_length(self):
    # Setup example stroke sequence in stroke-3 format.
    strokes3, num_seps, _ = self._get_strokes_and_glyph_ids()

    # Initialize and tokenize.
    max_length = 1000
    tokenizer = self._get_tokenizer(max_length)
    tokens = tokenizer.encode(jnp.array(strokes3))
    self.assertEqual(lib.Token.START_OF_SKETCH, tokens[0])
    self.assertEqual(max_length, tokens.shape[0])
    eos_pos = _STROKE_LENGTH + 2 + num_seps
    self.assertEqual(lib.Token.END_OF_SKETCH, tokens[eos_pos - 1])
    self.assertEqual(lib.Token.PAD, tokens[eos_pos])

    # Decode the tokens. While not comparing the delta coordinates because of
    # the quantization error, the pen states should match exactly.
    new_strokes3 = tokenizer.decode(tokens)
    np.testing.assert_array_equal(strokes3.shape, new_strokes3.shape)
    np.testing.assert_array_equal(strokes3[:, 2], new_strokes3[:, 2])

  def test_encode_no_pad(self):
    # Setup example stroke sequence in stroke-3 format.
    strokes3, _, _ = self._get_strokes_and_glyph_ids()

    # Initialize and tokenize.
    max_length = 500
    tokenizer = self._get_tokenizer(max_length)
    tokens = tokenizer.encode(jnp.array(strokes3))
    seq_length = max_length + 2  # Extra BOS and EOS.
    self.assertEqual(lib.Token.START_OF_SKETCH, tokens[0])
    self.assertEqual(seq_length, tokens.shape[0])
    eos_pos = seq_length
    self.assertEqual(lib.Token.END_OF_SKETCH, tokens[eos_pos - 1])

  def test_tf_encode_long_max_length(self):
    # Setup example stroke sequence in stroke-3 format.
    strokes3, num_seps, glyph_ids = self._get_strokes_and_glyph_ids()
    strokes3 = tf.convert_to_tensor(strokes3, dtype=tf.float32)
    glyph_ids = tf.convert_to_tensor(glyph_ids, dtype=tf.int32)

    # Initialize and tokenize.
    max_length = 1000
    tokenizer = self._get_tokenizer(max_length)
    tokens, glyph_ids, real_length = tokenizer.tf_encode(strokes3, glyph_ids)
    self.assertEqual(len(tokens), len(glyph_ids))
    self.assertEqual(lib.Token.START_OF_SKETCH, tokens[0])
    self.assertEqual(lib.Token.START_OF_SKETCH, glyph_ids[0])
    eos_pos = _STROKE_LENGTH + 2 + num_seps
    self.assertEqual(real_length, eos_pos - 2)
    self.assertEqual(lib.Token.END_OF_SKETCH, tokens[eos_pos - 1])
    self.assertEqual(lib.Token.END_OF_SKETCH, glyph_ids[eos_pos - 1])
    self.assertEqual(tokens[-1], lib.Token.PAD)
    self.assertEqual(glyph_ids[-1], lib.Token.PAD)

  def test_tf_encode_no_pad(self):
    # Setup example stroke sequence in stroke-3 format.
    strokes3, _, glyph_ids = self._get_strokes_and_glyph_ids()
    strokes3 = tf.convert_to_tensor(strokes3, dtype=tf.float32)
    glyph_ids = tf.convert_to_tensor(glyph_ids, dtype=tf.int32)

    # Initialize and tokenize.
    max_length = 500
    tokenizer = self._get_tokenizer(max_length)
    tokens, glyph_ids, real_length = tokenizer.tf_encode(strokes3, glyph_ids)
    self.assertEqual(real_length, max_length - 2)
    self.assertEqual(lib.Token.START_OF_SKETCH, tokens[0])
    self.assertEqual(lib.Token.START_OF_SKETCH, glyph_ids[0])
    self.assertEqual(tokens.shape[0], max_length)
    self.assertEqual(glyph_ids.shape[0], max_length)
    self.assertEqual(lib.Token.END_OF_SKETCH, tokens[max_length - 1])
    self.assertEqual(lib.Token.END_OF_SKETCH, glyph_ids[max_length - 1])

  def test_tf_insert(self):
    """Tests the core `tf_insert` API."""
    x = tf.constant([15])
    result = lib.tf_insert(x, 0, -1)
    np.testing.assert_array_equal(result, [-1, 15])
    result = lib.tf_insert(x, 1, -1)
    np.testing.assert_array_equal(result, [15, -1])

    x = tf.constant([25, 32, 14, 29, 54, 98])
    insert_pos = tf.constant([1, 3, 5])
    result = lib.tf_insert(x, insert_pos, 100)
    np.testing.assert_array_equal(
        result, [25, 100, 32, 14, 100, 29, 54, 100, 98])

    x = tf.constant([1, 0, 0, 1, 1])
    insert_pos = tf.constant([0, 3, 4])
    result = lib.tf_insert(x, insert_pos, -1)
    np.testing.assert_array_equal(
        result, [-1, 1, 0, 0, -1, 1, -1, 1])

  def test_tf_first_occurrence_index_above_max(self):
    sequence = [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 8.0, 10.0, 10.0]
    sequence = tf.convert_to_tensor(sequence)
    idx = lib.tf_first_occurrence_index_above_max(sequence, 1)
    self.assertEqual(idx, 0)
    idx = lib.tf_first_occurrence_index_above_max(sequence, 3)
    self.assertEqual(idx, 2)
    idx = lib.tf_first_occurrence_index_above_max(sequence, 5)
    self.assertEqual(idx, 4)
    idx = lib.tf_first_occurrence_index_above_max(sequence, 10)
    self.assertEqual(idx, 8)

  def test_tf_mark_concept_strokes(self):
    # Setup example stroke sequence in stroke-3 format.
    strokes3 = [
        [0.0, 0.0, 0.0], [0.903, 98.65, 1.0], [419.32, -59.74, 0.0],
        [-12.56, 126.28, 1.0], [494.62, -134.066, 0.0],
        [-18.27, 113.92, 1.0], [335.63, -158.26, 0.0], [-14.47, 17.116, 0.0],
        [-1.90, 11.97, 0.0], [8.17, 2.17, 0.0], [10.09, -4.05, 0.0],
        [8.28, -14.64, 0.0], [2.88, -18.63, 0.0], [-7.81, -2.82, 1.0]
    ]
    glyph_ids_per_stroke = [
        312, 312, 311, 311, 28, 28, 110, 110, 110, 110, 110, 110, 110, 110
    ]
    glyph_tokens = [312, 311, 28, 110]
    glyph_types = [1, 1, 2, 2]
    strokes3 = tf.convert_to_tensor(strokes3, dtype=tf.float32)
    glyph_ids_per_stroke = tf.convert_to_tensor(
        glyph_ids_per_stroke, dtype=tf.int32
    )
    glyph_tokens = tf.convert_to_tensor(glyph_tokens, dtype=tf.int32)
    glyph_types = tf.convert_to_tensor(glyph_types, dtype=tf.int32)

    # Initialize and tokenize.
    tokenizer = self._get_tokenizer(max_seq_length=20)
    stroke_tokens, glyph_ids_per_stroke, _ = tokenizer.tf_encode(
        strokes3, glyph_ids_per_stroke
    )

    # Mark the concepts. There should be one extra stroke token and one extra
    # glyph ID.
    stroke_tokens_new, glyph_ids_per_stroke = lib.tf_mark_concept_tokens(
        stroke_tokens, glyph_ids_per_stroke, glyph_tokens, glyph_types
    )
    self.assertEqual(stroke_tokens_new.shape[0], stroke_tokens.shape[0] + 1)
    self.assertEqual(glyph_ids_per_stroke.shape[0], stroke_tokens.shape[0] + 1)
    stroke_tokens_new = stroke_tokens_new.numpy().tolist()
    glyph_ids_per_stroke = glyph_ids_per_stroke.numpy().tolist()
    end_of_numbers_pos = 7  # Includes stroke separator tokens.
    self.assertEqual(
        end_of_numbers_pos, stroke_tokens_new.index(lib.Token.END_OF_NUMBERS)
    )
    self.assertEqual(
        end_of_numbers_pos,
        glyph_ids_per_stroke.index(stroke_utils.END_OF_STROKE)
    )


if __name__ == "__main__":
  absltest.main()
