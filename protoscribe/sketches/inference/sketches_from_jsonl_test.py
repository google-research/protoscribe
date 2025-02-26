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

"""Simple test for JSONL parser for sketches."""

import os
import random
from typing import Any

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
import ml_collections
import numpy as np
from protoscribe.corpus.reader import dataset_defaults as ds_lib
from protoscribe.sketches.inference import json_utils
from protoscribe.sketches.inference import sketches_from_jsonl as lib
from protoscribe.sketches.utils import stroke_stats as stats_lib
from protoscribe.texts import generate as generate_lib

_MAX_STROKE_SEQUENCE_LENGTH = 100
_STROKE_NORMALIZATION_TYPE = "sketch-rnn"
_TOKENIZER_FILE_NAME = "vocab2048_normalized_sketchrnn.npy"
_DATASET_DIR = (
    "protoscribe/sketches/inference/testdata"
)

FLAGS = flags.FLAGS

_EXPECTED_NUM_DOCS = 100
_NUM_DOCS_TO_SAMPLE = 10


class SketchesFromJsonlTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    FLAGS.dataset_dir = os.path.join(FLAGS.test_srcdir, _DATASET_DIR)
    FLAGS.skip_plots = True

    self.config = ml_collections.FrozenConfigDict({
        "max_stroke_sequence_length": _MAX_STROKE_SEQUENCE_LENGTH,
        "stroke_normalization_type": _STROKE_NORMALIZATION_TYPE,
        "stroke_tokenizer": ml_collections.FrozenConfigDict({
            "vocab_filename": _TOKENIZER_FILE_NAME,
        })
    })
    self.stroke_stats = stats_lib.load_stroke_stats(
        self.config, ds_lib.sketch_stroke_stats_file()
    )
    self.stroke_tokenizer = ds_lib.get_stroke_tokenizer(self.config)
    self.glyph_vocab = ds_lib.get_glyph_vocab()
    self.pronunciation_lexicon, _ = generate_lib.load_phonetic_forms(
        main_lexicon_file=ds_lib.main_lexicon_file(),
        number_lexicon_file=ds_lib.number_lexicon_file()
    )

  def _check_glyphs(self, d: dict[str, Any]) -> None:
    """Checks glyph information."""
    for feature in [
        "glyph.names", "glyph.scores", "glyph.names.best", "glyph.prons.best"
    ]:
      self.assertIn(feature, d)
      self.assertNotEmpty(d[feature])

    for feature in ["glyph.names", "glyph.prons"]:
      for hyp in d[feature]:
        self.assertIsInstance(hyp, list)
        self.assertNotEmpty(hyp)

    self.assertIn("sketch.confidence", d)
    self.assertGreater(d["sketch.confidence"], 0.)
    self.assertIn("glyph.confidence", d)
    self.assertGreater(d["glyph.confidence"], 0.)

  def _check_common_features(self, d: dict[str, Any]) -> None:
    """Checks that all the common features are there and sane."""
    for feature in [
        "concept.name", "concept.pron", "number.name",
        "text.sampa", "text.words", "text.inputs", "strokes.nbest.deltas"
    ]:
      self.assertIn(feature, d)
      self.assertNotEmpty(d[feature])

    self.assertIn("doc.id", d)
    self.assertGreater(d["doc.id"], 0)
    self.assertIsInstance(d["concept.pron"], list)

    # Basic check for decoded strokes in three-tuple representation. We need
    # to iterate over the n-best stroke sequences as these have varying length.
    self.assertIsInstance(d["strokes.nbest.deltas"], list)
    for stroke_seq in d["strokes.nbest.deltas"]:
      strokes = np.array(stroke_seq, np.float32)
      self.assertLen(strokes.shape, 2)
      self.assertEqual(strokes.shape[1], 3)

  @flagsaver.flagsaver
  def test_plain_sketch_tokens(self):
    FLAGS.recognizer_json = False
    FLAGS.combined_glyphs_and_strokes = False

    dicts = json_utils.load_jsonl(os.path.join(
        FLAGS.test_srcdir, _DATASET_DIR, "infer_eval_sketch_only.jsonl"
    ))
    self.assertLen(dicts, _EXPECTED_NUM_DOCS)
    for plain_dict in random.sample(dicts, _NUM_DOCS_TO_SAMPLE):
      d = lib.json_to_sketch(
          self.config, plain_dict, self.stroke_stats, self.stroke_tokenizer,
          self.glyph_vocab, self.pronunciation_lexicon, FLAGS.test_tmpdir
      )
      self._check_common_features(d)

      self.assertIn("sketch.confidence", d)
      self.assertEqual(d["sketch.confidence"], 0.)  # Beam-less test data.

  @flagsaver.flagsaver
  def test_glyph_recognition_from_sketch(self):
    # Running over the results of a recognizer.
    FLAGS.recognizer_json = True
    FLAGS.combined_glyphs_and_strokes = False

    dicts = json_utils.load_jsonl(os.path.join(
        FLAGS.test_srcdir, _DATASET_DIR,
        "infer_eval_glyph_recognition_from_sketch.jsonl"
    ))
    self.assertLen(dicts, _EXPECTED_NUM_DOCS)
    for plain_dict in random.sample(dicts, _NUM_DOCS_TO_SAMPLE):
      d = lib.json_to_sketch(
          self.config, plain_dict, self.stroke_stats, self.stroke_tokenizer,
          self.glyph_vocab, self.pronunciation_lexicon, FLAGS.test_tmpdir
      )
      self._check_common_features(d)
      self._check_glyphs(d)

  @flagsaver.flagsaver
  def test_combined_glyphs_and_sketch_tokens(self):
    FLAGS.recognizer_json = False
    FLAGS.combined_glyphs_and_strokes = True

    dicts = json_utils.load_jsonl(os.path.join(
        FLAGS.test_srcdir, _DATASET_DIR,
        "infer_eval_combined_strokes_and_glyphs.jsonl"
    ))
    self.assertLen(dicts, _EXPECTED_NUM_DOCS)
    for plain_dict in random.sample(dicts, _NUM_DOCS_TO_SAMPLE):
      d = lib.json_to_sketch(
          self.config, plain_dict, self.stroke_stats, self.stroke_tokenizer,
          self.glyph_vocab, self.pronunciation_lexicon, FLAGS.test_tmpdir
      )
      self._check_common_features(d)
      self._check_glyphs(d)

      # TODO: Since strokes and glyphs are in the same sequence, we
      # don't distinguish the scores by the sequence length yet.
      self.assertEqual(d["sketch.confidence"], d["glyph.confidence"])


if __name__ == "__main__":
  absltest.main()
