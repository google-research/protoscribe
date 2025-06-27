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

"""Test for discrete glyph prediction parser."""

import os
from typing import Any

from absl import flags
from absl.testing import absltest
from protoscribe.corpus.reader import dataset_defaults as ds_lib
from protoscribe.sketches.inference import glyphs_from_jsonl as lib
from protoscribe.sketches.inference import json_utils
from protoscribe.texts import generate as generate_lib

FLAGS = flags.FLAGS

_EXPECTED_NUM_DOCS = 100
_DATASET_DIR = (
    "protoscribe/sketches/inference/testdata"
)


class GlyphsFromJsonlTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    FLAGS.dataset_dir = os.path.join(FLAGS.test_srcdir, _DATASET_DIR)
    self.glyph_vocab = ds_lib.get_glyph_vocab()
    self.pronunciation_lexicon, _ = generate_lib.load_phonetic_forms(
        main_lexicon_file=ds_lib.main_lexicon_file(),
        number_lexicon_file=ds_lib.number_lexicon_file()
    )

  def _check_glyphs(self, d: dict[str, Any]) -> None:
    """Checks glyph information."""

    # Check input features copied to the output.
    self.assertIn("doc.id", d)
    self.assertIsInstance(d["doc.id"], int)
    for feature in [
        "concept.name", "number.name",
        "text.words", "text.inputs", "concept.pron"
    ]:
      self.assertIn(feature, d)
      self.assertNotEmpty(d[feature])

    # Checks predictions.
    for feature in [
        "glyph.names", "glyph.prons"
    ]:
      self.assertIn(feature, d)
      self.assertNotEmpty(d[feature])

    for feature in ["glyph.names", "glyph.prons"]:
      for hyp in d[feature]:
        self.assertIsInstance(hyp, list)
        self.assertNotEmpty(hyp)

    self.assertIn("glyph.confidence", d)
    self.assertGreater(d["glyph.confidence"], 0.)

  def test_basic_parse(self):
    dicts = json_utils.load_jsonl(os.path.join(
        FLAGS.test_srcdir, _DATASET_DIR, "infer_eval_discrete_glyphs.jsonl"
    ))
    self.assertLen(dicts, _EXPECTED_NUM_DOCS)
    for plain_dict in dicts:
      inputs, glyph_outputs, d, has_error = lib.json_to_glyphs(
          self.glyph_vocab, plain_dict, self.pronunciation_lexicon
      )
      self.assertNotEmpty(inputs)
      self.assertNotEmpty(glyph_outputs)
      self.assertFalse(has_error)
      self._check_glyphs(d)


if __name__ == "__main__":
  absltest.main()
