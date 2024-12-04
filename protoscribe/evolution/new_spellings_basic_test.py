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

"""Unit test for basic algorithm for spelling extension."""

import os

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from protoscribe.evolution import confidence_pruning
from protoscribe.evolution import new_spellings_basic as lib

FLAGS = flags.FLAGS

_TEST_DATA_DIR = "protoscribe/evolution/testdata"

# Total number of extensions corresponds to the total number of
# non-administrative categories.
_NUM_EXTENSIONS = 468


class NewSpellingsBasicTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    FLAGS.semantic_jsonl_file = os.path.join(
        FLAGS.test_srcdir, _TEST_DATA_DIR, "semantic_results.jsonl"
    )
    FLAGS.phonetic_jsonl_file = os.path.join(
        FLAGS.test_srcdir, _TEST_DATA_DIR, "phonetic_results.jsonl"
    )

  @flagsaver.flagsaver
  def test_all_biglyph_extensions(self):
    """Extends all spellings on the basis of biglyphs only."""

    # Setting the pruning method to `none` means that nothing gets pruned
    # and phonetic and semantics `streams` will both propose all the
    # available extensions.
    FLAGS.pruning_method = confidence_pruning.Method.NONE
    extensions = lib.glyph_extensions()
    self.assertLen(extensions, _NUM_EXTENSIONS)
    for concept, glyphs, concept_pron, glyph_pron, _ in extensions:
      self.assertIsNotNone(concept)
      self.assertIsNotNone(concept_pron)
      self.assertIsNotNone(glyph_pron)
      # Make sure that we have a bi-glyph.
      self.assertIsNotNone(glyphs)
      self.assertLen(glyphs.split(), 2)

  @flagsaver.flagsaver
  def test_no_biglyph_extensions(self):
    """Extends all spellings either pure semantically or phonetically."""

    # Select extension with the best confidence from phonetic and semantic
    # streams.
    FLAGS.pruning_method = confidence_pruning.Method.TOP_K
    FLAGS.phonetic_top_k = 1
    FLAGS.semantic_top_k = 1

    # We are expecting exactly two extensions. One glyph extended by phonetics
    # and one by semantics.
    extensions = lib.glyph_extensions()
    self.assertLen(extensions, 2)

    # Pure semantic extension.
    concept, glyphs, concept_pron, glyph_pron, _ = extensions[0]
    self.assertEqual(concept, "little_ADJ")
    self.assertIsNotNone(concept_pron)
    self.assertIsNotNone(glyph_pron)
    self.assertEqual(glyph_pron, "NA")  # Semantic extension.
    self.assertIsNotNone(glyphs)
    self.assertLen(glyphs.split(), 1)

    # Phonetic extension.
    concept, glyphs, concept_pron, glyph_pron, _ = extensions[1]
    self.assertEqual(concept, "dwarf_NOUN")
    self.assertIsNotNone(concept_pron)
    self.assertIsNotNone(glyph_pron)
    self.assertGreater(len(glyph_pron.split()), 1)  # Phonetic extension.
    self.assertIsNotNone(glyphs)
    self.assertLen(glyphs.split(), 1)


if __name__ == "__main__":
  absltest.main()
