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

"""Simple tests for glyph vocabulary."""

import os

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from protoscribe.glyphs import glyph_vocab as lib

FLAGS = flags.FLAGS

_TEST_DATA_DIR = "protoscribe/glyphs/testdata"


def _vocab_path() -> str:
  return os.path.join(FLAGS.test_srcdir, _TEST_DATA_DIR, "glyph_vocab.json")


class GlyphVocabTest(absltest.TestCase):

  @flagsaver.flagsaver
  def testBuildOrLoadGlyphVocab(self) -> None:
    FLAGS.glyph_vocab_file = _vocab_path()
    vocab = lib.load_or_build_glyph_vocab()
    self.assertLess(0, len(vocab))

  @flagsaver.flagsaver
  def testFindAllSVGFiles(self) -> None:
    svg_paths = lib.find_all_svg_files()
    self.assertLess(0, len(svg_paths))

    # Point at more glyphs.
    FLAGS.extension_glyphs_svg_dir = os.path.join(
        FLAGS.test_srcdir, _TEST_DATA_DIR
    )
    new_svg_paths = lib.find_all_svg_files()
    self.assertGreater(len(new_svg_paths), len(svg_paths))

  def testLoadOnly(self) -> None:
    vocab = lib.load_glyph_vocab(_vocab_path())
    self.assertLess(0, len(vocab))
    self.assertEqual(lib.GLYPH_PAD, vocab.name_to_id("<pad>"))
    self.assertEqual("<pad>", vocab.id_to_name(lib.GLYPH_PAD))

  def testGlyphVocabPath2Key(self) -> None:
    glyph_key = "administrative_categories_svgs/mammal"
    self.assertEqual(glyph_key, lib.vocab_svg_path2key(
        "1/2/3/4/administrative_categories_svgs/mammal.svg",
        strip_category=False))
    self.assertEqual("mammal", lib.vocab_svg_path2key(
        "1/2/3/4/administrative_categories_svgs/mammal.svg",
        strip_category=True))
    self.assertEqual(glyph_key, lib.vocab_svg_path2key(
        "1/2/3/4/administrative_categories_svgs/mammal_3.svg",
        strip_category=False))
    self.assertEqual("mammal", lib.vocab_svg_path2key(
        "1/2/3/4/administrative_categories_svgs/mammal_3.svg",
        strip_category=True))

  @flagsaver.flagsaver
  def testGlyphTokenizer(self) -> None:
    FLAGS.glyph_vocab_file = _vocab_path()
    glyph_vocab = lib.load_or_build_glyph_vocab()
    names = ["X", "X", "I", "I", "clay", "brick"]
    tokens, types_mask = glyph_vocab.tokenize(
        names, [True, True, True, True, False, False]
    )
    self.assertLen(tokens, len(names) + 2)  # + EOS and BOS.
    self.assertLen(tokens, len(types_mask))
    self.assertListEqual(
        tokens,
        [
            1, 313, 313, 309, 309, 60, 33, 2,
        ]
    )
    self.assertListEqual(
        types_mask,
        [
            0, 1, 1, 1, 1, 2, 2, 0,
        ]
    )
    detok_names = glyph_vocab.detokenize([1, 313, 313, 309, 309, 60, 33, 2])
    self.assertListEqual(
        detok_names,
        ["<s>"] + names + ["</s>"]
    )

  @flagsaver.flagsaver
  def testFindMultipleVariantSVGs(self) -> None:
    FLAGS.glyph_vocab_file = _vocab_path()
    vocab = lib.load_or_build_glyph_vocab()

    def sample_to_set(name: str) -> set[str]:
      num_trials = 10
      svgs = set()
      for _ in range(num_trials):
        svg = vocab.find_svg_from_name(name)
        self.assertIsNot(svg, None)
        svgs.add(svg[0])  # Path.
      return svgs

    FLAGS.include_glyph_variants = False
    svgs = sample_to_set("apple")
    self.assertLen(svgs, 1)

    FLAGS.include_glyph_variants = True
    svgs = sample_to_set("apple")
    self.assertGreater(len(svgs), 1)

  def testGlyphIsSingleNumeral(self):
    self.assertFalse(lib.glyph_is_single_numeral("cat"))
    self.assertFalse(lib.glyph_is_single_numeral("Cat"))
    self.assertTrue(lib.glyph_is_single_numeral("C"))
    self.assertFalse(lib.glyph_is_single_numeral("CI"))

  def testSpecialGlyphs(self):
    names = lib.special_glyph_names()
    self.assertIn("X", names)
    self.assertIn("<pad>", names)
    self.assertIn("DUMMY", names)
    vocab = lib.load_glyph_vocab(_vocab_path())
    ids = vocab.special_glyph_ids()
    self.assertLen(ids, len(names))
    self.assertIn(lib.GLYPH_PAD, ids)


if __name__ == "__main__":
  absltest.main()
