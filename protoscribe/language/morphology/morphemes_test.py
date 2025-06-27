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

"""Test for morpheme manipulation."""

import os

from absl import flags
from absl.testing import absltest
from protoscribe.language.morphology import morphemes as mor

_SAVED_LEXICON = flags.DEFINE_string(
    "saved_lexicon",
    "protoscribe/language/morphology/testdata/"
    "saved_lexicon.tsv",
    "Saved lexicon.",
)
_GENERATE_TESTDATA = flags.DEFINE_bool(
    "generate_testdata", False, "Generate the test data."
)

FLAGS = flags.FLAGS


class MorphemesTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.phoible_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/data/phonology/phoible-phonemes.tsv",
    )
    cls.features_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/data/phonology/",
        "phoible-segments-features.tsv",
    )
    cls.syllable_templates_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/data/phonology/",
        "wiktionary_syllable_stats.tsv",
    )
    cls.sesqui_template_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/language/phonology/",
        "testdata/sesqui.textproto",
    )
    cls.saved_lexicon = os.path.join(FLAGS.test_srcdir, _SAVED_LEXICON.value)

    cls.meanings = ["@HORSE", "@GOAT", "@CHEESE", "@WINE"]
    cls.crib_lexicon = mor.Lexicon()
    cls.crib_lexicon.add_morpheme(mor.Morpheme("@HORSE", "u m a"))
    cls.crib_lexicon.add_morpheme(mor.Morpheme("@GOAT", "y a g i"))
    cls.crib_lexicon.add_morpheme(mor.Morpheme("@WINE", "s a k e"))
    cls.crib_lexicon.add_morpheme(mor.Morpheme("@BRIDGE", "h a s i"))

  def testSimpleLexicon(self) -> None:
    lexicon = mor.make_lexicon(
        self.sesqui_template_path,
        self.meanings,
        phoible_path=self.phoible_path,
        phoible_features_path=self.features_path,
        phoible_syllable_templates=self.syllable_templates_path,
    )
    self.assertIn("@GOAT", lexicon.meanings_to_morphemes)

  def testCribLexicon(self) -> None:
    lexicon = mor.make_lexicon(
        self.sesqui_template_path,
        self.meanings,
        phoible_path=self.phoible_path,
        phoible_features_path=self.features_path,
        phoible_syllable_templates=self.syllable_templates_path,
        crib_lexicon=self.crib_lexicon,
    )
    self.assertGreaterEqual(
        lexicon.meanings_to_morphemes["@GOAT"].phonological_weight(),
        lexicon.meanings_to_morphemes["@HORSE"].phonological_weight())

  def testSavedLexicon(self) -> None:
    if _GENERATE_TESTDATA.value:
      lexicon = mor.make_lexicon(
          self.sesqui_template_path,
          self.meanings,
          phoible_path=self.phoible_path,
          phoible_features_path=self.features_path,
          phoible_syllable_templates=self.syllable_templates_path,
      )
      lexicon.dump_lexicon(self.saved_lexicon)
    else:
      lexicon = mor.Lexicon()
      lexicon.load_lexicon(self.saved_lexicon)
      self.assertEqual(lexicon.meanings_to_morphemes["@HORSE"].form, "p ə s o")
      self.assertEqual(lexicon.meanings_to_morphemes["@GOAT"].form, "b ə k a w")


if __name__ == "__main__":
  absltest.main()
