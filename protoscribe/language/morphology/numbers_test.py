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

"""Test for `protoscribe.language.numbers`."""

import os
import random

from absl import flags
from absl import logging
from absl.testing import absltest
from protoscribe.language.morphology import morphemes
from protoscribe.language.morphology import numbers
from protoscribe.language.phonology import sampa

import pynini
from pynini.lib import byte

flags.DEFINE_integer("num_test_iter", 10, "Number of iterations of tests")

FLAGS = flags.FLAGS


class NumbersTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.phoible_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/data/phonology/phoible-phonemes.tsv",
    )
    self.features_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/data/phonology/phoible-segments-features.tsv",
    )
    self.syllable_templates_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/data/phonology/",
        "wiktionary_syllable_stats.tsv",
    )
    self.sesqui_template_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/language/phonology/",
        "testdata/sesqui.textproto",
    )
    self.power_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "[E1]",
                       "[E2]", "[E3]"]
    self.crib_lexicon = morphemes.Lexicon()
    self.crib_lexicon.add_morpheme(morphemes.Morpheme("1", "u n o"))
    self.crib_lexicon.add_morpheme(morphemes.Morpheme("2", "d o s"))
    self.crib_lexicon.add_morpheme(morphemes.Morpheme("3", "t r e s"))
    self.crib_lexicon.add_morpheme(morphemes.Morpheme("4", "k w a t r o"))
    self.crib_lexicon.add_morpheme(morphemes.Morpheme("5", "s i n k o"))
    self.crib_lexicon.add_morpheme(morphemes.Morpheme("6", "s e i s"))
    self.crib_lexicon.add_morpheme(morphemes.Morpheme("7", "s i e t e"))
    self.crib_lexicon.add_morpheme(morphemes.Morpheme("8", "o č o"))
    self.crib_lexicon.add_morpheme(morphemes.Morpheme("9", "n w e b e"))
    self.crib_lexicon.add_morpheme(morphemes.Morpheme("[E1]", "d i e s"))
    self.crib_lexicon.add_morpheme(morphemes.Morpheme("[E2]", "s i e n"))
    self.crib_lexicon.add_morpheme(morphemes.Morpheme("[E3]", "m i l"))
    sigstar = pynini.closure(byte.BYTE)
    nasal = pynini.accep("m") | "n" | "ŋ"
    nasalization = (
        pynini.cross("p", "m")
        | pynini.cross("t", "n")
        | pynini.cross("k", "ŋ")
        | pynini.cross("b", "m")
        | pynini.cross("d", "n")
        | pynini.cross("ɡ", "ŋ")
    )
    wb_suffix = f" {sampa.WORD_BOUNDARY} "
    self.rules = [
        pynini.cdrewrite(nasalization, nasal + wb_suffix, "", sigstar)
    ]

  def number_tester(self, one_deletion: bool = False) -> None:
    lexicon = morphemes.make_lexicon(
        self.sesqui_template_path,
        self.power_list,
        phoible_path=self.phoible_path,
        phoible_features_path=self.features_path,
        phoible_syllable_templates=self.syllable_templates_path,
        crib_lexicon=self.crib_lexicon,
    )
    number_grammar = numbers.Numbers(lexicon, self.rules, one_deletion)
    for power, name in number_grammar.powers:
      logging.info("%s -> %s", power, name)
    for _ in range(FLAGS.num_test_iter):
      number = random.randint(1, 9999)
      name = number_grammar.verbalize(number)
      logging.info("%s -> %s", number, name)
    if one_deletion:
      self.assertLen(
          number_grammar.verbalize(1111).split(sampa.WORD_BOUNDARY), 4
      )
    else:
      self.assertLen(
          number_grammar.verbalize(1111).split(sampa.WORD_BOUNDARY), 7
      )

  def testNumbers(self) -> None:
    for i in range(FLAGS.num_test_iter):
      logging.info("Iteration %s", i)
      self.number_tester()

  def testNumbersOneDeletion(self) -> None:
    for i in range(FLAGS.num_test_iter):
      logging.info("Iteration %s", i)
      self.number_tester(one_deletion=True)


if __name__ == "__main__":
  absltest.main()
