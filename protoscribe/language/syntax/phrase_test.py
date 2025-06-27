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

"""Test for `protoscribe.language.syntax.phrase`."""

import os
import random

from absl import flags
from absl import logging
from absl.testing import absltest
from protoscribe.language.morphology import morphemes
from protoscribe.language.morphology import morphology
from protoscribe.language.morphology import numbers
from protoscribe.language.syntax import phrase

import pynini as py
from pynini.lib import byte

flags.DEFINE_integer("num_test_iter", 10, "Number of iterations of tests")

FLAGS = flags.FLAGS


class MorphologyTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    phoible_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/data/phonology/phoible-phonemes.tsv"
    )
    features_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/data/phonology/",
        "phoible-segments-features.tsv",
    )
    syllable_templates_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/data/phonology",
        "wiktionary_syllable_stats.tsv",
    )
    sesqui_template_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/language/phonology/",
        "testdata/sesqui.textproto",
    )
    self.meanings = ["@HORSE", "@GOAT", "@CHEESE", "@WINE"]
    crib_lexicon = morphemes.Lexicon()
    crib_lexicon.add_morpheme(morphemes.Morpheme("@HORSE", "u m a"))
    crib_lexicon.add_morpheme(morphemes.Morpheme("@GOAT", "y a g i"))
    crib_lexicon.add_morpheme(morphemes.Morpheme("@WINE", "s a k e"))
    crib_lexicon.add_morpheme(morphemes.Morpheme("@BRIDGE", "h a s i"))
    # TODO. Somewhere we need to make the writing of rules a little more
    # user-friendly. Any sequence of phonemes must be separated by spaces, so
    # that has to be reflected in the rule. Morphological boundaries are assumed
    # to be placed thus:
    #
    # s t e m+ s u f+ su f
    #
    # i.e. with a space after the boundary but not before, so this also needs to
    # be reflected in the rule
    voice = (
        py.cross("p", "b")
        | py.cross("t", "d")
        | py.cross("k", "ɡ")
        | py.cross("f", "v")
        | py.cross("s", "z")
        | py.cross("x", "ɣ")
    )
    v = py.accep("a") | "e" | "i" | "o" | "u" | "ə"
    # Only voice at boundaries for the sake of testing ease.
    self.voicing_rule_template = voice, v + " ", py.closure("+", 1) + " " + v
    self.lexicon = morphemes.make_lexicon(
        sesqui_template_path,
        self.meanings,
        phoible_path=phoible_path,
        phoible_features_path=features_path,
        phoible_syllable_templates=syllable_templates_path,
        crib_lexicon=crib_lexicon,
    )
    self.affix_specification = ["VC"]
    crib_lexicon = morphemes.Lexicon()
    crib_lexicon.add_morpheme(morphemes.Morpheme("1", "u n o"))
    crib_lexicon.add_morpheme(morphemes.Morpheme("2", "d o s"))
    crib_lexicon.add_morpheme(morphemes.Morpheme("3", "t r e s"))
    crib_lexicon.add_morpheme(morphemes.Morpheme("4", "k w a t r o"))
    crib_lexicon.add_morpheme(morphemes.Morpheme("5", "s i n k o"))
    crib_lexicon.add_morpheme(morphemes.Morpheme("6", "s e i s"))
    crib_lexicon.add_morpheme(morphemes.Morpheme("7", "s i e t e"))
    crib_lexicon.add_morpheme(morphemes.Morpheme("8", "o č o"))
    crib_lexicon.add_morpheme(morphemes.Morpheme("9", "n w e b e"))
    crib_lexicon.add_morpheme(morphemes.Morpheme("[E1]", "d i e s"))
    crib_lexicon.add_morpheme(morphemes.Morpheme("[E2]", "s i e n"))
    crib_lexicon.add_morpheme(morphemes.Morpheme("[E3]", "m i l"))
    sigstar = py.closure(byte.BYTE)
    nasal = py.accep("m") | "n" | "ŋ"
    nasalization = (
        py.cross("p", "m")
        | py.cross("t", "n")
        | py.cross("k", "ŋ")
        | py.cross("b", "m")
        | py.cross("d", "n")
        | py.cross("ɡ", "ŋ")
    )
    self.number_rules = [py.cdrewrite(nasalization, nasal + " # ", "", sigstar)]
    power_list = (
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "[E1]",
        "[E2]",
        "[E3]",
    )
    self.number_lexicon = morphemes.make_lexicon(
        sesqui_template_path,
        power_list,
        phoible_path=phoible_path,
        phoible_features_path=features_path,
        phoible_syllable_templates=syllable_templates_path,
        crib_lexicon=crib_lexicon,
        max_homophony=1,
    )

  def counted_noun_tester(self, config, conc) -> None:
    nouns = morphology.BasicNounMorphologyBuilder(
        self.lexicon,
        self.affix_specification,
        num_feats=config["num_feats"],
        cas_feats=config["cas_feats"],
        zero_feat_vals=config["zero_feat_vals"],
        rule_templates=[self.voicing_rule_template],
    )
    number_grammar = numbers.Numbers(
        self.number_lexicon,
        self.number_rules,
        one_deletion=config["one_deletion"],
    )
    for number in list(range(1, 10)) + [10, 100, 1000]:
      logging.info("%d -> %s", number, number_grammar.verbalize(number))
    np = phrase.CountedNoun(
        nouns,
        number_grammar,
        one=config["one"],
        two=config["two"],
        many=config["many"])

    def phrase_splitter(phr):
      words = phrase.phrase_to_words(phr)
      number_name = " # ".join(words[:-1])
      noun = words[-1]
      return number_name, noun

    for phr in [
        f"1 {conc}",
        f"2 {conc}",
        f"12 {conc}",
        f"{random.randint(1,9999)} {conc}",
    ]:
      verb = np.verbalize(phr)
      logging.info("CONFIG: %s", config)
      logging.info("%s -> %s", phr, verb)
      number_name, noun = phrase_splitter(verb)
      self.assertIn(conc, nouns.lookup_meanings(noun))
      number_deverbalization, num_interpretations = number_grammar.deverbalize(
          number_name,
          return_num_interpretations=True,
      )
      if num_interpretations == 1:
        self.assertEqual(number_deverbalization, phr.split()[0])
      else:
        msg = f"{number_name} has {num_interpretations} interpretations"
        logging.info(msg)

  CONFIGS = [
      {
          "num_feats": ["sg", "du", "pl"],
          "cas_feats": ["nom", "obl"],
          "zero_feat_vals": [("num", "sg"), ("cas", "nom")],
          "one_deletion": False,
          "one": ["num=sg", "cas=nom"],
          "two": ["num=du", "cas=nom"],
          "many": ["num=pl", "cas=nom"]
      },
      {
          "num_feats": ["sg", "pl"],
          "cas_feats": None,
          "zero_feat_vals": [("num", "sg")],
          "one_deletion": True,
          "one": ["num=sg"],
          "two": [],
          "many": ["num=pl"]
      },
      # TODO: I need to fix this so that I can genuinely have no
      # features. This is a hack for now.
      {
          "num_feats": ["unmarked"],
          "cas_feats": None,
          "zero_feat_vals": [("num", "unmarked")],
          "one_deletion": True,
          "one": ["num=unmarked"],
          "two": None,
          "many": None
      },
  ]

  def testCountedNoun(self) -> None:
    for i in range(FLAGS.num_test_iter):
      logging.info("Iteration %s", i)
      for conc in self.meanings:
        for config in self.CONFIGS:
          self.counted_noun_tester(config, conc)


if __name__ == "__main__":
  absltest.main()
