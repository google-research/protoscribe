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

"""Test the morphological APIs, such as lexicon generation."""

import logging
import os
import re

from absl import flags
from absl import logging
from absl.testing import absltest
from protoscribe.language.morphology import morphemes
from protoscribe.language.morphology import morphology
from protoscribe.language.phonology import phoible_templates as pt

import pynini as py
from pynini.lib import features

_TESTDATA_DIR = "protoscribe/language/morphology/testdata"

_NUM_TEST_ITER = flags.DEFINE_integer(
    "num_test_iter", 10,
    "Number of iterations of tests."
)

_MORPHOLOGY_PARAMETERS = flags.DEFINE_string(
    "morphology_parameters",
    f"{_TESTDATA_DIR}/morphology_parameters.textproto",
    "Saved morphology parameters.",
)

_RULES_FAR = flags.DEFINE_string(
    "rules_far",
    f"{_TESTDATA_DIR}/rules.far",
    "FAR of saved rules.",
)

_SAVED_LEXICON = flags.DEFINE_string(
    "saved_lexicon",
    f"{_TESTDATA_DIR}/saved_lexicon.tsv",
    "Saved lexicon. This can be regenerated by morphemes_test.",
)

_SAVED_AFFIXES = flags.DEFINE_string(
    "saved_affixes",
    f"{_TESTDATA_DIR}/saved_affixes.tsv",
    "Saved affixes.",
)

_GENERATE_TESTDATA = flags.DEFINE_bool(
    "generate_testdata", False,
    "Generate the test parameter files."
)

FLAGS = flags.FLAGS


class MorphologyTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.phoible_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/data/phonology/phoible-phonemes.tsv"
    )
    self.features_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/data/phonology/",
        "phoible-segments-features.tsv"
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
    self.templatic_template_path = os.path.join(
        FLAGS.test_srcdir,
        f"{_TESTDATA_DIR}/templatic.textproto",
    )
    self.saved_lexicon = os.path.join(
        FLAGS.test_srcdir, _SAVED_LEXICON.value
    )
    self.saved_affixes = os.path.join(
        FLAGS.test_srcdir, _SAVED_AFFIXES.value
    )
    self.morphology_parameters = os.path.join(
        FLAGS.test_srcdir, _MORPHOLOGY_PARAMETERS.value
    )
    self.rules_far = os.path.join(
        FLAGS.test_srcdir, _RULES_FAR.value
    )
    self.meanings = ["@HORSE", "@GOAT", "@CHEESE", "@WINE"]
    self.crib_lexicon = morphemes.Lexicon()
    self.crib_lexicon.add_morpheme(morphemes.Morpheme("@HORSE", "u m a"))
    self.crib_lexicon.add_morpheme(morphemes.Morpheme("@GOAT", "y a g i"))
    self.crib_lexicon.add_morpheme(morphemes.Morpheme("@WINE", "s a k e"))
    self.crib_lexicon.add_morpheme(morphemes.Morpheme("@BRIDGE", "h a s i"))
    self.num_feats = ["sg", "du", "pl"]
    self.cas_feats = ["nom", "obl"]
    self.featval = re.compile(r"\[[a-z]*=[a-z]*\]")
    # TODO. Somewhere we need to make the writing of rules a little more
    # user-friendly. Any sequence of phonemes must be separated by spaces, so
    # that has to be reflected in the rule. Morphological boundaries are assumed
    # to be placed thus:
    #
    # s t e m+ s u f+ su f
    #
    # i.e. with a space after the boundary but not before, so this also needs to
    # be reflected in the rule.
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
    self.affix_specification = ["VC"]
    #############################################################################
    # Templatic morphology rules for testTemplaticMorphology
    #
    # TODO: One weakness is that to build any feature sensitive rules, we
    # have to have the Category associated with the set of features, which is
    # created when the morphology is built. One can create it outside as
    # follows, but then one has to make sure one creates the same category.
    # Another weakness is that we need to know the phonemes so that we can
    # categorize at least into consonants and vowels. We can get that by
    # building a PhoibleTemplates instance, and then picking a superset of
    # phonemes that we are expecting to actually use.
    cat = features.Category(
        features.Feature("num", *self.num_feats),
        features.Feature("cas", *self.cas_feats),
    )
    self.templatic_phoible = pt.PhoibleTemplates(
        self.phoible_path, self.features_path, self.syllable_templates_path
    )
    phoneme_superset = self.templatic_phoible.phoible.top_k_segments(40)
    v = py.union(
        *self.templatic_phoible.phoible.matching_phonemes(
            ["+syllabic"], phoneme_superset
        )
    )
    c = py.union(
        *self.templatic_phoible.phoible.matching_phonemes(
            ["-syllabic"], phoneme_superset
        )
    )

    def template(v1: py.Fst, v2: py.Fst) -> py.Fst:
      return (
          c + " " + v1 + " " + c + " " + (c + " ").ques + v2 + (" " + c).ques
      ).optimize()

    css = cat.sigma_star

    def feature_acceptor(setting: list[str]) -> py.Fst:
      return css + features.FeatureVector(cat, *setting).acceptor

    #
    singular = template(py.cross(v, "a"), py.cross(v, "a"))
    self.singular_rule = singular, "", feature_acceptor(["num=sg"])
    #
    dual = template(py.cross(v, ""), py.cross(v, "u"))
    self.dual_rule = dual, "", feature_acceptor(["num=du"])
    #
    plural = template(py.cross(v, "i"), py.cross(v, ""))
    self.plural_rule = plural, "", feature_acceptor(["num=pl"])

  def noun_morphology_tester(self) -> None:
    lexicon = morphemes.make_lexicon(
        self.sesqui_template_path,
        self.meanings,
        phoible_path=self.phoible_path,
        phoible_features_path=self.features_path,
        phoible_syllable_templates=self.syllable_templates_path,
        crib_lexicon=self.crib_lexicon,
    )
    nouns = morphology.BasicNounMorphologyBuilder(
        lexicon,
        self.affix_specification,
        num_feats=self.num_feats,
        cas_feats=self.cas_feats,
        zero_feat_vals=[("num", "sg"), ("cas", "nom")],
        rule_templates=[self.voicing_rule_template],
    )
    forms = nouns.forms("@HORSE")
    self.assertLen(forms, 6)
    for infl, form in nouns.forms("@HORSE"):
      rinfl = self.featval.sub("", infl).replace("+", "")
      self.assertEqual(rinfl, form)
      logging.info("%s --> %s", infl, form)
    # Tests the lemma FeatureVector
    infl, form = nouns.inflected_form("@HORSE", nouns.lemma_vector)
    # ++ because we have defined nom and sg as zero morphemes
    self.assertEndsWith(infl, "++[cas=nom][num=sg]")
    self.assertEqual(form, lexicon.meanings_to_morphemes["@HORSE"].form)
    logging.info("@HORSE[cas=nom][num=sg] --> %s", form)
    # Tests another FeatureVector
    infl, form = nouns.inflected_form(
        "@HORSE", features.FeatureVector(nouns.cat, "num=du", "cas=obl"))
    self.assertEndsWith(infl, "[cas=obl][num=du]")
    logging.info("@HORSE[cas=obl][num=du] --> %s", form)

  def testMorphology(self) -> None:
    for i in range(_NUM_TEST_ITER.value):
      logging.info("Iteration %s", i)
      self.noun_morphology_tester()

  def testSavedMorphology(self) -> None:
    lexicon = morphemes.Lexicon()
    lexicon.load_lexicon(self.saved_lexicon)
    if _GENERATE_TESTDATA.value:
      nouns = morphology.BasicNounMorphologyBuilder(
          lexicon,
          self.saved_affixes,
          num_feats=self.num_feats,
          cas_feats=self.cas_feats,
          zero_feat_vals=[("num", "sg"), ("cas", "nom")],
          rule_templates=[self.voicing_rule_template],
      )
      # Note that we also have a dump_affixes method but we don't use
      # it here since we want to use the already saved affixes.
      nouns.dump_parameters(_MORPHOLOGY_PARAMETERS.value, _RULES_FAR.value)
      return

    params = morphology.BasicNounMorphologyBuilder.load_parameters(
        os.path.join(FLAGS.test_srcdir, _MORPHOLOGY_PARAMETERS.value),
        os.path.join(FLAGS.test_srcdir, _RULES_FAR.value)
    )
    nouns = morphology.BasicNounMorphologyBuilder(
        lexicon,
        self.saved_affixes,
        num_feats=params["num_feats"],
        cas_feats=params["cas_feats"],
        gen_feats=params["gen_feats"],
        zero_feat_vals=params["zero_feat_vals"],
        rule_templates=params["rules"],
    )
    ####
    infl, form = nouns.inflected_form(
        "@HORSE", features.FeatureVector(nouns.cat, "num=du", "cas=nom")
    )
    logging.info("@HORSE[cas=nom][num=du] --> %s", form)
    logging.info("@HORSE[cas=nom][num=du] --> %s", infl)
    self.assertEqual(form, "p ə s o o t")
    ####
    infl, form = nouns.inflected_form(
        "@HORSE", features.FeatureVector(nouns.cat, "num=du", "cas=obl")
    )
    logging.info("@HORSE[cas=obl][num=du] --> %s", form)
    logging.info("@HORSE[cas=obl][num=du] --> %s", infl)
    self.assertEqual(form, "p ə s o o d e ŋ")
    ####
    infl, form = nouns.inflected_form(
        "@HORSE", features.FeatureVector(nouns.cat, "num=pl", "cas=nom")
    )
    logging.info("@HORSE[cas=nom][num=du] --> %s", form)
    logging.info("@HORSE[cas=nom][num=du] --> %s", infl)
    self.assertEqual(form, "p ə s o e p")
    infl, form = nouns.inflected_form(
        "@HORSE", features.FeatureVector(nouns.cat, "num=pl", "cas=obl")
    )
    ####
    logging.info("@HORSE[cas=obl][num=du] --> %s", form)
    logging.info("@HORSE[cas=obl][num=du] --> %s", infl)
    self.assertEqual(form, "p ə s o e b e ŋ")
    _, form = nouns.inflected_form(
        "@GOAT", features.FeatureVector(nouns.cat, "num=sg", "cas=nom"))
    self.assertEqual(form, "b ə k a w")
    _, form = nouns.inflected_form(
        "@HORSE", features.FeatureVector(nouns.cat, "num=sg", "cas=nom"))
    self.assertEqual(form, "p ə s o")
    self.assertIn("@HORSE", nouns.lookup_meanings("p ə s o o d e ŋ"))

  def templatic_noun_morphology_tester(self) -> None:
    lexicon = morphemes.make_lexicon(
        self.templatic_template_path,
        self.meanings,
        crib_lexicon=self.crib_lexicon,
        phoible=self.templatic_phoible,
    )
    nouns = morphology.BasicNounMorphologyBuilder(
        lexicon,
        self.affix_specification,
        num_feats=["sg", "du", "pl"],
        cas_feats=["nom", "obl"],
        zero_feat_vals=[("num", "sg"), ("cas", "nom")],
        rule_templates=[self.singular_rule, self.dual_rule, self.plural_rule],
    )
    forms = nouns.forms("@HORSE")
    self.assertLen(forms, 6)
    for infl, form in nouns.forms("@HORSE"):
      rinfl = self.featval.sub("", infl).replace("+", "")
      self.assertEqual(rinfl, form)
      logging.info("%s --> %s", infl, form)
      meanings = nouns.lookup_meanings(form)
      logging.info("Check meanings: %s -> %s", form, meanings)
      self.assertIn("@HORSE", meanings)
    # Tests the lemma FeatureVector
    infl, form = nouns.inflected_form("@HORSE", nouns.lemma_vector)
    # ++ because we have defined nom and sg as zero morphemes
    self.assertEndsWith(infl, "++[cas=nom][num=sg]")
    logging.info("@HORSE[cas=nom][num=sg] --> %s", form)
    # Tests another FeatureVector
    infl, form = nouns.inflected_form(
        "@HORSE", features.FeatureVector(nouns.cat, "num=du", "cas=obl")
    )
    self.assertEndsWith(infl, "[cas=obl][num=du]")
    logging.info("@HORSE[cas=obl][num=du] --> %s", form)

  def testTemplaticMorphology(self) -> None:
    for i in range(_NUM_TEST_ITER.value):
      logging.info("Iteration %s", i)
      self.templatic_noun_morphology_tester()


if __name__ == "__main__":
  absltest.main()