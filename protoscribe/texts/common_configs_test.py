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

import os

from absl import flags
from absl import logging
from absl.testing import absltest
from protoscribe.language.phonology import phoible_templates as pt
from protoscribe.texts import common_configs

from pynini.lib import features

FLAGS = flags.FLAGS

# pytype: disable=attribute-error


class TemplaticRulesTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    phoible_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/data/phonology/phoible-phonemes.tsv"
    )
    features_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/data/phonology/",
        "phoible-segments-features.tsv"
    )
    syllable_templates_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/data/phonology/",
        "wiktionary_syllable_stats.tsv",
    )
    templatic_template_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/language/morphology/",
        "testdata/templatic.textproto",
    )
    phoible_templates = pt.PhoibleTemplates(
        phoible_path, features_path, syllable_templates_path
    )
    num_feats = ["sg", "du", "pl"]
    cas_feats = ["nom", "obl"]
    cls.cat = features.Category(
        features.Feature("num", *num_feats),
        features.Feature("cas", *cas_feats),
    )
    cls.templatic_rules = common_configs.TemplaticRules(
        phoible_templates, templatic_template_path, cls.cat
    )

  def make_input(self, string, feats):
    return (
        string
        + common_configs.BOUND
        + features.FeatureVector(self.cat, *feats).acceptor
    ).optimize()

  def printable(self, fst):
    return list((fst @ self.templatic_rules.label_rewriter).paths().ostrings())

  EXAMPLES = (
      (
          "d a n u p",
          (
              "d a n a p+[cas=nom][num=sg]",
              "d n u p+[cas=nom][num=du]",
              "d i n p+[cas=nom][num=pl]",
          ),
      ),
      (
          "d a n u",
          (
              "d a n a+[cas=nom][num=sg]",
              "d n u+[cas=nom][num=du]",
              "d i n+[cas=nom][num=pl]",
          ),
      ),
      (
          "d a n j u",
          (
              "d a n j a+[cas=nom][num=sg]",
              "d n j u+[cas=nom][num=du]",
              "d i n j+[cas=nom][num=pl]",
          ),
      ),
      (
          "d a n j u p",
          (
              "d a n j a p+[cas=nom][num=sg]",
              "d n j u p+[cas=nom][num=du]",
              "d i n j p+[cas=nom][num=pl]",
          ),
      ),
      (
          "m e s u",
          (
              "m a s a+[cas=nom][num=sg]",
              "m s u+[cas=nom][num=du]",
              "m i s+[cas=nom][num=pl]",
          ),
      ),
  )

  def _rule_tester(self, inp_string, outputs) -> None:
    i = 0
    for template, feat in common_configs.NOUN_TEMPLATE_RULE_EXAMPLES:
      inp_feat = ["cas=nom"] + feat
      inp = self.make_input(inp_string, inp_feat)
      rule = self.templatic_rules.compile_rule(template, feat)
      out = self.printable(inp @ rule)[0]
      logging.info("%s+[%s] --> %s", inp_string, "][".join(inp_feat), out)
      self.assertEqual(out, outputs[i])
      i += 1

  def testRules(self) -> None:
    for inp_string, outputs in self.EXAMPLES:
      self._rule_tester(inp_string, outputs)


class CommonConfigsTest(absltest.TestCase):

  def test_parse_number_config(self) -> None:
    path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/texts/configs",
        "number_config_sg.textproto"
    )
    params = common_configs.parse_number_config_file(path)
    self.assertNotEmpty(params)
    for key in ["NUM_FEATS", "ZERO_FEAT_VALS", "ONE_DELETION", "ONE_CONFIG"]:
      self.assertIn(key, params)
    self.assertNotIn("CAS_FEATS", params)
    self.assertNotIn("GEN_FEATS", params)

    self.assertLen(params["NUM_FEATS"], 1)
    self.assertEqual(params["NUM_FEATS"][0], "sg")
    self.assertLen(params["ZERO_FEAT_VALS"], 1)
    self.assertEqual(params["ZERO_FEAT_VALS"][0], ("num", "sg"))
    self.assertTrue(params["ONE_DELETION"])
    self.assertLen(params["ONE_CONFIG"], 1)
    self.assertEqual(params["ONE_CONFIG"][0], "num=sg")


if __name__ == "__main__":
  absltest.main()
