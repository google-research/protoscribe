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

"""Test for `protoscribe.language.phonology.phoible_templates`."""

import os

from absl import flags
from absl.testing import absltest
from protoscribe.language.phonology import phoible_templates

FLAGS = flags.FLAGS


class PhoibleTemplatesTest(absltest.TestCase):

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
        "phoible-segments-features.tsv",
    )
    syllable_templates_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/data/phonology/",
        "wiktionary_syllable_stats.tsv",
    )
    cls.phoible_templates = phoible_templates.PhoibleTemplates(
        path=phoible_path,
        features_path=features_path,
        syllable_templates=syllable_templates_path,
    )
    cls._mono_template_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/language/phonology/",
        "testdata/monosyllabic.textproto",
    )
    cls._sesqui_template_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/language/phonology/",
        "testdata/sesqui.textproto",
    )
    cls._di_template_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/language/phonology/",
        "testdata/disyllabic.textproto",
    )
    cls._di_template_path_lan = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/language/phonology/",
        "testdata/disyllabic_lan.textproto",
    )

  def testPhoibleTemplates(self) -> None:
    phonemes = self.phoible_templates.choose_phonemes(14)
    templates = self.phoible_templates.populate_syllables(
        phonemes, ["CV", "CVC", "VC", "V"], 2
    )
    cvc = list(("CVC" @ templates).paths().ostrings())
    self.assertIn("p a n", cvc)
    cv_cvc = list(("CV.CVC" @ templates).paths().ostrings())
    self.assertIn("p a n i t", cv_cvc)
    cvc_vc = list(("CVC.VC" @ templates).paths().ostrings())
    self.assertEmpty(cvc_vc)

  def testPhoibleTemplatesWithSesquiSyllable(self) -> None:
    phonemes = self.phoible_templates.choose_phonemes(14)
    templates = self.phoible_templates.sesquisyllabify(
        phonemes,
        self.phoible_templates.populate_syllables(phonemes, ["CV", "CVC"], 1),
    ).optimize()
    cx_cvc = list(("Cx.CVC" @ templates).paths().ostrings())
    self.assertIn("t ə b i m", cx_cvc)

  def testTemplateLoadingSesquiSyllabic(self) -> None:
    templates = self.phoible_templates.template_spec_to_fst(
        self._sesqui_template_path
    )
    cx_cvc = list(("Cx.CVC" @ templates).paths().ostrings())
    self.assertIn("t ə b i m", cx_cvc)

  def testTemplateLoadingDisyllabic(self) -> None:
    templates = self.phoible_templates.template_spec_to_fst(
        self._di_template_path
    )
    cx_cvc = list(("Cx.CVC" @ templates).paths().ostrings())
    self.assertNotIn("t ə b i m", cx_cvc)
    cx_cvc = list(("CV.CVC" @ templates).paths().ostrings())
    self.assertIn("t i b i m", cx_cvc)

  def testTemplateLoadingDisyllabicLanguage(self) -> None:
    templates = self.phoible_templates.template_spec_to_fst(
        self._di_template_path_lan
    )
    cx_cvc = list(("CV.CVC" @ templates).paths().ostrings())
    self.assertIn("h u l u p", cx_cvc)
    self.assertIn("n a p o l", cx_cvc)
    self.assertIn("h o s ɛ h", cx_cvc)
    self.assertIn("m a s a n", cx_cvc)
    self.assertNotIn("h u l u b", cx_cvc)

  def testTemplateLoadingMonosyllabicOverrideLanguage(self) -> None:
    templates = self.phoible_templates.template_spec_to_fst(
        self._mono_template_path,
        phonology_languages=["haw"]  # Hawaiian.
    )
    cvc = list(("CVC" @ templates).paths().ostrings())
    self.assertIn("m u p", cvc)
    self.assertIn("s i n", cvc)
    self.assertIn("h a k", cvc)

  def testTemplateLoadingSesquiSyllabicNumPhonemes(self) -> None:
    templates = self.phoible_templates.template_spec_to_fst(
        self._sesqui_template_path, number_of_phonemes=50
    )
    cx_cvc = list(("Cx.CVC" @ templates).paths().ostrings())
    # Rather weird phoneme is not among the 50 most common:
    self.assertNotIn("b ə b ẽ mb", cx_cvc)

    templates = self.phoible_templates.template_spec_to_fst(
        self._sesqui_template_path, number_of_phonemes=100
    )
    cx_cvc = list(("Cx.CVC" @ templates).paths().ostrings())
    # But if you increase to 100...
    self.assertIn("b ə b ẽ mb", cx_cvc)

  def testRandgen(self) -> None:
    templates = self.phoible_templates.template_spec_to_fst(
        self._sesqui_template_path
    )
    self.assertLen(self.phoible_templates.randgen(templates, n=10), 10)

  def testMaxSingleSegment(self) -> None:
    templates = self.phoible_templates.template_spec_to_fst(
        self._mono_template_path
    )
    forms = self.phoible_templates.randgen(
        templates,
        n=10_000,
        max_homophony=5,
        max_number_of_single_segment_forms=10,
    )
    n_single_segment = 0
    for form in forms:
      if self.phoible_templates.single_segment_form(form):
        n_single_segment += 1
    self.assertLessEqual(n_single_segment, 10)


if __name__ == "__main__":
  absltest.main()
