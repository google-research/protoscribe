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

"""Test for `protoscribe.language.phonology.sampa`."""

import os

from absl import flags
from absl.testing import absltest
from protoscribe.language.phonology import sampa

FLAGS = flags.FLAGS


class IpaToSampaTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    sampa_path = os.path.join(FLAGS.test_srcdir, sampa.SAMPA_PATH)
    phoible_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/data/phonology/phoible-phonemes.tsv"
    )
    phoible_features_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/data/phonology/",
        "phoible-segments-features.tsv"
    )
    cls.ipa_to_sampa = sampa.IpaToSampaConverter(
        sampa_path=sampa_path,
        phoible_path=phoible_path,
        phoible_features_path=phoible_features_path,
    )

  def testDi(self) -> None:
    self.assertEqual(self.ipa_to_sampa.convert("ɡ a l o ŋ"), 'g a" . l o% N')
    self.assertEqual(self.ipa_to_sampa.convert("ɡ ə l o ŋ"), 'g @- . l o" N')
    self.assertEqual(
        self.ipa_to_sampa.convert("ɡ a m l o ŋ"), 'g a" m . l o% N'
    )
    self.assertEqual(self.ipa_to_sampa.convert("ɣ e ŋ"), 'G e" N')
    # This one uses normal ASCII "g" rather than the IPA "ɡ", but should still
    # work:
    self.assertEqual(
        self.ipa_to_sampa.convert("g a m l o ŋ"), 'g a" m . l o% N'
    )

  def testVowel(self) -> None:
    self.assertTrue(sampa.is_vowel("o\""))
    self.assertTrue(sampa.is_vowel("o%"))
    self.assertTrue(sampa.is_vowel("o-"))
    self.assertFalse(sampa.is_vowel("g"))


if __name__ == "__main__":
  absltest.main()
