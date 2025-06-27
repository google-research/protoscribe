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

"""Tests for phonological inventories and features derived from PHOIBLE."""

from absl import flags
from absl.testing import absltest
from protoscribe.language.phonology import phoible_segments

FLAGS = flags.FLAGS


class PhoibleSegmentsTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.phoible = phoible_segments.PhoibleSegments()

  def testPhoible(self) -> None:
    self.assertEqual(self.phoible.classes["t"], "consonant")
    self.assertEqual(self.phoible.classes["uɪ"], "vowel")
    self.assertEqual(self.phoible.classes["ɑː"], "vowel")
    self.assertEqual(self.phoible.classes["˥˩"], "tone")
    self.assertIn("eng", self.phoible.languages["æ"])
    self.assertIn("fra", self.phoible.languages["t"])
    self.assertEqual(self.phoible.stats[0][1], "m")

  def testFeatures(self) -> None:
    t_feats = self.phoible.features("t")
    self.assertIn("+consonantal", t_feats)
    self.assertIn("-continuant", t_feats)
    self.assertIn("-lateral", t_feats)
    self.assertEqual(self.phoible.sonority("t"), 0)
    s_feats = self.phoible.features("s")
    self.assertIn("+consonantal", s_feats)
    self.assertIn("+continuant", s_feats)
    self.assertIn("-lateral", s_feats)
    self.assertEqual(self.phoible.sonority("s"), 1)
    l_feats = self.phoible.features("l")
    self.assertIn("+consonantal", l_feats)
    self.assertIn("+continuant", l_feats)
    self.assertIn("+lateral", l_feats)
    self.assertEqual(self.phoible.sonority("l"), 2)
    a_feats = self.phoible.features("a")
    self.assertIn("-consonantal", a_feats)
    self.assertIn("+continuant", a_feats)
    self.assertIn("-lateral", a_feats)
    self.assertEqual(self.phoible.sonority("a"), 4)
    schwa_feats = self.phoible.features("ə")
    self.assertIn("-consonantal", schwa_feats)
    self.assertIn("+continuant", schwa_feats)
    self.assertIn("-lateral", schwa_feats)
    self.assertEqual(self.phoible.sonority("a"), 4)
    w_feats = self.phoible.features("w")
    self.assertIn("-consonantal", w_feats)
    self.assertIn("-syllabic", w_feats)
    self.assertIn("+continuant", w_feats)
    self.assertIn("-lateral", w_feats)
    self.assertEqual(self.phoible.sonority("w"), 3)
    syllabic_eng_feats = self.phoible.features("ŋ̩")
    self.assertIn("+syllabic", syllabic_eng_feats)
    self.assertEqual(self.phoible.sonority("DUMMY"), -1)

  def testSyllabification(self) -> None:
    self.assertEqual(
        self.phoible.syllabify("k a ʔ p a l c ɑ m l ɑː ŋ"),
        (("k", "a", "ʔ"), ("p", "a", "l"), ("c", "ɑ", "m"), ("l", "ɑː", "ŋ")),
    )
    self.assertEqual(
        self.phoible.syllabify("k a p a l c ɑ m p l ɑː ŋ"),
        (("k", "a"), ("p", "a", "l"), ("c", "ɑ", "m"), ("p", "l", "ɑː", "ŋ")),
    )
    self.assertEqual(self.phoible.syllabify("a p a"), (("a",), ("p", "a")))
    self.assertEqual(
        self.phoible.syllabify("ɐ b ɐ k u m ɨ t͡ɕ ə mʲ ɪ"),
        (("ɐ",), ("b", "ɐ"), ("k", "u"), ("m", "ɨ"), ("t͡ɕ", "ə"), ("mʲ", "ɪ")),
    )
    self.assertEqual(
        self.phoible.syllabify("v z ɡ lʲ æ dʲ e"),
        (("v", "z", "ɡ", "lʲ", "æ"), ("dʲ", "e")),
    )
    self.assertEqual(
        self.phoible.syllabify("a b a s u ʁ d i ʁ"),
        (("a",), ("b", "a"), ("s", "u", "ʁ"), ("d", "i", "ʁ")),
    )
    self.assertEqual(
        self.phoible.syllabify("e ɪ e ɪ k æ p"),
        (("e",), ("ɪ",), ("e",), ("ɪ",), ("k", "æ", "p")),
    )
    self.assertEqual(
        self.phoible.syllabify("eɪ eɪ k æ p"),
        (("eɪ",), ("eɪ",), ("k", "æ", "p")),
    )
    self.assertEqual(
        self.phoible.syllabify("ɑ ɹ d w ʊ l v z"),
        (("ɑ", "ɹ"), ("d", "w", "ʊ", "l", "v", "z")),
    )
    self.assertEqual(
        self.phoible.syllabify("ʁ ʊ ŋ k ŋ̩ s t n̩"),
        (("ʁ", "ʊ", "ŋ"), ("k", "ŋ̩", "s"), ("t", "n̩")),
    )
    self.assertEqual(
        self.phoible.syllabify("ʁ ʊ ŋ k ŋ̩ s t n̩", parse_onsets_and_rhymes=True),
        ((("ʁ",), ("ʊ", "ŋ")), (("k",), ("ŋ̩", "s")), (("t",), ("n̩",))),
    )
    self.assertEqual(
        self.phoible.syllabify("eɪ eɪ k æ p", parse_onsets_and_rhymes=True),
        (((), ("eɪ",)), ((), ("eɪ",)), (("k",), ("æ", "p"))),
    )
    self.assertEqual(
        self.phoible.template(self.phoible.syllabify("ɑ ɹ d w ʊ l v z")),
        "VC.CCVCCC",
    )
    self.assertEqual(
        self.phoible.sonority_profile(("d", "w", "ʊ", "l", "v", "z")),
        (0, 3, 4, 2, 1, 1),
    )

  def testTopK(self) -> None:
    top_k = self.phoible.top_k_segments(14)
    self.assertEqual(
        top_k,
        ["a", "b", "i", "j", "k", "l", "m", "n", "p", "s", "t", "u", "w", "ŋ"],
    )
    top_k = self.phoible.top_k_segments(12, proportion=0.8)
    self.assertLen(top_k, 9)

  def testTopKLanguageRestricted(self) -> None:
    # All Italian segments.
    top_k = self.phoible.top_k_segments(-1, languages=["ita"])
    self.assertEqual(
        top_k, [
            "b", "d", "dz", "d̠ʒ", "e", "f", "i", "j", "k", "l", "m", "n", "o",
            "p", "r", "s", "t", "ts", "t̠ʃ", "u", "v", "w", "z", "ɐ", "ɔ", "ɛ",
            "ɡ", "ɲ", "ʃ", "ʎ"
        ]
    )
    # Spanish and Italian.
    top_k = self.phoible.top_k_segments(14, languages=["spa", "ita"])
    self.assertEqual(
        top_k,
        ["a", "b", "e", "i", "j", "k", "l", "m", "n", "p", "s", "t", "u", "w"],
    )
    # English and Wu Chinese.
    top_k = self.phoible.top_k_segments(14, languages=["eng", "wuu"])
    self.assertEqual(
        top_k,
        ["a", "b", "i", "j", "k", "l", "m", "n", "p", "s", "t", "u", "w", "ŋ"],
    )
    # Romanian: The top segments should be similar to Spanish and Italian, at
    # least in theory.
    top_k = self.phoible.top_k_segments(14, languages=["ron"])
    self.assertEqual(
        top_k,
        ["a", "b", "e", "i", "j", "k", "l", "m", "n", "p", "s", "t", "u", "w"],
    )

  def testAllSequences(self) -> None:
    top_k = self.phoible.top_k_segments(14)
    template = self.phoible.all_sequences(top_k, (0, 4, 0))
    self.assertEqual(
        template,
        [["b", "k", "p", "t"], ["a", "i", "u"], ["b", "k", "p", "t"]],
    )

  def testMatchingPhonemes(self) -> None:
    voiced = self.phoible.matching_phonemes(
        ["+periodicGlottalSource"], ["p", "b", "t", "d", "k", "ɡ", "f", "v"]
    )
    self.assertEqual(voiced, ["b", "d", "v", "ɡ"])
    voiced_continuant = self.phoible.matching_phonemes(
        ["+periodicGlottalSource", "+continuant"],
        ["p", "b", "t", "d", "k", "ɡ", "f", "v"],
    )
    self.assertEqual(voiced_continuant, ["v"])
    nasal = self.phoible.matching_phonemes(
        ["+nasal"],
        ["p", "b", "t", "d", "k", "ɡ", "f", "v", "n", "m", "ŋ", "m̥"],
    )
    self.assertEqual(nasal, ["m", "m̥", "n", "ŋ"])
    unvoiced_nasal = self.phoible.matching_phonemes(
        ["+nasal", "-periodicGlottalSource"],
        [
            "p",
            "b",
            "t",
            "d",
            "k",
            "ɡ",
            "f",
            "v",
            "n",
            "m",
            "ŋ",
            "m̥",
            "n̥",
            "ŋ̥",
        ],
    )
    self.assertEqual(unvoiced_nasal, ["m̥", "n̥", "ŋ̥"])
    segments = [
        "p",
        "b",
        "t",
        "d",
        "k",
        "ɡ",
        "n",
        "m",
        "ŋ",
        "m̥",
        "n̥",
        "ŋ̥",
    ]

    def plus_minus(feat):
      return [f"+{feat}", f"-{feat}"]

    result = []
    for labial in plus_minus("labial"):
      for coronal in plus_minus("coronal"):
        for voicing in plus_minus("periodicGlottalSource"):
          oral = self.phoible.matching_phonemes(
              [labial, coronal, voicing, "-nasal"], segments
          )
          nasal = self.phoible.matching_phonemes(
              [labial, coronal, voicing, "+nasal"], segments
          )
          if oral and nasal:
            result.append((oral[0], nasal[0]))
    self.assertEqual(
        result,
        [
            ("b", "m"),
            ("p", "m̥"),
            ("d", "n"),
            ("t", "n̥"),
            ("ɡ", "ŋ"),
            ("k", "ŋ̥"),
        ],
    )

  def testPhoneticDistance(self) -> None:
    dist = self.phoible.phonetic_distance("t", "t")
    self.assertEqual(dist, 0)
    dist = self.phoible.phonetic_distance("t", "k")
    self.assertEqual(dist, 2)
    dist = self.phoible.phonetic_distance("k", "m")
    self.assertEqual(dist, 5)
    dist = self.phoible.phonetic_distance("e", "i")
    self.assertEqual(dist, 1)
    dist = self.phoible.phonetic_distance("t", "d")
    self.assertEqual(dist, 1)
    dist = self.phoible.phonetic_distance("t", "n")
    self.assertEqual(dist, 3)
    dist = self.phoible.phonetic_distance("t", "a")
    self.assertEqual(dist, 8)
    dist = self.phoible.phonetic_distance("t", "i")
    self.assertEqual(dist, 8)
    dist = self.phoible.phonetic_distance("a", "i")
    self.assertEqual(dist, 3)
    dist = self.phoible.phonetic_distance("iː", "i")
    self.assertEqual(dist, 1)
    # Testing 0 for empty alignment: the default distance to an empty phoneme
    # corresponds to the overall number of distinctive features.
    dist = self.phoible.phonetic_distance(phoible_segments.EMPTY_PHONEME, "t")
    self.assertEqual(dist, 38)
    dist = self.phoible.phonetic_distance(
        phoible_segments.EMPTY_PHONEME, "t", dist_of_empty=10
    )
    self.assertEqual(dist, 10)

  def testStringDistance(self) -> None:
    dist, _ = self.phoible.string_distance("t i m", "t i m")
    self.assertEqual(dist, 0)
    dist, path = self.phoible.string_distance("t i m", "d i m")
    self.assertEqual(dist, 1)
    self.assertEqual(path, [("t", "d"), ("i", "i"), ("m", "m")])
    dist, _ = self.phoible.string_distance("t i m", "d e m")
    self.assertEqual(dist, 2)
    dist, _ = self.phoible.string_distance("t i m", "d a m")
    self.assertEqual(dist, 4)
    dist, _ = self.phoible.string_distance("t i m", "l o l")
    self.assertEqual(dist, 15)
    dist, _ = self.phoible.string_distance("t i m", "o l")
    self.assertEqual(dist, 20)
    dist, path = self.phoible.string_distance(
        "t i m o f e k o", "o l p a n i m u"
    )
    self.assertEqual(dist, 33)
    self.assertEqual(
        path,
        [
            ("t", "o"),
            ("i", "l"),
            ("m", "p"),
            ("o", "a"),
            ("f", "n"),
            ("e", "i"),
            ("k", "m"),
            ("o", "u"),
        ],
    )
    dist, path = self.phoible.string_distance("t i m".split(), "o l".split())
    self.assertEqual(dist, 20)
    self.assertEqual(path, [("t", None), ("i", "o"), ("m", "l")])


if __name__ == "__main__":
  absltest.main()
