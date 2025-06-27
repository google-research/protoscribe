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

"""Test for `protoscribe.language.phonology.similar_sound_distance`."""

import os

from absl import flags
from absl.testing import absltest
from protoscribe.language.phonology import phoible_segments
from protoscribe.language.phonology import similar_sound_distance as lib

import glob
import os

FLAGS = flags.FLAGS


class SimilarSoundDistanceTest(absltest.TestCase):
  phoible: phoible_segments.PhoibleSegments
  similar_sound_distance: lib.SimilarSoundDistance

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
    cls.phoible = phoible_segments.PhoibleSegments(
        path=phoible_path, features_path=features_path
    )
    cls.similar_sound_distance = lib.SimilarSoundDistance(cls.phoible)
    monosyllables_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/language/phonology/testdata",
        "monosyllables.txt",
    )
    cls.monosyllables = []
    with open(monosyllables_path) as stream:
      for line in stream:
        word = line.strip()
        cls.monosyllables.append(word)

  def distance(self, s1: str, s2: str) -> float:
    s1 = self.phoible.syllabify(s1, parse_onsets_and_rhymes=True)[0]
    s2 = self.phoible.syllabify(s2, parse_onsets_and_rhymes=True)[0]
    return self.similar_sound_distance.syllable_distance(s1, s2)

  def testDistanceSanity(self) -> None:
    pairs_and_distances = [
        (("b a l k", "b a k"), 0.0),
        (("p a l k", "b a k"), 0.0),
        (("m a l k", "b a k"), 0.0),
        (("a l k", "a k"), 0.0),
        (("a l k", "a ɡ"), 0.0),
        (("p a l k", "b i k"), 3.0),
        (("a l k", "b a k"), 10.0),
        (("m o p", "m e b"), 3.0),
        (("e k", "z o m"), 18.0),
        (("b ə s", "k r o k"), 17.0),
    ]
    for (p1, p2), d in pairs_and_distances:
      self.assertEqual(d, self.distance(p1, p2))

  def testDistanceSanity2(self) -> None:
    k = len(self.monosyllables)

    def find_closest(s1):
      others = []
      for j in range(k):
        s2 = self.monosyllables[j]
        if s1 == s2:
          continue
        d = self.distance(s1, s2)
        others.append((d, s2))
      others.sort()
      return others[:5]

    self.assertEqual(
        find_closest("a b"),
        [(0.0, "a p"), (2.0, "a m"), (2.0, "e b"), (2.0, "e p"), (3.0, "a t")],
    )
    self.assertEqual(
        find_closest("a j"),
        [(0.0, "a"), (1.0, "i"), (2.0, "e"), (2.0, "e j"), (3.0, "a w")],
    )
    self.assertEqual(
        find_closest("a m"),
        [(0.0, "a"), (0.0, "a n"), (2.0, "a b"), (2.0, "e"), (2.0, "e m")],
    )
    self.assertEqual(
        find_closest("a n"),
        [(0.0, "a"), (0.0, "a m"), (2.0, "e"), (2.0, "e m"), (2.0, "e n")],
    )
    self.assertEqual(
        find_closest("a p"),
        [(0.0, "a b"), (2.0, "a t"), (2.0, "e b"), (2.0, "e p"), (3.0, "a m")],
    )
    self.assertEqual(
        find_closest("a s"),
        [(2.0, "e s"), (3.0, "a t"), (3.0, "i s"), (3.0, "o s"), (4.0, "a p")],
    )
    self.assertEqual(
        find_closest("a t"),
        [(2.0, "a p"), (2.0, "e t"), (3.0, "a b"), (3.0, "a n"), (3.0, "a s")],
    )
    self.assertEqual(
        find_closest("a w"),
        [(0.0, "a"), (1.0, "u"), (2.0, "e"), (2.0, "e w"), (2.0, "o")],
    )
    self.assertEqual(
        find_closest("b a"),
        [
            (0.0, "b a j"),
            (0.0, "b a n"),
            (0.0, "m a"),
            (0.0, "m a j"),
            (0.0, "m a m"),
        ],
    )
    self.assertEqual(
        find_closest("b a b"),
        [
            (0.0, "b a p"),
            (0.0, "m a b"),
            (0.0, "m a p"),
            (0.0, "p a b"),
            (0.0, "p a p"),
        ],
    )

  def testWordDistance(self) -> None:
    pairs_and_distances = [
        (("m a b a p", "m a b a p"), 0.0),
        (("m a l b a p", "m a b a p"), 0.0),
        (("m a l b a p", "m a k a p"), 3.0),
        (("b a p", "m a k a p"), 13.0),
        (("k o b ə s", "k r o k"), 27.0),
        (("b o k ə s", "k r o k"), 27.0),
        (("b o k r ə s", "k r o k"), 37.0),
        (("b o k r ə k", "k r o k"), 33.0),
        (("b o n k r o k", "k r o k"), 10.0),
    ]
    for (w1, w2), d in pairs_and_distances:
      self.assertEqual(self.similar_sound_distance.word_distance(w1, w2), d)


if __name__ == "__main__":
  absltest.main()
