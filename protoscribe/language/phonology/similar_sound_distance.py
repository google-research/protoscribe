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

"""Phonetic distance with reference to notion of rebus intersubstitutability.

Attempts to define a more restricted and hence more defensible notion of
phonetic closeness, following

Baxter, William. 1992. A Handbook of Old Chinese Phonology. Number 64 in Trends
in Linguistics, Studies and Monographs, Berlin: Mouton de Gruyter.

who states (page 348) that for Ancient Chinese:

`In order to be written with the same phonetic element, words must normally have
identical main vowels and codas, and their initial consonants must have the same
position of articulation.`

In other early writing the restrictions seem to have been somewhat looser than
this, but Baxter's observation is a good starting point, and is better than
erring on the side of being too permissive.
"""

from protoscribe.language.phonology import phoible_segments


class SimilarSoundDistance:
  """Computes whether two syllables sound similar returning a distance."""

  SAME_POA_FEATURES = ["+labial", "+labiodental", "+coronal", "+dorsal"]

  def __init__(
      self,
      phoible: phoible_segments.PhoibleSegments,
  ):
    self._phoible = phoible
    self._syllabified_words = {}

  def _match_for_rebus(self, ph1: str, ph2: str, coda: bool = False) -> bool:
    """Matches a pair of phonemes given features and position.

    This largely implements Baxter's generalization in the introductory comment.

    Args:
      ph1: A string representing a phoneme
      ph2: A string representing a phoneme
      coda: If True, then the phoneme is in the code

    Returns:
      bool
    """
    if ph1 == ph2:
      return True
    feats1 = self._phoible.features(ph1, keep_unspecified=True) if ph1 else []
    feats2 = self._phoible.features(ph2, keep_unspecified=True) if ph2 else []
    if not ph1 or not ph2:
      # Allow sonorant insertion in coda
      if coda:
        if not ph1 and "+sonorant" in feats2:
          return True
        if not ph2 and "+sonorant" in feats1:
          return True
      return False

    def same_place_of_articulation(feats1, feats2):
      for f in self.SAME_POA_FEATURES:
        if f in feats1 and f in feats2:
          return True
      return False

    def mismatch_on_feature(feats1, feats2, feature_name):
      return (
          f"+{feature_name}" in feats1 and f"-{feature_name}" in feats2
      ) or (f"-{feature_name}" in feats1 and f"+{feature_name}" in feats2)

    if coda:
      if "+nasal" in feats1 and "+nasal" in feats2:
        return True
      if same_place_of_articulation(feats1, feats2):
        # Don't allow, say, "b" and "m" to match in coda
        if mismatch_on_feature(feats1, feats2, "sonorant"):
          return False
        # But "b" and "p" should match
        if mismatch_on_feature(feats1, feats2, "periodicGlottalSource"):
          return True
      return False
    else:
      if "-consonantal" in feats1 or "-consonantal" in feats2:
        return False
      return (
          # POA must match, but don't allow, say "s" and "t" to match.
          same_place_of_articulation(feats1, feats2)
          and not mismatch_on_feature(feats1, feats2, "continuant")
      )

  def syllable_distance(
      self,
      syl1: phoible_segments.ParsedSyllables,
      syl2: phoible_segments.ParsedSyllables,
  ) -> float:
    """Computes distance between 2 syllables, using rebus matching.

    Args:
      syl1: A tuple representing a parsed syllable.
      syl2: A tuple representing a parsed syllable.

    Returns:
      float
    """
    empty_dist = 10.0
    o1, c1 = syl1
    o2, c2 = syl2
    dist = 0.0
    if len(o1) == len(o2):
      for i in range(len(o1)):
        ph1, ph2 = o1[i], o2[i]
        if self._match_for_rebus(ph1, ph2, coda=False):
          continue
        else:
          dist += self._phoible.phonetic_distance(ph1, ph2)
    else:
      dist += empty_dist
    _, alignment = self._phoible.string_distance(c1, c2)
    for ph1, ph2 in alignment:
      if self._match_for_rebus(ph1, ph2, coda=True):
        continue
      else:
        if not ph1 or not ph2:
          dist += empty_dist
        else:
          dist += self._phoible.phonetic_distance(ph1, ph2)
    return dist

  def _syllabify(self, w):
    if w not in self._syllabified_words:
      self._syllabified_words[w] = self._phoible.syllabify(
          w,
          parse_onsets_and_rhymes=True,
      )
    return self._syllabified_words[w]

  def word_distance(self, w1: str, w2: str) -> float:
    """Computes distance between 2 words.

    Args:
      w1: String representing the first word.
      w2: String representing the second word.

    Returns:
      float
    """
    sylls1 = self._syllabify(w1)
    sylls2 = self._syllabify(w2)
    empty_syl = ((), ())
    l1 = len(sylls1)
    l2 = len(sylls2)
    if l1 < l2:
      count = l2
      sylls1 = [empty_syl] * (l2 - l1) + list(sylls1)
    else:
      count = l1
      sylls2 = [empty_syl] * (l1 - l2) + list(sylls2)
    dist = 0.0
    for i in range(count):
      dist += self.syllable_distance(sylls1[i], sylls2[i])
    return dist
