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

"""An alternative approach to phonetic embeddings.

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

import collections
from typing import Callable

import numpy as np
from protoscribe.language.embeddings import embedder
from protoscribe.language.phonology import phoible_segments
from sklearn.decomposition import PCA

import glob
import os


class RebusEmbeddings:
  """Constructs and stores embeddings based on rebuses."""

  def __init__(
      self,
      phoible: phoible_segments.PhoibleSegments,
      embedding_len: int = embedder.DEFAULT_EMBEDDING_DIM,
  ) -> None:
    self._phoible = phoible
    self._embedding_len = embedding_len
    self._empty_embedding = [0.0] * embedding_len
    self._empty_embedding[0] = 1.0
    self._empty_embedding = np.array(self._empty_embedding)
    self._embeddings = collections.defaultdict(list)
    self._words = []

  SAME_POA_FEATURES = ["+labial", "+labiodental", "+coronal", "+dorsal"]

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

  def _match_syllables(
      self,
      syl1: phoible_segments.ParsedSyllables,
      syl2: phoible_segments.ParsedSyllables,
  ) -> bool:
    """Matches two syllables.

    Args:
      syl1: A tuple representing a parsed syllable.
      syl2: A tuple representing a parsed syllable.

    Returns:
      bool
    """
    assert len(syl1) >= 2
    o1, c1 = syl1
    assert len(syl2) >= 2
    o2, c2 = syl2
    if len(o1) != len(o2):
      return False
    for i in range(len(o1)):
      ph1, ph2 = o1[i], o2[i]
      if not self._match_for_rebus(ph1, ph2, coda=False):
        return False
    _, alignment = self._phoible.string_distance(c1, c2)
    for ph1, ph2 in alignment:
      if not self._match_for_rebus(ph1, ph2, coda=True):
        return False
    return True

  def match_words(self, w1: str, w2: str) -> bool:
    """Matches two words as potential rebus substitutions.

    Args:
      w1: a word
      w2: a word

    Returns:
      bool
    """
    w1 = self._phoible.syllabify(w1, parse_onsets_and_rhymes=True)
    w2 = self._phoible.syllabify(w2, parse_onsets_and_rhymes=True)
    if len(w1) != len(w2):
      return False
    for i in range(len(w1)):
      syl1, syl2 = w1[i], w2[i]
      if not self._match_syllables(syl1, syl2):
        return False
    return True

  def embedding(self, word: str) -> np.ndarray:
    """Returns the embedding for the word, if any.

    Args:
      word: a word

    Returns:
      np.array for the embedding
    """
    if word in self._embeddings:
      return self._embeddings[word]
    return self._empty_embedding

  @property
  def embeddings(self) -> collections.defaultdict:
    return self._embeddings

  @property
  def empty_embedding(self) -> np.ndarray:
    return self._empty_embedding

  def build_embeddings(
      self,
      words: list[str],
      equiv: Callable[[str, str], bool] | None = None,
  ) -> None:
    """Builds the embeddings for a wordlist.

    For each word in the vocabulary, we first compute its set of matching
    rebuses among the other words. Then the initial embedding for that word will
    simply be the zero vector of length V, the size of the vocabulary, with 1's
    in the position of every possible rebus subsitution, including the word
    itself.  We then use PCA to reduce the dimensionality of the vector space to
    embedding_len (typically 300).

    NB: For PCA to work there must be at least embedding_len number of
    vocabulary items. Since this will typically be the case, we don't implement
    a backup strategy.

    Args:
      words: list of words
      equiv: An optional function for computing whether two words are in the
             same equivalence class. The function should take two string
             arguments and return a boolean.
    """
    if not equiv:
      equiv = self.match_words
    self._words = sorted(words)
    indices = {}
    idx = 0
    for word in self._words:
      indices[word] = idx
      idx += 1
    k = len(self._words)
    raw_embeddings = {}
    for w in self._words:
      raw_embeddings[w] = [0.0] * k
    for i in range(k - 1):
      w1 = self._words[i]
      raw_embeddings[w1][indices[w1]] = 1.0
      for j in range(i + 1, k):
        w2 = self._words[j]
        if equiv(w1, w2):
          raw_embeddings[w1][indices[w2]] = 1.0
          raw_embeddings[w2][indices[w1]] = 1.0
    input_embeddings = []
    for w in self._words:
      input_embeddings.append(raw_embeddings[w])
    input_embeddings = np.array(input_embeddings)
    assert input_embeddings.shape[0] > self._embedding_len
    pca = PCA(n_components=self._embedding_len)
    output_embeddings = pca.fit_transform(input_embeddings)
    for i in range(k):
      self._embeddings[self._words[i]] = output_embeddings[i]

  def write_embeddings(self, path: str) -> None:
    """Writes the embeddings to a text file.

    Args:
       path: path to file
    """
    with open(path, "w") as s:
      for w in self._embeddings:
        emb = " ".join([str(v) for v in self._embeddings[w]])
        s.write(f"{w}\t{emb}\n")

  def read_embeddings(self, path: str) -> None:
    """Reads the embeddings from a text file.

    Args:
       path: path to file
    """
    with open(path) as s:
      for line in s:
        w, emb = line.strip().split("\t")
        emb = np.array([float(v) for v in emb.split()])
        self._embeddings[w] = emb

  def distance(self, e1: np.ndarray, e2: np.ndarray) -> float:
    return embedder.distance(e1, e2)
