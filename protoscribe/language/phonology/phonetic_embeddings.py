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

"""Class definition for phonetic embeddings.

Follows loosely the method in:

Sproat, Richard. 2023. Symbols: An Evolutionary History from the Stone Age to
the Present. Heidelberg: SpringerNature. Chapter 7.

A phonetic embedding set depends on a term list, where the terms are ordered in
decreasing frequency. The first k terms are collected as top_k, where k is the
embedding size. A term's embedding is then the normalized k-dimensional vector
of phonetic distances, where the ith entry is the distance between the term and
the ith entry in top_k.
"""

import logging

import numpy as np
from protoscribe.language.embeddings import embedder as embedder_lib
from protoscribe.language.phonology import phoible_segments
from protoscribe.language.phonology import similar_sound_distance

import glob
import os


class PhoneticEmbeddings:
  """Container for phonetic embedding data."""

  def __init__(
      self,
      phoible_seg: phoible_segments.PhoibleSegments,
      embedding_len: int = embedder_lib.DEFAULT_EMBEDDING_DIM,
      norm_order: int = 1,
  ):
    self._phoible_segments = phoible_seg
    self._similar_sound_distance = similar_sound_distance.SimilarSoundDistance(
        self._phoible_segments,
    )
    self._embedding_len = embedding_len
    self._embeddings = {}
    self._top_k = []
    self._initialized = False
    self._empty_embedding = [0.0] * embedding_len
    self._empty_embedding[0] = 1.0
    self._norm_order = norm_order

  def compute_initial_embeddings(
      self, frequency_ordered_form_list: list[str]
  ) -> None:
    """Computes the initial embeddings for a form list.

    Args:
      frequency_ordered_form_list: a list of forms ordered in *descending*
        frequency. Must be at least of length `self.embedding_len_`.

    Raises:
      ValueError if there are two few unique phonetic forms to construct an
      embedding.
    """
    if len(frequency_ordered_form_list) < self._embedding_len:
      raise ValueError(
          "The number of unique phonetic forms (%d) is less than embedding "
          "dimension (%d)!" % (
              len(frequency_ordered_form_list), self._embedding_len
          )
      )
    self._initialized = True
    # -1 because we reserve the zeroeth dimension for the null embedding.
    self._top_k = frequency_ordered_form_list[: self._embedding_len - 1]
    for form in frequency_ordered_form_list:
      self._embeddings[form] = self.embedding(form)

  def embedding(self, form: str) -> np.ndarray:
    """Looks up or computes the embedding of a form.

    Args:
      form: a string.

    Returns:
      An embedding array.
    """
    assert self._initialized
    if form in self._embeddings:
      return self._embeddings[form]

    embedding = [0.0]
    for i in range(self._embedding_len - 1):
      other_form = self._top_k[i]
      if form == other_form:
        embedding.append(0.0)
        continue
      embedding.append(
          self._similar_sound_distance.word_distance(
              form,
              other_form,
          )
      )
    if sum(embedding) != 0:
      norm = np.linalg.norm(embedding, ord=self._norm_order)
      embedding = [e / norm for e in embedding]

    embedding = np.array(embedding, dtype=np.float64)
    self._embeddings[form] = embedding
    return embedding

  def build_embeddings(self, frequency_ordered_form_list: list[str]) -> None:
    """Computes the embeddings for a form list.

    Args:
      frequency_ordered_form_list: a list of forms ordered in *descending*
        frequency. Must be at least of length `self.embedding_len_`.
    """
    self.compute_initial_embeddings(frequency_ordered_form_list)
    for w in frequency_ordered_form_list:
      _ = self.embedding(w)

  def find_closest_term(self, term: str) -> str | None:
    """Finds the closest term to another term based on embedding.

    Args:
     term: a string

    Returns:
      The closest string, or None if none found.
    """
    assert self._initialized
    min_dist = 1_000_000
    closest = None
    embedding = self.embedding(term)
    if embedding is None:
      return closest
    for other_term in self._embeddings:
      if term == other_term:
        continue
      dist = embedder_lib.distance(embedding, self.embedding(other_term))
      if dist < min_dist:
        min_dist = dist
        closest = other_term
    return closest

  def get_k_nearest_neighbors(
      self, term: str, k: int, allowed_terms: list[str] | None = None
  ) -> list[tuple[str, float]]:
    """Finds k closest neighbors to a form.

    Args:
     term: a string
     k: an integer
     allowed_terms: set of allowed terms or None

    Returns:
      List of pairs of term and distance.
    """
    assert self._initialized
    neighbors = []
    embedding = self.embedding(term)
    if embedding is None:
      return neighbors
    allowed_terms = allowed_terms if allowed_terms else self.keys
    for other_term in allowed_terms:
      if term == other_term:
        continue
      neighbors.append(
          (other_term, embedder_lib.distance(
              embedding, self.embedding(other_term)
          )))
    neighbors.sort(key=lambda x: x[1])
    return neighbors[:k]

  @property
  def empty_embedding(self):
    return self._empty_embedding

  @property
  def keys(self):
    return self._embeddings.keys()

  def distance(self, emb1: np.ndarray, emb2: np.ndarray):
    return embedder_lib.distance(emb1, emb2)

  def write_embeddings(self, path: str) -> None:
    """Writes the embeddings to a text file.

    Args:
       path: path to file
    """
    assert self._initialized
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
    self._initialized = True

  def dump_all_distances(self, path: str):
    """Dumps all distances so that we can study the shape of the distribution.

    Last written line is the means for each of the |V| distances.

    Args:
      path: Path to output file
    """
    assert self._initialized
    all_distances = []
    for k1 in self._embeddings:
      e1 = self._embeddings[k1]
      distances = []
      for k2 in self._embeddings:
        e2 = self._embeddings[k2]
        distances.append((self.distance(e1, e2), k2))
      distances.sort()
      all_distances.append((k1, distances))
    with open(path, "w") as s:
      just_distances = []
      for k1, distances in all_distances:
        line = " ".join([f"{d:.08f}, {k2}" for (d, k2) in distances])
        s.write(f"{k1}\t{line}\n")
        just_distances.append([d for (d, _) in distances])
      means = np.mean(np.array(just_distances), axis=0)
      line = " ".join([f"{d:.08f}" for d in means])
      s.write(f"{line}\n")


def load_phonetic_embedder(
    embeddings_file_path: str,
    phoible_phonemes_path: str = phoible_segments.PHOIBLE,
    phoible_features_path: str = phoible_segments.PHOIBLE_FEATURES,
) -> PhoneticEmbeddings:
  """Creates a phonetic embedder.

  Args:
    embeddings_file_path: Path to a file containing the embeddings.
    phoible_phonemes_path: File containing PHOIBLE phonemes.
    phoible_features_path: File containing PHOIBLE features.

  Returns:
    An instance of PhoneticEmbeddings object.
  """
  logging.info("Loading phonetic embeddings from %s ...", embeddings_file_path)

  phoible = phoible_segments.PhoibleSegments(
      phoible_phonemes_path,
      phoible_features_path,
  )
  embedder = PhoneticEmbeddings(
      phoible, embedding_len=embedder_lib.DEFAULT_EMBEDDING_DIM
  )
  embedder.read_embeddings(embeddings_file_path)
  logging.info("Loaded %d phonetic embeddings.", len(embedder.keys))
  return embedder
