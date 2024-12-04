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

"""K-Nearest Neighbors."""

from typing import Optional

import numpy as np
from protoscribe.language.embeddings import embedder
import scipy

import glob
import os


def load_concepts(concepts_files: Optional[str]) -> list[str]:
  """Loads categories from files."""
  concepts = set()
  for path in concepts_files:
    with open(path) as stream:
      for line in stream:
        toks = line.strip().split()
        if len(toks) > 1:
          concepts.add(toks[0])
          concepts.add(toks[1])
        else:
          concepts.add(toks[0])
  return list(concepts)


def load_and_filter(
    concepts_files: Optional[list[str]],
    embeddings: embedder.Embeddings,
    try_tagless: bool = True
) -> embedder.Embeddings:
  """Loads the concepts and filters the embeddings."""
  concepts = set(load_concepts(concepts_files))
  if try_tagless:
    concepts.update([c.split("_")[0] for c in concepts])
  new_embeddings = embedder.Embeddings(embeddings.embedding_type)
  for k in embeddings.embeddings:
    if k in concepts:
      new_embeddings[k] = embeddings[k]
  return new_embeddings


def get_k_nearest_neighbours(
    word: str, word_emb: np.ndarray, embeddings: embedder.Embeddings, k: int
) -> list[tuple[str, float]]:
  """Returns a sorted list of $k$ closest neighbors for a given word."""
  neighbors = []
  for w in embeddings.embeddings:
    if w == word:
      continue
    dist = scipy.spatial.distance.cosine(word_emb, embeddings[w])
    neighbors.append((w, float(dist)))
  neighbors = sorted(neighbors, key=lambda neighbor: neighbor[1])
  if k > 0:
    neighbors = neighbors[:k]
  return neighbors
