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

"""Extracts a sequence of BNC embeddings for an input sequence of concepts."""

import dataclasses
import traceback

from absl import logging
import numpy as np
import scipy

# Internal resources dependency

DEFAULT_EMBEDDING_DIM = 300
NUM_EMB_FILENAME = "numbers.txt"
CONC_EMB_FILENAME = "embeddings.txt"


@dataclasses.dataclass(frozen=True)
class _Config:
  """Embedding type configuration."""
  concept_col_id: int  # Column ID for the concept name.
  vector_offset: int   # Offset for the actual embedding.
  dimension: int       # Embedding dimension for concepts and numbers.


# Configuration for number/concept embeddings.
_EMBEDDING_CONFIG = {
    "bnc": _Config(
        concept_col_id=1, vector_offset=2, dimension=DEFAULT_EMBEDDING_DIM
    ),
}
EMBEDDING_TYPES = sorted(_EMBEDDING_CONFIG.keys())

# When printing the trace of an error, limit the number of entries to this
# number.
_TRACEBACK_LIMIT = 5


def _null_embedding(embedding_type: str) -> np.ndarray:
  """Returns null embedding vector of given dimension."""
  return np.zeros(_EMBEDDING_CONFIG[embedding_type].dimension)


def distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
  """Computes the (cosine) distance between two embeddings.

  Args:
    emb1: First embedding, an array.
    emb2: Second embedding, an array.

  Returns:
    Cosine distance between emb1 and emb2.
  """
  return scipy.spatial.distance.cosine(emb1, emb2)


# TODO: We probably will need to think of a way to speed this up.
def distance_or_zero(
    key: np.ndarray, val: np.ndarray, embeddings: dict[str, np.ndarray]
) -> float:
  """If val = closest(key, embeddings), then 0.0 else distance(key, val).

  Args:
    key: First embedding, an array.
    val: Second embedding, an array.
    embeddings: A dictionary mapping strings to arrays.

  Returns:
    0.0 if the key's closest embedding in embeddings is the same as val,
        otherwise the distance between key and val.
  """

  def is_closest_distance(key: np.ndarray, min_dist: float) -> bool:
    for emb in embeddings.values():
      dist = distance(key, emb)
      if dist < min_dist:
        return False
    return True

  dist = distance(key, val)
  if is_closest_distance(key, dist):
    return 0.0
  return dist


def get_k_nearest_neighbours(
    query: np.ndarray,
    embeddings: dict[str, np.ndarray],
    k: int,
    allowed_entries: set[str] | None = None,
    try_tagless: bool = True,
) -> list[tuple[str, float]]:
  """Given a vector query, return k-nearest entries."""
  neighbors = []
  allowed_keys = allowed_entries if allowed_entries else embeddings.keys()
  for key in allowed_keys:
    emb = embeddings[key] if key in embeddings else None
    if emb is None and try_tagless:
      tagless_key = key.split("_")[0]
      emb = embeddings[tagless_key] if tagless_key in embeddings else None
    if emb is None:
      continue  # Silently ignore lookup failure.
    dist = distance(query, emb)
    neighbors.append((key, dist))
  neighbors = sorted(neighbors, key=lambda neighbor: neighbor[1])
  if k > 0:
    neighbors = neighbors[:k]
  return neighbors


class Embeddings:
  """Container for embeddings, created by load_embeddings."""

  def __init__(self, embedding_type):
    self._embeddings = {}
    self._embedding_type = embedding_type
    self._null_embedding = _null_embedding(embedding_type)
    self._dimension = _EMBEDDING_CONFIG[embedding_type].dimension

  def __getitem__(self, key: str) -> np.ndarray | None:
    # Note: this method does not throw if the key is missing.
    if key not in self._embeddings:
      key = key.split("_")[0]  # Split off pos.
    return self._embeddings[key] if key in self._embeddings else None

  def __setitem__(self, key: str, value: np.ndarray):
    self._embeddings[key] = value

  def __len__(self) -> int:
    return len(self._embeddings)

  @property
  def embeddings(self) -> dict[str, np.ndarray]:
    return self._embeddings

  @property
  def embedding_type(self) -> str:
    return self._embedding_type

  @property
  def dimension(self) -> int:
    return self._dimension

  @property
  def null_embedding(self) -> np.ndarray:
    return self._null_embedding

  def embed(
      self,
      text: str,
      embedding_type: str,
      add_null_embedding: bool = True,
      try_tagless: bool = True,
  ) -> list[np.ndarray]:
    """Embeds a text as a list of arrays.

    Args:
      text: a string of numbers and concepts.
      embedding_type: a string specifying embedding type.
      add_null_embedding: if True, add a null embedding for cases where the
        concept/number is not found.
      try_tagless: if True, try removing the tag and then looking up.

    Returns:
      A list of arrays.
    """
    embseq = []
    for concept in text.split():
      emb = self._embeddings[concept] if concept in self._embeddings else None
      if emb is None and try_tagless:
        concept_no_pos = concept.split("_")[0]
        if concept_no_pos in self._embeddings:
          emb = self._embeddings[concept_no_pos]
        else:
          emb = None
      if emb is None:
        logging.error(
            "Cannot find embedding for `%s` in `%s`. Traceback: %s",
            concept, embedding_type, "".join(
                traceback.format_stack(limit=_TRACEBACK_LIMIT)[:-1]
            )
        )
        if add_null_embedding:
          embseq.append(self._null_embedding)
      else:
        embseq.append(emb)
    return embseq

  # Redundant static method just to make calls to distance/distance_or_zero have
  # the same look-and-feel.
  @staticmethod
  def distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return distance(emb1, emb2)

  def distance_or_zero(self, key: np.ndarray, val: np.ndarray) -> float:
    return distance_or_zero(key, val, self._embeddings)

  def get_k_nearest_neighbours(
      self,
      query: np.ndarray,
      k: int,
      allowed_entries: set[str] | None = None,
      try_tagless: bool = True,
  ) -> list[tuple[str, float]]:
    return get_k_nearest_neighbours(
        query, self.embeddings, k,
        allowed_entries=allowed_entries,
        try_tagless=try_tagless
    )


def load_embeddings(
    embedding_type: str, num_emb_path: str, conc_emb_path: str
) -> Embeddings:
  """Loads the embeddings for numbers and concepts.

  Args:
    embedding_type: type of the embeddings (e.g., `bnc`).
    num_emb_path: path to number embeddings.
    conc_emb_path: path to concept embeddings.

  Returns:
    An Embeddings class instance.
  """
  logging.info(
      "Loading `%s` semantic embeddings from %s and %s ...",
      embedding_type, num_emb_path, conc_emb_path
  )
  embeddings = Embeddings(embedding_type)

  # Do this for numbers since we don't need the "_NUM" tag.
  def remove_tag(conc):
    return conc.split("_")[0]

  def conv(emb):
    return [float(e) for e in emb]

  config = _EMBEDDING_CONFIG[embedding_type]
  with open(num_emb_path, mode="rt", encoding="utf-8") as s:
    for line in s:
      line = line.split()
      num, emb = remove_tag(line[0]), conv(line[1:])
      if len(emb) < config.dimension:
        emb += [0.] * (config.dimension - len(emb))
      embeddings[num] = np.array(emb, dtype=np.float64)

  with open(conc_emb_path, mode="rt", encoding="utf-8") as s:
    for line in s:
      line = line.split()
      concept = line[config.concept_col_id]
      embedding = conv(line[config.vector_offset:])
      # Make sure the dimensions for concepts and numbers match.
      if len(embedding) < config.dimension:
        embedding += [0.] * (config.dimension - len(embedding))
      embeddings[concept] = np.array(embedding, dtype=np.float64)

  if not embeddings:
    raise ValueError(f"{embedding_type}: No embeddings loaded!")
  logging.info("Loaded %d semantic embeddings.", len(embeddings))
  return embeddings


def _embedding_file_path(
    embedding_type: str,
    file_name: str,
    prefix: str,
) -> str:
  if not prefix:
    prefix = "protoscribe/data/semantics"
  return prefix + f"/{embedding_type}/{file_name}"


def load_embeddings_from_type(
    embedding_type: str,
    prefix: str | None = None,
) -> Embeddings:
  """Loads the embeddings for numbers and concepts given the embedding type.

  Args:
    embedding_type: Name of the embedding directory (e.g., "enwiki").
    prefix: Optional path specifying location of embedding data.

  Returns:
    An Embeddings class instance.
  """
  return load_embeddings(
      embedding_type,
      _embedding_file_path(embedding_type, NUM_EMB_FILENAME, prefix),
      _embedding_file_path(embedding_type, CONC_EMB_FILENAME, prefix),
  )


def embedding_dim_from_type(embedding_type: str) -> int:
  """Returns embeddings dimension from type."""
  return _EMBEDDING_CONFIG[embedding_type].dimension
