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

"""Miscellaneous utilities."""

import logging

from gensim.models.keyedvectors import KeyedVectors
import numpy as np

import glob
import os


def load_keyvec_embeddings(path: str) -> dict[str, np.ndarray]:
  """Loads the embeddings stored in KeyedVectors text format.

  Args:
    path: Path to a file.

  Returns:
    A mapping from strings to floating point vectors.

  Raises:
    ValueError if error occurred during parsing.
  """
  embeddings = {}
  with open(path, "rt") as f:
    toks = f.readline().strip().split()
    if len(toks) != 2:
      raise ValueError("Expected two columns <num> and <dim> in first line!")
    emb_dim = int(toks[1])
    for line in f:
      toks = line.strip().split()
      emb_vec = [float(v) for v in toks[1:]]
      emb_vec = np.array(emb_vec, dtype=np.float64)
      if len(emb_vec.shape) != 1:
        raise ValueError(
            f"Expected one-dimensional vector! Got {len(emb_vec.shape)} dims"
        )
      if emb_vec.shape[0] != emb_dim:
        raise ValueError(
            f"Expected dimension {emb_dim}. Got {emb_vec.shape[0]}!"
        )
      embeddings[toks[0]] = emb_vec
  logging.info("Loaded %d embeddings.", len(embeddings))
  return embeddings


def save_keyvec_embeddings_from_dict(
    embeddings: dict[str, np.ndarray], output_file: str
) -> None:
  """Given the dictionary saves the embeddings.

  Args:
    embeddings: Mapping from keys to embedding vectors.
    output_file: Path where to save the vectors.
  """
  num_vecs = len(embeddings)
  keys = sorted(embeddings.keys())
  with open(output_file, mode="wt") as f:
    vector_size = embeddings[keys[0]].shape[0]
    f.write(f"{num_vecs} {vector_size}\n")
    for key in keys:
      vec = embeddings[key]
      vec = vec / np.maximum(np.linalg.norm(vec, ord=2), 1e-12)
      str_vec = " ".join(str(val) for val in vec.tolist())
      f.write(f"{key} {str_vec}\n")
  logging.info("Saved %d vectors to %s.", num_vecs, output_file)


def save_keyvec_embeddings(model: KeyedVectors, output_file: str) -> None:
  """Given an instance of `KeyedVectors` model saves the embeddings.

  Args:
    model: Model representation as gensim `KeyedVectors`.
    output_file: Path where to save the vectors.
  """
  embeddings = {}
  for key in model.vocab.keys():
    row = model.get_vector(key)
    embeddings[key] = row
  save_keyvec_embeddings_from_dict(embeddings, output_file)
