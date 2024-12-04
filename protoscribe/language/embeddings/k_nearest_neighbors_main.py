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

r"""Simple tool for computing $k$-nearest neighbors given the embedding.

Example:
--------
Find five nearest seen concepts for each unseen concept:
  python protoscribe/language/embeddings/k_nearest_neighbors_main.py \
    --embedding_type bnc \
    --top_k 5 \
    --seen_concepts_files protoscribe/data/concepts/administrative_categories.txt \
    --unseen_concepts_files protoscribe/data/concepts/non_administrative_categories.txt \
    --output_file /tmp/results.txt \
    --logtostderr
"""

import logging
from typing import Sequence

from absl import app
from absl import flags
from protoscribe.language.embeddings import embedder
from protoscribe.language.embeddings import k_nearest_neighbors as lib

import glob
import os

_EMBEDDING_TYPE = flags.DEFINE_enum(
    "embedding_type",
    "bnc",
    embedder.EMBEDDING_TYPES,
    "Type of embedding. Default: `bnc`.",
)

_TOP_K = flags.DEFINE_integer(
    "top_k", 3,
    "Keep best k candidates. If negative, compute for all entries."
)

_SEEN_CONCEPTS_FILES = flags.DEFINE_list(
    "seen_concepts_files", None,
    "Path to list of concepts seen during training."
)

_UNSEEN_CONCEPTS_FILES = flags.DEFINE_list(
    "unseen_concepts_files", None,
    "Path to list of concepts not seen during training."
)

_OUTPUT_FILE = flags.DEFINE_string(
    "output_file", None,
    "Path to the file containing all the closest neighbors."
)

_TAGLESS_LOOKUP = flags.DEFINE_bool(
    "tagless_lookup", True,
    "When looking up the embeddings in the available concept set employ "
    "(POS) tagless lookup if the embedding key is not present."
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  logging.info("Loading `%s` embeddings ...", _EMBEDDING_TYPE.value)
  embeddings = embedder.load_embeddings_from_type(_EMBEDDING_TYPE.value)

  if _SEEN_CONCEPTS_FILES.value:
    if not _UNSEEN_CONCEPTS_FILES.value:
      raise ValueError("Specify --unseen_concepts_files as well")
    logging.info("Loading seen concepts and filtering ...")
    seen_embeddings = lib.load_and_filter(
        _SEEN_CONCEPTS_FILES.value,
        embeddings,
        try_tagless=_TAGLESS_LOOKUP.value
    )
    logging.info("Retained %d seen embeddings.", len(seen_embeddings))

  if _UNSEEN_CONCEPTS_FILES.value:
    if not _SEEN_CONCEPTS_FILES.value:
      raise ValueError("Specify --seen_concepts_files as well")
    logging.info("Loading unseen concepts and filtering ...")
    unseen_embeddings = lib.load_and_filter(
        _UNSEEN_CONCEPTS_FILES.value,
        embeddings,
        try_tagless=_TAGLESS_LOOKUP.value
    )
    logging.info("Retained %d unseen embeddings.", len(unseen_embeddings))

  logging.info("Computing nearest neighbors ...")
  if not _SEEN_CONCEPTS_FILES.value and not _UNSEEN_CONCEPTS_FILES.value:
    query_embeddings = embeddings
    source_embeddings = embeddings
  else:
    query_embeddings = unseen_embeddings
    source_embeddings = seen_embeddings

  f = None
  if _OUTPUT_FILE.value:
    logging.info("Saving results to %s ...", _OUTPUT_FILE.value)
    f = open(_OUTPUT_FILE.value, mode="wt")
  for word in sorted(query_embeddings.embeddings.keys()):
    word_neighbors = lib.get_k_nearest_neighbours(
        word, query_embeddings[word], source_embeddings, _TOP_K.value)
    if f:
      f.write("{}: {}\n".format(word, " ".join(map(str, word_neighbors))))
    else:
      print("{}: {}".format(word, " ".join(map(str, word_neighbors))))
  if f:
    f.close()
  logging.info("Done.")


if __name__ == "__main__":
  app.run(main)
