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

"""Generates phonetic embeddings for a fixed set of lexicons.

Note that this does NOT take into account phonological rules or morphology. We
need to address that in future work.
"""

import collections
import logging

from absl import app
from absl import flags
from protoscribe.language.embeddings import embedder
from protoscribe.language.phonology import phoible_segments
from protoscribe.language.phonology import phonetic_embeddings

import glob
import os

_EMBEDDINGS = flags.DEFINE_string(
    "embeddings", None, "Path to output embeddings.",
    required=True
)
_MAIN_LEXICON = flags.DEFINE_string(
    "main_lexicon", None, "Path to main lexicon.",
    required=True
)
_NUMBER_LEXICON = flags.DEFINE_string(
    "number_lexicon", None, "Path to number lexicon.",
    required=True
)
_PHOIBLE_PATH = flags.DEFINE_string(
    "phoible_path", phoible_segments.PHOIBLE, "Path to Phoible segments."
)
_PHOIBLE_FEATURES_PATH = flags.DEFINE_string(
    "phoible_features_path",
    phoible_segments.PHOIBLE_FEATURES,
    "Path to Phoible features.",
)
_NORM_ORDER = flags.DEFINE_integer(
    "norm_order", 1,
    "Embedding vector normalization (L1 by default). Set to 2 for L2."
)


def _lexicon_to_words(path: str, words: dict[str, int]) -> None:
  """Load lexicon into set of words.

  Args:
    path: path to lexicon
    words: a sequence of counts and words
  """

  with open(path) as stream:
    for line in stream:
      _, phon = line.strip("\n").split("\t")
      words[phon] += 1


def main(unused_argv):
  phoible = phoible_segments.PhoibleSegments(
      path=_PHOIBLE_PATH.value,
      features_path=_PHOIBLE_FEATURES_PATH.value,
  )
  embeddings = phonetic_embeddings.PhoneticEmbeddings(
      phoible_seg=phoible,
      embedding_len=embedder.DEFAULT_EMBEDDING_DIM,
      norm_order=_NORM_ORDER.value,
  )
  words = collections.defaultdict(int)
  _lexicon_to_words(_MAIN_LEXICON.value, words)
  _lexicon_to_words(_NUMBER_LEXICON.value, words)
  freq_ordered_list = []
  for k, v in words.items():
    freq_ordered_list.append((v, k))
  freq_ordered_list.sort(reverse=True)
  freq_ordered_list = [w for _, w in freq_ordered_list]
  logging.info(
      "Building embeddings from %d unique pronunciations ...",
      len(freq_ordered_list)
  )
  embeddings.build_embeddings(freq_ordered_list)
  logging.info("Saving embeddings to %s ...", _EMBEDDINGS.value)
  embeddings.write_embeddings(_EMBEDDINGS.value)


if __name__ == "__main__":
  app.run(main)
