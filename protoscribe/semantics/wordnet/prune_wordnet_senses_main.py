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

"""A tool that prunes the word senses in embeddings derived from WordNet.

For example, the following sequence of keys

  ray.n.02, ray.n.03, ray.n.06, ray.n.07, ray.v.01

will be transformed into

  ray.n, ray.v

with the corresponding embedding vectors reduced by the supplied method.
"""

import collections
from collections.abc import Sequence
import logging

from absl import app
from absl import flags
import numpy as np

from protoscribe.semantics.wordnet import utils
from protoscribe.semantics.wordnet import wordnet_utils

_INPUT_FILE = flags.DEFINE_string(
    "input_file", None,
    "Input text file in gensim `KeyedVectors` format containing the input "
    "embeddings.",
    required=True
)

_OUTPUT_FILE = flags.DEFINE_string(
    "output_file", None,
    "Output text file in the similar format to --input_file containing the "
    "resulting embeddings after pruning away WordNet senses.",
    required=True
)

_PRUNE_METHOD = flags.DEFINE_enum(
    "prune_method", "mean",
    ["first", "mean"],
    "Reduction method. Select `first` sense we encounter or take `mean` over "
    "all the senses."
)

_WORDNET_POS_TO_BNC = flags.DEFINE_boolean(
    "wordnet_pos_to_bnc", True,
    "If enabled, after the pruning convert the WordNet part-of-speech tag to "
    "BNC-style, e.g., `zanzibar.n` will become `zanzibar_NOUN`."
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  logging.info("Reading embeddings from %s ...", _INPUT_FILE.value)
  embeddings = utils.load_keyvec_embeddings(_INPUT_FILE.value)

  # Construct a mapping from keys (without senses) to the lists of corresponding
  # embedding vectors.
  keys2joined_senses = collections.defaultdict(list)
  for key in sorted(embeddings.keys()):
    toks = key.strip().split(".")
    new_key = f"{toks[0]}.{toks[1]}" if len(toks) == 3 else key
    keys2joined_senses[new_key].append(embeddings[key])

  # For each key reduce multiple embeddings by the requested method.
  new_embeddings = {}
  for key in sorted(keys2joined_senses.keys()):
    emb_vecs = keys2joined_senses[key]
    if _PRUNE_METHOD.value == "first":
      emb_vec = emb_vecs[0]
    elif _PRUNE_METHOD.value == "mean":
      emb_vec = np.mean(emb_vecs, axis=0) if len(emb_vecs) > 1 else emb_vecs[0]
    else:
      raise ValueError("Unknown prune method!")
    new_embeddings[key] = emb_vec

  # Convert POS tags, if enabled.
  if _WORDNET_POS_TO_BNC.value:
    logging.info("Converting POS to BNC ...")
    num_converted = 0
    keys = list(new_embeddings.keys())
    for key in keys:
      toks = key.strip().split(".")
      new_key = key
      if len(toks) == 2:
        if toks[1] in wordnet_utils.WORDNET_POS_TO_PROTOSCRIBE:
          pos = wordnet_utils.WORDNET_POS_TO_PROTOSCRIBE[toks[1]]
          new_key = f"{toks[0]}_{pos}"
      if new_key != key:
        emb_vec = new_embeddings.pop(key)
        new_embeddings[new_key] = emb_vec
        num_converted += 1
    logging.info("Converted %d entries with POS tags.", num_converted)

  logging.info(
      "Saving embeddings with pruned senses to %s ...", _OUTPUT_FILE.value
  )
  utils.save_keyvec_embeddings_from_dict(new_embeddings, _OUTPUT_FILE.value)


if __name__ == "__main__":
  app.run(main)
