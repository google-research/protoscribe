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

"""Computes k-NN for concepts using the phonetic embeddings.

This is somewhat similar in function to `phonetic_embeddings_distances` tool,
but supports lookups via and filtering by the category names.
"""

from collections.abc import Sequence
import csv
import itertools
import logging

from absl import app
from absl import flags
from protoscribe.language.phonology import phoible_segments
from protoscribe.language.phonology import phonetic_embeddings
from protoscribe.texts import generate as generate_lib

import glob
import os

_INPUT_EMBEDDINGS_FILE = flags.DEFINE_string(
    "input_embeddings_file", None,
    "Path to the input phonetic embeddings file in TSV format.",
    required=True
)

_PHOIBLE_PATH = flags.DEFINE_string(
    "phoible_path", phoible_segments.PHOIBLE, "Path to PHOIBLE segments."
)

_PHOIBLE_FEATURES_PATH = flags.DEFINE_string(
    "phoible_features_path",
    phoible_segments.PHOIBLE_FEATURES,
    "Path to PHOIBLE features.",
)

_TOP_K = flags.DEFINE_integer(
    "top_k", 3,
    "Keep best k candidates. If negative, compute for all entries."
)

_OUTPUT_TSV_FILE = flags.DEFINE_string(
    "output_tsv_file", None,
    "Path to the output file in TSV format containing all the closest "
    "neighbors from the seen set.",
    required=True
)

# Following will expose the category and lexicon command-line flags via FLAGS.
# In particular, we will need the main and number lexicons, and the
# administrative and non-administrative categories.
FLAGS = flags.FLAGS


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Load phonetic embeddings.
  embeddings = phonetic_embeddings.load_phonetic_embedder(
      embeddings_file_path=_INPUT_EMBEDDINGS_FILE.value,
      phoible_phonemes_path=_PHOIBLE_PATH.value,
      phoible_features_path=_PHOIBLE_FEATURES_PATH.value
  )

  # Load administrative (seen) and non-administrative (unseen) concepts.
  # Make sure both are specified.
  if not FLAGS.concepts or not FLAGS.unseen_concepts:
    raise ValueError("Specify paths to both seens and unseen concepts!")

  _, seen_concepts = generate_lib.load_concepts(FLAGS.concepts)
  _, unseen_concepts = generate_lib.load_concepts(FLAGS.unseen_concepts)

  # Load category and number lexicon.
  if not FLAGS.main_lexicon or not FLAGS.number_lexicon:
    raise ValueError("Specify --main_lexicon and --number_lexicon!")

  lexicon, _ = generate_lib.load_phonetic_forms(
      main_lexicon_file=FLAGS.main_lexicon,
      number_lexicon_file=FLAGS.number_lexicon
  )
  logging.info("Loaded total of %d pronunciations.", len(lexicon))

  # Cache the embeddings for seen concepts.
  all_terms = embeddings.keys
  seen_terms = []
  for concept in seen_concepts:
    concept = concept.split("_")[0]  # POS kludge.
    if concept not in lexicon:
      raise ValueError(f"Concept {concept} not found in pronunciation lexicon!")
    pron = " ".join(lexicon[concept])
    if pron not in all_terms:
      raise ValueError(f"No embedding found for pronunciation '{pron}'!")
    seen_terms.append(pron)

  # For each concept in unseen set compute its $k$-nearest neighbors.
  logging.info("Saving results to %s ...", _OUTPUT_TSV_FILE.value)
  with open(_OUTPUT_TSV_FILE.value, mode="wt") as f:
    writer = csv.writer(f, delimiter="\t")
    top_k_header = [
        (f"Pron{k}", f"Dist{k}") for k in range(1, _TOP_K.value + 1)
    ]
    writer.writerow(
        ["NewConcept", "NewPron"] + list(itertools.chain(*top_k_header))
    )
    for concept in unseen_concepts:
      # Lookup the pronunciation.
      concept = concept.split("_")[0]  # POS kludge.
      if concept not in lexicon:
        raise ValueError(
            f"Concept {concept} not found in pronunciation lexicon!")
      pron = " ".join(lexicon[concept])

      # Compute nearest K pronunciations.
      nearest = embeddings.get_k_nearest_neighbors(
          pron, _TOP_K.value, allowed_terms=seen_terms
      )
      nearest = [(other_p, float(d)) for other_p, d in nearest]
      nearest = list(itertools.chain(*nearest))
      writer.writerow([concept, pron] + nearest)
    logging.info("Processed %d concepts.", len(unseen_concepts))


if __name__ == "__main__":
  app.run(main)
