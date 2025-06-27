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

r"""Utility for extracting KeyedVector embeddings for given concepts.

The provenance of original embeddings does not matter as long as they are in
gensim's KeyedVector text format. WordNet synsets are used to fill in the
missing concepts by their synonyms.

Please be warned that this tool is hacky. It's been tested at various times
on different types of embeddings
  - WordNet-derived embeddings constructed using Protoscribe tools in `wordnet`
    subdirectory,
  - ConceptNet NumberBatch from
    https://github.com/commonsense/conceptnet-numberbatch
  - WordNet embeddings from Chakaveh, et al. (Saedi, Chakaveh, António Branco,
    João António Rodrigues and João Ricardo Silva, 2018, "WordNet Embeddings",
    In Proceedings, 3rd Workshop on Representation Learning for Natural Language
    Processing (RepL4NLP), 56th Annual Meeting of the ACL".
but still requires substantial amount of refactoring. The major difficulty is
that we internally do not support word senses and neither do some of the
embeddings that we tried.

Example:
--------
python protoscribe/semantics/filter_keyvec_embeddings_main.py \
  --concepts_file protoscribe/data/concepts/all_concepts_without_pos.txt \
  --original_embeddings_file ~/WordNetEmbeddings/outputs/wn2vec.txt \
  --wordnet_dir ~/projects/WordNetsAndSimilarities/WordNet-3.1/dict \
  --missing_concepts_map_file ~//missing_concepts_map.json \
  --output_file embeddings.txt \
  --logtostderr
"""

import collections
from collections.abc import Sequence
import json
import logging

from absl import app
from absl import flags
import numpy as np
from protoscribe.semantics.wordnet import utils
from protoscribe.semantics.wordnet import wordnet_utils

import glob
import os

_WORDNET_DIR = flags.DEFINE_string(
    "wordnet_dir", None,
    "Directory that contains downloaded Princeton WordNet that should have "
    "files named `data.pos`, where `pos` is a part-of-speech, e.g. `noun`.",
    required=True
)

_ORIGINAL_EMBEDDINGS_FILE = flags.DEFINE_string(
    "original_embeddings_file", None,
    "Text file containing the original WordNet embeddings (either downloaded "
    "or trained locally. The first line contains space-separated number of "
    "entries and embedding dimension. The rest of the lines contain "
    "space-separated words and their corresponding embedding vectors.",
    required=True
)

_CONCEPTS_FILE = flags.DEFINE_string(
    "concepts_file", None,
    "Text file containing new-line separated list of concepts we require.",
    required=True
)

_OUTPUT_FILE = flags.DEFINE_string(
    "output_file", None,
    "Resulting file for the selected embeddings in text format. The first "
    "column is the word followed by the embedding vectors.",
    required=True
)

_MISSING_CONCEPTS_MAP_FILE = flags.DEFINE_string(
    "missing_concepts_map_file", None,
    "Text file in JSON format containing a mapping from missing concepts to "
    "concepts that are found in the existing embedding file. Useful for "
    "filling in the missing concepts `by hand`."
)

_POS_MAPPING_SCHEME = flags.DEFINE_string(
    "pos_mapping_scheme", "wordnet",
    "Scheme for mapping concept POS tags to the format in the embeddings file.",
)

_EMBEDDING_POS = flags.DEFINE_boolean(
    "embedding_pos", False,
    "If enabled, assumes that the keys in the original embedding file "
    "include POS tag."
)

# Pointer symbols (relations) that we ignore when we look at other synsets.
_RELATIONS_TO_IGNORE = set([
    "!",  # Antonyms.
])

_MAX_SYNSETS = 50  # Maximum number of synsets to consider.

# Mapping between Protoscribe POS tags to the set of tags used in embeddings.
_CONCEPT_POS_FROM_PSCRIBE = {
    "wordnet": {
        "ADJ": "a",
        "ADV": "r",
        "NOUN": "n",
        "VERB": "v",
        "a": "a",
        "r": "r",
        "n": "n",
        "v": "v",
    },
}

# Mapping from Wordnet POS tags to Protoscribe tags.
_CONCEPT_POS_TO_PSCRIBE = wordnet_utils.WORDNET_POS_TO_PROTOSCRIBE


def _lookup_concept_wordnet_style(
    embeddings: dict[str, np.ndarray],
    concept_name: str,
    concept_pos: str
) -> np.ndarray | None:
  """Looks up the concept WordNet-style."""
  if concept_pos not in _CONCEPT_POS_FROM_PSCRIBE[_POS_MAPPING_SCHEME.value]:
    raise ValueError(f"Invalid POS tag: {concept_pos}")

  pos = _CONCEPT_POS_FROM_PSCRIBE[_POS_MAPPING_SCHEME.value][concept_pos]
  for idx in range(1, _MAX_SYNSETS):
    key = f"{concept_name}.{pos}.{idx:02d}"
    if key in embeddings:
      return embeddings[key]
  return None


def _lookup_concept(
    embeddings: dict[str, np.ndarray],
    concept_name: str,
    concept_pos: str
) -> np.ndarray | None:
  """Looks up concept in embeddings."""
  if _POS_MAPPING_SCHEME.value == "wordnet":
    return _lookup_concept_wordnet_style(embeddings, concept_name, concept_pos)
  else:
    raise ValueError("Unknown POS mapping scheme!")
  return None


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Load all the required concepts.
  logging.info("Loading concepts from %s ...", _CONCEPTS_FILE.value)
  concepts = set()
  with open(_CONCEPTS_FILE.value, "r") as f:
    for line in f:
      line = line.strip()
      if line:
        # line = line.split("_")[0]  # Remove POS if present.
        concepts.add(line)
  logging.info("Loaded %d concepts.", len(concepts))

  # Load the original embeddings.
  logging.info(
      "Loading original embeddings from %s ...", _ORIGINAL_EMBEDDINGS_FILE.value
  )
  embeddings = utils.load_keyvec_embeddings(_ORIGINAL_EMBEDDINGS_FILE.value)

  # Fill in the exact matches and find the missing embeddings.
  logging.info("Filling in the required embeddings ...")
  required_embeddings = {}
  missing_concepts = set()
  for concept in concepts:
    if concept in embeddings:
      required_embeddings[concept] = embeddings[concept]
    else:
      toks = concept.split("_")
      if toks == 1 or not _EMBEDDING_POS.value:  # No POS.
        missing_concepts.add((concept, None))
      else:
        emb_vec = _lookup_concept(embeddings, toks[0], toks[1])
        if emb_vec is not None:
          required_embeddings[concept] = emb_vec
        else:
          pos = _CONCEPT_POS_FROM_PSCRIBE[_POS_MAPPING_SCHEME.value][toks[1]]
          missing_concepts.add((toks[0], pos))
  logging.info(
      "Found %d exact hits, %d misses.",
      len(required_embeddings), len(missing_concepts)
  )

  # Load WordNet synsets by syntactic category.
  all_synsets = wordnet_utils.load_synsets_from_data(_WORDNET_DIR.value)

  # Step 1: Check all synsets and fill in the missing embeddings from words
  # that belong to the same synset for the given part of speech.
  word_count = collections.defaultdict(int)
  concepts_in_synsets = set()
  missing_embeddings = {}
  all_words = set()
  already_used_sources = set()
  for suffix in sorted(wordnet_utils.WORDNET_FILE_SUFFIX_TO_POS.keys()):
    pos = wordnet_utils.WORDNET_FILE_SUFFIX_TO_POS[suffix]
    synsets = all_synsets[pos]
    for offset in sorted(synsets.keys()):
      words, _ = synsets[offset]
      # Lookup all the embedding vectors for the words in this synset.
      emb_vecs = {}
      for word in words:
        word_count[word] += 1
        all_words.add((word, offset, pos))
        emb_vec = _lookup_concept(embeddings, word, pos)
        if emb_vec is not None:
          emb_vecs[word] = emb_vec

      # For any of the words in synset that are missing concepts, choose an
      # existing embedding for its synonym, if present. When choosing the
      # synonyms we make sure that no source embedding for the new concept is
      # used more than once.
      for word in words:
        if (word, pos) in missing_concepts:
          concepts_in_synsets.add(word)
          for other_word in sorted(emb_vecs.keys()):
            if (other_word, pos) in already_used_sources:
              continue
            pscribe_pos = _CONCEPT_POS_TO_PSCRIBE[pos]
            full_key = f"{word}_{pscribe_pos}"
            missing_embeddings[full_key] = emb_vecs[other_word]
            emb_vecs.pop(other_word)
            already_used_sources.add((other_word, pos))
            missing_concepts.remove((word, pos))
            break

  logging.info(
      "Collected counts for %d unique words, total count is %d.",
      len(word_count), len(all_words)
  )
  logging.info(
      "Found %d missing concepts in synsets. Filled %d missing embeddings, %d "
      "concepts still missing.",
      len(concepts_in_synsets), len(missing_embeddings), len(missing_concepts)
  )
  required_embeddings.update(missing_embeddings)

  # Step 2: Follow the pointers to other synsets one level down.
  # TODO: This is not perfect as we're not following the pointers all
  # the way to the leaf notes, but does the job for most of the missing
  # concepts.
  logging.info("Looking for extra concepts by following pointers ...")
  num_extra = 0
  for suffix in sorted(wordnet_utils.WORDNET_FILE_SUFFIX_TO_POS.keys()):
    pos = wordnet_utils.WORDNET_FILE_SUFFIX_TO_POS[suffix]
    synsets = all_synsets[pos]
    for offset in sorted(synsets.keys()):
      words, targets = synsets[offset]
      for target_synset, relation_type, target_pos in targets:
        if relation_type in _RELATIONS_TO_IGNORE:
          continue
        target_words = all_synsets[target_pos][target_synset][0]
        expanded_words = [(word, pos) for word in words] + [
            (word, target_pos) for word in target_words
        ]

        # Similar to Step 1, first fill in the existing embeddings for
        # the words in the expanded list.
        emb_vecs = {}
        expanded_words = sorted(set(expanded_words))
        for word, word_pos in expanded_words:
          emb_vec = _lookup_concept(embeddings, word, word_pos)
          if (
              emb_vec is not None and
              (word, word_pos) not in already_used_sources
          ):
            emb_vecs[word] = emb_vec

        # For the missing concepts choose the synonym so that no chosen
        # embedding for the new concept is used more than once.
        for word, word_pos in expanded_words:
          if (word, word_pos) in missing_concepts and emb_vecs:
            for other_word in sorted(emb_vecs.keys()):
              if (other_word, word_pos) in already_used_sources:
                continue
              pscribe_pos = _CONCEPT_POS_TO_PSCRIBE[word_pos]
              full_key = f"{word}_{pscribe_pos}"
              required_embeddings[full_key] = emb_vecs[other_word]
              emb_vecs.pop(other_word)
              missing_concepts.remove((word, word_pos))
              already_used_sources.add((other_word, word_pos))
              num_extra += 1
              break

  logging.info(
      "Filled in %d extra concepts. Missing %d concepts: %s",
      num_extra, len(missing_concepts), sorted(missing_concepts)
  )

  # Step 3: Attempt to fill the remaining missing concepts using a manual
  # mapping to existing embeddings.
  missing_map = {}
  if _MISSING_CONCEPTS_MAP_FILE.value:
    logging.info(
        "Loading missing concepts map from %s ...",
        _MISSING_CONCEPTS_MAP_FILE.value
    )
    with open(_MISSING_CONCEPTS_MAP_FILE.value) as f:
      missing_map = json.load(f)
  for missing in sorted(missing_map.keys()):
    concept = missing_map[missing]
    if concept not in embeddings:
      raise ValueError(f"Concept `{concept}` embedding not found!")
    if missing not in required_embeddings:
      required_embeddings[missing] = embeddings[concept]
      missing_concepts.remove(missing)
  if missing_concepts:
    logging.info("Final missing concepts: %s", sorted(missing_concepts))

  # Save selected embeddings.
  logging.info(
      "Saving %d embeddings to %s ...",
      len(required_embeddings), _OUTPUT_FILE.value
  )
  with open(_OUTPUT_FILE.value, mode="w") as f:
    for word in sorted(required_embeddings.keys()):
      emb_vec = [str(v) for v in required_embeddings[word].tolist()]
      f.write("%s %s\n" % (word, " ".join(emb_vec)))


if __name__ == "__main__":
  app.run(main)
