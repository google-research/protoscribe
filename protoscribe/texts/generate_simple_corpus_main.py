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

r"""Helper tool for generating simple (textual-only) data for model training.

Example:
--------
Generates the initial data using defaults.

python protoscribe/texts/generate_simple_corpus_main.py \
  --dataset_dir /tmp/protoscribe \
  --logtostderr
"""

from collections.abc import Sequence
import logging

from absl import app
from absl import flags
from protoscribe.utils import file_utils
from protoscribe.utils import subprocess_utils

import glob
import os

_DATASET_DIR = flags.DEFINE_string(
    "dataset_dir", None,
    "Parent directory for the dataset.",
    required=True
)

_MAX_HOMOPHONY = flags.DEFINE_integer(
    "max_homophony", 5,
    "Maximum amount of homophony."
)

_NUMBER_CONFIG = flags.DEFINE_string(
    "number_config", "number_config_sg_du_pl.textproto",
    "Number generation configuration."
)

_NUM_SETS = flags.DEFINE_integer(
    "num_sets", 5,
    "Number of sets. of accounting texts to generate."
)

_NUM_TEXTS = flags.DEFINE_integer(
    "num_texts", 10_000,
    "Number of accounting documents to generate."
)

_MAX_COMMODITY = flags.DEFINE_integer(
    "max_commodity", 99,
    "Maximum cardinal representing the number of commodities."
)

_PROBABILITY_OF_SUPERCATEGORY_GLYPH = flags.DEFINE_float(
    "probability_of_supercategory_glyph", 0.25,
    "Probability of generating a supercategory glyph if one is available."
)

_SRC_DIR = file_utils.SRC_DIR
_RESOURCE_DIR = file_utils.RESOURCE_DIR
_TEXT_GENERATOR = f"{_RESOURCE_DIR}/texts/generate"
_VOCAB_BUILDER = f"{_RESOURCE_DIR}/texts/make_vocab_files"
_PHONETIC_EMBEDDINGS_BUILDER = (
    f"{_RESOURCE_DIR}/language/phonology/build_phonetic_embeddings"
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Prepare output directories.
  initial_dir = f"{_DATASET_DIR.value}/initial_texts"
  params_dir = f"{initial_dir}/params"
  concepts_dir = f"{_SRC_DIR}/data/concepts"
  if not os.path.exists(params_dir):
    os.makedirs(params_dir, exist_ok=True)

  # Generate lexicon resources.
  logging.info("Generating the lexicon with ALL concepts in %s ...", params_dir)
  concept_files = [
      f"{concepts_dir}/administrative_categories.txt",
      f"{concepts_dir}/non_administrative_categories.txt",
  ]
  number_config_file = f"{_SRC_DIR}/texts/configs/{_NUMBER_CONFIG.value}"
  subprocess_utils.run_subprocess(
      _TEXT_GENERATOR,
      args=[
          "--generate_lexical_resources", "true",
          "--concepts", ",".join(concept_files),
          "--affix_lexicon", f"{params_dir}/affixes.tsv",
          "--main_lexicon", f"{params_dir}/lexicon.tsv",
          "--morphology_params", f"{params_dir}/morphology_params.textproto",
          "--number_lexicon", f"{params_dir}/number_lexicon.tsv",
          "--number_phon_rules", f"{params_dir}/number_phon_rules.far",
          "--phon_rules", f"{params_dir}/phon_rules.far",
          "--number_config_file", number_config_file,
          "--max_homophony", _MAX_HOMOPHONY.value,
      ]
  )

  # Now generate the accounting texts.
  output_dir = f"{initial_dir}/output"
  if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
  for set_idx in range(_NUM_SETS.value):
    logging.info("Generating accounting texts set %d ...", set_idx)
    subprocess_utils.run_subprocess(
        _TEXT_GENERATOR,
        args=[
            "--concepts", ",".join(concept_files),
            "--affix_lexicon", f"{params_dir}/affixes.tsv",
            "--main_lexicon", f"{params_dir}/lexicon.tsv",
            "--morphology_params", f"{params_dir}/morphology_params.textproto",
            "--number_lexicon", f"{params_dir}/number_lexicon.tsv",
            "--number_phon_rules", f"{params_dir}/number_phon_rules.far",
            "--phon_rules", f"{params_dir}/phon_rules.far",
            "--number_config_file", number_config_file,
            "--num_texts", _NUM_TEXTS.value,
            "--probability_of_supercategory_glyph",
            _PROBABILITY_OF_SUPERCATEGORY_GLYPH.value,
            "--max_commodity", _MAX_COMMODITY.value,
            "--output_texts", f"{output_dir}/accounts_{set_idx}.txt",
        ]
    )

  # Create the vocabulary files.
  logging.info("Making the vocabulary files ...")
  subprocess_utils.run_subprocess(
      _VOCAB_BUILDER,
      args=[
          "--texts_glob", f"{output_dir}/accounts_[0-{_NUM_SETS.value}].txt",
          "--glyph_syms", f"{params_dir}/glyphs.syms",
          "--word_syms", f"{params_dir}/words.syms",
      ]
  )

  # Build phonetic embeddings.
  logging.info("Building phonetic embeddings ...")
  subprocess_utils.run_subprocess(
      _PHONETIC_EMBEDDINGS_BUILDER,
      args=[
          "--main_lexicon", f"{params_dir}/lexicon.tsv",
          "--number_lexicon", f"{params_dir}/number_lexicon.tsv",
          "--embeddings", f"{params_dir}/phonetic_embeddings.tsv",
      ]
  )

  # Copy semantic embeddings.
  logging.info("Copying semantic embeddings ...")
  sem_dir = f"{_RESOURCE_DIR}/data/semantics/bnc"
  for filename in ["embeddings.txt", "numbers.txt"]:
    src_file = file_utils.resource_path(f"{sem_dir}/{filename}")
    file_utils.copy_full_path(src_file, params_dir)

  logging.info("Initial corpus generated in %s.", initial_dir)


if __name__ == "__main__":
  app.run(main)
