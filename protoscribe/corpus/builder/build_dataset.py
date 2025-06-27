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

"""Helper tool for dataset building.

Assembles the core language resources followed by the generation of dataset
splits in `tf.train.Example` format for training, validation and test. More
specifically, following artifacts for the generation of accounting texts are
created:

  - Prerequisites for accounting document generation including
    pronunciation lexicons (created by text generator).
  - Validation and test text data in TSV format (created by text generator).
  - The actual data for model training (train, validation and test splits) and
    inference (created by corpus builder Beam pipeline).

The resulting data residing in the specified output directory is used to train
the models and perform the inference.

This is essentially a shell script written in Python primarily to benefit from
absl flags support. To keep things simple this tool has minimal code
dependencies, we spawn individual Protoscribe builders as separate processes
rather than depending on individual libraries.
"""

import logging
import os
from typing import Any

from absl import flags
from protoscribe.utils import file_utils
from protoscribe.utils import subprocess_utils

import glob
import os

_RESOURCE_DIR = file_utils.RESOURCE_DIR
_TEXT_GENERATOR = f"{_RESOURCE_DIR}/texts/generate"
_PHONETIC_EMBEDDINGS_BUILDER = (
    f"{_RESOURCE_DIR}/language/phonology/build_phonetic_embeddings"

)
_CORPUS_BUILDER_PIPELINE = (
    f"{_RESOURCE_DIR}/corpus/builder/corpus_builder_beam"
)

_MAX_LOCAL_WORKERS = flags.DEFINE_integer(
    "max_local_workers", 1,
    "Maximum number of local worker threads."
)

# --------------------------------------
# Protoscribe language generation flags:
# --------------------------------------

_ADMINISTRATIVE_CATEGORIES = flags.DEFINE_string(
    "administrative_categories",
    f"{file_utils.SRC_DIR}/data/concepts/administrative_categories.txt",
    "Path to administrative categories to use."
)

_NON_ADMINISTRATIVE_CATEGORIES = flags.DEFINE_string(
    "non_administrative_categories",
    f"{file_utils.SRC_DIR}/data/concepts/non_administrative_categories.txt",
    "Path to non-administrative categories to use."
)

_CONCEPT_SPELLINGS = flags.DEFINE_string(
    "concept_spellings", "",
    "Path to spellings of concept/words if available."
)

_GENERATE_LANGUAGE = flags.DEFINE_bool(
    "generate_language", True,
    "If enabled, generates all the necessary files for the language."
)

_EXCLUDE_CONCEPTS_FILE = flags.DEFINE_string(
    "exclude_concepts_file", "",
    "Text file containing new-line separated lists of concepts one needs to "
    "exclude from the files provided by '--concepts' and '--unseen_concepts'."
)

_MAX_HOMOPHONY = flags.DEFINE_integer(
    "max_homophony", 5,
    "Maximum amount of homophony."
)

_MORPHEME_SHAPE = flags.DEFINE_enum(
    "morpheme_shape", "MORPHEME_CORE",
    [
        "MORPHEME_CORE",
        "MORPHEME_CORE_SESQUI",
        "MORPHEME_CORE_DI",
        "MORPHEME_CORE_TEMPLATIC",
        "MORPHEME_CVCC_MONO"
    ],
    "Morpheme shape. For available values see "
    f"{file_utils.SRC_DIR}/texts/common_configs.py."
)

_NUMBER_CONFIG = flags.DEFINE_string(
    "number_config", "number_config_sg.textproto",
    "Number generation configuration."
)

_PHONOLOGY_LANGUAGES = flags.DEFINE_string(
    "phonology_languages", "",
    "Select the phonemes from a subset of phonologies corresponding to this "
    "list of comma-separated ISO 639-3 language codes. By default the phonemes "
    "are sampled from any language supported by PHOIBLE. If non-None, this "
    "settings overrides the languages defined in the prosodic template proto."
)

_NUMBER_OF_PHONEMES = flags.DEFINE_integer(
    "number_of_phonemes", -1,
    "Overrides ProsodyTemplate's number_of_phonemes."
)

_NUM_TEXTS = flags.DEFINE_integer(
    "num_texts", 500000,
    "Number of accounting documents to generate."
)

_MAX_COMMODITY = flags.DEFINE_integer(
    "max_commodity", 10,
    "Maximum cardinal representing the number of commodities."
)

_NUM_TEST_EXAMPLES = flags.DEFINE_integer(
    "num_test_examples", 3000,
    "Number of test examples of non-administrative categories."
    "Approximately ${FLAGS_max_commodity} * number of non-admin concepts."
)

_PROBABILITY_OF_SUPERCATEGORY_GLYPH = flags.DEFINE_float(
    "probability_of_supercategory_glyph", 0.25,
    "Probability of generating a supercategory glyph if one is available."
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", None,
    "Output directory for the dataset splits.",
    required=True
)

# ---------------------------------------
# Protoscribe Beam pipeline runner flags:
# ---------------------------------------

_FILE_DATA_FORMAT = "tfr"
_FILE_PREFIX_SPEC_SUFFIX = ""

_TRAIN_FILE_PREFIX_SPEC = flags.DEFINE_string(
    "train_file_prefix_spec",
    f"train.{_FILE_DATA_FORMAT}{_FILE_PREFIX_SPEC_SUFFIX}",
    "File pattern for the resulting records for the training split."
)

_VALIDATION_FILE_PREFIX_SPEC = flags.DEFINE_string(
    "validation_file_prefix_spec",
    f"validation.{_FILE_DATA_FORMAT}{_FILE_PREFIX_SPEC_SUFFIX}",
    "File pattern for the resulting records for the validation split."
)

_TEST_FILE_PREFIX_SPEC = flags.DEFINE_string(
    "test_file_prefix_spec",
    f"test.{_FILE_DATA_FORMAT}{_FILE_PREFIX_SPEC_SUFFIX}",
    "File pattern for the resulting records for the test split."
)

_INCLUDE_UNSEEN_CONCEPTS_IN_TRAIN = flags.DEFINE_bool(
    "include_unseen_concepts_in_train", False,
    "Whether to include the unseen concepts in training."
)

_INCLUDE_GLYPH_VARIANTS = flags.DEFINE_bool(
    "include_glyph_variants", False,
    "Whether to include glyph variants."
)

_EXTENSION_GLYPHS_SVG_DIR = flags.DEFINE_string(
    "extension_glyphs_svg_dir", "",
    "Directory housing discovered extension glyphs in SVG format."
)

_PREFER_CONCEPT_SVG = flags.DEFINE_bool(
    "prefer_concept_svg", False,
    "Attempt to lookup SVG by concept name first, by glyph names second."
)


def _output_file(filename: str) -> str:
  return os.path.join(_OUTPUT_DIR.value, filename)


def _language_file(filename: str) -> str:
  return os.path.join(_OUTPUT_DIR.value, "language", filename)


def _excluded_concepts_path() -> str:
  return (
      _output_file(
          "excluded_concepts.txt"
      ) if  _EXCLUDE_CONCEPTS_FILE.value else ""
  )


def _common_language_args() -> list[Any]:
  """Returns common parameters for generating the language."""
  return [
      "--affix_lexicon", _language_file("affixes.tsv"),
      "--main_lexicon", _language_file("lexicon.tsv"),
      "--morphology_params", _language_file("morphology_params.textproto"),
      "--number_lexicon", _language_file("number_lexicon.tsv"),
      "--number_config_file", _language_file(_NUMBER_CONFIG.value),
      "--number_phonological_rules", "",
      "--word_phonological_rule_templates", "",
  ]


def _prepare_language_components() -> None:
  """Prepares all the necessary language components under output directory."""

  # Create language directory.
  output_dir = _OUTPUT_DIR.value
  language_dir = os.path.join(output_dir, "language")
  logging.info("Language directory: %s", language_dir)
  if not os.path.exists(language_dir):
    os.makedirs(
        language_dir, exist_ok=True
    )

  # Copy these files that are copyable verbatim.
  file_utils.copy_src_file("texts/configs", _NUMBER_CONFIG.value, language_dir)
  file_utils.copy_full_path(_ADMINISTRATIVE_CATEGORIES.value, output_dir)
  file_utils.copy_full_path(_NON_ADMINISTRATIVE_CATEGORIES.value, output_dir)
  all_concept_paths = [
      _output_file("administrative_categories.txt"),
      _output_file("non_administrative_categories.txt"),
  ]
  excluded_concepts_path = _excluded_concepts_path()
  if excluded_concepts_path:
    file_utils.copy_file(_EXCLUDE_CONCEPTS_FILE.value, excluded_concepts_path)

  # Generate the actual core language and embeddings files.
  common_language_args = _common_language_args()
  if _GENERATE_LANGUAGE.value:
    logging.info("Generating new language in %s ...", language_dir)
    subprocess_utils.run_subprocess(
        _TEXT_GENERATOR,
        args=common_language_args + [
            "--generate_lexical_resources", "true",
            "--concepts", ",".join(all_concept_paths),
            "--max_homophony", _MAX_HOMOPHONY.value,
            "--morpheme_shape", _MORPHEME_SHAPE.value,
            "--number_of_phonemes", _NUMBER_OF_PHONEMES.value,
            "--phonology_languages", _PHONOLOGY_LANGUAGES.value,
        ]
    )
    logging.info("Generating phonetic embeddings in %s ...", language_dir)
    subprocess_utils.run_subprocess(
        _PHONETIC_EMBEDDINGS_BUILDER,
        args=[
            "--main_lexicon", _language_file("lexicon.tsv"),
            "--number_lexicon", _language_file("number_lexicon.tsv"),
            "--norm_order", 2,
            "--embeddings", _language_file("phonetic_embeddings.tsv"),
        ]
    )
  else:
    logging.info("Will use previously generated language in %s.", language_dir)

  # Generates in-domain held-out data.
  logging.info("Generating in-domain held-out data ...")
  subprocess_utils.run_subprocess(
      _TEXT_GENERATOR,
      args=common_language_args + [
          "--concepts", _output_file("administrative_categories.txt"),
          "--exclude_concepts_file", excluded_concepts_path,
          "--concept_spellings", _CONCEPT_SPELLINGS.value,
          "--output_texts", _output_file("in_domain_test_examples.tsv"),
          "--num_texts", _NUM_TEST_EXAMPLES.value,
          "--max_commodity", _MAX_COMMODITY.value,
          "--probability_of_supercategory_glyph",
          _PROBABILITY_OF_SUPERCATEGORY_GLYPH.value,
      ]
  )

  # Generates test data.
  logging.info("Generating test data ...")
  subprocess_utils.run_subprocess(
      _TEXT_GENERATOR,
      args=common_language_args + [
          "--concepts", _output_file("non_administrative_categories.txt"),
          "--exclude_concepts_file", excluded_concepts_path,
          "--concept_spellings", _CONCEPT_SPELLINGS.value,
          "--output_texts", _output_file("test_examples.tsv"),
          "--num_texts", _NUM_TEST_EXAMPLES.value,
          "--max_commodity", _MAX_COMMODITY.value,
          "--probability_of_supercategory_glyph",
          _PROBABILITY_OF_SUPERCATEGORY_GLYPH.value,
      ]
  )


def _corpus_builder_args() -> dict[str, Any]:
  """Builds the argument dictionary for the corpus builder pipeline."""

  # Convert common language arguments to a dictionary sanitizing the long option
  # prefixes if found.
  common_language_args = [
      arg.replace("--", "") for arg in _common_language_args()
  ]
  args_it = iter(common_language_args)
  common_language_args = dict(zip(args_it, args_it))

  args = dict(
      # Language generation flags.
      # --------------------------
      **common_language_args,
      morpheme_shape=_MORPHEME_SHAPE.value,
      probability_of_supercategory_glyph=
      _PROBABILITY_OF_SUPERCATEGORY_GLYPH.value,
      num_texts=_NUM_TEXTS.value,
      max_commodity=_MAX_COMMODITY.value,
      phonetic_embeddings=_language_file("phonetic_embeddings.tsv"),
      #
      # Core corpus builder flags.
      # --------------------------
      concepts=_output_file("administrative_categories.txt"),
      unseen_concepts=_output_file("non_administrative_categories.txt"),
      exclude_concepts_file=_excluded_concepts_path(),
      concept_spellings=_CONCEPT_SPELLINGS.value,
      output_train_file_prefix=_output_file(_TRAIN_FILE_PREFIX_SPEC.value),
      output_validation_file_prefix=_output_file(
          _VALIDATION_FILE_PREFIX_SPEC.value
      ),
      output_test_file_prefix=_output_file(_TEST_FILE_PREFIX_SPEC.value),
      glyph_vocab_file=_output_file("glyph_vocab.json"),
      concept_vocab_file=_output_file("concept_vocab.json"),
      sketch_stroke_stats_file=_output_file("sketch_stroke_stats.json"),
      include_unseen_concepts_in_train=_INCLUDE_UNSEEN_CONCEPTS_IN_TRAIN.value,
      include_glyph_variants=_INCLUDE_GLYPH_VARIANTS.value,
      extension_glyphs_svg_dir=_EXTENSION_GLYPHS_SVG_DIR.value,
      prefer_concept_svg=_PREFER_CONCEPT_SVG.value,
  )
  return args


def _run_corpus_builder_local(args: dict[str, Any]) -> None:
  """Runs the corpus builder pipeline."""
  logging.info("Running corpus builder pipeline ...")
  builder_args = []
  for arg_name, arg_value in args.items():
    builder_args.extend([f"--{arg_name}", arg_value])

  subprocess_utils.run_subprocess(_CORPUS_BUILDER_PIPELINE, args=builder_args)


def _run_corpus_builder() -> None:
  """Runs corpus builder pipeline."""
  args = _corpus_builder_args()
  _run_corpus_builder_local(args)


def build_dataset() -> None:
  """Runs dataset building tools as standalone subprocesses."""

  _prepare_language_components()
  _run_corpus_builder()
