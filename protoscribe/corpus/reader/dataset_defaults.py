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

"""Command-line flags and helper I/O methods."""

import logging
import os

from absl import flags
import ml_collections
from protoscribe.glyphs import glyph_vocab as glyph_lib
from protoscribe.sketches.utils import stroke_stats as stroke_stats_lib
from protoscribe.sketches.utils import stroke_tokenizer as tokenizer_lib
from protoscribe.utils import file_utils

import glob
import os

StrokeStats = stroke_stats_lib.FinalStrokeStats
StrokeTokenizer = tokenizer_lib.StrokeTokenizer

_DEFAULT_DATASET_DIR = None
_DEFAULT_DATASET_FORMAT = "tfr"

DATASET_DIR = flags.DEFINE_string(
    "dataset_dir",
    _DEFAULT_DATASET_DIR,
    "Dataset root directory. This may be overriden for small datasets if we "
    "want to have the dataset locally for certain postprocessing or if we want "
    "to point the training at the different version of the data."
)

DATASET_FORMAT = flags.DEFINE_string(
    "dataset_format",
    _DEFAULT_DATASET_FORMAT,
    "Default dataset format: `tfr` (`TFRecord`)"
)

# The dictionary containing corpus stroke statistics used for data
# normalization.
_SKETCH_STROKE_STATS_FILE = "sketch_stroke_stats.json"

# The glyph vocabulary file.
_GLYPH_VOCAB_FILE = "glyph_vocab.json"

# Concepts vocabulary file.
_CONCEPTS_VOCAB_FILE = "concept_vocab.json"

# Main lexicon
_MAIN_LEXICON_FILE = "language/lexicon.tsv"

# Number lexicon
_NUMBER_LEXICON_FILE = "language/number_lexicon.tsv"

# Phonetic embeddings
_PHONETIC_EMBEDDINGS = "language/phonetic_embeddings.tsv"

# Pretrained stroke tokenizer. Because pretraining takes a long time it is a
# separate step from dataset builder.
_STROKE_TOKENIZER_ROOT_DIR = (
    "protoscribe/data/glyphs/tokenizer/generic"
)


def dataset_dir(read_ahead: bool = True) -> str:
  # Based on past experience readahead is sometimes crucial to make sure the
  # TPUs don't get starved. Value based on TFDS code.
  readahead_prefix = "/readahead/256M" if read_ahead else ""
  full_dataset_dir = f"{readahead_prefix}{DATASET_DIR.value}"
  logging.info("Dataset directory is %s", full_dataset_dir)
  return full_dataset_dir


def glyph_vocab_file() -> str:
  """Glyph vocabulary file."""
  return os.path.join(dataset_dir(read_ahead=False), _GLYPH_VOCAB_FILE)


def _concept_vocab_file() -> str:
  """Concepts vocabulary file."""
  return os.path.join(dataset_dir(read_ahead=False), _CONCEPTS_VOCAB_FILE)


def main_lexicon_file() -> str:
  """Main lexicon."""
  return os.path.join(dataset_dir(read_ahead=False), _MAIN_LEXICON_FILE)


def number_lexicon_file() -> str:
  """Number lexicon."""
  return os.path.join(dataset_dir(read_ahead=False), _NUMBER_LEXICON_FILE)


def phonetic_embeddings_file() -> str:
  """Path to precomputed phonetic embeddings."""
  return os.path.join(dataset_dir(read_ahead=False), _PHONETIC_EMBEDDINGS)


def stroke_tokenizer_file(file_name: str) -> str:
  """Stroke quantizer file for tokenizing sketches."""
  return file_utils.resource_path(
      os.path.join(_STROKE_TOKENIZER_ROOT_DIR, file_name)
  )


def sketch_stroke_stats_file() -> str:
  """Stroke statistics."""
  return os.path.join(dataset_dir(read_ahead=False), _SKETCH_STROKE_STATS_FILE)


def get_stroke_tokenizer(
    config: ml_collections.FrozenConfigDict
) -> StrokeTokenizer | None:
  """Helper method for constructing a tokenizer from configuration."""
  stroke_tokenizer = None
  if "stroke_tokenizer" in config:
    stroke_tokenizer = StrokeTokenizer(
        stroke_tokenizer_file(config.stroke_tokenizer.vocab_filename),
        config.max_stroke_sequence_length
    )
  return stroke_tokenizer


def get_sketch_stroke_stats(
    config: ml_collections.FrozenConfigDict
) -> StrokeStats:
  """Helper method for constructing sketch stroke statistics."""
  return stroke_stats_lib.load_stroke_stats(config, sketch_stroke_stats_file())


def get_glyph_vocab() -> glyph_lib.GlyphVocab:
  """Helper method for loading the default glyph vocabulary."""
  glyph_vocab_path = glyph_vocab_file()
  logging.info("Reading glyph vocab from %s ...", glyph_vocab_path)
  with open(glyph_vocab_path, mode="r"):
    glyph_vocab = glyph_lib.load_glyph_vocab(glyph_vocab_path)
    logging.info("Loaded %d glyphs.", len(glyph_vocab))
  return glyph_vocab
