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

r"""A tool for sampling from the dataset.

This focuses on sampling the glyph representations and then testing that various
associated reconstructions in form of raster images and vector graphics make
sense.

Example:
--------
DATASET_DIR=...
python protoscribe/corpus/tools/sample_from_corpus_main.py \
  --dataset_dir "${DATASET_DIR}" \
  --sample_size 100 \
  --max_stroke_sequence_length 1000 \
  --stroke_normalization_type "z-standardize" \
  --stroke_tokenizer_vocab_file protoscribe/data/glyphs/tokenizer/generic/vocab2048_normalized_sketchrnn.npy \
  --stroke_tokenizer_mode all \
  --output_dir /tmp/tmp \
  --logtostderr
"""

from collections.abc import Sequence
import itertools
import logging
import os
from typing import Any, Iterable

from absl import app
from absl import flags
import ml_collections
import numpy as np
from protoscribe.corpus.reader import dataset_defaults as ds_lib
from protoscribe.corpus.reader import tasks as tasks_lib
from protoscribe.corpus.tools import sample_from_corpus
from protoscribe.sketches.utils import stroke_stats as norm_lib
from protoscribe.sketches.utils import stroke_tokenizer as tokenizer_lib
from protoscribe.sketches.utils import stroke_utils as strokes_lib
import t5
import tensorflow as tf

Array = np.ndarray
TokenizerMode = sample_from_corpus.TokenizerMode

_SAMPLE_SIZE = flags.DEFINE_integer(
    "sample_size", 10,
    "Number of sketches to randomly sample from the dataset."
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", None,
    "Output directory for dumping images.",
    required=True,
)

_SPLIT = flags.DEFINE_string(
    "split", "train",
    "Dataset split."
)

_MAX_STROKE_SEQUENCE_LENGTH = flags.DEFINE_integer(
    "max_stroke_sequence_length", 1_000,
    "Maximum number of the points in the stroke."
)

_STROKE_NORMALIZATION_TYPE = flags.DEFINE_enum(
    "stroke_normalization_type", "none",
    [
        "none",
        "z-standardize",
        "min-max",
        "mean-norm",
        "sketch-rnn",
        "det-covar"
    ],
    "Stroke normalization type."
)

_STROKE_TOKENIZER_VOCAB_FILE = flags.DEFINE_string(
    "stroke_tokenizer_vocab_file", None,
    "If supplied, tokenize and reconstruct the sketch."
)

_STROKE_TOKENIZER_MODE = flags.DEFINE_enum_class(
    "stroke_tokenizer_mode",
    default=TokenizerMode.SLOW,
    enum_class=TokenizerMode,
    help="Stroke tokenizer mode to use. This can be either the default slow "
    "mode, JAX or TensorFlow."
)

_STROKE_RANDOM_SCALE_FACTOR = flags.DEFINE_float(
    "stroke_random_scale_factor", 0.,
    "Random stretch factor for sketch."
)

_SAVE_SVG = flags.DEFINE_bool(
    "save_svg", False,
    "Save SVGs in addition to PNGs."
)

_FIND_CONCEPTS = flags.DEFINE_list(
    "find_concepts", [],
    "Only sample documents that relate to the given list of concepts."
)

_TASK_NAME = "sampler"


def _sample_examples(ds: tf.data.Dataset) -> Iterable[dict[str, Any]]:
  """Samples collection of examples from a dataset.

  Args:
    ds: Dataset instance.

  Returns:
    An iterable containing documents represented as feature dictionaries.

  Raises:
    ValueError if concept is not defined in one of the documents.
  """
  ds_iter = ds.as_numpy_iterator()
  if not _FIND_CONCEPTS.value:
    return itertools.islice(ds_iter, _SAMPLE_SIZE.value)

  logging.info(
      "Searching for %d concepts in %s ...",
      _SAMPLE_SIZE.value, _FIND_CONCEPTS.value
  )
  num_found = 0
  examples = []
  for doc_id, features in enumerate(ds_iter):
    if "concept.name" not in features:
      raise ValueError(
          f"Concept name not found in features for example {doc_id}"
      )
    concept = features["concept.name"].decode("utf-8")
    concept = concept.split("_")[0]
    if concept in _FIND_CONCEPTS.value:
      logging.info("[%d] Found %s in doc %d ...", num_found, concept, doc_id)
      num_found += 1
      examples.append(features)
      if num_found == _SAMPLE_SIZE.value:
        return examples

  return examples


def _polylines_summary(polylines: list[Array]) -> str:
  """Summarizes the given polylines."""
  return "-".join([str(len(strokes)) for strokes in polylines])


def _save_sketch(
    sketch_3_or_5: Array, sketch_id: int, sketch_name: str
) -> None:
  """Saves sketch with stroke-5 structure in various formats.

  Args:
    sketch_3_or_5: Array of shape (L, 3) or (L, 5) in either of sketch-3 or
      sketch-5 formats.
    sketch_id: Integer identifying the sketch.
    sketch_name: Name of the sketch.
  """
  sketch3 = (
      strokes_lib.stroke5_to_stroke3(sketch_3_or_5)
      if sketch_3_or_5.shape[1] == 5 else sketch_3_or_5
  )
  polylines = strokes_lib.stroke3_deltas_to_polylines(sketch3)
  polylines_summary = _polylines_summary(polylines)

  # Save stroke-3 format strokes to text.
  output_file = os.path.join(_OUTPUT_DIR.value, f"{sketch_name}_stroke3.txt")
  np.savetxt(output_file, sketch3, fmt="%.4f")

  # Save regular raster image.
  image = strokes_lib.polylines_to_raster_image(polylines)
  output_file = os.path.join(_OUTPUT_DIR.value, f"{sketch_name}.png")
  logging.info(
      "[%d] [%s] Saving %s (%d polylines, strokes: %s) ...",
      sketch_id, sketch_name, output_file, len(polylines), polylines_summary
  )
  image.save(output_file)

  # Save SVG.
  if _SAVE_SVG.value:
    output_file = os.path.join(_OUTPUT_DIR.value, f"{sketch_name}.svg")
    logging.info(
        "[%d] [%s] Saving %s (%d polylines, strokes: %s) ...",
        sketch_id, sketch_name, output_file, len(polylines), polylines_summary
    )
    strokes_lib.stroke3_strokes_to_svg_file(sketch3, output_file)


def _tokenize_and_reconstruct_to_file(
    config: ml_collections.FrozenConfigDict,
    normalized_sketch5: Array,
    glyph_affiliations: Array,
    sketch_stroke_stats: ds_lib.StrokeStats,
    tokenizer: tokenizer_lib.StrokeTokenizer,
    tokenizer_mode: TokenizerMode,
    doc_id: int,
    doc_text: str
) -> None:
  """Tokenizes, reconstructs and saves the result to file.

  Args:
    config: Configuraiton for tokenizer and normalizer.
    normalized_sketch5: Sketch in stroke5 format.
    glyph_affiliations: Glyph affiliations for each stroke.
    sketch_stroke_stats: Stroke statistics.
    tokenizer: Tokenizer object.
    tokenizer_mode: One of the three possible modes of tokenization.
      One of: slow (regular mode, default), Jax or TensorFlow.
    doc_id: ID of the current document.
    doc_text: Contents of the accounting document.
  """
  normalized_sketch3 = sample_from_corpus.tokenize_and_reconstruct(
      normalized_sketch5,
      glyph_affiliations=glyph_affiliations,
      tokenizer=tokenizer,
      tokenizer_mode=tokenizer_mode
  )
  sketch3_for_display = normalized_sketch3
  if norm_lib.should_normalize_strokes(config):
    sketch3_for_display = norm_lib.denormalize_strokes_array(
        config, sketch_stroke_stats, normalized_sketch3
    )

  vocab_size = tokenizer.codebook.shape[0]
  tokenizer_mode = tokenizer_mode.value.lower()
  sketch_name = f"{doc_text}_tok_{tokenizer_mode}_{vocab_size}"
  _save_sketch(
      sketch_3_or_5=sketch3_for_display,
      sketch_id=doc_id,
      sketch_name=sketch_name
  )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  config = ml_collections.FrozenConfigDict({
      "max_stroke_sequence_length": _MAX_STROKE_SEQUENCE_LENGTH.value,
      "stroke_normalization_type": _STROKE_NORMALIZATION_TYPE.value,
      "stroke_random_scale_factor": _STROKE_RANDOM_SCALE_FACTOR.value,
  })
  sketch_stroke_stats = ds_lib.get_sketch_stroke_stats(config)
  task_name = tasks_lib.register(
      task_name=_TASK_NAME,
      max_stroke_sequence_length=config.max_stroke_sequence_length,
      stroke_normalization_type=config.stroke_normalization_type,
      stroke_random_scale_factor=config.stroke_random_scale_factor
  )
  task = t5.data.TaskRegistry.get(task_name)
  ds = task.get_dataset(split=_SPLIT.value)

  tokenizer = None
  if _STROKE_TOKENIZER_VOCAB_FILE.value:
    tokenizer = tokenizer_lib.StrokeTokenizer(
        _STROKE_TOKENIZER_VOCAB_FILE.value, config.max_stroke_sequence_length
    )

  examples = _sample_examples(ds)
  for i, features in enumerate(examples):
    # Form name of the file.
    if "text.text" not in features:
      raise ValueError(f"[{i}] Bad dataset: text expected!")
    text = features["text.text"].decode("utf-8")
    text = text.replace(" ", "_")
    if "doc.id" not in features:
      raise ValueError(f"[{i}] Document ID not found!")
    doc_id = int(features["doc.id"])
    text = f"{text}_{doc_id}"

    if "sketch.glyph_affiliations.ids" not in features:
      raise ValueError(f"[{i}] Bad dataset: No glyph affiliations!")
    glyph_affiliations = features["sketch.glyph_affiliations.ids"]

    # Read the strokes (stroke-5 format). This are normalized by the corpus
    # reader. When creating the strokes-5 format the corpus reader inserts
    # explicit BOS and EOS vectors to start and end of the stroke sequence.
    if "strokes" not in features:
      raise ValueError(f"[{i}] Bad dataset: strokes expected!")
    normalized_sketch5 = features["strokes"]
    sketch5_for_display = normalized_sketch5
    if norm_lib.should_normalize_strokes(config):
      sketch5_for_display = norm_lib.denormalize_strokes_array(
          config, sketch_stroke_stats, normalized_sketch5
      )
    _save_sketch(
        sketch_3_or_5=sketch5_for_display, sketch_id=i, sketch_name=text
    )

    # If tokenizer is configured, test tokenization/detokenization process.
    if tokenizer:
      if _STROKE_TOKENIZER_MODE.value != TokenizerMode.ALL:
        _tokenize_and_reconstruct_to_file(
            config,
            normalized_sketch5,
            glyph_affiliations=glyph_affiliations,
            sketch_stroke_stats=sketch_stroke_stats,
            tokenizer=tokenizer,
            tokenizer_mode=_STROKE_TOKENIZER_MODE.value,
            doc_id=i,
            doc_text=text
        )
      else:
        for mode in [
            TokenizerMode.SLOW,
            TokenizerMode.JAX,
            TokenizerMode.TF
        ]:
          _tokenize_and_reconstruct_to_file(
              config,
              normalized_sketch5,
              glyph_affiliations=glyph_affiliations,
              sketch_stroke_stats=sketch_stroke_stats,
              tokenizer=tokenizer,
              tokenizer_mode=mode,
              doc_id=i,
              doc_text=text
          )


if __name__ == "__main__":
  app.run(main)
