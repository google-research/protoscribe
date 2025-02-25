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
  --split test \
  --stroke_normalization_type "z-standardize" \
  --stroke_tokenizer_vocab_file protoscribe/data/glyphs/tokenizer/generic/vocab2048_normalized_sketchrnn.npy \
  --fast_stroke_tokenizer \
  --output_dir /tmp/tmp \
  --logtostderr
"""

from collections.abc import Sequence
import itertools
import logging
import os

from absl import app
from absl import flags
import ml_collections
import numpy as np
from protoscribe.corpus.reader import dataset_defaults as ds_lib
from protoscribe.corpus.reader import tasks as tasks_lib
from protoscribe.sketches.utils import stroke_stats as norm_lib
from protoscribe.sketches.utils import stroke_tokenizer as tokenizer_lib
from protoscribe.sketches.utils import stroke_utils as strokes_lib
import t5
import tensorflow as tf

Array = np.ndarray

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

_FAST_STROKE_TOKENIZER = flags.DEFINE_bool(
    "fast_stroke_tokenizer", False,
    "Use fast tokenization (only applies if tokenization is enabled above)."
)

_TF_STROKE_TOKENIZER = flags.DEFINE_bool(
    "tf_stroke_tokenizer", False,
    "If enabled, use TF tokenizer rather than Jax-based one."
)

_STROKE_RANDOM_SCALE_FACTOR = flags.DEFINE_float(
    "stroke_random_scale_factor", 0.,
    "Random stretch factor for sketch."
)

_SAVE_SVG = flags.DEFINE_bool(
    "save_svg", False,
    "Save SVGs in addition to PNGs."
)

_TASK_NAME = "sampler"


def _polylines_summary(polylines: list[Array]) -> str:
  """Summarizes the given polylines."""
  return "-".join([str(len(strokes)) for strokes in polylines])


def _save_sketch(
    sketch_3_or_5: Array,
    sketch_id: int,
    sketch_name: str
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

  # Save regular raster image.
  image = strokes_lib.polylines_to_raster_image(polylines)
  output_file = os.path.join(_OUTPUT_DIR.value, f"{sketch_name}.png")
  logging.info(
      "[%d] [%s] Saving %s (%d polylines, strokes: %s) ...",
      sketch_id, sketch_name, output_file,
      len(polylines), _polylines_summary(polylines)
  )
  image.save(output_file)

  # Save SVG.
  if _SAVE_SVG.value:
    output_file = os.path.join(_OUTPUT_DIR.value, f"{sketch_name}.svg")
    logging.info("[%d] [%s] Saving %s ...", sketch_id, sketch_name, output_file)
    strokes_lib.stroke3_strokes_to_svg_file(sketch3, output_file)


def _tokenize_and_reconstruct(
    normalized_sketch5: Array,
    glyph_affiliations: Array,
    tokenizer: tokenizer_lib.StrokeTokenizer
) -> Array:
  """Checks tokenization/detokenization.

  Note that the tokenizer is trained on normalized sketches. Sketch
  needs de-normalization before saving.

  Args:
    normalized_sketch5: Sketch in stroke5 format.
    glyph_affiliations: Glyph affiliations for each stroke.
    tokenizer: Tokenizer object.

  Returns:
    Reconstructed strokes in stroke3 format.
  """
  sketch3 = strokes_lib.stroke5_to_stroke3(normalized_sketch5)
  if _FAST_STROKE_TOKENIZER.value:
    if not _TF_STROKE_TOKENIZER.value:
      tokens = tokenizer.encode(sketch3)
    else:
      tokens, _, _ = tokenizer.tf_encode(
          tf.convert_to_tensor(sketch3, dtype=tf.float32),
          tf.convert_to_tensor(glyph_affiliations, dtype=tf.int32)
      )
      tokens = tokens.numpy()
  else:
    tokens = tokenizer.slow_encode(sketch3)
  sketch3 = tokenizer.decode(tokens)
  return np.float32(sketch3)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  config = ml_collections.ConfigDict()
  config.max_stroke_sequence_length = _MAX_STROKE_SEQUENCE_LENGTH.value
  config.stroke_normalization_type = _STROKE_NORMALIZATION_TYPE.value
  config.stroke_random_scale_factor = _STROKE_RANDOM_SCALE_FACTOR.value

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

  examples = itertools.islice(ds.as_numpy_iterator(), _SAMPLE_SIZE.value)
  for i, features in enumerate(examples):
    if "text.text" not in features:
      raise ValueError(f"[{i}] Bad dataset: text expected!")
    text = features["text.text"].decode("utf-8")
    text = text.replace(" ", "_")
    if "sketch.glyph_affiliations.ids" not in features:
      raise ValueError(f"[{i}] Bad dataset: No glyph affiliations!")
    glyph_affiliations = features["sketch.glyph_affiliations.ids"]

    # Read the strokes (stroke-5 format). This are normalized by the corpus
    # reader.
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
      normalized_sketch3 = _tokenize_and_reconstruct(
          normalized_sketch5, glyph_affiliations, tokenizer
      )
      sketch3_for_display = normalized_sketch3
      if norm_lib.should_normalize_strokes(config):
        sketch3_for_display = norm_lib.denormalize_strokes_array(
            config, sketch_stroke_stats, normalized_sketch3
        )

      vocab_size = tokenizer.codebook.shape[0]
      sketch_name = f"{text}_tok{vocab_size}"
      _save_sketch(
          sketch_3_or_5=sketch3_for_display,
          sketch_id=i,
          sketch_name=sketch_name
      )


if __name__ == "__main__":
  app.run(main)
