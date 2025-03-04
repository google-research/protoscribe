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

"""Utilities for parsing stroke representations."""

from typing import Optional

import ml_collections
import numpy as np
from protoscribe.glyphs import glyph_vocab as glyph_lib
from protoscribe.sketches.utils import stroke_stats as stroke_stats_lib
from protoscribe.sketches.utils import stroke_tokenizer as tokenizer_lib
from protoscribe.sketches.utils import stroke_utils
import tensorflow as tf

StrokeStats = stroke_stats_lib.FinalStrokeStats

_END_OF_STROKE = stroke_utils.END_OF_STROKE
_FINAL_PEN_STATE = 1


def _random_scale_strokes(
    scale_factor: float, x: tf.Tensor, y: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
  """Augment data by stretching x and y axis randomly."""
  if scale_factor != 0.:
    x_scale_factor = (np.random.random() - 0.5) * 2 * scale_factor + 1.0
    y_scale_factor = (np.random.random() - 0.5) * 2 * scale_factor + 1.0
    return x * x_scale_factor, y * y_scale_factor
  else:
    return x, y


def _pad_strokes(
    inputs: tf.Tensor, max_stroke_sequence_length: int
) -> tf.Tensor:
  """Pads the inputs as necessary."""
  pad_size = max_stroke_sequence_length - tf.shape(inputs)[0]
  if len(tf.shape(inputs)) == 2:
    paddings = [[0, pad_size], [0, 0]]
  else:  # Assume 1D inputs.
    paddings = [[0, pad_size]]
  return tf.pad(inputs, paddings)


def _points_to_strokes3(
    config: ml_collections.FrozenConfigDict,
    stroke_stats: StrokeStats,
    x_strokes: tf.Tensor,
    y_strokes: tf.Tensor,
    is_training: bool,
    stroke_random_scale_factor: float,
) -> tf.Tensor:
  """Generates stroke-3 tensor."""
  pen_state = tf.cast(x_strokes == _END_OF_STROKE, dtype=tf.float32)
  pen_state = tf.roll(pen_state, -1, 0)[:-1]
  pen_state = tf.concat([pen_state, [_FINAL_PEN_STATE]], 0)
  indices = tf.reshape(tf.where(x_strokes != _END_OF_STROKE), (-1,))
  x_strokes = tf.gather(x_strokes, indices)
  y_strokes = tf.gather(y_strokes, indices)
  if is_training:
    # Random scaling is performed *before* the normalization. The interaction
    # between the two is non-trivial because we don't perform random scaling at
    # dataset building time. This means that random scaling is not taked into
    # account when computing global stroke statistics. Setting the scaling
    # factor value too high is likely to result in corrupt sketches. An
    # empirical upper bound for the scaling value \alpha is $\alpha < 0.15$.
    x_strokes, y_strokes = _random_scale_strokes(
        stroke_random_scale_factor, x_strokes, y_strokes
    )
  x_strokes, y_strokes = stroke_stats_lib.normalize_strokes(
      config, stroke_stats, x_strokes, y_strokes
  )
  pen_state = tf.gather(pen_state, indices)
  strokes_3 = tf.stack([x_strokes, y_strokes, pen_state], axis=1)
  return strokes_3


def _strokes3_to_strokes5(
    config: ml_collections.FrozenConfigDict,
    strokes_3: tf.Tensor,
    max_stroke_sequence_length: int
):
  """Generates stroke-5 tensor from stroke-3 format tensor.

  Since the stroke-5 format is used for continuous sequence prediction, there is
  no tokenizer. This method explicitly inserts BOS and EOS vectors to the
  beginning and end of sequence, respectively.

  Args:
    config: Configuration dictionary.
    strokes_3: Sketch in stroke-3 format.
    max_stroke_sequence_length: Maximum sequence length.

  Returns:
    Sketch in stroke-5 format.
  """
  delta_x = strokes_3[:, 0]  # Δx.
  delta_y = strokes_3[:, 1]  # Δy.
  pen_touching = 1 - strokes_3[:, 2]  # $p_1$.
  pen_lifted = strokes_3[:, 2]  # $p_2$.
  pen_eos = tf.zeros_like(pen_touching)  # $p_3$.
  strokes_5 = tf.stack(
      [delta_x, delta_y, pen_touching, pen_lifted, pen_eos], axis=1
  )
  strokes_5 = strokes_5[:max_stroke_sequence_length - 2, :]

  # Explicitly insert BOS and EOS vectors.
  real_length = tf.shape(strokes_5)[0] + 1
  strokes_5 = tf.concat([
      tf.constant([[0., 0., 1, 0, 0]], dtype=tf.float32),
      strokes_5,
      tf.constant([[0., 0., 0, 0, 1]], dtype=tf.float32)
  ], axis=0)
  if config.manual_padding:
    strokes_5 = _pad_strokes(strokes_5, max_stroke_sequence_length)
  return strokes_5, real_length


def parse_sketch_strokes_or_tokens(
    config: ml_collections.FrozenConfigDict,
    features: dict[str, tf.Tensor],
    stroke_stats: StrokeStats,
    stroke_tokenizer: Optional[tokenizer_lib.StrokeTokenizer],
    is_training: bool,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Parses stroke information into stroke-5 format or tokens.

  Args:
    config: Configuration dictionary.
    features: Actual input tensors.
    stroke_stats: Various statistics collected for the strokes.
    stroke_tokenizer: Tokenizer.
    is_training: If true, dealing with the training partition.

  Returns:
    A tuple consisting of sketch in either stroke-5 or tokenized format,
    glyph affiliations (integers), and the real length before padding.
  """
  max_stroke_sequence_length = config.get("max_stroke_sequence_length")
  if not max_stroke_sequence_length:
    raise ValueError("Maximum sketch stroke/token sequence length not set")

  # Generate stroke-3 tensor.  The $p$ value (third value in a three-tuple) of 1
  # means lifting the pen from paper, which matches the original semantics in
  # Sketch-RNN and Graves' paper.
  x_strokes = features["strokes/x_stroke_points"]
  y_strokes = features["strokes/y_stroke_points"]
  if "stroke_random_scale_factor" not in config:
    raise ValueError("No stroke random scaling factor defined!")
  strokes_3 = _points_to_strokes3(
      config, stroke_stats, x_strokes, y_strokes, is_training,
      config.stroke_random_scale_factor
  )

  # Glyph affiliations have the same length as strokes.
  glyph_ids = features["strokes/glyph_affiliations/ids"]
  indices = tf.reshape(tf.where(glyph_ids != _END_OF_STROKE), (-1,))
  glyph_ids = tf.gather(glyph_ids, indices)

  if "stroke_tokenizer" in config and stroke_tokenizer is not None:
    # Tokenize the sketch. The real length excludes the special BOS and EOS
    # tokens. The tokens are encoded 1-D integer tensor. Each token is
    # affiliated with the glyph ID.
    tokens, glyph_ids, real_length = stroke_tokenizer.tf_encode(
        strokes_3, glyph_ids
    )
    # Insert special token marking the end of number subsequence.
    tokens, glyph_ids = tokenizer_lib.tf_mark_concept_tokens(
        stroke_tokens=tokens,
        stroke_glyph_ids=glyph_ids,
        glyph_tokens=features["text/glyph/tokens"],
        glyph_types=features["text/glyph/types"]
    )
    # Accommodate two extra tokens in real length (for either BOS or EOS and
    # number-concept separator).
    real_length = real_length + 2
    return tokens, glyph_ids, tf.cast(real_length, dtype=tf.int32)

  # Generate stroke-5 tensor and update the corresponding glyph affiliations.
  strokes_5, real_length = _strokes3_to_strokes5(
      config, strokes_3, max_stroke_sequence_length
  )
  glyph_ids = glyph_ids[:max_stroke_sequence_length - 2]
  glyph_ids = tf.concat(
      [[glyph_lib.GLYPH_BOS], glyph_ids, [glyph_lib.GLYPH_EOS]], axis=0
  )
  if config.manual_padding:
    glyph_ids = _pad_strokes(glyph_ids, max_stroke_sequence_length)
  return strokes_5, glyph_ids, real_length


def parse_formant_strokes(
    config: ml_collections.FrozenConfigDict,
    features: dict[str, tf.Tensor],
    feature_type: str,
    stroke_stats: StrokeStats,
    is_training: bool
) -> tuple[tf.Tensor, tf.Tensor]:
  """Parses stroke information into stroke-5 format $(x, y, p1, p2, p3)$.

  Parsing formants represented as strokes. Note that formant strokes are not
  the same as sketch strokes, so we need formant-specific configuration settings
  here that define maximum sequence length and random scaling factor.

  Args:
    config: Configuration dictionary.
    features: Actual input tensors.
    feature_type: A string that specifies the type of the formant track to
      parse from the features above, e.g., `last_word`.
    stroke_stats: Various statistics collected for the strokes.
    is_training: If true, dealing with the training partition.

  Returns:
    A tuple consisting of sketch in stroke-5 format and the actual length
    before padding.
  """
  max_stroke_sequence_length = config.get("max_formant_stroke_sequence_length")
  if not max_stroke_sequence_length:
    raise ValueError("Maximum formant stroke sequence length not set")

  # Generate strokes-3 tensor.
  x_strokes = features[f"speech/formants/{feature_type}/x_stroke_points"]
  y_strokes = features[f"speech/formants/{feature_type}/y_stroke_points"]
  strokes_3 = _points_to_strokes3(
      config, stroke_stats, x_strokes, y_strokes, is_training,
      config.get("formants_random_scale_factor", 0.15)
  )

  # Convert to strokes-5 tensor.
  return _strokes3_to_strokes5(config, strokes_3, max_stroke_sequence_length)
