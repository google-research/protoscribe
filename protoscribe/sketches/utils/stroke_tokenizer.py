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

"""Tokenization of the sketches.

This is based on the Sketchformer paper: Leo Sampaio Ferraz Ribeiro, Tu Bui,
John Collomosse, Moacir Ponti (2020): "Sketchformer: Transformer-based
Representation for Sketched Structure".
GitHub: https://github.com/leosampaio/sketchformer/
"""

import enum
import logging

import jax
import jax.numpy as jnp
import numpy as np
from protoscribe.glyphs import glyph_vocab
from protoscribe.sketches.utils import stroke_utils
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf

import glob
import os

Tensor = np.ndarray | jnp.ndarray


class Token(enum.IntEnum):
  """Special sketch tokens.

     The actual cluster centroids need to be shifted to account for these.
  """
  PAD = 0
  START_OF_SKETCH = 1
  # Pen is lifted from the paper. Coincides with discrete glyph "</s>" as well.
  STROKE_SEP = 2
  END_OF_SKETCH = 3  # Coincides with discrete glyph "<unk>".
  END_OF_NUMBERS = 4  # Special token separating numbers and concepts.

CENTROID_OFFSET = Token.END_OF_NUMBERS + 1


def _find_closest_codes(points: Tensor, codebook: Tensor) -> Tensor:
  """Returns the index of the closest code to each `point`.

  See VectorQuantizer docstring for meanings of symbols.

  Args:
    points:   [..., D]
    codebook: [C, D]

  Returns:
    A Tensor of int32 code indices, shape `points.shape[:-1]`.
  """
  # [N, D]
  points = points.reshape((-1, points.shape[-1]))

  # Compute L2 similarity (negative distance). For efficiency, we ignore the
  # norms of `points` because they don't affect the resulting code
  # assignments:
  #   code(i) = argmax_j -||X_i - C_j||² / 2
  #           = argmax_j (X_i · C_j) - ||C_j||² / 2
  # Similarity maxtrix: [C, N]
  similarities = (
      # [C, N]
      jax.lax.dot(codebook, points.transpose())
      # [C], broadcasted to [C, N]
      - (jnp.einsum("cd,cd->c", codebook, codebook) / 2.)[:, None])

  # Note when k=1 the function below is exact (not approximate), but can still
  # be faster than jnp.argmax.
  # [1, N]
  _, codes = jax.lax.approx_max_k(similarities, k=1, reduction_dimension=0)

  # [...]
  codes = jnp.reshape(codes, points.shape[:-1])
  return codes


@tf.function(jit_compile=True)
def _tf_find_closest_codes(points: tf.Tensor, codebook: tf.Tensor) -> tf.Tensor:
  """Returns the index of the closest code to each `point`.

  See VectorQuantizer docstring for meanings of symbols.

  Args:
    points:   [..., D]
    codebook: [C, D]

  Returns:
    A tf.Tensor of int32 code indices, shape `points.shape[:-1]`.
  """
  # [N, D]
  points = tf.reshape(points, [-1, points.shape[-1]])

  # Compute L2 similarity (negative distance). For efficiency, we ignore the
  # norms of `points` because they don't affect the resulting code
  # assignments:
  #   code(i) = argmax_j -||X_i - C_j||² / 2
  #           = argmax_j (X_i · C_j) - ||C_j||² / 2
  # Similarity maxtrix: [C, N]
  similarities = (
      # [C, N]
      tf.linalg.matmul(codebook, tf.transpose(points))
      # [C], broadcasted to [C, N]
      - (tf.einsum("cd,cd->c", codebook, codebook) / 2.)[:, None])

  # Note when k=1 the function below is exact (not approximate), but can still
  # be faster than jnp.argmax.
  # [1, N]
  _, codes = tf.math.approx_max_k(similarities, k=1, reduction_dimension=0)

  # [1, N] -> [N]
  codes = tf.squeeze(codes)
  return codes


def tf_insert(
    x: tf.Tensor, pos: int | tf.Tensor, value: int| tf.Tensor
) -> tf.Tensor:
  """Inserts `value` into positions of 1-D tensor `x` at specified indices."""
  if isinstance(pos, int):
    pos = tf.expand_dims(pos, -1)
  pos_len = tf.shape(pos)[0]
  result_size = tf.shape(x)[0] + pos_len  # Final array shape.
  # Positions in the 1-D result where to insert the `value`.
  insert_pos = pos + tf.range(0, pos_len)
  insert_pos = tf.expand_dims(insert_pos, -1)  # [[i_0], [i_1], ...]
  # Fill the specified positions `pos` with `value`. The rest of the elements
  # are zeros.
  insert_values = tf.repeat(value, pos_len)
  result = tf.scatter_nd(insert_pos,
                         insert_values,
                         [result_size])
  # Gather the original values of `x` into the result into the indexes with
  # zeros.
  zero_pos = tf.where(result != value)
  result = tf.tensor_scatter_nd_add(result, zero_pos, x)
  return result


def tf_first_occurrence_index_above_max(
    sequence: tf.Tensor, desired_max: int
) -> tf.Tensor:
  """Finds the first index where the data is above the maximum.

  Args:
    sequence: [length] tensor.
    desired_max: scalar.

  Returns:
    The index where the data is above the desirev maximum.
  """
  desired_max = tf.convert_to_tensor(desired_max, dtype=sequence.dtype)
  return tf.cast(
      tf.reduce_min(tf.where(sequence >= desired_max)), dtype=tf.int32
  )


def tf_mark_concept_tokens(
    stroke_tokens: tf.Tensor,
    stroke_glyph_ids: tf.Tensor,
    glyph_tokens: tf.Tensor,
    glyph_types: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
  """Inserts special end-of-numbers token based on glyph affiliations.

  Args:
    stroke_tokens: Tensor (L_s,) of integer tokens, where L_s is the number of
      strokes.
    stroke_glyph_ids: Tensor (L_s,) of glyph IDs corresponding to strokes.
    glyph_tokens: Integer tensor (L_g,) where L_g is the number of glyphs in
      a sequence.
    glyph_types: Integer tensor (L_g,) where L_g is the number of glyphs in
      a sequence. The integers take value of 0 for special tokens, 1 for numbers
      and 2 for concepts.

  Returns:
    Updated stroke tokens with an extra end-of-numbers token inserted along
    with updated glyph affiliations.
  """

  def extend_with_eos(arr, eos):
    return tf.concat([arr, [eos]], axis=0)

  glyph_tokens_plus_eos = extend_with_eos(
      glyph_tokens, stroke_utils.END_OF_STROKE
  )
  glyph_types_plus_eos = extend_with_eos(
      glyph_types, glyph_vocab.GLYPH_TYPE_MASK_PAD
  )
  idx = tf.where(
      glyph_types_plus_eos == glyph_vocab.GLYPH_TYPE_MASK_CONCEPT,
      glyph_tokens_plus_eos,
      tf.zeros_like(glyph_tokens_plus_eos) - 1,
  )
  glyphs_in_idx = tf.reduce_any(  # Equivalent to np.in1d.
      tf.equal(
          tf.expand_dims(idx, axis=0), tf.expand_dims(stroke_glyph_ids, axis=1)
      ), axis=1
  )
  concept_strokes_mask = tf.where(
      glyphs_in_idx,
      tf.ones_like(stroke_glyph_ids),
      tf.zeros_like(stroke_glyph_ids),
  )
  first_concept_idx = tf.expand_dims(
      tf_first_occurrence_index_above_max(
          concept_strokes_mask, desired_max=1
      ), axis=-1
  )
  stroke_tokens = tf_insert(
      stroke_tokens, pos=first_concept_idx, value=Token.END_OF_NUMBERS
  )
  stroke_glyph_ids = tf_insert(
      stroke_glyph_ids, pos=first_concept_idx, value=stroke_utils.END_OF_STROKE
  )
  return stroke_tokens, stroke_glyph_ids


class StrokeTokenizer:
  """Tokenize/detokenize the sketch given the quantization vocabulary."""

  def __init__(self, dict_path: str, max_sequence_length: int) -> None:
    self.max_sequence_length = max_sequence_length
    logging.info("Loading tokenizer vocab from %s ...", dict_path)
    with open(dict_path, mode="rb") as f:
      self.codebook = np.load(f)
    num_clusters = self.codebook.shape[0]
    logging.info("Loaded %d clusters.", num_clusters)
    self.knn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(
        self.codebook)

    # The vocabulary size corresponds to the number of clusters plus three extra
    # special tokens (BOS, EOS and the stroke separator). We exclude the padding
    # token.
    self.vocab_size = num_clusters + 3

  def encode(self, strokes3: Tensor) -> Tensor:
    """Encodes the sequence of strokes in stroke-3 format into tokens."""
    if len(strokes3.shape) != 2:
      raise ValueError("Only single sketches are supported!")

    out = _find_closest_codes(strokes3[:, :2], self.codebook)
    out += CENTROID_OFFSET  # Shift because of the control tokens.
    pen_up = jnp.where(strokes3[:, 2] == 1)[0]
    # Assemble insertion indices. TF inserts *before* the condition index,
    # hence we offset the pen lifted indices by one in order to insert the
    # separator after the condition.
    out = jnp.insert(out, pen_up + 1, Token.STROKE_SEP)
    out = jnp.insert(out, 0, Token.START_OF_SKETCH)
    out = jnp.append(out, Token.END_OF_SKETCH)
    num_pad = self.max_sequence_length - out.shape[0]
    if num_pad > 0:
      out = jnp.concatenate([out, jnp.full(num_pad, Token.PAD)], axis=0)
    else:
      out = out[:self.max_sequence_length]
      out = jnp.append(out, jnp.array([Token.STROKE_SEP, Token.END_OF_SKETCH]))
    return jnp.int32(out)

  def tf_encode(
      self,
      strokes3: tf.Tensor,
      glyph_affiliations_ids: tf.Tensor
  ) -> tuple[tf.Tensor, tf.Tensor, int | tf.Tensor]:
    """Encodes the sequence of strokes in stroke-3 format into tokens.

    Tensorflow version. Please note: This version also takes the glyph
    affiliations (IDs of the glyphs rather than their positions) into account
    and re-arranges these to match the resulting tokens.

    Args:
      strokes3: Tensor of strokes in stroke-3 format.
      glyph_affiliations_ids: Glyph affiliations tensor consisting of glyph IDs.

    Returns:
      A tuple of three tensors: the tokens, their glyph affiliations and
      original lengths before padding.
    """
    tf.debugging.assert_rank(
        strokes3, 2, "Wrong rank for strokes-3 tensor (%s)" % tf.rank(strokes3)
    )

    # Assemble insertion indices. TF inserts *before* the condition index,
    # hence we offset the pen lifted indices by one in order to insert the
    # separator after the condition.
    pen_up = tf.cast(tf.where(strokes3[:, 2] == 1), dtype=tf.int32)
    pen_up = tf.squeeze(pen_up + 1)
    # Find centroids. Shift the token values because of the control tokens.
    tokens = _tf_find_closest_codes(
        strokes3[:, :2],
        tf.dtypes.cast(self.codebook, dtype=tf.float32)
    )
    tokens += CENTROID_OFFSET

    # Insert the control tokens.
    tokens = tf_insert(tokens, pen_up, Token.STROKE_SEP)
    glyph_ids = tf_insert(glyph_affiliations_ids, pen_up, Token.STROKE_SEP)
    real_length = tf.shape(tokens)[0]
    tokens = tf_insert(tokens, 0, Token.START_OF_SKETCH)
    tokens = tf.concat([tokens, [Token.END_OF_SKETCH]], axis=0)
    glyph_ids = tf_insert(glyph_ids, 0, Token.START_OF_SKETCH)
    glyph_ids = tf.concat([glyph_ids, [Token.END_OF_SKETCH]], axis=0)
    num_pad = self.max_sequence_length - tf.shape(tokens)[0]
    if num_pad > 0:
      tokens = tf.concat([tokens, tf.fill([num_pad], Token.PAD)], axis=0)
      glyph_ids = tf.concat([glyph_ids, tf.fill([num_pad], Token.PAD)], axis=0)
    else:
      real_length = self.max_sequence_length - 2
      tokens = tokens[:real_length]
      tokens = tf.concat(
          [tokens, [Token.STROKE_SEP, Token.END_OF_SKETCH]], axis=0
      )
      glyph_ids = glyph_ids[:real_length]
      glyph_ids = tf.concat(
          [glyph_ids, [Token.STROKE_SEP, Token.END_OF_SKETCH]], axis=0
      )

    return tokens, glyph_ids, real_length

  def slow_encode(self, strokes3: Tensor) -> Tensor:
    """Slow (sequential) encoding of strokes in stroke-3 format."""
    out = self.knn.kneighbors(X=strokes3[:, :2], return_distance=False)
    out = out.flatten()
    out += CENTROID_OFFSET  # Shift because of the control tokens.
    out = list(out)
    # Insert SEP token.
    positions = jnp.where(strokes3[:, 2] == 1)[0]
    offset = 1
    for i in positions:
      out.insert(i + offset, Token.STROKE_SEP)
      offset += 1
    # Insert SOS and EOS.
    out = [Token.START_OF_SKETCH] + out + [Token.END_OF_SKETCH]
    num_pad = self.max_sequence_length - len(out)
    if num_pad > 0:
      out += [Token.PAD] * num_pad
    else:
      out = out[:self.max_sequence_length]
      out[-2:] = [Token.STROKE_SEP, Token.END_OF_SKETCH]
    return jnp.array(out, dtype=jnp.int32)

  def decode(self, tokens: Tensor) -> Tensor:
    """Decodes a sequence of integer tokens into stroke-3 sketch format."""
    if len(tokens.shape) != 1:
      raise ValueError("Only single token sequence is supported!")

    cluster_ids = []
    pen_states = []
    for i in range(tokens.shape[0]):
      tok = tokens[i]
      if tok >= CENTROID_OFFSET:
        cluster_ids.append(tok)
        pen_states.append(0)
      elif tok == Token.STROKE_SEP and pen_states:
        pen_states[-1] = 1
      elif tok == Token.END_OF_SKETCH:
        break

    if not cluster_ids:
      raise ValueError(
          "Empty sketch: Only special tokens were found among "
          f"{len(tokens.shape)} tokens!"
      )
    cluster_ids = jnp.array(cluster_ids)
    cluster_ids -= CENTROID_OFFSET
    dxy = self.codebook[cluster_ids]
    out = jnp.c_[dxy, jnp.array(pen_states)]
    return jnp.float32(out)
