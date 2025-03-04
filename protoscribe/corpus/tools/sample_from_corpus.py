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

"""Glyph sampling utilities."""

import enum

import numpy as np
from protoscribe.sketches.utils import stroke_tokenizer as tokenizer_lib
from protoscribe.sketches.utils import stroke_utils as strokes_lib
import tensorflow as tf

Array = np.ndarray


class TokenizerMode(enum.Enum):
  """Type of the tokenization to invoke."""
  ALL = "ALL"
  SLOW = "SLOW"
  JAX = "JAX"
  TF = "TF"


def tokenize_and_reconstruct(
    normalized_sketch5: Array,
    glyph_affiliations: Array,
    tokenizer: tokenizer_lib.StrokeTokenizer,
    tokenizer_mode: TokenizerMode = TokenizerMode.SLOW
) -> Array:
  """Checks tokenization/detokenization.

  Note that the tokenizer is trained on normalized sketches. Sketch
  needs de-normalization before saving.

  Args:
    normalized_sketch5: Sketch in stroke5 format.
    glyph_affiliations: Glyph affiliations for each stroke.
    tokenizer: Tokenizer object.
    tokenizer_mode: One of the three possible modes of tokenization.
      One of: slow (regular mode, default), Jax or TensorFlow.

  Returns:
    Reconstructed strokes in stroke3 format.

  Raises:
    ValueError if mode is unsupported.
  """
  sketch3 = strokes_lib.stroke5_to_stroke3(normalized_sketch5)
  if tokenizer_mode == TokenizerMode.SLOW:
    tokens = tokenizer.slow_encode(sketch3)
  elif tokenizer_mode == TokenizerMode.JAX:
    tokens = tokenizer.encode(sketch3)
  elif tokenizer_mode == TokenizerMode.TF:  # TensorFlow:
    tokens, _, _ = tokenizer.tf_encode(
        tf.convert_to_tensor(sketch3, dtype=tf.float32),
        tf.convert_to_tensor(glyph_affiliations, dtype=tf.int32)
    )
    tokens = tokens.numpy()
  else:
    raise ValueError(f"Unsupported mode: {tokenizer_mode}")

  sketch3 = tokenizer.decode(tokens)
  return np.float32(sketch3)
