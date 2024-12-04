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

"""Discrete glyph parser from JSONL files."""

import logging
from typing import Any

from absl import flags
from protoscribe.glyphs import glyph_vocab as glyph_lib
from protoscribe.glyphs import numbers_to_glyphs
from protoscribe.sketches.inference import json_utils

_IGNORE_ERRORS = flags.DEFINE_boolean(
    "ignore_errors", False,
    "If enabled, whenever an error is encountered (e.g., in detokenizer), "
    "do not bail out but produce a special error glyph. Not encouraged in "
    "normal operation."
)

_FORCE_ONE_CONCEPT_GLYPH = flags.DEFINE_boolean(
    "force_one_concept_glyph", False,
    "Trim the predictions to contain one concept glyph. Only used in debugging "
    "the cases when the model fails to predict the EOS token correctly. If "
    "enabled, use in conjunction with `--ignore_errors` flag. Not encounraged "
    "in normal operation."
)

# Note, this placeholder is nevertheless a valid glyph.
_BAD_GLYPH_NAME = "DUMMY"


def json_to_glyphs(
    glyph_vocab: glyph_lib.GlyphVocab,
    glyphs_dict: dict[str, Any],
    pronunciation_lexicon: dict[str, list[str]]
) -> tuple[str, str, dict[str, Any], bool]:
  """Generates discrete glyphs from predictions dictionary in `glyphs_dict`.

  Args:
    glyph_vocab: Global glyph vocabulary.
    glyphs_dict: Individual predictions from JSONL file as a dictionary.
    pronunciation_lexicon: Pronunciations.

  Returns:
    A four-tuple consisting of:
      - Original input `text`,
      - Decoded discrete glyph sequence,
      - Dictionary with additional information such as pronunciations for the
        decoded glyphs and confidence of prediction (for downstream tools),
      - Decoding status (True/False). If `--ignore_errors` is enabled, we still
        record the failed predictions.

  Raises:
    ValueError in case of errors when predictions cannot be detokenized,
    for example.
  """

  scorer_dict = json_utils.get_scorer_dict(glyphs_dict, pronunciation_lexicon)
  inputs = scorer_dict["text.inputs"]

  tokens = glyphs_dict["prediction"][-1]  # Select best hypothesis.
  error = False
  if tokens[0] != glyph_lib.GLYPH_BOS:
    error_msg = f"{inputs}: BOS token missing!"
    if not _IGNORE_ERRORS.value:
      raise ValueError(error_msg)
    else:
      error = True
      logging.warning(error_msg)
  tokens = tokens[1:]

  if glyph_lib.GLYPH_EOS not in tokens:
    error_msg = f"{inputs}: EOS token missing!"
    if not _IGNORE_ERRORS.value:
      raise ValueError(error_msg)
    else:
      error = True
      logging.warning(error_msg)

  if not error:
    eos_pos = tokens.index(glyph_lib.GLYPH_EOS)
    glyphs = glyph_vocab.detokenize(tokens[0:eos_pos])
  elif _FORCE_ONE_CONCEPT_GLYPH.value:
    # Detokenize up to the first concept glyph.
    glyphs = []
    for token in tokens:
      glyph = glyph_vocab.id_to_name(token)
      glyphs.append(glyph)
      if glyph not in numbers_to_glyphs.NUMBER_GLYPHS:
        break
  else:
    glyphs = [_BAD_GLYPH_NAME]
  scorer_dict["glyph.names"] = [glyphs]

  # Fills in pronunciation for the hypothesis that excludes the numbers.
  pron = []
  for glyph_name in glyphs:
    if glyph_name in pronunciation_lexicon:
      pron.append(pronunciation_lexicon[glyph_name])
    else:
      pron.append([])  # Number.
  scorer_dict["glyph.prons"] = [pron]

  # Computes confidence as the difference between the best and second best
  # predictions.
  scores = glyphs_dict["aux"]["scores"]
  confidence = scores[-1] - scores[-2]
  scorer_dict["glyph.confidence"] = confidence

  return inputs, " ".join(glyphs), scorer_dict, error
