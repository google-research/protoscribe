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

"""Utilities for manipulating final outputs computed by `glyphs_from_json`."""

import dataclasses
import json
import logging
from typing import Any

import numpy as np
from protoscribe.evolution import confidence_pruning
from protoscribe.glyphs import numbers_to_glyphs
from protoscribe.sketches.inference import json_utils

import glob
import os

ConfidencePruningMethod = confidence_pruning.Method


@dataclasses.dataclass(frozen=True)
class GlyphInfo:
  glyph_names: str
  confidence: float
  concept_pron: str
  glyph_pron: str

  # For sketches the strokes information will be present.
  strokes: np.ndarray | None = None

  # This is a probability estimate of extension if probability threshold pruning
  # is used.
  log_prob: float = 1000.


def _load_jsonl(path: str) -> list[dict[str, Any]]:
  """Loads JSONL file returning a list of dictionaries.

  Args:
    path: Path to JSONL file.

  Returns:
    A list of parsed dictionaries.
  """
  logging.info("Reading JSONL from %s ...", path)
  dicts = []
  with open(path, mode="rt") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      dicts.append(json.loads(line))
  logging.info("Read %d dicts.", len(dicts))
  return dicts


def load_and_prune_jsonl(
    path: str,
    pruning_method: ConfidencePruningMethod,
    **kwargs,
) -> dict[str, GlyphInfo]:
  """Loads a JSONL file, returning a dictionary.

  The dictionary's keys are concepts/words, and value is the best (according
  to the selected pruning method, if any) glyph candidate.

  Args:
    path: Path to a TSV file.
    pruning_method: An enum specifying the pruning method.
    **kwargs: Optional keyword arguments. One common usecase of this is passing
      the pruning parameters.

  Returns:
    A dictionary as describe above.

  Raises:
    ValueError if any of the parsed values have wrong types.
  """
  all_docs = _load_jsonl(path)
  pruned_docs = confidence_pruning.prune_docs(
      all_docs,
      pruning_method=pruning_method,
      **kwargs
  )

  results = {}
  for concept, result_dict in pruned_docs.items():
    concept_pron = " ".join(result_dict["concept.pron"])
    # Select the last (best) element from n-best list.
    glyph_pron = json_utils.glyph_pron(result_dict, k=-1)
    glyphs = result_dict["glyph.names"][-1]
    strokes = None
    if "strokes.nbest.deltas" in result_dict:
      strokes = np.array(
          result_dict["strokes.nbest.deltas"][-1], dtype=np.float32
      )
      if len(strokes.shape) != 2:
        raise ValueError(f"{concept}: Expected strokes array of rank 2!")

    # Drop the numbers.
    glyph = []
    for g in glyphs:
      if g in numbers_to_glyphs.NUMBER_GLYPHS:
        continue
      glyph.append(g)
    glyph = " ".join(glyph)

    # Update the results. Each key is a unique new concept for which we provide
    # hypothesis spellings.
    if concept in results:
      raise ValueError(f"Spellings for concept `{concept}` are not unique!")
    results[concept] = GlyphInfo(
        glyph_names=glyph,
        confidence=confidence_pruning.confidence(result_dict),
        concept_pron=concept_pron,
        glyph_pron=glyph_pron,
        strokes=strokes,
        log_prob=result_dict["log_prob"] if "log_prob" in result_dict else -1,
    )

  return results
