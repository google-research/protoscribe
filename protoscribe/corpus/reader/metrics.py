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

"""Miscellaneous metrics.

These API are special in that they assume that the inputs have not been
passed through seqIO postprocessor, i.e. they are not detokenized into
sequences of strings.
"""

import jiwer
import numpy as np


def _stringify_tokens(tokens: list[list[int]]) -> list[str]:
  """Converts lists of int token sequences to list of strings."""
  tokens = [list(map(str, seq)) for seq in tokens]
  return [" ".join(seq) for seq in tokens]


def sequence_accuracy(
    targets: list[list[int]],
    predictions: list[list[int]]
) -> dict[str, float]:
  """Computes per-sequence accuracy.

  For each example, returns 1.0 if the target sequence EXACTLY matches the
  predicted sequence. Else, 0.0.

  Args:
    targets: list of list of tokens.
    predictions: list of list of tokens.

  Returns:
    float. Average sequence-level accuracy.
  """
  target_toks = _stringify_tokens(targets)
  prediction_toks = _stringify_tokens(predictions)
  assert len(target_toks) == len(prediction_toks)
  seq_acc = 100. * np.mean(
      [p == t for p, t in zip(prediction_toks, target_toks)]
  )
  return {"sequence_accuracy": seq_acc}


def wer(
    targets: list[list[int]], predictions: list[list[int]]
) -> dict[str, float]:
  """Computes word-error rate (WER).

  Args:
    targets: list of list of tokens.
    predictions: list of list of tokens.

  Returns:
    WER across all targets and predictions.
  """
  wer_default = jiwer.wer(
      truth=_stringify_tokens(targets),
      hypothesis=_stringify_tokens(predictions),
      truth_transform=jiwer.transformations.wer_default,
      hypothesis_transform=jiwer.transformations.wer_default,
  )
  return {"wer": wer_default}
