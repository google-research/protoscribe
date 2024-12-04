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

"""Simple test for results parser."""

import os

from absl import flags
from absl.testing import absltest
import numpy as np
from protoscribe.evolution import results_parser_and_pruner as lib

ConfidencePruningMethod = lib.ConfidencePruningMethod

FLAGS = flags.FLAGS

_DATA_DIR = "protoscribe/evolution/testdata"
_GLYPH_RESULTS_FILENAME = "glyph_results.jsonl"
_SKETCH_RESULTS_FILENAME = "sketch-token_recognizer_results.jsonl"

# Pruning thresholds as absolute confidence values.
_LOW_CONFIDENCE_THRESHOLD = 0.5
_HIGH_CONFIDENCE_THRESHOLD = 5.

# Pruning thresholds as probabilities.
_LOW_PROBABILITY = 1e-3
_HIGH_PROBABILITY = 0.7

# Top-K pruning.
_TOP_K_BEST = 1
_TOP_K_ALL = 3

# Pruning threshold as cumulative probabilities.
_LOW_CUMULATIVE_PROBABILITY = 0.8
_HIGH_CUMULATIVE_PROBABILITY = 1.0


class ResultsParserTest(absltest.TestCase):

  def _load_and_prune_results(
      self,
      file_name: str,
      pruning_method: ConfidencePruningMethod,
      min_confidence_threshold: float = -1.,
      min_prob: float = 0.,
      top_k: int = -1,
      max_cumulative_prob: float = -1.
  ) -> dict[str, lib.GlyphInfo]:
    """Loads and prunes the results from JSONL file."""
    path = os.path.join(FLAGS.test_srcdir, _DATA_DIR, file_name)
    return lib.load_and_prune_jsonl(
        path,
        pruning_method=pruning_method,
        min_confidence_threshold=min_confidence_threshold,
        min_prob=min_prob,
        top_k=top_k,
        max_cumulative_prob=max_cumulative_prob
    )

  def test_discrete_glyph_prune_by_threshold(self):
    results = self._load_and_prune_results(
        _GLYPH_RESULTS_FILENAME,
        ConfidencePruningMethod.THRESHOLD,
        min_confidence_threshold=_LOW_CONFIDENCE_THRESHOLD
    )
    self.assertLen(results, 3)
    self.assertIn("marsh_NOUN", results)
    self.assertIn("tray_NOUN", results)
    self.assertIn("wear_VERB", results)

    results = self._load_and_prune_results(
        _GLYPH_RESULTS_FILENAME,
        ConfidencePruningMethod.THRESHOLD,
        min_confidence_threshold=_HIGH_CONFIDENCE_THRESHOLD)
    self.assertLen(results, 2)
    self.assertIn("marsh_NOUN", results)
    self.assertIn("wear_VERB", results)

  def test_sketch_recognizer_prune_by_threshold(self):
    results = self._load_and_prune_results(
        _SKETCH_RESULTS_FILENAME,
        ConfidencePruningMethod.THRESHOLD,
        min_confidence_threshold=_LOW_CONFIDENCE_THRESHOLD
    )
    self.assertLen(results, 3)
    self.assertIn("bypass_NOUN", results)
    self.assertIn("excited_ADJ", results)
    self.assertIn("navy_NOUN", results)

    results = self._load_and_prune_results(
        _SKETCH_RESULTS_FILENAME,
        ConfidencePruningMethod.THRESHOLD,
        min_confidence_threshold=_HIGH_CONFIDENCE_THRESHOLD
    )
    self.assertLen(results, 3)
    bypass = results["bypass_NOUN"]
    strokes = bypass.strokes
    self.assertIsNot(strokes, None)
    assert strokes is not None
    self.assertGreater(strokes.shape[0], 1)
    self.assertEqual(strokes.shape[1], 3)  # Stroke-3 format.

  def test_sketch_recognizer_prune_by_prob(self):
    results = self._load_and_prune_results(
        _SKETCH_RESULTS_FILENAME,
        ConfidencePruningMethod.PROBABILITY,
        min_prob=_LOW_PROBABILITY
    )
    self.assertLen(results, 3)
    for concept in ["bypass_NOUN", "excited_ADJ", "navy_NOUN"]:
      self.assertIn(concept, results)
      self.assertLess(results[concept].log_prob, 0)

    results = self._load_and_prune_results(
        _SKETCH_RESULTS_FILENAME,
        ConfidencePruningMethod.PROBABILITY,
        min_prob=_HIGH_PROBABILITY
    )
    self.assertLen(results, 1)
    self.assertIn("navy_NOUN", results)
    self.assertGreater(results["navy_NOUN"].log_prob, np.log(_HIGH_PROBABILITY))

  def test_sketch_recognizer_prune_top_k(self):
    results = self._load_and_prune_results(
        _SKETCH_RESULTS_FILENAME,
        ConfidencePruningMethod.TOP_K,
        top_k=_TOP_K_ALL
    )
    self.assertLen(results, 3)
    for concept in ["bypass_NOUN", "excited_ADJ", "navy_NOUN"]:
      self.assertIn(concept, results)

    results = self._load_and_prune_results(
        _SKETCH_RESULTS_FILENAME,
        ConfidencePruningMethod.TOP_K,
        top_k=_TOP_K_BEST
    )
    self.assertLen(results, 1)
    self.assertIn("navy_NOUN", results)

  def test_sketch_recognizer_prune_top_p(self):
    results = self._load_and_prune_results(
        _SKETCH_RESULTS_FILENAME,
        ConfidencePruningMethod.TOP_P,
        max_cumulative_prob=_HIGH_CUMULATIVE_PROBABILITY
    )
    self.assertLen(results, 3)
    for concept in ["bypass_NOUN", "excited_ADJ", "navy_NOUN"]:
      self.assertIn(concept, results)
      self.assertLess(results[concept].log_prob, 0)

    results = self._load_and_prune_results(
        _SKETCH_RESULTS_FILENAME,
        ConfidencePruningMethod.TOP_P,
        max_cumulative_prob=_LOW_CUMULATIVE_PROBABILITY
    )
    self.assertLen(results, 1)
    self.assertIn("navy_NOUN", results)
    self.assertAlmostEqual(results["navy_NOUN"].log_prob, -0.224, delta=3)


if __name__ == "__main__":
  absltest.main()
