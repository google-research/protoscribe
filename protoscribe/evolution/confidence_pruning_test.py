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

"""Basic checks for various confidence pruning methods."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from protoscribe.evolution import confidence_pruning as lib

_JSONL = [
    {
        "concept.name": "dog",
        "glyph.confidence": 10.0,
    },
    {
        "concept.name": "cat",
        "glyph.confidence": 8.5,
    },
    {
        "concept.name": "aardvark",
        "glyph.confidence": 11.3,
    },
    {
        "concept.name": "platypus",
        "glyph.confidence": 11.0,
    },
    {
        "concept.name": "mole",
        "glyph.confidence": 3.8,
    },
    {
        "concept.name": "mammoth",
        "glyph.confidence": 14.5,
    },
    {
        "concept.name": "mammoth",
        "glyph.confidence": 14.0,
    },
]


class ConfidencePruningTest(parameterized.TestCase):

  def _prune_results(
      self,
      pruning_method: lib.Method,
      use_softmax: bool = True,
      min_confidence_threshold: float = -1.,
      min_prob: float = 0.,
      top_k: int = -1,
      top_percentage: int = -1,
      max_cumulative_prob: float = -1.
  ) -> dict[str, dict[str, Any]]:
    """Loads and prunes the results from JSONL file."""
    return lib.prune_docs(
        _JSONL,
        pruning_method=pruning_method,
        use_softmax=use_softmax,
        min_confidence_threshold=min_confidence_threshold,
        min_prob=min_prob,
        top_k=top_k,
        top_percentage=top_percentage,
        max_cumulative_prob=max_cumulative_prob
    )

  def _check_log_probs(
      self, pruned_dicts: dict[str, dict[str, Any]]
  ) -> None:
    """Checks that probabilities in pruned dicts are valid."""
    sum_probs = 0.
    for key in pruned_dicts:
      self.assertIn("log_prob", pruned_dicts[key])
      self.assertGreaterEqual(0., pruned_dicts[key]["log_prob"])
      sum_probs += np.exp(pruned_dicts[key]["log_prob"])
    self.assertAlmostEqual(sum_probs, 1., delta=5)

  def test_no_pruning(self) -> None:
    results = self._prune_results(lib.Method.NONE)
    self.assertLen(results, 6)
    self.assertEqual(
        sorted(results.keys()),
        ["aardvark", "cat", "dog", "mammoth", "mole", "platypus"]
    )

  def test_confidence_access(self) -> None:
    for d in _JSONL:
      self.assertGreater(lib.confidence(d), 0.)

  def test_dedup_by_concepts(self) -> None:
    deduped = lib.dedup_by_concepts(_JSONL)
    # Prune one mammoth.
    self.assertLen(deduped, len(_JSONL) - 1)
    found_mammoth = False
    for d in deduped:
      if d["concept.name"] == "mammoth":
        # The only mamooth present in the deduped list should have the
        # higher confidence than the pruned one.
        self.assertEqual(d["glyph.confidence"], 14.5)
        found_mammoth = True
        break
    self.assertTrue(found_mammoth)

  @parameterized.parameters(False, True)
  def test_confidence_distribution(self, use_softmax: bool) -> None:
    log_probs = lib.confidence_distribution(
        _JSONL, log_domain=True, use_softmax=use_softmax
    )
    mass = 0.
    for log_prob in log_probs:
      self.assertGreater(0., log_prob)
      mass += np.exp(log_prob)
    self.assertAlmostEqual(mass, 1., delta=5)
    probs = lib.confidence_distribution(
        _JSONL, log_domain=False, use_softmax=use_softmax
    )
    for prob in probs:
      self.assertLess(0., prob)
      self.assertGreater(1., prob)
    self.assertAlmostEqual(np.sum(probs), 1., delta=5)

  @parameterized.parameters(
      # Low threshold.
      (4., ["aardvark", "cat", "dog", "mammoth", "platypus"]),
      # Medium threshold.
      (11., ["aardvark", "mammoth", "platypus"]),
      # High threshold.
      (14., ["mammoth"]),
  )
  def test_prune_by_threshold(
      self, threshold: float, pruned_concepts: list[str]
  ) -> None:
    results = self._prune_results(
        lib.Method.THRESHOLD, min_confidence_threshold=threshold
    )
    self.assertLen(results, len(pruned_concepts))
    self.assertEqual(sorted(results.keys()), pruned_concepts)

  @parameterized.parameters(
      # Low probability.
      (0.005, ["aardvark", "dog", "mammoth", "platypus"]),
      # Medium range probability.
      (0.02, ["aardvark", "mammoth", "platypus"]),
      # High probability.
      (0.8, ["mammoth"]),
  )
  def test_prune_by_prob(
      self, prob: float, pruned_concepts: list[str]
  ) -> None:
    results = self._prune_results(
        lib.Method.PROBABILITY, use_softmax=True, min_prob=prob
    )
    self.assertLen(results, len(pruned_concepts))
    self.assertEqual(sorted(results.keys()), pruned_concepts)
    self._check_log_probs(results)

  @parameterized.parameters(
      # Nearly all hypotheses.
      (5, ["aardvark", "cat", "dog", "mammoth", "platypus"]),
      # Medium range.
      (3, ["aardvark", "mammoth", "platypus"]),
      # Best hypotheses.
      (1, ["mammoth"]),
  )
  def test_prune_by_top_k(
      self, top_k: int, pruned_concepts: list[str]
  ) -> None:
    results = self._prune_results(lib.Method.TOP_K, top_k=top_k)
    self.assertLen(results, len(pruned_concepts))
    self.assertEqual(sorted(results.keys()), pruned_concepts)

  @parameterized.parameters(
      # Nearly all hypotheses.
      (100, ["aardvark", "cat", "dog", "mammoth", "mole", "platypus"]),
      # Medium range.
      (50, ["aardvark", "mammoth", "platypus"]),
      # Best hypotheses.
      (1, ["mammoth"]),
  )
  def test_prune_by_top_percentage(
      self, top_percentage: int, pruned_concepts: list[str]
  ) -> None:
    results = self._prune_results(
        lib.Method.TOP_PERCENTAGE, top_percentage=top_percentage
    )
    self.assertLen(results, len(pruned_concepts))
    self.assertEqual(sorted(results.keys()), pruned_concepts)

  @parameterized.parameters(
      # Conversion using softmax:
      # -------------------------
      # Low cumulative probability.
      (0.95, True, ["mammoth"]),
      # Medium range cumulative probability.
      (0.99, True, ["aardvark", "mammoth", "platypus"]),
      # High cumulative probability.
      (0.999, True, ["aardvark", "dog", "mammoth", "platypus"]),
      # Conversion using regular normalization:
      # ---------------------------------------
      # Low cumulative probability.
      (0.3, False, ["mammoth"]),
      # Medium range cumulative probability.
      (0.7, False, ["aardvark", "mammoth", "platypus"]),
      # High cumulative probability.
      (1., False, ["aardvark", "cat", "dog", "mammoth", "mole", "platypus"]),
  )
  def test_prune_by_top_p(
      self, prob: float, use_softmax: bool, pruned_concepts: list[str]
  ) -> None:
    results = self._prune_results(
        lib.Method.TOP_P, use_softmax=use_softmax, max_cumulative_prob=prob
    )
    self.assertLen(results, len(pruned_concepts))
    self.assertEqual(sorted(results.keys()), pruned_concepts)
    self._check_log_probs(results)


if __name__ == "__main__":
  absltest.main()
