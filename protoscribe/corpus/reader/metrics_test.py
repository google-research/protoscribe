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

"""Test for metrics."""

from absl.testing import absltest
from protoscribe.corpus.reader import metrics


class MetricsTest(absltest.TestCase):

  def test_wer(self):
    """Checks word-error rate."""
    self.assertEqual(
        metrics.wer(
            [[1, 20]], [[1, 30]]
        )["wer"], 0.5
    )
    self.assertAlmostEqual(
        metrics.wer(
            [[1, 20, 2]], [[1, 30, 3]]
        )["wer"], 2. / 3
    )
    self.assertAlmostEqual(
        metrics.wer(
            [[1, 2, 3], [4, 4, 4], [5]], [[1, 2, 3], [10], [7, 8]]
        )["wer"], 0.71, delta=1E-2
    )

  def test_sequence_accuracy(self):
    self.assertEqual(
        metrics.sequence_accuracy(
            [[1, 20]], [[1, 20]]
        )["sequence_accuracy"], 100.0
    )
    self.assertEqual(
        metrics.sequence_accuracy(
            [[3, 2, 1]], [[3, 2, 2]]
        )["sequence_accuracy"], 0.0
    )
    self.assertEqual(
        metrics.sequence_accuracy(
            [[3, 2, 1], [4]], [[3, 2, 1], [1, 1, 1, 1]]
        )["sequence_accuracy"], 50.0
    )


if __name__ == "__main__":
  absltest.main()
