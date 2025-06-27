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

"""Tests for utils.perplexity_utils."""

import math
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from protoscribe.pmmx.utils import perplexity_utils


class PerplexityUtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          logits=np.array([[[0.0, 0.0, 0.0]]]),
          targets=np.array([[1]]),
          weights=None,
          expect=3
      ),
      dict(
          logits=np.array([[[0.0, 0.0, 1.0]]]),
          targets=np.array([[1]]),
          weights=None,
          expect=(2 + math.e)
      ),
      dict(
          logits=np.array([[[0.0, 0.0, 1.0]]]),
          targets=np.array([[2]]),
          weights=None,
          expect=(2 + math.e) / math.e
      ),
      dict(
          logits=np.array([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]),
          targets=np.array([[1, 2]]),
          weights=None,
          expect=(2 + math.e) * math.exp(-.5)
      ),
      dict(
          logits=np.array([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]),
          targets=np.array([[1, 2]]),
          weights=np.array([[1.0, 0.0]]),
          expect=(2 + math.e)
      ),
      dict(
          logits=np.array([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]),
          targets=np.array([[1, 2]]),
          weights=np.array([[0.0, 1.0]]),
          expect=(2 + math.e) / math.e
      ),
  )
  def testSoftmaxPerplexity(self, logits, targets, weights, expect):
    actual = perplexity_utils.softmax_perplexity(logits, targets, weights)
    self.assertAlmostEqual(expect, actual, places=5)

if __name__ == '__main__':
  absltest.main()
