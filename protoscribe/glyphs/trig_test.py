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

import math
import random

from absl.testing import absltest
from protoscribe.glyphs import trig


class TrigTest(absltest.TestCase):

  def _test_angle(self, angle) -> None:
    sin, cos = trig.sin_cos(angle)
    golden_sin, golden_cos = (
        math.sin(math.radians(angle)),
        math.cos(math.radians(angle)),
    )
    self.assertAlmostEqual(sin, golden_sin, places=3)
    self.assertAlmostEqual(cos, golden_cos, places=3)

  def testTrig(self):
    for _ in range(10_000):
      angle = random.random() * 360 - 180
      self._test_angle(angle)


if __name__ == "__main__":
  absltest.main()
