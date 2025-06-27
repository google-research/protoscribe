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

from absl.testing import absltest
import numpy as np
from protoscribe.corpus.tools import quantize_sketches_simple as lib

_RANDOM_SEED = 123


class QuantizeSketchesSimpleTest(absltest.TestCase):

  def test_collect_points_basic(self):
    examples = list([
        {
            "strokes": np.array([
                [0., 0., 1., 0., 0.],
                [1., 1., 1., 0., 0.],
                [3., 3., 1., 0., 0.],
                [6., 6., 0., 1., 0.],
                [8., 8., 1., 0., 0.],
                [5., 5., 1., 0., 0.],
                [4., 4., 1., 0., 0.],
                [2., 2., 1., 0., 0.],
                [0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 1.],
            ], dtype=np.float32),
        },
    ])
    points_to_sample = 6
    points = lib.collect_points(
        examples,
        random_seed=_RANDOM_SEED,
        total_num_points=points_to_sample,
        lift_to_touch_ratio=0.
    )
    self.assertEqual(points.shape, (points_to_sample, 2))
    self.assertEqual(points.tolist(), [
        [4.0, 4.0], [5.0, 5.0], [2.0, 2.0], [3.0, 3.0], [1.0, 1.0], [6.0, 6.0]
    ])

  def test_collect_points_lift_touch(self):
    examples = list([
        {
            "strokes": np.array([
                [0., 0., 1., 0., 0.],
                [1., 1., 1., 0., 0.],
                [3., 3., 1., 0., 0.],
                [6., 6., 0., 1., 0.],
                [8., 8., 1., 0., 0.],
                [5., 5., 1., 0., 0.],
                [4., 4., 1., 0., 0.],
                [2., 2., 1., 0., 0.],
                [0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 1.],
            ], dtype=np.float32),
        },
        {
            "strokes": np.array([
                [0., 0., 1., 0., 0.],
                [1.5, 1.5, 1., 0., 0.],
                [3.5, 3.5, 1., 0., 0.],
                [6.5, 6.5, 0., 1., 0.],
                [8.5, 8.5, 1., 0., 0.],
                [5.5, 5.5, 1., 0., 0.],
                [4.5, 4.5, 1., 0., 0.],
                [2.5, 2.5, 1., 0., 0.],
                [0.5, 0.5, 0., 1., 0.],
                [0.1, 0.1, 0., 0., 1.],
                [0., 0., 0., 0., 1.],
            ], dtype=np.float32),
        },
    ])
    points_to_sample = 6
    points = lib.collect_points(
        examples,
        random_seed=_RANDOM_SEED,
        total_num_points=points_to_sample,
        lift_to_touch_ratio=.2,
    )
    self.assertEqual(points.shape, (points_to_sample, 2))
    self.assertEqual(points.tolist(), [
        [1.0, 1.0], [4.5, 4.5], [5.0, 5.0], [0.0, 0.0], [5.5, 5.5], [8.0, 8.0]
    ])


if __name__ == "__main__":
  absltest.main()
