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

"""Simple tests for stroke stats."""

import os

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import ml_collections
import numpy as np
from protoscribe.sketches.utils import stroke_stats as lib

FLAGS = flags.FLAGS

_STROKE_LENGTH = 800
_STROKE_DIM = 3
_TRANSLATE = -2.0
_SCALE = 9.0


class StrokeStatsTest(parameterized.TestCase):

  @parameterized.parameters("none", "z-standardize", "min-max", "mean-norm",
                            "sketch-rnn", "det-covar")
  def test_normalization(self, norm_type):
    stats_file = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/sketches/utils",
        "testdata/stroke_stats.json")
    config = ml_collections.ConfigDict()
    self.assertFalse(lib.should_normalize_strokes(config))
    config.stroke_normalization_type = norm_type
    if norm_type == "none":
      self.assertFalse(lib.should_normalize_strokes(config))
    else:
      self.assertTrue(lib.should_normalize_strokes(config))
    stats = lib.load_stroke_stats(config, stats_file)
    x = _TRANSLATE + _SCALE * np.random.rand(_STROKE_LENGTH, _STROKE_DIM)
    y = _TRANSLATE + _SCALE * np.random.rand(_STROKE_LENGTH, _STROKE_DIM)
    new_x, new_y = lib.normalize_strokes(config, stats, x, y)
    if norm_type == "none":
      np.testing.assert_array_equal(new_x, x)
      np.testing.assert_array_equal(new_y, y)
      return
    reconstructed_x, reconstructed_y = lib.denormalize_strokes(
        config, stats, new_x, new_y)
    np.testing.assert_allclose(reconstructed_x, x, rtol=1e-7, atol=0.)
    np.testing.assert_allclose(reconstructed_y, y, rtol=1e-7, atol=0.)


if __name__ == "__main__":
  absltest.main()