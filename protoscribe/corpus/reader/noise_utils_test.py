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

"""Tests for noise utilities."""

import numpy as np
from protoscribe.corpus.reader import noise_utils as lib
import tensorflow as tf

_RANDOM_SEED = 42
_BALL_RADIUS = 1.
_NUM_SAMPLES = 50
_SAMPLE_DIM = 300


class NoiseUtilsTest(tf.test.TestCase):

  def test_order_inf(self):
    for _ in range(_NUM_SAMPLES):
      val = lib.tf_random_lp_ball_vector(
          shape=[_SAMPLE_DIM], order="INF", radius=_BALL_RADIUS,
          seed=_RANDOM_SEED
      ).numpy()
      self.assertEqual(val.shape, (_SAMPLE_DIM,))
      np.testing.assert_array_less(x=val, y=_BALL_RADIUS + 1e-8)
      np.testing.assert_array_less(x=-_BALL_RADIUS - 1e-8, y=val)

    val = lib.tf_random_lp_ball_vector(
        shape=[10, 5, 2], order="INF", radius=_BALL_RADIUS, seed=_RANDOM_SEED
    ).numpy().ravel()
    self.assertAlmostEqual(
        np.linalg.norm(val, ord=np.inf), 0.9, delta=0.1
    )

  def test_order_l1(self):
    for _ in range(_NUM_SAMPLES):
      val = lib.tf_random_lp_ball_vector(
          shape=[_SAMPLE_DIM], order="1", radius=_BALL_RADIUS, seed=_RANDOM_SEED
      ).numpy()
      np.testing.assert_array_less(x=val, y=_BALL_RADIUS + 1e-8)
      np.testing.assert_array_less(x=-_BALL_RADIUS - 1e-8, y=val)
      self.assertAlmostEqual(
          np.linalg.norm(val, ord=1), 0.99, delta=0.1
      )

  def test_order_l2(self):
    for _ in range(_NUM_SAMPLES):
      val = lib.tf_random_lp_ball_vector(
          shape=[_SAMPLE_DIM], order="2", radius=_BALL_RADIUS, seed=_RANDOM_SEED
      ).numpy()
      np.testing.assert_array_less(x=val, y=_BALL_RADIUS + 1e-8)
      np.testing.assert_array_less(x=-_BALL_RADIUS - 1e-8, y=val)
      self.assertAlmostEqual(
          np.linalg.norm(val, ord=2), 0.99, delta=0.1
      )

    val = lib.tf_random_lp_ball_vector(
        shape=[32, 64], order="2", radius=_BALL_RADIUS, seed=_RANDOM_SEED
    ).numpy().ravel()
    np.testing.assert_almost_equal(
        np.linalg.norm(val, ord=2), 0.99, decimal=1
    )

if __name__ == "__main__":
  tf.test.main()
