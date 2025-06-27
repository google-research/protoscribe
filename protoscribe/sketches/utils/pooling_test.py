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

"""Tests for the pooling utilities."""

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from protoscribe.sketches.utils import pooling as lib


class PoolingTest(absltest.TestCase):

  def test_pooling(self):
    encoded_inputs = jnp.array([
        [[0.2, 0.4], [0.22, 0.42], [0.23, 0.43], [0.24, 0.44]],
        [[0.3, 0.6], [-0.32, 0.62], [0.33, -0.63], [-0.34, -0.64]],
        [[-0.4, 0.8], [0.42, -0.82], [-0.43, 0.83], [0.44, -0.84]],
    ], dtype=jnp.float32)
    input_masks = jnp.array([
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 0, 0],
    ], dtype=jnp.int32)

    encodings = lib.get_pooling("mean", encoded_inputs, input_masks)
    np.testing.assert_array_almost_equal(
        encodings,
        jnp.array([
            [0.216667, 0.416667],
            [-0.0075, -0.0125],
            [0.01, -0.01],
        ], dtype=jnp.float32),
    )

    encodings = lib.get_pooling("max", encoded_inputs, input_masks)
    np.testing.assert_array_equal(
        encodings,
        jnp.array([
            [0.23, 0.43],
            [0.33, 0.62],
            [0.42, 0.8],
        ], dtype=jnp.float32),
    )

    encodings = lib.get_pooling("first", encoded_inputs, input_masks)
    np.testing.assert_array_equal(
        encodings,
        jnp.array([
            [0.2, 0.4],
            [0.3, 0.6],
            [-0.4, 0.8],
        ], dtype=jnp.float32),
    )

    encodings = lib.get_pooling("last", encoded_inputs, input_masks)
    np.testing.assert_array_equal(
        encodings,
        jnp.array([
            [0.23, 0.43],
            [-0.34, -0.64],
            [0.42, -0.82],
        ], dtype=jnp.float32),
    )


if __name__ == "__main__":
  absltest.main()
