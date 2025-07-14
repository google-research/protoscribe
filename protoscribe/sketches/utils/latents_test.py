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
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from protoscribe.sketches.utils import latents


class LatentsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng_key = jax.random.key(0)

  @parameterized.named_parameters(
      (
          "zero-mean_zero_logvar",
          np.zeros([10, 1, 10]),
          np.zeros([10, 1, 10]),
          (0., 0.01),
      ),
      (
          "unity-mean_zero-logvar",
          np.ones([10, 1, 10]),
          np.zeros([10, 1, 10]),
          (50., 50.1),
      ),
      (
          "unity-mean_unity-logvar",
          np.ones([10, 1, 10]),
          np.ones([10, 1, 10]),
          (85.8, 86.),
      )
  )
  def test_kl_gaussian_loss(
      self,
      mean: np.ndarray,
      logvar: np.ndarray,
      target_range: tuple[float, float]
  ):
    target_low, target_high = target_range
    mean = jnp.array(mean, dtype=jnp.float32)
    logvar = jnp.array(logvar, dtype=jnp.float32)
    loss, normalizing_factor = latents.kl_regularization_loss(mean, logvar)
    value = jnp.mean(loss)
    self.assertBetween(value, target_low, target_high)
    self.assertEqual(normalizing_factor, np.prod(mean.shape))

  @parameterized.named_parameters(
      (
          "zero-mean_zero_logvar",
          np.zeros([10, 1, 10]),
          np.zeros([10, 1, 10]),
          np.zeros([10, 1, 10]),
          np.full([10, 1, 10], 0.1),
      ),
      (
          "unity-mean_zero-logvar",
          np.ones([10, 1, 10]),
          np.zeros([10, 1, 10]),
          np.ones([10, 1, 10]),
          np.full([10, 1, 10], 1.1),
      ),
      (
          "unity-mean_unity-logvar",
          np.ones([10, 1, 10]),
          np.ones([10, 1, 10]),
          np.ones([10, 1, 10]),
          np.full([10, 1, 10], 1 + 0.1 * np.exp(0.5)),
      )
  )
  def test_sample_gaussian(
      self,
      mean: np.ndarray,
      logvar: np.ndarray,
      ref_z_no_randomness: np.ndarray,
      ref_z_uniform_eps: np.ndarray,
  ):
    mean = jnp.array(mean, dtype=jnp.float32)
    logvar = jnp.array(logvar, dtype=jnp.float32)

    # Sample deterministically.
    z = latents.sample_gaussian(mean, logvar)
    self.assertEqual(z.shape, (10, 1, 10))
    np.testing.assert_array_equal(np.array(z), ref_z_no_randomness)

    # Uniformly distributed epsilon noise.
    z_eps = jnp.full(mean.shape, 0.1)
    z = latents.sample_gaussian(mean, logvar, z_eps=z_eps)
    np.testing.assert_array_almost_equal(np.array(z), ref_z_uniform_eps)

    # Sample with normally distributed noise.
    z_eps = jax.random.normal(self.rng_key, shape=mean.shape)
    z = latents.sample_gaussian(mean, logvar, z_eps=z_eps)

  @parameterized.named_parameters(
      (
          "none-regular",
          latents.AnnealingType.NONE,
          False,
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
      ),
      (
          "none-cyclical",
          latents.AnnealingType.NONE,
          True,
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
      ),
      (
          "linear-regular",
          latents.AnnealingType.LINEAR,
          False,
          [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0],
      ),
      (
          "linear-cyclical",
          latents.AnnealingType.LINEAR,
          True,
          [0.0, 0.2, 0.4, 0.6, 0.8, 0.0, 0.2, 0.4, 0.6, 0.8],
      ),
      (
          "cosine-regular",
          latents.AnnealingType.COSINE,
          False,
          [0., 0.09549, 0.34549, 0.6545, 0.9045, 1., 1., 1., 1., 1.],
      ),
      (
          "cosine-cyclical",
          latents.AnnealingType.COSINE,
          True,
          [
              0., 0.09549, 0.34549, 0.6545, 0.9045,
              0., 0.09549, 0.34549, 0.6545, 0.9045,
          ],
      ),
      (
          "logistic-regular",
          latents.AnnealingType.LOGISTIC,
          False,
          [
              0.0758, 0.1824, 0.3775, 0.6224, 0.8175,
              0.9241, 0.9241, 0.9241, 0.9241, 0.9241,
          ],
      ),
      (
          "logistic-cyclical",
          latents.AnnealingType.LOGISTIC,
          True,
          [
              0.0758, 0.1824, 0.3775, 0.6224, 0.8175,
              0.0758, 0.1824, 0.3775, 0.6224, 0.8175,
          ],
      ),
  )
  def test_kl_annealing(
      self,
      annealing: latents.AnnealingType,
      is_cyclical: bool,
      reference_weights: list[float]
  ):
    annealer = latents.KLAnnealing(
        annealing_type=annealing,
        total_num_steps=len(reference_weights) // 2,
        cyclical=is_cyclical
    )
    weights = [annealer.step() for _ in range(len(reference_weights))]
    self.assertLen(weights, len(reference_weights))
    for weight, reference_weight in zip(weights, reference_weights):
      self.assertAlmostEqual(weight, reference_weight, places=3)


if __name__ == "__main__":
  absltest.main()
