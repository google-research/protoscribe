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

"""Simple test for losses."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from protoscribe.sketches.utils import continuous_bernoulli as lib


class LossesTest(parameterized.TestCase):

  @parameterized.parameters(
      (5), (10), (30), (60),
  )
  def test_clamp_probs(self, feature_dim: int):
    rng_key = jax.random.PRNGKey(42)
    batch = 2
    seqlen = 5
    eps = 1e-5
    logits = -2. * jax.random.normal(
        rng_key, shape=(batch, seqlen, feature_dim), dtype=jnp.float32
    )
    probs = lib.clamp_probs(
        jax.nn.softmax(logits, axis=-1), eps=eps
    )
    self.assertLessEqual(np.max(probs), 1. - eps)
    self.assertGreaterEqual(np.min(probs), eps)

  @parameterized.parameters(
      (5, 5), (10, 5), (100, 4), (500, 3)
  )
  def test_continuous_bernoulli(
      self, feature_dim: int, precision: int
  ):
    rng_key = jax.random.PRNGKey(42)
    batch = 2
    seqlen = 5
    logits = jax.random.uniform(
        rng_key,
        minval=-4.0,
        maxval=4.0,
        shape=(batch, seqlen, feature_dim),
        dtype=jnp.float32,
    )
    probs = lib.clamp_probs(jax.nn.softmax(logits, axis=-1))
    log_norm_c = lib.cb_log_norm_const(probs)
    self.assertTupleEqual(log_norm_c.shape, (batch, seqlen, feature_dim))

    def _log_norm_const_paper() -> jnp.ndarray:
      r"""Log normalizing constant of continuous categorical distribution.

      See $C(\lambda)$ in Equation (7) of:
        Gabriel Loaiza-Ganem and John P. Cunningham: `The continuous Bernoulli:
        fixing a pervasive error in variational autoencoders.`
        https://arxiv.org/pdf/1907.06845

      Returns:
        Array with the same shape as `probs`.
      """
      # Do not evaluate close to 0.5.
      return np.float32(
          np.log(np.abs(2.0 * np.arctanh(1.0 - 2.0 * probs))) - np.log(
              np.abs(1.0 - 2.0 * probs)))

    # Make sure the (almost) numerically stable version computed above matches
    # the exact but unstable one from the paper.
    log_norm_c_paper = _log_norm_const_paper()
    np.testing.assert_array_almost_equal(
        log_norm_c, log_norm_c_paper, decimal=precision
    )

    # Check loss.
    target_probs = jax.random.uniform(
        rng_key,
        shape=(batch, seqlen, feature_dim),
        dtype=jnp.float32,
    )
    loss = lib.cb_cross_entropy_with_logits(logits, target_probs)
    self.assertTupleEqual(loss.shape, (batch, seqlen))

    # Check KL divergence.
    kl_p_q = lib.cb_kl(logits, target_probs)
    self.assertTupleEqual(kl_p_q.shape, (batch, seqlen, feature_dim))
    self.assertGreater(jnp.min(kl_p_q), 0.)
    kl_q_p = lib.cb_kl(jnp.log(target_probs), jax.nn.softmax(logits, axis=-1))
    self.assertGreater(jnp.min(kl_q_p), 0.)


if __name__ == "__main__":
  absltest.main()
