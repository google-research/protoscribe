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

"""Tests for the loss combiner."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import ml_collections
from protoscribe.sketches.utils import loss_combiner as lib


class LossCombinerTest(absltest.TestCase):

  def test_homoscedastic_uncertainty(self):
    num_losses = 10
    losses = [jnp.array(i / 20.) for i in range(num_losses)]
    rngs = {"params": jax.random.PRNGKey(0)}

    loss_combiner = lib.HUWSigmaSquareLossCombiner(num_losses)
    (
        (loss, loss_weights, new_losses),
        state
    ) = loss_combiner.init_with_output(rngs, losses)
    self.assertAlmostEqual(1.125, loss, places=3)
    self.assertEqual(loss_weights.shape, (num_losses,))
    self.assertLen(new_losses, num_losses)
    self.assertIn("loss_weights", state["params"])

    loss_combiner = lib.HUWLogSigmaLossCombiner(num_losses)
    (
        (loss, loss_weights, new_losses),
        state
    ) = loss_combiner.init_with_output(rngs, losses)
    self.assertAlmostEqual(2.25, loss, places=3)
    self.assertEqual(loss_weights.shape, (num_losses,))
    self.assertLen(new_losses, num_losses)
    self.assertIn("loss_weights", state["params"])

    loss_combiner = lib.HUWLiebelKoernerLossCombiner(num_losses)
    (
        (loss, loss_weights, new_losses),
        state
    ) = loss_combiner.init_with_output(rngs, losses)
    self.assertAlmostEqual(8.056, loss, places=3)
    self.assertEqual(loss_weights.shape, (num_losses,))
    self.assertLen(new_losses, num_losses)
    self.assertIn("loss_weights", state["params"])

  def test_factory(self):
    config = ml_collections.ConfigDict()
    combiner = lib.get_loss_combiner(config, num_losses=3)
    self.assertIsNone(combiner)
    config.loss_combiner_type = "None"
    combiner = lib.get_loss_combiner(config, num_losses=3)
    self.assertIsNone(combiner)


if __name__ == "__main__":
  absltest.main()
