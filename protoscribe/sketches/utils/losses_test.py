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
import optax
from protoscribe.sketches.utils import losses


class LossesTest(parameterized.TestCase):

  def test_cosine_distance(self):
    key = jax.random.PRNGKey(0)
    key_a, key_b = jax.random.split(key, num=2)
    shape = (4, 3, 10)
    a = jax.random.uniform(key_a, shape=shape, dtype=jnp.float32)
    b = jax.random.uniform(key_b, shape=shape, dtype=jnp.float32)
    loss = jnp.mean(optax.cosine_distance(a, b))
    self.assertLess(0., loss)  # `a` and `b` are really different.
    loss = jnp.mean(optax.cosine_distance(a, -b))
    self.assertLess(1., loss)  # `a` and `-b` are drastically different.
    loss = jnp.mean(optax.cosine_distance(a, a))
    self.assertAlmostEqual(0, loss, places=6)  # Minimum.
    loss = jnp.mean(optax.cosine_distance(a, -a))
    self.assertAlmostEqual(2., loss, places=6)  # Maximum.

  def test_gini_loss(self):
    array = jnp.array([1, 2, 3, 0, 0, 1, 2, 19, 100])
    loss = losses.gini_loss(array)
    self.assertAlmostEqual(loss, 0.802, places=3)
    array = jnp.array([0] * 9999 + [100])
    loss = losses.gini_loss(array)
    self.assertAlmostEqual(loss, 1.0, places=3)
    array = jnp.array([1] * 10_000)
    loss = losses.gini_loss(array)
    self.assertAlmostEqual(loss, 0.0, places=3)

  def test_joint_embeddings_loss(self):
    key = jax.random.PRNGKey(0)
    key_a, key_b = jax.random.split(key, num=2)
    batch = 2
    seqlen = 5
    v = 20
    logits = jax.random.uniform(
        key_a,
        shape=(batch, seqlen, v),
        dtype=jnp.float32,
    )
    emb = 100
    targets = jax.random.uniform(
        key_a,
        shape=(batch, seqlen, 2, emb),
        dtype=jnp.float32,
    )
    embeddings = jax.random.uniform(key_b, shape=(v, 2, emb), dtype=jnp.float32)
    loss, semantic_loss, phonetic_loss = losses.joint_embeddings_loss(
        logits,
        targets,
        embeddings,
        use_semantic_embedding=True,
        use_phonetic_embedding=True,
        per_sequence_min=False,
    )
    for l in [loss, semantic_loss, phonetic_loss]:
      self.assertGreaterEqual(l, 0)
    loss, semantic_loss, phonetic_loss = losses.joint_embeddings_loss(
        logits,
        targets,
        embeddings,
        use_semantic_embedding=True,
        use_phonetic_embedding=True,
        per_sequence_min=True,
    )
    for l in [loss, semantic_loss, phonetic_loss]:
      self.assertGreaterEqual(l, 0)
    semantic_mask = jnp.array([[0, 0, 1, 0, 0], [0, 1, 0, 0, 0]])
    phonetic_mask = jnp.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0]])
    loss, semantic_loss, phonetic_loss = losses.joint_embeddings_loss(
        logits,
        targets,
        embeddings,
        use_semantic_embedding=True,
        use_phonetic_embedding=True,
        semantic_mask=semantic_mask,
        phonetic_mask=phonetic_mask,
    )
    for l in [loss, semantic_loss, phonetic_loss]:
      self.assertGreaterEqual(l, 0)
    loss, semantic_loss, phonetic_loss = losses.joint_embeddings_loss(
        logits,
        targets,
        embeddings,
        use_semantic_embedding=True,
        use_phonetic_embedding=False,
    )
    for l in [loss, semantic_loss]:
      self.assertGreaterEqual(l, 0)
    self.assertEqual(phonetic_loss, 0)
    loss, semantic_loss, phonetic_loss = losses.joint_embeddings_loss(
        logits,
        targets,
        embeddings,
        use_semantic_embedding=True,
        use_phonetic_embedding=True,
        sum_losses=True,
    )
    for l in [loss, semantic_loss]:
      self.assertGreaterEqual(l, 0)

  def test_long_loss(self):
    key = jax.random.PRNGKey(0)
    batch = 2
    seqlen = 5
    v = 20
    logits = jax.random.uniform(
        key,
        shape=(batch, seqlen, v),
        dtype=jnp.float32,
    )
    mask1 = jnp.array([[0, 0, 1, 0, 0], [0, 1, 1, 0, 0]])
    mask2 = jnp.array([[0, 1, 1, 0, 0], [0, 1, 1, 0, 0]])
    loss1 = losses.long_loss(logits, mask1)
    loss2 = losses.long_loss(logits, mask2)
    self.assertLess(loss2.item(), loss1.item())


if __name__ == "__main__":
  absltest.main()
