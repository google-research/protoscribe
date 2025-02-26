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

"""Test for glyph embeddings."""

import os

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from protoscribe.glyphs import glyph_vocab as glyph_lib
from protoscribe.language.embeddings import embedder
from protoscribe.sketches.utils import glyph_embeddings as lib

FLAGS = flags.FLAGS

_VOCAB_SIZE = 314


def matmul(one_hot, embeddings):
  """Wrapper for einsum.

  Args:
    one_hot: a [V] array
    embeddings: a [V, 2, emb_size] array

  Returns:
    The einsum of one_hot and embeddings, a [2, emb_size] array.
  """
  return jnp.einsum("i,ijk->jk", one_hot, embeddings)


class GlyphEmbeddingsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    glyph_vocab_file = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/sketches/utils",
        "testdata/glyph_vocab.json",
    )
    self.glyph_vocab = glyph_lib.load_glyph_vocab(glyph_vocab_file)

  def _load_real_embeddings(self) -> jnp.ndarray:
    config = ml_collections.FrozenConfigDict({
        "concept_embedding_type": "bnc",
        "main_lexicon": os.path.join(
            FLAGS.test_srcdir,
            "protoscribe/sketches/utils",
            "testdata/lexicon.tsv",
        ),
        "number_lexicon": os.path.join(
            FLAGS.test_srcdir,
            "protoscribe/sketches/utils",
            "testdata/number_lexicon.tsv",
        ),
        "phonetic_embeddings": os.path.join(
            FLAGS.test_srcdir,
            "protoscribe/sketches/utils",
            "testdata/phonetic_embeddings.tsv",
        ),
    })
    return lib.glyphs_to_embeddings(config, self.glyph_vocab)

  def _best_k(self, probs: jnp.ndarray, k: int = 3) -> list[str]:
    """Checks k closest glyphs to the given one according to semantics."""
    closest = np.argpartition(probs, -k)[-k:]
    return [self.glyph_vocab.id_to_name(idx) for idx in closest]

  def test_phonetic_embeddings(self):
    embeddings = self._load_real_embeddings()
    self.assertSequenceEqual(
        embeddings.shape, [_VOCAB_SIZE, 2, embedder.DEFAULT_EMBEDDING_DIM]
    )

    # "cook" = index 70
    cook = matmul(jax.nn.one_hot(70, _VOCAB_SIZE), embeddings)
    # Phonetic
    cook_first_five = [
        0.0,
        0.00417287,
        0.00208644,
        0.00685544,
        0.00387481,
    ]
    self.assertAlmostEqual(cook[1][0], cook_first_five[0], places=3)
    self.assertAlmostEqual(cook[1][1], cook_first_five[1], places=3)
    self.assertAlmostEqual(cook[1][2], cook_first_five[2], places=3)
    self.assertAlmostEqual(cook[1][3], cook_first_five[3], places=3)
    self.assertAlmostEqual(cook[1][4], cook_first_five[4], places=3)

    # Semantic
    cook_first_five = [-0.062899, 0.015054, 0.077151, -0.052424, -0.137374]
    self.assertAlmostEqual(cook[0][0], cook_first_five[0], places=3)
    self.assertAlmostEqual(cook[0][1], cook_first_five[1], places=3)
    self.assertAlmostEqual(cook[0][2], cook_first_five[2], places=3)
    self.assertAlmostEqual(cook[0][3], cook_first_five[3], places=3)
    self.assertAlmostEqual(cook[0][4], cook_first_five[4], places=3)

    # "I", the Roman numeral, index 311
    i = matmul(jax.nn.one_hot(311, _VOCAB_SIZE), embeddings)
    i_first_five = [1.0, 0.0, 0.0, 0.0, 0.0]
    # Semantic
    self.assertAlmostEqual(i[0][0], i_first_five[0], places=3)
    self.assertAlmostEqual(i[0][1], i_first_five[1], places=3)
    self.assertAlmostEqual(i[0][2], i_first_five[2], places=3)
    self.assertAlmostEqual(i[0][3], i_first_five[3], places=3)
    self.assertAlmostEqual(i[0][4], i_first_five[4], places=3)
    # Phonetic
    self.assertAlmostEqual(i[1][0], i_first_five[0], places=3)
    self.assertAlmostEqual(i[1][1], i_first_five[1], places=3)
    self.assertAlmostEqual(i[1][2], i_first_five[2], places=3)
    self.assertAlmostEqual(i[1][3], i_first_five[3], places=3)
    self.assertAlmostEqual(i[1][4], i_first_five[4], places=3)

  def test_construct_batch_glyph_embeddings(self):
    # Batch size is 3, embedding length is 5
    batch_glyph_embeddings = jnp.array([
        [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]],
        [[6, 7, 8, 9, 10], [10, 9, 8, 7, 6]],
        [[2, 4, 6, 8, 10], [10, 8, 6, 4, 2]],
    ])
    # Sequence length is 4
    glyph_types = jnp.array([[0, 1, 2, 0], [0, 2, 2, 2], [2, 2, 2, 1]])

    targets = lib.construct_batch_glyph_embeddings(
        batch_glyph_embeddings,
        glyph_types,
    )
    golden_targets = [
        [
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
            [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]],
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
        ],
        [
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
            [[6, 7, 8, 9, 10], [10, 9, 8, 7, 6]],
            [[6, 7, 8, 9, 10], [10, 9, 8, 7, 6]],
            [[6, 7, 8, 9, 10], [10, 9, 8, 7, 6]],
        ],
        [
            [[2, 4, 6, 8, 10], [10, 8, 6, 4, 2]],
            [[2, 4, 6, 8, 10], [10, 8, 6, 4, 2]],
            [[2, 4, 6, 8, 10], [10, 8, 6, 4, 2]],
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
        ],
    ]
    self.assertEqual(targets.tolist(), golden_targets)

  def test_construct_batch_glyph_embeddings_long_sequence(self):
    # Same as above, except that now we assume a long sequence of stroke-based
    # glyph targets (sketch_xformer_model with stroke-glyph affiliations).
    # Batch size is 3, embedding length is 5
    batch_glyph_embeddings = jnp.array([
        [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]],
        [[6, 7, 8, 9, 10], [10, 9, 8, 7, 6]],
        [[2, 4, 6, 8, 10], [10, 8, 6, 4, 2]],
    ])
    # Sequence length is 4
    glyph_types = jnp.array([
        [0, 1, 1, 1, 2, 2, 0, 2, 2, 0],
        [0, 2, 2, 0, 2, 2, 1, 1, 1, 0],
        [2, 2, 2, 0, 2, 2, 2, 1, 0, 1],
    ])

    targets = lib.construct_batch_glyph_embeddings(
        batch_glyph_embeddings,
        glyph_types,
    )
    golden_targets = [
        [
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
            [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]],
            [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]],
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
            [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]],
            [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]],
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
        ],
        [
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
            [[6, 7, 8, 9, 10], [10, 9, 8, 7, 6]],
            [[6, 7, 8, 9, 10], [10, 9, 8, 7, 6]],
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
            [[6, 7, 8, 9, 10], [10, 9, 8, 7, 6]],
            [[6, 7, 8, 9, 10], [10, 9, 8, 7, 6]],
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
        ],
        [
            [[2, 4, 6, 8, 10], [10, 8, 6, 4, 2]],
            [[2, 4, 6, 8, 10], [10, 8, 6, 4, 2]],
            [[2, 4, 6, 8, 10], [10, 8, 6, 4, 2]],
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
            [[2, 4, 6, 8, 10], [10, 8, 6, 4, 2]],
            [[2, 4, 6, 8, 10], [10, 8, 6, 4, 2]],
            [[2, 4, 6, 8, 10], [10, 8, 6, 4, 2]],
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
        ],
    ]
    # Show that the same thing works for a mask as returned by
    # construct_loss_mask_for_stroke_glyph_affiliations, multiplied by 2 to
    # identify the "real" glyphs.
    glyph_types = (
        jnp.array([
            [0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
            [0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
        ])
        * 2
    )

    targets = lib.construct_batch_glyph_embeddings(
        batch_glyph_embeddings,
        glyph_types,
    )
    self.assertEqual(targets.tolist(), golden_targets)

  def test_construct_loss_mask_for_stroke_glyph_affiliations(self):
    # Batch size is 2
    # Sequence length is 7
    # Stroke sequence length is 40
    glyphs = jnp.array([
        [1, 312, 312, 311, 309, 28, 2],
        [1, 311, 311, 42, 92, 2, 0],
    ])
    glyph_types = jnp.array([
        [0, 1, 1, 1, 1, 2, 0],
        [0, 1, 1, 2, 2, 0, 0],
    ])
    # pyformat: disable
    stroke_glyph_affiliations = jnp.array([
        [312, 312,
         1_000_000,
         312, 312,
         1_000_000,
         312, 312,
         1_000_000,
         312, 312,
         1_000_000,
         311, 311, 311,
         1_000_000,
         309, 309,
         1_000_000,
         28, 28, 28, 28, 28, 28,
         1_000_000,
         28, 28, 28, 28, 28, 28, 28,
         1_000_000,
         28, 28,
         1_000_000,
         0, 0, 0,
        ],
        [311, 311,
         1_000_000,
         311, 311,
         1_000_000,
         42, 42, 42, 42,
         1_000_000,
         42, 42,
         1_000_000,
         92, 92, 92, 92, 92, 92,
         1_000_000,
         92, 92,
         1_000_000,
         92, 92,
         1_000_000,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         ],
    ])
    # pyformat: enable
    mask = lib.construct_loss_mask_for_stroke_glyph_affiliations(
        glyphs,
        glyph_types,
        stroke_glyph_affiliations,
    )
    # pyformat: disable
    golden_mask = [
        [0, 0,  # 312, 312,
         0,  # 1_000_000,
         0, 0,  # 312, 312,
         0,  # 1_000_000,
         0, 0,  # 312, 312,
         0,  # 1_000_000,
         0, 0,  # 312, 312,
         0,  # 1_000_000,
         0, 0, 0,  # 311, 311, 311,
         0,  # 1_000_000,
         0, 0,  # 309, 309,
         0,  # 1_000_000,
         1, 1, 1, 1, 1, 1,  # 28, 28, 28, 28, 28, 28,
         0,  # 1_000_000,
         1, 1, 1, 1, 1, 1, 1,  # 28, 28, 28, 28, 28, 28, 28,
         0,  # 1_000_000,
         1, 1,  # 28, 28,
         0,  # 1_000_000,
         0, 0, 0,
        ],
        [0, 0,  # 311, 311,
         0,  # 1_000_000,
         0, 0,  # 311, 311,
         0,  # 1_000_000,
         1, 1, 1, 1,  # 42, 42, 42, 42,
         0,  # 1_000_000,
         1, 1,  # 42, 42,
         0,  # 1_000_000,
         1, 1, 1, 1, 1, 1,  # 92, 92, 92, 92, 92, 92,
         0,  # 1_000_000,
         1, 1,  # 92, 92,
         0,  # 1_000_000,
         1, 1,  # 92, 92,
         0,  # 1_000_000,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         ],
    ]
    # pyformat: enable
    self.assertEqual(mask.tolist(), golden_mask)

  def test_construct_loss_mask(self):
    # Batch size is 3
    # Sequence length is 4
    glyph_types = jnp.array([[0, 1, 2, 0], [0, 2, 2, 2], [2, 2, 2, 1]])
    mask = lib.construct_loss_mask(glyph_types)
    golden_mask = [[0, 0, 1, 0], [0, 1, 1, 1], [1, 1, 1, 0]]
    self.assertEqual(mask.tolist(), golden_mask)
    mask = lib.construct_loss_mask(glyph_types, concepts=False)
    golden_mask = [[1, 1, 0, 1], [1, 0, 0, 0], [0, 0, 0, 1]]
    self.assertEqual(mask.tolist(), golden_mask)

  def test_construct_targeted_sem_phon_loss_mask(self):
    # Batch size is 3
    # Sequence length is 4
    mask = jnp.array([[0, 0, 1, 0], [0, 1, 1, 1], [1, 1, 1, 0]])
    sem_golden, phon_golden = (
        [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
        [[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    )
    sem, phon = lib.construct_targeted_sem_phon_loss_mask(mask)
    self.assertEqual(sem.tolist(), sem_golden)
    self.assertEqual(phon.tolist(), phon_golden)

  @parameterized.parameters(0.5, 1.0, 1.5)
  def test_pairwise_cosine_similarity(self, temperature):
    rng = jax.random.PRNGKey(0)
    vocab_size = 8
    embeddings = jax.random.normal(rng, shape=(vocab_size, 4))
    sim = lib.pairwise_cosine_similarity(
        embeddings, self.glyph_vocab, temperature=temperature
    )
    self.assertTupleEqual(sim.shape, (vocab_size, vocab_size))

  def test_pairwise_cosine_similarity_with_valid_diagonal(self):
    rng = jax.random.PRNGKey(0)
    vocab_size = 8
    embeddings = jax.random.normal(rng, shape=(vocab_size, 4))
    sim = lib.pairwise_cosine_similarity(
        embeddings, self.glyph_vocab, mask_diagonal=False
    )
    self.assertTupleEqual(sim.shape, (vocab_size, vocab_size))
    self.assertAlmostEqual(jnp.sum(sim), vocab_size, places=2)
    sim = lib.pairwise_cosine_similarity(
        embeddings, self.glyph_vocab, mask_diagonal=False,
        similarity_to_self=2.0
    )
    self.assertTupleEqual(sim.shape, (vocab_size, vocab_size))
    self.assertAlmostEqual(jnp.sum(sim), vocab_size, places=2)

  def test_real_closest_glyphs(self):
    embeddings = self._load_real_embeddings()
    vocab_size = embeddings.shape[0]
    sim = lib.build_closest_glyphs(embeddings, self.glyph_vocab, temperature=1.)
    self.assertTupleEqual(sim.shape, (vocab_size, 2, vocab_size))
    self.assertAlmostEqual(jnp.sum(sim), 2.0 * vocab_size, places=2)
    # Check special token "<pad>" = index 0. These are equidistant from
    # everything else which is fine because we zero them out later on in loss
    # computation.
    self.assertEqual(sim[0, 0, 0], 1. / vocab_size)
    self.assertEqual(sim[0, 1, 0], 1. / vocab_size)
    # Check semantically closest concepts to "cook" (id=70) in order of
    # increasing similarity.
    self.assertEqual(sim[70, 0, 0], 0.0)  # Prob of special token (semantics).
    self.assertEqual(sim[70, 1, 0], 0.0)  # Prob of special token (phonetics).
    closest = self._best_k(sim[70, 0, :], k=3)
    self.assertListEqual(["servant", "vegetable", "baker"], closest)
    # Check semantic neighborhood of "pony" (id=202).
    self.assertEqual(sim[202, 0, 2], 0.0)  # Prob of special token (semantics).
    self.assertEqual(sim[202, 1, 2], 0.0)  # Prob of special token (phonetics).
    closest = self._best_k(sim[202, 0, :], k=5)
    self.assertListEqual(["mare", "dog", "donkey", "cart", "horse"], closest)

  @parameterized.parameters(
      # Semantics.
      (0, True, ["servant", "vegetable", "baker"], ["donkey", "cart", "horse"]),
      (0, False, ["vegetable", "baker", "cook"], ["cart", "pony", "horse"]),
      # Phonetics.
      (1, True, ["lamb", "tree", "milk"], ["loaf", "tool", "hall"]),
      # TODO: Following is a "feature" that may be interpreted as a
      # potential bug. When diagonal is not masked the closest (phonetically)
      # concept to a given one is not necessarily itself because there may be
      # more than glyphs with the same pronunciation. So the output in this case
      # is as follows:
      #   (1, False, ["milk", "lamb", "cook"], ["hall", "pony", "tool"]),
      # Note the disappearing "tree" in the first list and "tool" being inserted
      # in the second.
  )
  def test_closest_glyphs_soft_targets(
      self, modality, mask_diagonal, close_to_cook, close_to_pony
  ):
    embeddings = self._load_real_embeddings()
    vocab_size = embeddings.shape[0]
    close_glyphs = lib.build_closest_glyphs(
        embeddings, self.glyph_vocab, mask_diagonal=mask_diagonal
    )
    self.assertTupleEqual(close_glyphs.shape, (vocab_size, 2, vocab_size))

    targets = jnp.array([
        [1, 309, 309, 70, 2, 0, 0],     # <bos> I I cook <eos> <pad> <pad>.
        [1, 312, 312, 312, 202, 2, 0],  # <bos> X X X pony <eos> <pad>.
        [1, 310, 70, 202, 2, 0, 0],     # <bos> L cook pony <eos> <pad> <pad>.
    ], dtype=jnp.int32)
    batch_size = targets.shape[0]
    seq_len = targets.shape[1]
    soft_targets = lib.closest_glyphs_soft_targets(targets, close_glyphs)
    self.assertTupleEqual(
        soft_targets.shape, (batch_size, seq_len, 2, vocab_size)
    )
    self.assertAlmostEqual(
        jnp.sum(soft_targets), 2.0 * batch_size * seq_len, places=2
    )

    m = modality
    self.assertAlmostEqual(np.sum(soft_targets[0, 3, m]), 1.0, places=4)
    np.testing.assert_array_equal(soft_targets[0, 3, m], soft_targets[2, 2, m])
    closest = self._best_k(soft_targets[0, 3, m])
    self.assertListEqual(close_to_cook, closest)
    self.assertEqual(soft_targets[0, 3, m, 0], 0.0)  # <pad>.
    self.assertEqual(soft_targets[0, 3, m, 1], 0.0)  # <bos>.
    self.assertEqual(soft_targets[0, 3, m, 2], 0.0)  # <eos>.
    self.assertEqual(soft_targets[0, 3, m, 309], 0.0)  # I.
    if mask_diagonal:
      self.assertEqual(soft_targets[0, 3, m, 70], 0.0)  # cook (self).
    else:
      self.assertEqual(
          soft_targets[0, 3, m, 70], np.max(soft_targets[0, 3, m, :])
      )

    # Check "pony".
    self.assertAlmostEqual(np.sum(soft_targets[1, 4, m]), 1.0, places=4)
    np.testing.assert_array_equal(soft_targets[1, 4, m], soft_targets[2, 3, m])
    closest = self._best_k(soft_targets[1, 4, m])  # Pony.
    self.assertListEqual(close_to_pony, closest)
    self.assertEqual(soft_targets[1, 4, m, 0], 0.0)  # <pad>.
    self.assertEqual(soft_targets[1, 4, m, 1], 0.0)  # <bos>.
    self.assertEqual(soft_targets[1, 4, m, 2], 0.0)  # <eos>.
    self.assertEqual(soft_targets[1, 4, m, 312], 0.0)  # X.
    if mask_diagonal:
      self.assertEqual(soft_targets[1, 4, m, 202], 0.0)  # pony (self).
    else:
      self.assertEqual(
          soft_targets[1, 4, 0, 202], np.max(soft_targets[1, 4, m, :])
      )


if __name__ == "__main__":
  absltest.main()
