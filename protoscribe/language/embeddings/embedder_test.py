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

"""Unit tests for `protoscribe.language.embeddings.embedder`."""

import os
import random

from absl import flags
from absl.testing import absltest
import numpy as np
from protoscribe.language.embeddings import embedder

FLAGS = flags.FLAGS

_TEST_DATA_DIR = (
    "protoscribe/language/embeddings/testdata"
)


class MakeTextTestTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.num_emb = os.path.join(
        FLAGS.test_srcdir, f"{_TEST_DATA_DIR}/nums.txt",
    )
    cls.conc_emb = os.path.join(
        FLAGS.test_srcdir, f"{_TEST_DATA_DIR}/conc.txt",
    )
    cls.full_conc_emb = os.path.join(
        FLAGS.test_srcdir, f"{_TEST_DATA_DIR}/embeddings_10K.txt",
    )

  def testBNCGoat(self) -> None:
    bnc = embedder.load_embeddings("bnc", self.num_emb, self.conc_emb)
    self.assertEqual(bnc.dimension, 300)
    text = "2 goat_NOUN"
    embeddings = bnc.embed(text, "bnc")
    self.assertEqual(
        embeddings[0][:10].tolist(),
        [
            0.06174,
            0.063877,
            0.013623,
            0.024748,
            -0.048319,
            -0.040389,
            -0.07687,
            0.077987,
            -0.01947,
            -0.157572,
        ],
    )
    self.assertEqual(
        embeddings[1][:10].tolist(),
        [
            -0.004167,
            -0.016891,
            0.122412,
            -0.1345,
            0.003767,
            -0.049549,
            0.02205,
            0.041371,
            -0.092431,
            0.030102,
        ],
    )

  def testBNCChimpanzee(self) -> None:
    """Checks missing concept."""
    bnc = embedder.load_embeddings("bnc", self.num_emb, self.conc_emb)
    text = "2 chimpanzee_NOUN"
    embeddings = bnc.embed(text, "bnc")
    self.assertEqual(embeddings[1][:10].tolist(), [0.0] * 10)

  def testLoadAll(self) -> None:
    num_random_keys = 20
    for emb_type in embedder.EMBEDDING_TYPES:
      embeddings = embedder.load_embeddings_from_type(emb_type)
      self.assertNotEmpty(embeddings.embeddings)
      concepts = list(embeddings.embeddings.keys())
      for _ in range(num_random_keys):
        concept_id = random.randrange(len(concepts))
        concept_vector = embeddings[concepts[concept_id]]
        self.assertLen(
            concept_vector,
            embedder.embedding_dim_from_type(emb_type),
            f'Embedding: "{emb_type}"',
        )

  def testDistanceOrZero(self) -> None:
    # Tests with the full set of BNC embeddings to make sure this is fast
    # enough.
    bnc = embedder.load_embeddings("bnc", self.num_emb, self.full_conc_emb)
    goat_emb = bnc.embed("goat_NOUN", "bnc")[0]
    self.assertEqual(bnc.distance_or_zero(goat_emb, goat_emb), 0.0)
    goat_near_copy = goat_emb.copy()
    goat_near_copy[100] = goat_near_copy[100] * 2
    self.assertNotEqual(bnc.distance(goat_emb, goat_near_copy), 0.0)
    self.assertEqual(bnc.distance_or_zero(goat_near_copy, goat_emb), 0.0)
    horse_emb = bnc.embed("horse_NOUN", "bnc")[0]
    self.assertEqual(
        bnc.distance_or_zero(horse_emb, goat_emb),
        bnc.distance(horse_emb, goat_emb),
    )
    cow_emb = bnc.embed("cow_NOUN", "bnc")[0]
    self.assertEqual(
        bnc.distance_or_zero(cow_emb, goat_emb),
        bnc.distance(cow_emb, goat_emb),
    )
    fella_emb = bnc.embed("fella_NOUN", "bnc")[0]
    self.assertEqual(
        bnc.distance_or_zero(fella_emb, goat_emb),
        bnc.distance(fella_emb, goat_emb),
    )
    cheese_emb = bnc.embed("cheese_NOUN", "bnc")[0]
    milk_emb = bnc.embed("milk_NOUN", "bnc")[0]
    cheese_near_copy = cheese_emb.copy()
    cheese_near_copy[100] = cheese_near_copy[100] * 2
    self.assertNotEqual(bnc.distance(cheese_emb, cheese_near_copy), 0.0)
    self.assertEqual(bnc.distance_or_zero(cheese_near_copy, cheese_emb), 0.0)
    self.assertEqual(
        bnc.distance_or_zero(cheese_near_copy, milk_emb),
        bnc.distance(cheese_near_copy, milk_emb),
    )

  def testKNearestNeighbors(self) -> None:
    bnc = embedder.load_embeddings("bnc", self.num_emb, self.full_conc_emb)
    cow_emb = bnc.embed("cow_NOUN", "bnc")[0]
    allowed_list = set([
        "idiosyncratic_ADJ",
        "monkey_NOUN",
        "milk_NOUN",
        "inflation_NOUN",
        "dog_NOUN",
        "ocean_NOUN",
        "biological_ADJ"
    ])
    top_k = bnc.get_k_nearest_neighbours(
        cow_emb, k=3, allowed_entries=allowed_list
    )
    self.assertLen(top_k, 3)
    self.assertEqual(top_k[0][0], "milk_NOUN")
    self.assertEqual(top_k[1][0], "dog_NOUN")
    self.assertEqual(top_k[2][0], "monkey_NOUN")

  def testNorms(self) -> None:
    bnc = embedder.load_embeddings("bnc", self.num_emb, self.full_conc_emb)
    for concept in bnc.embeddings:
      vec = np.array(bnc.embeddings[concept], dtype=np.float32)
      l2_norm = np.linalg.norm(vec, ord=2)
      self.assertAlmostEqual(l2_norm, 1., delta=1e-5)


if __name__ == "__main__":
  absltest.main()
