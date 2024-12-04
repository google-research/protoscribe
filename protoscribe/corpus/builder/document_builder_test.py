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

"""Very simple unit test for single document builder."""

import logging
import random

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from protoscribe.corpus.builder import document_builder as lib
from protoscribe.corpus.builder import test_utils
import tensorflow as tf

FLAGS = flags.FLAGS

_MAX_DOC_IDS = 1000

_SRC_DIR = "protoscribe"


def _as_string(doc: tf.train.Example, name: str) -> str:
  """Returns a particular feature as string.

  Args:
    doc: Actual document.
    name: Name of the string feature.

  Returns:
    Feature value as string.
  """
  return doc.features.feature[name].bytes_list.value[0].decode("utf-8")


def _as_ints(doc: tf.train.Example, name: str) -> list[int]:
  """Returns feature as a list of ints.

  Args:
    doc: Actual document.
    name: Name of the string feature.

  Returns:
    Feature value as a list of ints.
  """
  return doc.features.feature[name].int64_list.value


def _as_floats(doc: tf.train.Example, name: str) -> list[float]:
  """Returns feature as a list of floats.

  Args:
    doc: Actual document.
    name: Name of the string feature.

  Returns:
    Feature value as a list of floats.
  """
  return doc.features.feature[name].float_list.value


class DocumentBuilderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self._params = test_utils.init_document_builder_params()

  def _generate_doc(self, concept: str) -> tf.train.Example:
    """Generates single document using current settings.

    Args:
      concept: Name of the concept for which document is needed.

    Returns:
      Actual document.
    """
    logging.info("Generating document for `%s` ...", concept)
    self.assertIsNotNone(self._params)
    doc_id = random.randint(0, _MAX_DOC_IDS)
    new_doc_id, doc, stroke_stats = lib.build_tf_example(
        doc_id=doc_id,
        concept=concept,
        params=self._params
    )
    self.assertEqual(new_doc_id, doc_id)
    self.assertIsNotNone(doc)
    self.assertIsNotNone(stroke_stats)
    return doc

  @parameterized.named_parameters(
      # Administrative categories.
      ("admin-bee", "bee_NOUN", True),
      ("admin-grape", "grape_NOUN", True),
      ("admin-wolf", "wolf_NOUN", True),
      # Non-administrative categories.
      ("non_admin-barge", "barge_NOUN", False),
      ("non_admin-foot", "foot_NOUN", False),
      ("non_admin-worship", "worship_VERB", False),
  )
  def test_simple(self, concept: str, in_domain: bool) -> None:
    # Note on sketches: For out-of-domain sketches we still generate valid
    # strokes using the placeholder `DUMMY` glyph.
    expected_names = [
        "doc/id",
        "concept/name",
        "concept/unseen",
        "concept/id",
        "number/name",
        "strokes/glyph_affiliations/ids",
        "strokes/glyph_affiliations/text_pos",
        "strokes/x_stroke_points",
        "strokes/y_stroke_points",
        "text/glyph/tokens",
        "text/glyph/types",
        "text/sampa",
        "text/words",
        "text/gloss",
        "text/bnc/concept_emb",
        "text/bnc/number_emb",
        "text/phonetic_embedding/emb",
    ]
    doc = self._generate_doc(concept)
    features = doc.features.feature
    self.assertNotEmpty(features)
    for name in expected_names:
      self.assertIn(name, features)

    self.assertEqual(_as_string(doc, "concept/name"), concept)

    unseen_feat = _as_ints(doc, "concept/unseen")
    self.assertLen(unseen_feat, 1)
    self.assertBetween(unseen_feat[0], 0, 1)
    self.assertEqual(bool(unseen_feat[0]), not in_domain)

    self.assertIsNotNone(_as_string(doc, "number/name"))
    self.assertIsNotNone(_as_string(doc, "text/gloss"))
    self.assertGreater(len(_as_ints(doc, "text/glyph/tokens")), 1)
    self.assertGreater(len(_as_ints(doc, "text/glyph/types")), 1)
    self.assertGreater(len(_as_floats(doc, "strokes/x_stroke_points")), 1)
    self.assertGreater(len(_as_floats(doc, "strokes/y_stroke_points")), 1)
    self.assertGreater(len(_as_ints(doc, "strokes/glyph_affiliations/ids")), 1)
    self.assertGreater(
        len(_as_ints(doc, "strokes/glyph_affiliations/text_pos")), 1
    )

    sampa = _as_string(doc, "text/sampa")
    self.assertGreater(len(sampa.split()), 1)
    words = _as_string(doc, "text/words")
    self.assertGreater(len(words.split()), 1)


if __name__ == "__main__":
  absltest.main()
