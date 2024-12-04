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

"""Utilities for testing."""

import os

from absl import flags
from protoscribe.corpus.builder import document_builder

FLAGS = flags.FLAGS

_SRC_DIR = "protoscribe"
_CONCEPTS_DIR = "data/concepts"
_TEST_DATA_DIR = "corpus/builder/testdata"


def init_document_builder_params() -> document_builder.Params:
  """Initializes document builder parameters using test data.

  Returns:
    An instance of initialized parameters.
  """
  src_dir = os.path.join(FLAGS.test_srcdir, _SRC_DIR)
  FLAGS.concepts = [
      os.path.join(
          src_dir, _CONCEPTS_DIR, "administrative_categories.txt"
      ),
  ]
  FLAGS.unseen_concepts = [
      os.path.join(
          src_dir, _CONCEPTS_DIR, "non_administrative_categories.txt"
      ),
  ]
  FLAGS.main_lexicon = os.path.join(
      src_dir, _TEST_DATA_DIR, "lexicon.tsv"
  )
  FLAGS.number_lexicon = os.path.join(
      src_dir, _TEST_DATA_DIR, "number_lexicon.tsv"
  )
  FLAGS.number_config_file = os.path.join(
      src_dir, _TEST_DATA_DIR, "number_config_sg.textproto"
  )
  FLAGS.morphology_params = os.path.join(
      src_dir, _TEST_DATA_DIR, "morphology_params.textproto"
  )
  FLAGS.affix_lexicon = os.path.join(
      src_dir, _TEST_DATA_DIR, "affixes.tsv"
  )
  FLAGS.glyph_vocab_file = os.path.join(
      src_dir, _TEST_DATA_DIR, "glyph_vocab.json"
  )
  phonetic_embeddings_path = os.path.join(
      src_dir, _TEST_DATA_DIR, "phonetic_embeddings.tsv"
  )
  concepts_vocab_file = os.path.join(FLAGS.test_tmpdir, "concept_vocab.json")

  return document_builder.init_params(
      phonetic_embeddings_path=phonetic_embeddings_path,
      concepts_vocab_file=concepts_vocab_file
  )
