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

"""Simple tests for miscelaneous system evolution utilities."""

import os

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from protoscribe.evolution import confidence_pruning
from protoscribe.evolution import new_spellings_utils as lib

FLAGS = flags.FLAGS

_TEST_DATA_DIR = "protoscribe/evolution/testdata"

# This is the total number of unique concepts found among the results.
_TOTAL_NUM_CONCEPTS = 468


class NewSpellingsUtilsTest(absltest.TestCase):

  @flagsaver.flagsaver
  def test_load_and_prune_semantics_and_phonetics(self):
    FLAGS.semantic_jsonl_file = os.path.join(
        FLAGS.test_srcdir, _TEST_DATA_DIR, "semantic_results.jsonl"
    )
    FLAGS.phonetic_jsonl_file = os.path.join(
        FLAGS.test_srcdir, _TEST_DATA_DIR, "phonetic_results.jsonl"
    )

    # Test no pruning.
    FLAGS.pruning_method = confidence_pruning.Method.NONE
    semantics, phonetics, join = lib.load_and_prune_semantic_phonetic_jsonls()
    self.assertLen(semantics, _TOTAL_NUM_CONCEPTS)
    self.assertEqual(len(semantics), len(phonetics))
    self.assertEqual(len(join), len(semantics))

    # Test top-P (cumulative probability pruning). We set the maximum cumulative
    # probability (nucleus cut-off) threshold so that actual weeding of less
    # probable hypotheses does happen.
    FLAGS.pruning_method = confidence_pruning.Method.TOP_P
    FLAGS.semantic_top_p = 0.4
    FLAGS.phonetic_top_p = 0.8
    semantics, phonetics, join = lib.load_and_prune_semantic_phonetic_jsonls()
    self.assertLen(semantics, 74)
    self.assertLen(phonetics, 189)
    self.assertLess(len(join), len(semantics))
    self.assertLess(len(join), len(phonetics))


if __name__ == "__main__":
  absltest.main()
