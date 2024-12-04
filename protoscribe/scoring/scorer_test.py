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

"""Test for scorer_lib."""

import os

from absl import flags
from absl.testing import absltest
from protoscribe.language.phonology import phoible_segments
from protoscribe.scoring import scorer as lib

FLAGS = flags.FLAGS

_PREDICTION_A = {
    "number.name": "11",
    "concept.name": "swim_NOUN",
    "text.words": "u # t e # n e",
    "glyph.names": [["X", "I", "water", "lake"]],
    "glyph.prons": [[[], [], ["e", "s"], ["n", "e"]]],
}

_PREDICTION_B = {
    "number.name": "16",
    "concept.name": "baked_ADJ",
    "text.words": "u # o # a t",
    "glyph.names": [["X", "V", "I", "meat", "steak"]],
    "glyph.prons": [[[], [], [], ["b", "i"], ["a", "k"]]],
}

_TESTDATA_DIR = "protoscribe/scoring/testdata"

_CONCEPTS_FILE = f"{_TESTDATA_DIR}/administrative_categories.txt"
_PHONETIC_EMBEDDINGS_FILE = f"{_TESTDATA_DIR}/phonetic_embeddings.tsv"
_MAIN_LEXICON_FILE = f"{_TESTDATA_DIR}/lexicon.tsv"
_NUMBER_LEXICON_FILE = f"{_TESTDATA_DIR}/number_lexicon.tsv"


class ScorerTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()

    FLAGS.concepts = [os.path.join(FLAGS.test_srcdir, _CONCEPTS_FILE)]
    FLAGS.k_nearest_concepts = 5
    FLAGS.k_nearest_prons = 5

    self._phoible_phonemes_path = os.path.join(
        FLAGS.test_srcdir, phoible_segments.PHOIBLE)
    self._phoible_features_path = os.path.join(
        FLAGS.test_srcdir, phoible_segments.PHOIBLE_FEATURES)
    self._phonetic_embeddings_path = os.path.join(
        FLAGS.test_srcdir, _PHONETIC_EMBEDDINGS_FILE)
    self._main_lexicon_path = os.path.join(
        FLAGS.test_srcdir, _MAIN_LEXICON_FILE)
    self._number_lexicon_path = os.path.join(
        FLAGS.test_srcdir, _NUMBER_LEXICON_FILE)

    self.scorer = lib.ComparativeScorer(
        phoible_phonemes_path=self._phoible_phonemes_path,
        phoible_features_path=self._phoible_features_path,
        phonetic_embeddings_path=self._phonetic_embeddings_path,
        main_lexicon_path=self._main_lexicon_path,
        number_lexicon_path=self._number_lexicon_path
    )

  def test_scorer_from_dicts(self) -> None:
    single_pred_scorer = lib.Scorer()
    score1 = single_pred_scorer.eval_single_prediction(
        self.scorer.prepare_for_scoring(_PREDICTION_A), cumulate_scores=True
    )
    golden1 = lib.Score(
        num_match=True,
        multi_glyph=True,
        sem_top_k_score=3,
        sem_top_k_accuracy=1,
        phon_top_k_score=0,
        phon_top_k_accuracy=1,
        sem_phon_top_k_accuracy=1
    )
    self.assertEqual(score1, golden1)
    score2 = single_pred_scorer.eval_single_prediction(
        self.scorer.prepare_for_scoring(_PREDICTION_B), cumulate_scores=True
    )
    golden2 = lib.Score(
        num_match=True,
        multi_glyph=True,
        sem_top_k_score=2,
        sem_top_k_accuracy=1,
        phon_top_k_score=1,
        phon_top_k_accuracy=1,
        sem_phon_top_k_accuracy=1
    )
    self.assertEqual(score2, golden2)
    self.assertEqual(single_pred_scorer.find_score(
        "total_num_match↑"), (2, 2, 1.0))
    self.assertEqual(single_pred_scorer.find_score(
        "total_multi_glyph↑"), (2, 2, 1.0))
    self.assertEqual(single_pred_scorer.find_score(
        "sem_top_k_score↓"), (2, 5, 2.5))
    self.assertEqual(single_pred_scorer.find_score(
        "phon_top_k_score↓"), (2, 1, 0.5))

  def test_comparative_scorer_from_dirs(self) -> None:
    dir1 = os.path.join(FLAGS.test_srcdir, f"{_TESTDATA_DIR}/dir1")
    dir2 = os.path.join(FLAGS.test_srcdir, f"{_TESTDATA_DIR}/dir2")

    self.scorer.score_dirs(
        [dir1, dir2], pretty_names=["phonemb", "no_phonemb"]
    )
    self.assertEqual(
        self.scorer.scorers["phonemb"].find_score("total_num_match↑"),
        self.scorer.scorers["no_phonemb"].find_score("total_num_match↑"),
    )
    self.assertEqual(
        self.scorer.scorers["phonemb"].find_score("total_multi_glyph↑"),
        self.scorer.scorers["no_phonemb"].find_score("total_multi_glyph↑"),
    )
    self.assertGreater(
        self.scorer.scorers["phonemb"].find_score("sem_top_k_score↓")[1],
        self.scorer.scorers["no_phonemb"].find_score("sem_top_k_score↓")[1],
    )
    self.assertLess(
        self.scorer.scorers["phonemb"].find_score("phon_top_k_score↓")[1],
        self.scorer.scorers["no_phonemb"].find_score("phon_top_k_score↓")[1],
    )


if __name__ == "__main__":
  absltest.main()
