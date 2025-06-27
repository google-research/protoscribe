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

"""Unit test for registering and accessing SeqIO tasks."""

import logging
import os

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
from protoscribe.corpus.reader import tasks as lib
import seqio

FLAGS = flags.FLAGS

_TEST_DATA_DIR = (
    "protoscribe/corpus/reader/testdata"
)


class TasksTest(parameterized.TestCase):

  def _dataset_dir(self, dataset_format: str) -> str:
    """Returns fully qualified dataset directory path."""
    dataset_dir = os.path.join(FLAGS.test_srcdir, _TEST_DATA_DIR)
    return dataset_dir

  def _grab_feature(
      self, task_name: str, split: str
  ) -> dict[str, np.ndarray]:
    """Selects the first feature."""
    task = seqio.TaskRegistry.get(task_name)
    self.assertIsNotNone(task)
    ds = task.get_dataset(split=split).take(1)
    ds = list(ds.as_numpy_iterator())
    self.assertLen(ds, 1)
    return ds[0]

  def _test_generation_task_features(
      self,
      task_name: str,
      split: str,
      continuous_strokes: bool = True,
  ) -> None:
    """Basic sanity test for one document."""
    features = self._grab_feature(task_name, split)
    self.assertIn("concept.id", features)
    self.assertIn("concept.unseen", features)
    self.assertIn("doc.id", features)
    self.assertIn("text.concept_embedding", features)
    self.assertIn("text.concept_embedding.original", features)
    self.assertIn("text.phonetic_embedding", features)
    self.assertIn("text.phonetic_embedding.original", features)
    self.assertIn("text.glyph.tokens", features)
    self.assertIn("text.glyph.types", features)
    self.assertIn("text.sampa", features)
    self.assertIn("text.words", features)
    self.assertIn("speech.log_mel_spectrum", features)
    if continuous_strokes:
      self.assertIn("strokes", features)
    else:
      self.assertIn("sketch_tokens", features)

    self.assertIn("inputs", features)
    self.assertIn("targets", features)

  @parameterized.named_parameters(
      dict(testcase_name="tfrecord_format", dataset_format="tfr"),
  )
  @flagsaver.flagsaver
  def test_bnc_synthesize_continuous_strokes(self, dataset_format: str) -> None:
    FLAGS.dataset_format = dataset_format
    task_name = lib.register(
        task_name=f"bnc_synthesize_strokes_{dataset_format}",
        dataset_dir=self._dataset_dir(dataset_format),
        concept_embedding_type="bnc"
    )
    logging.info("Testing task `%s` ...", task_name)
    for split in lib.CANONICAL_SPLITS:
      self._test_generation_task_features(
          task_name, split, continuous_strokes=True
      )

  @parameterized.named_parameters(
      dict(testcase_name="tfrecord_format", dataset_format="tfr"),
  )
  @flagsaver.flagsaver
  def test_bnc_synthesize_sketch_tokens(self, dataset_format: str) -> None:
    FLAGS.dataset_format = dataset_format
    task_name = lib.register(
        task_name=f"bnc_synthesize_sketch_tokens_{dataset_format}",
        dataset_dir=self._dataset_dir(dataset_format),
        concept_embedding_type="bnc",
        stroke_token_vocab_filename="vocab2048_normalized_sketchrnn.npy",
    )
    logging.info("Testing task `%s` ...", task_name)
    for split in lib.CANONICAL_SPLITS:
      self._test_generation_task_features(
          task_name, split, continuous_strokes=False
      )

  @parameterized.named_parameters(
      dict(testcase_name="tfrecord_format", dataset_format="tfr"),
  )
  @flagsaver.flagsaver
  def test_bnc_synthesize_glyphs_only(self, dataset_format: str) -> None:
    FLAGS.dataset_format = dataset_format
    task_name = lib.register(
        task_name=f"bnc_synthesize_glyphs_only_{dataset_format}",
        dataset_dir=self._dataset_dir(dataset_format),
        concept_embedding_type="bnc",
        glyph_only_targets=True,
    )
    logging.info("Testing task `%s` ...", task_name)
    for split in lib.CANONICAL_SPLITS:
      self._test_generation_task_features(task_name, split)

  @parameterized.named_parameters(
      dict(testcase_name="tfrecord_format", dataset_format="tfr"),
  )
  @flagsaver.flagsaver
  def test_bnc_recognize_glyphs(self, dataset_format: str) -> None:
    FLAGS.dataset_format = dataset_format
    task_name = lib.register(
        task_name=f"bnc_recognize_glyphs_{dataset_format}",
        dataset_dir=self._dataset_dir(dataset_format),
        concept_embedding_type="bnc",
        glyph_recognition=True,
        stroke_token_vocab_filename="vocab2048_normalized_sketchrnn.npy"
    )
    logging.info("Testing task `%s` ...", task_name)
    for split in lib.CANONICAL_SPLITS:
      features = self._grab_feature(task_name, split)
      self.assertIn("sketch_tokens", features)
      self.assertIn("targets", features)


if __name__ == "__main__":
  absltest.main()
