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

"""Simple unit tests for Apache Beam corpus building APIs."""

import collections
import logging
import os

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.options import pipeline_options
from protoscribe.corpus.builder import corpus_builder_beam as lib
from protoscribe.corpus.builder import test_utils
import tensorflow as tf

import glob
import os

FLAGS = flags.FLAGS


class CorpusBuilderBeamTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self._doc_builder_params = test_utils.init_document_builder_params()

  def _run_pipeline(self, pipeline: beam.Pipeline) -> None:
    """Executes the supplied pipeline.

    Args:
      pipeline: Initialized pipeline.
    """
    logging.info("Executing pipeline ...")
    options = pipeline_options.PipelineOptions(
        runner="DirectRunner",
        direct_running_mode="in_memory",
        direct_num_workers=1
    )
    p = beam.Pipeline(options=options)
    pipeline(p)
    p.run().wait_until_finish()
    logging.info("Pipeline finished.")

  @parameterized.parameters(
      (5, [0.98, 0.01, 0.01]),
      (10, [0.98, 0.01, 0.01]),
      (100, [0.98, 0.01, 0.01]),
      (5, [0.4, 0.3, 0.3]),
      (10, [0.4, 0.3, 0.3]),
      (100, [0.8, 0.1, 0.1]),
  )
  @flagsaver.flagsaver
  def test_split_assignments(self, num_texts: int, split_ratios: list[float]):
    """Basic test that split assignments are sane."""
    FLAGS.num_texts = num_texts
    docs = list(
        lib.sample_split_assignments(
            params=self._doc_builder_params, split_ratios=split_ratios
        )
    )
    split_to_num_docs = collections.defaultdict(int)
    split_to_concepts = collections.defaultdict(set)
    doc_ids = set()
    self.assertLen(docs, num_texts)
    for doc_id, split_id, concept in docs:
      doc_ids.add(doc_id)
      split_to_num_docs[split_id] += 1
      split_to_concepts[split_id].add(concept)
    self.assertLen(doc_ids, num_texts)

    # Check individual splits. Each split should have at least one document.
    self.assertLen(split_to_num_docs, 3)
    docs_in_splits = 0
    for split_id in split_to_num_docs:
      docs_in_splits += split_to_num_docs[split_id]
    self.assertEqual(docs_in_splits, num_texts)

    # Check that the concepts come from the administrative/non-administrative
    # assignments.
    concept_sets = [
        # Train.
        self._doc_builder_params.concepts,
        # Validation.
        self._doc_builder_params.concepts,
        # Test.
        self._doc_builder_params.unseen_concepts,
    ]
    for split_id in split_to_concepts:
      for concept in split_to_concepts[split_id]:
        self.assertIn(concept, concept_sets[split_id])

  @flagsaver.flagsaver
  def test_end_to_end_pipeline(self):
    """Simple test that runs the pipeline end-to-end."""
    FLAGS.num_texts = 5

    # Output paths for the artifacts.
    output_train_file_prefix = os.path.join(FLAGS.test_tmpdir, "train")
    output_validation_file_prefix = os.path.join(
        FLAGS.test_tmpdir, "validation"
    )
    output_test_file_prefix = os.path.join(FLAGS.test_tmpdir, "test")
    sketch_stroke_stats_file = os.path.join(
        FLAGS.test_tmpdir, "stroke_stats.json"
    )

    # Instantiate and run the pipeline.
    pipeline = lib.init_beam_pipeline_from_params(
        params=self._doc_builder_params,
        output_train_file_prefix=output_train_file_prefix,
        output_validation_file_prefix=output_validation_file_prefix,
        output_test_file_prefix=output_test_file_prefix,
        sketch_stroke_stats_file=sketch_stroke_stats_file,
        split_ratios=[0.5, 0.25, 0.25]
    )
    self._run_pipeline(pipeline)

    # Check the corpus splits.
    num_docs = 0
    for split_path_prefix in [
        output_train_file_prefix,
        output_validation_file_prefix,
        output_test_file_prefix,
    ]:
      logging.info("Checking `%s` ...", split_path_prefix)
      paths = [path for path in glob.glob(f"{split_path_prefix}*")]
      self.assertLen(paths, 1)  # One shard.
      dataset = tf.data.TFRecordDataset(paths)
      for record in dataset:
        example = tf.train.Example()
        example.ParseFromString(record.numpy())
        num_docs += 1
    self.assertEqual(num_docs, FLAGS.num_texts)


if __name__ == "__main__":
  absltest.main()
