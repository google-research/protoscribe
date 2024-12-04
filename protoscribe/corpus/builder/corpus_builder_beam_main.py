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

r"""Creates training data for ProtoScribe models using Apache Beam.

We don't run this tool manually but via the helper runner that generates the
synthetic language first and then invokes this tool with the necessary
parameters. Please see the helper runner in
  protoscribe/corpus/builder/build_dataset_main.py.
"""

import logging
from typing import Sequence

from absl import app
from absl import flags
import apache_beam as beam
from protoscribe.corpus.builder import corpus_builder_beam as beam_lib

_PHONETIC_EMBEDDINGS = flags.DEFINE_string(
    "phonetic_embeddings", None,
    "Path to output of build_phonetic_embeddings.",
    required=True
)

_CONCEPT_VOCAB_FILE = flags.DEFINE_string(
    "concept_vocab_file", None,
    "Concept vocabulary file in JSON format. A mapping from concept names to "
    "unique IDs and other associated information, such as a boolean flag "
    "indicating whether the concepts in `new` or not.",
    required=True,
)

_OUTPUT_TRAIN_FILE_PREFIX = flags.DEFINE_string(
    "output_train_file_prefix", None,
    "File prefix for the resulting records for the training split, e.g. "
    "`${OUTPUT_DIR}/protoscribe_train@*`.",
    required=True
)

_OUTPUT_VALIDATION_FILE_PREFIX = flags.DEFINE_string(
    "output_validation_file_prefix", None,
    "File prefix for the resulting records for the validation split, e.g. "
    "`${OUTPUT_DIR}/protoscribe_validation@*`.",
    required=True
)

_OUTPUT_TEST_FILE_PREFIX = flags.DEFINE_string(
    "output_test_file_prefix", None,
    "File prefix for the resulting records for the test split, e.g. "
    "`${OUTPUT_DIR}/protoscribe_test@*`.",
    required=True
)

_SKETCH_STROKE_STATS_FILE = flags.DEFINE_string(
    "sketch_stroke_stats_file", None,
    "File in JSON containing stroke statistics for normalizing the sketch "
    "data. The stats will only be computed if `--generate_strokes` is enabled.",
    required=True
)

_SPLIT_RATIOS = flags.DEFINE_list(
    "split_ratios", ["98", "1", "1"],
    "Ratios for the splits. Must be a list with three elements specifying "
    "the percentages for training, validation and test splits."
)

_MAX_WORKERS = flags.DEFINE_integer(
    "max_workers", 1,
    "Maximum number of worker threads."
)

FLAGS = flags.FLAGS


def _run_beam_pipeline() -> None:
  """Runs the Apache Beam pipeline."""

  if len(_SPLIT_RATIOS.value) != 3:
    raise ValueError("Three splits are expected in --split_ratios!")
  split_ratios = [int(val) / 100.0 for val in _SPLIT_RATIOS.value]
  pipeline = beam_lib.init_beam_pipeline(
      phonetic_embeddings_path=_PHONETIC_EMBEDDINGS.value,
      concepts_vocab_file=_CONCEPT_VOCAB_FILE.value,
      output_train_file_prefix=_OUTPUT_TRAIN_FILE_PREFIX.value,
      output_validation_file_prefix=_OUTPUT_VALIDATION_FILE_PREFIX.value,
      output_test_file_prefix=_OUTPUT_TEST_FILE_PREFIX.value,
      sketch_stroke_stats_file=_SKETCH_STROKE_STATS_FILE.value,
      split_ratios=split_ratios
  )
  options = beam.options.pipeline_options.PipelineOptions(
     runner="DirectRunner",
     direct_running_mode="multi_threading",
     direct_num_workers=_MAX_WORKERS.value
  )
  p = beam.Pipeline(options=options)
  pipeline(root=p)
  run_result = p.run()
  run_result.wait_until_finish()


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  _run_beam_pipeline()


if __name__ == "__main__":
  flags.mark_flag_as_required("concepts")
  flags.mark_flag_as_required("unseen_concepts")
  flags.mark_flag_as_required("glyph_vocab_file")
  app.run(main)
