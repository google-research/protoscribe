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

"""Apache Beam utilities for generating the dataset."""

import json
from typing import Callable, Iterable

from absl import flags
import apache_beam as beam
import numpy as np
from protoscribe.corpus.builder import document_builder
from protoscribe.corpus.builder import speech_pipeline as speech
from protoscribe.sketches.utils import stroke_stats as stroke_stats_lib
from protoscribe.texts import generate_lib
import tensorflow as tf

FLAGS = flags.FLAGS

StrokeStats = stroke_stats_lib.StrokeStats

_INCLUDE_UNSEEN_CONCEPTS_IN_TRAIN = flags.DEFINE_bool(
    "include_unseen_concepts_in_train", False,
    "Whenever unseen concepts are suplied to the dataset builder using "
    "`--unseen_concepts`, the default behaviour is only to sample from the "
    "unseen concepts for the test split. Enabling this flag will also include "
    "unseen concepts in training data. In this case the target glyphs, strokes "
    "and other types will be represented by the special dummy glyph "
    "(`DUMMY_GLYPH_NAME`) and the corresponding graphics."
)

# Always construct three splits.
_SPLIT_NAMES = ["train", "validation", "test"]


def sample_split_assignments(
    params: document_builder.Params,
    split_ratios: list[float]
) -> Iterable[tuple[int, int, str]]:
  """Yields document IDs and concepts for three different splits.

  Args:
    params: Document builder parameters.
    split_ratios: list[float]

  Yields:
    Tuples consisting of unique document ID, its split assignment and the
    corresponding concept name.

  Raises:
    ValueError if split ratios are invalid.
  """
  if np.sum(split_ratios) != 1.:
    raise ValueError("Split ratios should sum to 1.")

  num_texts = generate_lib.FLAGS.num_texts
  train_r, validation_r, _ = split_ratios
  validation_offset = int(num_texts * train_r)
  test_offset = int(num_texts * (train_r + validation_r))
  if test_offset == validation_offset and test_offset == num_texts - 1:
    # The test and validation partitions are likely too small. Make sure the
    # test partition has at least one document.
    validation_offset -= 2
    test_offset -= 1

  rng = np.random.default_rng()
  for doc_id in range(num_texts):
    if doc_id < validation_offset:
      split_name = "train"
      concepts = params.concepts
      if _INCLUDE_UNSEEN_CONCEPTS_IN_TRAIN.value:
        concepts = params.all_concepts
    elif doc_id >= validation_offset and doc_id < test_offset:
      concepts = params.concepts
      split_name = "validation"
    else:
      concepts = params.unseen_concepts
      split_name = "test"

    yield doc_id + 1, _SPLIT_NAMES.index(split_name), rng.choice(concepts)


class StrokeStatsCombinerFn(beam.CombineFn):
  """Combiner for accumulating stroke statistics."""

  def create_accumulator(self, *args, **kwargs) -> StrokeStats:
    """Returns a fresh, empty accumulator for the combine operation."""
    return stroke_stats_lib.StrokeStats()

  def add_input(
      self,
      mutable_accumulator: StrokeStats,
      element: StrokeStats,
      *args,
      **kwargs
  ) -> StrokeStats:
    """Returns result of folding element into accumulator."""
    mutable_accumulator.accumulate(element)
    return mutable_accumulator

  def merge_accumulators(
      self, accumulators: list[StrokeStats], *args, **kwargs
  ) -> StrokeStats:
    """Merging several accumulators to a single accumulator value."""
    accumulators = iter(accumulators)
    total = next(accumulators)
    for accumulator in accumulators:
      total.accumulate(accumulator)
    return total

  def extract_output(
      self, accumulator: StrokeStats
  ) -> stroke_stats_lib.FinalStrokeStats:
    return accumulator.finalize()


class BuildTfExampleFn(beam.DoFn):
  """Beam function for emitting a single document given the parameters."""

  def __init__(self, params: document_builder.Params) -> None:
    self._params = params
    self._speech_pipeline = speech.SpeechPipeline()

  def setup(self) -> None:
    """Called to prepare an instance for processing bundles of elements."""
    super().setup()
    self._speech_pipeline.setup()

  def process(
      self, element: tuple[int, StrokeStats, str, str], *args, **kwargs
  ) -> Iterable[tuple[int, tf.train.Example, StrokeStats | None]]:
    """Invoked to process single element from a PCollection."""
    doc_id, _, concept, split_name = element

    # Build the core example that excludes the speech features.
    (
        doc_id, core_example, sketch_stroke_stats
    ) = document_builder.build_tf_example(doc_id, concept, self._params)

    # Add speech features.
    try:
      doc_id, example = self._speech_pipeline.process_example(
          doc_id, core_example
      )
      # Yield a fully prepared document and its associated glyph stroke stats.
      beam.metrics.Metrics.counter(
          "BuildTfExampleFn", f"success-{split_name}"
      ).inc()
      yield doc_id, example, sketch_stroke_stats
    except ValueError as error:
      beam.metrics.Metrics.counter(
          "BuildTfExampleFn", f"speech-error-{split_name}"
      ).inc()
      raise ValueError(
          f"{doc_id}: Failed to generate speech features in {split_name} split"
      ) from error


@beam.ptransform_fn
def write_results_fn(
    pcoll: beam.PCollection, split_name: str, path_prefix: str
) -> beam.PCollection:
  """Writes the results to the sink."""
  return (
      pcoll
      | f"ValuesOnly-{split_name}" >> beam.Values()
      | f"WriteToTFRecord-{split_name}" >> beam.io.WriteToTFRecord(
          file_path_prefix=path_prefix,
          coder=beam.coders.ProtoCoder(tf.train.Example)
      )
  )


def init_beam_pipeline(
    phonetic_embeddings_path: str,
    concepts_vocab_file: str,
    output_train_file_prefix: str,
    output_validation_file_prefix: str,
    output_test_file_prefix: str,
    sketch_stroke_stats_file: str,
    split_ratios: list[float]
) -> Callable[[beam.Pipeline], None]:
  """Builds Apache Beam pipeline.

  Args:
    phonetic_embeddings_path: Path to phonetic embeddings.
    concepts_vocab_file: Concept vocabulary file in JSON format.
    output_train_file_prefix: Prefix path for the train split.
    output_validation_file_prefix: Prefix path for the validation split.
    output_test_file_prefix: Prefix path for the test split.
    sketch_stroke_stats_file: Path to stroke statistics file.
    split_ratios: A list of three float values in (0., 1.) specifying the
      percentage of train, validation and test splits.

  Returns:
    Functor for instantiating the pipeline.
  """

  params = document_builder.init_params(
      phonetic_embeddings_path=phonetic_embeddings_path,
      concepts_vocab_file=concepts_vocab_file
  )
  return init_beam_pipeline_from_params(
      params=params,
      output_train_file_prefix=output_train_file_prefix,
      output_validation_file_prefix=output_validation_file_prefix,
      output_test_file_prefix=output_test_file_prefix,
      sketch_stroke_stats_file=sketch_stroke_stats_file,
      split_ratios=split_ratios
  )


def init_beam_pipeline_from_params(
    params: document_builder.Params,
    output_train_file_prefix: str,
    output_validation_file_prefix: str,
    output_test_file_prefix: str,
    sketch_stroke_stats_file: str,
    split_ratios: list[float]
) -> Callable[[beam.Pipeline], None]:
  """Builds Apache Beam pipeline.

  Args:
    params: Initialized parameters for the document builder.
    output_train_file_prefix: Prefix path for the train split.
    output_validation_file_prefix: Prefix path for the validation split.
    output_test_file_prefix: Prefix path for the test split.
    sketch_stroke_stats_file: Path to stroke statistics file.
    split_ratios: A list of three float values in (0., 1.) specifying the
      percentage of train, validation and test splits.

  Returns:
    Functor for instantiating the pipeline.
  """

  def pipeline(root: beam.Pipeline) -> None:
    """Actual pipeline topology."""
    docs_and_splits_info = sample_split_assignments(params, split_ratios)
    docs_and_splits = (
        root
        | beam.Create(docs_and_splits_info)
        | "Partition" >> beam.Partition(
            lambda doc_concept_and_split_ids, num_partitions: (
                doc_concept_and_split_ids[1]  # Partition (split) ID.
            ),
            len(_SPLIT_NAMES)
        )
    )
    docs_and_stats = []
    doc_builder_fn = BuildTfExampleFn(params)
    for i in range(len(_SPLIT_NAMES)):
      split_name = _SPLIT_NAMES[i]
      docs_and_stats.append(
          docs_and_splits[i]
          | f"ReshardBeforeBuild-{split_name}" >> beam.transforms.Reshuffle()
          | f"InjectSplitName-{split_name}" >> beam.Map(
              lambda elem, add_elem=split_name: (
                  elem[0], elem[1], elem[2], add_elem
              )
          )
          | f"BuildDocsDoFn-{split_name}" >> beam.ParDo(doc_builder_fn)
          | f"ReshardAfterBuild-{split_name}" >> beam.transforms.Reshuffle()
      )

    # Write the dataset splits.
    outputs = [
        output_train_file_prefix,
        output_validation_file_prefix,
        output_test_file_prefix,
    ]
    for i in range(len(outputs)):
      split_name = _SPLIT_NAMES[i]
      _ = (
          docs_and_stats[i]
          | f"DocsOnly-{split_name}" >> beam.Map(
              lambda elem: (str(elem[0]).zfill(6), elem[1]))
          | f"ReshardBeforeWrite-{split_name}" >> beam.transforms.Reshuffle()
          | f"WriteResults-{split_name}" >> write_results_fn(
              split_name=split_name, path_prefix=outputs[i]
          )
      )

    # Accumulate and save the sketch stroke statistics for the training set.
    _ = (
        docs_and_stats[0]  # Train.
        | "SketchStrokeStatsOnly" >> beam.Map(lambda elem: elem[2])
        | "CombineSketchStrokeStats" >> beam.CombineGlobally(
            StrokeStatsCombinerFn())
        | "FormatSketchStrokeStatsToJson" >> beam.Map(json.dumps)
        | "WriteSketchStrokeStats" >> beam.io.WriteToText(
            sketch_stroke_stats_file, shard_name_template=""
        )
    )

  return pipeline
