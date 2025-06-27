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

"""A task for running the glyph recognizer over the predicted sketch.

This provides an inference-time only interface from JSON files produced by
running the predictions for sketch generators.
"""

import functools
import json
import logging
from typing import Any, Callable

import gin
import numpy as np
from protoscribe.glyphs import glyph_vocab as glyph_lib
from protoscribe.sketches.inference import json_utils
from protoscribe.sketches.utils import stroke_tokenizer as tokenizer_lib
import seqio
import tensorflow as tf


def _convert_json_example(
    json_example: bytes, max_stroke_sequence_length: int
) -> tuple[list[int], int, str, str, str, str, float]:
  """Converts an example from json format."""

  # Mandatory input features: sketch tokens.
  json_dict = json.loads(json_example.decode("utf-8"))
  if "prediction" not in json_dict:
    raise ValueError("Expecting `prediction` key to be present in json!")
  prediction = json_dict["prediction"]
  if isinstance(prediction[0], list):
    # When running the annotation, we only look at the top hypothesis.
    prediction = prediction[-1]

  # Pad sketch tokens to maximum length, if needed.
  pad_amount = max_stroke_sequence_length - len(prediction)
  if pad_amount > 0:
    prediction = np.pad(prediction, [[0, pad_amount]])

  # Pass-through features. Add more as required. The `doc.id` (integer) tensor
  # gets dumped as float for some reason. Cast it to int.
  if "inputs" not in json_dict:
    raise ValueError("Expected additional features in `inputs` key.")
  doc_id = int(json_dict["inputs"]["doc.id"])
  number_name = json_dict["inputs"]["number.name"]
  concept_name = json_dict["inputs"]["concept.name"]

  # We only return pronunciation and words if these are present.
  pron_sampa = ""
  if "text.sampa" in json_dict["inputs"]:
    pron_sampa = json_dict["inputs"]["text.sampa"]
  words = ""
  if "text.words" in json_dict["inputs"]:
    words = json_dict["inputs"]["text.words"]

  # Since we are only looking at the top generation hypothesis, there is
  # probably no need to propagate full generation scores, but we still want to
  # compute the generation confidence and pass it on.
  confidence = json_utils.get_confidence(json_dict)
  return (
      [prediction], doc_id, number_name, concept_name, pron_sampa, words,
      confidence
  )


def _make_line_parser(
    parse_fn: Callable[[bytes], Any],
    max_stroke_sequence_length: int,
    max_glyph_sequence_length: int,
    stateful: bool = False
) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
  """Wraps a `fn` that takes bytes (single JSON) and outputs inputs+targets."""

  def tf_parse(line: tf.Tensor) -> dict[str, tf.Tensor]:
    (
        inputs, doc_id, number_name, concept_name,
        pron_sampa, words, confidence
    ) = tf.numpy_function(
        parse_fn,
        inp=[
            line,
            tf.constant(max_stroke_sequence_length)  # For padding.
        ],
        Tout=[
            tf.int64,    # Inputs: sketch tokens.
            tf.int64,    # Document ID.
            tf.string,   # Number name.
            tf.string,   # Concept name.
            tf.string,   # SAMPA pronunciation.
            tf.string,   # Words.
            tf.float64,  # Generation confidence.
        ],
        stateful=stateful
    )
    # The reshape is necessary as otherwise the tensor has unknown rank.
    inputs.set_shape([1, max_stroke_sequence_length])
    inputs = tf.cast(tf.squeeze(inputs), tf.int32)
    scalars = [doc_id, number_name, concept_name, pron_sampa, words, confidence]
    for scalar_tensor in scalars:
      scalar_tensor.set_shape([])

    # Glyph targets: Set those to dummy values.
    targets = tf.zeros([max_glyph_sequence_length], dtype=tf.int32)
    return {
        # Mandatory features.
        "inputs": inputs,
        "sketch_tokens": inputs,  # Same as above.
        "targets": targets,  # Dummy.
        # Pass-through features.
        "doc.id": doc_id,
        "number.name": number_name,
        "concept.name": concept_name,
        "text.sampa": pron_sampa,
        "text.words": words,
        "sketch.confidence": confidence,
    }

  return lambda ds: ds.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)


@gin.configurable
def register_for_inference(
    task_name_prefix: str = gin.REQUIRED,
    jsonl_file_path: str = gin.REQUIRED,
    max_stroke_sequence_length: int = 250,
    max_glyph_sequence_length: int = 20,
) -> str:
  """Registers a single inference task from jsonl files."""

  task_name = f"{task_name_prefix}_sketch_annotation"
  logging.info(
      "Registering task `%s` to read from `%s`", task_name, jsonl_file_path
  )

  preprocessors = [
      _make_line_parser(
          parse_fn=functools.partial(_convert_json_example),
          max_stroke_sequence_length=max_stroke_sequence_length,
          max_glyph_sequence_length=max_glyph_sequence_length
      ),
  ]
  output_features = {
      "inputs": seqio.Feature(
          vocabulary=seqio.PassThroughVocabulary(
              size=0, eos_id=tokenizer_lib.Token.END_OF_SKETCH
          ),
          add_eos=False,
          dtype=tf.int32,
          required=True,
          rank=1,
      ),
      "targets": seqio.Feature(
          vocabulary=seqio.PassThroughVocabulary(
              size=0, eos_id=glyph_lib.GLYPH_EOS
          ),
          add_eos=False,
          dtype=tf.int32,
          required=True,
          rank=1,
      ),
  }
  seqio.TaskRegistry.add(
      name=task_name,
      source=seqio.TextLineDataSource(
          split_to_filepattern={
              "test": jsonl_file_path,
          },
      ),
      preprocessors=preprocessors,
      output_features=output_features,
      metric_fns=[],
  )
  return task_name
