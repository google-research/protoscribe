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

"""Test for sketch annotation task."""

import os

from absl import flags
from absl.testing import absltest
import numpy as np
from protoscribe.sketches.inference import sketch_annotation_task as lib
from protoscribe.sketches.utils import stroke_tokenizer as tokenizer_lib
import seqio

FLAGS = flags.FLAGS

_JSONL_PATH = (
    "protoscribe/sketches/inference/testdata/"
    "infer_eval_sketch_only.jsonl"
)

_TASK_NAME_PREFIX = "test"
_DEFAULT_SPLIT_NAME = "test"
_NUM_EXAMPLES = 5
_MAX_INPUT_TOKENS = 250
_MAX_GLYPH_TOKENS = 20


class SketchAnnotationTaskTest(absltest.TestCase):

  def test_simple(self):
    """Tests that the output examples are generally sane."""

    full_task_name = lib.register_for_inference(
        task_name_prefix=_TASK_NAME_PREFIX,
        jsonl_file_path=os.path.join(FLAGS.test_srcdir, _JSONL_PATH),
        max_stroke_sequence_length=_MAX_INPUT_TOKENS,
    )
    self.assertEqual(full_task_name, "test_sketch_annotation")

    task = seqio.TaskRegistry.get(full_task_name)
    self.assertIsNotNone(task)
    ds = task.get_dataset(split=_DEFAULT_SPLIT_NAME).take(_NUM_EXAMPLES)
    ds = list(ds.as_numpy_iterator())
    self.assertLen(ds, _NUM_EXAMPLES)
    for example in ds:
      self.assertEqual(list(sorted(example.keys())), [
          "concept.name", "doc.id", "inputs", "number.name",
          "sketch.confidence", "sketch_tokens", "targets", "text.sampa",
          "text.words",
      ])

      # Check mandatory features.
      inputs = example["inputs"]
      self.assertEqual(inputs.shape, (_MAX_INPUT_TOKENS,))
      self.assertEqual(inputs.dtype, np.int32)
      self.assertEqual(inputs[0], tokenizer_lib.Token.START_OF_SKETCH)
      self.assertEqual(inputs[-1], tokenizer_lib.Token.PAD)
      inputs = inputs.tolist()
      end_of_numbers_pos = inputs.index(tokenizer_lib.Token.END_OF_NUMBERS)
      self.assertGreater(end_of_numbers_pos, 0)
      end_of_sketch_pos = inputs.index(tokenizer_lib.Token.END_OF_SKETCH)
      self.assertGreater(end_of_sketch_pos, 0)
      self.assertGreater(end_of_sketch_pos, end_of_numbers_pos)
      pad_first_pos = inputs.index(tokenizer_lib.Token.PAD)
      self.assertGreater(pad_first_pos, end_of_sketch_pos)
      targets = example["targets"]
      self.assertEqual(targets.shape, (_MAX_GLYPH_TOKENS,))

      # Check pass-through features.
      self.assertIn("doc.id", example)
      self.assertGreater(example["doc.id"], 0.)
      self.assertIn("number.name", example)
      number_name = example["number.name"].decode("utf-8")
      self.assertNotEmpty(number_name)
      self.assertIn("concept.name", example)
      concept_name = example["concept.name"].decode("utf-8")
      self.assertNotEmpty(concept_name)
      self.assertIn("text.sampa", example)
      text_sampa = example["text.sampa"].decode("utf-8")
      self.assertNotEmpty(text_sampa)
      self.assertIn("text.words", example)
      text_words = example["text.words"].decode("utf-8")
      self.assertNotEmpty(text_words)
      self.assertIn("sketch.confidence", example)
      self.assertEqual(example["sketch.confidence"], 0.)  # Beam-less test data.


if __name__ == "__main__":
  absltest.main()
