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

import os

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import ml_collections
from protoscribe.models.flax import input_pipeline

FLAGS = flags.FLAGS

_TEST_DATA_DIR = (
    "protoscribe/corpus/reader/testdata"
)
_NUM_LOCAL_DEVICES = 1
_MAX_STROKE_SEQUENCE_LENGTH = 50
_MAX_INPUT_EMBEDDING_LENGTH = 2
_INPUT_EMBEDDING_DIM = 300  # BNC.
_INPUTS_SHAPE = [
    _NUM_LOCAL_DEVICES,
    _MAX_INPUT_EMBEDDING_LENGTH,
    _INPUT_EMBEDDING_DIM
]
_TARGETS_SHAPE = [_NUM_LOCAL_DEVICES, _MAX_STROKE_SEQUENCE_LENGTH]


def _dataset_dir(dataset_format: str) -> str:
  """Returns fully qualified dataset directory path.

  Args:
    dataset_format: Type of the dataset.

  Returns:
    Fully qualified path to the dataset shards.
  """
  dataset_dir = os.path.join(absltest.get_default_test_srcdir(), _TEST_DATA_DIR)
  return dataset_dir


def _mock_config() -> ml_collections.ConfigDict:
  """Returns mock configuration.

  Returns:
    Configuration dictionary.
  """

  # Protoscribe section.
  config = ml_collections.ConfigDict()
  config.max_stroke_sequence_length = _MAX_STROKE_SEQUENCE_LENGTH
  config.max_glyph_sequence_length = 20
  config.stroke_normalization_type = None  # No need for stroke stats.
  config.stroke_random_scale_factor = 0.
  config.stroke_token_vocab_filename = "vocab2048_normalized_sketchrnn.npy"
  config.vocab_size = 2064  # Sketch token vocabulary size.
  config.noisify_neftune_alphas = {}

  config.features = ml_collections.ConfigDict()
  config.features.mandatory = {
      "inputs": (
          "text.concept_embedding", _MAX_INPUT_EMBEDDING_LENGTH,
      ),
      "targets": (
          "sketch_tokens", config.max_stroke_sequence_length
      ),
  }

  # Main section.
  main_config = ml_collections.ConfigDict()
  main_config.protoscribe = config
  main_config.per_device_batch_size = 1
  return main_config


class InputPipelineTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="tfrecord_format", dataset_format="tfr"),
  )
  @flagsaver.flagsaver
  def test_dataset_readers_train_and_eval(self, dataset_format: str):
    """Tests training and development splits."""
    FLAGS.dataset_format = dataset_format
    config = _mock_config()
    dataset_dir = _dataset_dir(dataset_format)
    train_ds, eval_ds = input_pipeline.get_train_and_eval_datasets(
        config=config, dataset_dir=dataset_dir, num_devices=_NUM_LOCAL_DEVICES
    )

    for batch in train_ds.take(3):
      self.assertIn("inputs", batch)
      self.assertEqual(batch["inputs"].shape, _INPUTS_SHAPE)
      self.assertIn("targets", batch)
      self.assertEqual(batch["targets"].shape, _TARGETS_SHAPE)

    for batch in eval_ds.take(3):
      self.assertIn("inputs", batch)
      self.assertEqual(batch["inputs"].shape, _INPUTS_SHAPE)
      self.assertIn("targets", batch)
      self.assertEqual(batch["targets"].shape, _TARGETS_SHAPE)

  @parameterized.named_parameters(
      dict(testcase_name="tfrecord_format", dataset_format="tfr"),
  )
  @flagsaver.flagsaver
  def test_dataset_reader_test(self, dataset_format: str):
    """Tests test split."""

    # Create mock configuration and add pass-through features.
    config = _mock_config()
    config.protoscribe.features.passthrough = ["doc.id"]

    FLAGS.dataset_format = dataset_format
    dataset_dir = _dataset_dir(dataset_format)
    ds = input_pipeline.get_test_dataset(
        config=config, dataset_dir=dataset_dir, num_devices=_NUM_LOCAL_DEVICES
    )
    for batch in ds.take(3):
      self.assertIn("inputs", batch)
      self.assertEqual(batch["inputs"].shape, _INPUTS_SHAPE)
      self.assertIn("doc.id", batch)
      self.assertLen(batch["doc.id"], 1)
      self.assertGreater(batch["doc.id"][0], 0)


if __name__ == "__main__":
  absltest.main()
