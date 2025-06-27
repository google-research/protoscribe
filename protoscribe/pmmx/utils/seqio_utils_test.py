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

"""Tests for pmmx.seqio_utils."""

from protoscribe.pmmx.utils import seqio_utils
import seqio
from t5x import utils
import tensorflow as tf


class SeqioUtilsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.dataset_config = utils.DatasetConfig(
        mixture_or_task_name="awesome_task",
        task_feature_lengths={"inputs": 1, "targets": 1},
        split="train",
        batch_size=1,
        shuffle=False,
        seed=1,
    )
    seqio.TaskRegistry.reset()

  def test_get_dataset_defines_task_if_not_defined(self):
    def define_task_fn(task_name: str) -> None:
      del task_name
      register_dummy_task("awesome_task")

    self.assertNotIn("awesome_task", seqio.TaskRegistry.names())
    seqio_utils.get_dataset(
        self.dataset_config,
        shard_id=0,
        num_shards=1,
        feature_converter_cls=seqio.EncDecFeatureConverter,
        define_task_fn=define_task_fn,
    )
    self.assertIn("awesome_task", seqio.TaskRegistry.names())

  def test_get_dataset_does_not_redefine_task(self):
    def define_task_fn(task_name: str) -> None:
      del task_name
      register_dummy_task("awesome_task")

    register_dummy_task("awesome_task")
    seqio_utils.get_dataset(
        self.dataset_config,
        shard_id=0,
        num_shards=1,
        feature_converter_cls=seqio.EncDecFeatureConverter,
        define_task_fn=define_task_fn,
    )
    self.assertIn("awesome_task", seqio.TaskRegistry.names())


def register_dummy_task(task_name: str) -> None:
  """Register a dummy task."""
  x = [
      {"inputs": [7, 8, 5, 6, 9, 4, 3], "targets": [3, 9]},
      {"inputs": [8, 4], "targets": [4]},
      {"inputs": [5, 6, 7], "targets": [6, 5]},
  ]

  def dataset_fn(split, shuffle_files, seed=0):
    del split, shuffle_files, seed
    ds = seqio.test_utils.create_default_dataset(
        x, feature_names=("inputs", "targets")
    )
    return ds

  seqio.TaskRegistry.add(
      task_name,
      source=seqio.FunctionDataSource(
          dataset_fn=dataset_fn, splits=["train", "validation"]
      ),
      preprocessors=[
          seqio.preprocessors.append_eos_after_trim,
      ],
      output_features={
          feat: seqio.Feature(seqio.test_utils.sentencepiece_vocab())
          for feat in ("inputs", "targets")
      },
      metric_fns=[],
  )


if __name__ == "__main__":
  tf.test.main()
