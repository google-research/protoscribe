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

"""Input pipeline for Protoscribe dataset.

This is implemented on top of ``standard`` Protoscribe SeqIO task.
"""

import functools
import logging
from typing import Mapping

import ml_collections
from protoscribe.corpus.reader import tasks
from protoscribe.corpus.reader import utils
import seqio
import tensorflow as tf


class MinimalFeatureConverter(seqio.feature_converters.FeatureConverter):
  """This feature converter processes input and output features only."""

  def __init__(
      self,
      config: ml_collections.ConfigDict,
      use_passthrough_features: bool = False
  ):
    self._config = config
    self._use_passthrough_features = use_passthrough_features

  def _filter_and_pad(
      self, example: dict[str, tf.Tensor]
  ) -> dict[str, tf.Tensor]:
    """Retains all the necessary features from the ones provided by SeqIO task.

    Args:
      example: A dictionary containing all the features pulled by the task.

    Returns:
      A dictionary of filtered and padded features.

    Raises:
      ValueError if none of the required features are provided by configuration.
    """
    features = self._config.get("features")
    if not features:
      raise ValueError("Config should have `features` subsection defined!")
    mandatory_features = features.get("mandatory")
    if not mandatory_features:
      raise ValueError("No mandatory features defined in confuguration!")

    min_example = {}
    for name in mandatory_features:
      original_name, max_length = self._config.features.mandatory[name]
      # Workaround ConfigDict not accepting dots in field names.
      name = name.replace("/", ".")
      min_example[name] = example[original_name]
      if max_length > 0:
        min_example[name] = utils.pad_or_trim_sequence(
            min_example[name], max_length
        )
      if (
          self._use_passthrough_features and
          "passthrough" in self._config.features
      ):
        for name in self._config.features.passthrough:
          min_example[name] = example[name]
    return min_example

  def __call__(
      self, ds: tf.data.Dataset, task_feature_lengths: Mapping[str, int]
  ) -> tf.data.Dataset:
    """Applies feature converter to the dataset.

    Args:
      ds: Dataset reader.
      task_feature_lengths: Maximum sequence length per feature.

    Returns:
      The dataset reader with filtering and padding applied.
    """
    return  ds.map(
        functools.partial(self._filter_and_pad),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

  def _convert_features(
      self,
      ds: tf.data.Dataset,
      task_feature_lengths: Mapping[str, int]
  ):
    """This method is required to be overridden but is unused."""
    pass

  def get_model_feature_lengths(self, task_feature_lengths: Mapping[str, int]):
    """This method is required to be overridden but is unused."""
    pass


def _protoscribe_dataset(
    config: ml_collections.ConfigDict,
    dataset_dir: str,
    split_name: str,
    batch_size: int,
    is_training: bool = False,
    use_passthrough_features: bool = False
) -> tf.data.Dataset:
  """Registers and returns a dataset reader for a split.

  This function is a wrapper over Protoscribe SeqIO task registerer.

  Args:
    config: Configuration dictionary.
    dataset_dir: Directory where the dataset shards reside.
    split_name: Name of the split, e.g., "train".
    batch_size: Size of the batch.
    is_training: Whether the dataset is training or evaluation/test.
    use_passthrough_features: If enabled, pass through additional features
      useful for inference or debugging.

  Returns:
    Dataset reader.
  """
  task_name = tasks.register(
      task_name=split_name,
      dataset_dir=dataset_dir,
      max_stroke_sequence_length=config.max_stroke_sequence_length,
      max_glyph_sequence_length=config.max_glyph_sequence_length,
      stroke_normalization_type=config.stroke_normalization_type,
      stroke_random_scale_factor=config.stroke_random_scale_factor,
      stroke_token_vocab_filename=config.stroke_token_vocab_filename,
      noisify_embeddings=is_training,
      noisify_neftune_alphas=config.noisify_neftune_alphas,
      is_training=is_training
  )
  logging.info("Registered Protoscribe task: %s", task_name)
  ds = seqio.dataset_providers.get_dataset(
      mixture_or_task_name=task_name,
      task_feature_lengths={},
      feature_converter=MinimalFeatureConverter(
          config,
          use_passthrough_features=use_passthrough_features
      ),
      dataset_split=split_name,
      shuffle=is_training,
      num_epochs=None if is_training else 1,
      batch_size=batch_size,
      trim_output_features=False,
  )
  return ds


def get_train_and_eval_datasets(
    config: ml_collections.ConfigDict,
    dataset_dir: str,
    num_devices: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
  """Sets up and returns training and eval dataset readers.

  Args:
    config: Configuration dictionary.
    dataset_dir: Directory where the dataset splits reside.
    num_devices: Number of devices used for parallel reading.

  Returns:
    A tuple of training and evaluation dataset readers.

  Raises:
    ValueError if the configuration is malformed.
  """
  if "protoscribe" not in config:
    raise ValueError(
        "Expected Protoscribe sub-config in the main configuration!"
    )
  batch_size = config.per_device_batch_size * num_devices
  protoscribe_config = config.protoscribe
  train_ds = _protoscribe_dataset(
      protoscribe_config,
      dataset_dir,
      split_name="train",
      batch_size=batch_size,
      is_training=True
  )
  eval_ds = _protoscribe_dataset(
      protoscribe_config,
      dataset_dir,
      split_name="validation",
      batch_size=batch_size,
  )
  return train_ds, eval_ds


def get_test_dataset(
    config: ml_collections.ConfigDict,
    dataset_dir: str,
    num_devices: int,
) -> tf.data.Dataset:
  """Sets up and returns dataset reader for testing/inference.

  Args:
    config: Configuration dictionary.
    dataset_dir: Directory where the dataset splits reside.
    num_devices: Number of devices used for parallel reading.

  Returns:
    Dataset reader.

  Raises:
    ValueError if the configuration is malformed.
  """
  if "protoscribe" not in config:
    raise ValueError(
        "Expected Protoscribe sub-config in the main configuration!"
    )
  batch_size = config.per_device_batch_size * num_devices
  protoscribe_config = config.protoscribe
  return _protoscribe_dataset(
      protoscribe_config,
      dataset_dir,
      split_name="test",
      batch_size=batch_size,
      use_passthrough_features=True
  )
