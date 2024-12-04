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

"""Feature converters for Multimodal-Encoder Sequence-Decoder tasks.
"""

import abc
import dataclasses
import functools
from typing import Any, Iterable, Mapping, Sequence

from absl import logging
import seqio
import tensorflow.compat.v2 as tf

FeatureSpecConfig = tuple[str, str, int]


@dataclasses.dataclass
class FeatureSpec:
  """Simple container class for a feature's name, dtype, an rank."""

  name: str
  dtype: tf.DType
  rank: int

  @classmethod
  def to_map(
      cls, feature_specs: Iterable[FeatureSpecConfig]) -> Mapping[str, Any]:
    feature_spec_map = {}
    for name, dtype_str, rank in feature_specs:
      if name in feature_spec_map:
        raise ValueError(f"duplicate feature_spec={name} in config")
      feature_spec_map[name] = FeatureSpec(name, getattr(tf, dtype_str), rank)
    return feature_spec_map

  @classmethod
  def from_config(cls, config: FeatureSpecConfig):
    name, dtype_str, rank = config
    return FeatureSpec(name, getattr(tf, dtype_str), rank)


def _convert_task_feature_lengths_to_input_features(
    task_feature_lengths: Mapping[str, int],
    feature_specs: Iterable[FeatureSpecConfig]) -> Sequence[FeatureSpec]:
  """Converts the user-specified task feature lengths to input features.

  Args:
    task_feature_lengths: mapping of feature name to length
    feature_specs: configuration of features from gin

  Returns:
    a sequence of FeatureSpecs
  """
  # Prepare input features features.
  task_feature_lengths = {
      k: v
      for (k, v) in task_feature_lengths.items()
      if not k.startswith("targets")
  }

  feature_spec_map = FeatureSpec.to_map(feature_specs)
  return [feature_spec_map[name] for name in task_feature_lengths]


class ExportableFeatureConverter(metaclass=abc.ABCMeta):
  """Interface to support model export and feature conversion without datasets."""

  @abc.abstractmethod
  def convert_example(
      self,
      task_feature_lengths: Mapping[str, int],
      features: Mapping[str, tf.Tensor],
  ) -> Mapping[str, tf.Tensor]:
    raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class MultimodalEncDecFeatureConverterFactory(object):
  """Layer of indirection for `MultimodalEncDecFeatureConverter`."""

  task_feature_lengths: Mapping[str, int]
  feature_specs: Iterable[FeatureSpecConfig]
  apply_length_check: bool = True

  def __call__(self, pack: bool = False, use_custom_packing_ops: bool = False):
    input_features = _convert_task_feature_lengths_to_input_features(
        self.task_feature_lengths, self.feature_specs)
    return MultimodalEncDecFeatureConverter(
        input_features=input_features,
        pack=pack,
        use_custom_packing_ops=use_custom_packing_ops,
        apply_length_check=self.apply_length_check)


class MultimodalEncDecFeatureConverter(
    seqio.FeatureConverter, ExportableFeatureConverter
):
  """Feature converter for an encoder-decoder architecture.

  Unlike the P5X FeatureConverters, this class has dynamic `input_features`,
  which are configurable via gin. This prevents us from having to plumb each
  new PMMX modality through many layers of code.

  To use packing, pass pack = True argument to the FeatureConverter's
  constructor. When packing is done, two additional fields are added for each of
  the features.

  Example for a packed dataset:

  The input dataset has two examples each with "inputs" and "targets".

  ds = [{"text_tokens": [7, 8, 5, 1], "targets": [3, 9, 1]},
        {"text_tokens": [8, 4, 9, 3, 1], "targets": [4, 1]}]

  task_feature_lengths = {"text_tokens": 10, "targets": 7}

  First, the `text_tokens` are packed together, padded to length 10 and assigned
  to `text_tokens` field. The `targets` are processed similarly.

  The "*_segment_id" fields are generated from the packing operation. For the
  explanation of these fields, see the module docstring.

  The `decoder_loss_weights` is a binary mask indicating where non-padding
  positions are, i.e., value of 1 indicates non-padding and 0 for padding. This
  class assumes that the loss is taken only on the decoder side.

  converted_ds = [{
                            "text_tokens": [7, 8, 5, 1, 8, 4, 9, 3, 1, 0],
                "text_tokens_segment_ids": [1, 1, 1, 1, 2, 2, 2, 2, 2, 0],
                  "text_tokens_positions": [0, 1, 2, 3, 0, 1, 2, 3, 4, 0],
                   "decoder_input_tokens": [0, 3, 9, 0, 4, 0, 0],
                  "decoder_target_tokens": [3, 9, 1, 4, 1, 0, 0],
      "decoder_target_tokens_segment_ids": [1, 1, 1, 2, 2, 0, 0],
        "decoder_target_tokens_positions": [0, 1, 2, 0, 1, 0, 0],
                   "decoder_loss_weights": [1, 1, 1, 1, 1, 0, 0],
  }]

  Note that two examples are packed together into one example.
  """
  input_features: Sequence[FeatureSpec] = ()

  def __init__(self,
               input_features: Sequence[FeatureSpec],
               pack: bool = False,
               use_custom_packing_ops: bool = False,
               apply_length_check: bool = True):
    self.input_features = input_features
    # Do not use custom ops for packing.
    use_custom_packing_ops = False
    super().__init__(pack=pack, use_custom_packing_ops=use_custom_packing_ops,
                     apply_length_check=apply_length_check)

  @property
  def TASK_FEATURES(self):
    feature_specs = {
        f.name: seqio.FeatureConverter.FeatureSpec(dtype=f.dtype, rank=f.rank)
        for f in self.input_features}
    # There should always be a `targets`.
    feature_specs.update({
        "targets": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32)
    })
    return feature_specs

  @property
  def MODEL_FEATURES(self):
    feature_specs = {
        f.name: seqio.FeatureConverter.FeatureSpec(dtype=f.dtype, rank=f.rank)
        for f in self.input_features}
    feature_specs.update({
        "decoder_target_tokens": seqio.FeatureConverter.FeatureSpec(
            dtype=tf.int32),
        "decoder_input_tokens": seqio.FeatureConverter.FeatureSpec(
            dtype=tf.int32),
        "decoder_loss_weights": seqio.FeatureConverter.FeatureSpec(
            dtype=tf.int32),
    })
    return feature_specs

  @property
  def PACKING_FEATURE_DTYPES(self):
    packing_feature_dtypes = {}
    for f in self.input_features:
      packing_feature_dtypes.update({
          f"{f.name}_segment_ids": tf.int32,
          f"{f.name}_positions": tf.int32,
      })
    packing_feature_dtypes.update({
        "decoder_target_tokens_segment_ids": tf.int32,
        "decoder_target_tokens_positions": tf.int32
    })
    return packing_feature_dtypes

  def convert_example(
      self,
      task_feature_lengths: Mapping[str, int],
      features: Mapping[str, tf.Tensor]
  ) -> Mapping[str, tf.Tensor]:
    # targets_segment_id is present only for a packed dataset.
    decoder_input_tokens = seqio.autoregressive_inputs(
        features["targets"],
        sequence_id=features.get("targets_segment_ids", None))

    d = {"decoder_target_tokens": features["targets"],
         "decoder_input_tokens": decoder_input_tokens,
         # Loss is computed for all but the padding positions.
         "decoder_loss_weights": seqio.non_padding_position(
             features["targets"])}

    for f in self.input_features:
      d[f.name] = features[f.name]

    if self.pack:
      for name in self.PACKING_FEATURE_DTYPES:
        if name.startswith("decoder_target_tokens"):
          input_name = name.replace("decoder_target_tokens", "targets")
        else:
          input_name = name
        d[name] = features[input_name]

    logging.info("Converted features: %s", d)

    return d

  def _convert_features(
      self, ds: tf.data.Dataset,
      task_feature_lengths: Mapping[str, int]) -> tf.data.Dataset:
    """Convert the dataset to be fed to the encoder-decoder model.

    The conversion process involves three steps

    1. Each feature in the `task_feature_lengths` is trimmed/padded and
       optionally packed depending on the value of self.pack.
    2. "inputs" fields are mapped to the encoder input and "targets" are mapped
       to decoder input (after being shifted) and target.

    All the keys in the `task_feature_lengths` should be present in the input
    dataset, which may contain some extra features that are not in the
    `task_feature_lengths`. They will not be included in the output dataset.
    One common scenario is the "inputs_pretokenized" and "targets_pretokenized"
    fields.

    Args:
      ds: an input tf.data.Dataset to be converted.
      task_feature_lengths: a mapping from feature to its length.

    Returns:
      ds: the converted dataset.
    """
    ds = self._pack_or_pad(ds, task_feature_lengths)
    convert_example = functools.partial(
        self.convert_example, task_feature_lengths)
    return ds.map(
        convert_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]) -> Mapping[str, int]:
    """Define the length relationship between input and output features."""

    model_feature_lengths = {}
    decoder_length = task_feature_lengths["targets"]
    for k in self.MODEL_FEATURES:
      if k.startswith("decoder"):
        model_feature_lengths[k] = decoder_length
      else:
        model_feature_lengths[k] = task_feature_lengths[k]

    if self.pack:
      for k in self.PACKING_FEATURE_DTYPES:
        if k.startswith("decoder"):
          model_feature_lengths[k] = decoder_length
        else:
          f_k = k.replace("_segment_ids", "").replace("_positions", "")
          model_feature_lengths[k] = task_feature_lengths[f_k]

    return model_feature_lengths
