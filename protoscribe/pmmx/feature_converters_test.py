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

"""Tests for feature_converters."""

from protoscribe.pmmx import feature_converters
import seqio
import tensorflow.compat.v2 as tf

tf.compat.v1.enable_eager_execution()

assert_dataset = seqio.test_utils.assert_dataset


def create_default_dataset(x):

  feature_names = list(x[0].keys())
  name2type = {
      "text_tokens": tf.int32,
      "more_tokens": tf.int32,
      "image_dense": tf.float32,
      "targets": tf.int32,
  }
  name2shape = {
      "text_tokens": [None],
      "more_tokens": [None],
      "image_dense": [None, 2],
      "targets": [None],
  }

  return seqio.test_utils.create_default_dataset(
      x,
      feature_names=feature_names,
      output_types={k: name2type[k] for k in feature_names},
      output_shapes={k: name2shape[k] for k in feature_names},  # pytype: disable=wrong-arg-types
  )


class MultimodalEncDecFeatureConverterTest(tf.test.TestCase):

  def test_encoder_decoder_unpacked(self):
    x = [{"text_tokens": [9, 4, 3, 8, 1],
          "image_dense": [[.5, .25], [.125, .375]],
          "targets": [3, 9, 4, 1]}]
    ds = create_default_dataset(x)
    task_feature_lengths = {"text_tokens": 7, "image_dense": 3, "targets": 5}
    feature_specs = [("text_tokens", "int32", 1), ("image_dense", "float32", 2)]
    input_features = (
        feature_converters._convert_task_feature_lengths_to_input_features(
            task_feature_lengths, feature_specs))
    converter = feature_converters.MultimodalEncDecFeatureConverter(
        input_features=input_features, pack=False)
    converted_ds = converter(ds, task_feature_lengths)

    expected = {
        "text_tokens": [9, 4, 3, 8, 1, 0, 0],
        "image_dense": [[.5, .25], [.125, .375], [0., 0.]],
        "decoder_target_tokens": [3, 9, 4, 1, 0],
        # mtf.transformer.autoregressive_inputs does not zero out the last eos
        # when the data is not packed. This test mimic the behavior.
        "decoder_input_tokens": [0, 3, 9, 4, 1],
        "decoder_loss_weights": [1, 1, 1, 1, 0],
    }
    assert_dataset(converted_ds, expected)

  def test_encoder_decoder_targets_max_length(self):
    x = [{
        "text_tokens": [9, 4, 3, 8, 1],
        "image_dense": [[.5, .25], [.125, .375]],
        "targets": [3, 9, 4, 5, 1]
    }]
    ds = create_default_dataset(x)
    task_feature_lengths = {"text_tokens": 5, "image_dense": 2, "targets": 5}
    feature_specs = [("text_tokens", "int32", 1), ("image_dense", "float32", 2)]
    input_features = (
        feature_converters._convert_task_feature_lengths_to_input_features(
            task_feature_lengths, feature_specs))
    converter = feature_converters.MultimodalEncDecFeatureConverter(
        input_features=input_features, pack=False)
    converted_ds = converter(ds, task_feature_lengths)

    expected = {
        "text_tokens": [9, 4, 3, 8, 1],
        "image_dense": [[.5, .25], [.125, .375]],
        "decoder_target_tokens": [3, 9, 4, 5, 1],
        "decoder_input_tokens": [0, 3, 9, 4, 5],
        "decoder_loss_weights": [1, 1, 1, 1, 1],
    }
    assert_dataset(converted_ds, expected)

  def test_encoder_decoder_extra_long_inputs(self):
    x = [{"text_tokens": [9, 4, 3, 8, 4, 5, 1],
          "image_dense": [[.5, .25], [.125, .375]],
          "targets": [3, 9, 4, 7, 8, 1]}]
    ds = create_default_dataset(x)
    task_feature_lengths = {"text_tokens": 5, "image_dense": 2, "targets": 8}
    feature_specs = [("text_tokens", "int32", 1), ("image_dense", "float32", 2)]
    input_features = (
        feature_converters._convert_task_feature_lengths_to_input_features(
            task_feature_lengths, feature_specs))
    expected_msg = (
        r".*Feature \\'text_tokens\\' has length not less than or equal to "
        r"the expected length of 5 during input_validation.*")
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError, expected_msg):
      converter = feature_converters.MultimodalEncDecFeatureConverter(
          input_features=input_features, pack=False)
      converted_ds = converter(ds, task_feature_lengths)
      list(converted_ds.as_numpy_iterator())

  def test_encoder_decoder_pretokenized_field(self):
    x = [{
        "text_tokens": [7, 8, 5, 1],
        "image_dense": [[.5, .25], [.125, .375]],
        "targets": [3, 9, 1],
        "targets_pretokenized": "abc"
    }, {
        "text_tokens": [8, 4, 9, 3, 1],
        "image_dense": [[-.5, -.25]],
        "targets": [4, 1],
        "targets_pretokenized": "def"
    }]
    types = {
        "text_tokens": tf.int32,
        "image_dense": tf.float32,
        "targets": tf.int32,
        "targets_pretokenized": tf.string
    }
    shapes = {"text_tokens": [None], "image_dense": [None, None],
              "targets": [None], "targets_pretokenized": []}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=types, output_shapes=shapes)

    task_feature_lengths = {
        "text_tokens": 10,
        "image_dense": 2,
        "targets": 7
    }
    feature_specs = [("text_tokens", "int32", 1), ("image_dense", "float32", 2)]
    input_features = (
        feature_converters._convert_task_feature_lengths_to_input_features(
            task_feature_lengths, feature_specs))
    converter = feature_converters.MultimodalEncDecFeatureConverter(
        input_features=input_features, pack=False)
    # Check whether convert_features raise error because targets_pretokenized is
    # present in the ds but not in the task_feature_lengths
    converter(ds, task_feature_lengths)


if __name__ == "__main__":
  tf.test.main()
