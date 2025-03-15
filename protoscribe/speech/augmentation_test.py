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

"""Tests for speech augmentation utilities."""

import numpy as np
from protoscribe.speech import augmentation as lib
import tensorflow as tf

_NUM_TIME_STEPS = 120
_NUM_FREQ_BINS = 128
_NUM_INSTANCES = 10


class AugmentationTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    lib.tf_spec_augment_init()

  def test_with_default_config(self):
    input_shape = [_NUM_TIME_STEPS, _NUM_FREQ_BINS]
    inputs = tf.ones(input_shape, dtype=tf.float32)
    outputs = lib.tf_spec_augment(inputs)
    np.testing.assert_array_equal(tf.shape(outputs), input_shape)
    for _ in range(_NUM_INSTANCES):
      other_outputs = lib.tf_spec_augment(inputs)
      np.testing.assert_array_equal(tf.shape(other_outputs), input_shape)
      self.assertFalse(np.all(np.equal(outputs, other_outputs)))


if __name__ == "__main__":
  tf.test.main()
