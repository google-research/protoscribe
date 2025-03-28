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
from protoscribe.speech import augmentation_lingvo as lib
import tensorflow as tf


class AugmentationTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    lib.tf_spec_augment_init()

  def test_with_time_mask(self):
    tf.random.set_seed(127)
    batch_size = 5
    inputs = tf.ones([batch_size, 20, 2, 2], dtype=tf.float32)
    paddings = []
    for i in range(batch_size):
      paddings.append(
          tf.concat([tf.zeros([1, i + 12]),
                     tf.ones([1, 8 - i])], axis=1))
    paddings = tf.concat(paddings, axis=0)

    config = lib.AugmenterConfig(
        freq_mask_max_bins=0,
        time_mask_max_frames=5,
        time_mask_count=2,
        time_mask_max_ratio=1.0,
        random_seed=23456
    )
    expected_output = np.array([
        [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
        [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[0., 0.], [0., 0.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
        [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[0., 0.], [0., 0.]],
         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
        [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[0., 0.], [0., 0.]],
         [[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
        [[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]]
    ])
    h, _ = lib.tf_spec_augment_lingvo(config, inputs, paddings)
    self.assertAllClose(h, expected_output)

  def test_dynamic_size_time_mask(self):
    tf.random.set_seed(127)
    batch_size = 3
    inputs = tf.ones([batch_size, 20, 2, 2], dtype=tf.float32)
    paddings = []
    for i in range(batch_size):
      paddings.append(
          tf.concat([tf.zeros([1, 8 * i + 3]),
                     tf.ones([1, 17 - 8 * i])],
                    axis=1))
    paddings = tf.concat(paddings, axis=0)

    config = lib.AugmenterConfig(
        freq_mask_max_bins=0,
        time_mask_max_ratio=0.4,
        time_mask_count=1,
        use_dynamic_time_mask_max_frames=True,
        random_seed=12345
    )
    expected_output = np.array([
        [[[1., 1.], [1., 1.]], [[0., 0.], [0., 0.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
        [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
        [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]]
    ])
    h, _ = lib.tf_spec_augment_lingvo(config, inputs, paddings)
    self.assertAllClose(h, expected_output)

  def test_dynamic_multiplicity_time_mask(self):
    tf.random.set_seed(127)
    batch_size = 4
    inputs = tf.ones([batch_size, 22, 2, 2], dtype=tf.float32)
    paddings = []
    for i in range(batch_size):
      paddings.append(
          tf.concat([tf.zeros([1, 5 * i + 5]),
                     tf.ones([1, 16 - 5 * i])],
                    axis=1))
    paddings = tf.concat(paddings, axis=0)

    config = lib.AugmenterConfig(
        freq_mask_max_bins=0,
        time_mask_max_frames=5,
        time_mask_count=10,
        time_masks_per_frame=0.2,
        random_seed=67890
    )
    expected_output = np.array([
        [[[1., 1.], [1., 1.]], [[0., 0.], [0., 0.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
        [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[0., 0.], [0., 0.]],
         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
        [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[0., 0.], [0., 0.]],
         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]],
         [[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
        [[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[0., 0.], [0., 0.]],
         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]]
    ])
    h, _ = lib.tf_spec_augment_lingvo(config, inputs, paddings)
    self.assertAllClose(h, expected_output)

  def test_dynamic_size_and_multiplicity_time_mask(self):
    tf.random.set_seed(127)
    batch_size = 4
    inputs = tf.ones([batch_size, 22, 2, 2], dtype=tf.float32)
    paddings = []
    for i in range(batch_size):
      paddings.append(
          tf.concat([tf.zeros([1, 5 * i + 5]),
                     tf.ones([1, 16 - 5 * i])],
                    axis=1))
    paddings = tf.concat(paddings, axis=0)

    config = lib.AugmenterConfig(
        freq_mask_max_bins=0,
        time_mask_max_frames=5,
        time_mask_count=10,
        time_masks_per_frame=0.2,
        time_mask_max_ratio=0.4,
        use_dynamic_time_mask_max_frames=True,
        random_seed=67890,
    )
    expected_output = np.array([
        [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
        [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
        [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]],
         [[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
        [[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[1., 1.], [1., 1.]], [[0., 0.], [0., 0.]],
         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
         [[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]]
    ])
    h, _ = lib.tf_spec_augment_lingvo(config, inputs, paddings)
    self.assertAllClose(h, expected_output)

  def test_frequency_mask(self):
    tf.random.set_seed(1234)
    inputs = tf.ones([3, 5, 10, 1], dtype=tf.float32)
    paddings = tf.zeros([3, 5])

    config = lib.AugmenterConfig(
        freq_mask_max_bins=6,
        freq_mask_count=2,
        time_mask_max_frames=0,
        random_seed=34567
    )
    expected_output = np.array([
        [[[1.], [1.], [1.], [0.], [0.], [0.], [0.], [0.], [0.], [1.]],
         [[1.], [1.], [1.], [0.], [0.], [0.], [0.], [0.], [0.], [1.]],
         [[1.], [1.], [1.], [0.], [0.], [0.], [0.], [0.], [0.], [1.]],
         [[1.], [1.], [1.], [0.], [0.], [0.], [0.], [0.], [0.], [1.]],
         [[1.], [1.], [1.], [0.], [0.], [0.], [0.], [0.], [0.], [1.]]],
        [[[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [1.], [1.]],
         [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [1.], [1.]],
         [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [1.], [1.]],
         [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [1.], [1.]],
         [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [1.], [1.]]],
        [[[1.], [1.], [0.], [0.], [1.], [1.], [0.], [1.], [1.], [1.]],
         [[1.], [1.], [0.], [0.], [1.], [1.], [0.], [1.], [1.], [1.]],
         [[1.], [1.], [0.], [0.], [1.], [1.], [0.], [1.], [1.], [1.]],
         [[1.], [1.], [0.], [0.], [1.], [1.], [0.], [1.], [1.], [1.]],
         [[1.], [1.], [0.], [0.], [1.], [1.], [0.], [1.], [1.], [1.]]]
    ])
    h, _ = lib.tf_spec_augment_lingvo(config, inputs, paddings)
    self.assertAllClose(h, expected_output)

  def test_block_mask(self):
    tf.random.set_seed(1234)
    inputs = tf.ones([3, 5, 10, 1], dtype=tf.float32)
    paddings = tf.zeros([3, 5])

    config = lib.AugmenterConfig(
        freq_mask_max_bins=0,
        time_mask_max_frames=0,
        block_mask_prob=0.8,
        block_mask_size=dict(t=2, f=3),
        random_seed=34567
    )
    expected_output = [
        [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 0., 0., 0., 1., 1., 1., 0.],
         [1., 1., 1., 0., 0., 0., 1., 1., 1., 0.],
         [0., 0., 0., 1., 1., 1., 1., 1., 1., 1.]],
        [[1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
         [1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 1., 1., 1., 0.],
         [0., 0., 0., 0., 0., 0., 1., 1., 1., 0.],
         [0., 0., 0., 0., 0., 0., 1., 1., 1., 1.]],
        [[0., 0., 0., 1., 1., 1., 0., 0., 0., 0.],
         [0., 0., 0., 1., 1., 1., 0., 0., 0., 0.],
         [1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
         [1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
         [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.]]
    ]
    h, _ = lib.tf_spec_augment_lingvo(config, inputs, paddings)
    h = tf.squeeze(h, -1)
    self.assertAllClose(h, expected_output)

  def test_warp_matrix_constructor(self):
    inputs = tf.broadcast_to(tf.cast(tf.range(10), dtype=tf.float32), (4, 10))
    origin = tf.cast([2, 4, 4, 5], dtype=tf.float32)
    destination = tf.cast([3, 2, 6, 8], dtype=tf.float32)
    choose_range = tf.cast([4, 8, 8, 10], dtype=tf.float32)

    expected_output = np.array([
        [0.0000000, 0.6666667, 1.3333333, 2.0000000, 4.0000000,
         5.0000000, 6.0000000, 7.0000000, 8.0000000, 9.0000000],
        [0.0000000, 2.0000000, 4.0000000, 4.6666667, 5.3333333,
         6.0000000, 6.6666667, 7.3333333, 8.0000000, 9.0000000],
        [0.0000000, 0.6666667, 1.3333333, 2.0000000, 2.6666667,
         3.3333333, 4.0000000, 6.0000000, 8.0000000, 9.0000000],
        [0.0000000, 0.6250000, 1.2500000, 1.8750000, 2.5000000,
         3.1250000, 3.7500000, 4.3750000, 5.0000000, 7.5000000]
    ])
    warp_matrix = lib._construct_warp_matrix(
        batch_size=4,
        matrix_size=10,
        origin=origin,
        destination=destination,
        choose_range=choose_range,
        dtype=tf.float32
    )
    outputs = tf.einsum("bij,bj->bi", warp_matrix, inputs)
    self.assertAllClose(outputs, expected_output)

  def test_frequency_warping(self):
    tf.random.set_seed(1234)
    inputs = tf.broadcast_to(
        tf.cast(tf.range(8), dtype=tf.float32), (5, 1, 8))
    inputs = tf.expand_dims(inputs, -1)
    paddings = tf.zeros([3, 2])

    config = lib.AugmenterConfig(
        freq_mask_max_bins=0,
        time_mask_max_frames=0,
        freq_warp_max_bins=4,
        time_warp_max_frames=0,
        random_seed=345678,
    )
    expected_output = np.array([
        [[0.0, 4.0, 4.5714283, 5.142857, 5.714286, 6.285714, 6.8571434,
          3.999998]],
        [[0.0, 0.8, 1.6, 2.4, 3.2, 4.0, 5.3333335, 6.6666665]],
        [[0.0, 0.6666667, 1.3333334, 2.0, 3.2, 4.4, 5.6000004, 6.8]],
        [[0.0, 1.3333334, 2.6666667, 4.0, 4.8, 5.6000004, 6.3999996,
          5.5999947]],
        [[0.0, 2.0, 2.857143, 3.7142859, 4.571429, 5.4285717, 6.2857146,
          5.999997]]
    ])
    h, _ = lib.tf_spec_augment_lingvo(config, inputs, paddings)
    h = tf.squeeze(h, -1)
    self.assertAllClose(h, expected_output)

  def test_time_warping(self):
    tf.random.set_seed(1234)
    inputs = tf.broadcast_to(tf.cast(tf.range(10), dtype=tf.float32), (3, 10))
    inputs = tf.expand_dims(tf.expand_dims(inputs, -1), -1)
    paddings = []
    for i in range(3):
      paddings.append(
          tf.concat([tf.zeros([1, i + 7]),
                     tf.ones([1, 3 - i])], axis=1))
    paddings = tf.concat(paddings, axis=0)

    config = lib.AugmenterConfig(
        freq_mask_max_bins=0,
        time_mask_max_frames=0,
        time_warp_max_frames=8,
        time_warp_max_ratio=1.0,
        time_warp_bound="static",
        random_seed=34567
    )
    expected_output = np.array([
        [[[0.0000000]], [[0.6666667]], [[1.3333334]], [[2.0000000]],
         [[2.6666667]], [[3.3333335]], [[4.0000000]], [[7.0000000]],
         [[8.0000000]], [[9.0000000]]],
        [[[0.0000000]], [[3.0000000]], [[6.0000000]], [[6.3333334]],
         [[6.6666665]], [[7.0000000]], [[7.3333334]], [[7.6666667]],
         [[8.0000000]], [[9.0000000]]],
        [[[0.0000000]], [[0.5000000]], [[1.0000000]], [[1.5000000]],
         [[2.0000000]], [[3.4000000]], [[4.8000000]], [[6.2000000]],
         [[7.6000000]], [[9.0000000]]]
    ])
    h, _ = lib.tf_spec_augment_lingvo(config, inputs, paddings)
    self.assertAllClose(h, expected_output)

  def test_dynamic_time_warping(self):
    tf.random.set_seed(1234)
    inputs = tf.broadcast_to(tf.cast(tf.range(10), dtype=tf.float32), (3, 10))
    inputs = tf.expand_dims(tf.expand_dims(inputs, -1), -1)
    paddings = []
    for i in range(3):
      paddings.append(
          tf.concat([tf.zeros([1, 2 * i + 5]),
                     tf.ones([1, 5 - 2 * i])],
                    axis=1))
    paddings = tf.concat(paddings, axis=0)

    config = lib.AugmenterConfig(
        freq_mask_max_bins=0,
        time_mask_max_frames=0,
        time_warp_max_ratio=0.5,
        time_warp_bound="dynamic",
        random_seed=34567
    )

    expected_output = np.array([
        [[[0.0000000]], [[1.0000000]], [[2.0000000]], [[3.0000000]],
         [[4.0000000]], [[5.0000000]], [[6.0000000]], [[7.0000000]],
         [[8.0000000]], [[9.0000000]]],
        [[[0.0000000]], [[0.8333333]], [[1.6666666]], [[2.5000000]],
         [[3.3333333]], [[4.1666665]], [[5.0000000]], [[7.0000000]],
         [[8.0000000]], [[9.0000000]]],
        [[[0.0000000]], [[2.0000000]], [[2.8750000]], [[3.7500000]],
         [[4.6250000]], [[5.5000000]], [[6.3750000]], [[7.2500000]],
         [[8.1250000]], [[9.0000000]]]
    ])
    h, _ = lib.tf_spec_augment_lingvo(config, inputs, paddings)
    self.assertAllClose(h, expected_output)

  def test_frequency_noise(self):
    tf.random.set_seed(1234)
    inputs = tf.broadcast_to(
        tf.cast(tf.range(8), dtype=tf.float32), (5, 1, 8))
    inputs = tf.expand_dims(inputs, -1)
    paddings = tf.zeros([3, 2])

    config = lib.AugmenterConfig(
        freq_noise_max_stddev=0.1,
        freq_mask_max_bins=0,
        time_mask_max_frames=0,
        freq_warp_max_bins=0,
        time_warp_max_frames=0,
        freq_noise_warmup_steps=0,
        random_seed=345678
    )
    # pylint: disable=bad-whitespace
    expected_output = np.array([
        [[0.      , 1.031674, 1.876506, 3.088444, 3.944183,
          5.056358, 6.092408, 7.001149]],
        [[0.      , 0.890657, 2.022598, 3.283400, 3.796596,
          4.939937, 6.306836, 6.721056]],
        [[0.      , 1.000362, 2.028894, 2.968441, 3.859807,
          5.032872, 6.005741, 6.991328]],
        [[0.      , 1.045312, 2.125809, 2.765134, 4.301796,
          4.97742 , 5.714708, 6.979644]],
        [[0.      , 1.005502, 2.019461, 3.001168, 3.968645,
          4.990373, 6.064750, 7.040662]]
    ])
    # pylint: enable=bad-whitespace
    h, _ = lib.tf_spec_augment_lingvo(config, inputs, paddings)
    h = tf.squeeze(h, -1)
    self.assertAllClose(h, expected_output)

  def test_noisify(self):
    tf.random.set_seed(127)
    batch_size = 2
    inputs = tf.ones([batch_size, 20, 2, 2], dtype=tf.float32)
    paddings = []
    for i in range(batch_size):
      paddings.append(
          tf.concat([tf.zeros([1, 8 * i + 3]),
                     tf.ones([1, 17 - 8 * i])],
                    axis=1))
    paddings = tf.concat(paddings, axis=0)

    config = lib.AugmenterConfig(
        freq_mask_max_bins=0,
        time_mask_max_ratio=0.4,
        time_mask_count=1,
        use_dynamic_time_mask_max_frames=True,
        use_noise=True,
        gaussian_noise=False,
        random_seed=12345
    )

    expected_output = np.array([
        [[[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[-0.00113627, -0.00113627],
          [0.08975883, 0.08975883]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]]],
        [[[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[0.09341543, 0.09341543],
          [-0.11914382, -0.11914382]],
         [[0.04238122, 0.04238122],
          [0.115249, 0.115249]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]]]
    ])
    h, _ = lib.tf_spec_augment_lingvo(config, inputs, paddings)
    self.assertAllClose(h, expected_output)

  def test_gaussian_noisify(self):
    tf.random.set_seed(127)
    batch_size = 2
    inputs = tf.ones([batch_size, 20, 2, 2], dtype=tf.float32)
    paddings = []
    for i in range(batch_size):
      paddings.append(
          tf.concat([tf.zeros([1, 8 * i + 3]),
                     tf.ones([1, 17 - 8 * i])],
                    axis=1))
    paddings = tf.concat(paddings, axis=0)

    config = lib.AugmenterConfig(
        freq_mask_max_bins=0,
        time_mask_max_ratio=0.4,
        time_mask_count=1,
        use_dynamic_time_mask_max_frames=True,
        use_noise=True,
        gaussian_noise=True,
        random_seed=12345
    )
    expected_output = np.array([
        [[[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[-0.00798237, -0.00798237],
          [0.6305642, 0.6305642]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]]],
        [[[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[0.6562522, 0.6562522],
          [-0.83699656, -0.83699656]],
         [[0.29773206, 0.29773206],
          [0.8096351, 0.8096351]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]],
         [[1.00000000, 1.00000000],
          [1.00000000, 1.00000000]]]
    ])
    h, _ = lib.tf_spec_augment_lingvo(config, inputs, paddings)
    self.assertAllClose(h, expected_output)

  def test_stateless_random_ops(self):
    batch_size = 5
    inputs1 = tf.random.uniform(
        shape=[batch_size, 20, 2, 2], minval=0, maxval=1, dtype=tf.float32)
    inputs2 = tf.random.uniform(
        shape=[batch_size, 20, 2, 2], minval=0, maxval=1, dtype=tf.float32)
    paddings = []
    for i in range(batch_size):
      paddings.append(
          tf.concat([tf.zeros([1, i + 12]),
                     tf.ones([1, 8 - i])], axis=1))
    paddings = tf.concat(paddings, axis=0)

    config = lib.AugmenterConfig(
        freq_mask_count=1,
        freq_mask_max_bins=1,
        time_mask_max_frames=5,
        time_mask_count=2,
        time_mask_max_ratio=1.0,
        use_input_dependent_random_seed=True
    )
    h1, _ = lib.tf_spec_augment_lingvo(config, inputs1, paddings)
    h2, _ = lib.tf_spec_augment_lingvo(config, inputs2, paddings)
    self.assertAllEqual(np.shape(h1), np.array([5, 20, 2, 2]))
    self.assertNotAllEqual(h1, h2)


if __name__ == "__main__":
  tf.test.main()
