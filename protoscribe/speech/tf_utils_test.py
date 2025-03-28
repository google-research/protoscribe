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

"""Tests for miscellaneous Tensorflow helper APIs."""

from protoscribe.speech import tf_utils
import tensorflow as tf


class TfUtilsTest(tf.test.TestCase):

  def test_global_step(self):
    _ = tf_utils.get_or_create_global_step_var()
    for i in range(10):
      step = tf_utils.get_global_step()
      self.assertIsNotNone(step)
      self.assertEqual(step.name, "global_step:0")
      self.assertEqual(step.numpy(), i)
      tf.compat.v1.assign(tf_utils.get_or_create_global_step_var(), i + 1)


if __name__ == "__main__":
  tf.test.main()
