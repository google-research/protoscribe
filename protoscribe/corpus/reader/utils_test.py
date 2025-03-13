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

from absl.testing import absltest
from protoscribe.corpus.reader import utils
import tensorflow as tf


class UtilsTest(absltest.TestCase):

  def _pad_or_trim_list(
      self, inputs: list[int] | list[list[int]], max_sequence_length: int
  ) -> list[int] | list[list[int]]:
    """Helper for performing the operation returning a list."""
    return utils.pad_or_trim_sequence(
        tf.constant(inputs), max_sequence_length
    ).numpy().tolist()

  def test_1D_trim_or_pad(self):
    max_sequence_length = 7
    inputs = [1, 2, 3, 4]
    self.assertEqual(
        self._pad_or_trim_list(inputs, max_sequence_length=len(inputs)),
        inputs
    )
    self.assertEqual(
        self._pad_or_trim_list(inputs, max_sequence_length),
        [1, 2, 3, 4, 0, 0, 0]
    )
    inputs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    self.assertEqual(
        self._pad_or_trim_list(inputs, max_sequence_length),
        [1, 2, 3, 4, 5, 6, 7]
    )

  def test_2D_trim_or_pad(self):
    inputs = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    self.assertEqual(
        self._pad_or_trim_list(inputs, max_sequence_length=len(inputs)),
        inputs
    )
    self.assertEqual(
        self._pad_or_trim_list(inputs, max_sequence_length=7),
        [
            [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12],
            [0, 0, 0], [0, 0, 0], [0, 0, 0]
        ]
    )
    self.assertEqual(
        self._pad_or_trim_list(inputs, max_sequence_length=3),
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    )


if __name__ == "__main__":
  absltest.main()
