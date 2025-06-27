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

"""Basic tests for builder utilities."""

from absl.testing import absltest
from protoscribe.corpus.builder import prepare_utils


class PrepareUtilsTest(absltest.TestCase):

  def testFlattenEmbedding(self):
    inputs_2d = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    dims, outputs_1d = prepare_utils.flatten_embedding(inputs_2d)
    self.assertLen(dims, 2)
    self.assertEqual(dims, [4, 3])
    self.assertLen(outputs_1d, 12)
    self.assertEqual(outputs_1d, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])


if __name__ == "__main__":
  absltest.main()
