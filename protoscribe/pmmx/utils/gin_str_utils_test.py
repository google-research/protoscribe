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

"""Tests for pmmx.utils.gin_str_utils."""

from absl.testing import absltest
import gin
from protoscribe.pmmx.utils import gin_str_utils


class GinStrUtilsTest(absltest.TestCase):
  """A test class for mask_utils."""

  def setUp(self):
    super(GinStrUtilsTest, self).setUp()
    gin.clear_config()  # Clearing the config is required before every test.

  def test_join(self):
    """Test that join() properly joins its input values."""
    # Test with default delimiter.
    result = gin_str_utils.join(['a', 'b', 'c'])
    self.assertEqual(result, 'a,b,c')

    # Test with default a non-default delimiter.
    result = gin_str_utils.join(['a', 'b', 'c'], delimiter='/')
    self.assertEqual(result, 'a/b/c')

    # Test with mixed string/non-string values and the empty-string delimiter.
    result = gin_str_utils.join(['a', 1, 2.0], delimiter='')
    self.assertEqual(result, 'a12.0')

  def test_gin_join(self):
    """Test that join() works properly in gin."""
    gin.bind_parameter('join.values', ['a', 'b', 'c'])
    gin.bind_parameter('join.delimiter', '/')
    self.assertEqual('a/b/c', gin_str_utils.join())

if __name__ == '__main__':
  absltest.main()
