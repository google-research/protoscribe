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

"""Very basic tests for conversion from vector graphics to raster format."""

from absl.testing import absltest
from protoscribe.glyphs import vector_to_raster as lib

_RASTER_SIZE = 20


class VectorToRasterTest(absltest.TestCase):

  def test_stroke_points_conversion(self):
    # Each tuple consists of lists of x and y coordinates of equal length.
    stroke_points = [
        ([1., 2.], [1., 2.]),
        ([4., 5., 6.], [4., 5., 6.]),
        ([7., 8., 9., 10.], [7., 8., 9., 10.]),
    ]
    image = lib.stroke_points_to_raster(
        stroke_points=stroke_points,
        source_side=11,
        target_side=_RASTER_SIZE,
        padding=0.
    )
    self.assertEqual(image.size, (_RASTER_SIZE, _RASTER_SIZE))
    self.assertEqual(image.mode, "RGBA")


if __name__ == "__main__":
  absltest.main()
