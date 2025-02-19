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

"""Tests for stroke utilities."""

import json
import logging
import os

from absl.testing import absltest
import numpy as np
from protoscribe.glyphs import make_text
from protoscribe.glyphs import svg_to_strokes
from protoscribe.sketches.utils import stroke_utils as lib

import glob
import os

_POINTS_PER_SEGMENT = 10

_TEST_DATA_DIR = (
    "protoscribe/sketches/utils/testdata"
)


class StrokeUtilsTest(absltest.TestCase):

  def test_concrete_svg_build_time_stroke_points(self):
    """Tests the point generation build-time API."""

    glyph_path = os.path.join(
        absltest.get_default_test_srcdir(),
        _TEST_DATA_DIR,
        "field.svg"
    )
    _, svg_for_strokes, _, _ = make_text.concat_svgs(
        [glyph_path], ["field"]
    )
    strokes, glyph_affiliations = svg_to_strokes.svg_tree_to_strokes_for_test(
        svg_for_strokes, points_per_segment=_POINTS_PER_SEGMENT
    )
    num_strokes = len(strokes)
    self.assertNotEmpty(strokes)
    x_stroke_points, y_stroke_points, _, _, _ = lib.stroke_points(
        strokes, glyph_affiliations
    )
    self.assertNotEmpty(x_stroke_points)
    self.assertEqual(len(x_stroke_points), len(y_stroke_points))

    # Reconstruct individual strokes.
    sketch = []
    stroke = []
    x = None
    for x in x_stroke_points:
      if x == lib.END_OF_STROKE:
        self.assertNotEmpty(stroke)
        sketch.append(stroke)
      else:
        stroke.append(x)
    self.assertEqual(x, lib.END_OF_STROKE)
    self.assertNotEmpty(sketch)
    self.assertLen(sketch, num_strokes)

  def test_stroke3_strokes_to_svg_file(self):
    """Tests the stroke-to-SVG conversion."""

    # Loads strokes from n-best list of stroke sequences in stroke-3 format.
    path = os.path.join(
        absltest.get_default_test_srcdir(),
        _TEST_DATA_DIR,
        "cherry_nbest_strokes3.json"
    )
    with open(path, mode="r") as f:
      stroke_dict = json.loads(f.read())
      self.assertIn("strokes", stroke_dict)
      strokes = np.array(stroke_dict["strokes"][-1], dtype=np.float32)
    self.assertLen(strokes.shape, 2)
    self.assertEqual(strokes.shape[1], 3)

    # Supply `--test_tmpdir=/tmp` to examine the resulting SVG.
    for scale_factor in  [1.0, 0.5, 1.5]:
      svg_filename = os.path.join(
          absltest.get_default_test_tmpdir(), f"strokes_scale{scale_factor}.svg"
      )
      logging.info("Saving strokes to `%s` ...", svg_filename)
      lib.stroke3_strokes_to_svg_file(
          strokes, svg_filename, scale_factor=scale_factor
      )

  def test_stroke3_from_strokes(self):
    """Tests conversion of regular strokes to stroke-3 format."""

    strokes = [
        [(1., 1.), (3., 3.), (6., 6.)],
        [(8., 8.), (5., 5.), (4., 4.), (2., 2.), (0., 0.)],
    ]
    strokes3 = lib.stroke3_from_strokes(
        strokes, convert_to_deltas=False
    ).tolist()
    self.assertEqual(
        strokes3, [
            [1., 1., 0.],
            [3., 3., 0.],
            [6., 6., 1.],
            [8., 8., 0.],
            [5., 5., 0.],
            [4., 4., 0.],
            [2., 2., 0.],
            [0., 0., 1.],
        ]
    )
    strokes3_deltas = lib.stroke3_from_strokes(
        strokes, convert_to_deltas=True
    ).tolist()
    self.assertEqual(
        strokes3_deltas, [
            [1., 1., 0.],
            [2., 2., 0.],
            [3., 3., 1.],
            [2., 2., 0.],
            [-3., -3., 0.],
            [-1., -1., 0.],
            [-2., -2., 0.],
            [-2., -2., 1.],
        ]
    )

    test_strokes = [strokes3_deltas[0]]
    for i in range(1, len(strokes3_deltas)):
      d_x, d_y, lift_pen = strokes3_deltas[i]
      test_strokes.append(
          [
              test_strokes[i - 1][0] + d_x,
              test_strokes[i - 1][1] + d_y,
              lift_pen
          ]
      )
    self.assertEqual(test_strokes, strokes3)


if __name__ == "__main__":
  absltest.main()
