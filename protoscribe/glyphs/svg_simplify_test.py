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

import xml.etree.ElementTree as ET

from absl.testing import absltest
from protoscribe.glyphs import svg_simplify as lib
from svgpathtools import svg_to_paths

_SVG_CONTENTS = """<?xml version="1.0" encoding="utf-8"?>
<svg
  version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
  width="600" height="600" viewBox="0 0 600 600"
>
<path style="fill: none; stroke: 000000;" d="M0 92.42 C0 92.42 1.923 87.873 2.78 86.16 3.4 84.921 3.784 84.307 4.53 83.03 5.739 80.96 7.563 78.081 9.44 74.98 11.997 70.757 15.561 64.315 18.45 59.99 20.781 56.499 22.918 54.188 25.22 50.83 27.883 46.944 30.745 41.956 33.39 37.95 35.756 34.367 38.197 31.063 40.3 27.89 42.149 25.1 43.783 22.401 45.38 19.94 46.777 17.789 48.023 15.925 49.35 13.92 50.677 11.915 52.235 9.713 53.34 7.91 54.209 6.493 54.925 5.24 55.53 4.03 56.038 3.015 56.412 1.895 56.79 1.16 57.038 0.678 57.48 0 57.48 0 57.48 0 57.48 0 57.48 0 " transform="matrix(1.4872455600702508,0.08934196970717541,-0.08934196970717542,1.487245560070251,281.87712285963966,201.64123466839595)"/>
<path style="fill: none; stroke: 000000;" d="M0 31.11 C0 31.11 3.131 26.033 4.46 23.68 5.629 21.611 6.558 19.647 7.6 17.76 8.571 16.001 9.747 14.302 10.5 12.71 11.129 11.38 11.403 10.102 11.94 8.91 12.443 7.794 13.114 6.691 13.6 5.76 13.986 5.022 14.29 4.434 14.63 3.77 14.97 3.107 15.407 2.312 15.64 1.78 15.789 1.439 15.866 1.214 15.96 0.92 16.056 0.621 16.21 0 16.21 0 " transform="matrix(1.4872455600702508,0.08934196970717541,-0.08934196970717542,1.487245560070251,331.00165305723215,270.31044763510755)"/>
</svg>
"""

_SVG_WITH_GLYPH_AFFILIATION_CONTENTS = """<?xml version="1.0" encoding="utf-8"?>
<svg
  version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
  width="600" height="600" viewBox="0 0 600 600"
>
<path style="fill: none; stroke: 000000;" d="M0 92.42 C0 92.42 1.923 87.873 2.78 86.16 3.4 84.921 3.784 84.307 4.53 83.03 5.739 80.96 7.563 78.081 9.44 74.98 11.997 70.757 15.561 64.315 18.45 59.99 20.781 56.499 22.918 54.188 25.22 50.83 27.883 46.944 30.745 41.956 33.39 37.95 35.756 34.367 38.197 31.063 40.3 27.89 42.149 25.1 43.783 22.401 45.38 19.94 46.777 17.789 48.023 15.925 49.35 13.92 50.677 11.915 52.235 9.713 53.34 7.91 54.209 6.493 54.925 5.24 55.53 4.03 56.038 3.015 56.412 1.895 56.79 1.16 57.038 0.678 57.48 0 57.48 0 57.48 0 57.48 0 57.48 0 " transform="matrix(1.4872455600702508,0.08934196970717541,-0.08934196970717542,1.487245560070251,281.87712285963966,201.64123466839595)" position-and-glyph="0,aardvark"/>
<path style="fill: none; stroke: 000000;" d="M0 31.11 C0 31.11 3.131 26.033 4.46 23.68 5.629 21.611 6.558 19.647 7.6 17.76 8.571 16.001 9.747 14.302 10.5 12.71 11.129 11.38 11.403 10.102 11.94 8.91 12.443 7.794 13.114 6.691 13.6 5.76 13.986 5.022 14.29 4.434 14.63 3.77 14.97 3.107 15.407 2.312 15.64 1.78 15.789 1.439 15.866 1.214 15.96 0.92 16.056 0.621 16.21 0 16.21 0 " transform="matrix(1.4872455600702508,0.08934196970717541,-0.08934196970717542,1.487245560070251,331.00165305723215,270.31044763510755)" position-and-glyph="1,platypus"/>
</svg>
"""

# Total number of Bezier cubic segments in the above sketch.
_NUM_SEGMENTS = 23


class SvgSimplifyTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    ET.register_namespace("", lib.XML_SVG_NAMESPACE)

  def test_num_segments(self):
    paths, _ = svg_to_paths.svgstr2paths(_SVG_CONTENTS)
    self.assertEqual(lib.num_segments(paths), _NUM_SEGMENTS)

  def test_simplify_svg_tree(self):
    tree = ET.ElementTree(ET.fromstring(_SVG_CONTENTS))
    simplified_tree, num_paths, num_segments = lib.simplify_svg_tree(tree)
    self.assertEqual(2, num_paths)
    self.assertEqual(_NUM_SEGMENTS, num_segments)

    # Check that all the original XML attributes excluding the transforms are
    # in the modified tree.
    paths = simplified_tree.findall("path")
    self.assertLen(paths, num_paths)
    for path_elt in paths:
      self.assertIn("d", path_elt.attrib)
      self.assertIn("fill", path_elt.attrib)
      self.assertIn("stroke", path_elt.attrib)
      self.assertNotIn("transform", path_elt.attrib)

  def test_simplify_svg_tree_with_glyph_affiliations(self):
    tree = ET.ElementTree(ET.fromstring(_SVG_WITH_GLYPH_AFFILIATION_CONTENTS))
    simplified_tree, num_paths, num_segments = lib.simplify_svg_tree(tree)
    self.assertEqual(2, num_paths)
    self.assertEqual(_NUM_SEGMENTS, num_segments)

    # Check that the paths have glyph affiliations.
    paths = simplified_tree.findall("path")
    self.assertLen(paths, num_paths)
    self.assertIn(lib.XML_SVG_POSITION_AND_GLYPH, paths[0].attrib)
    self.assertEqual(
        "0,aardvark", paths[0].attrib[lib.XML_SVG_POSITION_AND_GLYPH]
    )
    self.assertIn(lib.XML_SVG_POSITION_AND_GLYPH, paths[1].attrib)
    self.assertEqual(
        "1,platypus", paths[1].attrib[lib.XML_SVG_POSITION_AND_GLYPH]
    )


if __name__ == "__main__":
  absltest.main()
