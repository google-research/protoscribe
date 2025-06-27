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

import os
import xml.etree.ElementTree as ET

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from protoscribe.glyphs import svg_to_strokes as lib

import glob
import os

FLAGS = flags.FLAGS

# Don't apply stroke pruning.
FLAGS.apply_rdp = False


class SVGToStrokesTest(parameterized.TestCase, absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.rabbit_svg = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/glyphs/testdata/rabbit.svg",
    )
    golden_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/glyphs/testdata/rabbit_points.txt",
    )
    cls.golden_points = []
    with open(golden_path) as stream:
      for line in stream:
        x, y = line.split()
        cls.golden_points.append((float(x), float(y)))
    cls.golden_deltas = []
    prev_x, prev_y = 0., 0.
    for x, y in cls.golden_points:
      cls.golden_deltas.append((x - prev_x, y - prev_y))
      prev_x, prev_y = x, y
    # These were manually entered into the SVG, and are a bit silly since
    # normally the rabbit would be a single glyph. This is just to make sure
    # that the various parts get passed through.
    cls.rabbit_affiliations = (
        [(0, "RABBIT_BODY")] * 93  # Largest piece, the body.
        + [(1, "RABBIT_RUMP")] * 6  # Next largest piece, the "rump" stroke.
        + [(2, "RABBIT_EYEBALL")] * 2  # Then the eye.
    )

  def _check_strokes_type(self, strokes) -> None:
    """Checks that the strokes container is generally sane."""
    self.assertIsInstance(strokes, list)
    self.assertGreater(len(strokes), 1)
    self.assertIsInstance(strokes[0], list)
    self.assertGreater(len(strokes[0]), 1)
    self.assertLen(strokes[0][0], 2)
    self.assertIsInstance(strokes[0][0], tuple)

  @flagsaver.flagsaver(deltas=False)
  def testSimplePointConversion(self) -> None:
    strokes, glyph_affiliations = lib.svg_file_to_strokes(self.rabbit_svg)
    self._check_strokes_type(strokes)
    self.assertEqual(len(strokes), len(glyph_affiliations))
    self.assertEqual(glyph_affiliations, self.rabbit_affiliations)
    points = []
    for stroke in strokes:
      points.extend([p for p in stroke])
    self.assertEqual(self.golden_points, points)

  @flagsaver.flagsaver(deltas=True)
  def testDeltaPointConversion(self) -> None:
    strokes, glyph_affiliations = lib.svg_file_to_strokes(self.rabbit_svg)
    self.assertEqual(glyph_affiliations, self.rabbit_affiliations)
    points = []
    for stroke in strokes:
      points.extend([p for p in stroke])
    self.assertEqual(self.golden_deltas, points)

  def testSvgStringToStrokes(self) -> None:
    rabbit = ET.parse(self.rabbit_svg)
    strokes, glyph_affiliations = lib.svg_tree_to_strokes(rabbit)
    self.assertEqual(glyph_affiliations, self.rabbit_affiliations)
    points = []
    for stroke in strokes:
      points.extend([p for p in stroke])
    self.assertEqual(self.golden_points, points)

  @parameterized.named_parameters(
      # `path_is_stroke` denoted `pis` and `first_point_is_origin` by `fpio`
      # in test name annotations.
      ("rdp-nflip-ndeltas-npis-nfpio", True, False, False, False, False),
      ("nrdp-nflip-ndeltas-npis-nfpio", False, False, False, False, False),
      ("rdp-flip-ndeltas-npis-nfpio", True, True, False, False, False),
      ("nrdp-flip-ndeltas-npis-nfpio", False, True, False, False, False),
      ("rdp-nflip-deltas-npis-nfpio", True, False, True, False, False),
      ("nrdp-nflip-deltas-npis-nfpio", False, False, True, False, False),
      ("rdp-flip-deltas-npis-nfpio", True, True, True, False, False),
      ("nrdp-flip-deltas-npis-nfpio", False, True, True, False, False),
      ("rdp-nflip-ndeltas-pis-nfpio", True, False, False, True, False),
      ("nrdp-nflip-ndeltas-pis-nfpio", False, False, False, True, False),
      ("rdp-flip-ndeltas-pis-nfpio", True, True, False, True, False),
      ("nrdp-flip-ndeltas-pis-nfpio", False, True, False, True, False),
      ("rdp-nflip-deltas-pis-nfpio", True, False, True, True, False),
      ("nrdp-nflip-deltas-pis-nfpio", False, False, True, True, False),
      ("rdp-flip-deltas-pis-nfpio", True, True, True, True, False),
      ("nrdp-flip-deltas-pis-nfpio", False, True, True, True, False),
      ("rdp-nflip-ndeltas-npis-fpio", True, False, False, False, True),
      ("nrdp-nflip-ndeltas-npis-fpio", False, False, False, False, True),
      ("rdp-flip-ndeltas-npis-fpio", True, True, False, False, True),
      ("nrdp-flip-ndeltas-npis-fpio", False, True, False, False, True),
      ("rdp-nflip-deltas-npis-fpio", True, False, True, False, True),
      ("nrdp-nflip-deltas-npis-fpio", False, False, True, False, True),
      ("rdp-flip-deltas-npis-fpio", True, True, True, False, True),
      ("nrdp-flip-deltas-npis-fpio", False, True, True, False, True),
      ("rdp-nflip-ndeltas-pis-fpio", True, False, False, True, True),
      ("nrdp-nflip-ndeltas-pis-fpio", False, False, False, True, True),
      ("rdp-flip-ndeltas-pis-fpio", True, True, False, True, True),
      ("nrdp-flip-ndeltas-pis-fpio", False, True, False, True, True),
      ("rdp-nflip-deltas-pis-fpio", True, False, True, True, True),
      ("nrdp-nflip-deltas-pis-fpio", False, False, True, True, True),
      ("rdp-flip-deltas-pis-fpio", True, True, True, True, True),
      ("nrdp-flip-deltas-pis-fpio", False, True, True, True, True),
  )
  def testSimplePointPathIsStrokeConversion(
      self,
      apply_rdp: bool,
      flip_vertical: bool,
      deltas: bool,
      path_is_stroke: bool,
      first_point_is_origin: bool
  ) -> None:
    rabbit = ET.parse(self.rabbit_svg)
    strokes, glyph_affiliations = lib.svg_tree_to_strokes_for_test(
        rabbit,
        path_is_stroke=path_is_stroke,
        apply_rdp=apply_rdp,
        flip_vertical=flip_vertical,
        deltas=deltas,
        first_point_is_origin=first_point_is_origin
    )
    self._check_strokes_type(strokes)
    self.assertNotEmpty(glyph_affiliations)

    if path_is_stroke:
      # There are three paths in `rabbit.svg`: two real paths and a circle.
      num_paths = 3
      self.assertLen(strokes, num_paths)

    if first_point_is_origin:
      self.assertEqual(strokes[0][0], (0., 0.))


if __name__ == "__main__":
  absltest.main()
