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

"""Simple tests for text generation library."""

import os
import tempfile

from absl import flags
from absl.testing import absltest
from protoscribe.glyphs import make_text_lib

import glob
import os

_GENERATE_GOLDEN = flags.DEFINE_bool(
    "generate_golden", False,
    "Generate golden data."
)

FLAGS = flags.FLAGS

_TEST_DATA_DIR = "protoscribe/glyphs/testdata"


def _load_svg(path: str) -> str:
  """Returns SVG string buffer."""
  with open(path, mode="rt") as s:
    return "".join([c.strip() for c in s.readlines()])


class MakeTextTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.glyphs = ["X", "X", "I", "I", "clay", "brick"]
    cls.svgs = []
    for glyph in cls.glyphs:
      cls.svgs.append(
          os.path.join(FLAGS.test_srcdir, _TEST_DATA_DIR, f"{glyph}.svg")
      )
    cls.golden_path = os.path.join(
        FLAGS.test_srcdir, _TEST_DATA_DIR, "golden.svg"
    )
    cls.golden = _load_svg(cls.golden_path)
    cls.golden_for_strokes_path = os.path.join(
        FLAGS.test_srcdir, _TEST_DATA_DIR, "golden_for_strokes.svg"
    )
    cls.golden_for_strokes = _load_svg(cls.golden_for_strokes_path)

  def testSvgMatch(self) -> None:
    svg, svg_for_strokes, _, _ = make_text_lib.concat_svgs(
        self.svgs,
        self.glyphs,
    )
    if _GENERATE_GOLDEN.value:
      svg.write(self.golden_path)
      svg_for_strokes.write(self.golden_for_strokes_path)
    else:
      for glyph, golden_glyph in [
          (svg, self.golden),
          (svg_for_strokes, self.golden_for_strokes),
      ]:
        tmpfile = tempfile.NamedTemporaryFile(suffix=".svg", prefix="/tmp/")
        glyph.write(tmpfile.name)
        test = _load_svg(tmpfile.name)
        self.assertEqual(golden_glyph, test)


if __name__ == "__main__":
  absltest.main()