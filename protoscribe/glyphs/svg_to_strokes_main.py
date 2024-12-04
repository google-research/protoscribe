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

"""Convert SVG into a sequence of strokes as points."""

import logging
import xml.etree.ElementTree as ET

from absl import app
from absl import flags
from protoscribe.glyphs import svg_to_strokes_lib
from protoscribe.glyphs import vector_to_raster as raster_lib

import glob
import os

_SVG_FILE = flags.DEFINE_string(
    "svg_file", None,
    "Path to SVG file.",
    required=True
)

_OUTPUT_POINTS_FILE = flags.DEFINE_string(
    "output_points_file", None,
    "Output strokes to a file as points."
)

_OUTPUT_PNG_FILE = flags.DEFINE_string(
    "output_png_file", None,
    "Convert the strokes to a rasterized image and save it in PNG format."
)

_RASTER_SIZE = flags.DEFINE_integer(
    "raster_size", 256,
    "Size of the raster image in pixels."
)

FLAGS = flags.FLAGS

# Hardcode stroke width to 1 for testing the generation of PNG from strokes.
FLAGS.stroke_width = 1


def _get_svg_dimensions() -> tuple[float, float]:
  """Attempts to read SVG dimensions."""

  def _get_dims(elt) -> tuple[float, float]:
    # Some of the SVGs lack `width` and `height attributes, in which
    # case we try the `viewBox`.
    try:
      w = float(elt.attrib["width"].replace("px", ""))
      h = float(elt.attrib["height"].replace("px", ""))
    except KeyError:
      viewbox = elt.attrib["viewBox"].split()
      w = float(viewbox[-2])
      h = float(viewbox[-1])
    return w, h

  with open(_SVG_FILE.value, "r") as stream:
    tree = ET.parse(stream)
    return _get_dims(tree.getroot())


def main(unused_argv):
  strokes, _ = svg_to_strokes_lib.svg_to_strokes(_SVG_FILE.value)
  if _OUTPUT_POINTS_FILE.value:
    with open(_OUTPUT_POINTS_FILE.value, "w") as stream:
      svg_to_strokes_lib.print_text(strokes, stream)

  if _OUTPUT_PNG_FILE.value:
    # Massage the points into the format expected by the `raster_lib` API below.
    drawing = []
    avg_num_points = 0.
    for stroke in strokes:
      x = [p[0] for p in stroke]
      y = [p[1] for p in stroke]
      avg_num_points += len(x)
      drawing.append((x, y))
    logging.info("Points / stroke: %f", avg_num_points / len(strokes))

    # Generate RGBA image.
    svg_width, svg_height = _get_svg_dimensions()
    logging.info("SVG Dimensions: (%d, %d)", svg_width, svg_height)
    if svg_width <= 0 or svg_width != svg_height:
      raise ValueError("Expecting non-empty symmetric SVG image!")
    image = raster_lib.stroke_points_to_raster(
        drawing,
        source_side=int(svg_width),
        target_side=_RASTER_SIZE.value
    )
    # Convert to single-channel grayscale and save.
    image.convert("L").save(_OUTPUT_PNG_FILE.value, format="PNG")


if __name__ == "__main__":
  app.run(main)
