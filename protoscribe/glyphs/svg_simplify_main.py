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

"""Tool for experimenting with offline SVG file simplification.

Please note, this is a tool for experimentation rather than utility for
production.
"""

from collections.abc import Sequence
import logging
import os
import random
import xml.etree.ElementTree as ET

from absl import app
from absl import flags
import numpy as np
from protoscribe.glyphs import make_text
from protoscribe.glyphs import numbers_to_glyphs
from protoscribe.glyphs import svg_simplify
from protoscribe.glyphs import svg_to_strokes
from protoscribe.sketches.utils import stroke_utils
from svgpathtools import svg_to_paths

import glob
import os
# Internal resources dependency

_INPUT_SVG_DIR = flags.DEFINE_string(
    "input_svg_dir", None,
    "Input directory containing the SVG files to be processed.",
    required=True
)

_OUTPUT_SVG_DIR = flags.DEFINE_string(
    "output_svg_dir", None,
    "Output directory for simplified SVGs.",
    required=True
)

_ADD_NUMBERS = flags.DEFINE_boolean(
    "add_numbers", False,
    "Also generate SVGs that include numbers."
)

_INCLUDE_LIFT_POINTS = flags.DEFINE_boolean(
    "include_lift_points", False,
    "When computing the final number points in a document, include the lift "
    "points in the calculation. When tokenizing, additional token is inserted "
    "after each lift point."
)

FLAGS = flags.FLAGS

# Set some defaults for stroke conversion (see the flags values
# in `svg_to_strokes`).
FLAGS.flip_vertical = False
FLAGS.deltas = True
FLAGS.path_is_stroke = True

_MAX_NUMBER_VALUE = 10
_MAX_NUMBER_GLYPH_VARIANTS = 5
_NUMBERS_DIR = "protoscribe/data/glyphs/generic/numbers"


def _to_strokes_svg_str(file_path: str) -> tuple[str, int]:
  """Convert SVG to strokes and return the resulting SVG.

  Args:
    file_path: File containing the SVG.

  Returns:
    Tuple consisting of string representation of a stroke-based sketch in XML
    format and the resulting number of strokes.
  """
  svg_tree = ET.parse(file_path)

  # Convert the SVG paths consisting mostly of cubic Bezier curves to
  # simple strokes. Then convert these to stroke-3 format.
  strokes, _ = svg_to_strokes.svg_tree_to_strokes(svg_tree)
  strokes3 = stroke_utils.stroke3_from_strokes(
      strokes, convert_to_deltas=not FLAGS.deltas
  )

  # Convert stroke-3 format to SVG and compute number of points.
  strokes_svg_str = stroke_utils.stroke3_strokes_to_svg(strokes3)
  num_points = strokes3.shape[0]
  if _INCLUDE_LIFT_POINTS.value:
    # Add the points which will turn into extra end-of-path tokens.
    num_points += np.sum(strokes3[:, 2] == 1.)

  return strokes_svg_str, num_points


def _to_strokes_and_points_svg_str(file_path: str) -> tuple[str, int]:
  """Convert to strokes, then to points and return the resulting SVG.

  Similar to above, but here we test the conversion through an intermediat
  point representation which is used to store strokes in the dataset.

  Args:
    file_path: File containing the SVG.

  Returns:
    Tuple consisting of string representation of a stroke-based sketch in XML
    format and the resulting number of strokes.
  """
  svg_tree = ET.parse(file_path)

  # First convert to individual x and y points.
  strokes, _ = svg_to_strokes.svg_tree_to_strokes(svg_tree)
  stroke_glyph_affiliations = [(0, 0)] * len(strokes)
  x_strokes, y_strokes, _, _, _ = stroke_utils.stroke_points(
      strokes, stroke_glyph_affiliations
  )
  strokes3 = stroke_utils.points_to_strokes3(
      np.array(x_strokes, dtype=np.float32),
      np.array(y_strokes, dtype=np.float32)
  )

  # Convert stroke-3 format to SVG and compute number of points.
  strokes_svg_str = stroke_utils.stroke3_strokes_to_svg(strokes3)
  num_points = strokes3.shape[0]
  if _INCLUDE_LIFT_POINTS.value:
    # Add the points which will turn into extra end-of-path tokens.
    num_points += np.sum(strokes3[:, 2] == 1.)

  return strokes_svg_str, num_points


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  ET.register_namespace("", "http://www.w3.org/2000/svg")
  logging.info("Processing SVGs in %s ...", _INPUT_SVG_DIR.value)
  max_num_strokes = -1
  for file_path in glob.glob(os.path.join(_INPUT_SVG_DIR.value, "*.svg")):
    # Use the `standard` API to parse the SVG. Retain the transformations. We
    # use this for comparing with a newer document-based API.
    vanilla_paths, _ = svg_to_paths.svg2paths(file_path)
    vanilla_num_segments = svg_simplify.num_segments(vanilla_paths)
    vanilla_xml = ET.parse(file_path)

    # Following API will parse the SVG into the paths objects and apply all
    # the transforms on the paths, thus `flattening` them.
    file_name = os.path.basename(file_path)
    logging.info("[no transforms] Processing %s ...", file_path)
    svg_xml, num_paths, num_segments = svg_simplify.simplify_svg_tree(
        vanilla_xml
    )
    if num_paths != len(vanilla_paths):
      raise ValueError(
          f"{file_name}: Different number of paths after removing transforms!"
      )
    if num_segments != vanilla_num_segments:
      raise ValueError(
          f"{file_name}: Different number of segments after transform removal!"
      )

    # Dump the resulting XML to a file and re-read from a file. For some reason
    # the `Drawing` get_xml() returns XML with no root element.
    file_name = os.path.basename(file_path)
    glyph_path = os.path.join(_OUTPUT_SVG_DIR.value, file_name)
    svg_xml.write(glyph_path)

    # Form an accounting document by randomly picking a number.
    if _ADD_NUMBERS.value:
      # Load the individual digit glyphs.
      quantity = random.randint(1, _MAX_NUMBER_VALUE)
      digits = numbers_to_glyphs.pseudo_roman(quantity)

      digit_xmls = []
      digit_names = []
      for digit_name in digits:
        number_glyph_variant = random.randint(1, _MAX_NUMBER_GLYPH_VARIANTS)
        digit_path = os.path.join(
            _NUMBERS_DIR, f"{digit_name}_{number_glyph_variant}.svg"
        )
        with open(digit_path, mode="rt", encoding="utf-8") as f:
          digit_xmls.append(ET.parse(f))
        digit_names.append(digit_name)

      # Build accounting document XML and save the simplified version.
      make_text.FLAGS.simplify_svg_trees = True
      glyph_name = os.path.splitext(file_name)[0]
      doc_path = os.path.join(_OUTPUT_SVG_DIR.value, f"{quantity}_{file_name}")
      doc_path = doc_path.replace(".svg", "_simplified.svg")
      all_trees = digit_xmls + [vanilla_xml]
      all_glyphs = digit_names + [glyph_name]
      doc_xml, _, _ = make_text.concat_xml_svgs(
          trees=all_trees, glyphs=all_glyphs, clone_trees=True
      )
      doc_xml.write(doc_path)
      concat_xml_path = doc_path

      # Converts the accounting document to strokes.
      strokes_svg_str, num_strokes = _to_strokes_svg_str(concat_xml_path)
      doc_path = doc_path.replace("_simplified.svg", "_strokes.svg")
      ET.ElementTree(ET.fromstring(strokes_svg_str)).write(doc_path)
      max_num_strokes = max(max_num_strokes, num_strokes)

      # Convert to strokes, then points and go back to an SVG.
      strokes_svg_str, _ = _to_strokes_and_points_svg_str(concat_xml_path)
      doc_path = doc_path.replace("_strokes.svg", "_strokes_points.svg")
      ET.ElementTree(ET.fromstring(strokes_svg_str)).write(doc_path)

  logging.info("Done. Maximum number of strokes: %d", max_num_strokes)


if __name__ == "__main__":
  app.run(main)
