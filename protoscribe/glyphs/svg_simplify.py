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

"""SVG simplification utilities."""

import xml.etree.ElementTree as ET

from absl import flags
from svgpathtools import document as svg_doc_lib
from svgpathtools import path as path_lib
from svgpathtools import paths2svg
from svgpathtools import svg_to_paths

STROKE_WIDTH = flags.DEFINE_float(
    "stroke_width", 5.0,
    "Stroke width used when generating SVG from individual strokes."
)

MARGIN_SIZE = flags.DEFINE_float(
    "margin_size", 0.1,
    "The minimum margin (empty area framing the collection of paths) size used "
    "for creating the canvas and background of the SVG."
)

# XML namespace for the parser.
XML_SVG_NAMESPACE = "http://www.w3.org/2000/svg"

# Name of glyph affiliation attribute.
XML_SVG_POSITION_AND_GLYPH = "position-and-glyph"


def _path_sort_key(path: path_lib.Path) -> tuple[int, float, float]:
  """Returns sorting key tuple for the given path.

  Make sure the document is "spelled" left-to-right and top-to-botoom by
  sorting the paths using either of the following critera:
    - If `position-and-glyph` attribute is present, sort first by the ascending
      glyph position and as a secondary and tertiary keys by the ascending
      offsets of the path along the x- and y-axis.
    - If no glyph information is present sort by the ascending offset of the
      path along the x- and y-axis.

  Args:
    path: SVG path instance.

  Returns: A three-tuple where
    - The primary key is the glyph position.
    - The secondary key is the x offset.
    - The tertiary key is the y offset.
  """
  assert path.element is not None
  primary = (
      int(path.element.attrib[XML_SVG_POSITION_AND_GLYPH].split(",")[0])
      if XML_SVG_POSITION_AND_GLYPH in path.element.attrib
      else -1
  )
  return primary, float(path.start.real), float(path.start.imag)


def num_segments(paths: list[path_lib.Path]) -> int:
  """Returns total number of segments in the paths.

  Args:
    paths: Multi-segment paths.

  Returns:
   Total number of segments.
  """
  n = 0
  for p in paths:
    n += len(p)
  return n


def simplify_svg_tree(tree: ET.ElementTree) -> tuple[ET.ElementTree, int, int]:
  """Simplifies the structure of SVG XML tree.

  Will flatten all the paths by removing the transforms and convert all the
  non-path elements to path objects.

  Args:
    tree: XML representation of an SVG.

  Returns:
    A tuple consisting of:
      - A simplified SVG with no transforms and zooming on the actual content.
      - Number of paths in an SVG.
      - Total number of segments.
  """
  ET.register_namespace("", XML_SVG_NAMESPACE)

  # Fetch the original paths, ignore attributes which are retrieved from later
  # from the flattened paths.
  sanity_check_paths, _ = svg_to_paths.svgstr2paths(
      ET.tostring(tree.getroot()).decode("utf8")
  )

  # Generate flattened version of the paths using the `Document` interface.
  # There should be the same number of those as in the original.
  doc = svg_doc_lib.Document.from_svg_string(
      ET.tostring(tree.getroot()).decode("utf8")
  )
  flat_paths = doc.paths()
  if len(sanity_check_paths) != len(flat_paths):
    raise ValueError(
        f"Number of flattened paths {len(flat_paths)} should match the number "
        f"of original paths {len(sanity_check_paths)}"
    )

  # Sort the paths and build the corresponding simplified attributes.
  flat_paths = sorted(flat_paths, key=_path_sort_key)
  attributes = []
  for path in flat_paths:
    attrib = {
        "fill": "none",
        "stroke": "#000000",
        "stroke-width": f"{STROKE_WIDTH.value}",
    }
    if XML_SVG_POSITION_AND_GLYPH in path.element.attrib:
      attrib[XML_SVG_POSITION_AND_GLYPH] = (
          path.element.attrib[XML_SVG_POSITION_AND_GLYPH]
      )
    attributes.append(attrib)

  # Convert the paths to a simple XML element tree. Note, when converting the
  # attributes to `svgwrite.Drawing`, `svgwrite` replaces all the underscores
  # with dashes in attribute names.
  stroke_widths = [STROKE_WIDTH.value] * len(flat_paths)
  simplified_svg = paths2svg.paths2Drawing(
      paths=flat_paths,
      attributes=attributes,
      stroke_widths=stroke_widths,
      margin_size=MARGIN_SIZE.value
  )
  simplified_xml_tree = ET.ElementTree(simplified_svg.get_xml())

  # Sanity check that the paths are preserved.
  new_paths = simplified_xml_tree.getroot().findall("path")
  if len(new_paths) != len(flat_paths):
    raise ValueError(
        f"Number of flattened paths {len(flat_paths)} should match the number "
        f"of new paths {len(new_paths)}"
    )

  ET.indent(simplified_xml_tree)
  return simplified_xml_tree, len(flat_paths), num_segments(flat_paths)


def simplify_svg_str(svg: str) -> tuple[str, int, int]:
  """Simplifies the structure of SVG XML expressed as string.

  Will flatten all the paths by removing the transforms and convert all the
  non-path elements to path objects.

  Args:
    svg: String representation of an SVG XML.

  Returns:
    A tuple consisting of:
      - A simplified SVG XML as a string.
      - Number of paths in an SVG.
      - Total number of segments.
  """
  simplified_tree, num_paths, num_path_segments = simplify_svg_tree(
      ET.ElementTree(ET.fromstring(svg))
  )
  simplified_svg = (
      """<?xml version="1.0" encoding="utf-8"?>\n""" +
      ET.tostring(simplified_tree.getroot()).decode("utf8")
  )
  return simplified_svg, num_paths, num_path_segments
