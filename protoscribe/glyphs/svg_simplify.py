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

"""SVG simplification utilities."""

import math
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

_MARGIN_SIZE = flags.DEFINE_float(
    "margin_size", 0.05,
    "The minimum margin (empty area framing the collection of paths) size used "
    "for creating the canvas and background of the SVG."
)

_FIXED_SVG_WIDTH = flags.DEFINE_integer(
    "fixed_svg_width", -1,
    "Width of the SVG, also the width of the viewbox."
)

_FIXED_SVG_HEIGHT = flags.DEFINE_integer(
    "fixed_svg_height", -1,
    "Height of the SVG, also the height of the viewbox."
)

_SVG_MIN_DIM = flags.DEFINE_integer(
    "svg_min_dim", 600,
    "Fixed minimal dimension (width or height) or the resulting SVG. "
    "This preserves the aspect ratio. Used when `fix_svg_size` flag "
    "is disabled."
)

FLAGS = flags.FLAGS

# XML namespace for the parser.
XML_SVG_NAMESPACE = "http://www.w3.org/2000/svg"

# Name of glyph affiliation attribute.
XML_SVG_POSITION_AND_GLYPH = "position-and-glyph"

# XML tag used for searching for paths.
XML_SVG_FIND_TAG = ".//{%s}" % XML_SVG_NAMESPACE


def basic_svg_xml_header(width: float, height: float) -> str:
  """Returns basic XML SVG header for the document.

  Base units (e.g., pixels `px`) are omitted.

  Args:
    width: Width of the document.
    height: Height of the document.

  Returns:
    Header string.
  """
  return (
      "<?xml version=\"1.0\" encoding=\"utf-8\" ?>\n"
      "<svg version=\"1.1\" xmlns=\"http://www.w3.org/2000/svg\" "
      "xmlns:xlink=\"http://www.w3.org/1999/xlink\" "
      f"width=\"{int(width)}\" height=\"{int(height)}\" "
      f"viewBox=\"0 0 {width} {height}\">\n"
  )


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


def _bbox_with_margins_and_style(
    paths: list[path_lib.Path],
) -> tuple[float, float, float, float]:
  """Returns the bounding box that takes into account margins and style.

  Args:
    paths: List of paths.

  Returns:
    The new viewbox (min_x, min_y, width, height).
  """
  # Bounding box that takes into account all the segment curves.
  min_x, max_x, min_y, max_y = paths2svg.big_bounding_box(paths)
  width, height = max_x - min_x, max_y - min_y
  width = 1 if width == 0 else width
  height = 1 if height == 0 else height

  # Adjust the bounding box taking into account the style (currently
  # the stroke width only) and margin size.
  extra_style = STROKE_WIDTH.value
  margin_size = _MARGIN_SIZE.value
  min_x -= (margin_size * width + extra_style / 2.)
  min_y -= (margin_size * height + extra_style / 2.)
  width += (2 * margin_size * width + extra_style)
  height += (2 * margin_size * height + extra_style)
  return min_x, min_y, width, height


def resize_paths(
    paths: list[path_lib.Path],
    new_width: int,
    new_height: int,
    min_dim: int = -1
) -> tuple[
    list[path_lib.Path], tuple[float, float, float, float]
]:
  """Resizes all the paths to the given size.

  Args:
    paths: List of paths.
    new_width: New width.
    new_height: New height.
    min_dim: Minimal dimension (width or height). If positive, overrides the
      fixed dimensions provided by `new_width` and `new_height`.

  Returns:
    A three-tuple consisting of:
      - Resized paths (translated and scaled appropriately).
      - The new viewbox (min_x, min_y, width, height).
  """
  min_x, min_y, width, height = _bbox_with_margins_and_style(paths)

  # Check the requested new dimensions. These can be both fixed (if `min_dim`
  # is negative) or one of them is specified by `min_dim`.
  if min_dim > 0:
    # When only one fixed dimension is specified, preserve the aspect ratio.
    if width > height:
      new_width = min_dim
      new_height = int(math.ceil(min_dim * height / width))
    else:
      new_width = int(math.ceil(min_dim * width / height))
      new_height = min_dim
  elif new_width <= 0 or new_height <= 0:
    raise ValueError("Both fixed width and height need to be specified!")
  view_box = (0, 0, new_width, new_height)
  scale_x = new_width / width
  scale_y = new_height / height

  # Transform the paths.
  new_paths = []
  for path in paths:
    new_paths.append(
        path
        .translated(-complex(min_x, min_y))
        .scaled(scale_x, scale_y)
    )
  return new_paths, view_box


def _paths_to_svg_tree(
    paths: list[path_lib.Path], glyphs_info: list[str]
) -> ET.ElementTree:
  """Converts the paths along with the glyph information to an XML tree.

  Args:
    paths: Path instances.
    glyphs_info: Glyph information strings, one per path.

  Returns:
    SVG element tree.
  """
  paths, view_box = resize_paths(
      paths,
      new_width=_FIXED_SVG_WIDTH.value,
      new_height=_FIXED_SVG_HEIGHT.value,
      min_dim=_SVG_MIN_DIM.value
  )
  buf = basic_svg_xml_header(view_box[2], view_box[3])
  for p, glyph_info in zip(paths, glyphs_info):
    buf += (
        f"<path d=\"{p.d()}\" "
        f"fill=\"none\" stroke=\"#000000\" stroke-width=\"%f\" %s/>\n" % (
            STROKE_WIDTH.value,
            "%s=\"%s\" " % (
                XML_SVG_POSITION_AND_GLYPH, glyph_info
            ) if glyph_info else ""
        )
    )
  buf += "</svg>"
  return ET.ElementTree(ET.fromstring(buf))


def find_paths(tree: ET.ElementTree) -> list[ET.Element]:
  """Finds all the paths under the given XML element.

  Args:
    tree: XML tree.

  Returns:
    A list of paths as XML tree elements.
  """
  return tree.getroot().findall(f"{XML_SVG_FIND_TAG}path")


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

  # Sort the paths and collect the glyph information, if present.
  flat_paths = sorted(flat_paths, key=_path_sort_key)
  glyphs_info = []
  for path in flat_paths:
    glyph_info = None
    if XML_SVG_POSITION_AND_GLYPH in path.element.attrib:
      glyph_info = path.element.attrib[XML_SVG_POSITION_AND_GLYPH]
    glyphs_info.append(glyph_info)

  # Convert the paths to a simple XML element tree and check that the paths
  # are preserved.
  simplified_xml_tree = _paths_to_svg_tree(flat_paths, glyphs_info)
  new_paths = find_paths(simplified_xml_tree)
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
