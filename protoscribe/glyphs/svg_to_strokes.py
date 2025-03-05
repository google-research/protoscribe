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

"""Methods to converts from SVG paths to strokes represnted as points."""

import sys
import xml.etree.ElementTree as ET

from absl import flags
import numpy as np
from protoscribe.glyphs import svg_simplify
from protoscribe.glyphs import trig
from svgpathtools.path import Path
from svgpathtools.svg_to_paths import svg2paths
from svgpathtools.svg_to_paths import svgstr2paths

_FLIP_VERTICAL = flags.DEFINE_bool(
    "flip_vertical", True,
    "The svgpathtools extracts something that is flipped vertically. "
    "Usually we want to flip it back."
)

_FIRST_POINT_IS_ORIGIN = flags.DEFINE_bool(
    "first_point_is_origin", False,
    "If true, make the first point be (0, 0)."
)

_DELTAS = flags.DEFINE_bool(
    "deltas", False,
    "If true, render not actual points but deltas."
)

_POINTS_PER_SEGMENT = flags.DEFINE_integer(
    "points_per_segment", 100,
    "Points per segment for stroke generation."
)

_APPLY_RDP = flags.DEFINE_bool(
    "apply_rdp", True,
    "Apply Ramer–Douglas–Peucker (RDP) algorithm for line simplification on "
    "each stroke."
)

_RDP_TOLERANCE = flags.DEFINE_float(
    "rdp_tolerance", 1e-5,
    "Tolerance for Ramer–Douglas–Peucker algorithm. Prune stroke points whose "
    "distance is beyond tolerance from the original. The larger the tolerance "
    "the smaller number of points are retained."
)

_PATH_IS_STROKE = flags.DEFINE_bool(
    "path_is_stroke", False,
    "Make a path a stroke rather than the individual segments. Has the "
    "potential to reduce point count if the Ramer-Douglas-Peucker algorithm "
    "is also used."
)

FLAGS = flags.FLAGS


def _distances_to_lines(
    points: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
  """Computes the distances from points to lines.

  The distances are computed from from N ``points`` to N lines given
  by the points `start` and `end`.

  Args:
    points: An array of points.
    start: An array containing the beginnings of lines.
    end: An array of line ends.

  Returns:
    An array of N distances.
  """
  def _cross2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A scalar `cross product` of two-dimensional vectors."""
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

  if np.all(start == end):
    return np.linalg.norm(points - start, axis=1)

  vec = end - start
  cross = _cross2d(vec, start - points)
  return np.divide(abs(cross), np.linalg.norm(vec))


def _simplify_rdp(points: np.ndarray, tolerance: float = 0.) -> np.ndarray:
  """Ramer–Douglas–Peucker (RDP) algorithm for line simplification.

  Takes an array of `points` of shape (N, 2) and removes excess points in the
  line. The remaining points form an identical line to within `tolerance` from
  the original. See:
  https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm

  From https://github.com/fhirschmann/rdp/issues/7
  originally written by Kirill Konevets https://github.com/kkonevets

  Args:
    points: Array of points with dimension (N, 2).
    tolerance: Algorithm's tolerance.

  Returns:
    Simplified array of points.
  """
  curve = np.asarray(points)
  start, end = curve[0], curve[-1]
  dists = _distances_to_lines(curve, start, end)

  index = np.argmax(dists)
  max_dist = dists[index]

  if max_dist > tolerance:
    result1 = _simplify_rdp(curve[: index + 1], tolerance)
    result2 = _simplify_rdp(curve[index:], tolerance)
    result = np.vstack((result1[:-1], result2))
  else:
    result = np.array([start, end])

  return result


def simplify_rdp(
    points: list[tuple[float, float]], tolerance: float = 0.
) -> list[tuple[float, float]]:
  """Ramer–Douglas–Peucker (RDP) algorithm for line simplification.

  Args:
    points: A lists of points representing an individual stroke.
    tolerance: Algorithm's tolerance.

  Returns:
    Simplified lists of points (simplified stroke).
  """
  points_array = np.array(points, dtype=np.float64)
  result = _simplify_rdp(points_array, tolerance=tolerance)
  result = [(float(x), float(y)) for x, y in result.tolist()]
  return result


def _flip_vertical(
    strokes: list[list[tuple[float, float]]]
) -> list[list[tuple[float, float]]]:
  """Flips sequence of strokes vertically."""
  max_y = np.finfo(np.float32).min
  for stroke in strokes:
    max_y = max(np.max([p[1] for p in stroke]), max_y)
  return [[(p[0], max_y - p[1]) for p in stroke] for stroke in strokes]


def _rotate(x: float, y: float, angle: float) -> tuple[float, float]:
  """Rotates a point by an angle."""
  sin, cos = trig.sin_cos(angle)
  return x * cos - y * sin, x * sin + y * cos


def _parse_transformation(
    attribute: dict[str, str], idx: int
) -> tuple[float, float, float, float, list[float], int]:
  """Parses SVG transform element given the attribute."""
  trans_x = 0.
  trans_y = 0.
  angle = 0
  scale = 1
  mat = []
  if "transform" in attribute:
    transform = [c.strip() for c in attribute["transform"].split(")")]
    for trans in transform:
      if trans.startswith("translate("):
        trans = trans.replace("translate(", "").split()
        trans_x = float(trans[0])
        trans_y = float(trans[1])
      elif trans.startswith("rotate("):
        trans = trans.replace("rotate(", "")
        angle = float(trans)
      elif trans.startswith("scale("):
        trans = trans.replace("scale(", "")
        scale = float(trans)
      elif trans.startswith("matrix("):
        # Something of the form `transform="matrix(1,0,0,1,599,266)"`
        trans = trans.replace("matrix(", "").replace("(", "").replace(")", "")
        trans = trans.split(",")
        mat = [float(v) for v in trans]
  return trans_x, trans_y, angle, scale, mat, idx


def _parse_position_and_glyph(
    attribute: dict[str, str]
) -> tuple[int, str | None]:
  """Parses glyph affiliations from the attribute."""
  if svg_simplify.XML_SVG_POSITION_AND_GLYPH in attribute:
    position, glyph = (
        attribute[svg_simplify.XML_SVG_POSITION_AND_GLYPH].split(",")
    )
    position = int(position)
    return position, glyph
  else:
    return -1, None


def svg_to_strokes(
    paths: list[Path],
    attributes: list[dict[str, str]],
    flip_vertical: bool = True,
    path_is_stroke: bool = False,
    apply_rdp: bool = True,
    rdp_tolerance: float = 1e-5,
    deltas: bool = False,
    first_point_is_origin: bool = False,
    points_per_segment: int = 100,
):
  """Converts from paths and attributes to strokes consisting of points.

  Args:
    paths: a set of paths extracted from the SVG.
    attributes: a list of attributes extracted from the SVG.
    flip_vertical: flip the strokes vertically.
    path_is_stroke: turn each path into a single stroke.
    apply_rdp: apply RDP segment simplification.
    rdp_tolerance: numeric tolerance for RDP algorithm.
    deltas: use deltas instead of absolute coordinates.
    first_point_is_origin: make the first point (0, 0).
    points_per_segment: How many points to assign to each path segment.

  Returns:
    A tuple consisting of
      - a list of lists of x-y points, and
      - a list of glyph affiliations associated with each stroke.
  """
  strokes = []
  glyph_affiliations = []

  # Get the translations and other attributes. Since svg2paths does
  # not extract these things in any sensible order, we need to reorder
  # things from left to right based on the x translations, otherwise
  # the pen in the sketch will be flying all over the place.
  transformations = []
  positions_and_glyphs = []
  for idx, attribute in enumerate(attributes):
    transformations.append(_parse_transformation(attribute, idx))
    positions_and_glyphs.append(_parse_position_and_glyph(attribute))

  transformations.sort()
  sorted_paths = []
  sorted_positions_and_glyphs = []
  for transformation in transformations:
    sorted_paths.append(paths[transformation[-1]])
    sorted_positions_and_glyphs.append(positions_and_glyphs[transformation[-1]])

  for path, transformation, position_and_glyph in zip(
      sorted_paths,
      transformations,
      sorted_positions_and_glyphs,
  ):
    trans_x, trans_y, angle, scale, mat, _ = transformation
    path_strokes = []
    for segment in path:
      stroke = []
      for t in range(points_per_segment + 1):
        point = segment.point(t / points_per_segment)
        x, y = float(point.real), float(point.imag)
        # See
        # https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/transform
        # for a description of what a transformation matrix represents.
        if mat:
          matrix = np.array(
              [[mat[0], mat[2], mat[4]], [mat[1], mat[3], mat[5]], [0, 0, 1]]
          )
          x, y, _ = np.matmul(matrix, [x, y, 1]).tolist()
        x, y = _rotate(x, y, angle)
        x, y = scale * x + trans_x, scale * y + trans_y
        stroke.append((float(x), float(y)))
      if path_is_stroke:
        path_strokes.extend(stroke)
      else:
        path_strokes.append(stroke)
    if path_is_stroke:
      path_strokes = [path_strokes]
    strokes.extend(path_strokes)
    position_and_glyph = [position_and_glyph] * len(path_strokes)
    glyph_affiliations.extend(position_and_glyph)

  if strokes and flip_vertical:
    strokes = _flip_vertical(strokes)

  if strokes and first_point_is_origin:
    first_x, first_y = strokes[0][0]
    new_strokes = []
    for stroke in strokes:
      new_stroke = []
      for x, y in stroke:
        new_stroke.append((x - first_x, y - first_y))
      new_strokes.append(new_stroke)
    strokes = new_strokes

  if apply_rdp:
    # Apply Ramer–Douglas–Peucker algorithm to each stroke. Make sure we do this
    # before we perform delta conversion below.
    strokes = [simplify_rdp(stroke, rdp_tolerance) for stroke in strokes]

  if strokes and deltas:
    # Compute relative instead of absolute point positions. The first point in
    # the first stroke is absolute, the rest are absolute. This is to be
    # compatible with various stroke-to-polylines conversion APIs in
    # protoscribe/sketches/utils/stroke_utils.py.
    prev_x, prev_y = 0., 0.
    new_strokes = []
    for stroke in strokes:
      new_stroke = []
      for x, y in stroke:
        new_stroke.append((x - prev_x, y - prev_y))
        prev_x, prev_y = x, y
      new_strokes.append(new_stroke)
    strokes = new_strokes

  return strokes, glyph_affiliations


def svg_file_to_strokes(svg_file: str):
  """Converts from SVG file to strokes consisting of points.

  Args:
    svg_file: a path to an SVG file

  Returns:
    A tuple consisting of
      - a list of lists of x-y points, and
      - a list of glyph affiliations associated with each stroke.
  """
  paths, attributes = svg2paths(svg_file)
  return svg_to_strokes(
      paths,
      attributes,
      flip_vertical=_FLIP_VERTICAL.value,
      path_is_stroke=_PATH_IS_STROKE.value,
      apply_rdp=_APPLY_RDP.value,
      rdp_tolerance=_RDP_TOLERANCE.value,
      deltas=_DELTAS.value,
      first_point_is_origin=_FIRST_POINT_IS_ORIGIN.value,
      points_per_segment=_POINTS_PER_SEGMENT.value
  )


def svg_tree_to_strokes(svg_tree: ET.ElementTree):
  """Converts from SVG tree to strokes consisting of points.

  Args:
    svg_tree: an SVG tree.

  Returns:
    A tuple consisting of
      - a list of lists of x-y points, and
      - a list of glyph affiliations associated with each stroke.
  """
  ET.register_namespace("", svg_simplify.XML_SVG_NAMESPACE)
  svg_str = ET.tostring(svg_tree.getroot()).decode("utf8")
  paths, attributes = svgstr2paths(svg_str)
  return svg_to_strokes(
      paths,
      attributes,
      flip_vertical=_FLIP_VERTICAL.value,
      path_is_stroke=_PATH_IS_STROKE.value,
      apply_rdp=_APPLY_RDP.value,
      rdp_tolerance=_RDP_TOLERANCE.value,
      deltas=_DELTAS.value,
      first_point_is_origin=_FIRST_POINT_IS_ORIGIN.value,
      points_per_segment=_POINTS_PER_SEGMENT.value
  )


def svg_tree_to_strokes_for_test(
    svg_tree: ET.ElementTree,
    flip_vertical: bool = True,
    path_is_stroke: bool = False,
    apply_rdp: bool = True,
    rdp_tolerance: float = 1e-5,
    deltas: bool = False,
    first_point_is_origin: bool = False,
    points_per_segment: int = 100
):
  """Converts from SVG tree to strokes consisting of points.

  Note: This is a version intended for the tests only. The other
  two APIs above give us a more controlled access using command-line flags.

  Args:
    svg_tree: an SVG tree.
    flip_vertical: flip the strokes vertically.
    path_is_stroke: turn each path into a single stroke.
    apply_rdp: apply RDP segment simplification.
    rdp_tolerance: numeric tolerance for RDP algorithm.
    deltas: use deltas instead of absolute coordinates.
    first_point_is_origin: make the first point (0, 0).
    points_per_segment: How many points to assign to each path segment.

  Returns:
    A tuple consisting of
      - a list of lists of x-y points, and
      - a list of glyph affiliations associated with each stroke.
  """
  ET.register_namespace("", svg_simplify.XML_SVG_NAMESPACE)
  svg_str = ET.tostring(svg_tree.getroot()).decode("utf8")
  paths, attributes = svgstr2paths(svg_str)
  return svg_to_strokes(
      paths,
      attributes,
      flip_vertical=flip_vertical,
      path_is_stroke=path_is_stroke,
      apply_rdp=apply_rdp,
      rdp_tolerance=rdp_tolerance,
      deltas=deltas,
      first_point_is_origin=first_point_is_origin,
      points_per_segment=points_per_segment
  )


def print_text(
    strokes: list[list[tuple[float, float]]], stream=sys.stdout
) -> None:
  """Utility to print a text as a set of points.

  Args:
    strokes: output of svg_to_strokes.
    stream: output stream.
  """
  for stroke in strokes:
    for x, y in stroke:
      stream.write(f"{x} {y}\n")
