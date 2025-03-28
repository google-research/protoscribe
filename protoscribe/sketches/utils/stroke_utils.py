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

"""Miscellaneous utilities for stroke manipulation."""

from absl import flags
import chex
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
import PIL.Image
from protoscribe.glyphs import svg_simplify as svg_simplify_lib
from protoscribe.sketches.utils import stroke_stats as stroke_stats_lib

import glob
import os

Array = np.ndarray | jnp.ndarray
StrokeStats = stroke_stats_lib.StrokeStats

_MAX_STROKE_POINTS = flags.DEFINE_integer(
    "max_stroke_points", 1_000,
    "Maximum number of stroke points which are used for computing the "
    "statistics for stroke normalization."
)

# End of stroke value.
END_OF_STROKE = 1_000_000


def stroke3_to_stroke5(strokes_3: Array, max_len: int) -> tuple[Array, int]:
  """Converts to stroke-5 bigger format as described in paper and pads.

  A sketch is a list of points, and each point is a vector consisting of 5
  elements: (∆x, ∆y, p1, p2, p3).  The first two elements are the offset
  distance in the x and y directions of the pen from the previous point. The
  last 3 elements represents a binary one-hot vector of 3 possible states. The
  first pen state, p1, indicates that the pen is currently touching the paper,
  and that a line will be drawn connecting the next point with the current
  point. The second pen state, p2, indicates that the pen will be lifted from
  the paper after the current point, and that no line will be drawn next. The
  final pen state, p3, indicates that the drawing has ended, and subsequent
  points, including the current point, will not be rendered.

  Args:
    strokes_3: Array in stroke-3 format.
    max_len: Maximum length of a sequence.

  Returns:
    A tuple consisting of sketch in stroke-5 format and its length.
  """
  chex.assert_rank(strokes_3, 2)
  chex.assert_shape(strokes_3, (None, 3))

  length = strokes_3.shape[0]
  if length + 1 > max_len:
    raise ValueError(f"The length of stroke ({length}) (plus BOS) "
                     f"exceeds the maximum allowed length ({max_len}).")
  result = np.zeros((max_len, 5), dtype=np.float32)
  result[0, 2] = 1  # BOS.
  result[1:length + 1, :2] = strokes_3[:, :2]  # ∆x, ∆y.
  result[1:length + 1, 2] = 1 - strokes_3[:, 2]  # $p_1$ (touching).
  result[1:length + 1, 3] = strokes_3[:, 2]  # $p_2$ (lifted).
  result[length + 1:, 4] = 1  # $p_3$ (drawing ended).
  return result, length  # Don't include BOS.


def stroke5_to_stroke3(strokes5: Array) -> Array:
  """Converts from stroke-5 format back to stroke-3.

  The conversion below assumes that the input strokes contain BOS and EOS
  vectors, which are omitted from the conversion. The stroke-3 format does
  not require them.

  Args:
    strokes5: Array of strokes in strokes-5 format.

  Returns:
    Array of strokes in stroke-3 format.
  """

  chex.assert_rank(strokes5, 2)
  chex.assert_shape(strokes5, (None, 5))

  if strokes5[0, :].tolist() == [0., 0., 1., 0., 0.]:
    strokes5 = strokes5[1:, :]  # Strip BOS.

  length = 0
  for i in range(strokes5.shape[0]):
    if strokes5[i, 4] > 0:
      length = i
      break
  if length == 0:
    length = strokes5.shape[0]
  result = np.zeros((length, 3), dtype=np.float32)
  result[:, 0:2] = strokes5[0:length, 0:2]
  result[:, 2] = strokes5[0:length, 3]
  return result


def stroke3_deltas_to_polylines(strokes: Array) -> list[Array]:
  """Converts stroke-3 delta format to lines.

  Each sample is stored as list of coordinate offsets: ∆x, ∆y, and a binary
  value $p$ representing whether the pen is lifted away from the paper.
  This format, referred to as stroke-3, is described in:
     Alex Graves (2013). `Generating Sequences With Recurrent Neural Networks`
     https://arxiv.org/abs/1308.0850

  Args:
    strokes: Array in stroke-3 delta format.

  Returns:
    List of drawable points converted from deltas.
  """
  chex.assert_rank(strokes, 2)
  chex.assert_shape(strokes, (None, 3))

  x = 0.
  y = 0.
  lines = []
  line = []
  for i in range(strokes.shape[0]):
    if int(strokes[i, 2]) == 1:  # Pen lifted.
      x += strokes[i, 0]
      y += strokes[i, 1]
      line.append([x, y])
      lines.append(np.array(line, dtype=np.float32))
      line = []
    else:  # Continuing the stroke.
      x += strokes[i, 0]
      y += strokes[i, 1]
      line.append([x, y])
  return lines


def stroke5_deltas_to_polylines(strokes: Array) -> list[Array]:
  """Converts strokes in stroke-5 format to lines."""
  chex.assert_rank(strokes, 2)
  chex.assert_shape(strokes, (None, 5))

  # Cumulative sums of $(\Delta x, \Delta y)$ to get $(x, y)$.
  coords = np.cumsum(strokes[:, 0:2], axis=0)  # (N, 2).
  # Create a new numpy array of the form $(x, y, q_2)$.
  seq = np.zeros((coords.shape[0], 3), dtype=np.float32)
  seq[:, 0:2] = coords
  seq[:, 2] = strokes[:, 3]  # Pen is lifted.

  # Split the array at points where $q_2$ is $1$, where the pen is lifted
  # from the paper. This yields a list of sequences of strokes.
  pen_up_indices = np.where(seq[:, 2] > 0.)[0]
  strokes_list = np.split(seq, pen_up_indices + 1)
  return strokes_list


def polylines_to_raster_image(polylines: list[Array]) -> PIL.Image.Image:
  """Generates raster image from polylines."""
  plt.clf()
  for lines in polylines:
    plt.plot(lines[:, 0], -lines[:, 1])
    # Don't show axes.
    plt.axis("off")

  # Convert to an image.
  canvas = plt.get_current_fig_manager().canvas
  canvas.draw()
  return PIL.Image.frombytes(
      "RGB", canvas.get_width_height(), canvas.tostring_rgb()
  )


def stroke3_strokes_to_svg(
    strokes_3: Array, scale_factor: float = 1.0
) -> str:
  """Converts stroke-3 format to an SVG string buffer.

  Args:
    strokes_3: Array representing strokes in stroke-3 format (point offsets).
    scale_factor: Ratio of the transformed dimension to the original (a floating
      point number between).

  Returns:
    XML string representation of an SVG.
  """

  chex.assert_rank(strokes_3, 2)
  chex.assert_shape(strokes_3, (None, 3))
  polylines = stroke3_deltas_to_polylines(strokes_3)

  all_x = []
  all_y = []
  for lines in polylines:
    all_x.extend(lines[:, 0].tolist())
    all_y.extend(lines[:, 1].tolist())
  x_min, x_max = np.min(all_x), np.max(all_x)
  y_min, y_max = np.min(all_y), np.max(all_y)

  offset = 10.  # Margins for the sketch.
  width = x_max - x_min + 2 * offset
  height = y_max - y_min + 2 * offset
  buf = (
      svg_simplify_lib.basic_svg_xml_header(width, height) +
      f"<g transform=\"translate({-x_min} {-y_min})"
  )
  if scale_factor != 1.0:
    buf += f" scale({scale_factor})"
  buf += "\">\n"
  for lines in polylines:
    if lines.shape[0] == 1:
      # Disallow empty lines that only have an origin point. This can happen
      # to decoder outputs if the model overpredicts the stroke separator
      # tokens.
      continue
    path_buf = (
        f"<path d=\"M{lines[0][0]} {lines[0][1]}"
    )
    for i in range(1, lines.shape[0]):
      path_buf += f" L{lines[i][0]} {lines[i][1]}"
    path_buf += (
        "\" fill=\"none\" stroke=\"#000000\" stroke-width=\"%f\" />\n" % (
            svg_simplify_lib.STROKE_WIDTH.value
        )
    )
    buf += path_buf
  buf += "</g></svg>\n"

  # Return simplified representation where all the elements are the paths and
  # the paths are flattened (no transforms).
  xml_str, _, _ = svg_simplify_lib.simplify_svg_str(buf)
  return xml_str


def stroke3_strokes_to_svg_file(
    sketch: Array, svg_filename: str, scale_factor: float = 1.0
) -> None:
  """Converts stroke-3 format to an SVG and save."""
  svg = stroke3_strokes_to_svg(sketch, scale_factor=scale_factor)
  with open(svg_filename, mode="wt") as f:
    f.write(svg)


def stroke_points(
    strokes: list[list[tuple[float, float]]],
    stroke_glyph_affiliations: list[tuple[int, int]]
) -> tuple[
    list[float], list[float], StrokeStats, list[int], list[int]
]:
  """Converts strokes to separate coordinate points.

  Constructs the x and y stroke vectors, interspersing each stroke
  with an end-of-stroke token. Record glyph affiliation for each
  resulting points as well. This API is used at dataset building time.

  Args:
    strokes: Collection of strokes.
    stroke_glyph_affiliations: List of tuples recording the glyph
      position in text and its identity that this stroke belongs too.

  Returns:
    A tuple consisting of flat lists of x and y stroke coordinates,
    accumulated statistics and glyph affiliations for each coordinate.
  """
  if len(strokes) != len(stroke_glyph_affiliations):
    raise ValueError(
        "The lengths of strokes and glyph affiliations must match!"
    )
  x_stroke_points = []
  y_stroke_points = []
  glyph_affiliations_text_pos = []
  glyph_affiliations_ids = []
  num_points = 0
  stats = stroke_stats_lib.StrokeStats()
  for i, stroke in enumerate(strokes):
    for x, y in stroke:
      x_stroke_points.append(x)
      y_stroke_points.append(y)
      glyph_affiliations_text_pos.append(stroke_glyph_affiliations[i][0])
      glyph_affiliations_ids.append(stroke_glyph_affiliations[i][1])
      if num_points < _MAX_STROKE_POINTS.value:
        stats.update_point(x, y)
      num_points += 1
    x_stroke_points.append(END_OF_STROKE)
    y_stroke_points.append(END_OF_STROKE)
    glyph_affiliations_ids.append(END_OF_STROKE)
    glyph_affiliations_text_pos.append(END_OF_STROKE)

  stats.max_sample_size = num_points
  return (
      x_stroke_points, y_stroke_points, stats,
      glyph_affiliations_text_pos, glyph_affiliations_ids
  )


def stroke3_from_strokes(
    strokes: list[list[tuple[float, float]]],
    convert_to_deltas: bool
) -> Array:
  """Converts stroke points to stroke-3 (∆x, ∆y, p) format.

  Takes the stroke point format generated by `svg_to_strokes` library, i.e., a
  sketch is represented as sequence of strokes, where a stroke is a sequence of
  points.

  Args:
    strokes: A sequence of strokes consisting of individual points.
    convert_to_deltas: If enabled, the original `strokes` are assumed to contain
      absolute coordinates which are then converted to deltas.

  Returns:
    Stroke representation in stroke-3 format.

  Raises:
    ValueError if empty strokes are passed in.
  """
  if not strokes or not strokes[0]:
    raise ValueError("Stroke sequence is empty!")
  flat_points = []
  prev_x, prev_y = 0, 0
  for stroke in strokes:
    for pos, (x, y) in enumerate(stroke):
      lift_pen = pos == len(stroke) - 1
      cur_x, cur_y = (x - prev_x, y - prev_y) if convert_to_deltas else (x, y)
      flat_points.append((cur_x, cur_y, lift_pen))
      prev_x, prev_y = x, y
  return np.array(flat_points, dtype=np.float32)


def points_to_strokes3(x_strokes: Array, y_strokes: Array) -> Array:
  """Generates stroke-3 array from individual point arrays.

  One dimensional arrays of x and y strokes. The lift points are indicated
  by `END_OF_STROKE`.

  Args:
    x_strokes: Array of x points for strokes of shape (L,).
    y_strokes: Array of y points for strokes of shape (L,).

  Returns:
    Array for strokes in strokes-3 format.
  """
  pen_state = x_strokes == END_OF_STROKE
  pen_state = pen_state.astype(np.float32)
  pen_state = np.roll(pen_state, -1, axis=0)[:-1]
  pen_state = np.concatenate([pen_state, [1.]], axis=0)
  indices = np.reshape(np.where(x_strokes != END_OF_STROKE), (-1,))
  x_strokes = np.take(x_strokes, indices)
  y_strokes = np.take(y_strokes, indices)
  pen_state = np.take(pen_state, indices)
  strokes_3 = np.stack([x_strokes, y_strokes, pen_state], axis=1)
  return strokes_3
