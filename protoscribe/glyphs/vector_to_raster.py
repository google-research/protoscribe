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

"""Miscellaneous utilities for conversion to raster image formats."""

import os
import random
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from absl import flags
import cairo
import numpy as np
import PIL.Image
import PIL.ImageColor
from protoscribe.glyphs import svg_simplify

_RANDOM_LINE_WIDTH = flags.DEFINE_bool(
    "random_line_width", False,
    "Randomly set line width to be between 1 and `--stroke_width`.")

_RANDOM_SCALE = flags.DEFINE_bool(
    "random_scale", False,
    "Randomly resize between 0.75 and 1.")

_ALPHA_CHANNEL = flags.DEFINE_boolean(
    "alpha_channel", False,
    "Keep alpha channel in the output or remove it.")


def _line_width() -> int:
  """Returns line width in pixels."""
  stroke_width = int(svg_simplify.STROKE_WIDTH.value)
  line_width = (
      random.randint(1, stroke_width)
      if _RANDOM_LINE_WIDTH.value else stroke_width
  )
  return line_width


def _cairo_bgra_rgba(surface: cairo.ImageSurface) -> PIL.Image.Image:
  """Converts a Cairo surface from default BGRA format to a RBGA string."""
  image = PIL.Image.frombuffer(
      "RGBA",
      (surface.get_width(), surface.get_height()),
      surface.get_data().tobytes(), "raw", "BGRA", 0, 1
  )
  return image


def _scale_image(
    input_image: PIL.Image.Image, padding: float
) -> PIL.Image.Image:
  """Scales the input image using random factor."""
  input_width, input_height = input_image.size
  image = PIL.Image.new(
      input_image.mode,
      (input_width, input_height),
      PIL.ImageColor.getrgb("white")
  )
  random_scale = 0.5 + 0.5 * random.random()
  new_width = int(input_width * random_scale)
  new_height = int(input_height * random_scale)
  scaled_image = input_image.resize(
      (new_width, new_height), PIL.Image.Resampling.BICUBIC
  )
  image.paste(scaled_image, box=(int(padding / 2), int(padding / 2)))
  return image


# TODO: Experiment with using random brush strokes instead of using
# a fixed value.
def stroke_points_to_raster(
    stroke_points: list[tuple[list[float], list[float]]],
    source_side: int = 256,
    target_side: int = 28,
    padding: float = 16.,
    bg_color: tuple[int, int, int, int] = (1, 1, 1, 0),  # Transparent white.
    fg_color: tuple[int, int, int, int] = (0, 0, 0, 1)   # Opaque black.
) -> PIL.Image.Image:
  """Converts a list of SVGs to corresponding bitmap (rasterized) format."""
  # Padding and line width are relative to the default original 256x256
  # image.
  surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, target_side, target_side)
  ctx = cairo.Context(surface)
  ctx.set_antialias(cairo.ANTIALIAS_BEST)
  ctx.set_line_cap(cairo.LINE_CAP_ROUND)
  ctx.set_line_join(cairo.LINE_JOIN_ROUND)
  line_width = _line_width()
  ctx.set_line_width(line_width)  # Width of the stroke.

  # scale to match the new size
  # add padding at the edges for the line width
  # and add additional padding to account for antialiasing
  total_padding = padding * 2. + line_width
  new_scale = float(target_side) / float(source_side + total_padding)
  ctx.scale(new_scale, new_scale)
  ctx.translate(total_padding / 2., total_padding / 2.)

  # Clear background.
  if _ALPHA_CHANNEL.value:
    ctx.set_source_rgba(*bg_color)
  else:
    ctx.set_source_rgb(bg_color[0], bg_color[1], bg_color[2])
  ctx.set_operator(cairo.OPERATOR_SOURCE)
  ctx.paint()
  ctx.set_operator(cairo.OPERATOR_OVER)

  bbox = np.hstack(stroke_points).max(axis=1)
  offset = ((source_side, source_side) - bbox) / 2.
  offset = offset.reshape(-1, 1)
  centered = [stroke + offset for stroke in stroke_points]

  # Draw strokes, this is the most cpu-intensive part.
  if _ALPHA_CHANNEL.value:
    ctx.set_source_rgba(*fg_color)
  else:
    ctx.set_source_rgb(fg_color[0], fg_color[1], fg_color[2])
  ctx.set_operator(cairo.OPERATOR_SOURCE)
  for xv, yv in centered:
    ctx.move_to(xv[0], yv[0])
    for x, y in zip(xv, yv):
      ctx.line_to(x, y)
    ctx.stroke()
  image = _cairo_bgra_rgba(surface)

  # Scale if necessary.
  if _RANDOM_SCALE.value:
    image = _scale_image(image, total_padding)
  return image


def vector_to_raster(
    svg: ET.ElementTree,
    output_file: str,
    output_width: int,
    output_height: int,
    output_background: str = "ivory"
) -> None:
  """Calls inkscape to rasterize the SVG.

  Args:
    svg: An SVG tree.
    output_file: Path to raster output (in PNG format).
    output_width: width of an image.
    output_height: height of an image.
    output_background: optional background color.
  """
  with tempfile.NamedTemporaryFile(
      suffix=".svg", prefix=os.path.basename(__file__)
  ) as f:
    svg.write(f.name)
    cmd = (
        f"/usr/bin/inkscape -z "
        f"--export-filename={output_file} "
        f"--export-width={output_width} "
        f"--export-height={output_height} "
        f"--export-background=\"{output_background}\" {f.name}"
    )
    subprocess.call(cmd, shell=True)
