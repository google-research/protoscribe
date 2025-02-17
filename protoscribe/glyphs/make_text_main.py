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

r"""Constructs an SVG `text` out of a set of SVGs of glyphs.

Allows for random rotations and scaling of glyphs.

Example:
--------
python protoscribe/glyphs/make_text_main.py \
  --concepts "X,X,I,I,I,apple" \
  --output_svg_file /tmp/test.svg \
  --output_raster_file /tmp/test.png \
  --logtostderr
"""

import logging

from absl import app
from absl import flags
from protoscribe.glyphs import glyph_vocab as glyph_lib
from protoscribe.glyphs import make_text
from protoscribe.glyphs import vector_to_raster as raster_lib

_CONCEPTS = flags.DEFINE_list(
    "concepts", None,
    "List of concept names, e.g., `I,dog`.",
    required=True
)

_OUTPUT_RASTER_FILE = flags.DEFINE_string(
    "output_raster_file", None,
    "Output raster image in PNG format.",
    required=True
)

_OUTPUT_SVG_FILE = flags.DEFINE_string(
    "output_svg_file", None,
    "SVG output file."
)

_OUTPUT_SVG_FOR_STROKES_FILE = flags.DEFINE_string(
    "output_svg_for_strokes_file", None,
    "SVG-for-strokes output file."
)

_BACKGROUND_COLOR = flags.DEFINE_string(
    "background_color", "ivory",
    "Background color for rastered image."
)

_SCALE = flags.DEFINE_float(
    "scale", 2.5,
    "Scaling info for raster image converter (Inkscape)."
)

FLAGS = flags.FLAGS


def main(unused_argv):
  # Prepare the requested glyphs and compose them.
  svgs = []
  glyph_vocab = glyph_lib.load_or_build_glyph_vocab()
  for concept in _CONCEPTS.value:
    path_and_id = glyph_vocab.find_svg_from_name(concept)
    if path_and_id:
      svgs.append(path_and_id[0])
  if not svgs:
    raise ValueError("No glyphs found!")

  logging.info("Composing %d glyphs ...", len(svgs))
  concat, concat_for_strokes, width, height = make_text.concat_svgs(
      svgs, glyphs=_CONCEPTS.value,
  )

  # Save vector graphics format.
  if _OUTPUT_SVG_FILE.value:
    logging.info("Saving SVG to %s ...", _OUTPUT_SVG_FILE.value)
    concat.write(_OUTPUT_SVG_FILE.value)

  if _OUTPUT_SVG_FOR_STROKES_FILE.value:
    logging.info(
        "Saving SVG-for-strokes to %s ...", _OUTPUT_SVG_FOR_STROKES_FILE.value
    )
    concat_for_strokes.write(_OUTPUT_SVG_FOR_STROKES_FILE.value)

  # Convert and save to raster format.
  def scale(x: float) -> int:
    return int(x * _SCALE.value)

  logging.info(
      "Converting SVG to raster image in %s ...", _OUTPUT_RASTER_FILE.value
  )
  raster_lib.vector_to_raster(
      concat,
      output_file=_OUTPUT_RASTER_FILE.value,
      output_width=scale(width),
      output_height=scale(height),
      output_background=_BACKGROUND_COLOR.value
  )


if __name__ == "__main__":
  app.run(main)
