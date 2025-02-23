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
import os

from absl import app
from absl import flags
from protoscribe.glyphs import glyph_vocab as glyph_lib
from protoscribe.glyphs import make_text
from protoscribe.glyphs import svg_to_strokes
from protoscribe.glyphs import vector_to_raster as raster_lib
from protoscribe.sketches.utils import stroke_utils

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

_BACKGROUND_COLOR = flags.DEFINE_string(
    "background_color", "ivory",
    "Background color for rastered image."
)

_SCALE = flags.DEFINE_float(
    "scale", 2.5,
    "Scaling info for raster image converter (Inkscape)."
)

_NUM_VARIANTS = flags.DEFINE_integer(
    "num_variants", 1,
    "Number of documents to generate. Useful for debugging when random "
    "transformations are enabled."
)

FLAGS = flags.FLAGS


def _doc_path(file_path: str, doc_id: int) -> str:
  """Adds document ID to the file path."""
  file_name = os.path.basename(file_path)
  return os.path.join(os.path.dirname(file_path), f"{doc_id}_{file_name}")


def _make_document(svgs: list[str], doc_id: int) -> None:
  """Generates a single document."""
  logging.info("[%d] Composing %d glyphs ...", doc_id, len(svgs))
  concat_svg, width, height = make_text.concat_svgs(
      svgs, glyphs=_CONCEPTS.value
  )

  # Save vector graphics format.
  if _OUTPUT_SVG_FILE.value:
    svg_path = _OUTPUT_SVG_FILE.value
    if _NUM_VARIANTS.value > 1:
      svg_path = _doc_path(svg_path, doc_id)

    logging.info("[%d] Saving SVG to %s ...", doc_id, svg_path)
    concat_svg.write(svg_path)

    # Convert to SVG to strokes and then back again.
    strokes, _ = svg_to_strokes.svg_tree_to_strokes(concat_svg)
    strokes3 = stroke_utils.stroke3_from_strokes(
        strokes, convert_to_deltas=not FLAGS.deltas
    )
    svg_path = svg_path.replace(".svg", "_strokes.svg")
    stroke_utils.stroke3_strokes_to_svg_file(strokes3, svg_path)

  # Convert and save to raster format.
  def scale(x: float) -> int:
    return int(x * _SCALE.value)

  raster_path = _OUTPUT_RASTER_FILE.value
  if _NUM_VARIANTS.value > 1:
    raster_path = _doc_path(raster_path, doc_id)
  logging.info(
      "[%d] Converting SVG to raster image in %s ...", doc_id, raster_path
  )
  raster_lib.vector_to_raster(
      concat_svg,
      output_file=raster_path,
      output_width=scale(width),
      output_height=scale(height),
      output_background=_BACKGROUND_COLOR.value
  )


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

  # Generate the requested number of documents.
  for doc_id in range(_NUM_VARIANTS.value):
    _make_document(svgs, doc_id)


if __name__ == "__main__":
  app.run(main)
