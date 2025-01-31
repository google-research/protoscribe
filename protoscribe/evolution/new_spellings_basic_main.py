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

"""Extends spellings to new terms based on semantics and/or phonetics.

Uses the confidences from the semantic and phonetic runs to decide whether to
extend a spelling to a new concept/word based on the confidences with which the
model projects those glyph spellings during inference.

The algorithm was described in Richard Sproat (2023) "Symbols: An Evolutionary
History from the Stone Age to the Future.", Heidelberg, Springer, 2023.
See also: https://github.com/rwsproat/symbols/tree/main
"""

import logging
import os

from absl import app
from absl import flags
from protoscribe.evolution import new_spellings_basic as basic_lib
from protoscribe.evolution import new_spellings_utils as utils

import glob
import os

_DATA_LOCATION = flags.DEFINE_string(
    "data_location", None,
    "Path to base data location.",
    required=True,
)

_ADMINISTRATIVE_CATEGORIES = flags.DEFINE_string(
    "administrative_categories", None,
    "Path to current `administrative categories`.",
    required=True,
)

_NON_ADMINISTRATIVE_CATEGORIES = flags.DEFINE_string(
    "non_administrative_categories", None,
    "Path to current `non-administrative categories`.",
    required=True,
)

_PREVIOUS_SPELLINGS = flags.DEFINE_string(
    "previous_spellings", None,
    "Spellings from a previous round, if any, to be integrated into "
    "current spellings.",
)

_OUTPUT_GLYPH_GRAPHICS_DIR = flags.DEFINE_string(
    "output_glyph_graphics_dir", None,
    "Directory where to store the glyph extensions for new concepts in SVG "
    "format."
)


def main(unused_argv):
  # Compute extensions.
  glyphs = basic_lib.glyph_extensions()

  # Create a summary of all the extensions in this round.
  output_dir = os.path.join(_DATA_LOCATION.value, "inference_extensions")
  if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
  extensions_path = os.path.join(output_dir, "extensions.tsv")
  logging.info("Writing extensions to %s ...", extensions_path)
  with open(extensions_path, "w") as s:
    s.write("Concept\tGlyph(s)\tConcept pron\tPhonetic extension glyph pron\n")
    for concept, glyph, concept_pron, glyph_pron, _ in glyphs:
      s.write(f"{concept}\t{glyph}\t{concept_pron}\t{glyph_pron}\n")

  # Update the global list of spellings with the spellings from this round and
  # update the lists of seen and unseen concepts.
  spellings = []
  if _PREVIOUS_SPELLINGS.value:
    logging.info(
        "Reading previous round spellings from %s ...",
        _PREVIOUS_SPELLINGS.value
    )
    with open(_PREVIOUS_SPELLINGS.value) as s:
      spellings = [l.strip() for l in s.readlines()]
  for concept, glyph, _, _, _ in glyphs:
    glyph = " ".join([f"{g}_GLYPH" for g in glyph.split()])
    spellings.append(f"{concept}\t{glyph}")
  spellings.sort()
  spellings_path = os.path.join(output_dir, "spellings.tsv")
  logging.info("Writing spellings to %s ...", spellings_path)
  with open(spellings_path, "w") as s:
    for line in spellings:
      s.write(f"{line}\n")
  new_spelling_categories = set([c[0] for c in glyphs])
  utils.rejigger_categories(
      new_spelling_categories,
      _ADMINISTRATIVE_CATEGORIES.value,
      _NON_ADMINISTRATIVE_CATEGORIES.value,
      output_dir,
  )

  if _OUTPUT_GLYPH_GRAPHICS_DIR.value:
    utils.save_glyph_graphics(glyphs, _OUTPUT_GLYPH_GRAPHICS_DIR.value)


if __name__ == "__main__":
  app.run(main)
