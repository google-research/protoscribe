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

"""Constructs an HTML page to view the proposed spellings with glyph images."""

import csv
import logging
import os

from absl import flags
from protoscribe.utils import file_utils

import glob
import os

_EXTENSIONS_FILE = flags.DEFINE_string(
    "extensions_file", "",
    "Path to the text file in TSV format containing glyph extensions.",
)

_SVG_SRC_DIR = flags.DEFINE_string(
    "svg_src_dir", "",
    "Directory containing source graphics in SVG format for the glyph "
    "extensions.",
)

_OUTPUT_HTML_DIR = flags.DEFINE_string(
    "output_html_dir", None,
    "Path to output directory for `index.html`.",
    required=True
)

_IMAGE_HEIGHT = flags.DEFINE_integer(
    "image_height", 60,
    "Height of the glyph images (in pixels)."
)


# HTML header.
_HTML_HEADER = """<!DOCTYPE html>
<html>
<head>
<meta charset=utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Glyph Extensions</title>
</head>
<body>"""

# HTML decorations wrapping each glyph.
_HTML_FRAGMENT = """
<hr>
@LINE@
<p>
<img src="@IMG_PATH@" height=@HEIGHT@>
</p>
"""

# HTML footer.
_HTML_FOOTER = "</body></html>"


def _compose_page(
    extensions: str, output_html_dir: str
) -> set[str]:
  """Composes the HTML page.

  Args:
    extensions: Path to extensions TSV file.
    output_html_dir: Path to output directory for `index.html`.

  Returns:
    Set of concepts for which glyph extensions have been proposed.
  """
  index = os.path.join(output_html_dir, "index.html")
  concepts = set()
  logging.info("Composing page `%s` ...", index)
  with open(index, "wt") as out_f:
    out_f.write(f"{_HTML_HEADER}\n")
    with open(extensions, "rt") as ext_f:
      reader = csv.reader(ext_f, delimiter="\t", quotechar='"')
      rows = [row for row in reader]
      for row in rows[1:]:
        concept, glyphs, concept_pron, phon_glyph_pron = row
        concept = concept.split("_")[0]
        concepts.add(concept)
        svg_path = f"svgs/{concept}.svg"
        if concept_pron == "NA":
          line = f"{concept} = {glyphs}"
        else:
          line = (
              f"{concept} (/{concept_pron}/) = {glyphs} (/{phon_glyph_pron}/)"
          )
        fragment = (
            _HTML_FRAGMENT.replace("@LINE@", line)
            .replace("@HEIGHT@", str(_IMAGE_HEIGHT.value))
            .replace(
                "@IMG_PATH@",
                svg_path,
            )
            .strip()
        )
        out_f.write(f"{fragment}\n")
    out_f.write(f"{_HTML_FOOTER}\n")

  return concepts


def _copy_svgs(concepts: set[str], source_dir: str, target_dir: str) -> None:
  """Prepares SVG glyph directory in the requested location.

  Args:
    concepts: A set of concepts to be copied.
    source_dir: Source directory.
    target_dir: Target directory.

  Raises:
    FileNotFoundError: when source file is not found.
  """
  paths = []
  for concept in concepts:
    filename = f"{concept}.svg"
    source_path = os.path.join(source_dir, filename)
    if not os.path.exists(source_path):
      raise FileNotFoundError(f"Source SVG {source_path} not found")
    paths.append(source_path)

  file_utils.copy_files(paths, target_dir)


def make_html() -> None:
  """Creates the HTML page with contents."""

  # Create the output graphics directory if it doesn't exist.
  output_svg_dir = os.path.join(_OUTPUT_HTML_DIR.value, "svgs")
  if not os.path.exists(output_svg_dir):
    logging.info("Making directory %s ...", output_svg_dir)
    os.makedirs(
        output_svg_dir, exist_ok=True
    )

  # Create index page.
  if not _EXTENSIONS_FILE.value:
    raise ValueError("Specify --extensions_file!")
  concepts = _compose_page(_EXTENSIONS_FILE.value, _OUTPUT_HTML_DIR.value)

  # Copy graphics.
  if not _SVG_SRC_DIR.value:
    raise ValueError("Specify --svg_src_dir!")
  logging.info("Copying %d SVGs from %s ...", len(concepts), _SVG_SRC_DIR.value)
  _copy_svgs(concepts, _SVG_SRC_DIR.value, output_svg_dir)
