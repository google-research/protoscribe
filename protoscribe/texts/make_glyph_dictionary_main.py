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

r"""Creates table of concepts/phonetics -> SVG, to make for easy eyeballing.

For example:

DATABASE_DIR=...
python protoscribe/text_generation:make_glyph_dictionary_main.py \
  --lexicon ${DATABASE_DIR}/language/lexicon.tsv \
  --html example.html
"""

import logging
import os
from typing import List, Tuple

from absl import app
from absl import flags
from protoscribe.glyphs import glyph_vocab as glyph_lib

import glob
import os

_LEXICON = flags.DEFINE_string(
    "lexicon", None,
    "Path to the input lexicon in two-column TSV format consisting of "
    "concept name followed by pronunciation.",
    required=True
)

_HTML = flags.DEFINE_string(
    "html", None,
    "Path to output HTML page.",
    required=True
)

_COLS_PER_ROW = flags.DEFINE_integer(
    "cols_per_row", 10,
    "Number of columns per row."
)


def create_svg_table() -> List[Tuple[str, str, str]]:
  """Creates table of SVGs for seen categories.

  Returns:
    List of tuples of concept, phonetic form and SVG path.
  """
  glyph_vocab = glyph_lib.load_or_build_glyph_vocab()
  table = []
  logging.info("Reading lexicon in %s ...", _LEXICON.value)
  with open(_LEXICON.value) as stream:
    for line in stream:
      concept, phon = line.strip().split("\t")
      glyph_and_id = glyph_vocab.find_svg_from_name(concept)
      if not glyph_and_id:
        continue
      table.append((concept, phon, glyph_and_id[0]))
  return table


_HDR = '<html>\n<body>\n\n<table border="2">\n'
_COL = "<TD>@@</TD>"
_ROW = "<TR>@@</TR>\n"
_FTR = "</table>\n\n</body>\n</html>\n"
_URL_BASE = (
    "protoscribe/data/glyphs/generic/administrative_categories"
)


def main(unused_argv):
  table = create_svg_table()
  with open(_HTML.value, "w") as stream:
    stream.write(_HDR)
    lexicon_path = str(_LEXICON.value)
    stream.write(
        f'<FONT SIZE="+1">LEXICON: <B>{lexicon_path}</B></FONT>\n<P/>\n'
    )
    column_entries = []
    for concept, phon, glyph in table:
      glyph = glyph.split("/")[-1]
      glyph_path = os.path.join(_URL_BASE, glyph)
      img = f'<IMG SRC="{glyph_path}" WIDTH="150">'
      img = f'<A HREF="{glyph_path}" TARGET="_blank">{img}</A>'
      img = f"<CENTER>{img}</CENTER>"
      entry = (
          f'<FONT COLOR="blue"><B>{concept}</B>&nbsp;&nbsp;/{phon}/'
          f"<BR/><BR/>{img}<FONT>"
      )
      column_entries.append(_COL.replace("@@", entry))
    left = 0
    right = _COLS_PER_ROW.value
    while True:
      row = column_entries[left:right]
      if not row:
        break
      row = _ROW.replace("@@", "\n".join(row))
      stream.write(f"{row}\n")
      left = right
      right = left + _COLS_PER_ROW.value
    stream.write(_FTR)


if __name__ == "__main__":
  app.run(main)
