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

"""Produces various glyph representations of numerals."""

from typing import Union

# All Roman numerals.
NUMBER_GLYPHS = frozenset("IVXLMCD")


def pseudo_roman(number: Union[int, str]) -> str:
  """Converts number to pseudo-Roman numeral.

  Args:
    number: integer or string.

  Returns:
    String with Roman numeral representation.
  """
  r = int(number)
  m = r // 1000
  r = r % 1000
  d = r // 500
  r = r % 500
  c = r // 100
  r = r % 100
  l = r // 50
  r = r % 50
  x = r // 10
  r = r % 10
  v = r // 5
  i = r % 5
  return "M" * m + "D" * d + "C" * c + "L" * l + "X" * x + "V" * v + "I" * i


def number_and_other_glyphs(glyphs: list[str]) -> tuple[list[str], list[str]]:
  """Return sequences of number and other glyphs in larger sequence.

  Assumes the sequence consists of numbers followed by other glyphs.

  Args:
    glyphs: A list of string glyphs.

  Returns:
    Tuple of number glyphs and non-number glyphs.
  """
  numbers = []
  concepts = []
  for glyph in glyphs:
    if glyph in NUMBER_GLYPHS:
      numbers.append(glyph)
    else:
      concepts.append(glyph)
  return numbers, concepts
