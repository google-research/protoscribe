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

"""Simple number-name grammar.

Based on the tried and true grammatical model used in speech/numbers,
but simplified since we can limit the size of the numbers for this
application. Currently this goes to 9999.

Limitations:

1) We assume a decimal system. (Mesoamerican systems were
vigesimal...)

2) We assume that number names are largely compositional. (Nothing
like the separate word for each value up to 99 as in North Indian
languages.)
"""

from protoscribe.language.morphology import morphemes

import pynini
from pynini.lib import byte

Fst = pynini.Fst
FstLike = pynini.FstLike

_d = byte.DIGIT
_D = _d - "0"


def d(expr: FstLike) -> Fst:
  """Short-named deletion helper.

  Args:
    expr: FST-like

  Returns:
    FST deleting expr
  """
  return pynini.cross(expr, "")


def i(expr: FstLike) -> Fst:
  """Short named insertion helper.

  Args:
    expr: FST-like

  Returns:
    FST inserting expr
  """
  return pynini.cross("", expr)


# Definition of powers, including versions that can be initial
# (capitalized) and versions that can involve deletion of the zero
# (lower-case).
_THOUSAND = _D + i("[E3]")
_HUNDRED = _D + i("[E2]")
_DECADE = _D + i("[E1]")
_UNIT = _D
_hundred = _HUNDRED | d("0")
_decade = _DECADE | d("0")
_unit = _UNIT | d("0")

# One deletion rule. Right now this either applies to all powers of
# ten or to none.
_ONE_DELETION = pynini.cdrewrite(
    d("1"), "", pynini.union("[E1]", "[E2]", "[E3]"),
    (byte.BYTE | "[E1]" | "[E2]" | "[E3]").closure())

_NUMBER_GRAMMAR = ((_THOUSAND + _hundred + _decade + _unit) |
                   (_HUNDRED + _decade + _unit) | (_DECADE + _unit) | _UNIT)

# Version of the number grammar with one-deletion.
_NUMBER_GRAMMAR_ONE_DELETION = _NUMBER_GRAMMAR @ _ONE_DELETION


class Numbers:
  """Definition of number names.

  Map between digits and numbers is given in a lexicon. Optional rules
  can be applied to the output. If one_deletion is True then use the
  version of the number grammar with one deletion.
  """

  def __init__(
      self,
      lexicon: morphemes.Lexicon,
      rules: list[Fst] | None = None,
      one_deletion: bool = False
  ) -> None:
    self._powers = lexicon.meanings_to_forms()
    self._rules = [] if rules is None else rules
    self._grammar = pynini.union(
        *[pynini.cross(p, m) for (p, m) in self._powers])
    self._grammar = self._grammar + (i(" # ") + self._grammar).closure()
    number_grammar = (
        _NUMBER_GRAMMAR_ONE_DELETION if one_deletion else _NUMBER_GRAMMAR)
    self._grammar = number_grammar @ self._grammar
    for rule in self._rules:
      self._grammar @= rule
    self._grammar.optimize()
    self._igrammar = pynini.invert(self._grammar)

  def verbalize(self, number: int) -> str:
    number_name = (
        str(int(number)) @ self._grammar).project("output").paths().ostrings()
    return list(number_name)[0]

  def deverbalize(
      self, number_name: str, return_num_interpretations: bool = False
  ) -> str | tuple[str, int]:
    number = (number_name @ self._igrammar).project("output").paths().ostrings()
    number = list(number)
    if return_num_interpretations:
      return number[0], len(number)
    else:
      return number[0]

  @property
  def powers(self) -> list[tuple[str, str]]:
    return sorted(self._powers)
