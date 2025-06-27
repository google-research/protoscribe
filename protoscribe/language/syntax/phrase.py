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

"""Definition of simple phrases."""

import enum
import itertools
from typing import Callable

from protoscribe.language.morphology import morphology
from protoscribe.language.morphology import numbers as numbers_lib
from protoscribe.language.phonology import sampa

from pynini.lib import features

# TODO: `#` is specified in Phrase.verbalize() and in phrase_to_words,
# separately, which is a tad risky.


class Phrase:
  """Base class for phrases.

  Defines a verbalizer which takes whitespace-delimited phrase and
  applies rules to each token as defined by the derived class.
  """

  def __init__(self) -> None:
    pass

  def verbalize(
      self,
      phrase: str,
      delimiter: str = " ",
      out_delimiter: str = f" {sampa.WORD_BOUNDARY} "
  ) -> str:
    phrase = phrase.split(delimiter)
    result = []
    rules = self._genrules(phrase)
    for rule, elt in itertools.zip_longest(rules, phrase):
      result.append(rule(elt) if rule else elt)
    return out_delimiter.join(result)

  def _genrules(self, unused_phrase):
    return []


class CountedNoun(Phrase):
  """Verbalize noun phrases of the form number noun."""

  class NumberType(enum.Enum):
    ONE = 1
    TWO = 2
    MANY = 3

  def number_type(self, num: str) -> NumberType:
    num = int(num)
    if num == 1:
      return self.NumberType.ONE
    elif num == 2:
      return self.NumberType.TWO
    return self.NumberType.MANY

  def __init__(
      self,
      noun_morphology: morphology.BasicNounMorphologyBuilder,
      numbers: numbers_lib.Numbers,
      one: list[str],
      two: list[str] | None = None,
      many: list[str] | None = None
  ) -> None:
    super().__init__()
    self._noun_morphology = noun_morphology
    self._numbers = numbers
    self._one = one
    self._many = many if many else one
    self._two = self._many
    self._two = two if two else self._two

  def _genrules(self, phrase: str) -> list[Callable[[str], str]]:
    number, _ = phrase
    number_type = self.number_type(number)
    if number_type == self.NumberType.ONE:
      feats = self._one
    elif number_type == self.NumberType.TWO:
      feats = self._two
    else:
      feats = self._many
    fv = features.FeatureVector(self._noun_morphology.cat, *feats)
    # pylint: disable=unnecessary-lambda
    return [
        lambda t: self._numbers.verbalize(t),
        lambda t: self._noun_morphology.inflected_form(t, fv)[1]
    ]
    # pylint: enable=unnecessary-lambda


def phrase_to_words(
    phrase: str, wb_marker: str = sampa.WORD_BOUNDARY
) -> list[str]:
  """Helper function to split phrase on wb_marker.

  Args:
    phrase: a string.
    wb_marker: word boundary marker.

  Returns:
    list of words.
  """
  return [w.strip() for w in phrase.split(wb_marker)]
