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

"""Classes and functions for basic morpheme form generation."""

import collections
import sys
from typing import Callable

from protoscribe.language.phonology import phoible_segments
from protoscribe.language.phonology import phoible_templates as pt

import glob
import os


# TODO: Provide other definitions of phonological weight.
def _phonological_weight(x: str, func: Callable[[str], float] = len) -> float:
  """Defines a simple phonological weight function.

  Args:
    x: A phonological string
    func: A function to compute the phonological weight

  Returns:
    Phonological weight according to func.
  """
  return func(x)


# Phonological forms are now space delimited.
def _phonological_len(x: str) -> int:
  return len(x.split())


class Morpheme:
  """A morpheme consists of a meaning and a form."""

  def __init__(self, meaning: str, form: str) -> None:
    self._meaning = meaning
    self._form = form

  def __repr__(self) -> str:
    return f"{self._meaning}/{self._form}"

  @property
  def meaning(self) -> str:
    return self._meaning

  @property
  def form(self) -> str:
    return self._form

  def phonological_weight(
      self, func: Callable[[str], float] = _phonological_len
  ) -> float:
    return _phonological_weight(self._form, func=func)


class Lexicon:
  """A lexicon consists of a set of morphemes, and provides accessors for these."""

  def __init__(self) -> None:
    self._meanings_to_morphemes = collections.defaultdict(Morpheme)
    self._forms_to_morphemes = collections.defaultdict(list)
    self._phoible_templates = None

  def __repr__(self) -> str:
    return "\n".join(map(repr, self._meanings_to_morphemes.values()))

  def set_phoible_templates(
      self, phoible_templates: pt.PhoibleTemplates
  ) -> None:
    self._phoible_templates = phoible_templates

  @property
  def phoible_templates(self) -> pt.PhoibleTemplates | None:
    return self._phoible_templates

  def add_morpheme(self, morpheme: Morpheme) -> None:
    self._meanings_to_morphemes[morpheme.meaning] = morpheme
    self._forms_to_morphemes[morpheme.form].append(morpheme)

  @property
  def meanings_to_morphemes(self) -> collections.defaultdict:
    return self._meanings_to_morphemes

  @property
  def forms_to_morphemes(self) -> collections.defaultdict:
    return self._forms_to_morphemes

  def meanings_to_forms(self) -> list[tuple[str, str]]:
    return [(m, self._meanings_to_morphemes[m].form)
            for m in self._meanings_to_morphemes]

  def dump_lexicon(self, path: str) -> None:
    with open(path, "w") as stream:
      for m, f in sorted(self.meanings_to_forms()):
        stream.write(f"{m}\t{f}\n")

  def load_lexicon(self, path: str) -> None:
    self._meanings_to_morphemes.clear()
    self._forms_to_morphemes.clear()
    with open(path) as stream:
      for line in stream:
        m, f = line.strip("\n").split("\t")
        self.add_morpheme(Morpheme(m, f))


def make_lexicon(
    phoible_proto: str,
    meanings: list[str] | tuple[str, ...],
    crib_lexicon: Lexicon | None = None,
    weight_func: Callable[[str], float] = _phonological_len,
    max_homophony: int = sys.maxsize,
    max_number_of_single_segment_forms: int = sys.maxsize,
    phoible_path: str = phoible_segments.PHOIBLE,
    phoible_features_path: str = phoible_segments.PHOIBLE_FEATURES,
    phoible_syllable_templates: str = pt.SYLLABLE_TEMPLATES,
    phoible: pt.PhoibleTemplates | None = None
):
  """Constructs a lexicon given a grammar, phoneme classes, and meanings.

  Forms from the prosodic grammar are randomly assigned to meanings.

  If crib_lexicon is not None, then this should be a Lexicon object.
  This is used to define a partial ordering for the morphemes in the
  new lexicon so that phonologically heavier forms in the new lexicon
  will correspond to phonologically heavier forms in the crib
  lexicon. This attempts to provide a quasi-natural distribution to
  the mapping between meaning and form on the assumption that meaning
  X has a longer form than meaning Y in some real language, then it is
  reasonable for meaning X to have a longer form than meaning Y in our
  artificial language.

  Args:
    phoible_proto: path to phoible template textproto
    meanings: a list of meanings
    crib_lexicon: a Lexicon
    weight_func: a phonological weight function
    max_homophony: attempt to limit the maximum number of times a form can be
      reused to this amount.
    max_number_of_single_segment_forms: maximum number of forms that can have a
      single (vowel) segment.
    phoible_path: path to PHOIBLE segments.
    phoible_features_path: path to PHOIBLE features.
    phoible_syllable_templates: path to PHOIBLE syllable templates.
    phoible: if not None, an already constructed PHOIBLE.

  Returns:
    A Lexicon
  """
  lexicon = Lexicon()
  if not phoible:
    phoible = pt.PhoibleTemplates(
        phoible_path, phoible_features_path, phoible_syllable_templates
    )
  template = phoible.template_spec_to_fst(phoible_proto)

  forms = phoible.randgen(
      template,
      n=len(meanings),
      max_homophony=max_homophony,
      max_number_of_single_segment_forms=max_number_of_single_segment_forms,
  )
  if crib_lexicon is not None:
    crib_meanings = list(crib_lexicon.meanings_to_morphemes.keys())
    crib_meanings.sort(
        # Whine whine whine: you can't have it both ways.
        # pylint: disable=g-long-lambda
        key=lambda x: crib_lexicon.meanings_to_morphemes[x].phonological_weight(
            weight_func),
        # pylint: enable=g-long-lambda
        reverse=True)
    forms.sort(key=lambda x: _phonological_weight(x, weight_func), reverse=True)
    meanings_set = set(meanings)
    crib_meanings_set = set(crib_meanings)
    shared_meanings = [m for m in crib_meanings if m in meanings_set]
    residual_meanings = [m for m in meanings if m not in crib_meanings_set]
    meanings = shared_meanings + residual_meanings
  for meaning, form in zip(meanings, forms):
    lexicon.add_morpheme(Morpheme(meaning, form))
  lexicon.set_phoible_templates(phoible)
  return lexicon
