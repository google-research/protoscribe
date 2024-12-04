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

"""Creates FST word generator from PhoibleSegments.

Instantiates a PhoibleSegments instance, and loads cross-linguistic
syllable-type statistics from the output of collect_wiktionary_syllable_types.sh
"""

import collections
import csv
import random
import sys

from absl import flags
from absl import logging
from protoscribe.language.phonology import phoible_segments
from protoscribe.language.phonology import prosody_template_pb2

from google.protobuf import text_format
import pynini as py
# Internal resources dependency

PHONOLOGY_LANGUAGES = flags.DEFINE_list(
    "phonology_languages",
    None,
    "Select the phonemes from a subset of phonologies corresponding to this "
    "list of comma-separated ISO 639-3 language codes. By default the phonemes "
    "are sampled from any language supported by PHOIBLE. If non-None, this "
    "settings overrides the languages defined in the prosodic template proto."
)

NUMBER_OF_PHONEMES = flags.DEFINE_integer(
    "number_of_phonemes",
    -1,
    "Override ProsodyTemplate's number_of_phonemes.",
)

SYLLABLE_TEMPLATES = (
    "protoscribe/data/phonology/"
    "wiktionary_syllable_stats.tsv"
)
# Maximum number of tries to get a not-overused form before giving up
# and just returning whatever form is produced.
_MAX_TRIES = 20


def load_proto(path: str) -> prosody_template_pb2.ProsodyTemplate:
  """Loads a template from a text proto.

  Args:
    path: path to text proto.

  Returns:
    ProsodyTemplate
  """
  proto = prosody_template_pb2.ProsodyTemplate()
  with open(path, mode="rt", encoding="utf-8") as f:
    proto = text_format.Parse(f.read(), proto)
  return proto


class PhoibleTemplates:
  """Phoible prosodic template generator."""

  def __init__(
      self,
      path: str = phoible_segments.PHOIBLE,
      features_path: str = phoible_segments.PHOIBLE_FEATURES,
      syllable_templates: str = SYLLABLE_TEMPLATES,
      min_count: int = 10,
  ) -> None:
    self._phoible = phoible_segments.PhoibleSegments(path, features_path)
    self._sonorities = collections.defaultdict(set)
    self._phonemes = []
    with open(syllable_templates, mode="rt", encoding="utf-8") as stream:
      reader = csv.reader(stream, delimiter="\t", quotechar='"')
      for row in reader:
        count, sonority_count, template, sonority = row
        count, sonority_count = int(count), int(sonority_count)
        if sonority_count < min_count:
          continue
        sonority = tuple([int(c) for c in sonority.split()])
        self._sonorities[template].add(sonority)

  def choose_phonemes(
      self, k: int, p: float = 1.0, languages: list[str] | None = None
  ) -> list[str]:
    self._phonemes = self._phoible.top_k_segments(
        k=k, proportion=p, languages=languages
    )
    return self._phonemes

  @property
  def phonemes(self) -> list[str]:
    return self._phonemes

  @property
  def phoible(self) -> phoible_segments.PhoibleSegments:
    return self._phoible

  def populate_syllables(
      self,
      phonemes: list[str],
      list_of_templates: list[str],
      max_syllables: int,
      min_syllables: int = 1
  ) -> py.Fst:
    """Populates syllables from list of templates.

    Args:
      phonemes: typically the output of choose_phonemes()
      list_of_templates: list such as ["CV", "CVC", ...]
      max_syllables: maximum number of syllables to allow
      min_syllables: minimum number of syllables to allow

    Returns:
      An FST mapping from template (sequences) to instantiations.
    """
    result = py.Fst()

    def build(choices):
      result = py.union(*choices[0])
      for choice in choices[1:]:
        result += " " + py.union(*choice)
      return result

    for template in self._sonorities:
      split_template = template.split(".")
      l = len(split_template)
      if l > max_syllables or l < min_syllables:
        continue
      valid = True
      for sub_template in split_template:
        if sub_template not in list_of_templates:
          valid = False
      if not valid:
        continue
      for sonority in self._sonorities[template]:
        result |= py.cross(
            template, build(self._phoible.all_sequences(phonemes, sonority))
        )
    return result.optimize()

  def sesquisyllabify(
      self, phonemes: list[str], syllable_grammar: py.Fst
  ) -> py.Fst:
    """Special constructor for schwa CV for sesqui-syllables.

    Args:
      phonemes: phoneme list.
      syllable_grammar: a base FST syllable grammar.

    Returns:
      FST
    """
    consonants = []
    for phoneme in phonemes:
      feats = self._phoible.features(phoneme)
      if "+consonantal" in feats and "-syllabic" in feats:
        consonants.append(phoneme)
    result = py.cross("Cx.", py.union(*consonants) + " " + "ə" + " ")
    result += syllable_grammar
    return result.optimize()

  def _template_spec_to_fst(
      self,
      k: int,
      p: float,
      list_of_templates: list[str],
      max_syllables: int,
      min_syllables: int = 1,
      use_sesquisyllabic: bool = False,
      languages: list[str] | None = None,
  ) -> py.Fst:
    """Creates an FST from a template specification.

    Args:
      k: max number of segments
      p: proportion of segments to choose
      list_of_templates: list such as ["CV", "CVC", ...]
      max_syllables: maximum number of syllables to allow from templates
      min_syllables: minimum number of syllables to allow from templates
      use_sesquisyllabic: add in sesquisyllables
      languages: optional list of languages

    Returns:
      FST grammar.
    """
    # If k is 0, then it means we have already generated a phoneme set and want
    # to reuse it.
    if k == 0:
      phonemes = self._phonemes
    else:
      language_list = [language for language in languages]
      phonemes = self.choose_phonemes(k, p, languages=language_list)
    assert phonemes
    fst = self.populate_syllables(
        phonemes, list_of_templates, max_syllables, min_syllables
    )
    if use_sesquisyllabic:
      fst = py.union(self.sesquisyllabify(phonemes, fst), fst)
    return fst

  def template_spec_to_fst(
      self,
      path: str,
      number_of_phonemes: int = -1,
      phonology_languages: list[str] | None = None
  ) -> py.Fst:
    """Loads a template from a text proto.

    Args:
      path: Path to text proto specifying the template.
      number_of_phonemes: Number of phonemes to sample. Callers should only
        provide this argument for testing. At run-time, the value of
        corresponding command-line flag or the proto is used.
      phonology_languages: Languages to sample from. Callers should only
        provide this argument for testing. At run-time, the value of
        corresponding command-line flag or the proto is used.

    Returns:
      FST grammar.

    Raises:
      ValueError in case of template configuration.
    """
    proto = load_proto(path)
    if not proto.list_of_templates:
      raise ValueError("Expecting non-empty list of templates!")

    # Number of phonemes can be specified either in the proto or on the
    # command-line. The non-default argument to this method is only used
    # by the test.
    num_of_phonemes = number_of_phonemes
    if number_of_phonemes == -1:
      if NUMBER_OF_PHONEMES.value > 0:
        proto.number_of_phonemes = NUMBER_OF_PHONEMES.value
      if proto.number_of_phonemes <= 0:
        raise ValueError("Number of phonemes not defined!")
      num_of_phonemes = proto.number_of_phonemes

    # Similar logic to the above for overriding the languages.
    languages = phonology_languages
    if not phonology_languages:
      if PHONOLOGY_LANGUAGES.value:
        proto.languages[:] = PHONOLOGY_LANGUAGES.value
      languages = list(proto.languages)

    return self._template_spec_to_fst(
        num_of_phonemes,
        proto.probability_of_selection,
        list(proto.list_of_templates),
        proto.max_syllables,
        proto.min_syllables,
        proto.use_sesquisyllabic,
        languages,
    )

  def single_segment_form(self, form: str) -> bool:
    return len(form.split()) == 1

  def randgen(
      self,
      template: py.Fst,
      n: int = 1,
      max_homophony: int = sys.maxsize,
      max_number_of_single_segment_forms: int = sys.maxsize,
  ) -> list[str]:
    """Generates n forms from a prosodic specification.

    Args:
      template: a template, e.g. as generated template_spec_to_fst.
      n: number of forms to generate
      max_homophony: attempt to limit the maximum number of times a form can be
        reused to this amount.
      max_number_of_single_segment_forms: if set, restrict the number of single
        segment forms to no more than this number.

    Returns:
      A list of forms of length n.
    """
    result = []
    form_counts = collections.defaultdict(int)
    number_of_single_segment_forms = 0

    def get_form():
      nonlocal number_of_single_segment_forms
      num_tries = 0
      while True:
        seed = random.randint(0, sys.maxsize)
        form = list(py.randgen(template, seed=seed).paths().ostrings())[0]
        num_tries += 1
        if (
            number_of_single_segment_forms >= max_number_of_single_segment_forms
            and self.single_segment_form(form)
        ):
          continue
        if num_tries > _MAX_TRIES:
          break
        if form_counts[form] < max_homophony:
          form_counts[form] += 1
          break
      if form_counts[form] > max_homophony:
        logging.warning(
            "Form `%s` has been used more than %d times", form, max_homophony
        )
      if self.single_segment_form(form):
        if number_of_single_segment_forms > max_number_of_single_segment_forms:
          logging.warning(
              "More than %d single-segments forms used",
              max_number_of_single_segment_forms,
          )
        number_of_single_segment_forms += 1
      return form

    for _ in range(n):
      form = get_form()
      result.append(form)
    return result
