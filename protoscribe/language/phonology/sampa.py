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

"""IPA to X-SAMPA converter.

See:
  https://wwwhomes.uni-bielefeld.de/gibbon/EGA/Formats/Sampa/sampa.html
  https://en.wikipedia.org/wiki/X-SAMPA
"""

import collections

from absl import logging
from protoscribe.language.phonology import phoible_segments

# Internal resources dependency

SAMPA_PATH = (
    "protoscribe/data/phonology/ipa_to_sampa.tsv"
)

_SYLL = "."  # This MUST be a single character.

WORD_BOUNDARY = "#"

# Various stress levels.
PRIMARY_STRESS_MARK = "\""
SECONDARY_STRESS_MARK = "%"
TERTIARY_STRESS_MARK = "-"  # Or unstressed.


def is_vowel(segment: str) -> bool:
  """Checks if the segment happens to be a vowel."""
  return segment[-1] in [
      PRIMARY_STRESS_MARK, SECONDARY_STRESS_MARK, TERTIARY_STRESS_MARK
  ]


class IpaToSampaConverter:
  """Converter for IPA to (T)SAMPA."""

  def __init__(
      self,
      sampa_path: str = SAMPA_PATH,
      phoible_path: str = phoible_segments.PHOIBLE,
      phoible_features_path: str = phoible_segments.PHOIBLE_FEATURES,
  ) -> None:
    self._ipa_to_sampa = collections.defaultdict(str)
    with open(sampa_path, mode="rt", encoding="utf-8") as stream:
      for line in stream:
        if line.startswith("#"):
          continue
        try:
          ipa, sampa = line.split()
          self._ipa_to_sampa[ipa] = sampa
        except ValueError:
          continue
    self._phoible = phoible_segments.PhoibleSegments(
        path=phoible_path, features_path=phoible_features_path
    )

  # TODO: This sets primary stress on the first vowel and
  # secondary stresses elsewhere, unless the vowel is /ə/, in which
  # case that always gets secondary stress. At some point we may want
  # a more flexible set of options.
  def convert(self, ipa: str) -> str:
    """Converts IPA to T-SAMPA.

    Args:
      ipa: IPA string

    Returns:
      T-SAMPA string
    """
    sampa = []
    stress_mark = PRIMARY_STRESS_MARK  # Primary stress.
    syllables = self._phoible.syllabify(ipa)
    sampa = []
    for syllable in syllables:
      for segment in syllable:
        s = self._ipa_to_sampa[segment]
        if not s:
          logging.error('No SAMPA equivalent for IPA: "%s"', segment)
          s = segment
        if self._phoible.is_vowel(segment):
          if segment == "ə":
            # Don't change stress_mark.
            stress = TERTIARY_STRESS_MARK
          else:
            stress = stress_mark  # First non-schwa gets primary
            stress_mark = SECONDARY_STRESS_MARK
          sampa.append(s + stress)
        else:
          sampa.append(s)
      sampa.append(_SYLL)
    sampa = sampa[:-1]  # Removes final _SYLL.
    return " ".join(sampa)
