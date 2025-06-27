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

r"""Displays SAMPA segments for specified set of languages.

Example:
--------
# Pirahã language.
python protoscribe/language/phonology/sampa_for_language_main.py \
  --phonology_languages myp --number_of_phonemes 10 --logtostderr

# Select 20 most frequent phonemes across all languages.
python protoscribe/language/phonology/sampa_for_language_main.py \
  --number_of_phonemes 20 --logtostderr

# Select full Russian inventory from UPSID (UCLA Phonological Segment Inventory
# Database).
python protoscribe/language/phonology/sampa_for_language_main.py \
  --phonology_languages rus --phoible_select_source upsid --logtostderr
"""

from collections.abc import Sequence
import logging

from absl import app
from absl import flags
from protoscribe.language.phonology import phoible_templates as phoible
from protoscribe.language.phonology import sampa

_REMOVE_STRESS = flags.DEFINE_bool(
    "remove_stress", True,
    "Removes stress for the resulting SAMPA phonemes."
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  num_phonemes = phoible.NUMBER_OF_PHONEMES.value
  languages = phoible.PHONOLOGY_LANGUAGES.value
  logging.info(
      "Choosing %s phonemes for languages %s ...",
      str(num_phonemes) if num_phonemes > 0 else "all",
      languages if languages else "all"
  )
  phonology = phoible.PhoibleTemplates()
  segments = phonology.choose_phonemes(k=num_phonemes, languages=languages)
  logging.info("Chose %d segments.", len(segments))
  logging.info("IPA segments: %s", " ".join(sorted(segments)))

  converter = sampa.IpaToSampaConverter()
  segments = [converter.convert(p) for p in segments]
  if _REMOVE_STRESS.value:
    segments = [p[:-1] if p.endswith("\"") else p for p in segments]
  logging.info("SAMPA segments: %s", " ".join(segments))


if __name__ == "__main__":
  app.run(main)
