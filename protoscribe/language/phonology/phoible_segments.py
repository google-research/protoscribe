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

"""Interface to PHOIBLE segments."""

import collections
import csv
import logging
import os
import random

from absl import flags
from protoscribe.language.phonology import aligner

# Internal resources dependency

_PHOIBLE_SELECT_SOURCE = flags.DEFINE_string(
    "phoible_select_source", None,
    "A string identifying the source of a particular set of cross-linguistic "
    "phonological inventories in PHOIBLE, e.g., `spa` (Stanford Phonology "
    "Archive) or `upsid` (UCLA Phonological Segment Inventory Database). If "
    "set, the search for language's inventory will be restricted to this "
    "source."
)

_PHOIBLE_DIR = "protoscribe/data/phonology"
PHOIBLE = os.path.join(_PHOIBLE_DIR, "phoible-phonemes.tsv")
PHOIBLE_FEATURES = os.path.join(_PHOIBLE_DIR, "phoible-segments-features.tsv")
EMPTY_PHONEME = "0"

# Artifact of syllabification.
ParsedSyllables = tuple[tuple[tuple[str, ...] | str, ...], ...]


class PhoibleSegments:
  """Holder for Phoible segment data."""

  def __init__(
      self,
      path: str = PHOIBLE,
      features_path: str = PHOIBLE_FEATURES
  ) -> None:
    """Initializes the class.

    Args:
      path: Path to the PHOILE phonemes file in TSV format.
      features_path: Path to the PHOIBLE features path (per-segment) in TSV
        format.

    Raises:
      ValueError if the source name filter is supplied and is invalid.
    """
    self._classes = collections.defaultdict(str)
    self._languages = collections.defaultdict(set)
    inventories = collections.defaultdict(str)
    ignored_inventories = set()
    sources = set()
    with open(path, mode="rt", encoding="utf-8") as stream:
      reader = csv.reader(stream, delimiter="\t", quotechar='"')
      hdr = next(reader)
      # Inventory ID is more accurate than `Source`. In PHOIBLE, the same source
      # may have several inventories with the same language code.
      inventory_index = hdr.index("InventoryID")
      source_index = hdr.index("Source")
      language_code_index = hdr.index("LanguageCode")
      phoneme_index = hdr.index("Phoneme")
      class_index = hdr.index("Class")
      for row in reader:
        source = row[source_index]
        sources.add(source)
        if (
            _PHOIBLE_SELECT_SOURCE.value and
            _PHOIBLE_SELECT_SOURCE.value != source
        ):
          # We are in source restriction mode. Ignore all the sources that don't
          # match the requested one.
          continue
        language_code = row[language_code_index]
        inventory = row[inventory_index]
        if (language_code in inventories and
            inventory != inventories[language_code]):
          # Allow only one phoneme inventory for language. If multiple
          # inventories are present we process the first inventory that
          # appears in the file.
          ignored_inventories.add((language_code, inventory))
          continue
        inventories[language_code] = inventory
        phoneme = row[phoneme_index]
        self._classes[phoneme] = row[class_index]
        self._languages[phoneme].add(language_code)
    logging.info(
        "Chose %d phoneme inventories. Ignored %d extra inventories.",
        len(inventories), len(ignored_inventories)
    )

    # Sanity check for the valid source name.
    if (
        _PHOIBLE_SELECT_SOURCE.value and
        _PHOIBLE_SELECT_SOURCE.value not in sources
    ):
      raise ValueError(f"Invalid source name: `{_PHOIBLE_SELECT_SOURCE.value}`")

    self._stats = [(len(self._languages[ph]), ph) for ph in self._languages]
    self._stats.sort(reverse=True)
    self._num_features = 0
    with open(features_path, mode="rt", encoding="utf-8") as stream:
      reader = csv.reader(stream, delimiter="\t", quotechar='"')
      hdr = next(reader)
      self._feature_to_offset = collections.defaultdict(int)
      self._offset_to_feature = collections.defaultdict(str)
      self._features = collections.defaultdict(list)
      for offset, feature in enumerate(hdr[1:]):  # First position is segment.
        self._feature_to_offset[feature] = offset
        self._offset_to_feature[offset] = feature
      for row in reader:
        self._features[row[0]] = row[1:]
        self._num_features = len(row[1:])
    assert self._num_features > 0

  @property
  def classes(self) -> collections.defaultdict:
    return self._classes

  @property
  def languages(self) -> collections.defaultdict:
    return self._languages

  @property
  def stats(self) -> list[tuple[int, str]]:
    return self._stats

  def is_vowel(self, segment: str) -> bool:
    return "+syllabic" in self.features(segment)

  def features(
      self, segment: str, keep_unspecified: bool = False
  ) -> list[str]:
    segment = self._clean_phoneme(segment)
    if segment not in self._features:
      return []
    result = []
    for offset, value in enumerate(self._features[segment]):
      if not keep_unspecified and value == "0":
        continue
      result.append(f"{value}{self._offset_to_feature[offset]}")
    return result

  def sonority(self, segment: str) -> int:
    """Computes the sonority of a segment.

    Args:
      segment: a segment

    Returns:
      integer representing the sonority from 0 to 4.
    """
    feats = self.features(segment)
    if not feats:
      return -1
    if "-consonantal" in feats:  # Vowel or Glide
      if "-syllabic" in feats:  # Glide
        return 3
      return 4  # Vowel
    if "+sonorant" in feats:
      return 2
    if "+continuant" in feats:
      return 1
    return 0  # Any other consonant

  def sonority_profile(
      self, sequence: list[str] | tuple[str, ...]
  ) -> tuple[int, ...]:
    """Returns sonority profile."""
    return tuple([self.sonority(ph) for ph in sequence])

  # Deal with things not in the Phoible set such as some ligatures.
  def _clean_phoneme(self, ph: str) -> str:
    ph = ph.replace("͡", "")
    return ph

  # TODO: Possibly improve the syllable boundary placement with
  # sonority considerations.
  def syllabify(
      self, phoneme_sequence: str, parse_onsets_and_rhymes: bool = False
  ) -> ParsedSyllables:
    """Syllabifies a phoneme sequence using a crude maxim onset criterion.

    Args:
      phoneme_sequence: space-separated string of phonemes
      parse_onsets_and_rhymes: bool

    Returns:
      Tuple of syllables. If parse_onsets_and_rhymes is True, then also parses
      each syllable into its onsets and rhymes.
    """
    phoneme_sequence = phoneme_sequence.split()
    classes = []
    segments = []

    for phoneme in phoneme_sequence:
      cphoneme = self._clean_phoneme(phoneme)
      clas = self._classes[cphoneme]
      if not clas:
        self._classes[cphoneme] = "unknown"
      if clas != "tone":
        segments.append(phoneme)
        if self.is_vowel(phoneme):
          clas = "vowel"
        classes.append(clas)

    def getclass(i, classes):
      if i >= len(classes):
        return None
      return classes[i]

    def has_syllabic(syllable):
      for phoneme in syllable:
        if "+syllabic" in self.features(self._clean_phoneme(phoneme)):
          return True
      return False

    def parse_onset_rhyme(syllable):
      onset = []
      coda = []
      constituent = onset
      for phoneme in syllable:
        if "+syllabic" in self.features(self._clean_phoneme(phoneme)):
          constituent = coda
        constituent.append(phoneme)
      return tuple(onset), tuple(coda)

    syllables = []
    syllable = []
    for i in range(len(classes)):
      clas = classes[i]
      syllable.append(segments[i])
      if clas == "vowel":
        if getclass(i + 1, classes) == "vowel":
          syllables.append(tuple(syllable))
          syllable = []
        elif (
            getclass(i + 1, classes) == "consonant"
            and getclass(i + 2, classes) == "vowel"
        ):
          syllables.append(tuple(syllable))
          syllable = []
      elif clas == "consonant":
        if has_syllabic(syllable) and getclass(i + 1, classes) == "consonant":
          syllables.append(tuple(syllable))
          syllable = []
    if syllable:
      if not syllables or has_syllabic(syllable):
        syllables.append(tuple(syllable))
      else:  # Trailing consonants
        syllables[-1] = tuple(list(syllables[-1]) + syllable)
    if parse_onsets_and_rhymes:
      new_syllables = []
      for syllable in syllables:
        new_syllables.append(parse_onset_rhyme(syllable))
      syllables = new_syllables
    return tuple(syllables)

  def template(self, syllables: ParsedSyllables) -> str:
    """Returns a CV template from a syllabified string.

    Args:
      syllables: the output of syllabify(), with parse_onsets_and_rhymes=False.

    Returns:
      A string of the form CVC.CV.CVC etc.
    """
    cv_template = []
    for syllable in syllables:
      cv_syllable = []
      for phone in syllable:
        phone = self._clean_phoneme(phone)
        if self._classes[phone] == "consonant":
          clas = "C"
        elif self._classes[phone] == "vowel":
          clas = "V"
        else:
          clas = "?"
        cv_syllable.append(clas)
      cv_template.append("".join(cv_syllable))
    return ".".join(cv_template)

  def top_k_segments(
      self, k: int, proportion: float = 1.0, languages: list[str] | None = None
  ) -> list[str]:
    """Select proportion of the k most frequent segments cross-linguistically.

    Args:
      k: number of segments to pick. Set to -1 to select all segments.
      proportion: proportion of k to actually use
      languages: if specified a list of ISO-639-2 language codes from which to
        select phonemes

    Returns:
      list of segments
    """
    if languages:
      stats = []
      for cnt, ph in self._stats:
        for language in languages:
          if language in self._languages[ph]:
            stats.append((cnt, ph))
            break
    else:
      stats = self._stats
    if k == -1:
      k = len(stats)
    segments = [p[1] for p in stats[:k]]
    if proportion < 1.0:
      random.shuffle(segments)
      segments = segments[: int(proportion * k)]
    segments.sort()
    return segments

  def all_sequences(
      self, segments: list[str], sonority_profile: tuple[int, ...]
  ) -> list[list[str]]:
    """All possible sequences given a sonority profile given segments.

    Args:
      segments: A list of segments
      sonority_profile: a sonority profile as output by sonority_profile()

    Returns:
      A template with specific segments filled out for each position.
    """
    template = []
    for index in sonority_profile:
      matching_segments = []
      for segment in segments:
        if self.sonority(segment) == index:
          matching_segments.append(segment)
      template.append(matching_segments)
    return template

  # Functionality for ease of writing phonological rules: for a given list of
  # features, and a given list of phonemes, find all phonemes that match the
  # features.
  def matching_phonemes(
      self, feature_list: list[str], phoneme_list: list[str]
  ) -> list[str]:
    """Returns matching phonemes for a list of features.

    Args:
      feature_list: a list of feature specifications.
      phoneme_list: a list of phonemes.

    Returns:
      List of matching phonemes.
    """
    result = []
    for phoneme in phoneme_list:
      feats = self.features(self._clean_phoneme(phoneme))
      matched = True
      for feature in feature_list:
        if feature not in feats:
          matched = False
      if matched:
        result.append(phoneme)
    result.sort()
    return result

  def phonetic_distance(
      self, phoneme1: str, phoneme2: str, dist_of_empty: float = -1
  ) -> float:
    """Computes the phonetic distance between two phonemes.

    Args:
      phoneme1: A phoneme.
      phoneme2: A phoneme.
      dist_of_empty: Distance to assign if one phoneme is the empty phoneme.

    Returns:
      Featural distance between the two phonemes.
    """
    if phoneme1 == phoneme2:
      return 0
    if EMPTY_PHONEME in [phoneme1, phoneme2]:
      return self._num_features if dist_of_empty == -1 else dist_of_empty
    feats1 = self.features(phoneme1, keep_unspecified=True)
    feats2 = self.features(phoneme2, keep_unspecified=True)
    dist = 0
    for c1, c2 in zip(feats1, feats2):
      if c1[0] == "0" or c2[0] == "0":
        continue
      dist += 0 if c1 == c2 else 1
    return dist

  def string_distance(
      self,
      s1: str | list[str] | tuple[str, ...],
      s2: str | list[str] | tuple[str, ...]
  ) -> tuple[float, list[tuple[str | None, str | None]]]:
    """Computes string edit distance with featural distance.

    Args:
      s1: a space-delimited phoneme string or list.
      s2: a space-delimited phoneme string or list.

    Returns:
      A pair of string distance and best path
    """

    ins_cost = 10

    def insert(unused_item):
      return ins_cost

    def delete(unused_item):
      return ins_cost

    cost, path = aligner.best_match(
        s1, s2, insert=insert, delete=delete, substitute=self.phonetic_distance
    )
    return cost, path
