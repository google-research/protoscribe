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

"""Glyph vocabulary manipulation."""

import collections
import dataclasses
import json
import os
import random
import re
from typing import Optional

from absl import flags
from absl import logging
from protoscribe.glyphs import numbers_to_glyphs as number_lib

import glob
import os
# Internal resources dependency

_GLYPH_VOCAB_FILE = flags.DEFINE_string(
    "glyph_vocab_file",
    None,
    "Glyph vocabulary file in JSON format. A one-to-one mapping between glyph "
    "names and the corresponding unique IDs (tokens). This can be a path of a "
    "prebuilt vocabulary *or*, if the file does not exist, to a vocabulary "
    "which needs to be created by this tool."
)

_INCLUDE_GLYPH_VARIANTS = flags.DEFINE_bool(
    "include_glyph_variants",
    False,
    "If enabled, we sample from multiple visual variants (if present) for a "
    "given glyph."
)

_EXTENSION_GLYPHS_SVG_DIR = flags.DEFINE_string(
    "extension_glyphs_svg_dir",
    None,
    "Directory housing discovered extension glyphs in SVG format."
)

_GLYPHS_ROOT_DIR = "protoscribe/data/glyphs"
GLYPH_SET_DIRS = [
    f"{_GLYPHS_ROOT_DIR}/generic/administrative_categories",
    f"{_GLYPHS_ROOT_DIR}/generic/numbers",
]

# Name of the dummy glyph used for the unseen concepts.
DUMMY_GLYPH_NAME = "DUMMY"

# Special glyph IDs.
GLYPH_PAD = 0
GLYPH_BOS = 1
GLYPH_EOS = 2
GLYPH_UNK = 3

# Glyph mask values. A typical glyph sequence would look like:
# ['X', 'X', 'X', 'X', 'I', 'I', 'I', 'water', 'river']
GLYPH_TYPE_MASK_NUMBER = 1
GLYPH_TYPE_MASK_CONCEPT = 2
GLYPH_TYPE_MASK_PAD = 0

_ROMAN_SINGLE_NUMERAL_GLYPHS_REGEX = r"^[CDILVXM]$"


def vocab_svg_path2key(svg_path: str, strip_category: bool = True) -> str:
  """Converts SVG path to the vocabulary key."""
  path_parts = svg_path.split("/")
  if len(path_parts) < 2:
    raise ValueError(f"Invalid path: {svg_path}")
  category_type = path_parts[-2]
  base = path_parts[-1].replace(".svg", "")
  base_parts = base.split("_")
  if len(base_parts) > 1:  # Multiple variants of the same glyph.
    base = "_".join(base_parts[:-1])
  return base if strip_category else f"{category_type}/{base}"


def glyph_is_single_numeral(name: str) -> bool:
  """Checks whether glyph is number."""
  return re.search(name, _ROMAN_SINGLE_NUMERAL_GLYPHS_REGEX) is not None


def special_glyph_names() -> list[str]:
  """Returns names of all the special glyphs including numbers."""
  return [
      "<pad>", "<s>", "</s>", "</unk>",
      DUMMY_GLYPH_NAME, *number_lib.NUMBER_GLYPHS,
  ]


def find_all_svg_files() -> list[str]:
  """Finds all SVGs in the configured directories."""
  svg_paths = []
  for glyph_set_dir in GLYPH_SET_DIRS:
    logging.info("Looking for SVGs under %s ...", glyph_set_dir)
    file_paths = []
    for dir_path, _, filenames in os.walk(glyph_set_dir):
      for filename in filenames:
        if filename.endswith(".svg"):
          file_paths.append(os.path.join(dir_path, filename))
    if not file_paths:
      raise ValueError(f"No SVG files found for category \"{glyph_set_dir}\"")
    svg_paths.extend(file_paths)

  if _EXTENSION_GLYPHS_SVG_DIR.value:
    path = os.path.join(_EXTENSION_GLYPHS_SVG_DIR.value, "*.svg")
    logging.info("Reading %s ...", _EXTENSION_GLYPHS_SVG_DIR.value)
    for path in glob.glob(path):
      svg_paths.append(path)

  svg_paths = [*collections.Counter(svg_paths)]  # Uniques.
  logging.info(
      "Found %d total SVGs in %d directories.",
      len(svg_paths),
      len(GLYPH_SET_DIRS),
  )
  return svg_paths


@dataclasses.dataclass
class GlyphVocab:
  """Glyph vocabulary container."""

  vocab: Optional[dict[str, tuple[int, list[str]]]] = None
  _glyph_names: Optional[list[str]] = None

  def init(self, glyph_id_offset: int = 0) -> None:
    """Builds vocabulary mapping glyphs to integer tokens."""
    logging.info("Building glyph vocab ...")
    svg_paths = find_all_svg_files()
    self.vocab = {
        "<pad>": (GLYPH_PAD, []),
        "<s>": (GLYPH_BOS, []),
        "</s>": (GLYPH_EOS, []),
        "</unk>": (GLYPH_UNK, []),
    }
    self._glyph_names = [  # Unique glyph names.
        "<pad>", "<s>", "</s>", "</unk>"
    ]
    glyph_id = glyph_id_offset if glyph_id_offset else len(self.vocab)
    for path in svg_paths:
      glyph_key = vocab_svg_path2key(path)
      if glyph_key in self.vocab and not _INCLUDE_GLYPH_VARIANTS.value:
        logging.warning("Glyph already exists for `%s`! Ignoring...", glyph_key)
        # Note: We still fill in all the available glyphs in here. This is by
        # design because it allows us to keep the same vocab for either the
        # single variant or multi-variant mode.
      if glyph_key not in self.vocab:
        glyph_id = len(self._glyph_names)
        self.vocab[glyph_key] = (glyph_id, [])
        self._glyph_names.append(glyph_key)
      self.vocab[glyph_key][1].append(path)

    num_glyphs = len(self.vocab) - 4  # Exclude special tokens.
    num_variants = (
        len(svg_paths) if _INCLUDE_GLYPH_VARIANTS.value else num_glyphs
    )
    logging.info("Initialized %d glyphs with %.2f variants per glyph.",
                 num_glyphs, num_variants / num_glyphs)
    if _INCLUDE_GLYPH_VARIANTS.value:
      max_variants = -1
      for name in self.vocab:
        if len(self.vocab[name][1]) > max_variants:
          max_variants = len(self.vocab[name][1])
      logging.info("Max variants per glyph: %d", max_variants)

  def tokenize(
      self,
      glyph_names: list[str],
      number_mask: list[bool],
  ) -> tuple[list[int], list[int]]:
    """Converts a list of glyphs to a list of integer tokens."""

    if not self.vocab:
      raise ValueError("Vocabulary is empty!")
    if len(glyph_names) != len(number_mask):
      raise ValueError("Lengths of glyph sequence and mask should match!")

    tokens = [self.vocab["<s>"][0]]
    number_concept_mask = [GLYPH_TYPE_MASK_PAD]
    for idx, glyph_key in enumerate(glyph_names):
      if glyph_key not in self.vocab:
        raise ValueError(f"Glyph {glyph_key} not found in the vocabulary!")
      tokens.append(self.vocab[glyph_key][0])
      if number_mask[idx]:
        number_concept_mask.append(GLYPH_TYPE_MASK_NUMBER)
      else:
        number_concept_mask.append(GLYPH_TYPE_MASK_CONCEPT)
    tokens.append(self.vocab["</s>"][0])
    number_concept_mask.append(GLYPH_TYPE_MASK_PAD)
    return tokens, number_concept_mask

  def detokenize(self, glyph_ids: list[int]) -> list[str]:
    """Converts list of glyph tokens to list of glyph names."""
    if not self._glyph_names:
      raise ValueError("Vocabulary is empty!")
    return [self._glyph_names[idx] for idx in glyph_ids]

  def find_svg_from_name(
      self,
      name: str,
  ) -> Optional[tuple[str, int]]:
    """Finds an SVG given a name.

    If we have multiple visual variants per glyph, we randomly sample from the
    available representations, otherwise we return the first SVG that we find.

    Args:
      name: a name, e.g. `horse`

    Returns:
      A path to an SVG along with the variant ID, or None if not found.
    """
    if not self.vocab:
      raise ValueError("Vocabulary not initialized!")

    # TODO: This removes the POS tag, if any, but eventually we
    # don't want to do that...
    name = name.split("_")[0]
    if name in self.vocab:
      if not _INCLUDE_GLYPH_VARIANTS.value:
        return self.vocab[name][1][0], 0
      else:
        idx = random.randrange(0, len(self.vocab[name][1]))
        return self.vocab[name][1][idx], idx
    else:
      return None

  def load(self, vocab_file: str) -> None:
    """Loads the vocabulary."""
    logging.info("Reading glyph vocab from %s ...", vocab_file)
    with open(vocab_file) as f:
      json_data = json.load(f)
      self.vocab = json_data["vocab"]
      self._glyph_names = json_data["glyph_names"]
    logging.info("Loaded %d glyphs.", len(self.vocab))

  def save(self) -> None:
    logging.info("Saving glyph vocab to %s ...", _GLYPH_VOCAB_FILE.value)
    with open(_GLYPH_VOCAB_FILE.value, mode="w") as f:
      json_data = {
          "vocab": self.vocab,
          "glyph_names": self._glyph_names,
      }
      json.dump(json_data, f)

  def __len__(self):
    return len(self.vocab) if self.vocab is not None else 0

  def id_to_name(self, idx: int) -> str:
    if idx >= len(self._glyph_names):
      raise ValueError(f"Invalid glyph ID {idx}!")
    return self._glyph_names[idx]

  def name_to_id(self, name: str) -> int:
    return self.vocab[name][0]

  def special_glyph_ids(self) -> list[int]:
    """Converts the names of special glyphs to vocab IDs."""
    return sorted([self.name_to_id(name) for name in special_glyph_names()])


def load_glyph_vocab(vocab_file: str) -> GlyphVocab:
  """Loads glyph vocabulary."""
  if not os.path.exists(vocab_file):
    raise ValueError(f"Glyph vocabulary file `{vocab_file}` does not exist!")
  vocab = GlyphVocab()
  vocab.load(vocab_file)
  return vocab


def load_or_build_glyph_vocab(vocab_file: Optional[str] = None) -> GlyphVocab:
  """Builds a new or loads the existing glyph vocabulary file."""
  vocab = GlyphVocab()
  if not vocab_file and _GLYPH_VOCAB_FILE.value:
    if os.path.exists(_GLYPH_VOCAB_FILE.value):
      vocab_file = _GLYPH_VOCAB_FILE.value

  if vocab_file:
    vocab.load(vocab_file)
  else:
    vocab.init()
    if _GLYPH_VOCAB_FILE.value:
      vocab.save()
  logging.info("Glyph vocab with %d entries", len(vocab))
  return vocab
