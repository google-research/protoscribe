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

"""Extends spellings to new terms based on semantics and/or phonetics.

Basic algorithm The algorithm described in Richard Sproat (2023)
"Symbols: An Evolutionary History from the Stone Age to the Future.",
Heidelberg, Springer, 2023.
See also: https://github.com/rwsproat/symbols/tree/main
"""

import logging

import numpy as np
from protoscribe.evolution import new_spellings_utils as utils


def glyph_extensions() -> list[
    tuple[str, str, str, str, list[np.ndarray | None]]
]:
  """Computes glyph extensions and associated debug information.

  Uses arguments passed through command-line flags defined in
  `new_spellings_utils.py`.

  Returns:
    A list of six-tuples, where the tuple consists of:
      - Concept name,
      - Glyph name(s) for the concept,
      - Concept pronunciation,
      - Glyph pronunciation,
      - A single-element list of strokes for the proposed glyph.
  """

  # Load extension hypotheses from semantic and phonetic streams.
  semantics, phonetics, join = utils.load_and_prune_semantic_phonetic_jsonls()

  # Semantic-phonetic extensions.
  glyphs = []
  for concept in sorted(join):
    semantic_extension = semantics[concept].glyph_names
    phonetic_extension, concept_pron, glyph_pron = (
        phonetics[concept].glyph_names,
        phonetics[concept].concept_pron,
        phonetics[concept].glyph_pron,
    )
    joint_glyph = f"{semantic_extension} {phonetic_extension}"
    glyphs.append(
        (concept, joint_glyph, concept_pron, glyph_pron, [
            semantics[concept].strokes, phonetics[concept].strokes
        ])
    )
  logging.info("Created %d semantic-phonetic bi-glyphs ...", len(join))

  # Pure semantic extensions.
  semantic_extensions = set()
  for concept in sorted(semantics):
    if concept in join:
      continue
    glyph = semantics[concept].glyph_names
    strokes = semantics[concept].strokes
    glyphs.append((concept, glyph, "NA", "NA", [strokes]))
    semantic_extensions.add(concept)
  logging.info("Created %d pure semantic glyphs ...", len(semantic_extensions))

  # Pure phonetic extensions.
  phonetic_extensions = set()
  for concept in sorted(phonetics):
    if concept in join:
      continue
    glyph, concept_pron, glyph_pron, strokes = (
        phonetics[concept].glyph_names,
        phonetics[concept].concept_pron,
        phonetics[concept].glyph_pron,
        phonetics[concept].strokes,
    )
    glyphs.append((concept, glyph, concept_pron, glyph_pron, [strokes]))
    phonetic_extensions.add(concept)
  logging.info("Created %d pure phonetic glyphs ...", len(phonetic_extensions))

  return glyphs
