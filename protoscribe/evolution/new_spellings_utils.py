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

"""Various utilities supporting the spelling extension algorithms."""

import logging
import os
import xml.etree.ElementTree as ET

from absl import flags
import numpy as np
from protoscribe.evolution import confidence_pruning
from protoscribe.evolution import results_parser_and_pruner
from protoscribe.glyphs import make_text
from protoscribe.sketches.utils import stroke_utils

import glob
import os

ConfidencePruningMethod = confidence_pruning.Method
GlyphInfo = results_parser_and_pruner.GlyphInfo

_SEMANTIC_JSONL_FILE = flags.DEFINE_string(
    "semantic_jsonl_file",
    None,
    "Path to JSONL file extracted with `glyphs_from_jsonl` (for discrete "
    "glyphs) or `sketches_from_jsonl` (for sketches) from semantic model "
    "inference.",
)

_PHONETIC_JSONL_FILE = flags.DEFINE_string(
    "phonetic_jsonl_file",
    None,
    "Path to JSONL file extracted with `glyphs_from_jsonl` (for discrete "
    "glyphs) or `sketches_from_jsonl` (for sketches) from phonetic model "
    "inference.",
)

_MINIMUM_SEMANTIC_CONFIDENCE = flags.DEFINE_float(
    "minimum_semantic_confidence",
    confidence_pruning.DEFAULT_MIN_CONFIDENCE_THRESHOLD,
    "Minimum semantic confidence (for `THRESHOLD` pruning)."
)

_MINIMUM_PHONETIC_CONFIDENCE = flags.DEFINE_float(
    "minimum_phonetic_confidence",
    confidence_pruning.DEFAULT_MIN_CONFIDENCE_THRESHOLD,
    "Minimum phonetic confidence (for `THRESHOLD` pruning)."
)

_MINIMUM_SEMANTIC_PROB = flags.DEFINE_float(
    "minimum_semantic_prob",
    confidence_pruning.DEFAULT_MIN_PROBABILITY,
    "Minimum semantic probability (for `PROBABILITY` pruning)."
)

_MINIMUM_PHONETIC_PROB = flags.DEFINE_float(
    "minimum_phonetic_prob",
    confidence_pruning.DEFAULT_MIN_PROBABILITY,
    "Minimum phonetic probability (for `PROBABILITY` pruning)."
)

_SEMANTIC_TOP_K = flags.DEFINE_integer(
    "semantic_top_k",
    confidence_pruning.DEFAULT_TOP_K,
    "Top K semantic results to retain (for `TOP_K` pruning)."
)

_PHONETIC_TOP_K = flags.DEFINE_integer(
    "phonetic_top_k",
    confidence_pruning.DEFAULT_TOP_K,
    "Top K phonetic results to retain (for `TOP_K` pruning)."
)

_SEMANTIC_TOP_PERCENTAGE = flags.DEFINE_integer(
    "semantic_top_percentage",
    confidence_pruning.DEFAULT_TOP_PERCENTAGE,
    "Top K percents of the semantic results to retain (for `TOP_PERCENTAGE` "
    "pruning)."
)

_PHONETIC_TOP_PERCENTAGE = flags.DEFINE_integer(
    "phonetic_top_percentage",
    confidence_pruning.DEFAULT_TOP_PERCENTAGE,
    "Top K percents of the phonetic results to retain (for `TOP_PERCENTAGE` "
    "pruning)."
)

_SEMANTIC_TOP_P = flags.DEFINE_float(
    "semantic_top_p",
    confidence_pruning.DEFAULT_MAX_CUMULATIVE_PROBABILITY,
    "The maximum cumulative probability for `TOP_P` nucleus pruning."
)

_PHONETIC_TOP_P = flags.DEFINE_float(
    "phonetic_top_p",
    confidence_pruning.DEFAULT_MAX_CUMULATIVE_PROBABILITY,
    "The maximum cumulative probability for `TOP_P` nucleus pruning."
)

_PRUNING_METHOD = flags.DEFINE_enum_class(
    "pruning_method",
    ConfidencePruningMethod.THRESHOLD,
    ConfidencePruningMethod,
    "Method for pruning the results during parsing."
)

_PRUNING_USE_SOFTMAX = flags.DEFINE_boolean(
    "pruning_use_softmax", False,
    "When pruning using the probability-based methods, use `softmax` when "
    "converting the confidences to distribution. Otherwise use regular "
    "normalization (dividing by the total sum)."
)


def rejigger_categories(
    concepts_with_new_spellings: set[str],
    administrative_categories_path: str,
    non_administrative_categories_path: str,
    output_dir: str,
) -> None:
  """Moves categories that have new spellings to administrative categories.

  Args:
    concepts_with_new_spellings: A set of categories.
    administrative_categories_path: Path to administrative categories.
    non_administrative_categories_path: Path to non-administrative categories.
    output_dir: Output directory to place new files.
  """
  logging.info(
      "Concepts with new spellings: %d", len(concepts_with_new_spellings)
  )
  with open(administrative_categories_path) as s:
    administrative_categories = [l.strip() for l in s]
  with open(non_administrative_categories_path) as s:
    non_administrative_categories = [l.strip() for l in s]
  new_non_administrative_categories = []
  for concept in non_administrative_categories:
    if concept in concepts_with_new_spellings:
      administrative_categories.append(concept)
    else:
      new_non_administrative_categories.append(concept)
  non_administrative_categories = new_non_administrative_categories
  with open(
      os.path.join(output_dir, "administrative_categories.txt"), mode="wt"
  ) as s:
    for category in administrative_categories:
      s.write(f"{category}\n")
  with open(
      os.path.join(output_dir, "non_administrative_categories.txt"), mode="wt"
  ) as s:
    for category in non_administrative_categories:
      s.write(f"{category}\n")


def save_glyph_graphics(
    glyphs: list[tuple[str, str, str, str, list[np.ndarray]]],
    output_glyph_graphics_dir: str
) -> None:
  """Saves glyph extensions in SVG format.

  Args:
     glyphs: Glyph information tuples. Each tuple consists of concept name, its
       spelling, the spelling, its pronunciation and a list of strokes per
       glyph. The list can contain one (semantic- or phonetic-only extension) or
       two (biglyph extension) stroke sequences.
     output_glyph_graphics_dir: Path to output glyphs directory.
  """
  logging.info(
      "Saving %d glyph SVGs to %s ...",
      len(glyphs),
      output_glyph_graphics_dir,
  )
  for concept, spelling, _, _, glyph_strokes in glyphs:
    glyph_names = spelling.split()
    concept = concept.split("_")[0]
    path = os.path.join(output_glyph_graphics_dir, f"{concept}.svg")
    logging.info("Saving %s with #%d glyph(s) ...", path, len(glyph_strokes))
    if len(glyph_strokes) == 1:
      stroke_utils.stroke3_strokes_to_svg_file(glyph_strokes[0], path)
      continue

    # For biglyphs, convert individual stroke sequences and concatenate.
    xml_semantic = stroke_utils.stroke3_strokes_to_svg(glyph_strokes[0])
    xml_phonetic = stroke_utils.stroke3_strokes_to_svg(glyph_strokes[1])
    combined_svg, _, _ = make_text.concat_xml_svgs(
        [
            ET.ElementTree(ET.fromstring(xml_semantic)),
            ET.ElementTree(ET.fromstring(xml_phonetic)),
        ],
        glyph_names
    )
    with open(path, mode="wt") as f:
      combined_svg.write(f, encoding="unicode")


def load_and_prune_semantic_phonetic_jsonls() -> tuple[
    dict[str, GlyphInfo], dict[str, GlyphInfo], set[str]
]:
  """Loads and prunes results for phonetics and semantics.

  Semantic stream may be represented by the output of semantic or vision models,
  while phonetic stream may be represented by phonological or speech model
  results. This method uses current values of the command-line flags to retrieve
  the necessary arguments to lower-level APIs.

  Returns:
    A tuple containing:
      - Pruned predictions from semantic stream,
      - Pruned predictions from phonetic stream,
      - Join between the two.
  """

  if not _SEMANTIC_JSONL_FILE.value:
    raise ValueError(
        "Flag --semantic_jsonl_file must have a value other than None."
    )
  semantics = results_parser_and_pruner.load_and_prune_jsonl(
      _SEMANTIC_JSONL_FILE.value,
      pruning_method=_PRUNING_METHOD.value,
      use_softmax=_PRUNING_USE_SOFTMAX.value,
      min_confidence_threshold=_MINIMUM_SEMANTIC_CONFIDENCE.value,
      min_prob=_MINIMUM_SEMANTIC_PROB.value,
      top_k=_SEMANTIC_TOP_K.value,
      top_percentage=_SEMANTIC_TOP_PERCENTAGE.value,
      max_cumulative_prob=_SEMANTIC_TOP_P.value,
  )

  if not _PHONETIC_JSONL_FILE.value:
    raise ValueError(
        "Flag --phonetic_jsonl_file must have a value other than None."
    )
  phonetics = results_parser_and_pruner.load_and_prune_jsonl(
      _PHONETIC_JSONL_FILE.value,
      pruning_method=_PRUNING_METHOD.value,
      use_softmax=_PRUNING_USE_SOFTMAX.value,
      min_confidence_threshold=_MINIMUM_PHONETIC_CONFIDENCE.value,
      min_prob=_MINIMUM_PHONETIC_PROB.value,
      top_k=_PHONETIC_TOP_K.value,
      top_percentage=_PHONETIC_TOP_PERCENTAGE.value,
      max_cumulative_prob=_PHONETIC_TOP_P.value,
  )

  join = set(semantics.keys()).intersection(set(phonetics.keys()))
  return semantics, phonetics, join
