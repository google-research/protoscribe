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

"""APIs for handling concept glosses."""

import collections
import json

from absl import flags

# Internal resources dependency

_CONCEPT_GLOSSES_FILE = flags.DEFINE_string(
    "concept_glosses_file",
    (
        "protoscribe/data/concepts/"
        "concept_descriptions.json"
    ),
    (
        "A JSON file containing the categories, concepts, MIDs and the "
        "corresponding glosses."
    ),
)

# This also defines the search order.
_CATEGORIES = [
    "administrative_categories",
    "non_administrative_categories",
]


def read_glosses() -> dict[str, dict[str, str]]:
  """Reads glosses repository returning concept-to-gloss mapping."""
  with open(_CONCEPT_GLOSSES_FILE.value, mode="rt", encoding="utf-8") as f:
    data = json.load(f)
  all_features = data["glyph_glosses"]
  all_glosses = collections.defaultdict(dict)
  for features in all_features:
    category = features["category"]
    concept = features["glyph_name"]
    all_glosses[category][concept] = features["gloss"]
  return all_glosses


def find_gloss(
    concept_name: str,
    glosses: dict[str, dict[str, str]],
    restrict_to_category: str | None = None
) -> str | None:
  """Find the gloss for a given concept. Returns empty string if not found."""
  # TODO: We shouldn't be stripping POS tags.
  concept_name = concept_name.split("_")[0]
  categories = [restrict_to_category] if restrict_to_category else _CATEGORIES
  for category in categories:
    if category not in glosses:
      raise ValueError(f"Invalid category: {category}")
    if concept_name in glosses[category]:
      return glosses[category][concept_name]
  # Nothing found.
  return None
