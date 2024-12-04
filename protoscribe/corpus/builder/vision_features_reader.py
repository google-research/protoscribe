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

"""Concrete readers for vision features."""

from protoscribe.corpus.builder import abstract_vision_features_reader as base_reader
from protoscribe.corpus.builder import prepare_utils as utils


class DummyVisionFeaturesReader(base_reader.AbstractVisionFeaturesReader):
  """Vision features reader returning empty features."""

  def _read_features(
      self,
      features_dir: str | None = None,
      concept_types: list[bool] | None = None,
      image_models: list[str] | None = None,
      feature_versions: list[str] | None = None,
  ) -> base_reader.VisionFeaturesRepo:
    """Populates the vision feature mapping.

    Args:
      features_dir: Directory hosting the features.
      concept_types: A list of booleans indicating whether the repository is for
        administrative (seen) or non-administrative (unseen) concept categories.
        For example, the default of [True, False] means including both seen and
        unseen concepts.
      image_models: A list of strings identifying model names (types).
      feature_versions: A list of strings identifying feature versions.

    Returns:
      The vision feature repository, which is a mapping defined as follows:
        Concept -> list[tuple[Model Type, Feature Version, Features],
    """
    return {}

  def features_for_concept(self, concept: str) -> dict[str, utils.Feature]:
    """Retrieves features for the specified concept.

    Args:
      concept: Concept name for which features need to retrieved.

    Returns:
      A dictionary between feature names and TensorFlow features
      of type `tf.train.Feature`.
    """
    return {}


# Currently no vision features are supported.
VisionFeaturesReader = DummyVisionFeaturesReader
