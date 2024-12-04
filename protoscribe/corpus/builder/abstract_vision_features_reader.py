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

"""Abstract reader for vision features."""

import abc

import numpy as np
import tensorflow as tf

# The following defines the mapping:
#  Concept -> list[tuple[Model Type, Feature Version, Features].
VisionFeaturesRepo = dict[str, list[tuple[str, str, np.ndarray]]]


class AbstractVisionFeaturesReader(abc.ABC):
  """Abstract interface for vision features readers."""

  repo: VisionFeaturesRepo | None = None

  def init(
      self,
      features_dir: str | None = None,
      concept_types: list[bool] | None = None,
      image_models: list[str] | None = None,
      feature_versions: list[str] | None = None,
  ) -> None:
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
      Vision features dictionary.
    """
    self.repo = self._read_features(
        features_dir=features_dir,
        concept_types=concept_types,
        image_models=image_models,
        feature_versions=feature_versions
    )

  @abc.abstractmethod
  def features_for_concept(self, concept: str) -> dict[str, tf.train.Feature]:
    """Retrieves features for the specified concept.

    Args:
      concept: Concept name for which features need to retrieved.

    Returns:
      A dictionary between feature names and TensorFlow features
      of type `tf.train.Feature`.
    """
    ...

  @abc.abstractmethod
  def _read_features(
      self,
      features_dir: str | None = None,
      concept_types: list[bool] | None = None,
      image_models: list[str] | None = None,
      feature_versions: list[str] | None = None,
  ) -> VisionFeaturesRepo:
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
    ...
