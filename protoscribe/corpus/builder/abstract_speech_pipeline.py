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

"""Helpers for generating the speech features."""

import abc
import logging

import tensorflow as tf


class AbstractSpeechPipeline(abc.ABC):
  """An abstract base class for pipelines for filling in speech features."""

  @abc.abstractmethod
  def setup(self) -> None:
    """Initializes the pipeline."""
    ...

  def process_example(
      self, doc_id: int, input_example: tf.train.Example
  ) -> tuple[int, tf.train.Example]:
    """Updates the example with speech features.

    Args:
      doc_id: Unique integer document ID.
      input_example: Input features for a document in `tf.train.Example` format.

    Returns:
      A pair consisting of `doc_id` and document annotated with additional
      speech features.

    Raises:
      ValueError if input document does not have the necessary features for
      speech processing or some other error was encountered.
    """

    tfe = tf.train.Example()
    tfe.CopyFrom(input_example)

    sampa_key = "text/sampa"
    if sampa_key not in tfe.features.feature:
      raise ValueError(f"{doc_id}: Phonemes not found in TF example!")
    sampa = tfe.features.feature[sampa_key].bytes_list.value[0].decode("utf-8")

    try:
      tfe = self._generate_speech_features(tfe, sampa)
    except Exception as e:  # pylint: disable=broad-except
      logging.exception(
          "[%d] '%s': Speech processing failed: %s", doc_id, sampa, str(e)
      )
      raise ValueError from e

    return doc_id, tfe

  @abc.abstractmethod
  def _generate_speech_features(
      self, input_example: tf.train.Example, sampa: str
  ) -> tf.train.Example:
    """Generates speech features and adds them to the document.

    Args:
      input_example: Input features for a document in `tf.train.Example` format.
      sampa: Space-separated pronunciation string in SAMPA format.

    Returns:
      Document annotated with additional speech features.
    """
    ...
