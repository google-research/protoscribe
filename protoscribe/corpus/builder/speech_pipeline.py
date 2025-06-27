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

"""Speech pipeline details."""

import random

from protoscribe.corpus.builder.abstract_speech_pipeline import AbstractSpeechPipeline

import tensorflow as tf

DUMMY_SAMPLE_RATE_HZ = 16_000
DUMMY_NUM_SAMPLES = 10


class DummySpeechPipeline(AbstractSpeechPipeline):
  """This implementation passes through the document unchanched."""

  def setup(self) -> None:
    """Called to prepare an instance for processing bundles of elements."""
    ...

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
    del sampa

    input_example.features.feature["audio/sample_rate"].int64_list.value.append(
        DUMMY_SAMPLE_RATE_HZ
    )
    input_example.features.feature["audio/waveform"].float_list.value.extend(
        [random.random() for _ in range(DUMMY_NUM_SAMPLES)]
    )
    return input_example


# Currently no speech generation pipelines are supported. Phoneme-based speech
# synthesizer is going to be added in due course.
SpeechPipeline = DummySpeechPipeline
