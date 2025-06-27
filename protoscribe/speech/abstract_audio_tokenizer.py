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

"""Interface for all the audio tokenizers."""

import abc

import tensorflow as tf


class AudioTokenizer(abc.ABC):
  """An abstract base class for generating token embeddings from audio."""

  def __init__(
      self,
      model_config_name_or_path: str,
      sample_rate: int,
      has_quantizer: bool = False
  ):
    """Initializes the tokenizer.

    Args:
      model_config_name_or_path: Name of the model configuration or path of
        the model. This is implementation-specific.
      sample_rate: Sampling rate in Hz.
      has_quantizer: True if the model has quantizer. In this case it should be
        possible to retrieve discrete tokens in addition to the embeddings.
    """
    del model_config_name_or_path
    del sample_rate
    del has_quantizer

  @abc.abstractmethod
  def get_embeddings(self, audio: tf.Tensor) -> tf.Tensor:
    """Turns the audio into embeddings.

    Args:
      audio: A floating-point tensor [L_a], where L_a is the number of
        audio samples.

    Returns:
      The floating-point embeddings tensor [L_t, D], where L_t is the number
      of tokens in a sequence and D is the feature dimension.
    """
    ...

  @abc.abstractmethod
  def get_tokens(self, audio: tf.Tensor) -> tf.Tensor:
    """Converts the audio to discrete tokens.

    Args:
      audio: A floating-point tensor [L_a], where L_a is the number of
        audio samples.

    Returns:
      The integer tokens tensor [L_t], where L_t is the number of tokens in
      a sequence.
    """
    ...
