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

"""Audio tokenization interfaces."""

from protoscribe.speech import abstract_audio_tokenizer

AudioTokenizer = abstract_audio_tokenizer.AudioTokenizer


def get_tokenizer(
    model_config_name_or_path: str,
    sample_rate: int,
    has_quantizer: bool = False
) -> AudioTokenizer | None:
  """Manufactures an instance of audio tokenizer.

  Args:
    model_config_name_or_path: Name of the model configuration or path of
      the model. This is implementation-specific.
    sample_rate: Sampling rate in Hz.
      has_quantizer: True if the model has quantizer. In this case it should be
        possible to retrieve discrete tokens in addition to the embeddings.

  Returns:
    Audio tokenizer instance.
  """
  # No audio tokenizers have been implemented yet.
  return None
