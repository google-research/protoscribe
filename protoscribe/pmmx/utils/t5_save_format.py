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

"""Implements the T5 save format API.
"""
from typing import Any, Union

from flax.core import frozen_dict

from flaxformer.architectures.t5 import t5_architecture


def load(
    model: Union[t5_architecture.EncoderDecoder, t5_architecture.DecoderOnly],
    save_format_params: Any,
):
  """Loads a T5 model.

  Args:
    model: Model class.
    save_format_params: Dictionary of save format parameters.

  Returns:
    New parameters in the same pytree structure as `model`.
  """
  del model  # unused
  return frozen_dict.freeze(save_format_params)
