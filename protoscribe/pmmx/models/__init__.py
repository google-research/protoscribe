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

"""Models API for PMMX.

If you add a new model here, it will be visible to the rest of the world as
simply `pmmx.models.MyModelName`, therefore when adding a new model, choose
a name that isn't already in use.

This is meant to be a central repository of different model flavors, which
roughly corresponds to the `p5x/models.py`, but with separate files per
model to keep the separation clean.
"""

from .multimodal_encoder_decoder import MultimodalEncoderDecoderModel
