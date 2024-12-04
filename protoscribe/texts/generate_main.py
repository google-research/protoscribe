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

"""Generates a new language, saving out the parameters.

One can then use this to generate texts in the language.
"""

from absl import app
from absl import flags
from protoscribe.texts import generate_lib

_GENERATE_LEXICAL_RESOURCES = flags.DEFINE_bool(
    "generate_lexical_resources", False,
    "Generates lexicon resources rather than texts.",
)


def main(unused_argv):
  generator = generate_lib.TextGenerator()
  if _GENERATE_LEXICAL_RESOURCES.value:
    generator.generate_lexical_resources()
  else:
    generator.generate_texts()


if __name__ == "__main__":
  flags.mark_flag_as_required("affix_lexicon")
  flags.mark_flag_as_required("concepts")
  flags.mark_flag_as_required("main_lexicon")
  flags.mark_flag_as_required("morphology_params")
  flags.mark_flag_as_required("number_lexicon")
  app.run(main)
