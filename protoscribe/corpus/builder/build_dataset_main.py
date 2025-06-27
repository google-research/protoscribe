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

"""Helper tool for dataset building.

Assembles the core language resources followed by the generation of dataset
splits in `tf.train.Example` format for training, validation and test. More
specifically, following artifacts for the generation of accounting texts are
created:

  - Prerequisites for accounting document generation including
    pronunciation lexicons (created by text generator).
  - Validation and test text data in TSV format (created by text generator).
  - The actual data for model training (train, validation and test splits) and
    inference (created by corpus builder Beam pipeline).

The resulting data residing in the specified output directory is used to train
the models and perform the inference.

This is essentially a shell script written in Python primarily to benefit from
absl flags support. To keep things simple this tool has minimal code
dependencies, we spawn individual Protoscribe builders as separate processes
rather than depending on individual libraries.
"""

from collections.abc import Sequence

from absl import app
from protoscribe.corpus.builder import build_dataset


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  build_dataset.build_dataset()


if __name__ == "__main__":
  app.run(main)
