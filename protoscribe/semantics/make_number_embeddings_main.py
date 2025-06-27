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

r"""Makes number embeddings with a given dimension.

Example:
--------
python protoscribe/semantics/make_number_embeddings_main.py \
  --embedding_dim 50 --max_number 10 --output_file /tmp/num.txt
"""

from collections.abc import Sequence
import logging

from absl import app
from absl import flags
import numpy as np

import glob
import os

_EMBEDDING_DIM = flags.DEFINE_integer(
    "embedding_dim", None,
    "Dimension of the embedding vectors.",
    required=True
)

_MAX_NUMBER = flags.DEFINE_integer(
    "max_number", 100,
    "Maximum number M to generate in [0, M)."
)

_OUTPUT_FILE = flags.DEFINE_string(
    "output_file", None,
    "Output embeddings file for numbers in text format.",
    required=True
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  if _EMBEDDING_DIM.value < 3:
    raise ValueError(f"Invalid dimension: {_EMBEDDING_DIM.value}")

  # Generate random positive vectors drawn from standard exponential
  # distribution scale them up and re-normalize.
  rng = np.random.default_rng()
  numbers = []
  for n in range(_MAX_NUMBER.value):
    vec = rng.standard_exponential(
        size=(_EMBEDDING_DIM.value), dtype=np.float64
    ) * (_MAX_NUMBER.value + n + 1)
    vec /= np.linalg.norm(vec)
    numbers.append(vec)

  logging.info("Writing to %s ...", _OUTPUT_FILE.value)
  with open(_OUTPUT_FILE.value, mode="w") as f:
    for n, vec in enumerate(numbers):
      vec_str = " ".join([str(v) for v in vec.tolist()])
      f.write(f"{n}_NUM {vec_str}\n")


if __name__ == "__main__":
  app.run(main)
