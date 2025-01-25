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

"""Computes sorted distance vectors for all embeddings."""

from absl import app
from absl import flags
from protoscribe.language.embeddings import embedder
from protoscribe.language.phonology import phoible_segments
from protoscribe.language.phonology import phonetic_embeddings

_INPUT_EMBEDDINGS_FILE = flags.DEFINE_string(
    "input_embeddings_file", None,
    "Path to the input embeddings file in TSV format.",
    required=True
)

_OUTPUT_DISTANCES_FILE = flags.DEFINE_string(
    "output_distances_file", None,
    "Path to output distances.",
    required=True
)

_PHOIBLE_PATH = flags.DEFINE_string(
    "phoible_path", phoible_segments.PHOIBLE, "Path to PHOIBLE segments."
)

_PHOIBLE_FEATURES_PATH = flags.DEFINE_string(
    "phoible_features_path",
    phoible_segments.PHOIBLE_FEATURES,
    "Path to PHOIBLE features.",
)


def main(unused_argv):
  phoible = phoible_segments.PhoibleSegments(
      path=_PHOIBLE_PATH.value,
      features_path=_PHOIBLE_FEATURES_PATH.value,
  )
  embeddings = phonetic_embeddings.PhoneticEmbeddings(
      phoible_seg=phoible,
      embedding_len=embedder.DEFAULT_EMBEDDING_DIM,
  )
  embeddings.read_embeddings(_INPUT_EMBEDDINGS_FILE.value)
  embeddings.dump_all_distances(_OUTPUT_DISTANCES_FILE.value)


if __name__ == "__main__":
  app.run(main)
