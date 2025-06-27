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

r"""Helper script for generating HTML page with results.

This tool is only relevant for sketch generation.

Example:
--------
  BASE_DIR=...
  python protoscribe/evolution/stages/make_html_main.py \
    --default_base_dir="${BASE_DIR} \
    --experiment_name="concept_to_glyph" \
    --round=0 \
    --output_html_dir=/tmp \
    --logtostderr
"""

from collections.abc import Sequence
import logging
import os

from absl import app
from absl import flags
from protoscribe.evolution import make_html
from protoscribe.evolution.stages import common_flags

_EXPERIMENT_NAME = flags.DEFINE_string(
    "experiment_name", None,
    "An experiment name which will define the directory in which the "
    "evolving system data is placed.",
    required=True
)

FLAGS = flags.FLAGS


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Infer the locations of source SVGs and glyph extension names.
  base_dir = common_flags.experiment_dir()
  svg_src_dir = os.path.join(base_dir, "glyph_extensions_svg")
  round_data_dir = common_flags.round_data_dir()
  logging.info("Using data location: %s", round_data_dir)
  extensions_tsv_file = os.path.join(
      round_data_dir, "inference_extensions", "extensions.tsv"
  )

  # Override the source flags and invoke the HTML builder.
  FLAGS.extensions_file = extensions_tsv_file
  FLAGS.svg_src_dir = svg_src_dir
  make_html.make_html()


if __name__ == "__main__":
  app.run(main)
