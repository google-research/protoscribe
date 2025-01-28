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

"""A stage script responsible for building dataset for a particular round.

Typically there is one corpus preparation stage for each round. Between the
rounds the setup needs to be different. When the initial corpus is created
in the first round, we need to generate the language. The subsequent rounds
need to use this language from the first round unchanged. At the same time,
each round needs to use the updated set of categories and glyphs from the
previous round of evolution.
"""

from collections.abc import Sequence
import logging
import os
from typing import Any

from absl import app
from absl import flags
from protoscribe.corpus.builder import build_dataset as builder_lib
from protoscribe.evolution.stages import common_flags
from protoscribe.utils import file_utils

import glob
import os

FLAGS = flags.FLAGS


def _setup_builder(round_data_dir: str) -> list[tuple[str, Any]]:
  """Sets up builder environment and updates the relevant flags.

  Args:
    round_data_dir: Data directory for the current round.

  Returns:
    A list of flags for the given round necessary for running the builder.
    These are categories flags contain flags to pick up the administrative and
    non-administrative categories lists, and the spellings created from the
    previous round for round > 0.
  """
  categories_flags = []

  # Figure out the locations for the data and perform the necessary sanity
  # checks.
  if os.path.isdir(round_data_dir):
    raise ValueError(
        f"Directory `{round_data_dir}` already exists: Cowardly unwilling to "
        "overwrite previous experiment."
    )
  round_id = common_flags.ROUND.value
  if round_id > 0:
    previous_data_dir = common_flags.previous_data_dir()
    if not os.path.isdir(previous_data_dir):
      raise ValueError(
          f"Directory `{previous_data_dir}` does not exist: did you run the "
          f"previous round {round_id - 1} needed for round {round_id}?"
      )

    # Next we check to see if we have correctly generated spelling extensions in
    # output directory `inference_extensions` on the previous generation's run.
    extensions_dir = f"{previous_data_dir}/inference_extensions"
    if not os.path.isdir(extensions_dir):
      # TODO: Revisit this when we get to Round 1, since actually the
      # *language* does not change. The only thing that changes is that more of
      # these will acquire spellings, meaning that we need to update the glyphs,
      # plus what gets put into the training versus held-out data.
      raise ValueError(
          f"Directory `{extensions_dir} does not exist: did you run the "
          f"previous round {round_id - 1}  needed for round {round_id}?"
      )

    # Prepare data for new round: make new round directory and copy over the
    # language definitions from the previous round.
    logging.info("Making %s ...", round_data_dir)
    language_dir = os.path.join(round_data_dir, "language")
    os.makedirs(language_dir)
    file_utils.copy_dir(
        os.path.join(previous_data_dir, "language"), language_dir
    )

    # Pick up categories and spellings.
    categories_flags.extend([
        (
            "administrative_categories", os.path.join(
                extensions_dir, "administrative_categories.txt"
            )
        ),
        (
            "non_administrative_categories", os.path.join(
                extensions_dir, "non_administrative_categories.txt"
            )
        ),
        ("concept_spellings", os.path.join(extensions_dir, "spellings.tsv")),
        ("prefer_concept_svg", "true"),
    ])

    # Check for directory containing SVG glyph extensions.
    extensions_svg_dir = os.path.join(round_data_dir, "glyph_extensions_svg")
    if os.path.isdir(extensions_svg_dir):
      categories_flags.append(
          ("extension_glyphs_svg_dir", extensions_svg_dir),
      )

  # At this stage it is safe to do this again.
  if not os.path.isdir(round_data_dir):
    os.makedirs(round_data_dir)
  logging.info(
      "Created `%s` for outputs for round %d.", round_data_dir, round_id
  )

  return categories_flags


def _run_builder(app_flags: list[tuple[str, Any]]) -> None:
  """Invokes dataset builder.

  Args:
    app_flags: A list of pairs mapping flag names to the respective values.
      These are the flags filled in by this script. Any other flags passed to
      this script by the caller are already parsed.
  """
  logging.info("Final local flags: %s", app_flags)
  for flag_name, flag_value in app_flags:
    FLAGS[flag_name].parse(flag_value)
  builder_lib.build_dataset()


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Set up the environment and prepare the flags. Generate language for the
  # first round only.
  round_data_dir = common_flags.round_data_dir()
  categories_flags = _setup_builder(round_data_dir)
  generate_language = common_flags.ROUND.value == 0
  logging.info("Categories and spellings flags: %s", categories_flags)

  # Uses most of the defaults set in `builder_lib`, the other flags are passed
  # to this binary directly by the calling scripts.
  logging.info("Done with setup. Running dataset builder ...")
  app_flags = [
      ("generate_language", generate_language),
      ("output_dir", round_data_dir),
      ("probability_of_supercategory_glyph", 0.0),
      ("logtostderr", True),
  ]
  if categories_flags:
    app_flags.extend(categories_flags)
  _run_builder(app_flags=app_flags)


if __name__ == "__main__":
  # Temporarily set the output directory flag required by the vanilla builder
  # to some temporary value. This is going to be overwritten programmatically
  # by the implementation above.
  FLAGS.output_dir = "tmp"

  app.run(main)
