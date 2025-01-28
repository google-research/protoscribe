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

"""Stage wrapper over basic spelling extension algorithm."""

from collections.abc import Sequence
import logging
import os
import tempfile

from absl import app
from absl import flags
from protoscribe.evolution import new_spellings_utils  # pylint: disable=unused-import Import flags.
from protoscribe.evolution.stages import common_flags
from protoscribe.evolution.stages import utils
from protoscribe.utils import file_utils
from protoscribe.utils import subprocess_utils

import glob
import os

_MODE = flags.DEFINE_enum(
    "mode", "sketch-token",
    [
        "sketch-token",
        "sketch-token-and-glyph",
    ],
    "Type of sketch mdoel. Can be 'sketch-token' for pure sketch generation or "
    "'sketch-token-and-glyph' for combined glyph and sletch prediction. "
    "This is a prefix part of the model configuration in 'configs' directory."
)

_EXPERIMENT_NAME = flags.DEFINE_string(
    "experiment_name", None,
    "An experiment name which will define the directory in which the "
    "evolving system data is placed.",
    required=True
)

FLAGS = flags.FLAGS

# Actual spelling extension tool.
_NEW_SPELLINGS_TOOL = (
    "protoscribe/evolution/new_spellings_basic"
)


def _results_jsonl_for_model_type(
    round_data_dir: str, model_type: str, experiment_id: str
) -> str:
  """Returns results JSONL for the given model type.

  Args:
    round_data_dir: Data directory for this round.
    model_type: Type of the model.
    experiment_id: XManager job ID.
  """

  # Figure out directory for the outputs.
  round_id = common_flags.ROUND.value
  experiment_name = (
      f"{_EXPERIMENT_NAME.value}:{round_id}:{_MODE.value}_{model_type}"
  )
  if _MODE.value == "sketch-token":
    experiment_name = f"{experiment_name}:reco"
  output_dir = utils.setup_inference_directories(
      round_data_dir=round_data_dir,
      experiment_name=experiment_name,
      experiment_id=experiment_id
  )
  jsonl_path = os.path.join(output_dir, "results.jsonl")
  logging.info("JSONL results in %s ...", jsonl_path)
  return jsonl_path


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  round_data_dir = common_flags.round_data_dir()
  logging.info("Using data location: %s", round_data_dir)

  # Find previous spellings extensions, if any.
  prev_spellings_file = None
  if common_flags.ROUND.value > 0:
    prev_spellings_file = os.path.join(
        common_flags.previous_data_dir(), "inference_extensions/spellings.tsv"
    )
    if not os.path.exists(prev_spellings_file):
      raise ValueError(
          f"Previous spelling extensions {prev_spellings_file} not found!"
      )

  # Get the paths to JSONL files containing the results from the current
  # round.
  semantics_results_jsonl = _results_jsonl_for_model_type(
      round_data_dir=round_data_dir,
      model_type=common_flags.SEMANTIC_MODEL.value,
      experiment_id=common_flags.SEMANTICS_XID.value
  )
  phonetics_results_jsonl = _results_jsonl_for_model_type(
      round_data_dir=round_data_dir,
      model_type=common_flags.PHONETIC_MODEL.value,
      experiment_id=common_flags.PHONETICS_XID.value
  )

  # Setup command-line flags to call the actual spellings extension tool.
  admin_categories = f"{round_data_dir}/administrative_categories.txt"
  non_admin_categories = f"{round_data_dir}/non_administrative_categories.txt"
  args = [
      "--data_location", round_data_dir,
      "--semantic_jsonl_file", semantics_results_jsonl,
      "--phonetic_jsonl_file", phonetics_results_jsonl,
      "--administrative_categories", admin_categories,
      "--non_administrative_categories", non_admin_categories,
      # TODO: The plumbing for flags below is not great. Maybe refactor
      # using protocol buffers.
      "--pruning_method", FLAGS.pruning_method,
      "--minimum_semantic_confidence", FLAGS.minimum_semantic_confidence,
      "--minimum_phonetic_confidence", FLAGS.minimum_phonetic_confidence,
      "--minimum_semantic_prob", FLAGS.minimum_semantic_prob,
      "--minimum_phonetic_prob", FLAGS.minimum_phonetic_prob,
      "--semantic_top_k", FLAGS.semantic_top_k,
      "--phonetic_top_k", FLAGS.phonetic_top_k,
      "--semantic_top_percentage", FLAGS.semantic_top_percentage,
      "--phonetic_top_percentage", FLAGS.phonetic_top_percentage,
      "--semantic_top_p", FLAGS.semantic_top_p,
      "--phonetic_top_p", FLAGS.phonetic_top_p,
  ]
  if common_flags.ROUND.value > 0:
    args.extend(["--previous_spellings", prev_spellings_file])

  # For sketches, we also need to set up the directory for outputing the
  # actual glyphs as SVGs.
  svg_temp_dir = tempfile.TemporaryDirectory()
  if _MODE.value == "sketch-token":
    output_glyph_graphics_dir = os.path.join(
        round_data_dir, "glyph_extensions_svg"
    )
    if not os.path.exists(output_glyph_graphics_dir):
      os.makedirs(output_glyph_graphics_dir)
    args.extend([
        "--output_glyph_graphics_dir", svg_temp_dir.name
    ])

  # Run the algorithm.
  subprocess_utils.run_subprocess(_NEW_SPELLINGS_TOOL, args=args)

  # Copy the extensions from temp directory.
  if _MODE.value == "sketch-token":
    logging.info("Copying glyph SVGs to %s ...", output_glyph_graphics_dir)
    file_utils.copy_dir(svg_temp_dir.name, output_glyph_graphics_dir)
  svg_temp_dir.cleanup()


if __name__ == "__main__":
  app.run(main)
