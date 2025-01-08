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

"""Stage-specific helper for postprocessing discrete glyph inference results.

This tool is intended to be used on outputs of discrete glyph predictor.
"""

from collections.abc import Sequence
import logging
import os

from absl import app
from absl import flags
from protoscribe.evolution.stages import common_flags
from protoscribe.utils import file_utils
from protoscribe.utils import subprocess_utils

_EXPERIMENT_NAME = flags.DEFINE_string(
    "experiment_name", None,
    "An experiment name which will define the directory in which the "
    "evolving system data is placed.",
    required=True
)

_SEMANTICS_XID = flags.DEFINE_string(
    "semantics_xid", None,
    "XManager job ID for the inference run with semantics model for this round."
)

_PHONETICS_XID = flags.DEFINE_string(
    "phonetics_xid", None,
    "XManager job ID for the inference run with phonetics model for this round."
)

_JSONL_FILE_NAME_GLYPHS = flags.DEFINE_string(
    "jsonl_file_name_glyphs", None,
    "File name used for storing the outputs of glyph inference.",
    required=True
)

# Actual inference post-processing tool.
_GLYPHS_TOOL = (
    "protoscribe/sketches/inference/glyphs_from_jsonl"
)

# Discrete glyph prediction mode.
_MODE = "glyph"


def _setup_inference_directories(
    round_data_dir: str,
    experiment_name: str,
    experiment_id: str | None
) -> str:
  """Sets up the directory for storing the post-processed inference outputs.

  Args:
    round_data_dir: Data directory for this round.
    experiment_name: Symbol name for the experiment.
    experiment_id: XManager Job ID (integer string).

  Returns:
    Output directory where postprocessed results will be stored.

  Raises:
    ValueError if output directory could not be determined.
  """
  output_dir = os.path.join(
      round_data_dir, f"{experiment_name}:inference_outputs"
  )
  if experiment_id:
    output_dir = os.path.join(output_dir, experiment_id)
  else:
    experiment_dirs = file_utils.list_subdirs(output_dir)
    if not experiment_dirs:
      raise ValueError(
          f"No inference experiment directories found under {output_dir}!"
      )
    output_dir = experiment_dirs[-1]

  logging.info("Reading and writing output data to %s ...", output_dir)
  return output_dir


def _glyphs_for_model_type(
    round_data_dir: str, model_type: str, experiment_id: str
) -> None:
  """Run glyph extractions from the inference run for a given model type.

  Args:
    round_data_dir: Data directory for this round.
    model_type: Type of the model.
    experiment_id: XManager job ID.
  """
  round_id = common_flags.ROUND.value
  experiment_name = (
      f"{_EXPERIMENT_NAME.value}:{round_id}:{_MODE}_{model_type}"
  )
  output_dir = _setup_inference_directories(
      round_data_dir=round_data_dir,
      experiment_name=experiment_name,
      experiment_id=experiment_id
  )
  jsonl_file = os.path.join(output_dir, _JSONL_FILE_NAME_GLYPHS.value)
  subprocess_utils.run_subprocess(
      _GLYPHS_TOOL,
      args=[
          "--dataset_dir", round_data_dir,
          "--input_jsonl_file", jsonl_file,
          "--output_tsv_file", f"{output_dir}/results.tsv",
          "--output_file_for_scorer", f"{output_dir}/results.jsonl",
          "--ignore_errors", True,
      ]
  )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  round_data_dir = common_flags.round_data_dir()
  logging.info("Using data location: %s", round_data_dir)

  # Post-process inference results for the semantic stream.
  _glyphs_for_model_type(
      round_data_dir=round_data_dir,
      model_type=common_flags.SEMANTIC_MODEL.value,
      experiment_id=_SEMANTICS_XID.value
  )
  # Post-process inference results for the phonetic stream.
  _glyphs_for_model_type(
      round_data_dir=round_data_dir,
      model_type=common_flags.PHONETIC_MODEL.value,
      experiment_id=_PHONETICS_XID.value
  )


if __name__ == "__main__":
  app.run(main)
