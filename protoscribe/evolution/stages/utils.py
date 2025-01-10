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

"""Miscellaneous stage-specific helpers."""

import logging
import os

from protoscribe.utils import file_utils


def setup_inference_directories(
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
    # TODO: We should probably be either returning the latest
    # created directory *or* not allow multiple directories at all.
    output_dir = experiment_dirs[-1]

  logging.info("Reading and writing output data to %s ...", output_dir)
  return output_dir
