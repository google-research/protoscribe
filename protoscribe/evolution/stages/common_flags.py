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

"""Flags common to all the stages."""

import os

from absl import flags

DEFAULT_BASE_DIR = flags.DEFINE_string(
    "default_base_dir", None,
    "Default base directory.",
    required=True
)

ROUND = flags.DEFINE_integer(
    "round", 0,
    "Identifies which round of the experiment we are running. Note that to run "
    "round N, for N>0, round N-1 must have been run."
)

SEMANTIC_MODEL = flags.DEFINE_enum(
    "semantic_model", "concepts",
    [
        "concepts",
        "vision"
    ],
    "Type of the semantics model to use."
)

PHONETIC_MODEL = flags.DEFINE_enum(
    "phonetic_model", "phonemes",
    [
        "phonemes",
        "logmel-spectrum",
    ],
    "Type of the phonetic model to use."
)


def experiment_dir() -> str:
  """Returns fully-qualified experiment directory path."""
  if not flags.FLAGS.experiment_name:
    raise ValueError("Experiment name is not provided with --experiment_name!")
  return os.path.join(DEFAULT_BASE_DIR.value, flags.FLAGS.experiment_name)


def round_data_dir() -> str:
  """Returns fully-qualified path to the dataset for this round."""
  return os.path.join(experiment_dir(), str(ROUND.value))


def previous_data_dir() -> str:
  """Returns fully-qualified path to the previous round's data."""
  return os.path.join(experiment_dir(), str(ROUND.value - 1))
