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

"""Helper tool for scoring predictions from multiple JSONL files."""

from collections.abc import Sequence
import logging
import os

from absl import app
from absl import flags
from protoscribe.utils import subprocess_utils

from pathlib import Path

_RESULTS_DIR = flags.DEFINE_string(
    "results_dir", None,
    "Directory in which to search for prediction files in JSONL format. "
    "Note, the search is recursive including possible subdirectories.",
    required=True
)

_OUTPUT_SUMMARY_TSV_FILE = flags.DEFINE_string(
    "output_summary_tsv_file", None,
    "Name of the output file in TSV format containing the final summary of the "
    "results.",
    required=True
)

_DATASET_DIR = flags.DEFINE_string(
    "dataset_dir", None,
    "Dataset directory that was used for training the models we are scoring.",
    required=True
)

_CONCEPT_EMBEDDING_TYPE = flags.DEFINE_string(
    "concept_embedding_type", "bnc",
    "Type of semantic embeddings. Note, it's currently not possible to score "
    "multiple systems produced with different semantic embedding types."
)

_SEEN_CONCEPTS_FILE = flags.DEFINE_string(
    "seen_concepts_file",
    "protoscribe/data/concepts/administrative_categories.txt",
    "Text file containing administrative categories seen during training."
)

_SCORER_TOOL = "protoscribe/scoring/scorer"


def _find_all_files(base_path: Path) -> list[str]:
  """Recursively finds all files under the specified path.

  Args:
    base_path: Base directory.

  Returns:
    A list of all the files.
  """
  paths = list()
  for path in base_path.iterdir():
    if path.is_dir():
      paths.extend(_find_all_files(path))
    else:
      paths.append(str(path))
  return paths


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Find all the JSONL files in results directory.
  files = _find_all_files(Path(_RESULTS_DIR.value))
  files = [path for path in files if path.endswith(".jsonl")]
  logging.info("Found JSONL files: %s", files)
  if not files:
    raise ValueError(f"No JSONL files found in {_RESULTS_DIR.value}!")

  # Establish system names from file names.
  system_names = []
  for path in files:
    name = os.path.basename(path)
    name = os.path.splitext(name)[0]
    if name == "scorer":
      # This is a typical scenario where the scoring file is located in a
      # separate directory alongside generated sketches. In this case assume
      # the system name is encoded in the name of a parent directory.
      parent_dir = os.path.dirname(path)
      name = os.path.basename(parent_dir)
    system_names.append(name)
  logging.info("Symbolic system names: %s", system_names)

  # Run the scorer.
  subprocess_utils.run_subprocess(
      _SCORER_TOOL,
      args=[
          "--concepts", _SEEN_CONCEPTS_FILE.value,
          "--jsonl_files", ",".join(files),
          "--names", ",".join(system_names),
          "--dataset_dir", _DATASET_DIR.value,
          "--concept_embedding_type", _CONCEPT_EMBEDDING_TYPE.value,
          "--output_tsv_file", _OUTPUT_SUMMARY_TSV_FILE.value
      ]
  )


if __name__ == "__main__":
  app.run(main)
