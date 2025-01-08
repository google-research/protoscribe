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

"""Miscellaneous file-related utilities."""

import logging
import os
import shutil

from absl import flags

import glob
import os
# Internal resources dependency

_NUM_COPY_WORKERS = flags.DEFINE_integer(
    "num_copy_workers", -1,
    "Number of workers when performing parallel copy."
)

# Source directory for all the code and data.
SRC_DIR = "protoscribe"
RESOURCE_DIR = "protoscribe"


def resource_path(path: str) -> str:
  """Returns fully qualified path for the given resource.

  Args:
    path: Path to resource.

  Returns:
    Full path.
  """
  return os.path.join(os.getcwd(), path)


def src_file(path: str) -> str:
  """Returns full path for the source file."""
  return os.path.join(SRC_DIR, path)


def copy_file(src_path: str, dst_path: str) -> None:
  """Copies full src path to destination path.

  Args:
    src_path: Fully-qualified source path.
    dst_path: Fully-qualified destination path.
  """
  logging.info("Copying %s -> %s ...", src_path, dst_path)
  shutil.copy2(src_path, dst_path)


def copy_src_file(source_dir: str, file_name: str, output_dir: str) -> None:
  """Copy a source file to a target directory.

  Target directory must exist.

  Args:
    source_dir: Source directory.
    file_name: File name or path in a `source_dir` to copy.
    output_dir: Target directory.
  """
  src_path = src_file(os.path.join(source_dir, file_name))
  dst_path = os.path.join(output_dir, file_name)
  copy_file(src_path, dst_path)


def copy_full_path(file_path: str, output_dir: str) -> None:
  """Copies a file provided by the full path to target directory.

  Target directory must exist.

  Args:
    file_path: Fully-qualified file path.
    output_dir: Output directory.
  """
  full_file_path = os.path.join(os.getcwd(), file_path)
  copy_src_file(
      source_dir=os.path.dirname(full_file_path),
      file_name=os.path.basename(full_file_path),
      output_dir=output_dir
  )


def copy_files(paths: list[str], target_dir: str) -> None:
  """Copies files to a target directory.

  Target directory must exist.

  Args:
    paths: List of file paths.
    target_dir: Target directory for copying.
  """
  logging.info("Copying %d files to %s ...", len(paths), target_dir)
  paths = [
      (
          path,
          os.path.join(target_dir, os.path.basename(path))
      ) for path in paths
  ]
  for source_path, target_path in paths:
    logging.info("Copying %s -> %s ...", source_path, target_path)
    shutil.copy(source_path, target_path)


def copy_dir(source_dir: str, target_dir: str) -> None:
  """Copies files from a source directory to a target directory.

  Source and target directories must exist. Important: This is NOT a recursive
  copy.

  Args:
    source_dir: Source directory from which the files will be recursively
      copied.
    target_dir: Target directory for copying.
  """
  if not os.path.isdir(source_dir):
    raise ValueError(f"Source directory {source_dir} does not exist!")

  logging.info("Copying `%s` to `%s` ...", source_dir, target_dir)
  source_paths = []
  for path in glob.glob(f"{source_dir}/*"):
    if not os.path.isdir(path):
      source_paths.append(path)
  copy_files(source_paths, target_dir)


def list_subdirs(root_dir: str) -> list[str]:
  """Retrieves all subdirectories under the specified root.

  Root directory must exist.

  Args:
    root_dir: Root directory.

  Returns:
    A list of subdirectories (may be empty).
  """
  if not os.path.isdir(root_dir):
    raise ValueError(f"Source directory {root_dir} does not exist!")

  logging.info("Searching for subdirs of `%s` ...", root_dir)
  subdir_paths = []
  for path in glob.glob(f"{root_dir}/*"):
    if os.path.isdir(path):
      subdir_paths.append(path)

  return subdir_paths
