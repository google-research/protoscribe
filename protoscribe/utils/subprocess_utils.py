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

"""Utilities for executing subprocesses."""

import logging
import subprocess
from typing import Any
import os
import sys

from protoscribe.utils import file_utils


def _val_to_string(value: Any) -> str:
  """Converts argument to string."""
  value = str(value)
  if value == "False" or value == "True":
    value = value.lower()
  return value


def run_subprocess(exec_path: str, args: list[Any]) -> None:
  """Runs executable as a subprocess using the supplied arguments.

  Args:
    exec_path: Path of the process executable to run.
    args: A flat list of argument names and values of the following format:
      [flag_name_1, flag_value_1, ..., flag_name_N, flag_value_N],
      where each flag must have the corresponding name and value present in
      the list.

  Raises:
    ValueError: if invalid argument list is passed.
    CalledProcessError: if execution failed for some reason.
  """
  if len(args) % 2 != 0:
    raise ValueError(
        f"The argument list should have an even length, got {len(args)}!"
    )

  # Determine the process to execute.
  exec_args = [
      sys.executable, file_utils.resource_path(f"{exec_path}_main.py")
  ]

  # Makes sure that all elements of the process' argument list are
  # in `name=value` format.
  process_args = []
  args_it = iter(args)
  for arg_name, arg_value in zip(args_it, args_it):
    value = _val_to_string(arg_value)
    process_args.append(f"{arg_name}={value}")

  # Execute the command in `exec_path` in a subprocess and wait for its
  # completion printing the logs as it executes.
  args = exec_args + process_args + ["--logtostderr"]
  logging.info("Executing: %s", args)
  proc = subprocess.Popen(
      args,
      shell=False,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      encoding="utf8",
      # Make sure the path is correct relative to the current working directory.
      env=os.environ.copy().update({
        "PYTHONPATH": ".",
      })
  )

  # Dump the logs from the process in real time.
  assert proc.stdout  # Placate type checker.
  for line in iter(proc.stdout.readline, ""):
    print(line.rstrip())

  # Wait for completion.
  status_code = proc.wait()
  if status_code != 0:
    raise subprocess.CalledProcessError(proc.returncode, proc.args)
