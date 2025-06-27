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

#!/bin/bash
#
# Helper script for testing Protoscribe using 'pytest'.

# Enable "exit immediately if any command fails" option.
set -e

# We must be running in a virtual environment. The following variable is
# definitely set by 'venv'.
if [ -z "${VIRTUAL_ENV}" ] ; then
  echo "Looks like the script is not run in virtual environment"
  exit 1
fi

PYTEST=$(which pytest)
if [ $? -ne 0 ] ; then
  echo "Pytest has not been installed!"
  exit 1
fi

# We can't run `pytest` collecting *all* the available tests automatically. This
# is because some tests define absl flags with similar names which results in
# 'absl.flags._exceptions.DuplicateFlagError: The flag 'X' is defined twice'
# exception. Therefore we run these test modules individualy.

INDIVIDUAL_COMPONENTS=(
  "protoscribe/corpus"
  "protoscribe/evolution"
  "protoscribe/glyphs"
  "protoscribe/language/embeddings"
  "protoscribe/language/morphology/morphemes_test.py"
  "protoscribe/language/morphology/morphology_test.py"
  "protoscribe/language/morphology/numbers_test.py"
  "protoscribe/language/phonology"
  "protoscribe/language/syntax"
  "protoscribe/models"
  "protoscribe/pmmx"
  "protoscribe/scoring"
  "protoscribe/semantics"
  "protoscribe/sketches/utils"
  "protoscribe/sketches/inference/glyphs_from_jsonl_test.py"
  "protoscribe/sketches/inference/json_utils_test.py"
  "protoscribe/sketches/inference/sketch_annotation_task_test.py"
  "protoscribe/sketches/inference/sketches_from_jsonl_test.py"
  "protoscribe/speech"
  "protoscribe/texts"
  # No tests in vision component.
  # "protoscribe/vision"
)
for test_component in "${INDIVIDUAL_COMPONENTS[@]}"; do
  "${PYTEST}" -v "${test_component}"
done
