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

#!/bin/bash
#
# Utility script for setting up Protoscribe for development.

# Enable "exit immediately if any command fails" option.
set -e

# Install non-Python dependencies.
# TODO: Ideally we shouldn't have any of those.
echo "Installing non-Python dependencies ..."
(sudo bash || bash) <<'EOF'
apt update && \
apt install -y libcairo2-dev protobuf-compiler git
EOF

# Quick sanity check that we have the necessary tools installed.
PROTOC_COMPILER=$(which protoc)
if [ $? -ne 0 ] ; then
  echo "Protocol buffer compiler 'protoc' is not installed!"
  exit 1
fi
GIT=$(which git)
if [ $? -ne 0 ] ; then
  echo "Git has not been installed!"
  exit 1
fi

# We must be running in a virtual environment. The following variable is
# definitely set by 'venv'.
if [ -z "${VIRTUAL_ENV}" ] ; then
  echo "Looks like the script is not run in virtual environment"
  exit 1
fi

# Install Python dependencies from requirements.txt
if [ ! -r requirements.txt ] ; then
  echo "The setup script should run from the home directory of " \
    "the Protoscribe project."
  exit 1
fi
echo "Installing Python required dependencies ..."
pip install --upgrade pip
pip3 install -U -r requirements.txt

# Compile the protocol buffers.
PROTOBUF_FILES=(
  "protoscribe/language/morphology/morphology_parameters.proto"
  "protoscribe/language/phonology/prosody_template.proto"
  "protoscribe/texts/number_config.proto"
)
echo "Compiling protocol buffer files ..."
for protobuf_path in "${PROTOBUF_FILES[@]}"; do
  echo "Compiling ${protobuf_path} ..."
  protobuf_dir=$(dirname "${protobuf_path}")
  "${PROTOC_COMPILER}" \
    -I. \
    --proto_path="${protobuf_dir}" \
    --python_out=. \
    "${protobuf_path}"
done

# Download BNC skipgram embeddings data.
echo "Getting BNC data ..."
CURRENT_DIR=$(pwd)
BNC_URL="https://github.com/rwsproat/symbols"
BNC_DIR="${CURRENT_DIR}/protoscribe/data/semantics/bnc"
if [ ! -d "${BNC_DIR}" ] ; then
  echo "BNC embeddings home directory not found!"
  exit 1
fi
cd "${BNC_DIR}"
rm -rf symbols  # Remove the previous export if present.
"${GIT}" clone -n --depth=1 --filter=tree:0 "${BNC_URL}"
cd symbols
"${GIT}" sparse-checkout set --no-cone bnc_embeddings_10k
"${GIT}" checkout
cp -f bnc_embeddings_10k/embeddings.txt "${BNC_DIR}"
cp -f bnc_embeddings_10k/numbers.txt "${BNC_DIR}"
cd "${CURRENT_DIR}"

# Download PHOIBLE data.
echo "Getting PHOIBLE data ..."
CURRENT_DIR=$(pwd)
PHOIBLE_URL="https://github.com/phoible/dev"
PHOIBLE_DIR=protoscribe/data/phonology
if [ ! -d "${PHOIBLE_DIR}" ] ; then
  echo "Phonology home directory not found!"
  exit 1
fi
cd "${PHOIBLE_DIR}"
rm -rf dev  # Remove the previous export if present.
"${GIT}" clone -n --depth=1 --filter=tree:0 "${PHOIBLE_URL}"
cd dev
"${GIT}" sparse-checkout set --no-cone data
"${GIT}" checkout
cd "${CURRENT_DIR}"
PHOIBLE_DATA_DIR="${PHOIBLE_DIR}/dev/data"
if [ ! -d "${PHOIBLE_DATA_DIR}" ] ; then
  echo "Failed to download PHOIBLE data!"
  exit 1
fi

# Process PHOIBLE data.
echo "Processing PHOIBLE data ..."
PROCESS_PHOIBLE=protoscribe/language/phonology/phoible_ingest_main.py
python "${PROCESS_PHOIBLE}" \
  --phoible_source_file="${PHOIBLE_DATA_DIR}/phoible.csv" \
  --phonemes_tsv_file="${PHOIBLE_DIR}/phoible-phonemes.tsv" \
  --segment_features_tsv_file="${PHOIBLE_DIR}/phoible-segments-features.tsv" \
  --logtostderr
if [ ! -f "${PHOIBLE_DIR}/phoible-phonemes.tsv" ] ; then
  echo "PHOIBLE phoneme inventories failed to be generated"
  exit 1
fi
if [ ! -f "${PHOIBLE_DIR}/phoible-segments-features.tsv" ] ; then
  echo "PHOIBLE segment inventories failed to be generated"
  exit 1
fi

# Install Princeton WordNet.
echo "Installing Princeton Wordnet ..."
python -m nltk.downloader wordnet
