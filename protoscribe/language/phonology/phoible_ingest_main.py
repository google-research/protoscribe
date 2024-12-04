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

"""Ingesting PHOIBLE data for consumption by Protoscribe."""

from collections.abc import Sequence
import logging

from absl import app
from absl import flags
import pandas as pd

import glob
import os

_PHOIBLE_SOURCE_FILE = flags.DEFINE_string(
    "phoible_source_file", None,
    "Input PHOIBLE file in CSV format from PHOIBLE development tree.",
    required=True
)

_SEGMENTS_FEATURES_TSV_FILE = flags.DEFINE_string(
    "segment_features_tsv_file", None,
    "Output text file in TSV format containing segment information.",
    required=True
)

_PHONEMES_TSV_FILE = flags.DEFINE_string(
    "phonemes_tsv_file", None,
    "Output text file in TSV format containing all the phoneme inventories. "
    "Each individual row represents a phoneme from the corresponding phoneme "
    "inventory for some language/dialect.",
    required=True
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  logging.info("Reading %s ...", _PHOIBLE_SOURCE_FILE.value)
  with open(_PHOIBLE_SOURCE_FILE.value, mode="rb") as f:
    df = pd.read_csv(f, sep=",", dtype=str, header=0, encoding="utf-8")
  logging.info("Read %d entries. Shape: %s", len(df), df.shape)

  # Create phoneme inventory data.
  logging.info("Generating %s ...", _PHONEMES_TSV_FILE.value)
  new_df = df[[
      "InventoryID", "Source", "ISO6393",
      "LanguageName", "Phoneme", "SegmentClass",
  ]].copy()
  new_df.rename(
      columns={
          "ISO6393": "LanguageCode",
          "SegmentClass": "Class",
      },
      inplace=True
  )
  with open(_PHONEMES_TSV_FILE.value, mode="wb") as f:
    new_df.to_csv(f, sep="\t", header=True, index=False, encoding="utf-8")

  # Create featurization data. The first feature column is `tone`.
  columns = df.columns.tolist()
  new_columns = ["Phoneme"] + columns[columns.index("tone"):]
  logging.info("Generating %s ...", _SEGMENTS_FEATURES_TSV_FILE.value)
  new_df = df[new_columns].copy()
  new_df.rename(columns={"Phoneme": "segment"}, inplace=True)
  new_df.drop_duplicates(inplace=True)

  with open(_SEGMENTS_FEATURES_TSV_FILE.value, mode="wb") as f:
    new_df.to_csv(f, sep="\t", header=True, index=False, encoding="utf-8")


if __name__ == "__main__":
  app.run(main)
