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

r"""Generation of data in our BNC format from non-BNC embeddings.

Given the 10K most frequent BNC types and numbers, prepares data from
other non-BNC models provided at

   http://vectors.nlpl.eu/explore/embeddings/en/models/

in the format that is similar to our BNC embeddings.

Example:
--------
  python protoscribe/language/embeddings/prepare_non_bnc_data_main.py \
    --embedding_text_file ~/projects/concept_embeddings/enwiki_upos_skipgram_300_2_2021/model.txt \
    --output_dir /tmp --logtostderr
"""

from collections.abc import Sequence
import logging
import os

from absl import app
from absl import flags
from protoscribe.language.embeddings import embedder

_EMBEDDING_TEXT_FILE = flags.DEFINE_string(
    "embedding_text_file", None,
    "Text file containing non-BNC embedding. When downloaded, it is "
    "located in `model.txt` file.")

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", None,
    "Output directory where concept and number embeddings will be written "
    "in our BNC format.")

_CHECK_EXACT_SUBSET = flags.DEFINE_boolean(
    "check_superset", False,
    "Check that BNC is exact subset of the generated set.")


def _remove_tag(conc: str) -> str:
  return conc.split("_")[0]


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Load BNC embeddings. An alternative option is to simply load the frequency
  # list.
  logging.info("Loading BNC embeddings ...")
  bnc = embedder.load_embeddings_from_type("bnc")
  logging.info("Got %d embeddings.", len(bnc))

  # Parse the non-BNC embedding and fetch the relevant concepts and numbers,
  # writing them in same format as our BNC embeddings.
  if not os.path.isdir(_OUTPUT_DIR.value):
    raise ValueError(f"Directory {_OUTPUT_DIR.value} does not exist")
  logging.info("Parsing %s ...", _EMBEDDING_TEXT_FILE.value)
  found = set()
  outfile_concepts = os.path.join(_OUTPUT_DIR.value, embedder.CONC_EMB_FILENAME)
  outfile_numbers = os.path.join(_OUTPUT_DIR.value, embedder.NUM_EMB_FILENAME)
  with open(outfile_concepts, mode="wt", encoding="utf8") as out_conc_f:
    with open(outfile_numbers, mode="wt", encoding="utf8") as out_num_f:
      with open(_EMBEDDING_TEXT_FILE.value, encoding="utf8") as input_emb_f:
        next(input_emb_f)
        for line in input_emb_f:
          line = line.split()
          if len(line) != embedder.DEFAULT_EMBEDDING_DIM + 1:
            raise ValueError(f"Invalid number tokens: {len(line)}")
          is_num = line[0].endswith("_NUM")
          concept = _remove_tag(line[0]) if is_num else line[0]
          if concept not in bnc.embeddings:
            # Check if concept is a proper noun (PROPN). In BNC these are just
            # nouns.
            if not concept.endswith("_PROPN"):
              continue
            concept = _remove_tag(concept).lower() + "_NOUN"
            if concept not in bnc.embeddings:
              continue

          # Legit to add.
          if concept not in found:
            found.add(concept)
            out_f = out_num_f if is_num else out_conc_f
            out_f.write(" ".join(["0", concept] + line[1:]) + "\n")

  # Sanity check.
  logging.info("Found %d entries in BNC.", len(found))
  missing = sorted(list(set(bnc.embeddings.keys()).difference(found)))
  if missing:
    logging.warning("Missing %d BNC concepts: %s", len(missing), missing)
    with open(os.path.join(_OUTPUT_DIR.value, "missing.txt"), mode="w") as f:
      f.write("\n".join(missing) + "\n")
  if _CHECK_EXACT_SUBSET.value and len(found) != len(bnc):
    raise ValueError(f"Not all BNC concepts found. Expected {len(bnc)}, "
                     f"found {len(found)}!")


if __name__ == "__main__":
  flags.mark_flag_as_required("embedding_text_file")
  flags.mark_flag_as_required("output_dir")
  app.run(main)
