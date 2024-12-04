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

"""Converts JSONL inference results to discrete glyph information."""

import json
import logging
from typing import Sequence

from absl import app
from absl import flags
from protoscribe.corpus.reader import dataset_defaults as ds_lib
from protoscribe.sketches.inference import glyphs_from_jsonl
from protoscribe.texts import generate_lib

import glob
import os

_INPUT_JSONL_FILE = flags.DEFINE_string(
    "input_jsonl_file", None,
    "Input file containing the decoding results in JSONL format.",
    required=True
)

_OUTPUT_TSV_FILE = flags.DEFINE_string(
    "output_tsv_file", None,
    "Human-friendly TSV file containing the target concepts, their "
    "pronunciations along with hypothesis glyphs and their pronunciations. "
    "The final column contains model's confidence in the best hypothesis.",
    required=True
)

_OUTPUT_FILE_FOR_SCORER = flags.DEFINE_string(
    "output_file_for_scorer", None,
    "Output file in JSONL format that contains all the necessary information "
    "to run the scorer.",
    required=True
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  glyph_vocab = ds_lib.get_glyph_vocab()
  pronunciation_lexicon, _ = generate_lib.load_phonetic_forms(
      main_lexicon_file=ds_lib.main_lexicon_file(),
      number_lexicon_file=ds_lib.number_lexicon_file()
  )

  logging.info("Loading results in %s ...", _INPUT_JSONL_FILE.value)
  results = []
  all_inputs = set()
  scorer_dicts = []
  num_errors = 0
  with open(_INPUT_JSONL_FILE.value, mode="r") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      glyphs_dict = json.loads(line)
      try:
        inputs, glyphs, scorer_dict, error = glyphs_from_jsonl.json_to_glyphs(
            glyph_vocab, glyphs_dict, pronunciation_lexicon
        )
      except IndexError as e:
        logging.exception(
            "Failed to detokenize: %s. Exception: %s", glyphs_dict, e
        )
        continue
      if inputs in all_inputs:  # Ignore duplicate query.
        continue
      all_inputs.add(inputs)
      results.append((inputs, glyphs, scorer_dict))
      scorer_dicts.append(scorer_dict)
      if error:
        num_errors += 1
  logging.info("Processed %d unique documents.", len(all_inputs))
  if num_errors:
    logging.warning("Encountered %d errors.", num_errors)

  results = sorted(results, key=lambda x: x[0])
  logging.info("Writing results %s ...", _OUTPUT_TSV_FILE.value)
  with open(_OUTPUT_TSV_FILE.value, mode="w") as f:
    f.write("Input concepts\tConcept Pron\tGlyphs\tGlyph Prons\tConfidence\n")
    for input_concepts, output_glyphs, scorer_dict in results:
      glyphs_pron = []
      for glyph_pron in filter(None, scorer_dict["glyph.prons"][0]):
        glyphs_pron.append(" ".join(glyph_pron))
      glyphs_pron = " # ".join(glyphs_pron)
      concept_pron = " ".join(scorer_dict["concept.pron"])
      confidence = scorer_dict["glyph.confidence"]
      line = (
          f"{input_concepts}\t{concept_pron}\t{output_glyphs}\t{glyphs_pron}\t"
          f"{confidence:0.2f}"
      )
      f.write(f"{line}\n")

  scorer_dicts = sorted(scorer_dicts, key=lambda x: x["doc.id"])
  logging.info("Writing scorer file %s ...", _OUTPUT_FILE_FOR_SCORER.value)
  with open(_OUTPUT_FILE_FOR_SCORER.value, "wt") as f:
    for score_dict in scorer_dicts:
      f.write(json.dumps(score_dict, sort_keys=True) + "\n")


if __name__ == "__main__":
  app.run(main)
