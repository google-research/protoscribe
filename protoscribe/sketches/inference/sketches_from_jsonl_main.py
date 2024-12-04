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

"""Utility for generating sketches from the inference results.

By default the raster image will be generated with the default (white)
background. For nicer results enable the transparency with `--transparent`.
If title is disabled with `--noshow_title` it is saved in a separate text file
alongside the image.
"""

import json
import logging

from absl import app
from absl import flags
import ml_collections
from protoscribe.corpus.reader import dataset_defaults as ds_lib
from protoscribe.sketches.inference import json_utils
from protoscribe.sketches.inference import sketches_from_jsonl as lib
from protoscribe.sketches.utils import stroke_stats as stats_lib
from protoscribe.texts import generate_lib

import glob
import os

_INPUT_JSONL_FILE = flags.DEFINE_string(
    "input_jsonl_file", None,
    "Input file containing the decoding results in JSONL format.",
    required=True
)

_TOKENIZER_FILE_NAME = flags.DEFINE_string(
    "tokenizer_file_name", "vocab2048_normalized_sketchrnn.npy",
    "Name of the tokenizer file. This will be loaded from the dataset "
    "directory."
)

_MAX_STROKE_SEQUENCE_LENGTH = flags.DEFINE_integer(
    "max_stroke_sequence_length", 250,
    "Maximum number of strokes in a sketch."
)

_STROKE_NORMALIZATION_TYPE = flags.DEFINE_string(
    "stroke_normalization_type", "sketch-rnn",
    "If non-empty, overrides stroke normalization with the specified type."
    "Empty value means that no de-normalization of strokes is performed. "
    "Note, the default value corresponds to the particular tokenizer type "
    "provided by the default setting for `--tokenizer_file_name`."
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", None,
    "Output directory for saving sketches.",
    required=True
)

_OUTPUT_FILE_FOR_SCORER = flags.DEFINE_string(
    "output_file_for_scorer", None,
    "Output file in JSONL format that contains all the necessary information "
    "from the glyph recognition/annotation to run the scorer. Only applies "
    "when `--recognizer_json` flag is enabled."
)

_OUTPUT_TSV_FILE = flags.DEFINE_string(
    "output_tsv_file", None,
    "Human-friendly TSV file containing the target concepts, their "
    "pronunciations along with hypothesis glyphs and their pronunciations. "
    "The final column contains model's confidence in the best hypothesis."
    "This option only applies when `--recognizer_json` flag is enabled."
)

_NUM_SKETCHES_TO_PROCESS = flags.DEFINE_integer(
    "num_sketches_to_process", -1,
    "If set to positive number, process only the specified number of sketches."
)

_DEDUP_INPUTS = flags.DEFINE_bool(
    "dedup_inputs", True,
    "If enabled, only keeps the sketches for which the inputs (number and "
    "concept combination) are unique. We may *not* want to dedup if sketch "
    "generation is not deterministic, e.g., if we sample from variational "
    "model. In this case we need to distinguish the duplicate documents "
    "by their ID."
)


def main(argv: list[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  logging.info("Initializing tokenizer ...")
  config = ml_collections.ConfigDict({
      "max_stroke_sequence_length": _MAX_STROKE_SEQUENCE_LENGTH.value,
      "stroke_normalization_type": _STROKE_NORMALIZATION_TYPE.value,
      "stroke_tokenizer": ml_collections.ConfigDict({
          "vocab_filename": _TOKENIZER_FILE_NAME.value,
      })
  })
  stroke_stats = stats_lib.load_stroke_stats(
      config, ds_lib.sketch_stroke_stats_file()
  )
  stroke_tokenizer = ds_lib.get_stroke_tokenizer(config)
  glyph_vocab = ds_lib.get_glyph_vocab()
  pronunciation_lexicon, _ = generate_lib.load_phonetic_forms(
      main_lexicon_file=ds_lib.main_lexicon_file(),
      number_lexicon_file=ds_lib.number_lexicon_file()
  )

  logging.info("Loading results in %s ...", _INPUT_JSONL_FILE.value)
  scorer_dicts = []
  unique_inputs = set()
  with open(_INPUT_JSONL_FILE.value, mode="r") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue

      sketch_dict = json.loads(line)
      scorer_dict = json_utils.get_scorer_dict(
          sketch_dict, pronunciation_lexicon
      )

      if _DEDUP_INPUTS.value:
        inputs = scorer_dict["text.inputs"]
        if inputs in unique_inputs:
          continue
        unique_inputs.add(inputs)

      sketch_to_score = lib.json_to_sketch(
          config, sketch_dict, stroke_stats, stroke_tokenizer, glyph_vocab,
          pronunciation_lexicon, _OUTPUT_DIR.value
      )
      scorer_dicts.append(sketch_to_score)
      if len(scorer_dicts) == _NUM_SKETCHES_TO_PROCESS.value:
        break
  logging.info("Processed %d sketches.", len(scorer_dicts))

  ok_for_scorer = (
      lib.RECOGNIZER_JSON.value or lib.COMBINED_GLYPHS_AND_STROKES.value
  )
  if ok_for_scorer and _OUTPUT_FILE_FOR_SCORER.value:
    scorer_dicts = sorted(scorer_dicts, key=lambda x: x["doc.id"])
    logging.info("Writing scorer file %s ...", _OUTPUT_FILE_FOR_SCORER.value)
    with open(_OUTPUT_FILE_FOR_SCORER.value, "wt") as f:
      for score_dict in scorer_dicts:
        f.write(json.dumps(score_dict, sort_keys=True) + "\n")

  # Write a more human-readable TSV file. Please note, the file will contain
  # duplicate entries corresponding to the input concepts because the real
  # inputs are sketch strokes and these *are* different.
  if ok_for_scorer and _OUTPUT_TSV_FILE.value:
    scorer_dicts = sorted(scorer_dicts, key=lambda x: x["text.inputs"])
    logging.info(
        "Writing %d results %s ...", len(scorer_dicts), _OUTPUT_TSV_FILE.value
    )
    with open(_OUTPUT_TSV_FILE.value, mode="w") as f:
      f.write(
          (
              "Doc Id\tInput concepts\tConcept Pron\tGlyphs\tGlyph Prons\t"
              "Confidence\tSketch Confidence\n"
          )
      )
      for scorer_dict in scorer_dicts:
        doc_id = scorer_dict["doc.id"]
        input_concepts = scorer_dict["text.inputs"]
        concept_pron = " ".join(scorer_dict["concept.pron"])
        confidence = scorer_dict["glyph.confidence"]
        output_glyphs = scorer_dict["glyph.names.best"]
        glyphs_pron = scorer_dict["glyph.prons.best"]
        sketch_confidence = scorer_dict["sketch.confidence"]
        line = (
            f"{doc_id}\t{input_concepts}\t{concept_pron}\t{output_glyphs}\t"
            f"{glyphs_pron}\t{confidence:0.4f}\t{sketch_confidence:0.4f}"
        )
        f.write(f"{line}\n")


if __name__ == "__main__":
  app.run(main)
