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

r"""Command-line tool for scorer.

Sample usage:
-------------

From directories containing individual predicions in JSON format:
 python protoscribe/scoring/scorer_main.py \
   --json_dirs=dir1,dir2 \
   --names=name1,name2

From individial JSONL files:
 python protoscribe/scoring/scorer_main.py \
   --jsonl_files=file1,file2 \
   --names=name1,name2

 where, if given, --names must have the same length as dirs/files.
"""

import logging
import timeit

from absl import app
from absl import flags
import pandas as pd
from protoscribe.corpus.reader import dataset_defaults as ds_lib
from protoscribe.language.phonology import phoible_segments
from protoscribe.scoring import scorer as lib

_JSON_DIRS = flags.DEFINE_list(
    "json_dirs", None,
    "List of directories, each containing individual prediction files in JSON "
    "format."
)
_JSONL_FILES = flags.DEFINE_list(
    "jsonl_files", None,
    "List of files each in JSONL format containing the predictions."
)
_NAMES = flags.DEFINE_list(
    "names", None,
    "List of names: if given must be the same length as directories or files."
)
_PRINT_CONTENTS = flags.DEFINE_bool(
    "print_contents", False, "Print individual file contents."
)
_OUTPUT_TSV_FILE = flags.DEFINE_string(
    "output_tsv_file", None, "Single TSV file with all the results."
)


def _display_scores(scorer: lib.ComparativeScorer) -> None:
  """Displays the scores."""
  all_scores = []
  for name in sorted(scorer.store):
    print("*" * 80)
    print(name)
    if _PRINT_CONTENTS.value:
      for fname in scorer.store[name]:
        for line in scorer.store[name][fname]:
          print(line)
        print("-" * 80)
    print(f"{'Type':20s}\tExamp.\tTotal\tMean")
    for score_type in lib.SCORE_TYPES:
      num_examples, total, mean = scorer.scorers[name].find_score(score_type)
      print(f"{score_type:20s}\t{num_examples}\t{total}\t{mean}")
      all_scores.append({
          "Name": name,
          "Type": score_type,
          "Num Examples": num_examples,
          "Total": total,
          "Mean": mean,
      })

  if _OUTPUT_TSV_FILE.value:
    logging.info("Saving results to %s ...", _OUTPUT_TSV_FILE.value)
    output_df = pd.DataFrame(all_scores)
    output_df.to_csv(_OUTPUT_TSV_FILE.value, sep="\t", index=False)


def main(unused_argv):
  if not _JSON_DIRS.value and not _JSONL_FILES.value:
    raise ValueError("Specify either --json_dirs or --jsonl_files")

  # Make sure the symbol names have been passed correctly.
  names = _NAMES.value
  if names:
    paths = _JSON_DIRS.value if _JSON_DIRS.value else _JSONL_FILES.value
    if len(set(paths)) != len(set(names)):
      raise ValueError("Number of unique paths and unique names has to match!")
  else:
    names = []

  # Initialize, collect and display the scores.
  logging.info("Initializing scorer ...")
  scorer = lib.ComparativeScorer(
      phoible_phonemes_path=phoible_segments.PHOIBLE,
      phoible_features_path=phoible_segments.PHOIBLE_FEATURES,
      phonetic_embeddings_path=ds_lib.phonetic_embeddings_file(),
      main_lexicon_path=ds_lib.main_lexicon_file(),
      number_lexicon_path=ds_lib.number_lexicon_file(),
  )

  logging.info("Scoring %d systems ...", len(names))
  start_time = timeit.default_timer()
  if _JSON_DIRS.value:
    scorer.score_dirs(_JSON_DIRS.value, pretty_names=names)
  else:
    scorer.score_jsonl_files(_JSONL_FILES.value, pretty_names=names)
  logging.info(
      "Elapsed scoring time: %0.4g sec", timeit.default_timer() - start_time
  )

  _display_scores(scorer)


if __name__ == "__main__":
  app.run(main)
