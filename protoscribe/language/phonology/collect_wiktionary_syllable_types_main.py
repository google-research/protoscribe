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

"""Collects statistics on syllable types from Wiktionary."""

import collections
import csv
import sys

from absl import app
from absl import flags
from protoscribe.language.phonology import phoible_segments

import glob
import os

flags.DEFINE_string("output", None, "Output file.")

flags.DEFINE_string(
    "phoible",
    "protoscribe/data/phonology/phoible-phonemes.tsv",
    "Path to PHOIBLE phonemes text file in TSV format.",
)

flags.DEFINE_string(
    "phoible_features",
    "protoscribe/data/phonology/phoible-segments-features.tsv",
    "Path to PHOIBLE phonemes with features in text file in TSV format.",
)

flags.DEFINE_list(
    "wiktionary_files", None,
    "List of Wiktionary TSV files to use."
)

FLAGS = flags.FLAGS


def main(unused_argv):
  phoible = phoible_segments.PhoibleSegments(
      path=FLAGS.phoible, features_path=FLAGS.phoible_features
  )
  syllable_stats = collections.defaultdict(lambda: collections.defaultdict(int))

  def flatten(syllables):
    result = []
    for syllable in syllables:
      for phoneme in syllable:
        result.append(phoneme)
    return result

  for path in FLAGS.wiktionary_files:
    sys.stderr.write(f"Parsing {path}\n")
    seen = set()
    with open(path) as stream:
      reader = csv.reader(stream, delimiter="\t", quotechar='"')
      for row in reader:
        syllables = phoible.syllabify(row[1])
        template = phoible.template(syllables)
        if (
            len(syllables) == len(template.split("."))
            and "?" not in template  # Error if not
        ):
          sonority = phoible.sonority_profile(flatten(syllables))
          if syllables not in seen:  # Not seen for this language
            syllable_stats[template][sonority] += 1
            seen.add(syllables)
  final_stats = []
  for template in syllable_stats:
    total = 0
    for sonority in syllable_stats[template]:
      total += syllable_stats[template][sonority]
    for sonority in syllable_stats[template]:
      final_stats.append(
          (total, syllable_stats[template][sonority], template, sonority)
      )
  final_stats.sort(reverse=True)
  with open(FLAGS.output, "w") as stream:
    # Write out single syllables first,
    for total, specific_total, template, sonority in final_stats:
      if "." in template:
        continue
      sonority = " ".join([str(v) for v in sonority])
      stream.write(f"{total}\t{specific_total}\t{template}\t{sonority}\n")
    # then multisyllables.
    for total, specific_total, template, sonority in final_stats:
      if "." not in template:
        continue
      sonority = " ".join([str(v) for v in sonority])
      stream.write(f"{total}\t{specific_total}\t{template}\t{sonority}\n")


if __name__ == "__main__":
  flags.mark_flag_as_required("output")
  flags.mark_flag_as_required("wiktionary_files")
  app.run(main)
