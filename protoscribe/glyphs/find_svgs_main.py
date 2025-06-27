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

"""Randomly selects SVGs associated with terms."""

import collections
import os
import random
import sys

from absl import app
from absl import flags

_DATA = "protoscribe/data"

flags.DEFINE_list("terms", ["HORSE"], "Comma-separated list of terms")
flags.DEFINE_string("mapfile", f"{_DATA}/wiktionary/map.tsv",
                    "Character to word map.")
# TODO: This needs to be changed to something reasonable
flags.DEFINE_string("svgs", f"{_DATA}/glyphs/oracle_bone/svgs", "Directory with SVGs.")

FLAGS = flags.FLAGS


def load_map():
  character_map = collections.defaultdict(set)
  with open(FLAGS.mapfile) as stream:
    for line in stream:
      char, terms = line.strip().split("\t")
      terms = [t.strip().replace(" ", "_") for t in terms.split(",")]
      for t in terms:
        character_map[t].add(char)
  return character_map


def main(unused_argv):
  character_map = load_map()
  paths = []
  for term in FLAGS.terms:
    chars = list(character_map[term])
    if not chars:
      sys.stderr.write(f"Missing char for {term}\n")
      continue
    char = random.choice(chars)
    svg_dir = os.path.join(FLAGS.svgs, char)
    svg = random.choice(os.listdir(svg_dir))
    paths.append(os.path.join(svg_dir, svg))
  print(",".join(paths))


if __name__ == "__main__":
  app.run(main)
