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

"""Generate the restricted and non-restricted vocab files."""

import csv

from absl import app
from absl import flags

import glob
import os

_ADMINISTRATIVE_CONCEPTS = flags.DEFINE_string(
    "administrative_concepts",
    "protoscribe/data/concepts/administrative_categories.txt",
    "Path to list of administrative_concepts.",
)

_TEXTS_GLOB = flags.DEFINE_string(
    "texts_glob", None,
    "Path to glob of generated texts."
)

_GLYPH_SYMS = flags.DEFINE_string(
    "glyph_syms", None,
    "Path to glyph symbols.",
    required=True
)

_WORD_SYMS = flags.DEFINE_string(
    "word_syms", None,
    "Path to word symbols.",
    required=True
)

FLAGS = flags.FLAGS


def load_administrative_concepts() -> set[str]:
  """Loads administrative concepts.

  Returns:
    A set
  """
  administrative_concepts = set()
  with open(_ADMINISTRATIVE_CONCEPTS.value, mode="rt") as stream:
    for line in stream:
      terms = line.split()
      # Administrative categories may include a hierarchy, in which case it is
      # the last term that is relevant.
      administrative_concepts.add(terms[-1])
  return administrative_concepts


def main(unused_argv):
  administrative_concepts = load_administrative_concepts()
  paths = glob.glob(_TEXTS_GLOB.value)
  administrative_glyphs = set()
  non_administrative_glyphs = set()
  word_vocab = set()
  for path in paths:
    with open(path) as stream:
      reader = csv.reader(stream, delimiter="\t", quotechar='"')
      for row in reader:
        text, _, _, glyphs = row
        text = text.split()
        for token in text:
          word_vocab.add(token)
          if token.isdigit():
            continue
          if token in administrative_concepts:
            administrative = True
          else:
            administrative = False
        glyphs = glyphs.split()
        for glyph in glyphs:
          if not glyph.endswith("_GLYPH"):
            administrative_glyphs.add(glyph)
          elif administrative:
            administrative_glyphs.add(glyph)
          else:
            non_administrative_glyphs.add(glyph)
  # Remove any cases where a non-administrative glyph is really administrative,
  # and tag this as restricted.
  glyphs = set()
  for glyph in administrative_glyphs:
    glyphs.add(glyph)
  for glyph in non_administrative_glyphs:
    if glyph in administrative_glyphs:
      continue
    glyphs.add(f"{glyph}_RESTRICTED")

  special_tokens = ["<pad>", "<s>", "</s>", "<unk>"]

  def create_vocab(vocab, path):
    with open(path, "w") as stream:
      idx = 0
      for token in special_tokens:
        stream.write(f"{token}\t{idx}\n")
        idx += 1
      for token in vocab:
        stream.write(f"{token}\t{idx}\n")
        idx += 1

  create_vocab(glyphs, _GLYPH_SYMS.value)
  create_vocab(word_vocab, _WORD_SYMS.value)


if __name__ == "__main__":
  app.run(main)
