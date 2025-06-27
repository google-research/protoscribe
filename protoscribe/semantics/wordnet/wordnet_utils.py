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

"""Helper WordNet parsing functions."""

import logging
import os

import networkx as nx
import nltk
from nltk.corpus import wordnet as wn

import glob
import os

Synset = nltk.corpus.reader.Synset

# This is a mapping between part-of-speech data file suffix in WordNet and the
# corresponding to syntactic category as defined in
# https://wordnet.princeton.edu/documentation/wndb5wn:
#   `n` for noun files, `v` for verb files, `a` for adjective files, `r` for
#   adverb files.
WORDNET_FILE_SUFFIX_TO_POS = {
    "adj": "a",
    "adv": "r",
    "noun": "n",
    "verb": "v",
}

# Mapping from Wordnet POS tags to Protoscribe tags.
WORDNET_POS_TO_PROTOSCRIBE = {
    "a": "ADJ",
    "n": "NOUN",
    "r": "ADV",
    "s": "ADJ",  # Adjective satellites.
    "v": "VERB",
}


def wordnet_data_path(wordnet_dir: str, suffix_pos: str) -> str:
  """Returns path to the WordNet data file given part-of-speech."""
  return os.path.join(wordnet_dir, f"data.{suffix_pos}")


def wordnet_parse_data_lines(
    lines: list[str]
) -> dict[str, tuple[list[str], list[tuple[str, str, str]]]]:
  """Parses lines from WordNet data file.

  See https://wordnet.princeton.edu/documentation/wndb5wn for the description
  of the structure of data files.

  Args:
     lines: List of lines as strings.

  Returns:
     A dictionary where each key is a synset offset and the value is a
     pair that consists of:
     - A list of words in this synset.
     - A list of pointers to other synsets, where each pointer consists of a
       3-tuple (pointee synset, relation type, POS).
  """
  synsets = {}
  for line in lines:
    if not line.strip():
      continue

    # Each data file begins with several lines containing a copyright notice,
    # version number, and license agreement. These lines all begin with two
    # spaces and the line number. All other lines are in the following format.
    # Integer fields are of fixed length and are zero-filled:
    #
    #   synset_offset lex_filenum ss_type w_cnt word lex_id
    #       [word lex_id...] p_cnt [ptr...] [frames...] | gloss

    # Skip documentation.
    if line[0:2] == "  ":
      continue

    toks = line.split(" ")
    # `synset_offset`: Current byte offset in the file represented as an 8 digit
    # decimal integer.
    if len(toks[0]) != 8:
      raise ValueError(
          "Expected 8-byte string representing 8 digit decimal integer. "
          f"Got {toks[0]} instead!"
      )

    # `w_cnt`: Two digit *hexadecimal* integer indicating the number of words in
    # the synset, `word`: ASCII form of a word as entered in the synset by the
    # lexicographer.
    synset_words = []
    num_words = int(toks[3], 16)
    offset = 4
    for _ in range(num_words):
      synset_words.append(toks[offset])
      offset += 2  # Skip `lex_id`.

    # `p_cnt`: Three digit decimal integer indicating the number of pointers
    # from this synset to other synsets.
    target_synsets = []
    num_ptr = int(toks[offset])
    offset += 1  # Skip pointer symbol.
    for _ in range(num_ptr):
      # There are four fields representing each pointer:
      #   pointer_symbol  synset_offset  pos  source/target

      # Relation type is known as `pointer_symbol`, e.g., for nouns, this can be
      # be `!` for `Antonym`, `~` for `Hyponym`, etc. See
      # https://wordnet.princeton.edu/documentation/wninput5wn.
      relation_type = toks[offset]
      target_synset = toks[offset + 1]
      # Syntactic category: `n` for noun files, `v` for verb files, `a` for
      # adjective files, `r` for adverb files.
      pos = toks[offset + 2]
      offset += 4  # Skip `source/target` field.
      target_synsets.append((target_synset, relation_type, pos))

    # Ignore `frames` and `gloss`.
    data = (synset_words, target_synsets)
    synsets[toks[0]] = data

  return synsets


def load_synsets_from_data(
    wordnet_dir: str
) -> dict[str, dict[str, tuple[list[str], list[tuple[str, str, str]]]]]:
  """Loads WordNet synsets by syntactic category."""
  all_synsets = {}
  for suffix, pos in WORDNET_FILE_SUFFIX_TO_POS.items():
    path = wordnet_data_path(wordnet_dir, suffix)
    with open(path) as f:
      logging.info("[%s] Reading %s ...", pos, path)
      synsets = wordnet_parse_data_lines(f.readlines())
      logging.info("[%s] Loaded %d synsets.", pos, len(synsets))
      all_synsets[WORDNET_FILE_SUFFIX_TO_POS[suffix]] = synsets
  return all_synsets


def parse_synset_name(synset_name: str) -> tuple[str, str, str]:
  """Splits Synset name into constituent components."""
  if len(synset_name.split(".")) < 3:
    raise ValueError("Invalid synset name! Expected at least three components")
  name, pos, synset_index = synset_name.rsplit(".", 2)
  return name, pos, synset_index


def build_graph(
    words: list[str] | None = None,
    pos: str | None = None,
    construct_dag: bool = False,
    prune: bool = False
) -> nx.Graph | nx.DiGraph:
  """Constructs a DAG or undirected graph that represents the WordNet hierarchy.

  Args:
    words: The vertices of the graph (https://www.nltk.org/howto/wordnet.html).
      Can be empty in which case all the synsets for the supplied part of speech
      will be used. The available POS are `wn.ADJ`, `wn.ADV`, `wn.NOUN` and
      `wn.VERB`.
    pos: Restrict the construction to a particular part-of-speech.
    construct_dag: Construct directed acyclic or undirected graph.
    prune: Prune half-nodes from the graph.

  Returns:
    A directed acyclic or undirected graph.

  Raises:
    ValueError if the resulting graph is not a DAG and a DAG was requested.
  """
  g = nx.DiGraph() if construct_dag else nx.Graph()

  if not words:
    words = [s.name() for s in wn.all_synsets(pos)]
  logging.info("Building graph from %d words ...", len(words))

  edges = set()
  for word in words:
    queue = [wn.synset(word)]
    while queue:
      synset = queue.pop()
      for hypernym in synset.hypernyms():
        queue.append(hypernym)
        g.add_edge(hypernym.name(), synset.name())
        _, pos, synset_index = parse_synset_name(hypernym.name())
        for lemma in hypernym.lemmas():
          lemma_full = f"{lemma.name()}.{pos}.{synset_index}"
          edges.add((lemma_full, synset.name()))

  for parent, child in edges:
    g.add_edge(parent, child)

  while prune and (nodes := [n for n, d in g.out_degree() if d == 1]):
    node = nodes[0]
    child = next(g.neighbors(node))
    for parent in g.predecessors(node):
      g.add_edge(parent, child)
    g.remove_node(node)

  if construct_dag and not nx.is_directed_acyclic_graph(g):
    raise ValueError("The resulting graph is not DAG!")
  logging.info("Graph: %s", g)

  return nx.freeze(g)
