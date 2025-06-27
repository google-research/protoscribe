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

"""Tests for `generic` WordNet parser APIs."""

from absl import flags
from absl.testing import absltest
import networkx as nx
from protoscribe.semantics.wordnet import wordnet_utils as lib

import glob
import os

FLAGS = flags.FLAGS

_DATA = {
    "adj": [
        "00098896 00 s 01 overnonviable 0 001 & 12345678 a 0000 | complicated",
    ],
    "adv": [
        "00155346 02 r 01 underastonished 0 000 | unimpressed",
    ],
    "noun": [
        "03130521 06 n 01 craft 0 000 | hovering thing",
        ("03552409 06 n 02 hovercraft 0 hingmy 0 001 @ 03130521 "
         "n 0000 | Ma hovercraft's breemin' ower wi eels"),
    ],
    "verb": [
        # Nothing.
    ],
}


class WordNetUtilsTest(absltest.TestCase):

  def test_data_lines_parser(self):
    synsets = lib.wordnet_parse_data_lines(_DATA["noun"])
    self.assertLen(synsets, 2)
    self.assertEqual(sorted(synsets.keys()), ["03130521", "03552409"])

    hovercraft_words, hovercraft_targets = synsets["03552409"]
    self.assertLen(hovercraft_words, 2)
    self.assertEqual(hovercraft_words[0], "hovercraft")
    self.assertEqual(hovercraft_words[1], "hingmy")
    self.assertLen(hovercraft_targets, 1)
    self.assertLen(hovercraft_targets[0], 3)
    self.assertEqual(hovercraft_targets[0][0], "03130521")
    self.assertEqual(hovercraft_targets[0][1], "@")
    self.assertEqual(hovercraft_targets[0][2], "n")

    craft_words, craft_targets = synsets["03130521"]
    self.assertLen(craft_words, 1)
    self.assertEqual(craft_words[0], "craft")
    self.assertEmpty(craft_targets)

  def test_load_synsets(self):
    wordnet_dir = FLAGS.test_tmpdir
    for suffix in lib.WORDNET_FILE_SUFFIX_TO_POS.keys():
      path = lib.wordnet_data_path(wordnet_dir, suffix)
      with open(path, mode="wt") as f:
        f.write("\n".join(_DATA[suffix]) + "\n")

    all_synsets = lib.load_synsets_from_data(wordnet_dir)
    self.assertLen(all_synsets, 4)
    for suffix in lib.WORDNET_FILE_SUFFIX_TO_POS.keys():
      self.assertIn(lib.WORDNET_FILE_SUFFIX_TO_POS[suffix], all_synsets)
    self.assertLen(all_synsets["a"].keys(), 1)
    self.assertLen(all_synsets["n"].keys(), 2)
    self.assertLen(all_synsets["r"].keys(), 1)
    self.assertEmpty(all_synsets["v"].keys())

  def test_parse_synset_name(self):
    self.assertEqual(
        lib.parse_synset_name("Altaic_language.n.02"),
        ("Altaic_language", "n", "02")
    )
    self.assertEqual(
        lib.parse_synset_name("I.E.D..n.01"),
        ("I.E.D.", "n", "01")
    )

  def test_simple_dag(self):
    g = lib.build_graph(
        words=["platypus.n.01"],
        pos="n",
        construct_dag=True,
        prune=False
    )
    self.assertTrue(nx.is_directed(g))
    self.assertEqual(sorted(g.nodes), [
        "animal.n.01",
        "animate_being.n.01",
        "animate_thing.n.01",
        "beast.n.01",
        "being.n.01",
        "brute.n.01",
        "chordate.n.01",
        "craniate.n.01",
        "creature.n.01",
        "egg-laying_mammal.n.01",
        "entity.n.01",
        "fauna.n.01",
        "living_thing.n.01",
        "mammal.n.01",
        "mammalian.n.01",
        "monotreme.n.01",
        "object.n.01",
        "organism.n.01",
        "physical_entity.n.01",
        "physical_object.n.01",
        "platypus.n.01",
        "prototherian.n.01",
        "unit.n.02",
        "vertebrate.n.01",
        "whole.n.02",
    ])
    self.assertEqual(sorted(g.edges), [
        ("animal.n.01", "chordate.n.01"),
        ("animate_being.n.01", "chordate.n.01"),
        ("animate_thing.n.01", "organism.n.01"),
        ("beast.n.01", "chordate.n.01"),
        ("being.n.01", "animal.n.01"),
        ("brute.n.01", "chordate.n.01"),
        ("chordate.n.01", "vertebrate.n.01"),
        ("craniate.n.01", "mammal.n.01"),
        ("creature.n.01", "chordate.n.01"),
        ("egg-laying_mammal.n.01", "platypus.n.01"),
        ("entity.n.01", "physical_entity.n.01"),
        ("fauna.n.01", "chordate.n.01"),
        ("living_thing.n.01", "organism.n.01"),
        ("mammal.n.01", "prototherian.n.01"),
        ("mammalian.n.01", "prototherian.n.01"),
        ("monotreme.n.01", "platypus.n.01"),
        ("object.n.01", "whole.n.02"),
        ("organism.n.01", "animal.n.01"),
        ("physical_entity.n.01", "object.n.01"),
        ("physical_object.n.01", "whole.n.02"),
        ("prototherian.n.01", "monotreme.n.01"),
        ("unit.n.02", "living_thing.n.01"),
        ("vertebrate.n.01", "mammal.n.01"),
        ("whole.n.02", "living_thing.n.01")
    ])


if __name__ == "__main__":
  absltest.main()
