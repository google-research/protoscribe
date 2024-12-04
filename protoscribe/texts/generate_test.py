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

"""Testsuite for APIs used in generating accounting texts."""

import os

from absl import flags
from absl.testing import absltest
from protoscribe.texts import generate_lib as gen

import glob
import os

FLAGS = flags.FLAGS


class GenerateLibTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    FLAGS.main_lexicon = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/texts/testdata",
        "lexicon.tsv",
    )
    FLAGS.number_lexicon = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/texts/testdata",
        "number_lexicon.tsv",
    )
    FLAGS.morphology_params = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/texts/testdata",
        "morphology_params.textproto",
    )
    FLAGS.phon_rules = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/texts/testdata",
        "phon_rules.far",
    )
    FLAGS.number_phon_rules = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/texts/testdata",
        "number_phon_rules.far",
    )
    FLAGS.affix_lexicon = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/texts/testdata",
        "affixes.tsv",
    )
    FLAGS.concepts = [
        os.path.join(
            FLAGS.test_srcdir,
            "protoscribe/data/concepts",
            "administrative_categories.txt",
        )
    ]
    FLAGS.number_config_file = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/texts/configs/",
        "number_config_sg_du_pl.textproto",
    )

  def test_same_order(self) -> None:
    for _ in range(5):
      fl1 = gen.TextGenerator().generate_initial_frequency_ordered_list()
      fl2 = gen.TextGenerator.generate_initial_frequency_ordered_list_lite()
      self.assertEqual(fl1, fl2)

  def test_load_concepts(self) -> None:
    concepts_file = os.path.join(FLAGS.test_tmpdir, "concepts.txt")
    with open(concepts_file, mode="wt") as f:
      f.write("animal cat\nanimal dog\ntree\n")

    supercategories, concepts = gen.load_concepts([concepts_file])
    self.assertLen(concepts, 3)
    self.assertIn("tree", concepts)
    self.assertLen(supercategories, 2)
    self.assertIn("cat", supercategories)
    self.assertEqual(supercategories["cat"], "animal")

    exclude_concepts_file = os.path.join(
        FLAGS.test_tmpdir, "exclude_concepts.txt"
    )
    with open(exclude_concepts_file, mode="wt") as f:
      f.write("tree\n")
    gen.FLAGS.exclude_concepts_file = exclude_concepts_file
    new_supercategories, concepts = gen.load_concepts([concepts_file])
    self.assertEqual(new_supercategories, supercategories)
    self.assertLen(concepts, 2)


if __name__ == "__main__":
  absltest.main()
