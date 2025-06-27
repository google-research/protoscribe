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

from absl.testing import absltest
from protoscribe.glyphs import numbers_to_glyphs as lib


class MorphemesTest(absltest.TestCase):

  def test_pseudo_roman(self) -> None:
    self.assertEqual(lib.pseudo_roman(3), "III")
    self.assertEqual(lib.pseudo_roman(8), "VIII")
    self.assertEqual(lib.pseudo_roman(10), "X")
    self.assertEqual(lib.pseudo_roman(11), "XI")
    self.assertEqual(lib.pseudo_roman(15), "XV")
    self.assertEqual(lib.pseudo_roman(23), "XXIII")
    self.assertEqual(lib.pseudo_roman(50), "L")
    self.assertEqual(lib.pseudo_roman(68), "LXVIII")
    self.assertEqual(lib.pseudo_roman(100), "C")
    self.assertEqual(lib.pseudo_roman(159), "CLVIIII")
    self.assertEqual(lib.pseudo_roman(500), "D")
    self.assertEqual(lib.pseudo_roman(572), "DLXXII")
    self.assertEqual(lib.pseudo_roman(1000), "M")
    self.assertEqual(lib.pseudo_roman(1572), "MDLXXII")
    self.assertEqual(lib.pseudo_roman(6572), "MMMMMMDLXXII")

  def test_number_concept_split(self) -> None:
    self.assertEqual(lib.number_and_other_glyphs([]), ([], []))
    self.assertEqual(lib.number_and_other_glyphs(["I"]), (["I"], []))
    self.assertEqual(lib.number_and_other_glyphs(
        ["platypus_NOUN"]), ([], ["platypus_NOUN"]))
    self.assertEqual(lib.number_and_other_glyphs(
        ["C", "D", "I", "X", "platypus_NOUN", "tree_NOUN"]
    ), (["C", "D", "I", "X"], ["platypus_NOUN", "tree_NOUN"]))


if __name__ == "__main__":
  absltest.main()
