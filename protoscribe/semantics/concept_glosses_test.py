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

"""Test for concept_glosses."""

from absl.testing import absltest
from protoscribe.semantics import concept_glosses


class GlossesTest(absltest.TestCase):

  def test_read_glosses(self):
    all_glosses = concept_glosses.read_glosses()
    self.assertNotEmpty(all_glosses)
    self.assertIn("administrative_categories", all_glosses)
    admin_glosses = all_glosses["administrative_categories"]
    self.assertIn("apple", admin_glosses)
    self.assertNotEmpty(admin_glosses["apple"])
    self.assertIn("non_administrative_categories", all_glosses)
    non_admin_glosses = all_glosses["non_administrative_categories"]
    self.assertIn("wash", non_admin_glosses)
    self.assertNotEmpty(admin_glosses["apple"])

  def test_find(self):
    all_glosses = concept_glosses.read_glosses()
    gloss = concept_glosses.find_gloss("apple", all_glosses)
    self.assertNotEmpty(gloss)
    gloss = concept_glosses.find_gloss("wash", all_glosses)
    self.assertNotEmpty(gloss)
    gloss = concept_glosses.find_gloss("aardvark", all_glosses)
    self.assertIs(gloss, None)

  def test_find_with_restrict(self):
    all_glosses = concept_glosses.read_glosses()
    gloss = concept_glosses.find_gloss(
        "apple", all_glosses, restrict_to_category="administrative_categories")
    self.assertNotEmpty(gloss)
    gloss = concept_glosses.find_gloss(
        "apple", all_glosses,
        restrict_to_category="non_administrative_categories")
    self.assertIs(gloss, None)

    gloss = concept_glosses.find_gloss(
        "wash", all_glosses,
        restrict_to_category="non_administrative_categories")
    self.assertNotEmpty(gloss)
    gloss = concept_glosses.find_gloss(
        "wash", all_glosses,
        restrict_to_category="administrative_categories")
    self.assertIs(gloss, None)


if __name__ == "__main__":
  absltest.main()
