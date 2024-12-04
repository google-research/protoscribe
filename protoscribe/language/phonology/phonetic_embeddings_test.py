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

"""Test for `protoscribe.language.phonology.phonetic_embeddings`."""

import os

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from protoscribe.language.phonology import phoible_segments
from protoscribe.language.phonology import phonetic_embeddings as lib

import glob
import os

flags.DEFINE_bool("gendata", False, "Regenerate test embeddings")

FLAGS = flags.FLAGS


class PhoneticEmbeddingsTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    phoible_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/data/phonology/phoible-phonemes.tsv"
    )
    features_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/data/phonology/",
        "phoible-segments-features.tsv"
    )
    word_hist_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/language/phonology/testdata",
        "word_hist.txt",
    )
    cls.freq_ordered_forms = []
    with open(word_hist_path) as stream:
      for line in stream:
        _, word = line.strip().split(" ", 1)
        cls.freq_ordered_forms.append(word)
    cls.test_embeddings_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/language/phonology/testdata",
        "phonetic_embeddings.tsv",
    )
    cls.closest_forms_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/language/phonology/testdata",
        "closest_forms.tsv",
    )
    cls.closest_forms = {}
    with open(cls.closest_forms_path) as stream:
      for line in stream:
        inp, out = line.strip().split("\t")
        cls.closest_forms[inp] = out

    cls.phoible = phoible_segments.PhoibleSegments(
        path=phoible_path, features_path=features_path
    )

  @parameterized.parameters(1, 2)
  def testClosestEmbeddings(self, norm_order: int) -> None:
    embeddings = lib.PhoneticEmbeddings(self.phoible, norm_order=norm_order)
    embeddings.build_embeddings(self.freq_ordered_forms)
    for w in self.freq_ordered_forms:
      self.assertEqual(
          self.closest_forms[w],
          embeddings.find_closest_term(w),
      )

  def testCreateReadWriteEmbeddings(self) -> None:
    embeddings = lib.PhoneticEmbeddings(self.phoible, norm_order=1)
    embeddings.build_embeddings(self.freq_ordered_forms)
    if FLAGS.gendata:
      embeddings.write_embeddings(self.test_embeddings_path)
    new_embeddings = lib.PhoneticEmbeddings(self.phoible, norm_order=1)
    new_embeddings.read_embeddings(self.test_embeddings_path)
    for w in self.freq_ordered_forms:
      old_e = embeddings.embedding(w)
      new_e = new_embeddings.embedding(w)
      for i in range(len(old_e)):
        self.assertAlmostEqual(old_e[i], new_e[i])

  def testSimpleLoad(self) -> None:
    embeddings = lib.load_phonetic_embedder(self.test_embeddings_path)
    self.assertNotEmpty(embeddings.keys)


if __name__ == "__main__":
  absltest.main()
