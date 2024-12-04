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

"""Test for `protoscribe.language.phonology.rebus_embeddings`."""

import logging
import os

from absl import flags
from absl.testing import absltest
from protoscribe.language.phonology import phoible_segments
from protoscribe.language.phonology import rebus_embeddings

import glob
import os

_REGENERATE_DATA = flags.DEFINE_bool(
    "regenerate_data", False, "Regenerate test embeddings."
)

FLAGS = flags.FLAGS


class RebusTest(absltest.TestCase):

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
    cls.phoible = phoible_segments.PhoibleSegments(
        path=phoible_path, features_path=features_path
    )
    cls.rebus = rebus_embeddings.RebusEmbeddings(cls.phoible)
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
        "rebus_embeddings.tsv",
    )

  def testRebusMatches(self) -> None:
    self.assertTrue(self.rebus.match_words("b a l k", "b a k"))
    self.assertTrue(self.rebus.match_words("p a l k", "b a k"))
    self.assertTrue(self.rebus.match_words("m a l k", "b a k"))
    self.assertTrue(self.rebus.match_words("a l k", "a k"))
    self.assertTrue(self.rebus.match_words("a l k", "a ɡ"))
    self.assertFalse(self.rebus.match_words("p a l k", "b i k"))
    self.assertFalse(self.rebus.match_words("a l k", "b a k"))
    self.assertFalse(self.rebus.match_words("m o p", "m e b"))

  def testDistancesMakeSense(self) -> None:
    """Tests that the distances have the right properties.

    It should always be the case that if w2 could be a rebus of w1, then the
    embedding for w2 should be closer to the embedding of w1, than the embedding
    of any w3 which cannot be a rebus for w1.
    """
    rebus = rebus_embeddings.RebusEmbeddings(self.phoible)
    rebus.read_embeddings(self.test_embeddings_path)
    cache = {}

    def cached_dist(w1, w2):
      if (w1, w2) not in cache:
        cache[w1, w2] = rebus.distance(rebus.embedding(w1), rebus.embedding(w2))
      return cache[w1, w2]

    def all_less(seq1, seq2):
      for w11, w12, s1 in seq1:
        for w21, w22, s2 in seq2:
          msg = f"{w11}, {w12}\t{w21}, {w22}"
          self.assertLessEqual(s1, s2, msg=msg)

    for w1 in self.freq_ordered_forms:
      ingroup = []
      outgroup = []
      for w2 in self.freq_ordered_forms:
        if rebus.match_words(w1, w2):
          ingroup.append((w1, w2, cached_dist(w1, w2)))
        else:
          outgroup.append((w1, w2, cached_dist(w1, w2)))
      all_less(ingroup, outgroup)


if __name__ == "__main__":
  absltest.main()
