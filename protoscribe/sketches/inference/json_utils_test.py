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

"""Simple tests for JSONL utils."""

import os

from absl import flags
from absl.testing import absltest
from protoscribe.sketches.inference import json_utils as lib

FLAGS = flags.FLAGS

_INPUT_DICT = {
    "inputs": {
        "sketch_tokens": [1, 131, 248, 1743, 2, 171, 997, 2, 817, 1654, 2],
        "doc.id": 29404.0,
        "number.name": "8",
        "concept.name": "seam_NOUN",
        "text.sampa": "o\" p # o\" n",
        "text.words": "o p # o n"
    },
    "prediction": [
        [1, 312, 309, 309, 309, 273, 2, 0],
        [1, 312, 309, 309, 309, 78, 2, 0],
    ],
    "aux": {
        "scores": [-2.968203544616699, -1.885766625404358]
    },
}

_PRONUNCIATION_LEXICON = {
    "seam": ["a", "b", "c"],
}


class JsonUtilsTest(absltest.TestCase):

  def test_get_scorer_dict(self):
    scorer_dict = lib.get_scorer_dict(_INPUT_DICT, _PRONUNCIATION_LEXICON)
    self.assertIn("doc.id", scorer_dict)
    self.assertEqual(scorer_dict["doc.id"], 29404)
    self.assertIn("text.inputs", scorer_dict)
    self.assertEqual(scorer_dict["text.inputs"], "8 seam_NOUN")
    self.assertIn("concept.pron", scorer_dict)
    self.assertListEqual(scorer_dict["concept.pron"], ["a", "b", "c"])

  def test_glyph_pron(self):
    scorer_dict = {
        "glyph.prons": [
            [[], [], [], ["s", "a"], [], ["s", "a"]],
            [[], [], [], ["s", "a"], ["m", "o"]],
            [[], [], ["j", "a"]],
            [["t", "a", "k"], [], [], [], [], ["s", "a"]],
        ],
    }
    self.assertEqual(lib.glyph_pron(scorer_dict, k=0), "s a # s a")
    self.assertEqual(lib.glyph_pron(scorer_dict, k=1), "s a # m o")
    self.assertEqual(lib.glyph_pron(scorer_dict, k=2), "j a")
    self.assertEqual(lib.glyph_pron(scorer_dict, k=3), "t a k # s a")

  def test_get_confidence(self):
    confidence = lib.get_confidence(_INPUT_DICT)
    self.assertGreater(confidence, 0.)

  def test_load_jsonl(self):
    path = os.path.join(
        FLAGS.test_srcdir,
        (
            "protoscribe/sketches/inference/testdata"
        ),
        "infer_eval_discrete_glyphs.jsonl"
    )
    dicts = lib.load_jsonl(path)
    self.assertNotEmpty(dicts)
    doc_ids = []
    for d in dicts:
      self.assertNotEmpty(d)
      self.assertIn("inputs", d)
      self.assertIn("doc.id", d["inputs"])
      doc_ids.append(d["inputs"]["doc.id"])
    self.assertLen(dicts, len(set(doc_ids)))


if __name__ == "__main__":
  absltest.main()
