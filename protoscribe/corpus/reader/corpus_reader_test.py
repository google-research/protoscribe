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

"""Simple tests for the dataset reader."""

import logging
import os

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import ml_collections
from protoscribe.corpus.reader import corpus_reader as lib
from protoscribe.glyphs import glyph_vocab as glyph_lib
import tensorflow.compat.v1 as tf

from google.protobuf import text_format
import glob
import os

FLAGS = flags.FLAGS

_MAX_STROKE_SEQUENCE_LENGTH = 50
_MAX_GLYPH_SEQUENCE_LENGTH = 20
_MAX_PHONETIC_SEQUENCE_LENGTH = 10
_MAX_SPEECH_FRAME_SEQUENCE_LENGTH = 100

_TEST_DATA_DIR = (
    "protoscribe/corpus/reader/testdata"
)


class CorpusReaderTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters("train", "validation", "test")
  def test_features(self, split_name):
    logging.info("Checking corpus split `%s` ...", split_name)

    # Create default dummy configuration.
    config = ml_collections.FrozenConfigDict({
        "concept_embedding_type": "bnc",
        "manual_padding": True,  # Pad sequences.
        "max_stroke_sequence_length": _MAX_STROKE_SEQUENCE_LENGTH,
        "max_glyph_sequence_length": _MAX_GLYPH_SEQUENCE_LENGTH,
        "max_phonetic_sequence_length": _MAX_PHONETIC_SEQUENCE_LENGTH,
        "max_speech_frame_sequence_length": _MAX_SPEECH_FRAME_SEQUENCE_LENGTH,
        "stroke_random_scale_factor": 0.,
        "stroke_combine_with_glyphs": False,
        "speech_framework_type": "dmvr",
        "speech_corpus_sample_rate": 16_000,
        "speech_frame_length_ms": 25.,
        "speech_frame_step_ms": 10.,
        "speech_num_mel_channels": 128,
        "speech_normalize_waveform": False,
        "speech_keep_waveform": False,
    })

    # Read and parse the example documents.
    path = os.path.join(
        FLAGS.test_srcdir, _TEST_DATA_DIR, f"{split_name}_example.textproto"
    )
    logging.info("Reading %s ...", path)
    with open(path, mode="rt") as f:
      example_proto = text_format.Parse(f.read(), tf.train.Example())

    features = lib.parse_example(
        example=example_proto.SerializeToString(),
        config=config,
        sketch_stroke_stats=dict(),
        stroke_tokenizer=None,
        speech_tokenizer=None,
        is_training=False
    )

    # Check text.
    self.assertIn("text/text", features)
    text = str(features["text/text"])
    self.assertNotEmpty(text)

    # Combined number/concept embedding.
    self.assertIn("text/concept_embedding", features)
    embedding = features["text/concept_embedding"]
    self.assertLen(embedding.shape, 2)
    self.assertEqual(2, embedding.shape[0])
    self.assertGreater(embedding.shape[1], 1)

    # Individual number/concept embeddings.
    self.assertIn("text/bnc/number_emb", features)
    embedding = features["text/bnc/number_emb"]
    self.assertLen(embedding.shape, 2)
    self.assertGreater(embedding.shape[1], 1)
    self.assertIn("text/bnc/concept_emb", features)
    embedding = features["text/bnc/concept_emb"]
    self.assertLen(embedding.shape, 2)
    self.assertGreater(embedding.shape[1], 1)

    # Check categorical glyphs.
    self.assertIn("text/glyph/tokens", features)
    glyph_tokens = features["text/glyph/tokens"]
    self.assertLen(glyph_tokens.shape, 1)
    self.assertEqual(glyph_tokens.shape[0], config.max_glyph_sequence_length)
    self.assertIn("text/glyph/types", features)
    glyph_types = features["text/glyph/types"]
    self.assertLen(glyph_types.shape, 1)
    self.assertEqual(glyph_types.shape[0], config.max_glyph_sequence_length)

    # Check phonetic embeddings.
    self.assertIn("text/phonetic_embedding", features)
    embedding = features["text/phonetic_embedding"]
    self.assertLen(embedding.shape, 2)
    self.assertEqual(_MAX_PHONETIC_SEQUENCE_LENGTH, embedding.shape[0])
    self.assertEqual(
        embedding.shape[1], lib.concept_embed_lib.DEFAULT_EMBEDDING_DIM
    )

    # Check strokes (these must be in strokes-5 format). The sequence has
    # been truncated to the specified length.
    self.assertIn("strokes", features)
    sketch = features["strokes"]
    self.assertIn("lengths", features)
    real_length = features["lengths"] + 1  # Include EOS for our test.
    self.assertLessEqual(real_length, _MAX_STROKE_SEQUENCE_LENGTH)
    self.assertLen(sketch.shape, 2)
    self.assertEqual(sketch.shape[0], _MAX_STROKE_SEQUENCE_LENGTH)
    self.assertEqual(5, sketch.shape[1])
    self.assertEqual(sketch[0][0], 0.)  # BOS.
    self.assertEqual(sketch[0][1], 0.)
    self.assertEqual(sketch[0][2], 1)
    self.assertEqual(sketch[0][3], 0)
    self.assertEqual(sketch[0][4], 0)
    self.assertEqual(sketch[real_length - 1][2], 0)  # EOS.
    self.assertEqual(sketch[real_length - 1][3], 0)
    self.assertEqual(sketch[real_length - 1][4], 1)

    # Check glyph affiliations.
    self.assertIn("sketch/glyph_affiliations/ids", features)
    glyph_affiliations = features["sketch/glyph_affiliations/ids"]
    self.assertLen(glyph_affiliations.shape, 1)
    self.assertEqual(glyph_affiliations.shape[0], _MAX_STROKE_SEQUENCE_LENGTH)
    self.assertEqual(glyph_affiliations[0], glyph_lib.GLYPH_BOS)
    self.assertEqual(glyph_affiliations[real_length - 1], glyph_lib.GLYPH_EOS)
    self.assertAllGreater(
        glyph_affiliations[1:real_length - 2], glyph_lib.GLYPH_UNK
    )

    # Check vision features, if any. This should be a single vector after
    # combining the N samples.
    for feature_name in lib.VISION_FEATURE_NAMES:
      self.assertIn(feature_name, features)
      feature = features[feature_name]
      self.assertLen(feature.shape, 2)

    # Check speech features. Do not keep the original waveform in the extracted
    # features, as configured.
    self.assertIn("speech/log_mel_spectrum", features)
    feature = features["speech/log_mel_spectrum"]
    self.assertLen(feature.shape, 2)
    self.assertNotIn("audio/waveform", features)


if __name__ == "__main__":
  absltest.main()
