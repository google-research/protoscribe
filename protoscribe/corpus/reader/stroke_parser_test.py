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

"""Tests for the stroke parser."""

import os
import xml.etree.ElementTree as ET

from absl import flags
from absl.testing import absltest
import ml_collections
from protoscribe.corpus.reader import stroke_parser as lib
from protoscribe.glyphs import svg_to_strokes_lib as strokes_lib
from protoscribe.sketches.utils import stroke_stats as stroke_stats_lib
from protoscribe.sketches.utils import stroke_utils
import tensorflow as tf

from google.protobuf import text_format
import glob
import os

FLAGS = flags.FLAGS

_MAX_STROKE_SEQUENCE_LENGTH = 200


class ProtoscribeDatasetTest(tf.test.TestCase):

  def setUp(self):
    super(tf.test.TestCase, self).setUp()

    # Setup parser-specific configuration.
    self.config = ml_collections.ConfigDict()
    self.config.max_stroke_sequence_length = _MAX_STROKE_SEQUENCE_LENGTH
    self.config.stroke_random_scale_factor = 0.
    self.config.manual_padding = True  # Pad sequences.

    # Load stroke statistics.
    stats_file = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/sketches/utils/testdata",
        "stroke_stats.json"
    )
    self.stroke_stats = stroke_stats_lib.load_stroke_stats(
        self.config, stats_file
    )

    # Read example proto in `tf.Example` format.
    feature_specs = {
        "strokes/glyph_affiliations/ids": tf.io.VarLenFeature(tf.int64),
        "strokes/x_stroke_points": tf.io.VarLenFeature(tf.float32),
        "strokes/y_stroke_points": tf.io.VarLenFeature(tf.float32),
        "text/glyph/tokens": tf.io.VarLenFeature(tf.int64),
        "text/glyph/types": tf.io.VarLenFeature(tf.int64),
    }
    example_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/corpus/reader/testdata",
        "rabbit_tf_example.textproto"
    )
    with open(example_path, mode="rt") as f:
      example_proto = text_format.Parse(f.read(), tf.train.Example())
    self.features = tf.io.parse_single_example(
        example_proto.SerializeToString(), features=feature_specs
    )
    for name in self.features:
      if isinstance(self.features[name], tf.SparseTensor):
        self.features[name] = tf.sparse.to_dense(self.features[name])

  def test_parse_sketch_strokes_or_tokens(self):
    (strokes_or_tokens, sketch_glyph_affiliations, lengths) = (
        lib.parse_sketch_strokes_or_tokens(
            features=self.features,
            config=self.config,
            stroke_stats=self.stroke_stats,
            stroke_tokenizer=None,
            is_training=False,
        )
    )
    self.assertEqual(strokes_or_tokens.shape, (_MAX_STROKE_SEQUENCE_LENGTH, 5))
    self.assertEqual(
        sketch_glyph_affiliations.shape,
        (_MAX_STROKE_SEQUENCE_LENGTH,),
    )
    self.assertEqual(lengths, 65)

  def test_reconstruction(self):
    rabbit_svg_path = os.path.join(
        FLAGS.test_srcdir,
        "protoscribe/glyphs/testdata",
        "rabbit.svg"
    )
    rabbit = ET.parse(rabbit_svg_path)
    strokes, stroke_glyph_affiliations = (
        strokes_lib.svg_tree_to_strokes_for_test(
            rabbit,
            # Important: Same settings as in the corpus builder.
            flip_vertical=False,
            deltas=True,
            apply_rdp=True,
            rdp_tolerance=2.0,
            path_is_stroke=True,
            points_per_segment=100,
        )
    )
    stroke_glyph_affiliations_ids = []
    part_names = {
        "RABBIT_BODY": 0,
        "RABBIT_RUMP": 1,
        "RABBIT_EYEBALL": 2,
    }
    for text_pos, glyph_name in stroke_glyph_affiliations:
      stroke_glyph_affiliations_ids.append((text_pos, part_names[glyph_name]))
    (
        x_stroke_points,
        y_stroke_points,
        stroke_stats,
        glyph_affiliations_text_pos,
        glyph_affiliations_ids,
    ) = stroke_utils.stroke_points(strokes, stroke_glyph_affiliations_ids)
    npoints = len(x_stroke_points)
    features = {
        "strokes/npoints": tf.constant(npoints, dtype=tf.int64),
        "strokes/x_stroke_points": tf.constant(
            x_stroke_points, dtype=tf.float32,
        ),
        "strokes/y_stroke_points": tf.constant(
            y_stroke_points, dtype=tf.float32,
        ),
        "strokes/glyph_affiliations/text_pos": tf.constant(
            glyph_affiliations_text_pos, dtype=tf.int64,
        ),
        "strokes/glyph_affiliations/ids": tf.constant(
            glyph_affiliations_ids, dtype=tf.int64,
        ),
    }
    (strokes_or_tokens, _, _) = lib.parse_sketch_strokes_or_tokens(
        config=self.config,
        features=features,
        stroke_stats=stroke_stats.finalize(),
        stroke_tokenizer=None,
        is_training=False,
    )
    self.assertEqual(strokes_or_tokens.shape, (_MAX_STROKE_SEQUENCE_LENGTH, 5))
    x = strokes_or_tokens[:, 0].numpy().tolist()
    y = strokes_or_tokens[:, 1].numpy().tolist()
    pen_lifted = strokes_or_tokens[:, 3].numpy().tolist()
    output_x_stroke_points = []
    output_y_stroke_points = []
    for px, py, pen in zip(x, y, pen_lifted):
      output_x_stroke_points.append(px)
      output_y_stroke_points.append(py)
      if pen == 1:
        output_x_stroke_points.append(lib._END_OF_STROKE)
        output_y_stroke_points.append(lib._END_OF_STROKE)
    # Note that the tensors have the points shifted one to the right, hence
    # i + 1 below.
    for i in range(npoints):
      self.assertAlmostEqual(
          x_stroke_points[i], output_x_stroke_points[i + 1], places=4,
      )
    for i in range(npoints):
      self.assertAlmostEqual(
          y_stroke_points[i], output_y_stroke_points[i + 1], places=4,
      )
    # End of rabbit reconstruction.


if __name__ == "__main__":
  absltest.main()
