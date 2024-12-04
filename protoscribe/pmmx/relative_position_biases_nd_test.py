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

"""Tests for relative_position_biases."""

from absl import logging  # pylint: disable=unused-import
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from protoscribe.pmmx import multimodal_feature
from protoscribe.pmmx import relative_position_biases_nd

from flaxformer import sharding
from flaxformer import testing_utils
from flaxformer.components import relative_position_biases

expected_files = testing_utils.ExpectedJsonFiles(
    'protoscribe/pmmx/testdata')


class MaybeInferDimsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          size=20,
          dims=(4, 5),
          expect=(4, 5),
          expect_error=None,
      ),
      dict(
          size=20,
          dims=(4, -1),
          expect=(4, 5),
          expect_error=None,
      ),
      dict(
          size=20,
          dims=(-1, 5),
          expect=(4, 5),
          expect_error=None,
      ),
      dict(
          size=20,
          dims=(-1, -1),
          expect=None,
          expect_error=r'at most one dim in [(]-1, -1[)] may be negative',
      ),
      dict(
          size=19,
          dims=(4, 5),
          expect=None,
          expect_error=r'size=19 does not match shape=[(]4, 5[)]',
      ),
      dict(
          size=0,
          dims=(4, -1),
          expect=None,
          expect_error=r'size=0 must be positive for dim inference',
      ),
      dict(
          size=8,
          dims=(3, 2, -1),
          expect=None,
          expect_error=r'8 is not a multiple of 6',
      ),
      dict(
          size=6,
          dims=(3, 2, -1),
          expect=(3, 2, 1),
          expect_error=None,
      ),
      dict(
          size=6,
          dims=(3, 2, 0),
          expect=(3, 2, 0),
          expect_error=None,
      ),
  )
  def test_infer_shape(self, size, dims, expect, expect_error):
    if expect_error:
      with self.assertRaisesRegex(ValueError, expect_error):
        relative_position_biases_nd.infer_shape(size, dims)
    else:
      actual = relative_position_biases_nd.infer_shape(size, dims)
      self.assertSequenceEqual(actual, expect)


class RelposNDTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          feature_bounds={
              'text_tokens': (0, 5),
          },
          feature_shapes={
              'text_tokens': [-1],
          },
          expect=(np.array([
              [0, 1, 2, 3, 4],
              [-1, 0, 1, 2, 3],
              [-2, -1, 0, 1, 2],
              [-3, -2, -1, 0, 1],
              [-4, -3, -2, -1, 0],
          ]),)),
      dict(
          feature_bounds={
              'image_dense': (0, 9),
          },
          feature_shapes={
              'image_dense': [3, 3],
          },
          expect=(
              np.array([
                  [0, 1, 2, 0, 1, 2, 0, 1, 2],
                  [-1, 0, 1, -1, 0, 1, -1, 0, 1],
                  [-2, -1, 0, -2, -1, 0, -2, -1, 0],
                  [0, 1, 2, 0, 1, 2, 0, 1, 2],
                  [-1, 0, 1, -1, 0, 1, -1, 0, 1],
                  [-2, -1, 0, -2, -1, 0, -2, -1, 0],
                  [0, 1, 2, 0, 1, 2, 0, 1, 2],
                  [-1, 0, 1, -1, 0, 1, -1, 0, 1],
                  [-2, -1, 0, -2, -1, 0, -2, -1, 0],
              ]),
              np.array([
                  [0, 0, 0, 1, 1, 1, 2, 2, 2],
                  [0, 0, 0, 1, 1, 1, 2, 2, 2],
                  [0, 0, 0, 1, 1, 1, 2, 2, 2],
                  [-1, -1, -1, 0, 0, 0, 1, 1, 1],
                  [-1, -1, -1, 0, 0, 0, 1, 1, 1],
                  [-1, -1, -1, 0, 0, 0, 1, 1, 1],
                  [-2, -2, -2, -1, -1, -1, 0, 0, 0],
                  [-2, -2, -2, -1, -1, -1, 0, 0, 0],
                  [-2, -2, -2, -1, -1, -1, 0, 0, 0],
              ]),
          )),
      dict(
          feature_bounds={
              'image_dense': (0, 9),
              'text_tokens': (9, 11),
          },
          feature_shapes={
              'image_dense': [0, 3, 3],
              'text_tokens': [-1, 0, 0],
          },
          expect=(
              np.array([
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
              ]),
              np.array([
                  [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0],
                  [-1, 0, 1, -1, 0, 1, -1, 0, 1, 0, 0],
                  [-2, -1, 0, -2, -1, 0, -2, -1, 0, 0, 0],
                  [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0],
                  [-1, 0, 1, -1, 0, 1, -1, 0, 1, 0, 0],
                  [-2, -1, 0, -2, -1, 0, -2, -1, 0, 0, 0],
                  [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0],
                  [-1, 0, 1, -1, 0, 1, -1, 0, 1, 0, 0],
                  [-2, -1, 0, -2, -1, 0, -2, -1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              ]),
              np.array([
                  [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0],
                  [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0],
                  [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0],
                  [-1, -1, -1, 0, 0, 0, 1, 1, 1, 0, 0],
                  [-1, -1, -1, 0, 0, 0, 1, 1, 1, 0, 0],
                  [-1, -1, -1, 0, 0, 0, 1, 1, 1, 0, 0],
                  [-2, -2, -2, -1, -1, -1, 0, 0, 0, 0, 0],
                  [-2, -2, -2, -1, -1, -1, 0, 0, 0, 0, 0],
                  [-2, -2, -2, -1, -1, -1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              ]),
          )),
      dict(
          feature_bounds={
              'image_dense': (0, 8),  # Two images, each comprised of a 2x2 grid
              'text_tokens': (8, 11),
          },
          feature_shapes={
              'image_dense': [0, 2, 2],
              'text_tokens': [-1, 0, 0],
          },
          expect=(
              np.array([
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
                  [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, -2, -1, 0],
              ]),
              np.array([
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                  [-1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                  [-1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                  [-1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                  [-1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              ]),
              np.array([
                  [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
                  [-1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0],
                  [-1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
                  [-1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0],
                  [-1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              ]),
          )),
      dict(
          feature_bounds={
              'image_dense': (0, 9),
              'text_tokens': (9, 11),
          },
          feature_shapes={
              'image_dense': [0, 3, 3, 0],
              'text_tokens': [-1, 0, 0, 0],
          },
          expect=(
              np.array([
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
              ]),
              np.array([
                  [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0],
                  [-1, 0, 1, -1, 0, 1, -1, 0, 1, 0, 0],
                  [-2, -1, 0, -2, -1, 0, -2, -1, 0, 0, 0],
                  [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0],
                  [-1, 0, 1, -1, 0, 1, -1, 0, 1, 0, 0],
                  [-2, -1, 0, -2, -1, 0, -2, -1, 0, 0, 0],
                  [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0],
                  [-1, 0, 1, -1, 0, 1, -1, 0, 1, 0, 0],
                  [-2, -1, 0, -2, -1, 0, -2, -1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              ]),
              np.array([
                  [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0],
                  [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0],
                  [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0],
                  [-1, -1, -1, 0, 0, 0, 1, 1, 1, 0, 0],
                  [-1, -1, -1, 0, 0, 0, 1, 1, 1, 0, 0],
                  [-1, -1, -1, 0, 0, 0, 1, 1, 1, 0, 0],
                  [-2, -2, -2, -1, -1, -1, 0, 0, 0, 0, 0],
                  [-2, -2, -2, -1, -1, -1, 0, 0, 0, 0, 0],
                  [-2, -2, -2, -1, -1, -1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              ]),
              np.array([
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              ]),
          )),
      dict(
          feature_bounds={
              'image_dense': (0, 9),
              'text_tokens': (9, 11),
          },
          feature_shapes={
              'image_dense': [0, 0, 3, 3],
              'text_tokens': [-1, 0, 0, 0],
          },
          expect=(
              np.array([
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
              ]),
              np.array([
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              ]),
              np.array([
                  [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0],
                  [-1, 0, 1, -1, 0, 1, -1, 0, 1, 0, 0],
                  [-2, -1, 0, -2, -1, 0, -2, -1, 0, 0, 0],
                  [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0],
                  [-1, 0, 1, -1, 0, 1, -1, 0, 1, 0, 0],
                  [-2, -1, 0, -2, -1, 0, -2, -1, 0, 0, 0],
                  [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0],
                  [-1, 0, 1, -1, 0, 1, -1, 0, 1, 0, 0],
                  [-2, -1, 0, -2, -1, 0, -2, -1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              ]),
              np.array([
                  [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0],
                  [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0],
                  [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0],
                  [-1, -1, -1, 0, 0, 0, 1, 1, 1, 0, 0],
                  [-1, -1, -1, 0, 0, 0, 1, 1, 1, 0, 0],
                  [-1, -1, -1, 0, 0, 0, 1, 1, 1, 0, 0],
                  [-2, -2, -2, -1, -1, -1, 0, 0, 0, 0, 0],
                  [-2, -2, -2, -1, -1, -1, 0, 0, 0, 0, 0],
                  [-2, -2, -2, -1, -1, -1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              ]),
          )),
      dict(
          feature_bounds={
              'image_dense': (0, 12),
              'text_tokens': (12, 15),
          },
          feature_shapes={
              'image_dense': [0, -1, 2, 2],
              'text_tokens': [-1, 0, 0, 0],
          },
          expect=(
              np.array([
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, 0],
              ]),
              np.array([
                  [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 0],
                  [-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, 0, 0, 0],
                  [-2, -1, 0, -2, -1, 0, -2, -1, 0, -2, -1, 0, 0, 0, 0],
                  [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 0],
                  [-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, 0, 0, 0],
                  [-2, -1, 0, -2, -1, 0, -2, -1, 0, -2, -1, 0, 0, 0, 0],
                  [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 0],
                  [-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, 0, 0, 0],
                  [-2, -1, 0, -2, -1, 0, -2, -1, 0, -2, -1, 0, 0, 0, 0],
                  [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 0],
                  [-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, 0, 0, 0],
                  [-2, -1, 0, -2, -1, 0, -2, -1, 0, -2, -1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              ]),
              np.array([
                  [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                  [-1, -1, -1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0],
                  [-1, -1, -1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0],
                  [-1, -1, -1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                  [-1, -1, -1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0],
                  [-1, -1, -1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0],
                  [-1, -1, -1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              ]),
              np.array([
                  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              ]),
          ),
      ),
      dict(
          feature_bounds={
              'image_dense': (0, 12),
              'text_tokens': (12, 15),
          },
          feature_shapes={
              'image_dense': [0, 2, 2, -1],
              'text_tokens': [-1, 0, 0, 0],
          },
          expect=(
              np.array([
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, 0],
              ]),
              np.array([
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                  [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                  [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                  [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                  [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                  [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                  [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              ]),
              np.array([
                  [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
                  [-1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0],
                  [-1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
                  [-1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0],
                  [-1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
                  [-1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0],
                  [-1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              ]),
              np.array([
                  [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0],
                  [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0],
                  [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0],
                  [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0],
                  [-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                  [-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                  [-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                  [-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                  [-2, -2, -2, -2, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0],
                  [-2, -2, -2, -2, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0],
                  [-2, -2, -2, -2, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0],
                  [-2, -2, -2, -2, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              ]),
          ),
      ),
      dict(
          feature_bounds={
              'image_dense': (0, 12),
              'text_tokens': (12, 15),
          },
          feature_shapes={
              'image_dense': [0, 2, 2, 0],
              'text_tokens': [-1, 0, 0, 0],
          },
          expect=(
              np.array([
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, 0],
              ]),
              np.array([
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                  [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                  [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                  [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                  [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                  [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                  [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              ]),
              np.array([
                  [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
                  [-1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0],
                  [-1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
                  [-1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0],
                  [-1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
                  [-1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0],
                  [-1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              ]),
              np.array([
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              ]),
          )),
  )
  def test_relpos_nd(self, feature_bounds, feature_shapes, expect):
    actual = relative_position_biases_nd.relpos_nd(
        feature_bounds, feature_shapes, computation_module=np)
    for a in actual:
      logging.info('%s', a)
    np.testing.assert_allclose(actual, expect)


class RelposWeightsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          feature_bounds={
              'text_tokens': (0, 5),
          },
          feature_shapes={
              'text_tokens': [-1],
          },
          expect=(
              np.full([5, 5], 1.0, dtype=np.float32),
          )
      ),
      dict(
          feature_bounds={
              'image_dense': (0, 9),
          },
          feature_shapes={
              'image_dense': [3, 3],
          },
          expect=(
              np.full([9, 9], .5, dtype=np.float32),
              np.full([9, 9], .5, dtype=np.float32)
          )
      ),
      dict(
          feature_bounds={
              'image_dense': (0, 9),
              'text_tokens': (9, 11),
          },
          feature_shapes={
              'image_dense': [0, 3, 3],
              'text_tokens': [-1, 0, 0],
          },
          expect=(
              # On the Text-axis, weight=1.0 because text is 1D
              np.pad(np.full([2, 2], 1.0, dtype=np.float32),
                     [(9, 0), (9, 0)]),
              # On the Spatial X-axis, weight=0.5 because image is 2D
              np.pad(np.full([9, 9], 0.5, dtype=np.float32),
                     [(0, 2), (0, 2)]),
              # On the Spatial Y-axis, weight=0.5 because image is 2D
              np.pad(np.full([9, 9], 0.5, dtype=np.float32),
                     [(0, 2), (0, 2)]),
          )
      ),
  )
  def test_relpos_weights(self, feature_bounds, feature_shapes, expect):
    actual = relative_position_biases_nd.relpos_weights(
        feature_bounds, feature_shapes, dtype=np.float32, computation_module=np)
    np.testing.assert_allclose(actual, expect)

  @parameterized.parameters(
      dict(
          feature_bounds={
              'image_dense': (0, 12),
              'text_tokens': (12, 15),
          },
          feature_shapes={
              'image_dense': [0, 2, 2, -1],
              'text_tokens': [-1, 0, 0, 0],
          },
          feature_weights={
              'image_dense': [0, 0.5, 0.5, 1],
          },
          expect=(
              # On the Text-axis, weight=1.0 because text is 1D
              np.pad(
                  np.full([3, 3], 1.0, dtype=np.float32), [(12, 0), (12, 0)]
              ),
              # On the Spatial X-axis, assigned weight=0.5
              np.pad(
                  np.full([12, 12], 0.5, dtype=np.float32), [(0, 3), (0, 3)]
              ),
              # On the Spatial Y-axis, assigned weight=0.5
              np.pad(
                  np.full([12, 12], 0.5, dtype=np.float32), [(0, 3), (0, 3)]
              ),
              # On the Spatial T-axis, assigned weight=1.0
              np.pad(
                  np.full([12, 12], 1.0, dtype=np.float32), [(0, 3), (0, 3)]
              ),
          ),
      ),
  )
  def test_relpos_weights_with_assigned_weights(
      self, feature_bounds, feature_shapes, feature_weights, expect
  ):
    actual = relative_position_biases_nd.relpos_weights(
        feature_bounds,
        feature_shapes,
        dtype=np.float32,
        computation_module=np,
        feature_weights=feature_weights,
    )
    np.testing.assert_allclose(actual, expect)


class RelativePositionBiasesTest(absltest.TestCase):

  def setUp(self):
    self.num_heads = 3
    self.seq_len = 7
    self.text_sequence_metadata = multimodal_feature.SequenceMetadata(
        modality_segment_ids=np.array([], dtype=np.int32),  # dummy value
        feature_name_to_segment_id_map={},  # dummy value
        feature_name_to_bounds_map={'text_tokens': (0, self.seq_len)}
    )
    self.feature_shapes = {
        'text_tokens': (1, 1, -1),
        'image_dense': (16, 16, 1),
    }
    self.relative_attention = (
        relative_position_biases_nd.RelativePositionBiasesND(
            num_buckets=12,
            max_distance=10,
            num_heads=3,
            shape_dim_names=('_x', '_y', ''),
            dtype=jnp.float32,
            feature_shapes=self.feature_shapes))
    super().setUp()

  def test_relative_attention_renamed_head_axis(self):
    """Tests that the head axis renaming is as expected."""
    self.relative_attention = (
        relative_position_biases_nd.RelativePositionBiasesND(
            num_buckets=12,
            max_distance=10,
            num_heads=3,
            shape_dim_names=('_x', '_y', ''),
            dtype=jnp.float32,
            head_axis_name='relpos_heads',
            feature_shapes=self.feature_shapes))
    variables = self.relative_attention.init(
        random.PRNGKey(0), self.seq_len, self.seq_len,
        self.text_sequence_metadata)
    sharding.check_params_and_axis_names_match(variables)
    for axis_names in jax.tree.leaves(sharding.get_axis_names(variables)):
      for axis_name in axis_names:
        self.assertIn(axis_name, {'relpos_heads', 'relpos_buckets'})
    expected_files.check_params_and_axes(
        variables['params'],
        variables['params_axes'],
        'relpos_bias_3d_renamed_head_axis.json')

  def test_relative_attention_bidirectional_params(self):
    """Tests that bidirectional relative position biases have expected params."""
    params = self.relative_attention.init(
        random.PRNGKey(0),
        self.seq_len,
        self.seq_len,
        self.text_sequence_metadata,
        bidirectional=True,
        mutable=['params'])
    param_shapes = jax.tree.map(lambda x: x.shape, params)
    self.assertEqual(param_shapes, {
        'params': {
            'rel_embedding_x': (3, 12),
            'rel_embedding_y': (3, 12),
            'rel_embedding': (3, 12),
        },
    })

  def test_regression_relative_attention_bidirectional_values(self):
    """Tests that ND module is backwards compatible with 1D text module."""
    seqlen = 5
    num_heads = 3
    relative_attention_1d = relative_position_biases.RelativePositionBiases(
        num_buckets=12,
        max_distance=10,
        num_heads=num_heads,
        dtype=jnp.float32)
    relative_attention_nd = (
        relative_position_biases_nd.RelativePositionBiasesND(
            num_buckets=12,
            max_distance=10,
            num_heads=num_heads,
            shape_dim_names=('', '_x', '_y'),
            dtype=jnp.float32,
            feature_shapes={
                'text_tokens': (-1, 0, 0)
            }))
    text_sequence_metadata = multimodal_feature.SequenceMetadata(
        modality_segment_ids=np.array([], dtype=np.int32),  # dummy value
        feature_name_to_segment_id_map={},  # dummy value
        feature_name_to_bounds_map={'text_tokens': (0, seqlen)}
    )
    outputs_nd, params = relative_attention_nd.init_with_output(
        random.PRNGKey(0), seqlen, seqlen,
        text_sequence_metadata, bidirectional=True)
    outputs_1d = relative_attention_1d.apply(params, seqlen, seqlen,
                                             bidirectional=True)
    self.assertEqual(outputs_nd.shape, (1, num_heads, seqlen, seqlen))
    self.assertEqual(outputs_1d.shape, (1, num_heads, seqlen, seqlen))
    np.testing.assert_allclose(outputs_1d, outputs_nd)

  def test_relative_attention_unidirectional_params(self):
    """Tests that unidirectional relative position biases have expected params."""
    params = self.relative_attention.init(
        random.PRNGKey(0),
        self.seq_len,
        self.seq_len,
        self.text_sequence_metadata,
        bidirectional=False,
        mutable=['params'])
    param_shapes = jax.tree.map(lambda x: x.shape, params)
    self.assertEqual(param_shapes, {
        'params': {
            'rel_embedding_x': (3, 12),
            'rel_embedding_y': (3, 12),
            'rel_embedding': (3, 12),
        },
    })

  def test_relative_attention_decode_cache_error_with_init(self):
    """Tests that relative embedding init fails with decode == True."""
    with self.assertRaisesRegex(
        ValueError,
        'decode-mode cannot be enabled during init. use model.apply to '
        'initialize the decoding cache.'):
      self.relative_attention.init(
          jax.random.PRNGKey(0),
          self.seq_len,
          self.seq_len,
          self.text_sequence_metadata,
          bidirectional=False,
          decode=True)

  def test_relative_attention_decode_cache_errror_with_bidirectional(self):
    """Tests that bidirectional relative embeddings fails when decoding."""
    params = self.relative_attention.init(
        jax.random.PRNGKey(0),
        self.seq_len,
        self.seq_len,
        self.text_sequence_metadata,
        bidirectional=False,
        decode=False)

    with self.assertRaisesRegex(
        ValueError,
        'bidirectional RelativePositionBiases are not supported when decode=True.'
    ):
      self.relative_attention.apply(
          params,
          self.seq_len,
          self.seq_len,
          self.text_sequence_metadata,
          bidirectional=True,
          decode=True,
          mutable=['cache'])

  def test_relative_attention_decode_cache(self):
    """Tests that relative embeddings are correctly cached when decode=True."""

    params = self.relative_attention.init(
        jax.random.PRNGKey(0),
        self.seq_len,
        self.seq_len,
        self.text_sequence_metadata,
        bidirectional=False,
        decode=False)

    # during init, cache is not actually initialized.
    self.assertNotIn('cache', params)

    outputs, state = self.relative_attention.apply(
        params,
        self.seq_len,
        self.seq_len,
        self.text_sequence_metadata,
        bidirectional=False,
        decode=True,
        mutable=['cache'])

    self.assertEqual(outputs.shape,
                     (1, self.num_heads, self.seq_len, self.seq_len))

    self.assertIn('cached_bias', state['cache'])

    cached_bias = state['cache']['cached_bias']

    np.testing.assert_array_equal(outputs, state['cache']['cached_bias'])

    params_with_cache = {
        **params,
        **state,
    }

    outputs, state = self.relative_attention.apply(
        params_with_cache,
        self.seq_len,
        self.seq_len,
        self.text_sequence_metadata,
        bidirectional=False,
        decode=True,
        mutable=['cache'])

    np.testing.assert_array_equal(cached_bias, state['cache']['cached_bias'])


if __name__ == '__main__':
  absltest.main()
