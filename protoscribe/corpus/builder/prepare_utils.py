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

"""Utility library for data generators."""

import numpy as np
import tensorflow as tf

Feature = tf.train.Feature


def bytes_feature(values: list[bytes]) -> Feature:
  return Feature(bytes_list=tf.train.BytesList(value=values))


def int64_feature(values: list[int]) -> Feature:
  return Feature(int64_list=tf.train.Int64List(value=values))


def float_feature(values: list[float]) -> Feature:
  return Feature(float_list=tf.train.FloatList(value=values))


def ndarray_feature(values: np.ndarray) -> Feature:
  return Feature(float_list=tf.train.FloatList(value=values.tolist()))


def flatten_embedding(
    emb_2d: list[list[float]]
) -> tuple[list[float], list[float]]:
  """Encodes 2d regular list as a 1d list."""
  emb_len = len(emb_2d)

  # Brilliantly, tf.train only seems to support one-dimensional vectors, so we
  # end up flattening this, but preserving the dimensionality of the 2D tensor
  # in the emb_dim feature.
  emb_1d = []
  for x in emb_2d:
    emb_1d.extend(list(x))
  # TODO: Storing the original length is enough.
  emb_1d_dims = [emb_len, int(len(emb_1d) / emb_len)]
  return emb_1d_dims, emb_1d
