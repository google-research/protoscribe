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

"""Noise utilities.
"""

import tensorflow as tf


def _tf_random_exponential(
    shape,
    rate: float = 1.0,
    dtype: tf.DType = tf.float32,
    seed: int | None = None
) -> tf.Tensor:
  """Samples from the exponential distribution."""
  return tf.random.gamma(shape, alpha=1, beta=1. / rate, dtype=dtype, seed=seed)


def _tf_random_laplace(
    shape,
    loc: float = 0.0,
    scale: float = 1.0,
    dtype: tf.DType = tf.float32,
    seed: int | None = None
) -> tf.Tensor:
  """Samples from Laplace distribution."""
  z1 = _tf_random_exponential(shape, loc, dtype=dtype, seed=seed)
  z2 = _tf_random_exponential(shape, scale, dtype=dtype, seed=seed)
  return z1 - z2


def tf_random_lp_ball_vector(
    shape,
    order: str,
    radius: float,
    dtype: tf.DType = tf.float32,
    seed: int | None = None
) -> tf.Tensor:
  """Samples random vectors from a norm ball of radius epsilon.

  Given a real or complex ball B(r) of radius r > 0, where ‖·‖ₚ, is the
  standard lₚ norm, the objective is to generate samples uniformly in B(r).

  Args:
    shape: Output shape of the random sample to be drawn from a norm ball of
      dimension `d1*d2*...*dn`.
    order: Order of the norm (string). Possible values: "INF", "1" or "2".
    radius: Radius (r) of the norm ball.
    dtype: Tensor data type.
    seed: Random number seed.

  Returns:
     Sampled tensor.
  """
  if order not in ["INF", "1", "2"]:
    raise ValueError(f"Unsupported order: {order}")

  if order == "INF":
    r = tf.random.uniform(
        shape, minval=-radius, maxval=radius, dtype=dtype, seed=seed
    )
  else:
    # For order=1 and order=2, we use the generic technique from (Calafiore
    # et al. 1998) to sample uniformly from a norm ball. Paper link (Calafiore
    # et al. 1998):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=758215&tag=1
    #
    # We first sample from the surface of the norm ball, and then scale by
    # a factor `w^(1/d)` where `w~U[0,1]` is a standard uniform random variable
    # and `d` is the dimension of the ball. In high dimensions, this is roughly
    # equivalent to sampling from the surface of the ball.

    dim = tf.reduce_prod(shape)

    if order == "1":
      x = _tf_random_laplace(
          [dim], loc=1.0, scale=1.0, dtype=dtype, seed=seed
      )
      norm = tf.reduce_sum(tf.abs(x))
    elif order == "2":
      # \mathcal{N}(0, 1).
      x = tf.random.normal(shape=[dim], dtype=dtype, seed=seed)
      norm = tf.sqrt(tf.reduce_sum(tf.square(x)))

    # \mathcal{U}(0, 1).
    w = tf.pow(tf.random.uniform(shape=[], dtype=dtype, seed=seed),
               1.0 / tf.cast(dim, dtype))
    r = radius * tf.reshape(w * x / norm, shape)

  return r
