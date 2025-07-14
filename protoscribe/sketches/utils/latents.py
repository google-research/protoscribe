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

"""Core utilities related to variational auto-encoders (VAE)."""

import dataclasses
import enum

import chex
from flax import typing
import jax
import jax.numpy as jnp
import numpy as np

Array = typing.Array


def sample_gaussian(
    z_mean: Array,
    z_log_var: Array,
    rng: Array | None = None,
    z_eps: Array | None = None,
) -> Array:
  """Take one sample from each multivariate Gaussian. Reparameterization trick.

  Args:
    z_mean: <float>[..., N] multivariate Gaussian mean(s).
    z_log_var: <float>[..., N] log of Gaussian variance(s).
    rng: Pseudo-randomness to use for sampling. May be None, in which case
      the function expects a valid `z_eps` below.
    z_eps: <float>[..., N] normally distributed epsilon noise.

  Returns:
    <float>[..., N] One sample from drawn from each multivariate Gaussian.
  """
  chex.assert_equal_shape([z_mean, z_log_var])
  if rng is not None:
    eps = jax.random.normal(rng, shape=z_mean.shape, dtype=z_mean.dtype)
  elif z_eps is not None:
    chex.assert_equal_shape([z_mean, z_eps])
    eps = z_eps
  else:
    eps = jnp.zeros_like(z_log_var, dtype=z_log_var.dtype)
  z_std_dev = jnp.exp(0.5 * z_log_var)
  return z_mean + z_std_dev * eps


def kl_regularization_loss(
    z_mean: Array,
    z_log_var: Array,
    weights: Array | None = None
) -> Array:
  """KL-divergence to zero-mean unit-variance Gaussian.

  TODO: The weights should be computed from the input sequence in the
  trainer if the latent variable rank > 2.

  Args:
    z_mean: <float>[..., N] Gaussian means.
    z_log_var: <float>[..., N] Log of Gaussian variance.
    weights: Sequence mask <int>[...].

  Returns:
    A tuple consiting of
      - <float>[...] the KL divergence between each of the multivariate
      Gaussians parameterized by the final dimension of mean/log_var and
      the zero-mean, unit-variance Gaussian.
      - normalizing factor computed from the weights.
  """
  chex.assert_equal_shape([z_mean, z_log_var])
  kld = 0.5 * jnp.sum(
      jnp.square(z_mean) + jnp.exp(z_log_var) - z_log_var - 1, axis=-1
  )
  normalizing_factor = np.prod(z_mean.shape)
  if weights is not None:
    kld *= weights
    normalizing_factor = weights.sum()
  return jnp.sum(kld), normalizing_factor


@enum.unique
class AnnealingType(enum.StrEnum):
  """Type of the annealing."""

  NONE = "none"
  LINEAR = "linear"
  COSINE = "cosine"
  LOGISTIC = "logistic"
  LIU_2019 = "liu_2019"  # DOI: 10.1109/IJCNN.2019.8852155


@dataclasses.dataclass
class KLAnnealing:
  """Annealing weights for KL-divergence.

  For cyclical annealing see `Cyclical Annealing Schedule: A Simple Approach
  to Mitigating KL Vanishing` (https://arxiv.org/abs/1903.10145).

  Based on https://github.com/hubertrybka/vae-annealing.
  """

  annealing_type: AnnealingType
  total_num_steps: int
  cyclical: bool
  current_step: int = 0
  weight_initial: float = 0.

  def step(self) -> float:
    """Returns KL weight for the current step and updates the step."""
    if self.annealing_type == AnnealingType.NONE:
      return 1.
    elif self.annealing_type == AnnealingType.LINEAR:
      y = self.current_step / self.total_num_steps
    elif self.annealing_type == AnnealingType.COSINE:
      y = (
          np.cos(np.pi * (self.current_step / self.total_num_steps - 1.)) + 1
      ) / 2.
      y = y.item()
    elif self.annealing_type == AnnealingType.LOGISTIC:
      exponent = (self.total_num_steps / 2.) - self.current_step
      y = 1. / (1. + np.exp(exponent))
      y = y.item()
    elif self.annealing_type == AnnealingType.LIU_2019:
      # See Liu, D. and Liu, G. (2019): "A Transformer-Based Variational
      # Autoencoder for Sentence Generation".
      # https://ieeexplore.ieee.org/document/8852155
      exponent = np.exp(-0.0025 * self.current_step + 6.25)
      y = 1. / (1. + exponent)
      y = y.item()
    else:
      raise ValueError("Unknown annealing type!")
    weight = y * (1. - self.weight_initial) + self.weight_initial

    # Updates the current step.
    if self.current_step < self.total_num_steps:
      self.current_step += 1
    if self.cyclical and self.current_step >= self.total_num_steps:
      self.current_step = 0

    return weight
