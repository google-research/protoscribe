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

"""Helper module for multi-task loss combination."""

import logging
from typing import List, Optional, Tuple

from flax import linen as nn
import jax.numpy as jnp
import ml_collections

JTensor = jnp.ndarray


class HUWSigmaSquareLossCombiner(nn.Module):
  r"""Homoscedastic Uncertainty Weighted loss optimization (for \sigma^2).

  Described in `Multi-Task Learning Using Uncertainty to Weigh Lossesfor Scene
  Geometry and Semantics`, by A. Kendall, Y. Gal, and R. Cipolla. CVPR 2018.
  https://arxiv.org/abs/1705.07115
  """

  num_losses: int
  dtype: jnp.dtype = jnp.float32

  def setup(self) -> None:
    """Layer setup."""
    self.sigma_sq = self.param(
        "loss_weights", nn.initializers.ones, (self.num_losses,), self.dtype
    )

  @nn.compact
  def __call__(self, losses: List[JTensor]) -> Tuple[
      JTensor, JTensor, List[JTensor]]:
    """Computes combination of individual losses.

    Args:
      losses: List of tensors representing individual losses.

    Returns:
      Tuple of combined total loss tensor, the corresponding tensor of weights
      and the individual reweighted losses.
    """
    if not losses:
      raise ValueError("No losses supplied!")
    if len(losses) != self.num_losses:
      raise ValueError(
          f"Mismatching number of losses. Configured {self.num_losses} losses!"
      )

    # Observable noise scalar: \sigma^2.
    total_loss = jnp.zeros_like(losses[0], dtype=self.dtype)
    reweighted_losses = []
    for i, _ in enumerate(losses):
      factor = 0.5 / self.sigma_sq[i]
      loss = factor * losses[i] + jnp.log(self.sigma_sq[i])
      reweighted_losses.append(loss)
      total_loss += loss

    return total_loss, self.sigma_sq, reweighted_losses


class HUWLogSigmaLossCombiner(nn.Module):
  r"""Homoscedastic Uncertainty Weighted loss optimization (for \log\sigma).

  Described in `Multi-Task Learning Using Uncertainty to Weigh Lossesfor Scene
  Geometry and Semantics`, by A. Kendall, Y. Gal, and R. Cipolla. CVPR 2018.
  https://arxiv.org/abs/1705.07115
  """

  num_losses: int
  dtype: jnp.dtype = jnp.float32

  def setup(self) -> None:
    """Layer setup."""
    self.log_sigma = self.param(
        "loss_weights", nn.initializers.zeros, (self.num_losses,), self.dtype
    )

  @nn.compact
  def __call__(self, losses: List[JTensor]) -> Tuple[
      JTensor, JTensor, List[JTensor]]:
    """Computes combination of individual losses.

    Args:
      losses: List of tensors representing individual losses.

    Returns:
      Tuple of combined total loss tensor, the corresponding tensor of weights
      and the individual reweighted losses.
    """
    if not losses:
      raise ValueError("No losses supplied!")
    if len(losses) != self.num_losses:
      raise ValueError(
          f"Mismatching number of losses. Configured {self.num_losses} losses!"
      )

    # Observable noise scalar: \log\sigma.
    total_loss = jnp.zeros_like(losses[0], dtype=self.dtype)
    reweighted_losses = []
    for i, _ in enumerate(losses):
      loss = jnp.exp(-self.log_sigma[i]) * losses[i] + self.log_sigma[i]
      reweighted_losses.append(loss)
      total_loss += loss

    return total_loss, self.log_sigma, reweighted_losses


class HUWLiebelKoernerLossCombiner(nn.Module):
  """Revised Homoscedastic Uncertainty Weighted loss optimization.

  Described in Liebel L, Körner M. (2018): `Auxiliary tasks in multi-task
  learning` (Equation 2) in https://arxiv.org/pdf/1805.06334.pdf.
  Source: https://github.com/Mikoto10032/AutomaticWeightedLoss
  """

  num_losses: int
  dtype: jnp.dtype = jnp.float32

  def setup(self) -> None:
    """Layer setup."""
    self.sigma = self.param(
        "loss_weights", nn.initializers.ones, (self.num_losses,), self.dtype
    )

  @nn.compact
  def __call__(self, losses: List[JTensor]) -> Tuple[
      JTensor, JTensor, List[JTensor]]:
    """Computes combination of individual losses.

    Args:
      losses: List of tensors representing individual losses.

    Returns:
      Tuple of combined total loss tensor, the corresponding tensor of weights
      and the individual reweighted losses.
    """
    if not losses:
      raise ValueError("No losses supplied!")
    if len(losses) != self.num_losses:
      raise ValueError(
          f"Mismatching number of losses. Configured {self.num_losses} losses!"
      )

    # Observable noise scalar: \sigma.
    total_loss = jnp.zeros_like(losses[0], dtype=self.dtype)
    reweighted_losses = []
    for i, _ in enumerate(losses):
      factor = 0.5 / (self.sigma[i] ** 2)
      loss = factor * losses[i] + jnp.log(1. + self.sigma[i] ** 2)
      reweighted_losses.append(loss)
      total_loss += loss

    return total_loss, self.sigma, reweighted_losses


def get_loss_combiner(
    config: ml_collections.ConfigDict,
    num_losses: int
) -> Optional[nn.Module]:
  """Manufactures configured loss combiner."""
  if "loss_combiner_type" not in config:
    return None

  combiner_type = config.loss_combiner_type
  logging.info("Loss combiner: %s", combiner_type)
  if combiner_type == "huw_sigma_square":
    return HUWSigmaSquareLossCombiner(num_losses)
  elif combiner_type == "huw_log_sigma":
    return HUWLogSigmaLossCombiner(num_losses)
  elif combiner_type == "huw_liebel_koerner":
    return HUWLiebelKoernerLossCombiner(num_losses)
  elif combiner_type.upper() == "NONE":
    return None
  else:
    raise ValueError(f"Unknown loss combiner: {combiner_type}")
