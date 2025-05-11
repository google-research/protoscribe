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

"""Utilities for continuous Bernoulli distribution.

Sources:
[1] `The continuous Bernoulli: fixing a pervasive error in variational
  autoencoders`, Gabriel Loaiza-Ganem and John P. Cunningham, NeurIPS 2019.
  https://arxiv.org/abs/1907.06845
[2] `The continuous categorical: a novel simplex-valued exponential family`
  Elliott Gordon-Rodriguez, Gabriel Loaiza-Ganem, and John P. Cunningham
  (2020). https://arxiv.org/pdf/2002.08563
[3] `Uses and Abuses of the Cross-Entropy Loss: Case Studies in Modern Deep
  Learning`. Elliott Gordon-Rodriguez, Gabriel Loaiza-Ganem, Geoff Pleiss,
  and John P. Cunningham (2020). https://arxiv.org/pdf/2011.05231

TODO: The APIs here don't quite work in a real scenario exhibiting
  numerical instabilities. Fix.
"""

import jax
import jax.numpy as jnp
from t5x import losses

JTensor = jnp.ndarray


def clamp_probs(
    probs: JTensor, eps: float = 1e-6
) -> JTensor:
  """Brings all probabilities in range [eps, 1-eps]."""
  return jnp.clip(probs, min=eps, max=1. - eps)


def cb_log_norm_const(
    probs: JTensor,
    lower_lim: float = 0.499,
    upper_lim: float = 0.501
) -> JTensor:
  """Computes the log normalizing constant for probabilities.

  Implementation from `torch.distributions.ContinuousBernoulli`.

  While the normalizing constant `C(probs)` is a continuous function of `probs`
  (even at `probs = 0.5`), computing it at values close to 0.5 can result in
  numerical instabilities due to 0/0 errors. A Taylor approximation of
  `C(probs)` is thus used for values of `probs` in a small interval around 0.5.

  Args:
    probs: Array with values in (0, 1). Note: these values should be clamped.
      Each element in the array parameterizes an independent continuous
      Bernoulli distribution. In other words, the values along the last
      dimension do not necessarily have to form a probability distribution (sum
      to 1).
    lower_lim: The safe upper limit above 0.5.
    upper_lim: The safe lower limit below 0.5.

  Returns:
    Array with the same dimension as `probs`.
  """

  outside_unstable_region = jnp.maximum(probs <= lower_lim, probs > upper_lim)
  cut_probs = jnp.where(
      outside_unstable_region, probs, lower_lim * jnp.ones_like(probs)
  )
  cut_probs_below_half = jnp.where(
      cut_probs <= 0.5, cut_probs, jnp.zeros_like(cut_probs)
  )
  cut_probs_above_half = jnp.where(
      cut_probs >= 0.5, cut_probs, jnp.ones_like(cut_probs)
  )
  log_norm = jnp.log(
      jnp.abs(jnp.log1p(-cut_probs) - jnp.log(cut_probs))
  ) - jnp.where(
      cut_probs <= 0.5,
      jnp.log1p(-2.0 * cut_probs_below_half),
      jnp.log(2.0 * cut_probs_above_half - 1.0),
  )
  x = jnp.pow(probs - 0.5, 2)
  taylor = jnp.log(2.0) + (4.0 / 3.0 + 104.0 / 45.0 * x) * x
  return jnp.where(outside_unstable_region, log_norm, taylor)


def cb_cross_entropy_with_logits(
    logits: JTensor, targets: JTensor
) -> JTensor:
  """Computes cross entropy loss with continuous Bernoulli correction.

  Computes a stabilized-gradient version of:
    -jnp.sum(targets * nn.log_softmax(logits), axis=-1)

  Args:
    logits: [batch, length, num_classes] float array.
    targets: Soft labels float array with size [batch, length, num_classes].

  Returns:
    Float array with size [batch, length].
  """
  clamped_probs = clamp_probs(jax.nn.softmax(logits))
  log_norm_c = cb_log_norm_const(clamped_probs)
  log_norm_c = -jnp.sum(log_norm_c, axis=-1)
  xent_loss, _ = losses.cross_entropy_with_logits(logits, targets, z_loss=0.)
  return log_norm_c + xent_loss


def cb_kl(
    p_logits: JTensor,
    q_probs: JTensor,
    lower_lim: float = 0.499,
    upper_lim: float = 0.501
) -> JTensor:
  """Computes KL divergence between two continuous Bernoulli distributions.

  Args:
    p_logits: Logits for the p distribution.
    q_probs: Array with values in (0, 1). Note: these values should be clamped.
      Each element in the array parameterizes an independent continuous
      Bernoulli distribution. In other words, the values along the last
      dimension do not necessarily have to form a probability distribution (sum
      to 1).
    lower_lim: The safe upper limit above 0.5.
    upper_lim: The safe lower limit below 0.5.

  Returns:
    Float array with the same dimension as `p_logits` and `q_probs`.
  """
  p_probs = clamp_probs(jax.nn.softmax(p_logits, axis=-1))
  q_probs = clamp_probs(q_probs)
  q_logits = jnp.log(q_probs)

  # Compute \mu(p).
  outside_unstable_region = jnp.maximum(
      p_probs <= lower_lim, p_probs > upper_lim)
  cut_probs = jnp.where(
      outside_unstable_region, p_probs, lower_lim * jnp.ones_like(p_probs)
  )
  mus = cut_probs / (2.0 * cut_probs - 1.0) + 1.0 / (
      jnp.log1p(-cut_probs) - jnp.log(cut_probs)
  )
  x = p_probs - 0.5
  taylor = 0.5 + (1.0 / 3.0 + 16.0 / 45.0 * jnp.pow(x, 2)) * x
  p_mean = jnp.where(outside_unstable_region, mus, taylor)

  # Compute KL(p||q).
  t1 = p_mean * (p_logits - q_logits)
  p_const = cb_log_norm_const(p_probs, lower_lim, upper_lim)
  q_const = cb_log_norm_const(q_probs, lower_lim, upper_lim)
  t2 = p_const + jnp.log1p(-p_probs)
  t3 = -q_const - jnp.log1p(-q_probs)
  return t1 + t2 + t3
