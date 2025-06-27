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

"""Utilities for computing perplexity.

This utility can be used by Model subclasses in their `compute_metrics` method
to plot perplexity in Tensorboard.

The `softmax_perplexity` may be used for models with softmax distribution, such
as generative encoder-decoder models.
"""

from typing import Optional
import jax
import jax.lax
import jax.numpy as jnp


def ce_perplexity(
    ce: jnp.ndarray):
  """Compute perplexity from the cross entropy."""
  return jnp.exp(ce)


def softmax_perplexity(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None):
  """Computes perplexity from softmax logits.

  If `weights` are not provided, all targets are assumed to have the same
  weight. In most PMMX models, the `weights` are automatically set by the
  MultimodalEncDecFeatureConverter, which assigns weight=0 to pad positions and
  weight=1 otherwise.

  Args:
    logits: float array of shape [batch_size, seq_len, vocab_size]
    targets: int array of shape [batch_size, seq_len]
    weights: optional weights of shape [batch_size, seq_len]

  Returns:
    perplexity of the model for the provided targets
  """
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  target_one_hot = jax.nn.one_hot(
      targets, num_classes=logits.shape[-1], dtype=logits.dtype)
  batch_dims = ((0, 1), (0, 1))  # (lhs_batch_dims, rhs_batch_dims)
  contract_dims = ((2,), (2,))  # (lhs_contract_dims, rhs_contract_dims)
  cross_entropy = -jax.lax.dot_general(
      log_probs, target_one_hot, (contract_dims, batch_dims))
  if weights is not None:
    exponent = jnp.sum(cross_entropy * weights) / jnp.sum(weights)
  else:
    exponent = jnp.mean(cross_entropy)
  return jnp.exp(exponent)
