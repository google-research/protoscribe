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

"""Common loss functions."""

import jax
import jax.numpy as jnp
import numpy as np
import optax

Array = jnp.ndarray | np.ndarray

# A small number, generally used to prevent dividing by zero.
_EPSILON = jnp.finfo(jnp.float32).eps


def mean_square_error_loss(
    predictions: Array, targets: Array, weights: Array) -> Array:
  """Computes MSE between predictions and targets."""
  numerator = jnp.sum(jnp.square(predictions - targets) * weights)
  denominator = jnp.sum(weights)
  return numerator / (denominator + _EPSILON)


def apply_label_smoothing(one_hot_targets: Array,
                          label_smoothing: float) -> Array:
  """Apply label smoothing to the one-hot targets.

  Applies label smoothing such that the on-values are transformed from 1.0 to
  `1.0 - label_smoothing + label_smoothing / num_classes`, and the off-values
  are transformed from 0.0 to `label_smoothing / num_classes`.
  https://arxiv.org/abs/1512.00567

  Note that another way of performing label smoothing (which we don't use here)
  is to take `label_smoothing` mass from the on-values and distribute it to the
  off-values; in other words, transform the on-values to `1.0 - label_smoothing`
  and the  off-values to `label_smoothing / (num_classes - 1)`.
  http://jmlr.org/papers/v20/18-789.html

  Args:
    one_hot_targets: One-hot targets for an example, a [batch, ..., num_classes]
      float array.
    label_smoothing: A scalar in [0, 1] used to smooth the labels.

  Returns:
    A float array of the same shape as `one_hot_targets` with smoothed label
    values.
  """
  on_value = 1.0 - label_smoothing
  num_classes = one_hot_targets.shape[-1]
  off_value = label_smoothing / num_classes
  one_hot_targets = one_hot_targets * on_value + off_value
  return one_hot_targets


def weighted_cross_entropy(
    logits: Array,
    targets: Array,
    weights: Array,
    label_smoothing: float | None = 0.0
) -> Array:
  """Compute weighted cross entropy and entropy for log probs and targets.

  This computes sum_(x,y) ce(x, y) for a single, potentially padded minibatch.
  If the minibatch is padded (that is it contains null examples) it is assumed
  that weights is a binary mask where 0 indicates that the example is null.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: one hot vector of shape [batch, ..., num_classes].
   weights: None or array of shape [batch x ...] (rank of one_hot_targets -1).
   label_smoothing: Label smoothing constant.

  Returns:
    Cross entropy loss computed per example, shape [batch, ...].
  """
  if logits.ndim != targets.ndim:
    raise ValueError("Incorrect shapes. Got shape %s logits and %s targets" %
                     (str(logits.shape), str(targets.shape)))
  soft_targets = apply_label_smoothing(targets, label_smoothing)
  loss = -jnp.sum(soft_targets * jax.nn.log_softmax(logits), axis=-1)
  loss = jnp.sum(
      loss * weights, axis=-1) / jnp.clip(jnp.sum(weights, axis=-1), _EPSILON)
  return jnp.mean(loss)


# Shamelessly copied from
# learning/deepmind/research/beneficial_rl/projects/trust_factory/mechanism_design/library/analysis_utils.py,
# get_gini_coefficient
def gini_loss(x: Array) -> Array:
  """Calculates the gini coefficient (jax-friendy, i.e., can be vmaped).

  Args:
    x: A unidimensional array of counts.

  Returns:
    A zero-dimensional value representing the Gini coefficient as a loss.
  """
  assert x.ndim == 1, f"Expected unidimensional array but {x.ndim}-dim array."
  outer_subtraction = x[:, None] - x[None, :]
  mean_abs_dif = jnp.abs(outer_subtraction).mean()
  # Relative mean absolute difference
  mean_x = jnp.mean(x)
  relative_mean_abs_dif = jnp.where(mean_x == 0.0, 0.0, mean_abs_dif / mean_x)
  gini_coefficient = 0.5 * relative_mean_abs_dif
  return gini_coefficient


def joint_embeddings_loss(
    logits: Array,
    targets: Array,
    embeddings: Array,
    use_semantic_embedding: bool,
    use_phonetic_embedding: bool,
    per_sequence_min: bool | None = True,
    sum_losses: bool | None = False,
    semantic_mask: Array | None = None,
    phonetic_mask: Array | None = None,
) -> tuple[Array, Array, Array]:
  """Computes the loss between logits and glyphs with reference to embeddings.

  Embeddings is a [V, 2, embedding_size] tensor mapping each glyph vocabulary
  element to one semantic and one phonetic embedding.

  Args:
    logits: a [batch, n, V] tensor of logits
    targets: a [batch, n, 2, embedding_size] float tensor of target embeddings
    embeddings: a [V, 2, embedding_size] precomputed matrix of embeddings
    use_semantic_embedding: whether to consider the first (semantic) dimension
      in the loss.
    use_phonetic_embedding: whether to consider the second (phonetic) dimension
      in the loss.
    per_sequence_min: if True, computes the min between the semantic and
      phonetic loss per sequence rather than per batch.
    sum_losses: if True, sum the two losses rather than taking the
      min. Overrides per_sequence_min since we won't be taking the min.
    semantic_mask: If not None, a [batch, n] 1,0 tensor that masks the semantic
      losses
    phonetic_mask: If not None, a [batch, n] 1,0 tensor that masks the phonetic
      losses

  Returns:
    Tuple of batch loss, semantic loss and phonetic loss.
  """
  assert embeddings.shape[0] == logits.shape[2]
  assert targets.shape[2] == embeddings.shape[1] == 2
  assert targets.shape[3] == embeddings.shape[2]
  x = jax.nn.softmax(logits, axis=2)
  matmul = jnp.einsum("ijk,klm->ijlm", x, embeddings)
  dist = optax.cosine_distance(matmul, targets, epsilon=_EPSILON)

  # Note: we are not normalizing the loss by sequence length or the batch size
  # to be compatible with the implementation of t5x cross-entropy loss.
  if semantic_mask is not None and phonetic_mask is not None:
    assert semantic_mask.shape == phonetic_mask.shape
    mask = jnp.stack([semantic_mask, phonetic_mask], axis=2)
    dist *= mask

  if use_semantic_embedding and use_phonetic_embedding:
    semantic_loss = jnp.sum(dist[:, :, 0])
    phonetic_loss = jnp.sum(dist[:, :, 1])
    if sum_losses:
      loss = semantic_loss + phonetic_loss
    elif per_sequence_min:
      componentwise_mins = jnp.min(dist, axis=2)
      loss = jnp.sum(componentwise_mins)
    else:
      loss = jnp.minimum(semantic_loss, phonetic_loss)
  elif use_semantic_embedding:
    semantic_loss = jnp.sum(dist[:, :, 0])
    phonetic_loss = jnp.array(0.)
    loss = semantic_loss
  elif use_phonetic_embedding:
    phonetic_loss = jnp.sum(dist[:, :, 1])
    semantic_loss = jnp.array(0.)
    loss = phonetic_loss
  else:
    raise ValueError("No embeddings are available")

  return loss, semantic_loss, phonetic_loss


def long_loss(
    logits: Array,
    mask: Array,
) -> Array:
  """A loss that favors longer and thus less-likely-to-be-ambiguous spellings.

  Args:
    logits: a [batch, n, V] tensor of logits
    mask: a [batch, n] 1,0 tensor that masks the losses

  Returns:
    Length loss.
  """
  masked_logits = jnp.einsum("ij,ijk->ijk", mask, logits)
  mass = jnp.sum(jnp.sum(masked_logits, axis=1), axis=1)
  loss = jnp.sum(1 / (mass + _EPSILON))
  return loss
