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

"""Python libs for distillation."""

from typing import Optional, Tuple, Union

from absl import logging  # pylint: disable=unused-import
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
from t5x import losses as t5x_losses


LossNormalizingFactor = Union[
    float, int, str, t5x_losses.SpecialLossNormalizingFactor]


def distillation_loss(student_logits,
                      teacher_logits,
                      weights,
                      loss_normalizing_factor: Optional[LossNormalizingFactor],
                      temperature=1):
  """Distillation loss from student/teacher logits.

  See also distillation loss functions in BigVision codebase:
  https://github.com/google-research/big_vision/blob/main/big_vision/evaluators/proj/distill/distance.py # pylint: disable=line-too-long

  Args:
    student_logits: [batch, length, num_classes] unnormalized student logits
    teacher_logits: [batch, length, num_classes] unnormalized teacher logits
    weights: [batch, length] weights for each sequence position derived e.g.
      from padding
    loss_normalizing_factor: normalizes the sequence loss the same way it would
      with discrete targets
    temperature: distillation hyperparameter

  Returns:
    loss scalar
  """
  t = jnp.full((), temperature, dtype=student_logits.dtype)
  loss = t**2 * _weighted_softmax_xent(
      logits=(student_logits / t),
      labels=jax.nn.softmax(teacher_logits / t),
      weights=weights)
  if loss_normalizing_factor is not None:
    loss /= loss_normalizing_factor
  return loss


def _weighted_softmax_xent(logits, labels, weights, kl: bool = True):
  """Compute weighted cross entropy.

  Args:
   logits: [batch, length, num_classes] float array.
   labels: [batch, length, num_classes] float array.
   weights: [batch, length] float array.
   kl: if using KL distance

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  if logits.shape != labels.shape:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(labels.shape)))
  if len(logits.shape) != 3:
    raise ValueError('Incorrect ranks. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(labels.shape)))
  loss = -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
  if kl:
    loss += jnp.sum(labels * jnp.log(jnp.clip(labels, 1e-8)), axis=-1)
  loss *= weights
  loss = loss.sum()
  return loss


def fix_teacher_logits(teacher_logits, decoder_target_tokens):
  """Reorders teacher logits s.t. the groundtruth ids are the top-1 prediction.

  Hyperparameter-free method for correcting inaccurate teacher logits that
  preserves the shape of the teacher distribution by reordering class
  probabilities.

  Args:
    teacher_logits: [batch, seq, vocab] logits as computed by the teacher
    decoder_target_tokens: [batch, seq] groundtruth ids

  Returns:
    array of the same shape and type as logits but reordered so that the
      groundtruth id is the top-1 logit
  """
  # Order the vocab ids from most likely to least likely.
  sort_idx = jnp.argsort(-teacher_logits, axis=-1)
  # Transpose the `sort_idx` array so that it maps from vocab id to sort index,
  # not vice-versa.
  sort_coords = jnp.lexsort((sort_idx,), axis=-1)
  decoder_target_tokens = jnp.expand_dims(decoder_target_tokens, -1)
  # Gather the sort coordinate of the groundtruth id.
  gt_sort_coord = jnp.take_along_axis(
      sort_coords, decoder_target_tokens, axis=2)
  # Reorder the coordinate of every vocab id so that the groundtruth is in the
  # top position, and every id with a greater likelihood moves from the kth
  # position to the (k+1)th position.
  reordered_coords = jnp.where(
      sort_coords < gt_sort_coord,
      sort_coords + 1,
      jnp.where(sort_coords > gt_sort_coord, sort_coords, 0))
  # Compute the new "groundtruth-corrected" teacher logits.
  sorted_teacher_logits = -jnp.sort(-teacher_logits, axis=-1)
  corrected_logits = jnp.take_along_axis(
      sorted_teacher_logits, reordered_coords, axis=-1)
  return corrected_logits


def compute_weighted_cross_entropy_total(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    loss_normalizing_factor: Optional[float] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Compute weighted cross entropy and entropy for log probs and targets.

  This is the same as t5x_losses.compute_weighted_cross_entropy, but also
  returns the total loss. The total loss is used for self-adaptive distillation.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch, length].
   label_smoothing: label smoothing constant, used to determine the on and off
     values.
   z_loss: coefficient for auxiliary z-loss loss term.
   loss_normalizing_factor: Constant to divide loss by. If not specified, loss
     will not be normalized. Intended for backward compatibility with T5-MTF
     training. Should not normally be used.

  Returns:
    Tuple of scalar loss, z_loss, weight sum, and total loss.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError(
        'Incorrect shapes. Got shape %s logits and %s targets'
        % (str(logits.shape), str(targets.shape))
    )
  vocab_size = logits.shape[-1]
  confidence = 1.0 - label_smoothing
  low_confidence = (1.0 - confidence) / (vocab_size - 1)
  normalizing_constant = -(
      confidence * jnp.log(confidence)
      + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
  )
  soft_targets = common_utils.onehot(
      targets, vocab_size, on_value=confidence, off_value=low_confidence
  )
  total_loss, total_z_loss = t5x_losses.cross_entropy_with_logits(
      logits, soft_targets, z_loss=z_loss
  )
  total_loss = total_loss - normalizing_constant

  weight_sum = np.prod(targets.shape)
  if weights is not None:
    total_loss = total_loss * weights
    total_z_loss = total_z_loss * weights
    weight_sum = jnp.sum(weights)

  # By default, we do not normalize loss based on anything.
  # We don't normalize based on batch size because the optimizers we use are
  # pretty much scale invariant, so this simplifies things.
  # We don't normalize based on number of non-padding tokens in order to treat
  # each token as equally important regardless of sequence length.
  if loss_normalizing_factor is not None:
    total_loss /= loss_normalizing_factor
    total_z_loss /= loss_normalizing_factor
  return jnp.sum(total_loss), jnp.sum(total_z_loss), weight_sum, total_loss


def losses_and_adaptive_weight(
    student_logits: jnp.ndarray,
    teacher_logits: jnp.ndarray,
    targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    loss_normalizing_factor: Optional[float] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Computes gt losses and adaptive weight for self-adaptive distillation.

  This uses student and teacher's logits to compute the adaptive weight for
  self-adaptive distillation.

  Args:
   student_logits: [batch, length, num_classes] float array.
   teacher_logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch, length].
   label_smoothing: label smoothing constant, used to determine the on and off
     values.
   z_loss: coefficient for auxiliary z-loss loss term.
   loss_normalizing_factor: Constant to divide loss by. If not specified, loss
     will not be normalized. Intended for backward compatibility with T5-MTF
     training. Should not normally be used.

  Returns:
    Tuple of scalar student_loss, student_z_loss, teacher_loss, teacher_z_loss,
    and adaptive_weight.
  """
  student_loss, student_z_loss, _, total_student_loss = (
      compute_weighted_cross_entropy_total(
          student_logits,
          targets=targets,
          weights=weights,
          label_smoothing=label_smoothing,
          z_loss=z_loss,
          loss_normalizing_factor=loss_normalizing_factor,
      )
  )

  teacher_loss, teacher_z_loss, _, total_teacher_loss = (
      compute_weighted_cross_entropy_total(
          teacher_logits,
          targets=targets,
          weights=weights,
          label_smoothing=label_smoothing,
          z_loss=z_loss,
          loss_normalizing_factor=loss_normalizing_factor,
      )
  )

  # Stop gradient on teacher.
  stop_grad_total_teacher_loss = jax.lax.stop_gradient(total_teacher_loss)

  return (
      student_loss,
      student_z_loss,
      teacher_loss,
      teacher_z_loss,
      total_student_loss / (1 + stop_grad_total_teacher_loss),
  )
