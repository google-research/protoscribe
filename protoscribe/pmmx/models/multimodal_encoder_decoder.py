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

"""PMMX Models.

This module builds a higher-level model structure and define methods
for the loss computation as well as a train, prediction, and
evaluation steps.
"""

import inspect
from typing import Any, Callable, Mapping, MutableMapping, Optional, Tuple, Union

from absl import logging  # pylint: disable=unused-import
from clu.metrics import Metric
from flax import linen as nn
from flax.core import scope as flax_scope
from flax.training import common_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from protoscribe.pmmx.utils import distillation_utils
from protoscribe.pmmx.utils import partitioning_utils
from protoscribe.pmmx.utils import perplexity_utils
import seqio
from t5x import decoding
from t5x import losses as t5x_losses
from t5x import metrics as metrics_lib
from t5x import models as t5x_models
from t5x import optimizers
from t5x import utils as t5x_utils
import tensorflow as tf


Array = Union[np.ndarray, jax.Array, tf.Tensor]
DType = jnp.dtype
MetricsMap = MutableMapping[str, Metric]
ConfigDict = ml_collections.ConfigDict
DecodeFnCallable = t5x_models.DecodeFnCallable
LossNormalizingFactor = Union[
    float, int, str, t5x_losses.SpecialLossNormalizingFactor]
PyTree = Any


class MultimodalEncoderDecoderModel(t5x_models.BaseTransformerModel):
  """Wrapper class for the models.Transformer nn.module."""

  FEATURE_CONVERTER_CLS: Callable[[bool, bool], seqio.FeatureConverter] = None
  _param_axes_names_override: partitioning_utils.ParamAxesNamesOverrideFn = None
  _annotation_mode: bool = False
  _distillation_mode: bool = False
  _distillation_alpha: float = 1.0
  _distillation_gt_fixing: bool = False
  _self_adaptive_distillation: bool = False

  def __init__(
      self,
      feature_converter_cls: Callable[[bool, bool], seqio.FeatureConverter],
      module: nn.Module,
      input_vocabulary: seqio.Vocabulary,
      output_vocabulary: seqio.Vocabulary,
      optimizer_def: optimizers.OptimizerDefType,
      decode_fn: DecodeFnCallable = decoding.beam_search,
      label_smoothing: float = 0.0,
      z_loss: float = 0.0,
      loss_normalizing_factor: Optional[
          Union[float, int, str, t5x_losses.SpecialLossNormalizingFactor]
      ] = None,
      param_axes_names_override: partitioning_utils.ParamAxesNamesOverrideFn = (
          partitioning_utils.legacy_vit_names_override
      ),
      annotation_mode: bool = False,
      distillation_mode: bool = False,
      distillation_alpha: float = 1.0,
      distillation_gt_fixing: bool = False,
      self_adaptive_distillation: bool = False,
  ):
    self.FEATURE_CONVERTER_CLS = feature_converter_cls  # pylint: disable=invalid-name
    self._param_axes_names_override = param_axes_names_override
    self._annotation_mode = annotation_mode
    self._distillation_mode = distillation_mode
    self._distillation_alpha = distillation_alpha
    self._distillation_gt_fixing = distillation_gt_fixing
    self._self_adaptive_distillation = self_adaptive_distillation
    super().__init__(
        module=module,
        input_vocabulary=input_vocabulary,
        output_vocabulary=output_vocabulary,
        optimizer_def=optimizer_def,
        decode_fn=decode_fn,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        loss_normalizing_factor=loss_normalizing_factor,
    )

  def get_initial_variables(
      self,
      rng: jnp.ndarray,
      input_shapes: Mapping[str, Array],
      input_types: Optional[Mapping[str, DType]] = None
  ) -> flax_scope.FrozenVariableDict:
    """Get the initial variables for an encoder-decoder model."""
    input_types = {} if input_types is None else input_types
    batch = {
        k: jnp.ones(shape, input_types.get(k, jnp.float32))
        for k, shape in input_shapes.items()
    }
    initial_variables = self.module.init(
        rng, batch=batch, decode=False, enable_dropout=False)
    param_axes_names_override = self._param_axes_names_override()
    return t5x_utils.override_params_axes_names(
        initial_variables,
        params_axes_names_override=param_axes_names_override)

  def _compute_logits(self,
                      params: PyTree,
                      batch: Mapping[str, jnp.ndarray],
                      dropout_rng: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Computes logits via a forward pass of `self.module`."""
    # Dropout is provided only for the training mode.
    rngs = {'dropout': dropout_rng} if dropout_rng is not None else None
    return self.module.apply({'params': params},
                             batch=batch,
                             decode=False,
                             enable_dropout=rngs is not None,
                             rngs=rngs)

  def _predict_batch_with_aux_embeddings_instead_of_tokens(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      rng: Optional[jax.Array] = None,
      decoder_params: Optional[MutableMapping[str, Any]] = None,
      return_all_decodes: bool = False,
      num_decodes: int = 1,
      prompt_with_targets: bool = False,
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Prediction fn that returns decoder embeddings for distillation."""
    decoded = self.module.apply(
        {'params': params},
        batch=batch,
        decode=False,
        enable_dropout=False,
        rngs=None,
    )
    decodes_dtype = jnp.int32
    if return_all_decodes:
      decodes_shape = decoded.shape[0:1] + [1] + decoded.shape[1:2]
    else:
      decodes_shape = decoded.shape[0:2]
    dummy_decodes = np.zeros(decodes_shape, dtype=decodes_dtype)
    return dummy_decodes, {
        'decoder_embeddings': decoded,
        'decoder_input_tokens': batch['decoder_input_tokens'],
        'decoder_target_tokens': batch['decoder_target_tokens'],
        'decoder_loss_weights': batch['decoder_loss_weights'],
    }

  def _compute_kv_cache(
      self,
      params,
      encoded_inputs: jnp.ndarray,
      encoder_mask: jnp.ndarray,
      encoder_segment_ids: Optional[jnp.ndarray],
      encoder_batch: Mapping[str, jnp.ndarray],
      decoder_batch: Mapping[str, jnp.ndarray],
      eos_id: int,
      prefill_decoder_prompt: bool = False,
  ) -> Tuple[PyTree, Optional[jnp.ndarray]]:
    """Initialize the key/value cache, with optional prompt.

    Args:
      params: The parameters of the model.
      encoded_inputs: Output of the encoder on the inputs.
      encoder_mask: Encoder self-attention mask.
      encoder_segment_ids: Encoder segment ids (if packing is used).
      encoder_batch: Batch of encoder-specific features.
      decoder_batch: Batch of decoder-specific features.
      eos_id: EOS token.
      prefill_decoder_prompt: Whether to prefill the cache using the decoder
        prompt.

    Returns:
      cache: The initialzed cache.
      initial_index: The index of the next position following prefill or None if
        `prefill_decoder_prompt` is False.
    """
    # Prepare zeroed-out autoregressive cache.
    decoder_input_tokens = decoder_batch['decoder_input_tokens']
    ones_batch = {
        k: jnp.ones(v.shape, self.module.dtype)
        for (k, v) in {**encoder_batch, **decoder_batch}.items()
    }
    _, initial_variables = self.module.apply(
        {'params': params},
        batch=ones_batch,
        decode=True,
        enable_dropout=False,
        mutable=['cache'],
    )

    cache = initial_variables['cache']

    if not prefill_decoder_prompt:
      return cache, None

    # Prefill the cache based on an (optional) prompt.
    # We assume the only 0 tokens are a BOS=0 token at the beginning of the
    # input and PAD=0 tokens at the end.
    decoder_prompt_inputs = decoder_input_tokens * (
        decoder_input_tokens != eos_id
    )
    inputs_lengths = jnp.sum(decoder_prompt_inputs != 0, axis=1)

    _, variables_with_cache = self.module.apply(
        {'params': params, 'cache': cache},
        encoded=encoded_inputs,
        encoder_mask=encoder_mask,
        encoder_segment_ids=encoder_segment_ids,
        decoder_batch=decoder_batch,
        enable_dropout=False,
        max_decode_length=decoder_input_tokens.shape[1],
        mutable=['cache'],
        prefill=True,
        prefill_lengths=inputs_lengths,
        method=self.module.decode,
    )

    cache = variables_with_cache['cache']
    if 'position_embedder' in cache['decoder']:
      # TODO: Instead have `module.decode` accept an index.
      cache['decoder']['position_embedder'][
          'position_embedder_index'
      ] = inputs_lengths

    return cache, inputs_lengths

  def predict_batch_with_aux(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      rng: Optional[jax.Array] = None,
      decoder_params: Optional[MutableMapping[str, Any]] = None,
      return_all_decodes: bool = False,
      num_decodes: int = 1,
      prompt_with_targets: bool = False
      ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Predict with fast decoding beam search on a batch.

    Here we refer to "parameters" for values that can be compiled into the
    model dynamically, as opposed to static configuration settings that require
    a recompile. For example, the model weights and the decoder brevity-penalty
    are parameters and can be modified without requiring a recompile. The number
    of layers, the batch size and the decoder beam size are configuration
    options that require recompilation if changed.

    If `prompt_with_targets = True`, then `decoder_prompt_inputs` is initialized
    from the batch's `decoder_input_tokens`. The EOS is stripped to avoid
    decoding to stop after the prompt by matching to `output_vocabulary.eos_id`.

    Args:
      params: model parameters.
      batch: a batch of inputs.
      rng: an optional RNG key to use during prediction (e.g., for decoding).
      decoder_params: additional (model-independent) parameters for the decoder.
      return_all_decodes: whether to return the entire beam or just the top-1.
      num_decodes: the number of beams to use in beam search.
      prompt_with_targets: Whether the force decode decoder_inputs.

    Returns:
      A tuple containing:
        the batch of predictions, with the entire beam if requested
        an auxiliary dictionary of decoder scores
    """
    if self._annotation_mode:
      return self._predict_batch_with_aux_embeddings_instead_of_tokens(
          params=params,
          batch=batch,
          rng=rng,
          decoder_params=decoder_params,
          return_all_decodes=return_all_decodes,
          num_decodes=num_decodes,
      )
    target_shape = batch['decoder_input_tokens'].shape

    # Split the batch into encoder/decoder features.
    encoder_batch = {
        k: v for (k, v) in batch.items() if not k.startswith('decoder')}
    decoder_batch = {
        k: v for (k, v) in batch.items() if k.startswith('decoder')}

    # Prepare transformer fast-decoder call.
    encoded_inputs, encoder_mask, encoder_segment_ids = self.module.apply(
        {'params': params},
        encoder_batch=encoder_batch,
        enable_dropout=False,
        method=self.module.encode)

    # TODO: Allow `prompt_with_targets` to be set dynamically
    # based on whether a prompt exists.
    if not prompt_with_targets:
      prefill_decoder_prompt = False
    elif 'initial_index' not in inspect.signature(self.decode_fn).parameters:
      logging.info(
          'Disabling prompt prefilling due to incompatible decode fn: %s.',
          self.decode_fn,
      )
      prefill_decoder_prompt = False
    elif 'prefill' not in inspect.signature(self.module.decode).parameters:
      logging.info(
          'Disabling prompt prefilling due to incompatible `module.decode`.'
      )
      prefill_decoder_prompt = False
    else:
      logging.info('Enabling prompt prefilling.')
      prefill_decoder_prompt = True
    cache, initial_index = self._compute_kv_cache(
        params,
        encoded_inputs,
        encoder_mask,
        encoder_segment_ids,
        encoder_batch,
        decoder_batch,
        eos_id=self.output_vocabulary.eos_id,
        prefill_decoder_prompt=prefill_decoder_prompt,
    )

    # We need to set up our decoder model to handle a batch size equal to
    # batch_size * num_decodes, where each batch item's data is expanded
    # in-place rather than tiled.
    # i.e. if we denote each batch element subtensor as el[n]:
    # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]
    tiled_decoder_batch = {
        k: decoding.flat_batch_beam_expand(f, num_decodes)
        for k, f in decoder_batch.items()
    }
    encoded_inputs = decoding.flat_batch_beam_expand(encoded_inputs,
                                                     num_decodes)
    encoder_mask = decoding.flat_batch_beam_expand(encoder_mask, num_decodes)
    if encoder_segment_ids is not None:
      encoder_segment_ids = decoding.flat_batch_beam_expand(
          encoder_segment_ids, num_decodes)

    def tokens_ids_to_logits(decoding_state: decoding.DecodingState):
      """Token slice to logits from decoder model."""
      flat_ids = decoding_state.cur_token
      flat_cache = decoding_state.cache
      decode_batch = dict(tiled_decoder_batch)
      decode_batch['decoder_target_tokens'] = flat_ids
      decode_batch['decoder_input_tokens'] = flat_ids
      # flat_ids: [batch * beam, seq_len=1]
      # cache is expanded inside beam_search to become flat_cache
      # flat_cache: [batch * beam, num_heads, depth_per_head, max_decode_len]
      # flat_logits: [batch * beam, seq_len=1, vocab]
      flat_logits, new_vars = self.module.apply(
          {'params': params, 'cache': flat_cache},
          encoded=encoded_inputs,
          encoder_mask=encoder_mask,
          encoder_segment_ids=encoder_segment_ids,
          decoder_batch=decode_batch,
          enable_dropout=False,
          decode=True,
          max_decode_length=target_shape[1],
          mutable=['cache'],
          method=self.module.decode,
      )  # pytype: disable=attribute-error
      if self._distillation_mode:
        (flat_logits, _) = flat_logits  # discard teacher logits
      # Remove sequence length dimension since it's always 1 during decoding.
      flat_logits = jnp.squeeze(flat_logits, axis=1)
      new_flat_cache = new_vars['cache']
      return flat_logits, new_flat_cache

    if decoder_params is None:
      decoder_params = {}
    if rng is not None:
      if decoder_params.get('decode_rng') is not None:
        raise ValueError(
            f'Got RNG both from the `rng` argument ({rng}) and '
            f"`decoder_params['decode_rng']` ({decoder_params['decode_rng']}). "
            'Please specify one or the other.')
      decoder_params['decode_rng'] = rng

    if initial_index is not None:
      # We only set initial_index when it's non-None since it is not supported
      # by all decoders.
      decoder_params['initial_index'] = initial_index

    # `decoder_prompt_inputs` is initialized from the batch's
    # `decoder_input_tokens`. The EOS is stripped to avoid decoding to stop
    # after the prompt by matching to `output_vocabulary.eos_id`.
    # These inputs are ignored by the beam search decode fn.
    if prompt_with_targets:
      decoder_prompt_inputs = batch['decoder_input_tokens']
      decoder_prompt_inputs = decoder_prompt_inputs * (
          decoder_prompt_inputs != self.output_vocabulary.eos_id)
    else:  # for batch_size and seq_len
      decoder_prompt_inputs = jnp.zeros_like(batch['decoder_input_tokens'])

    scanned = hasattr(self.module, 'scan_layers') and self.module.scan_layers
    decodes, scores = self._decode_fn(
        inputs=decoder_prompt_inputs,
        cache=cache,
        tokens_to_logits=tokens_ids_to_logits,
        num_decodes=num_decodes,
        eos_id=self.output_vocabulary.eos_id,
        cache_offset=1 if scanned else 0,
        **decoder_params)

    # Beam search returns [n_batch, n_beam, n_length] with beam dimension sorted
    # in increasing order of log-probability.
    # Return the highest scoring beam sequence.
    if return_all_decodes:
      return decodes, {'scores': scores}
    else:
      return decodes[:, -1, :], {'scores': scores[:, -1]}

  def score_batch(self,
                  params: PyTree,
                  batch: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    """Compute log likelihood score on a batch."""
    weights = batch['decoder_loss_weights']
    logits = self._compute_logits(params, batch)
    if self._distillation_mode:
      (logits, _) = logits  # discard teacher logits for scoring
    # Purposefully don't use config.z_loss because that term is for training
    # stability and shouldn't affect our reported scores.
    token_scores = -t5x_losses.cross_entropy_with_logits(
        logits,
        common_utils.onehot(
            batch['decoder_target_tokens'], logits.shape[-1], on_value=1,
            off_value=0),
        z_loss=0.0)[0] * weights
    sequence_scores = token_scores.sum(-1)
    return sequence_scores

  def _compute_metrics(self,
                       logits: jnp.ndarray,
                       targets: jnp.ndarray,
                       mask: jnp.ndarray,
                       loss: jnp.ndarray,
                       z_loss: Optional[Union[jnp.ndarray, float]] = None,
                       segment_ids: Optional[Mapping[str, jnp.ndarray]] = None,
                       ) -> MetricsMap:
    """Compute metrics given the logits, targets and loss."""
    metrics_map = super()._compute_metrics(
        logits=logits,
        targets=targets,
        mask=mask,
        loss=loss,
        z_loss=z_loss,
        segment_ids=segment_ids)
    perplexity = perplexity_utils.softmax_perplexity(logits, targets, mask)
    metrics_map['perplexity'] = metrics_lib.Sum.from_model_output(perplexity)
    return metrics_map

  def loss_fn(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.Array],
  ) -> Tuple[jnp.ndarray, MetricsMap]:
    """Loss function used for training with a cross-entropy loss."""
    if not self._distillation_mode:
      return super().loss_fn(params, batch, dropout_rng)

    loss_normalizing_factor: Optional[LossNormalizingFactor]
    (loss_normalizing_factor,
     weights) = t5x_losses.get_loss_normalizing_factor_and_weights(
         self._loss_normalizing_factor, batch)

    (student_logits, teacher_logits) = self._compute_logits(
        params, batch, dropout_rng)

    logging.info('batch.keys(): %s', batch.keys())

    # segment ids to compute packing, padding etc.
    segment_ids = {
        k[:-len('_segment_ids')]: v
        for k, v in batch.items()
        if k.endswith('_segment_ids')
    }
    # If these don't exist then we can create only padding mask.
    if not segment_ids:
      segment_ids = {
          k: v != 0
          for k, v in batch.items()
          if k in ('encoder_input_tokens', 'decoder_target_tokens')
      }

    if self._distillation_gt_fixing:
      teacher_logits = distillation_utils.fix_teacher_logits(
          teacher_logits, batch['decoder_target_tokens'])

    if self._self_adaptive_distillation:
      one_hot_loss, z_loss, teacher_loss, _, adaptive_weight = (
          distillation_utils.losses_and_adaptive_weight(
              student_logits,
              teacher_logits,
              batch['decoder_target_tokens'],
              weights,
              label_smoothing=self._label_smoothing,
              z_loss=self._z_loss,
              loss_normalizing_factor=loss_normalizing_factor,
          )
      )
      distillation_loss = distillation_utils.distillation_loss(
          student_logits=student_logits,
          teacher_logits=teacher_logits,
          weights=weights * adaptive_weight,
          loss_normalizing_factor=loss_normalizing_factor,
      )
    else:
      one_hot_loss, z_loss, _ = t5x_losses.compute_weighted_cross_entropy(
          student_logits,
          targets=batch['decoder_target_tokens'],
          weights=weights,
          label_smoothing=self._label_smoothing,
          z_loss=self._z_loss,
          loss_normalizing_factor=loss_normalizing_factor)
      teacher_loss, _, _ = t5x_losses.compute_weighted_cross_entropy(
          teacher_logits,
          targets=batch['decoder_target_tokens'],
          weights=weights,
          label_smoothing=self._label_smoothing,
          z_loss=self._z_loss,
          loss_normalizing_factor=loss_normalizing_factor,
      )
      distillation_loss = distillation_utils.distillation_loss(
          student_logits=student_logits,
          teacher_logits=teacher_logits,
          weights=weights,
          loss_normalizing_factor=loss_normalizing_factor,
      )

    if self._distillation_alpha < 1.0:
      if self._distillation_alpha < 0.0:
        raise ValueError('distillation_alpha must be in the interval [0,1]')
      loss = (self._distillation_alpha * distillation_loss +
              (1 - self._distillation_alpha) * one_hot_loss)
    else:
      loss = distillation_loss
      z_loss = 0.0

    student_metrics = self._compute_metrics(
        logits=student_logits,
        targets=batch['decoder_target_tokens'],
        mask=weights,
        loss=loss,
        z_loss=z_loss,
        segment_ids=segment_ids)

    teacher_metrics = self._compute_metrics(
        logits=teacher_logits,
        targets=batch['decoder_target_tokens'],
        mask=weights,
        loss=teacher_loss,  # loss vs gt one-hot distribution
        z_loss=z_loss,
        segment_ids=segment_ids)

    metrics: t5x_models.MetricsMap = {}
    metrics.update({f'student_{k}': v for (k, v) in student_metrics.items()})
    metrics.update({f'teacher_{k}': v for (k, v) in teacher_metrics.items()})
    return loss, metrics
