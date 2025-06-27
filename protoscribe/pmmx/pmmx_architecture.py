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

"""PMMX Layers and Flax modules."""

import abc
import dataclasses
import inspect
import math
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
from flax import linen as nn
import gin
import jax.lax
import jax.numpy as jnp
from protoscribe.pmmx import multimodal_feature
from protoscribe.pmmx import relative_position_biases_nd

from flaxformer import activation_partitioning
from flaxformer import transformer_common as common
from flaxformer.architectures.common import param_remapping
from flaxformer.architectures.t5 import t5_architecture
from flaxformer.components import dense
from flaxformer.components import embedding
from flaxformer.components import relative_position_biases
from flaxformer.components import rich_attention_position_scores
from flaxformer.components import transforms
from flaxformer.components.attention import dense_attention

# Type Stubs
Dtype = jnp.dtype
Array = jnp.ndarray

# pylint: disable=not-callable
# pytype: disable=not-callable


class DenseEmbed(dense.DenseGeneral):
  """Dense Embedding Module."""
  pass


class IdentityEmbed(nn.Module):
  """Identity Embedding Module."""

  @nn.compact
  def __call__(self, inputs: Array, *args: Any, **kwargs: Any) -> Array:
    return inputs


class SinusoidalEmbed(nn.Module):
  """Sinusoidal non-trainable position embed.

  This layer calculates the position encoding as a mix of sine and cosine
  functions with geometrically increasing wavelengths. Defined and formulized in
   "Attention is All You Need", section 3.5.
  (https://arxiv.org/abs/1706.03762).

  Original TF implementation:
  https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/position_embedding.py # pylint: disable=line-too-long

  Attributes:
    features: num. of embedding dims
    dtype: jnp.dtype to use
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position.
  """

  features: int
  dtype: Dtype = jnp.float32
  min_timescale: float = 1.0
  max_timescale: float = 1e4

  @nn.compact
  def __call__(self, position_ids: Array):
    """Returns fixed sinusoidal position embeddings."""
    position_ids = position_ids.astype(self.dtype)
    num_timescales = float(self.features // 2)
    log_timescale_increment = (
        math.log(float(self.max_timescale) / float(self.min_timescale)) /
        (num_timescales - 1))
    # [E // 2]
    inv_timescales = self.min_timescale * jnp.exp(
        jnp.arange(num_timescales, dtype=self.dtype) *
        -log_timescale_increment)
    # [L, E // 2]
    scaled_time = jnp.expand_dims(position_ids, -1) * jnp.expand_dims(
        inv_timescales, -2)
    # [L, E]
    position_embeddings = jnp.concatenate(
        [jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=2)
    return position_embeddings


@gin.register
class MeanPoolingEmbed(embedding.Embed):
  """Mean pooling Embedding Module."""

  def __call__(self, inputs, *args: Any, **kwargs: Any):
    """Mean-pooling of token embeddings.

    The reason for mean pooling is twofold:
      a) to reduce the sequence length, as 10 tokens per image would consume a
         lot of sequence, and
      b) with absolute position embeddings, the model can easily determine
         the correspondence between different features of the same image

    Args:
      inputs: input data, all dimensions are considered batch dimensions.
      *args: other args
      **kwargs: other kwargs

    Returns:
      Output which is embedded input data.  The output shape follows the input,
        with the final dimension replaced by a `features` dimension.
    """
    per_token_embeddings = super().__call__(inputs)
    return jnp.mean(per_token_embeddings, axis=-2)


class AttentionBiasApi(metaclass=abc.ABCMeta):
  """Interface for relative attention APIs that need the entire input vector."""

  @abc.abstractmethod
  def __call__(
      self,
      q_inputs: Array,
      k_inputs: Array,
      raw_input: Mapping[str, Array],
      sequence_metadata: Optional[multimodal_feature.SequenceMetadata] = None,
      bidirectional: bool = True,
      is_cross_attention: bool = False,
      decode: bool = False,
  ) -> Array:
    raise NotImplementedError()


class MultimodalEncoderLayer(nn.Module, param_remapping.ParameterRemappable):
  """Transformer encoder layer.

  Attributes:
    attention: The attention module.
    mlp: The MLP module, applied after attention.
    dropout_factory:  A callable that returns a new dropout instance. This is
      applied after the attention module.
    layer_norm_factory:  A callable that returns a new layer norm. This is
      applied before the attention module and before the MLP.
    relative_position_bias_factory:  A callable that returns relative position
      bias instances. This should only be used for per-layer relative position
      biases; please use `shared_relative_position_bias` if they are shared
      among layers.
    shared_relative_position_bias: Shared relative position bias module, usually
      owned by the Encoder.
    activation_partitioning_dims: When set to 2, partitions intermediate
      variables containing the input and output of the encoder layer.
    parallel: whether to call attention and mlp in parallel
    sow_intermediates: whether to track intermediates using Module.sow.
    scanned: whether this layer is being scanned over.
  """
  attention: nn.Module
  mlp: nn.Module
  dropout_factory: Callable[[], nn.Module]
  layer_norm_factory: Callable[[], nn.Module]
  relative_position_bias_factory: Optional[Callable[[], nn.Module]] = None
  shared_relative_position_bias: Optional[nn.Module] = None
  multimodal_relative_position_bias_factory: Optional[
      Callable[[], nn.Module]] = None
  shared_multimodal_relative_position_bias: Optional[nn.Module] = None
  activation_partitioning_dims: int = 1
  parallel: bool = False
  sow_intermediates: bool = False
  scanned: bool = False

  def setup(self):
    if (self.relative_position_bias_factory is not None and
        self.shared_relative_position_bias is not None):
      raise ValueError(
          'Please set at most one of relative_position_bias_factory and '
          'shared_relative_position_bias. '
          '(They can both be None however, e.g. for absolute position embeds.)')
    self.relpos_bias = (
        self.relative_position_bias_factory()
        if self.relative_position_bias_factory is not None else
        self.shared_relative_position_bias)
    self.multimodal_relpos_bias = (
        self.multimodal_relative_position_bias_factory()
        if self.multimodal_relative_position_bias_factory is not None else
        self.shared_multimodal_relative_position_bias)

    if self.parallel:
      self.layer_norm = self.layer_norm_factory()
      self.dropout = self.dropout_factory()
    else:
      self.pre_attention_layer_norm = self.layer_norm_factory()
      self.pre_mlp_layer_norm = self.layer_norm_factory()
      self.post_attention_dropout = self.dropout_factory()
      self.post_mlp_dropout = self.dropout_factory()

  def get_bias(self, layer_input: Array,
               sequence_metadata: multimodal_feature.SequenceMetadata,
               batch: Mapping[str, Array]) -> Optional[Array]:
    encoder_bias = None
    if self.relpos_bias:
      # TODO: Migrate other bias objects into shared API.
      if isinstance(self.relpos_bias, AttentionBiasApi):
        encoder_bias = self.relpos_bias(
            q_inputs=layer_input,
            k_inputs=layer_input,
            raw_input=batch,
            sequence_metadata=sequence_metadata,
            bidirectional=True,
        )
      elif isinstance(self.relpos_bias,
                      relative_position_biases.RelativePositionBiases):
        encoder_bias = self.relpos_bias(
            layer_input.shape[-2], layer_input.shape[-2], bidirectional=True)
      elif isinstance(self.relpos_bias,
                      relative_position_biases_nd.RelativePositionBiasesND):
        encoder_bias = self.relpos_bias(
            layer_input.shape[-2], layer_input.shape[-2],
            sequence_metadata=sequence_metadata, bidirectional=True)
      elif isinstance(self.relpos_bias,
                      rich_attention_position_scores.RichAttentionApi):
        encoder_bias = self.relpos_bias(
            layer_input, layer_input, bidirectional=True)
      else:
        raise TypeError(
            f'{type(self.relpos_bias)} is not a supported relative position '
            f'bias factory.\nInstance value: {self.relpos_bias}')

    if encoder_bias is not None and self.multimodal_relpos_bias is not None:
      encoder_bias = self.multimodal_relpos_bias(
          encoder_bias, sequence_metadata.modality_segment_ids)

    return encoder_bias

  def __call__(self,
               inputs: Array,
               sequence_metadata: multimodal_feature.SequenceMetadata,
               batch: Mapping[str, Array],
               encoder_mask: Optional[Array] = None,
               *,
               logit_mask: Optional[Array] = None,
               enable_dropout: bool = True):
    """Applies a single T5 encoder layer.

    Args:
      inputs: input data [batch, length, emb_dim].
      sequence_metadata: a SequenceMetadata containing info about the input
      batch: Batch of input examples before multimodal embeddings.
      encoder_mask: encoder self-attention mask.
      logit_mask: encoder logits mask.
      enable_dropout: Enables dropout if set to True.

    Returns:
      output after transformer encoder block.
    """
    layer_input = inputs
    del inputs

    assert layer_input.ndim == 3
    layer_input = activation_partitioning.with_sharding_migration(
        layer_input,
        self.activation_partitioning_dims,
        logical_axis_names=('batch', 'length', 'embed'))
    if self.parallel:
      x = self.layer_norm(layer_input)
      x = activation_partitioning.with_sharding_migration(
          x,
          self.activation_partitioning_dims,
          logical_axis_names=('batch', 'length', 'embed'))

      encoder_bias = self.get_bias(
          layer_input=x, sequence_metadata=sequence_metadata, batch=batch)

      y = (
          self.attention(
              x,
              x,
              encoder_mask,
              encoder_bias,
              enable_dropout=enable_dropout) +
          self.mlp(x, enable_dropout=enable_dropout))
      y *= 2**-0.5
      y = layer_input + self.dropout(y, deterministic=not enable_dropout)

    else:
      # Attention block.
      x = self.pre_attention_layer_norm(layer_input)
      x = activation_partitioning.with_sharding_migration(
          x,
          self.activation_partitioning_dims,
          logical_axis_names=('batch', 'length', 'embed'))

      if logit_mask is not None:
        x = logit_mask * x

      encoder_bias = self.get_bias(
          layer_input=x, sequence_metadata=sequence_metadata, batch=batch)

      # The shape should be maintained for the residual connection.
      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      x = self.attention(
          x, x, encoder_mask, encoder_bias, enable_dropout=enable_dropout)
      x = layer_input + self.post_attention_dropout(
          x, deterministic=not enable_dropout)
      x = activation_partitioning.with_sharding_migration(
          x,
          self.activation_partitioning_dims,
          logical_axis_names=('batch', 'length', 'embed'))

      # MLP block.
      y = self.pre_mlp_layer_norm(x)
      y = activation_partitioning.with_sharding_migration(
          y,
          self.activation_partitioning_dims,
          logical_axis_names=('batch', 'length', 'embed'))

      if logit_mask is not None:
        y = logit_mask * y

      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      y = self.mlp(y, enable_dropout=enable_dropout)
      y = x + self.post_mlp_dropout(y, deterministic=not enable_dropout)

    y = activation_partitioning.with_sharding_migration(
        y,
        self.activation_partitioning_dims,
        logical_axis_names=('batch', 'length', 'embed'))

    if self.sow_intermediates:
      self.sow('intermediates', 'activations', y)

    # scan expects functions to have a signature: fn(carry, in) --> carry, out
    # TODO: automate this detail.
    if self.scanned:
      return y, None
    else:
      return y


@dataclasses.dataclass
class EmbedCombineResult:
  """Class for storing embeddings and feature position indices."""
  embedded: Array
  feature_positions: Mapping[str, Tuple[int, int]]

  def to_dict(self, x):
    if x is not None:
      return {name: x[:, i:j]
              for name, (i, j) in self.feature_positions.items()}
    else:
      return None


class MultimodalEncoder(nn.Module, param_remapping.ParameterRemappable):
  """A stack of encoder layers w/ multimodal inputs.

  This is a fork of the Flaxformer T5 Encoder, which could be moved into
  that library at some point.

  Attributes:
    layer_factory: A callable that returns an EncoderLayer.
    input_dropout_factory: A callable that returns the dropout to apply to the
      input.
    output_dropout_factory: A callable that returns the dropout to apply to the
      output. Perhaps for legacy rather than essential reasons, the broadcasting
      pattern is sometimes different from input_dropout_factory().
    layer_norm_factory: A callable that returns a layer norm.
    num_layers: Number of layers to generate.
    dtype: Dtype to cast the embedded inputs.
    relative_position_bias_factory: A callable that returns a relative position
      bias instance which will be shared for all encoder layers. Only set this
      if using shared relative position biases.
    shared_token_embedder: A callable that returns a token embedder shared
      between both encoder and decoder.
    feature_dropout_spec: A dict of {feature_name: dropout_factory}. Used to
      override the input_dropout_factory config on the feature level.
    passthrough_features: A sequence of feature keys to ignore when creating a
      multimodal feature converter.
    sow_intermediates: whether to track intermediates using Module.sow.
  """
  layer_factory: t5_architecture.MakeEncoderLayerFn  # pytype: disable=module-attr
  input_dropout_factory: Callable[[], nn.Module]
  output_dropout_factory: Callable[[], nn.Module]
  layer_norm_factory: Callable[[], nn.Module]
  num_layers: int
  dtype: Dtype = jnp.float32
  layer_remat: str = 'legacy'
  scan_layers: bool = False
  spmd_annotations: Any = None
  sow_intermediates: bool = False
  shared_relative_position_bias_factory: Optional[Callable[[],
                                                           nn.Module]] = None
  shared_multimodal_relative_position_bias_factory: Optional[Callable[
      [], nn.Module]] = None

  # Text token embedder shared by encoder and decoder.
  # See MultimodalEncoderDecoder.
  shared_token_embedder: Optional[embedding.Embed] = None

  # List of feature names, in implicit sequence order, and their modality names.
  feature_spec: Optional[Sequence[Tuple[str, str]]] = None
  # List of feature names and their dropout factories.
  feature_dropout_spec: Optional[Mapping[str, Callable[[], nn.Module]]] = None

  # Mapping from primary feature name to a sequence of sub features.
  sub_features_spec: Optional[Mapping[str, Sequence[str]]] = None

  # List of modalities, in implicit id order (first id=0, second id=1, ...)
  modality_spec: Optional[Sequence[str]] = None
  # Mapping from modality to a callable or
  #                          a sequence of (feature_name, embedding_factory).
  modality_embedders_spec: Optional[Union[
      Mapping[str, Callable[[], nn.Module]],
      Mapping[str, Sequence[Tuple[str, Callable[[], nn.Module]]]]]] = None

  # Name of features to ignore in multimodal_features from a batch.
  passthrough_features: Sequence[str] = tuple()

  # When True, __call__ returns a dict from feature-name to feature-outputs,
  # where the feature-outputs were sliced from their corresponding positions.
  # Otherwise, the returned representation is seq_len long feature.
  outputs_as_dict: bool = False
  encoder_mask_fn: Callable[[Array, dtype],  # type: ignore  # jnp-type
                            Tuple[Array,
                                  Array]] = multimodal_feature.make_encoder_mask

  sequence_metadata_fn: Callable[  # pytype: disable=invalid-annotation  # jnp-type
      [Sequence[multimodal_feature.MultimodalFeature], dtype],
      multimodal_feature.SequenceMetadata,
  ] = multimodal_feature.make_sequence_metadata

  def setup(self):
    self.setup_without_encoder()

    layer_kwargs = dict(
        shared_relative_position_bias=self.relpos_bias,
        shared_multimodal_relative_position_bias=self.multimodal_relpos_bias)
    self.encoder = self.make_encoder(self.num_layers, layer_kwargs)

  @nn.nowrap
  def setup_without_encoder(self):
    """Set up for all the variables and submodules except for the encoder.

    We split up the setup into two functions: setup_without_encoder and
    make_encoder. This allows the subclasses of MultimodalEncoder to setup
    their own encoder while reusing all the non-encoder setup code.
    """
    def _add_embedder(embedders, embedder_key, embedder):
      if embedder_key in embedders:
        raise ValueError(
            f'Multiple embedders for key={embedder_key}')
      embedders[embedder_key] = embedder
      logging.info('adding embedder for %s', embedder_key)

    if self.feature_spec is None:
      raise ValueError('feature_spec must be set')
    if self.modality_spec is None:
      raise ValueError('modality_spec must be set')
    if self.modality_embedders_spec is None:
      self.modality_embedders_spec = {}

    # TODO: Find out why Optional does not get unwrapped by pytype.
    self.feature_spec: Sequence[Tuple[str, str]]
    self.modality_spec: Sequence[str]

    if self.use_master_multi_embedder:
      # This corresponds to the case where modality_embedders_spec maps
      # the modality to embedding_factory, which is the common embedder to
      # be shared by all features with this modality; in this case, feature
      # overlay information is solely expressed by subfeature_spec.  We create
      # one master MultiEmbed with embedders keyed on the modality name.
      #
      # Example:
      #   See unit test 'test_entire_transformer_shared_embeds_sub_features'
      #   //protoscribe/pmmx/configs/models/p1_t5_1_1_testing_shared_embeds_sub_features.gin # pylint: disable=line-too-long
      master_embedders = {}
      if self.shared_token_embedder is not None:
        master_embedders['text_tokens'] = self.shared_token_embedder

      for modality, spec in self.modality_embedders_spec.items():  # pylint: disable=not-an-iterable
        _add_embedder(master_embedders, modality,
                      spec(name=f'{modality}_embedder'))
      self.multi_embedder = embedding.MultiEmbed(master_embedders)
      logging.info('Master MultiEmbed created: %s', self.multi_embedder)

    else:
      # This corresponds to the case where modality_embedders_spec maps
      # modality to a sequence of (feature_name, embedding_factory).  You
      # can't have embedder sharing among features (except for the
      # 'text_tokens' modality, which gets special treatment) in this
      # case; but since embedders are specified at the feature level, it is
      # possible to allow different sub-features in a given modality to have
      # different embedders (this is not what ``modality'' is meant to be
      # used for, but we support this for backward compatibility).
      # Note that:
      # - embedders for all subfeatures associated with a main_feature of the
      #   modality need to be specified in the sequence.
      # - we currently only support specifying embedders for position_ids
      #   in this way.
      # To ensure backward compatibility for checkpoints, in this case we create
      # a MultiEmbed for each modality, with embedders keyed on feature name.
      #
      # Example:
      #   See extra1 and extra2 in
      #   unit test test_entire_transformer_shared_embeds_extra_features.
      #   //protoscribe/pmmx/configs/models/p1_t5_1_1_testing_extra_features.gin # pylint: disable=line-too-long
      for modality, spec in self.modality_embedders_spec.items():  # pylint: disable=not-an-iterable
        embedders = {}
        for feature_name, embedder_factory in spec:
          _add_embedder(embedders, feature_name, embedder_factory())
        if modality != 'text_tokens':
          embedder = embedding.MultiEmbed(embedders)
          setattr(self, f'{modality}_embedder', embedder)
        else:
          if self.shared_token_embedder is not None:
            embedders['text_tokens'] = self.shared_token_embedder
          self.embedder = embedding.MultiEmbed(embedders)
      if not hasattr(self, 'embedder'):
        self.embedder = embedding.MultiEmbed(
            {'text_tokens': self.shared_token_embedder})

    # Init dropout of all features to the default input_dropout config
    # TODO: Get rid of this hack (pytype complains).
    self.feature_spec: Sequence[Tuple[str, str]]
    feature_dropout_layers = {
        feature_name: self.input_dropout_factory()
        for feature_name, _ in self.feature_spec}

    # Override with feature specific params.
    if self.feature_dropout_spec:
      # TODO: Get rid of this hack  (pytype complains).
      self.feature_dropout_spec: Mapping[str, Callable[[], nn.Module]]
      for feature_name, dropout_factory in self.feature_dropout_spec.items():
        feature_dropout_layers[feature_name] = dropout_factory()

    self.feature_dropout_layers = feature_dropout_layers

    self.relpos_bias = (
        self.shared_relative_position_bias_factory()  # pylint: disable=not-callable
        if self.shared_relative_position_bias_factory is not None else None)
    self.multimodal_relpos_bias = (
        self.shared_multimodal_relative_position_bias_factory()  # pylint: disable=not-callable
        if self.shared_multimodal_relative_position_bias_factory is not None
        else None)
    self.encoder_norm = self.layer_norm_factory()
    self.output_dropout = self.output_dropout_factory()

  @nn.nowrap
  def make_encoder(self, num_layers, layer_kwargs):
    lyrf = lambda: self.layer_factory(**layer_kwargs)  # pylint: disable=unnecessary-lambda  # pytype: disable=wrong-keyword-args
    lyrf = t5_architecture.maybe_remat(
        lyrf, self.layer_remat, self.scan_layers, static_argnums=(1, 5))
    if not self.scan_layers:
      self.layers = [lyrf() for _ in range(num_layers)]
      encoder = common.TransparentLayerSequence(self.layers)
    else:
      initializing = self.is_mutable_collection('params')
      # We scan the parameters along axis 1 as an XLA layout optimization.
      SCAN_AXIS = 1  # pylint: disable=invalid-name
      params_spec = SCAN_AXIS if initializing else transforms.ScanIn(SCAN_AXIS)
      cache_spec = 0
      scan_annotation = (
          self.spmd_annotations['encoder']
          if self.spmd_annotations is not None else None)
      lyrf = transforms.factory_scan(
          lyrf,
          in_axes=(nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast,
                   nn.broadcast),
          variable_axes={
              'params': params_spec,
              'cache': cache_spec
          },
          split_rngs={
              'params': True,
              'dropout': True
          },
          length=num_layers,
          data_transform=transforms.inner_scan_spmd(scan_annotation, SCAN_AXIS),
          axes_collections=('params', 'cache')
      )
      encoder = lyrf()

    return encoder

  @property
  def use_master_multi_embedder(self):

    # If self.modality_embedders_spec is empty or if it consistently map
    # modality to embedding_factory, we use a master multi_embedder.
    if all(callable(spec) for spec in self.modality_embedders_spec.values()):
      return True
    # Otherwise, self.modality_embedders_spec should consistently map
    # modality to a sequence of (feature_name, embedding_factory).
    elif all(
        not callable(spec) for spec in self.modality_embedders_spec.values()):
      return False
    else:
      raise ValueError('modality_embedders_spec must either consistently map '
                       'modality to embedding_factory or consistently map '
                       'modality to a seq of (feature_name, embedding_factory)')

  def embed_and_combine_inputs(
      self, encoder_features: Sequence[multimodal_feature.MultimodalFeature],
      enable_dropout: bool = True
  ) -> Union[Array, EmbedCombineResult]:
    """Returns the embedded inputs for further encoding.

    Features may have different lengths and are embedded and returned
    in order.

    Args:
      encoder_features: linearized encoder features
      enable_dropout: whether dropout is enabled

    Returns:
      the embedded features
    """
    def _get_embedder(feature):
      if self.use_master_multi_embedder:
        return self.multi_embedder
      embedder_name = feature.embedder_name
      if not hasattr(self, embedder_name):
        raise ValueError(
            f'No embedder named {embedder_name}. You must define this if you '
            f'wish to embed features with modality={feature.modality_name}.')
      return getattr(self, embedder_name)

    def _check_embedder_key(embedder, embedder_key):
      if embedder_key in embedder.embedders:
        return embedder_key
      else:
        raise ValueError(f'No embedder found for {embedder_key}.')

    def _get_embedder_key(embedder, feature_name, modality_name):
      if self.use_master_multi_embedder:
        return _check_embedder_key(embedder, modality_name)
      # In the context of the old syntax of specifying embedders by
      # modality -> sequence(feature, embedding_factory), either the embedder
      # is keyed by feature_name in the MultiEmbed for that modality; or if
      # the feature has 'text_tokens' modality, it's the default token embedder
      # which is keyed by 'text_tokens' rather than feature name.
      elif feature_name in embedder.embedders:
        return feature_name
      elif modality_name == 'text_tokens':
        return _check_embedder_key(embedder, modality_name)
      else:
        raise ValueError(
            f'No embedder found for {feature_name} or {modality_name}')

    embedded = []
    modality_map = dict(self.feature_spec)
    for feature in encoder_features:
      logging.info('Embedding feature=%s', feature)
      embedder = _get_embedder(feature)

      embedder_kwargs = dict(deterministic=not enable_dropout)
      main_feature_embedder_key = _get_embedder_key(embedder,
                                                    feature.name,
                                                    feature.modality_name)
      embedder_kwargs[main_feature_embedder_key] = feature.values

      if (
          not self.use_master_multi_embedder and
          'position_ids' in embedder.embedders
      ):
        position_embedder_key = 'position_ids'
        if feature.positions is None:
          embedder_kwargs[position_embedder_key] = jnp.tile(
              jnp.arange(feature.values.shape[1])[None, :],
              [feature.values.shape[0], 1])
        else:
          embedder_kwargs[position_embedder_key] = feature.positions

      # Add or skip subfeatures.
      for subfeature in feature.subfeatures:
        subfeature_modality = modality_map[subfeature.name]
        subfeature_embedder_key = _get_embedder_key(embedder,
                                                    subfeature.name,
                                                    subfeature_modality)
        embedder_kwargs[subfeature_embedder_key] = subfeature.values

      logging.info('Feature=%s embedder_kwargs=%s', feature, embedder_kwargs)
      embeddings = embedder(**embedder_kwargs)  # pytype: disable=wrong-arg-types

      # Apply feature specific dropout or fallback to the default input dropout.
      dropout_layer = self.feature_dropout_layers[feature.name]
      embeddings = dropout_layer(embeddings, deterministic=not enable_dropout)
      embedded.append(embeddings)

    logging.info('Pre-concatenation embeddings: %s', embedded)
    embedded = jnp.concatenate(embedded, axis=1)

    if self.outputs_as_dict:
      logging.info('Post-concatenation: Computing feature indices.')
      # Extract the starting positions of each feature.
      # position ranges are start inclusive, end exclusive. i.e., [start, end)
      feature_positions = {}
      start_pos = 0
      for feature in encoder_features:
        end_pos = start_pos + feature.values.shape[1]
        feature_positions[feature.name] = (start_pos, end_pos)
        start_pos = end_pos

      result = EmbedCombineResult(embedded, feature_positions)
    else:
      result = embedded

    return result

  def encode_from_continuous_inputs(self,
                                    inputs,
                                    sequence_metadata,
                                    batch,
                                    encoder_mask=None,
                                    enable_dropout: bool = True):
    """Applies all the layers starting from the continuous (embedded) inputs."""
    # Apply all encoder layers. Because of residual connection, the width of the
    # network is kept at `cfg.emb_dim` throughout.
    encoder_outputs = self.encoder(
        inputs,
        sequence_metadata,
        batch,
        encoder_mask=encoder_mask,
        logit_mask=None,
        enable_dropout=enable_dropout)

    if self.scan_layers:
      encoder_outputs = encoder_outputs[0]

    # Post-process the outputs of the final encoder layer.
    # TODO: We could do this in the common encoder.
    encoder_outputs = self.encoder_norm(encoder_outputs)
    encoder_outputs = self.output_dropout(
        encoder_outputs, deterministic=not enable_dropout)

    return encoder_outputs

  def prepare_encoder_features(self, batch: Mapping[str, Array]):
    batch_encoder = {
        k: v
        for (k, v) in batch.items()
        if not k.startswith('targets') and not k.endswith('_loss_weights')
    }

    encoder_features = multimodal_feature.linearize_encoder_features(
        batch_encoder, self.feature_spec, self.modality_spec,
        self.sub_features_spec, self.passthrough_features)
    logging.info('Linearized encoder_features: %s', encoder_features)
    return encoder_features

  # TODO: This method might be doing too much for inspection APIs.
  def __call__(self,
               batch: Mapping[str, Array],
               enable_dropout: bool = True):
    """Applies Transformer model on the inputs.

    The features are embedded separately and concatenated to form the input
    sequence to pass to the Transformer.

    Zero-valued inputs are considered padding when populating the
    self-attention mask.

    Args:
      batch: feature name to values
      enable_dropout: whether dropout is disabled

    Returns:
      triple of (encoded values, encoder mask, encoder segment ids)
    """
    encoder_features = self.prepare_encoder_features(batch)

    embed_combine_result = self.embed_and_combine_inputs(
        encoder_features, enable_dropout=enable_dropout)
    if isinstance(embed_combine_result, EmbedCombineResult):
      embedded_input = embed_combine_result.embedded
    else:
      embedded_input = embed_combine_result

    # TODO: Revert this cast or move to embedder.
    embedded_input = embedded_input.astype(self.dtype)

    encoder_mask, encoder_segment_ids = self.encoder_mask_fn(
        encoder_features, self.dtype)

    sequence_metadata = self.sequence_metadata_fn(
        encoder_features, self.dtype)

    logging.info('SequenceMetadata.feature_name_to_bounds_map=%s',
                 sequence_metadata.feature_name_to_bounds_map)

    encoder_outputs = self.encode_from_continuous_inputs(
        embedded_input,
        encoder_mask=encoder_mask,
        sequence_metadata=sequence_metadata,
        batch=batch,
        enable_dropout=enable_dropout,
    )

    if self.sow_intermediates:
      for (k, v) in batch.items():
        self.sow('intermediates', k, v)
      self.sow('intermediates', 'final_encoder_outputs', encoder_outputs)

    encoder_mask = multimodal_feature.attention_mask_for_zeros(
        [pf.values for pf in encoder_features])

    if self.outputs_as_dict:
      # split the embeddings back to their features according their positions.
      encoder_outputs = embed_combine_result.to_dict(encoder_outputs)  # pytype: disable=attribute-error  # jax-ndarray
      encoder_mask = embed_combine_result.to_dict(encoder_mask)  # pytype: disable=attribute-error  # jax-ndarray
      encoder_segment_ids = embed_combine_result.to_dict(encoder_segment_ids)  # pytype: disable=attribute-error  # jax-ndarray

    return (encoder_outputs, encoder_mask, encoder_segment_ids)


class MultimodalEncoderDecoder(nn.Module, param_remapping.ParameterRemappable):
  """Transformer Model for multimodal-sequence to sequence translation.

  Attributes:
    encoder_factory: A callable that returns the lower-level Encoder object. If
      shared_token_embedder_factory is non-None, then the result of it will be
      passed as the `shared_token_embedder` argument to `encoder_factory`.
    decoder_factory: A callable that returns the lower-level Decoder object. If
      shared_token_embedder_factory is non-None, then the result of it will be
      passed as the `shared_token_embedder` argument to `decoder_factory`.
    dtype: Dtype for encoder/decoder to cast embedded inputs, and for attention
      mask generation.
    shared_token_embedder_factory: A callable that returns an embedder that can
      be shared between the encoder and decoder.
  """
  # Core components: encoder and decoder embedders and layers.
  encoder_factory: t5_architecture.MakeEncoderFn  # pytype: disable=module-attr
  decoder_factory: t5_architecture.MakeDecoderFn  # pytype: disable=module-attr

  # Configures behavior when the model is called. Many of these might eventually
  # be better as call parameters.
  dtype: Dtype = jnp.float32
  scan_layers: bool = False  # only used to smuggle this option to predict_fn.
  spmd_annotations: Any = None  # only used for scanned spmd layers

  # Set `shared_token_embedder_factory` to share the embeddings between encoder
  # and decoder.
  shared_token_embedder_factory: Optional[Callable[[], embedding.Embed]] = None
  # Otherwise set the embedder factories separately.
  encoder_token_embedder_factory: Optional[Callable[[], embedding.Embed]] = None
  decoder_token_embedder_factory: Optional[Callable[[], embedding.Embed]] = None

  # number of few shot examples
  num_shots: int = 0  # only used in episofic settings.

  # Distillation settings
  teacher_output_logits_factory: Optional[Callable[[], nn.Module]] = None
  distillation_mode: bool = False

  def setup(self):
    if self.shared_token_embedder_factory is None:
      assert (self.encoder_token_embedder_factory is not None and
              self.decoder_token_embedder_factory is not None)
      self.encoder_token_embedder_factory: Callable[[], embedding.Embed]
      self.decoder_token_embedder_factory: Callable[[], embedding.Embed]
      self.encoder_token_embedder = self.encoder_token_embedder_factory()
      self.decoder_token_embedder = self.decoder_token_embedder_factory()
      encoder_token_embedder = self.encoder_token_embedder
      decoder_token_embedder = self.decoder_token_embedder
    else:
      self.shared_token_embedder_factory: Callable[[], embedding.Embed]
      self.token_embedder = self.shared_token_embedder_factory()
      encoder_token_embedder = self.token_embedder
      decoder_token_embedder = self.token_embedder
    # TODO: Clean up SPMD annotation code.
    if self.spmd_annotations is None:
      encoder_annotations = None
      decoder_annotations = None
    else:
      encoder_annotations = self.spmd_annotations.get('encoder')
      decoder_annotations = self.spmd_annotations.get('decoder')
    encoder_factory_params = tuple(
        inspect.signature(self.encoder_factory).parameters.keys())
    if 'spmd_annotations' in encoder_factory_params:
      self.encoder = self.encoder_factory(
          shared_token_embedder=encoder_token_embedder,
          spmd_annotations=encoder_annotations)
    else:
      self.encoder = self.encoder_factory(
          shared_token_embedder=encoder_token_embedder)

    decoder_factory_params = tuple(
        inspect.signature(self.decoder_factory).parameters.keys())
    if 'spmd_annotations' in decoder_factory_params:
      self.decoder = self.decoder_factory(
          shared_token_embedder=decoder_token_embedder,
          spmd_annotations=decoder_annotations)
    else:
      self.decoder = self.decoder_factory(
          shared_token_embedder=decoder_token_embedder)

    if self.teacher_output_logits_factory is not None:
      self.teacher_logits_dense = self.teacher_output_logits_factory()

  def encode(self,
             encoder_batch: Mapping[str, Array],
             enable_dropout: bool = True):
    """Applies Transformer encoder-branch on the inputs.

    Args:
      encoder_batch: batch of encoder features, including packing info
      enable_dropout: whether dropout is enabled

    Returns:
      triple of (encoded feature array, encoder mask, encoder segment ids) from
        the transformer encoder.
    """
    return self.encoder(batch=encoder_batch, enable_dropout=enable_dropout)  # pytype: disable=attribute-error,wrong-keyword-args

  def decode(
      self,
      encoded: Array,
      encoder_mask: Array,
      encoder_segment_ids: Optional[Array],
      decoder_batch: Mapping[str, Array],
      enable_dropout: bool = True,
      decode: bool = False,
      max_decode_length: Optional[int] = None,
      prefill: bool = False,
      prefill_lengths: Optional[Array] = None,
  ):
    """Applies Transformer decoder-branch on encoded-input and target.

    Args:
      encoded: encoded input data from encoder.
      encoder_mask: encoder mask (for padding attention mask values)
      encoder_segment_ids: encoder segment ids (if packing is used)
      decoder_batch: features and targets for the decoder
      enable_dropout: Enables dropout if set to True.
      decode: Whether to prepare and use an autoregressive cache.
      max_decode_length: An optional integer specifying the maximum decoding
        length. Note that this is only used for defining the relative position
        embedding parameters.
      prefill: Whether to run a partial sequence to prefill the cache.
      prefill_lengths: The length of each partial sequence we are filling in the
        cache, lengths are inferred from the mask if not provided.

    Returns:
      logits array from transformer decoder.
    """
    packing = encoder_segment_ids is not None
    if self.num_shots > 0:
      # For packing encoder_segment_ids are used to denote different fewshot
      # examples rather than different training instances.
      packing = False

    decoder_input_tokens = decoder_batch['decoder_input_tokens']
    decoder_target_tokens = decoder_batch['decoder_target_tokens']
    if packing:
      decoder_segment_ids = decoder_batch['decoder_target_tokens_segment_ids']
      decoder_positions = decoder_batch['decoder_target_tokens_positions']
    else:
      decoder_segment_ids = None
      decoder_positions = None

    # Make padding attention masks.
    if decode:
      # fast autoregressive decoding uses only a special encoder-decoder mask
      decoder_mask = None
      encoder_decoder_mask = dense_attention.make_attention_mask(
          jnp.ones_like(decoder_target_tokens) > 0,
          encoder_mask,
          dtype=self.dtype)
    else:
      decoder_mask = dense_attention.make_decoder_mask(
          decoder_target_tokens=decoder_target_tokens,
          dtype=self.dtype,
          decoder_segment_ids=decoder_segment_ids)
      encoder_decoder_mask = dense_attention.make_attention_mask(
          decoder_target_tokens > 0, encoder_mask, dtype=self.dtype)

    # Add segmentation block-diagonal attention masks if using segmented data.
    if packing:
      if decode:
        raise ValueError(
            'During decoding, packing should not be used but '
            '`encoder_features[0].segment_ids` was passed to '
            '`Transformer.decode`.')
      if decoder_segment_ids is None:
        raise ValueError('decoder_segment_ids is required')
      if decoder_positions is None:
        raise ValueError('decoder_positions is required')
      encoder_decoder_mask = dense_attention.combine_masks(
          encoder_decoder_mask,
          dense_attention.make_attention_mask(
              decoder_segment_ids,
              encoder_segment_ids,
              jnp.equal,
              dtype=self.dtype))

    # When computing the logits, we don't need decoder_target_tokens, which is
    # needed for computing the loss.
    logits = self.decoder(  # pytype: disable=attribute-error
        encoded,
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        enable_dropout=enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length,
        prefill=prefill,
        prefill_lengths=prefill_lengths,
    )

    return logits.astype(self.dtype)

  def __call__(self,
               batch: Mapping[str, Array],
               enable_dropout: bool = True,
               decode: bool = False,
               max_decode_length: Optional[int] = None):
    """Applies Transformer model on the inputs.

    This method requires both decoder_target_tokens and decoder_input_tokens,
    which is a shifted version of the former. For a packed dataset, it usually
    has additional processing applied. For example, the first element of each
    sequence has id 0 instead of the shifted EOS id from the previous sequence.

    Args:
      batch: all encoder features, including `X_positions` and `X_segment_ids`
      enable_dropout: Enables dropout if set to True.
      decode: Whether to prepare and use an autoregressive cache.
      max_decode_length: An optional integer specifying the maximum decoding
        length. Note that this is only used for defining the relative position
        embedding parameters.

    Returns:
      logits array from full transformer.
    """
    encoder_batch = {
        k: v for (k, v) in batch.items() if not k.startswith('decoder')}
    decoder_batch = {
        k: v for (k, v) in batch.items() if k.startswith('decoder')}
    encoded, encoder_mask, encoder_segment_ids = self.encode(
        encoder_batch=encoder_batch, enable_dropout=enable_dropout)
    decode_result = self.decode(
        encoded,
        encoder_mask=encoder_mask,
        encoder_segment_ids=encoder_segment_ids,
        decoder_batch=decoder_batch,
        max_decode_length=max_decode_length,
        decode=decode,
        enable_dropout=enable_dropout)
    if self.distillation_mode:
      if 'decoder_embeddings' not in decoder_batch:
        raise ValueError('there must be a decoder_embeddings feature')
      if self.teacher_output_logits_factory is None:
        raise ValueError('must set `teacher_embeddings_factory`')
      student_logits = decode_result
      # Compute teacher_logits within the student.
      teacher_logits = self.teacher_logits_dense(
          decoder_batch['decoder_embeddings'])
      # Prevent backpropagation through the teacher embedding.
      teacher_logits = jax.lax.stop_gradient(teacher_logits)
      return (student_logits, teacher_logits)
    return decode_result  # logits
