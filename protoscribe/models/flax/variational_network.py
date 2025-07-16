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

"""Variational model along the lines of conditional VAE (CVAE).

The architecture consists of a dual encoder: one for conditional features
(the actual Protoscribe multimodal embeddings for semantics, speech, vision
and so on we care about) and one for the targets to be reconstructed (VAE).
"""

import dataclasses
import logging
import math

import chex
from flax import linen as nn
from flax import struct
import jax
import jax.numpy as jnp
import ml_collections
from protoscribe.models.flax import utils
from protoscribe.models.flax import vanilla_network
from protoscribe.sketches.utils import glyph_embeddings
from protoscribe.sketches.utils import latents
from protoscribe.sketches.utils import pooling


@struct.dataclass
class TransformerConfig(vanilla_network.TransformerConfig):
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  latent_size: int = -1
  latent_num_encoder_layers: int = 6
  encoder_pooling: str = "none"
  encoder_mask_non_concept_tokens: bool = False
  latent_share_token_embeddings: bool = False
  latent_blend_strategy: str = "encoder_concat"
  latent_conditional_layer_norm: bool = False
  latent_neftune_noise_alpha: float = 5.
  min_log_var: float = math.log(1e-5)
  max_log_var: float = math.log(160.0)


def _project_layer(config: TransformerConfig, features: int) -> nn.Dense:
  """Returns linear projection layer."""
  return nn.Dense(
      features=features,
      use_bias=True,
      kernel_init=config.kernel_init,
      bias_init=config.bias_init,
  )


def _apply_neftune(
    rng: chex.PRNGKey,
    embedding: chex.Array,
    noise_alpha: float = 5.
) -> chex.Array:
  r"""Applies NEFTune noisification to original embeddings.

  See https://arxiv.org/abs/2310.05914. This is a JAX implementation of an
  equivalent functionality in corpus reader. This only makes sense during
  training. See `_maybe_noisify` API in
  protoscribe/corpus/reader/corpus_reader.py.

  Args:
    rng: Random number generator.
    embedding: Array representing the embeddings to be noisifed with shape
      (B, L, D).
    noise_alpha: Noise factor \alpha.

  Returns:
    Embeddings with noise applied.
  """
  chex.assert_rank(embedding, 3)

  l_by_d = embedding.shape[1] * embedding.shape[2]
  scale = noise_alpha / jnp.sqrt(l_by_d)
  return embedding + jax.random.uniform(
      rng,
      shape=embedding.shape,
      dtype=jnp.float32,
      minval=-scale,
      maxval=scale
  )


class TokenEncoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    shared_embedding: a shared embedding layer to use.
  """

  config: TransformerConfig
  shared_embedding: nn.Module | None = None

  @nn.compact
  def __call__(
      self,
      inputs: chex.Array,
      encoder_mask: chex.Array | None = None,
      rng: chex.PRNGKey | None = None
  ) -> chex.Array:
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      encoder_mask: decoder self-attention mask.
      rng: Random number generator key that will only be used in training to
        apply noise.

    Returns:
      output of a transformer encoder.
    """
    config = self.config
    chex.assert_rank(inputs, 2)  # (batch, len)

    # Embed the inputs.
    if self.shared_embedding is None:
      input_embed = nn.Embed(
          num_embeddings=config.output_vocab_size,
          features=config.emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0),
      )
    else:
      input_embed = self.shared_embedding
    x = inputs.astype("int32")
    x = input_embed(x)

    # Apply noisification if configured.
    if rng is not None and config.latent_neftune_noise_alpha > 0.:
      x = _apply_neftune(
          rng=rng,
          embedding=x,
          noise_alpha=config.latent_neftune_noise_alpha
      )

    # Apply positional embeddings and dropout.
    x = vanilla_network.AddPositionEmbeddings(
        config=config, decode=False, name="token_posembed_input")(x)
    x = nn.Dropout(rate=config.dropout_rate)(
        x, deterministic=config.deterministic
    )

    x = x.astype(jnp.float32)

    # Apply encoder layers.
    for lyr in range(config.latent_num_encoder_layers):
      x = vanilla_network.Encoder1DBlock(
          config=config, name=f"token_encoder_block_{lyr}"
      )(
          x, encoder_mask
      )

    encoded = nn.LayerNorm(dtype=jnp.float32, name="token_encoder_norm")(x)

    return encoded


class ConditionalLayerNorm(nn.Module):
  """Conditional layer normalization (CLN).

  See, e.g.: https://arxiv.org/pdf/2103.00993
  """

  config: TransformerConfig

  @nn.compact
  def __call__(self, x: chex.Array, z: chex.Array) -> chex.Array:
    """Applies conditional layer normalization.

    Args:
      x: Input tensor with shape (B, L, D).
      z: Latent code tensor with shape (B, D_l).

    Returns:
      Output tensor with shape (B, L, D).
    """
    chex.assert_rank(x, 3)
    chex.assert_rank(z, 2)

    # Fallback to vanilla layer normalization if CLN is disabled.
    config = self.config
    if not config.latent_conditional_layer_norm:
      return nn.LayerNorm(dtype=jnp.float32)(x)

    # Use standard LayerNorm but disable its learned scale and bias parameters.
    # We only want it to compute (x - mean) / std.
    normalized_x = nn.LayerNorm(
        use_scale=False,  # We will use predicted scale instead.
        use_bias=False,   # We will use predicted bias instead.
        dtype=jnp.float32,
        name="core_layer_norm"
    )(x)

    # Predict scale and bias parameters (gamma and beta) from the latent code.
    # Use different linear layers for gamma and beta predictions.
    feature_dim = x.shape[2]
    gamma = nn.Dense(
        features=feature_dim,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
        dtype=jnp.float32,
        name="gamma_predictor"
    )(z)
    beta = nn.Dense(
        features=feature_dim,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
        dtype=jnp.float32,
        name="beta_predictor"
    )(z)

    # Apply the predicted gamma and beta.
    # gamma and beta are broadcast from (B, D) to (B, L, D).
    output = gamma[:, None, :] * normalized_x + beta[:, None, :]

    return output


class TargetsEmbedder(nn.Module):
  """Helper for embedding the targets.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """

  config: TransformerConfig
  token_embedder: nn.Module

  @nn.compact
  def __call__(self, targets: chex.Array) -> chex.Array:
    """Embeds the supplied targets.

    Args:
      targets: Target sketch tokens array (B, L).

    Returns:
      Embedded targets array (B, L, D).
    """
    chex.assert_rank(targets, 2)  # (batch, len)

    config = self.config
    y = targets.astype("int32")
    if not config.decode:
      y = utils.shift_right(y)
    y = self.token_embedder(y)

    return y


class EncoderDecoder1DBlock(nn.Module):
  """Transformer encoder-decoder layer taking into account latents.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """

  config: TransformerConfig

  @nn.compact
  def __call__(
      self,
      latent_z: chex.Array,
      targets: chex.Array,
      encoded: chex.Array,
      decoder_self_attention_mask: chex.Array | None = None,
      cross_attention_mask: chex.Array | None = None
  ) -> chex.Array:
    """Applies EncoderDecoder1DBlock module.

    Args:
      latent_z: sampled latent code
      targets: input data for decoder
      encoded: input data from encoder
      decoder_self_attention_mask: decoder self-attention mask.
      cross_attention_mask: encoder-decoder attention mask.

    Returns:
      output after transformer encoder-decoder block.
    """
    config = self.config

    # Decoder block.
    chex.assert_rank(targets, 3)
    x = ConditionalLayerNorm(config=config)(targets, latent_z)
    x = nn.MultiHeadDotProductAttention(
        num_heads=config.num_heads,
        dtype=jnp.float32,
        qkv_features=config.qkv_dim,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=config.attention_dropout_rate,
        deterministic=config.deterministic,
        decode=config.decode,
    )(x, mask=decoder_self_attention_mask)
    x = nn.Dropout(rate=config.dropout_rate)(
        x, deterministic=config.deterministic
    )
    x = x + targets

    # Encoder-Decoder block.
    y = ConditionalLayerNorm(config=config)(x, latent_z)
    y = nn.MultiHeadDotProductAttention(
        num_heads=config.num_heads,
        dtype=jnp.float32,
        qkv_features=config.qkv_dim,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=config.attention_dropout_rate,
        deterministic=config.deterministic,
    )(y, encoded, mask=cross_attention_mask)

    y = nn.Dropout(rate=config.dropout_rate)(
        y, deterministic=config.deterministic
    )
    y = y + x

    # MLP block.
    z = ConditionalLayerNorm(config=config)(y, latent_z)
    z = vanilla_network.MlpBlock(config=config)(z)

    return y + z


class Decoder(nn.Module):
  """Transformer Model Decoder for sequence to sequence translation.

  This decoder operates on continuous (embedded) representation of tokens
  rather than discrete tokens themselves.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """

  config: TransformerConfig

  @nn.compact
  def __call__(
      self,
      z: chex.Array,
      encoded: chex.Array,
      embedded_tokens: chex.Array,
      decoder_self_attention_mask: chex.Array | None = None,
      cross_attention_mask: chex.Array | None = None
  ) -> chex.Array:
    """Applies Transformer model on the input decoder tokens.

    Args:
      z: sampled latent code.
      encoded: encoded input data from encoder.
      embedded_tokens: target inputs.
      decoder_self_attention_mask: decoder self-attention mask.
      cross_attention_mask: encoder-decoder attention mask.

    Returns:
      output of a transformer decoder.
    """
    chex.assert_rank(z, 2)
    chex.assert_rank(encoded, 3)  # (batch, len, depth)
    chex.assert_rank(embedded_tokens, 3)  # (batch, len, dim)

    config = self.config
    y = embedded_tokens

    # Decode from continuous inputs.
    y = vanilla_network.AddPositionEmbeddings(
        config=config, decode=config.decode, name="posembed_output"
    )(y)
    y = nn.Dropout(rate=config.dropout_rate)(
        y, deterministic=config.deterministic
    )

    y = y.astype(jnp.float32)

    # Target-Input Decoder
    for lyr in range(config.num_decoder_layers):
      y = EncoderDecoder1DBlock(
          config=config, name=f"encoder_decoder_block_{lyr}"
      )(
          z,
          y,
          encoded,
          decoder_self_attention_mask=decoder_self_attention_mask,
          cross_attention_mask=cross_attention_mask,
      )
    y = nn.LayerNorm(dtype=jnp.float32, name="encoder_decoder_norm")(y)

    # Decoded Logits
    logits = nn.Dense(
        config.output_vocab_size,
        dtype=jnp.float32,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
        name="logit_dense",
    )(y)
    return logits


class LatentBlender(nn.Module):
  """Helper module for blending the latent code with encoder/decoder vectors.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """

  config: TransformerConfig

  @nn.compact
  def __call__(
      self,
      z: chex.Array,
      encoded_cond_inputs: chex.Array,
      decoder_inputs: chex.Array
  ) -> tuple[chex.Array, chex.Array]:
    """Blends the latent code with encoder outputs or decoder inputs.

    Args:
      z: Latent code (B, D_latent).
      encoded_cond_inputs: Encoder outputs for conditional embeddings of shape
        (B, L1, D), where the conditional embeddings represent the multimodal
        inputs (semantics, speech and so on).
      decoder_inputs: Embedded decoder input tokens (B, L2, D).

    Returns:
      Tuple consisting of encoded conditional embeddings and embedded decoder
      input tokens with latent code blended in.
    """
    chex.assert_rank(z, 2)
    chex.assert_rank(encoded_cond_inputs, 3)
    chex.assert_rank(decoder_inputs, 3)

    config = self.config
    encoded = encoded_cond_inputs
    y = decoder_inputs

    # Build encoder outputs for the decoder. In CVAE style, combine samples from
    # posterior distribution with encoded conditional features.
    blend_strategy = config.latent_blend_strategy
    if blend_strategy == "encoder_concat":
      z = z[:, None, :]
      z = jnp.repeat(z, encoded_cond_inputs.shape[1], axis=1)
      encoded = jnp.concatenate([encoded_cond_inputs, z], axis=-1)
      encoded = _project_layer(
          config, features=encoded_cond_inputs.shape[2]
      )(encoded)
    elif blend_strategy == "encoder_prepend":
      z = z[:, None, :]
      z = _project_layer(config, features=encoded_cond_inputs.shape[2])(z)
      # Prepend along the temporal axis: (B, L + 1, D).
      encoded = jnp.concatenate([z, encoded_cond_inputs], axis=1)
    elif blend_strategy == "decoder_concat":
      z = z[:, None, :]
      z = jnp.repeat(z, y.shape[1], axis=1)
      y_pre = y
      y = jnp.concatenate([y, z], axis=-1)
      y = _project_layer(config, features=y_pre.shape[2])(y)
    elif blend_strategy == "decoder_add":
      z = z[:, None, :]
      z = jnp.repeat(z, y.shape[1], axis=1)
      z = _project_layer(config, features=y.shape[2])(z)
      y = y + z
    elif blend_strategy == "no_conditional":
      z = z[:, None, :]
      encoded = _project_layer(config, features=y.shape[2])(z)
    elif blend_strategy == "no_latents":
      pass  # Use conditional encoded outputs and embedded target tokens as is.
    else:
      raise ValueError(f"Unknown latent blend strategy: {blend_strategy}")

    return encoded, y


class VariationalTransformer(nn.Module):
  """Variational transformer model.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """

  config: TransformerConfig

  def setup(self) -> None:
    config = self.config
    self.cond_encoder = (
        vanilla_network.Encoder(config=config)
        if config.latent_blend_strategy != "no_conditional" else None
    )
    self.token_embedder = nn.Embed(
        num_embeddings=config.output_vocab_size,
        features=config.emb_dim,
        embedding_init=nn.initializers.normal(stddev=1.0),
    )
    self.token_encoder = None
    if config.latent_blend_strategy != "no_latents":
      self.token_encoder = TokenEncoder(
          config=config,
          shared_embedding=(
              self.token_embedder
              if config.latent_share_token_embeddings else None
          )
      )
    self.targets_embedder = TargetsEmbedder(
        config=config, token_embedder=self.token_embedder
    )
    self.decoder = Decoder(config=config)

    assert config.latent_size > 0
    self.z_mean_head = _project_layer(config, features=config.latent_size)
    self.z_log_var_head = _project_layer(config, features=config.latent_size)
    self.latent_blender = LatentBlender(config=config)

  def encode(
      self,
      cond_inputs: chex.Array,
      input_tokens: chex.Array | None = None,
      concept_tokens_mask: chex.Array | None = None,
      rng: chex.PRNGKey | None = None
  ) -> tuple[chex.Array, chex.Array, chex.Array]:
    """Applies Transformer encoder-branch on the inputs.

    Args:
      cond_inputs: Conditional features for encoding (embeddings) which
        represent multimodal inputs (semantics, speech and so on).
      input_tokens: input data, same as targets which we are reconstructing.
        These are sketch tokens. Not specified in decoding mode.
      concept_tokens_mask: An array of the same shape as `input_tokens`
        representing a boolean mask where 1s correspond to concept-specific
        stroke tokens and 0th to everything else. Note that end-of-stroke
        tokens are always masked out regardless of their membership.
      rng: Random number generator key used during training.

    Returns:
      A tuple consisting of posterior distribution parameters and encoded
      conditional inputs. Either was these can be dummies if either the
      latent or the conditional subnetworks are disabled.
    """
    config = self.config
    chex.assert_rank(cond_inputs, 3)  # (batch, len, dim)

    # Conditional features: Make padding attention mask and encode through
    # transformer.
    if self.cond_encoder is not None:
      cond_inputs_mask = utils.nonzero_sequence_mask(cond_inputs)
      cond_encoder_mask = nn.make_attention_mask(
          cond_inputs_mask > 0, cond_inputs_mask > 0, dtype=jnp.float32
      )
      encoded_cond_inputs = self.cond_encoder(
          cond_inputs, encoder_mask=cond_encoder_mask
      )
    else:
      encoded_cond_inputs = jnp.zeros(
          (cond_inputs.shape[0], cond_inputs.shape[1], config.emb_dim),
          dtype=jnp.float32
      )

    # In decoding mode or when the latents are completely disabled, there is no
    # need to derive posterior distribution parameters for the sketch tokens.
    if config.decode or self.token_encoder is None:
      dummy_z_mean = jnp.zeros((cond_inputs.shape[0], config.latent_size))
      dummy_z_log_var = jnp.ones((cond_inputs.shape[0], config.latent_size))
      return dummy_z_mean, dummy_z_log_var, encoded_cond_inputs

    # Run the encoder over the sketch tokens. Before running the encoder compute
    # self-attention mask that excludes padding tokens. Also adjust the
    # encoder self-attention mask to exclude non-concept tokens, if configured.
    assert input_tokens is not None  # Placate pylint.
    chex.assert_rank(input_tokens, 2)  # (batch, len)
    chex.assert_equal_shape([input_tokens, concept_tokens_mask])

    valid_tokens_mask = input_tokens > 0
    if config.encoder_mask_non_concept_tokens:
      valid_tokens_mask *= concept_tokens_mask

    token_encoder_mask = nn.make_attention_mask(
        valid_tokens_mask, valid_tokens_mask, dtype=jnp.float32
    )
    encoded_input_tokens = self.token_encoder(
        inputs=input_tokens,
        encoder_mask=token_encoder_mask,
        rng=rng
    )

    # Pool the encoder outputs.
    if config.encoder_pooling != "none":
      encoded_and_pooled = pooling.get_pooling(
          pooling=config.encoder_pooling,
          enc=encoded_input_tokens,
          mask=valid_tokens_mask
      )
    else:
      encoded_and_pooled = jnp.reshape(
          encoded_input_tokens, (encoded_input_tokens.shape[0], -1)
      )

    # Estimate the parameters of posterior distribution q(z|x) of shape (B, S),
    # where S is the latent dimension.
    z_mean = self.z_mean_head(encoded_and_pooled)
    z_log_var = self.z_log_var_head(encoded_and_pooled)
    z_log_var = jnp.clip(
        z_log_var, min=config.min_log_var, max=config.max_log_var
    )
    return z_mean, z_log_var, encoded_cond_inputs

  def decode(
      self,
      encoded: tuple[chex.Array, chex.Array],
      cond_inputs: chex.Array,  # only needed for masks
      targets: chex.Array,
  ) -> chex.Array:
    """Applies Transformer decoder-branch on encoded-input and target.

    Args:
      encoded: This is a tuple consisting of a code sampled from the posterior
        distribution and the encoded conditional features (multimodal
        embeddings).
      cond_inputs: input data (only needed for masking). These are the original
        conditional features representing our multimodal embeddings (sematics,
        speech, and so on).
      targets: target tokens.

    Returns:
      logits array from transformer decoder.
    """

    # Embed target tokens and unpack the latent variable and encoded conditional
    # features.
    z, encoded_cond_inputs = encoded
    y = self.targets_embedder(targets)  # (B, L, D).

    # Blend with the latent code: This yields (possibly modified) encoder
    # outputs and decoder input token embeddings.
    encoded, y = self.latent_blender(
        z=z,
        encoded_cond_inputs=encoded_cond_inputs,
        decoder_inputs=y
    )

    # Compute the masks and run the decoder. Note that the masks are computed
    # *after* blending the latents because some blending modes may modify the
    # lengths of the sequences involved.
    decoder_self_attention_mask, cross_attention_mask = self._decoder_masks(
        encoded, targets
    )
    logits = self.decoder(
        z=z,
        encoded=encoded,
        embedded_tokens=y,
        decoder_self_attention_mask=decoder_self_attention_mask,
        cross_attention_mask=cross_attention_mask,
    )
    return logits

  def __call__(
      self, features: dict[str, chex.Array]
  ) -> dict[str, chex.Array]:
    """Applies Transformer model on the inputs.

    Args:
      features: A dictionary of arrays representing the input batch features.

    Returns:
      logits array from full transformer.
    """
    inputs = features["inputs"]
    targets = features["targets"]

    # Compute mask for token encoder. This will be used (if configured) to mask
    # out all the non-concept tokens. The mask will contain ones for all the
    # concept-related stroke tokens and zeros for everything else. Note that the
    # current implementation of masking always masks out end-of-stroke tokens
    # even if they belong to concept glyphs. In other words, the concept-related
    # stroke sequence looks like [... 1 1 1 0 1 1 0 1 1 1 1 0 ...], where 0s
    # correspond to end-of-stroke tokens.
    config = self.config
    concept_tokens_mask = None
    if not config.decode:
      # No need for concept masking in prediction mode.
      concept_tokens_mask = (
          glyph_embeddings.construct_loss_mask_for_stroke_glyph_affiliations(
              glyphs=features["text.glyph.tokens"],
              glyph_types=features["text.glyph.types"],
              stroke_glyph_affiliations=(
                  features["sketch.glyph_affiliations.ids"]
              ),
              concepts=True
          )
      )
      chex.assert_equal_shape([targets, concept_tokens_mask])

    # Encode the target tokens obtaining the parameters of the posterior
    # distribution. Also encode the embeddings (conditional inputs).
    z_mean, z_log_var, encoded_cond_inputs = self.encode(
        cond_inputs=inputs,
        input_tokens=targets,
        concept_tokens_mask=concept_tokens_mask,
        rng=None if config.deterministic else self.make_rng("latents")
    )

    # Obtain the latent code by sampling from the posterior.
    if not config.deterministic:
      z = latents.sample_gaussian(
          z_mean, z_log_var, rng=self.make_rng("latents")
      )
    else:
      # No randomness in evaluation mode.
      z = latents.sample_gaussian(z_mean, z_log_var)

    logits = self.decode(
        (z, encoded_cond_inputs),
        inputs,  # only used for masks
        targets,
    )

    model_outputs = dict(logits=logits)
    if not config.decode and config.latent_blend_strategy != "no_latents":
      model_outputs["z_mean"] = z_mean
      model_outputs["z_log_var"] = z_log_var
    return model_outputs

  def _decoder_masks(
      self,
      cond_inputs: chex.Array,
      targets: chex.Array
  ) -> tuple[chex.Array | None, chex.Array]:
    """Returns all the masks required for running the decoder.

    Args:
      cond_inputs: Conditional features for encoding (embeddings) which
        represent multimodal inputs to the model (semantics, speech, and so on).
      targets: Decoder target tokens.

    Returns:
      Tuple of decoder self-attention mask and encoder-decoder cross-attention
      mask.
    """
    # Make padding attention masks.
    cond_inputs_mask = utils.nonzero_sequence_mask(cond_inputs)
    if self.config.decode:
      # For fast autoregressive decoding only a special encoder-decoder mask is
      # used.
      decoder_self_attention_mask = None
      cross_attention_mask = nn.make_attention_mask(
          jnp.ones_like(targets) > 0, cond_inputs_mask > 0, dtype=jnp.float32
      )
    else:
      decoder_self_attention_mask = nn.combine_masks(
          nn.make_attention_mask(targets > 0, targets > 0, dtype=jnp.float32),
          nn.make_causal_mask(targets, dtype=jnp.float32),
      )
      cross_attention_mask = nn.make_attention_mask(
          targets > 0, cond_inputs_mask > 0, dtype=jnp.float32
      )

    return decoder_self_attention_mask, cross_attention_mask


def get_config(
    config: ml_collections.ConfigDict, run_type: utils.RunType
) -> TransformerConfig:
  """Constructs model configuration from main configuration.

  Args:
    config: Global configuration dictionary.
    run_type: Execution mode which can be one of: `TRAIN`, `EVAL`, `PREDICT`.

  Returns:
    Model configuration.
  """
  config = TransformerConfig(
      output_vocab_size=config.protoscribe.vocab_size,
      emb_dim=config.emb_dim,
      num_heads=config.num_heads,
      num_encoder_layers=config.num_encoder_layers,
      num_decoder_layers=config.num_decoder_layers,
      qkv_dim=config.qkv_dim,
      mlp_dim=config.mlp_dim,
      max_len=config.protoscribe.max_stroke_sequence_length,
      dropout_rate=config.dropout_rate,
      attention_dropout_rate=config.attention_dropout_rate,
      deterministic=False if run_type == utils.RunType.TRAIN else True,
      decode=True if run_type == utils.RunType.PREDICT else False,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.normal(stddev=1e-6),
      latent_size=config.latents.dimension,
      latent_num_encoder_layers=config.latents.num_encoder_layers,
      latent_share_token_embeddings=config.latents.share_token_embeddings,
      latent_conditional_layer_norm=config.latents.conditional_layer_norm,
      latent_neftune_noise_alpha=config.latents.neftune_noise_alpha,
      encoder_pooling=config.latents.encoder_pooling,
      encoder_mask_non_concept_tokens=(
          config.latents.encoder_mask_non_concept_tokens
      ),
      latent_blend_strategy=config.latents.blend_strategy
  )
  logging.info("Model configuration: %s", dataclasses.asdict(config))
  return config
