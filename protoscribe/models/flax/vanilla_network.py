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

"""Vanilla transformer-based machine translation model from Flax.

This implementation is loosely based on the WMT example that comes with Flax.
"""

from typing import Any, Callable

import chex
import flax
from flax import linen as nn
from flax import struct
from jax import lax
import jax.numpy as jnp
import ml_collections
import numpy as np
from protoscribe.models.flax import utils

Initializer = flax.typing.Initializer


@struct.dataclass
class TransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

  output_vocab_size: int
  emb_dim: int = 512
  num_heads: int = 8
  num_encoder_layers: int = 6
  num_decoder_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_len: int = 2048
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  deterministic: bool = False
  decode: bool = False
  kernel_init: Initializer = nn.initializers.xavier_uniform()
  bias_init: Initializer = nn.initializers.normal(stddev=1e-6)
  posemb_init: Initializer | None = None


def sinusoidal_init(
    max_len: int = 2048,
    min_scale: float = 1.0,
    max_scale: float = 10000.0
) -> Callable[[Any, Any, Any], chex.Array]:
  """1D Sinusoidal Position Embedding Initializer.

  Note: the `max_len` is a maximum *output* stroke token sequence length.
  This is also used to compute fixed position embeddings for the input
  Protoscribe multimodal embeddings, which are generally shorter.

  Args:
      max_len: maximum possible length for the input.
      min_scale: float: minimum frequency-scale in sine grating.
      max_scale: float: maximum frequency-scale in sine grating.

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key, dtype
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
    pe[:, : d_feature // 2] = np.sin(position * div_term)
    pe[:, d_feature // 2 : 2 * (d_feature // 2)] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class AddPositionEmbeddings(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    decode: whether to run in single-position autoregressive mode.
  """

  config: TransformerConfig
  decode: bool = False

  @nn.compact
  def __call__(self, inputs: chex.Array) -> chex.Array:
    """Applies AddPositionEmbeddings module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init in the configuration.

    Args:
      inputs: input data.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    config = self.config
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, (
        "Number of dimensions should be 3, but it is: %d" % inputs.ndim
    )
    length = inputs.shape[1]
    pos_emb_shape = (1, config.max_len, inputs.shape[-1])
    if config.posemb_init is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(max_len=config.max_len)(
          None, pos_emb_shape, None
      )
    else:
      pos_embedding = self.param(
          "pos_embedding", config.posemb_init, pos_emb_shape
      )
    pe = pos_embedding[:, :length, :]

    # We use a cache position index for tracking decoding position.
    if self.decode:
      is_initialized = self.has_variable("cache", "cache_index")
      cache_index = self.variable(
          "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.uint32)
      )
      if is_initialized:
        i = cache_index.value
        cache_index.value = i + 1
        _, _, df = pos_embedding.shape
        pe = lax.dynamic_slice(pos_embedding, jnp.array((0, i, 0)), (1, 1, df))

    return inputs + pe


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
  """

  config: TransformerConfig
  out_dim: int | None = None

  @nn.compact
  def __call__(self, inputs: chex.Array) -> chex.Array:
    """Applies Transformer MlpBlock module."""
    config = self.config
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        config.mlp_dim,
        dtype=jnp.float32,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
    )(inputs)
    x = nn.relu(x)
    x = nn.Dropout(rate=config.dropout_rate)(
        x, deterministic=config.deterministic
    )
    output = nn.Dense(
        actual_out_dim,
        dtype=jnp.float32,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
    )(x)
    output = nn.Dropout(rate=config.dropout_rate)(
        output, deterministic=config.deterministic
    )
    return output


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """

  config: TransformerConfig

  @nn.compact
  def __call__(
      self, inputs: chex.Array, encoder_mask: chex.Array | None = None
  ) -> chex.Array:
    """Applies Encoder1DBlock module.

    Args:
      inputs: input data.
      encoder_mask: encoder self-attention mask.

    Returns:
      output after transformer encoder block.
    """
    config = self.config

    # Attention block.
    chex.assert_rank(inputs, 3)
    x = nn.LayerNorm(dtype=jnp.float32)(inputs)
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
    )(x, mask=encoder_mask)

    x = nn.Dropout(rate=config.dropout_rate)(
        x, deterministic=config.deterministic
    )
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=jnp.float32)(x)
    y = MlpBlock(config=config)(y)

    return x + y


class EncoderDecoder1DBlock(nn.Module):
  """Transformer encoder-decoder layer.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """

  config: TransformerConfig

  @nn.compact
  def __call__(
      self,
      targets: chex.Array,
      encoded: chex.Array,
      decoder_self_attention_mask: chex.Array | None = None,
      cross_attention_mask: chex.Array | None = None
  ) -> chex.Array:
    """Applies EncoderDecoder1DBlock module.

    Args:
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
    x = nn.LayerNorm(dtype=jnp.float32)(targets)
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
    y = nn.LayerNorm(dtype=jnp.float32)(x)
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
    z = nn.LayerNorm(dtype=jnp.float32)(y)
    z = MlpBlock(config=config)(z)

    return y + z


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """

  config: TransformerConfig

  @nn.compact
  def __call__(
      self, inputs: chex.Array, encoder_mask: chex.Array | None = None
  ) -> chex.Array:
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      encoder_mask: decoder self-attention mask.

    Returns:
      output of a transformer encoder.
    """
    config = self.config
    chex.assert_rank(inputs, 3)  # (batch, len, dim)

    # Input projection.
    input_project = nn.Dense(
        config.emb_dim,
        use_bias=True,
        dtype=jnp.float32,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
    )
    x = inputs.astype("float32")
    x = input_project(x)
    x = AddPositionEmbeddings(
        config=config, decode=False, name="posembed_input"
    )(x)
    x = nn.Dropout(rate=config.dropout_rate)(
        x, deterministic=config.deterministic
    )

    x = x.astype(jnp.float32)

    # Input Encoder
    for lyr in range(config.num_encoder_layers):
      x = Encoder1DBlock(config=config, name=f"encoderblock_{lyr}")(
          x, encoder_mask
      )

    encoded = nn.LayerNorm(dtype=jnp.float32, name="encoder_norm")(x)

    return encoded


class Decoder(nn.Module):
  """Transformer Model Decoder for sequence to sequence translation.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """

  config: TransformerConfig

  @nn.compact
  def __call__(
      self,
      encoded: chex.Array,
      targets: chex.Array,
      decoder_self_attention_mask: chex.Array | None = None,
      cross_attention_mask: chex.Array | None = None
  ) -> chex.Array:
    """Applies Transformer model on the input decoder tokens.

    Args:
      encoded: encoded input data from encoder.
      targets: target inputs.
      decoder_self_attention_mask: decoder self-attention mask.
      cross_attention_mask: encoder-decoder attention mask.

    Returns:
      output of a transformer decoder.
    """
    chex.assert_rank(encoded, 3)  # (batch, len, depth)
    chex.assert_rank(targets, 2)  # (batch, len)

    # Embed inputs (current sketch tokens).
    config = self.config
    y = targets.astype("int32")
    if not config.decode:
      y = utils.shift_right(y)
    y = nn.Embed(
        num_embeddings=config.output_vocab_size,
        features=config.emb_dim,
        embedding_init=nn.initializers.normal(stddev=1.0),
    )(y)

    # Decode from continuous inputs.
    y = AddPositionEmbeddings(
        config=config, decode=config.decode, name="posembed_output"
    )(y)
    y = nn.Dropout(rate=config.dropout_rate)(
        y, deterministic=config.deterministic
    )

    y = y.astype(jnp.float32)

    # Target-Input Decoder
    for lyr in range(config.num_decoder_layers):
      y = EncoderDecoder1DBlock(
          config=config, name=f"encoderdecoderblock_{lyr}"
      )(
          y,
          encoded,
          decoder_self_attention_mask=decoder_self_attention_mask,
          cross_attention_mask=cross_attention_mask,
      )
    y = nn.LayerNorm(dtype=jnp.float32, name="encoderdecoder_norm")(y)

    # Decoded Logits
    logits = nn.Dense(
        config.output_vocab_size,
        dtype=jnp.float32,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
        name="logitdense",
    )(y)
    return logits


class Transformer(nn.Module):
  """Vanilla transformer Model adopted from sequence to sequence translation.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """

  config: TransformerConfig

  def setup(self):
    config = self.config
    self.encoder = Encoder(config=config)
    self.decoder = Decoder(config=config)

  def encode(self, inputs: chex.Array) -> chex.Array:
    """Applies Transformer encoder-branch on the inputs.

    Args:
      inputs: input data.

    Returns:
      encoded feature array from the transformer encoder.
    """
    # Make padding attention mask.
    inputs_mask = utils.nonzero_sequence_mask(inputs)
    encoder_mask = nn.make_attention_mask(
        inputs_mask > 0, inputs_mask > 0, dtype=jnp.float32
    )
    return self.encoder(
        inputs, encoder_mask=encoder_mask
    )

  def decode(
      self,
      encoded: chex.Array,
      inputs: chex.Array,  # only needed for masks
      targets: chex.Array,
  ) -> chex.Array:
    """Applies Transformer decoder-branch on encoded-input and target.

    Args:
      encoded: encoded input data from encoder.
      inputs: input data (only needed for masking).
      targets: target data.

    Returns:
      logits array from transformer decoder.
    """
    config = self.config

    # Make padding attention masks.
    inputs_mask = utils.nonzero_sequence_mask(inputs)
    if config.decode:
      # For fast autoregressive decoding only a special encoder-decoder mask is
      # used.
      decoder_self_attention_mask = None
      cross_attention_mask = nn.make_attention_mask(
          jnp.ones_like(targets) > 0, inputs_mask > 0, dtype=jnp.float32
      )
    else:
      decoder_self_attention_mask = nn.combine_masks(
          nn.make_attention_mask(targets > 0, targets > 0, dtype=jnp.float32),
          nn.make_causal_mask(targets, dtype=jnp.float32),
      )
      cross_attention_mask = nn.make_attention_mask(
          targets > 0, inputs_mask > 0, dtype=jnp.float32
      )

    logits = self.decoder(
        encoded,
        targets,
        decoder_self_attention_mask=decoder_self_attention_mask,
        cross_attention_mask=cross_attention_mask,
    )
    return logits.astype(jnp.float32)

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

    encoded = self.encode(inputs)
    logits = self.decode(
        encoded,
        inputs,  # only used for masks
        targets,
    )
    return dict(logits=logits)


def get_config(
    config: ml_collections.ConfigDict, run_type: utils.RunType
) -> TransformerConfig:
  """Constructs model configuration from global configuration.

  Args:
    config: Global configuration dictionary.
    run_type: Execution mode which can be one of: `TRAIN`, `EVAL`, `PREDICT`.

  Returns:
    Model configuration.
  """
  return TransformerConfig(
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
  )
