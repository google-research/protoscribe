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

"""Default Hyperparameter configuration.

Stroke token prediction from semantic category embeddings.
"""

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # Name of TFDS translation dataset to use.
  config.dataset_name = "protoscribe"

  # Name of the backend: `vanilla` (regular transformer) or `variational`
  # (variational architecture).
  config.backend_type = "variational"

  # Per device batch size for training.
  config.per_device_batch_size = 12

  # Number of training steps.
  config.num_train_steps = 500_000

  # Number of steps to take during evaluation.
  config.num_eval_steps = 20

  # Cross entropy loss label smoothing.
  config.label_smoothing = 0.1

  # Number of transformer encoder layers.
  config.num_encoder_layers = 4

  # Number of transformer decoder layers.
  config.num_decoder_layers = 16

  # Size of query/key/value for attention. This correspondings to the
  # product of number of attention heads by dimension of an individual
  # head.
  config.qkv_dim = 1024
  # Size of embeddings.
  config.emb_dim = 1024
  # Size of the MLP.
  config.mlp_dim = 2048
  # Number of attention heads.
  config.num_heads = 16

  # Dropout rate.
  config.dropout_rate = 0.1

  # Attention dropout rate.
  config.attention_dropout_rate = 0.1

  # Whether to save model checkpoints.
  config.save_checkpoints = True
  # Whether to restore from existing model checkpoints.
  config.restore_checkpoints = True

  # Save a checkpoint every these number of steps.
  config.checkpoint_every_steps = 20_000
  # Frequency of eval during training.
  config.eval_every_steps = 10_000

  # Integer for PRNG random seed.
  config.seed = 0

  # ----------------------------------------------------------------------------
  # Layers related to latent variables:
  # ----------------------------------------------------------------------------
  # This configuration section is ignored by the vanilla transformer
  # implementation.

  config.latents = ml_collections.ConfigDict()

  # For variational models: the following variable corresponds to the dimension
  # of the latents.
  config.latents.dimension = 128

  # Number of encoder layers for the latent variable.
  config.latents.num_encoder_layers = 16

  # For vatiational models: Type of encoder pooling. One of: `none`, `max`,
  # `mean`, `first` or `last`.
  config.latents.encoder_pooling = "mean"

  # If enabled mask out all the strokes corresponding to number glyphs and
  # special tokens in the encoder for the latents.
  config.latents.encoder_mask_non_concept_tokens = True

  # Share token embeddings between token encoder and decoder.
  config.latents.share_token_embeddings = True

  # How to blend the latents with the conditional features (multimodal
  # embeddings) in the decoder:
  #   `no_conditional`: Do not use conditional features at all, which
  #     corresponds to pure VAE architecture.
  #   `no_latents`: Disable variational component. Useful for testing.
  #   `encoder_concat`: Concatenate with each vector in encoder outputs.
  #   `encoder_prepend`: Project and prepend the latent to the encoder outputs.
  #   `decoder_concat`: Concatenate the latent code with embedded decoder input
  #      tokens.
  #   `decoder_add`: Add latent code to the embedded decoder input tokens.
  config.latents.blend_strategy = "decoder_concat"

  # Enable conditional layer normalization (CLN) to be used with latents.
  config.latents.conditional_layer_norm = False

  # Include Continuous Bernoulli normalizing factor in reconstruction loss. See
  # "The continuous Bernoulli: fixing a pervasive error in variational
  # autoencoders", Gabriel Loaiza-Ganem and John P. Cunningham, NeurIPS 2019.
  # https://arxiv.org/abs/1907.06845
  config.latents.continuous_bernoulli_log_norm_const = False

  # If greater than zero, applies uniform random noise to the input sketch token
  # embeddings in the latent encoder using the NEFTune algorithm during
  # training. See https://arxiv.org/abs/2310.05914.
  # Currently disabled. If enabled, the reasonable value of alpha seems to be 5,
  # also, the model training may need significantly more steps to converge
  # (e.g., 1M).
  config.latents.neftune_noise_alpha = 1.

  # ----------------------------------------------------------------------------
  # Decoder:
  # ----------------------------------------------------------------------------
  config.decoder = ml_collections.ConfigDict()

  # Decoding algorithm. Can be either beam search (`beam`) or temperature
  # sampling (`sampling`).
  config.decoder.algorithm = "sampling"

  # Number of steps to generate predictions. -1 will use the whole eval dataset.
  config.decoder.num_predict_steps = -1

  # Maximum length cutoff for predicted tokens.
  config.decoder.max_predict_length = 150

  # Beam size for inference if using beam search or number of hypotheses if
  # using stochastic sampling.
  config.decoder.num_hypotheses = 8

  # Brevity penalty for the decoder.
  config.decoder.brevity_alpha = 0.6

  # Temperature parameter for sampling. As it approaches zero this becomes
  # equivalent to greedy sampling.
  config.decoder.temperature = 1.0

  # Top K: if nonzero only use the top-k logits to sample next token, if
  # zero don't use any cutoff and sample from full logits over vocabulary.
  # Either `top_k` or `top_p` can be non-zero, otherwise exception will be
  # thrown.
  config.decoder.top_k = 0.

  # Top P (aka nucleus sampling): if nonzero only use the smallest number of
  # logits whose cumulative sum of probs adds up to (at least) `top_p`.
  config.decoder.top_p = 0.95

  # ----------------------------------------------------------------------------
  # Optimizer:
  # ----------------------------------------------------------------------------
  config.optimizer = ml_collections.ConfigDict()

  # Optimizer type: can be `adam` or `adafactor`.
  config.optimizer.name = "adafactor"

  # Base learning rate.
  config.optimizer.learning_rate = 0.001

  # Linear learning rate warmup.
  config.optimizer.warmup_steps = 1000

  # Decay factor for AdamW style weight decay.
  config.optimizer.weight_decay = 0.0

  # For variational models the following entry defines the weight annealing
  # strategy for KL-divergence. Can be one of: `linear`, `cosine`, `logistic`.
  # For cyclical annealing, `kl_num_cycles` defines the number of identical
  # annealing intervals to divide the total number of training steps into.
  # See https://arxiv.org/pdf/1903.10145.
  config.latents.kl_annealing = "cosine"
  config.latents.kl_cyclical = True
  config.latents.kl_num_cycles = 500
  config.latents.kl_multiplier = 0.01  # Fixed multiplier for KL loss.

  # ----------------------------------------------------------------------------
  # Protoscribe configuration.
  # ----------------------------------------------------------------------------
  config.protoscribe = ml_collections.ConfigDict()
  config.protoscribe.max_stroke_sequence_length = 150
  config.protoscribe.max_glyph_sequence_length = 20
  config.protoscribe.stroke_normalization_type = "sketch-rnn"
  config.protoscribe.stroke_random_scale_factor = 0.
  config.protoscribe.stroke_token_vocab_filename = (
      "vocab2048_normalized_sketchrnn.npy"
  )
  # Sketch token vocabulary size. This includes the actual number of tokens
  # supported by `stroke_token_vocab_filename` above, which is 2048 + <pad> +
  # <bos> + <eos> + <stroke-sep> + <end-of-numbers> = 2053, and possibly some
  # extra tokens for making the size of the vocabularity rounded to the
  # multiples of 16. Note, rounding may cause issues during the stochastic
  # sampling.
  config.protoscribe.vocab_size = 2064

  # NEFTune noisification.
  config.protoscribe.noisify_neftune_alphas = {
      "semantics": 0.01,
  }

  # Model features.
  # ---------------
  # This includes training, eval and inference features and their corresponding
  # lengths, where applicable.
  config.protoscribe.features = ml_collections.ConfigDict()

  # Features that are necessary for running training/eval and/or inference. Some
  # of these may not be used during inference (e.g., targets). Each dictionary
  # entry consists of a feature name and the corresponding maximum sequence
  # length (which may be -1 in which case the sequence dimension is left
  # unchanged).
  config.protoscribe.features.mandatory = {
      "inputs": (
          "text.concept_embedding", -1,
      ),
      "targets": (
          "sketch_tokens", config.protoscribe.max_stroke_sequence_length
      ),
      # ConfigDict does not accept dots in field names.
      "text/glyph/tokens": (
          "text.glyph.tokens", config.protoscribe.max_glyph_sequence_length
      ),
      "text/glyph/types": (
          "text.glyph.types", config.protoscribe.max_glyph_sequence_length
      ),
      "sketch/glyph_affiliations/ids": (
          "sketch.glyph_affiliations.ids",
          config.protoscribe.max_stroke_sequence_length
      ),
  }
  # Pass-through features we need to retain in the inference mode. These are
  # useful for inspecting the results.
  config.protoscribe.features.passthrough = [
      "doc.id",
      "concept.name",
      "number.name",
      "text.sampa",
      "text.words",
  ]
  return config


def metrics():
  return [
      "train_loss",
      "eval_loss",
      "eval_accuracy",
      "train_accuracy",
      "uptime",
      "steps_per_sec",
      "train_learning_rate",
  ]
