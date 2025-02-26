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

"""Reader for ProtoScribe dataset."""

from typing import Any

import ml_collections
from protoscribe.corpus.reader import dataset_defaults as ds_lib
from protoscribe.corpus.reader import noise_utils
from protoscribe.corpus.reader import stroke_parser as stroke_parser_lib
from protoscribe.language.embeddings import embedder as concept_embed_lib
from protoscribe.speech import audio_tokenizer
from protoscribe.speech import augmentation as speech_augment_lib
from protoscribe.speech import log_mel_spectrum as log_mel_lib
import tensorflow as tf

Features = dict[str, tf.Tensor | tf.SparseTensor]
StrokeStats = ds_lib.StrokeStats
StrokeTokenizer = ds_lib.StrokeTokenizer
AudioTokenizer = audio_tokenizer.AudioTokenizer

# When glyph and stroke tokens are combined in a single sequence, we use the
# stroke token vocabulary size as an affset.
STROKE_OFFSET_FOR_GLYPH_IDS = 2064

# Names of the vision features and the corresponding feature dimensions.
# The features have dimensions (N, D), where N is the number of images and
# D is the dimension of the feature.
VISION_FEATURE_NAMES = {
    # This list may be empty until we figure out which vision features are
    # the most optimal for this task.
}


# Keys to be used for various embedding types below.
class EmbeddingType:
  SEMANTICS = "semantics"
  PHONETICS = "phonetics"
  VISION = "vision"
  SPEECH = "speech"


def _build_text(number: tf.Tensor, concept_or_text: tf.Tensor,
                separator: str = " ") -> tf.Tensor:
  """Returns full text corresponding to number and concept."""
  concept_or_text = tf.strings.split(concept_or_text, "_")[0]  # Remove POS.
  return tf.strings.join([number, separator, concept_or_text])


def _maybe_noisify(
    config: ml_collections.FrozenConfigDict,
    embedding_type: str,
    embeddings: tf.Tensor,
    is_training: bool,
) -> tf.Tensor:
  """Noisifies the embedding tensors.

  The noise is randomly sampled from either a uniform distribution (when
  `noisify_order` is set to `INF`) or from the surface of a ball with a unit
  radius subject to lₚ norm (which is defined by `noisify_order` of `1` or
  `2`).

  Noisification amplitude is broadly based on NEFTune algorithm:
  Neel Jain, et al. (2023): `NEFTune: Noisy embeddings improve instruction
    finetuning.`, ICLR. https://arxiv.org/pdf/2310.05914.pdf

  Args:
    config: Configuration dictionary.
    embedding_type: Embedding type or modality.
    embeddings: Input tensor of dimension (L, D).
    is_training: Whether training or evaluating/testing.

  Returns:
    A (possibly) noisified tensor.
  """
  enabled = config.get("noisify_embeddings", False)
  if not enabled or not is_training:
    return embeddings

  if ("noisify_neftune_alphas" not in config or
      embedding_type not in config.noisify_neftune_alphas):
    return embeddings

  norm_order = config.get("noisify_order", "INF")
  alpha = config.noisify_neftune_alphas[embedding_type]
  shape = tf.shape(embeddings)
  epsilon = noise_utils.tf_random_lp_ball_vector(
      shape, order=norm_order, radius=1.0, dtype=tf.float32
  )
  l_by_d = tf.cast(shape[0] * shape[1], dtype=tf.float32)
  scale = alpha / tf.math.sqrt(l_by_d)
  return embeddings + scale * epsilon


def _preprocess_features(features: Features) -> Features:
  """Basic preprocessing of the features.

  This involves replacing sparse tensors in a dictionary of tensors with
  dense ones and using a smaller integer types.

  Args:
    features: Actual features.
  Returns:
    Massaged features.
  """
  for k in features:
    if isinstance(features[k], tf.SparseTensor):
      features[k] = tf.sparse.to_dense(features[k])
    if features[k].dtype == tf.int64:
      features[k] = tf.cast(features[k], dtype=tf.int32)
  return features


def _parse_discrete_glyphs(
    features: dict[str, tf.Tensor],
    config: ml_collections.FrozenConfigDict
) -> tuple[tf.Tensor, tf.Tensor]:
  """Parses discrete glyph sequences and their types."""
  glyph_tokens = features["text/glyph/tokens"]
  glyph_types = features["text/glyph/types"]
  pad_size = config.max_glyph_sequence_length - tf.shape(glyph_tokens)[0]
  if config.manual_padding and pad_size > 0:
    paddings = [[0, pad_size]]
    glyph_tokens = tf.pad(glyph_tokens, paddings)
    glyph_types = tf.pad(glyph_types, paddings)
  return glyph_tokens, glyph_types


def _parse_concept_embeddings(
    features: dict[str, tf.Tensor],
    config: ml_collections.FrozenConfigDict,
    is_training: bool,
) -> tuple[tf.Tensor, tf.Tensor, dict[str, tf.Tensor]]:
  """Parses concept semantic embeddings into combined and individual tensors."""
  concept_embedding_type = config.concept_embedding_type
  concept_embedding_names = [
      f"text/{concept_embedding_type}/number_emb",
      f"text/{concept_embedding_type}/concept_emb"
  ]
  # Combined tensor: (2, D).
  combined_embeddings = tf.stack(
      [features[name] for name in concept_embedding_names], axis=0
  )
  original_combined_embeddings = combined_embeddings
  combined_embeddings = _maybe_noisify(
      config, EmbeddingType.SEMANTICS, combined_embeddings, is_training
  )

  # Individual concept embedding tensors with dimensions (1, D) before and
  # after applying noise.
  individual_embeddings = {
      name: _maybe_noisify(
          config, EmbeddingType.SEMANTICS,
          tf.expand_dims(features[name], axis=0),
          is_training
      ) for name in concept_embedding_names
  }
  individual_embeddings.update({
      f"{name}/original": tf.expand_dims(
          features[name], axis=0
      ) for name in concept_embedding_names
  })
  return (
      combined_embeddings, original_combined_embeddings, individual_embeddings
  )


def _parse_phonetic_embeddings(
    features: dict[str, tf.Tensor],
    feature_name: str,
    config: ml_collections.FrozenConfigDict,
    is_training: bool
) -> tuple[tf.Tensor, tf.Tensor]:
  """Parses and pads 2D phonetic embeddings."""
  embeddings = tf.cast(features[feature_name], dtype=tf.float32)
  embed_dim = concept_embed_lib.DEFAULT_EMBEDDING_DIM  # Outer dimension.
  num_phones = int(tf.shape(embeddings)[0] / embed_dim)
  embeddings = tf.reshape(embeddings, (num_phones, embed_dim))
  original_embeddings = embeddings
  embeddings = _maybe_noisify(
      config, EmbeddingType.PHONETICS, embeddings, is_training
  )
  pad_size = config.max_phonetic_sequence_length - num_phones

  def _pad(emb: tf.Tensor) -> tf.Tensor:
    """Pads the inputs tensor."""
    if not config.manual_padding:
      return emb
    if pad_size > 0:
      paddings = [[0, pad_size], [0, 0]]
      emb = tf.pad(emb, paddings)
    elif pad_size < 0:
      emb = emb[:config.max_phonetic_sequence_length, :]
    return emb

  return _pad(embeddings), _pad(original_embeddings)


def _parse_vision_feature(
    config: ml_collections.FrozenConfigDict,
    features: dict[str, tf.Tensor],
    feature_name: str,
    is_training: bool
) -> tf.Tensor:
  """Parses 2D vision feature with a given name."""
  embeddings = features[f"{feature_name}/values"]

  # The `embeddings` array is a (N * D,) encoding of the original shape (N, D),
  # where N is the number of sampled images and D is the original embedding
  # dimension for each image.
  #
  # For each feature, we have three ways of representing it:
  #   - "average": average pooling across the batch, (N, D) -> (D).
  #   - "concat": concatenation of all the N samples along outer (feature)
  #     dimension: (N, D) -> (N * D).
  #   - "seq_concat": concatenation of all N samples along sequence dimension:
  #     identity mapping (N, D) -> (N, D).
  #   - "sample": uniform sampling from N samples, (N, D) -> (D).
  vision_combine_type = config.get("vision_combine_type", "average")
  if vision_combine_type == "concat":
    embeddings = tf.expand_dims(tf.cast(embeddings, tf.float32), axis=0)
    embeddings = tf.linalg.l2_normalize(embeddings, axis=1)
    return _maybe_noisify(config, EmbeddingType.VISION, embeddings, is_training)

  dims = VISION_FEATURE_NAMES[feature_name]
  embeddings = tf.reshape(embeddings, dims)
  if vision_combine_type == "average":
    embeddings = tf.math.reduce_mean(embeddings, axis=0)
    embeddings = tf.expand_dims(tf.cast(embeddings, tf.float32), axis=0)
  elif vision_combine_type == "sample":
    batch_size = tf.shape(embeddings)[0]
    idx = tf.random.uniform(
        shape=[], minval=0, maxval=batch_size, dtype=tf.int32
    )
    embeddings = embeddings[idx, :]
    embeddings = tf.expand_dims(tf.cast(embeddings, tf.float32), axis=0)
  elif vision_combine_type != "seq_concat":
    raise ValueError(
        f"Unsupported vision feature combination type: {vision_combine_type}"
    )

  # Returns a "sequence" with a unit length.
  return _maybe_noisify(config, EmbeddingType.VISION, embeddings, is_training)


def _speech_log_mel_spectrum(
    config: ml_collections.FrozenConfigDict,
    features: dict[str, tf.Tensor],
    is_training: bool
) -> dict[str, tf.Tensor]:
  """Parses various speech features."""

  # Process log mel-spectrum.
  log_mel_vals = log_mel_lib.log_mel_spectrogram(
      backend=config.speech_framework_type,
      samples=features["audio/waveform"],
      sample_rate=config.speech_corpus_sample_rate,
      normalize_waveform=config.speech_normalize_waveform,
      frame_length_ms=config.speech_frame_length_ms,
      frame_step_ms=config.speech_frame_step_ms,
      num_mel_channels=config.speech_num_mel_channels
  )

  # Perform normalization: either utterance-level cepstral mean and variance
  # normalization (CMVN), global using precomputed frame mean and variance or
  # none.
  frame_norm = config.get("speech_frame_normalization", "utterance")
  if frame_norm == "utterance":
    log_mel_vals = log_mel_lib.z_normalize_spectrogram(log_mel_vals)
  elif frame_norm == "global":
    log_mel_vals = log_mel_lib.normalize_spectrogram(
        log_mel_vals, log_domain=True, clip=False
    )

  # Augment the training features (if spectrum augmentation is configured
  # or simply noisify) and pad.
  spec_augment = config.get("speech_spectrum_augmentation", False)
  original_log_mel_vals = log_mel_vals
  if is_training:
    if spec_augment:
      log_mel_vals = speech_augment_lib.tf_spec_augment(log_mel_vals)
    else:
      log_mel_vals = _maybe_noisify(
          config, EmbeddingType.SPEECH, log_mel_vals, is_training
      )
  pad_size = (
      config.max_speech_frame_sequence_length - tf.shape(log_mel_vals)[0]
  )
  if config.manual_padding and pad_size > 0:
    paddings = [[0, pad_size], [0, 0]]
    log_mel_vals = tf.pad(log_mel_vals, paddings)
    original_log_mel_vals = tf.pad(original_log_mel_vals, paddings)

  # Extract audio patches from the spectrogram first converting it to a
  # (B, H, W, C) image. If spectrum augmentation is disabled, patches are
  # extracted from the original spectrum and then noisified.
  patch_size = config.get("speech_spectrum_patch_size", 16)
  patch_overlap = config.get("speech_spectrum_patch_overlap", 6)
  stride = patch_size - patch_overlap
  log_mel = log_mel_vals if spec_augment else original_log_mel_vals
  image = tf.expand_dims(tf.expand_dims(log_mel, axis=-1), axis=0)
  patches = tf.image.extract_patches(
      image,
      [1, patch_size, patch_size, 1],
      [1, stride, stride, 1],
      [1, 1, 1, 1],
      padding="VALID",
  )
  patches = tf.reshape(patches, [-1, patch_size ** 2])
  if not spec_augment:
    patches = _maybe_noisify(
        config, EmbeddingType.SPEECH, patches, is_training
    )

  speech_features = {
      "speech/log_mel_spectrum": log_mel_vals,
      "speech/spectrum_patches": patches,
  }
  if config.speech_keep_waveform:  # Useful by debugging.
    speech_features.update({
        "audio/sample_rate": features["audio/sample_rate"],
        "audio/waveform": features["audio/waveform"]
    })
  return speech_features


def _speech_tokens_or_embeddings_features(
    config: ml_collections.FrozenConfigDict,
    features: dict[str, tf.Tensor],
    speech_tokenizer: AudioTokenizer,
    is_training: bool
) -> dict[str, tf.Tensor]:
  """Generates token embeddings or discrete tokens for audio."""

  embeddings = speech_tokenizer.get_embeddings(audio=features["audio/waveform"])
  embeddings = _maybe_noisify(
      config, EmbeddingType.SPEECH, embeddings, is_training
  )
  speech_features = {
      "speech/embeddings": embeddings,
  }
  if config.speech_keep_waveform:  # Useful by debugging.
    speech_features.update({
        "audio/sample_rate": features["audio/sample_rate"],
        "audio/waveform": features["audio/waveform"]
    })
  return speech_features


def feature_specification(
    config: ml_collections.FrozenConfigDict
) -> dict[str, tf.io.FixedLenFeature | tf.io.VarLenFeature]:
  """Builds feature specification for parsing from TF examples."""
  features = {
      "audio/sample_rate": tf.io.FixedLenFeature([], tf.int64),
      "audio/waveform": tf.io.VarLenFeature(tf.float32),
      "concept/id": tf.io.FixedLenFeature([], tf.int64),
      "concept/name": tf.io.FixedLenFeature([], tf.string),
      "concept/unseen": tf.io.FixedLenFeature([], tf.int64),
      "doc/id": tf.io.FixedLenFeature([], tf.int64),
      "number/name": tf.io.FixedLenFeature([], tf.string),
      "strokes/glyph_affiliations/ids": tf.io.VarLenFeature(tf.int64),
      "strokes/x_stroke_points": tf.io.VarLenFeature(tf.float32),
      "strokes/y_stroke_points": tf.io.VarLenFeature(tf.float32),
      "text/glyph/tokens": tf.io.VarLenFeature(tf.int64),
      "text/glyph/types": tf.io.VarLenFeature(tf.int64),
      "text/phonetic_embedding/emb": tf.io.VarLenFeature(tf.float32),
      "text/sampa": tf.io.FixedLenFeature([], tf.string),
      "text/words": tf.io.FixedLenFeature([], tf.string),
  }

  # Concept and number embeddings.
  concept_embedding_type = config.get("concept_embedding_type")
  if not concept_embedding_type:
    raise ValueError("Expecting `concept_embedding_type` to be set to one of "
                     f"{concept_embed_lib.EMBEDDING_TYPES}")
  emb_dim = concept_embed_lib.embedding_dim_from_type(concept_embedding_type)
  features.update({
      f"text/{concept_embedding_type}/number_emb": tf.io.FixedLenFeature(
          [emb_dim], dtype=tf.float32),
      f"text/{concept_embedding_type}/concept_emb": tf.io.FixedLenFeature(
          [emb_dim], dtype=tf.float32),
  })

  # Concept gloss embeddings.
  if "t5" in config:
    features.update({
        "text/gloss": tf.io.FixedLenFeature([], tf.string),
    })

  return features


def parse_features(
    features: Features,
    config: ml_collections.FrozenConfigDict,
    sketch_stroke_stats: StrokeStats,
    stroke_tokenizer: StrokeTokenizer | None,
    speech_tokenizer: AudioTokenizer | None,
    is_training: bool
) -> dict[str, tf.Tensor]:
  """Parses the tensors read from tf.Example proto.

  Args:
    features: A dictionary of features.
    config: Configuration dictionary from `ml_collections`.
    sketch_stroke_stats: Sketch sufficient stats.
    stroke_tokenizer: Stroke tokenizer for sketches. Will be None for
      configurations that don't predict sketch tokens.
    speech_tokenizer: Tokenizer for audio. Will be None for configurations
      no requiring audio tokenization.
    is_training: Training or eval/test mode.

  Returns:
    All the feature tensors for a single document.
  """
  features = _preprocess_features(features)

  combined_concepts, original_combined_concepts, individual_concepts = (
      _parse_concept_embeddings(features, config, is_training)
  )
  phonetics, original_phonetics = _parse_phonetic_embeddings(
      features, "text/phonetic_embedding/emb", config, is_training
  )

  (
      strokes_or_tokens,
      sketch_glyph_affiliations,
      lengths
  ) = stroke_parser_lib.parse_sketch_strokes_or_tokens(
      config, features, sketch_stroke_stats, stroke_tokenizer, is_training
  )
  sketch_contents = "sketch_tokens" if stroke_tokenizer else "strokes"
  glyph_tokens, glyph_types = _parse_discrete_glyphs(features, config)

  parsed = {
      "concept/id": features["concept/id"],
      "concept/name": features["concept/name"],
      "doc/id": features["doc/id"],
      "number/name": features["number/name"],
      "text/sampa": features["text/sampa"],
      "text/words": features["text/words"],
      "text/concept_embedding": combined_concepts,
      "text/concept_embedding/original": original_combined_concepts,
      "text/phonetic_embedding": phonetics,
      "text/phonetic_embedding/original": original_phonetics,
      "text/glyph/tokens": glyph_tokens,
      "text/glyph/types": glyph_types,
      sketch_contents: strokes_or_tokens,
      "sketch/glyph_affiliations/ids": sketch_glyph_affiliations,
      "lengths": lengths,
      **individual_concepts,
  }
  if "concept/unseen" in features:
    # Each concept is marked either as `seen` or `unseen` (`new`). For `new`
    # concepts we don't know how to write glyphs during the training, these are
    # represented by the dummy glyph. The synthesis losses for the `new` glyph
    # are masked out during training.
    parsed["concept/unseen"] = features["concept/unseen"]
  else:  # No unseen concepts.
    parsed["concept/unseen"] = tf.constant(0, dtype=tf.int32)

  if config.get("retain_text", True):
    # We want to retain the original text for the sampler and for the testing.
    text = _build_text(features["number/name"], features["concept/name"])
    parsed["text/text"] = text

  # Combine glyph tokens with sketch tokens in a single sequence.
  if config.stroke_combine_with_glyphs and stroke_tokenizer:
    unpadded_glyph_tokens = (
        features["text/glyph/tokens"] + STROKE_OFFSET_FOR_GLYPH_IDS
    )
    unpadded_glyph_tokens = unpadded_glyph_tokens[1:-1]  # Exclude BOS/EOS.
    glyph_and_stroke_tokens = tf.concat(
        [unpadded_glyph_tokens, strokes_or_tokens], axis=0
    )
    parsed["sketch/glyphs_and_strokes"] = glyph_and_stroke_tokens

  # Make speech features. We either perform audio tokenization or generate
  # classical log-mel spectral features.
  if speech_tokenizer:
    speech_features = _speech_tokens_or_embeddings_features(
        config, features, speech_tokenizer, is_training
    )
  else:
    speech_features = _speech_log_mel_spectrum(
        config, features, is_training
    )
  parsed.update(speech_features)

  return parsed  # pytype: disable=bad-return-type


def parse_example(
    example: Any,
    config: ml_collections.FrozenConfigDict,
    sketch_stroke_stats: StrokeStats,
    stroke_tokenizer: StrokeTokenizer | None,
    speech_tokenizer: AudioTokenizer | None,
    is_training: bool
) -> dict[str, tf.Tensor]:
  """Returns a dictionary with the decoded features. Used by the test."""

  features = tf.io.parse_single_example(
      example, features=feature_specification(config)
  )
  return parse_features(
      features=features,
      config=config,
      sketch_stroke_stats=sketch_stroke_stats,
      stroke_tokenizer=stroke_tokenizer,
      speech_tokenizer=speech_tokenizer,
      is_training=is_training
  )
