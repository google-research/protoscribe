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

"""SeqIO tasks for ProtoScribe dataset."""

import functools
import logging
import os
from typing import Callable

from absl import flags
import gin
import ml_collections
from protoscribe.corpus.reader import corpus_reader as parser_lib
from protoscribe.corpus.reader import dataset_defaults as ds_lib
from protoscribe.corpus.reader import metrics
from protoscribe.glyphs import glyph_vocab as glyph_lib
from protoscribe.sketches.utils import stroke_tokenizer as tokenizer_lib
from protoscribe.speech import audio_tokenizer
from protoscribe.speech import augmentation
import seqio
import tensorflow as tf

FLAGS = flags.FLAGS

CANONICAL_SPLITS = ["train", "validation", "test"]

# Prefix added to the name of the task registered with gin.
_TASK_NAME_PREFIX = "protoscribe"

# Embedding types.
EmbeddingType = parser_lib.EmbeddingType
gin.constant("tasks.EMBEDDING_SEMANTICS", EmbeddingType.SEMANTICS)
gin.constant("tasks.EMBEDDING_PHONETICS", EmbeddingType.PHONETICS)
gin.constant("tasks.EMBEDDING_VISION", EmbeddingType.VISION)
gin.constant("tasks.EMBEDDING_SPEECH", EmbeddingType.SPEECH)


def _target_features(
    config: ml_collections.FrozenConfigDict,
) -> seqio.Feature:
  """Returns target features depending on configuration."""

  # Targets are always either discrete glyphs or continuous strokes.
  stroke_tokenizer = "stroke_tokenizer" in config
  if config.glyph_recognition or config.glyph_only_targets or stroke_tokenizer:
    dtype = tf.int32
    rank = 1
  else:
    dtype = tf.float32
    rank = 2

  targets_eos_id = None
  if config.glyph_recognition or config.glyph_only_targets:
    targets_eos_id = glyph_lib.GLYPH_EOS
  elif stroke_tokenizer:
    targets_eos_id = tokenizer_lib.Token.END_OF_SKETCH

  return seqio.Feature(
      # The token vocabulary size is set in the config files as
      # `SKETCH_TOKEN_VOCAB_SIZE` for tokenized sketches and
      # `GLYPH_VOCAB_SIZE` for glyph-only prediction.
      vocabulary=seqio.PassThroughVocabulary(
          size=0,
          eos_id=targets_eos_id,
      ),
      add_eos=False,
      dtype=dtype,
      required=True,
      rank=rank,
  )


def _output_features_for_synthesis(
    config: ml_collections.FrozenConfigDict
) -> dict[str, seqio.Feature]:
  """Adds generation-specific features (embeddings -> sketch tokens).

  Args:
    config: Configuration dictionary.

  Returns:
    Dictionary of the features to be used by the model. These only include the
    features that affect the model graph. There are more features that we use,
    but these are mostly passed-through the model.
  """
  features = {
      "text.concept_embedding": seqio.ContinuousFeature(
          dtype=tf.float32, required=True, rank=2
      ),
      "text.concept_embedding.original": seqio.ContinuousFeature(
          dtype=tf.float32, required=True, rank=2  # Before noise.
      ),
      "text.phonetic_embedding": seqio.ContinuousFeature(
          dtype=tf.float32, required=True, rank=2
      ),
      "text.phonetic_embedding.original": seqio.ContinuousFeature(
          dtype=tf.float32, required=True, rank=2  # Before noise.
      ),
      "text.glyph.tokens": seqio.Feature(
          vocabulary=seqio.PassThroughVocabulary(
              size=0, eos_id=glyph_lib.GLYPH_EOS
          ),
          add_eos=False,
          dtype=tf.int32,
          required=True,
          rank=1,
      ),
      "text.glyph.types": seqio.Feature(
          vocabulary=seqio.PassThroughVocabulary(size=0),
          add_eos=False,
          dtype=tf.int32,
          required=True,
          rank=1,
      ),
      # Inputs are ignored by the encoder. These are only created in order to
      # specify the input vocabulary. This is done to placate the following
      # error inside t5x: "inputs" and "targets" are not both present, and
      # vocabularies are different. TODO: Find a fix.
      "inputs": seqio.Feature(
          vocabulary=seqio.PassThroughVocabulary(size=0),
          add_eos=False,
          dtype=tf.string,
          required=True,
          rank=0,
      ),
  }

  # Fill in speech input features.
  if config.get("speech_tokenizer_name_or_path", False):
    speech_spec = {
        "speech.embeddings": seqio.ContinuousFeature(
            dtype=tf.float32, required=True, rank=2
        ),
    }
  else:
    speech_spec = {
        "speech.log_mel_spectrum": seqio.ContinuousFeature(
            dtype=tf.float32, required=True, rank=2
        ),
        "speech.spectrum_patches": seqio.ContinuousFeature(
            dtype=tf.float32, required=True, rank=2
        ),
    }
  features.update(speech_spec)

  return features


def _output_features_for_recognition(
    config: ml_collections.FrozenConfigDict
) -> dict[str, seqio.Feature]:
  """Adds recognition-specific features (sketch tokens -> glyph IDs).

  Args:
    config: Task configuration.

  Returns:
    Dictionary of the features to be used by the model. These only include the
    features that affect the model graph. There are more features that we use,
    but these are mostly passed-through the model.
  """
  if "stroke_tokenizer" not in config:
    raise ValueError("Recognition from continuous strokes not supported")

  # This setup is rather different from the synthesis setup. Because during
  # generation we are not running inference, we seem to be able to get away
  # with exposing all the Protoscribe features to PMMX. In recognition mode,
  # we can only expose the features we actually use in the model.
  # TODO: Investigate this further.
  features = {
      "sketch_tokens": seqio.Feature(
          vocabulary=seqio.PassThroughVocabulary(
              size=0,
              eos_id=tokenizer_lib.Token.END_OF_SKETCH),
          add_eos=False,
          dtype=tf.int32,
          required=True,
          rank=1),
  }
  # Placate the following error inside seqio: "inputs" and "targets" are not
  # both present, and vocabularies are different.
  # TODO: Find a fix.
  features["inputs"] = features["sketch_tokens"]
  return features


def _output_features(
    config: ml_collections.FrozenConfigDict,
) -> dict[str, seqio.Feature]:
  """Returns dictionary of feautures available to the model."""

  if config.glyph_recognition:
    features = _output_features_for_recognition(config)
  else:
    features = _output_features_for_synthesis(config)
  features.update({
      "targets": _target_features(config),
  })
  return features


def _get_file_patterns_by_split(
    dataset_dir: str, dataset_format: str
) -> dict[str, str]:
  """Returns file patterns for each dataset split."""
  return {
      split: os.path.join(dataset_dir, f"{split}.{dataset_format}-*-of-*")
      for split in CANONICAL_SPLITS
  }


def _get_reader_cls(dataset_format: str) -> Callable[..., tf.data.Dataset]:
  """Determines particular dataset reader class for the given file(s)."""
  if dataset_format == "tfr":
    return tf.data.TFRecordDataset
  else:
    raise ValueError(
        f"Unsupported reader type: {dataset_format}"
    )


def _inputs_and_targets_for_synthesis(
    config: ml_collections.FrozenConfigDict,
    example: dict[str, tf.Tensor],
) -> dict[str, tf.Tensor]:
  """Inputs/outputs for sketch generation."""
  # The inputs are not used by the multimodal encoder. These are for propagating
  # to the decoder at inference time.
  example["inputs"] = tf.strings.join(
      [example["number.name"], example["concept.name"]], separator=" "
  )

  # The targets are eigher discrete sketch tokens, continuous strokes or simply
  # discrete glyphs.
  stroke_tokenizer = "stroke_tokenizer" in config
  if config.glyph_only_targets:
    targets = example["text.glyph.tokens"]  # Discrete glyphs.
  elif stroke_tokenizer:
    # Glyphs combined with sketch tokens or just the sketch tokens.
    if config.stroke_combine_with_glyphs:
      targets = example["sketch.glyphs_and_strokes"]
    else:
      targets = example["sketch_tokens"]
  else:
    targets = example["strokes"]  # Continuous strokes.
  example["targets"] = targets
  return example


def _inputs_and_targets_for_recognition(
    config: ml_collections.FrozenConfigDict,
    example: dict[str, tf.Tensor],
) -> dict[str, tf.Tensor]:
  """Inputs/outputs for sketch generation."""
  stroke_tokenizer = "stroke_tokenizer" in config
  if stroke_tokenizer:
    example["inputs"] = example["sketch_tokens"]
  else:
    example["inputs"] = example["strokes"]
  example["targets"] = example["text.glyph.tokens"]
  return example


def _preprocess_tf_example(
    features: parser_lib.Features,
    config: ml_collections.FrozenConfigDict,
    sketch_stroke_stats: ds_lib.StrokeStats,
    stroke_tokenizer: tokenizer_lib.StrokeTokenizer | None,
    speech_tokenizer: audio_tokenizer.AudioTokenizer | None,
    is_training: bool,
) -> dict[str, tf.Tensor]:
  """Preprocesses a single example.

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
  example = parser_lib.parse_features(
      features,
      config,
      sketch_stroke_stats,
      stroke_tokenizer=stroke_tokenizer,
      speech_tokenizer=speech_tokenizer,
      is_training=is_training
  )
  # t5x/gin are confused by "/" in feature names. Replacing by something
  # more palatable.
  for key in list(example.keys()):
    if "/" in key:
      new_key = key.replace("/", ".")
      example[new_key] = example.pop(key)

  # Protoscribe dataset does not define input and target features explicitly.
  # Set them here.
  if config.glyph_recognition:
    example = _inputs_and_targets_for_recognition(config, example)
  else:
    example = _inputs_and_targets_for_synthesis(config, example)
  return example


def _dataset_preprocessor(
    dataset: tf.data.Dataset,
    config: ml_collections.FrozenConfigDict,
    sketch_stroke_stats: ds_lib.StrokeStats,
    stroke_tokenizer: ds_lib.StrokeTokenizer | None,
    speech_tokenizer: audio_tokenizer.AudioTokenizer | None,
    is_training: bool,
) -> tf.data.Dataset:
  """Protoscribe preprocessor.

  Args:
    dataset: the input dataset that all the necessary features parsed
      from tf.Example.
    config: Configuration dictionary from `ml_collections`.
    sketch_stroke_stats: Sketch sufficient stats.
    stroke_tokenizer: Stroke tokenizer for sketches. Will be None for
      configurations that don't predict sketch tokens.
    speech_tokenizer: Tokenizer for audio. Will be None for configurations
      no requiring audio tokenization.
    is_training: Training or eval/test mode.

  Returns:
    A dataset with parsed features (with some extra new ones).
  """
  return dataset.map(
      functools.partial(
          _preprocess_tf_example,
          config=config,
          sketch_stroke_stats=sketch_stroke_stats,
          stroke_tokenizer=stroke_tokenizer,
          speech_tokenizer=speech_tokenizer,
          is_training=is_training),
      num_parallel_calls=tf.data.AUTOTUNE,
  )


# Notes:
# ------------------------------------------------------------------------------
#   - By default the stroke random scaling (`stroke_random_scale_factor`) is
# disabled (set to 0.0). This is the default for the configurations where the
# sketch tokens are the targets. This default results in smooth training curves.
# For other configurations (such as annotation mode), where the strokes are on
# the input side, enabling random scaling does not hurt the performance.
#   - The NEFTune alpha parameters needs to be embedding-specific since it
# essentially controls the amplitude of the added noise. For now we just have it
# depend on the modality.
#   - Manual sequence padding is disabled for SeqIO tasks because padding is
# performed by SeqIO itself.
@gin.configurable
def register(
    task_name: str = gin.REQUIRED,
    dataset_dir: str | None = None,
    concept_embedding_type: str = "bnc",
    glyph_recognition: bool = False,
    glyph_only_targets: bool = False,
    manual_padding: bool = False,
    max_stroke_sequence_length: int = 250,
    max_glyph_sequence_length: int = 20,
    max_phonetic_sequence_length: int = 10,
    max_speech_frame_sequence_length: int = 120,
    noisify_embeddings: bool = False,
    noisify_order: str = "INF",  # Draw from uniform distribution.
    noisify_neftune_alphas: dict[str, float] | None = None,
    stroke_normalization_type: str = "none",
    stroke_token_vocab_filename: str | None = None,
    stroke_random_scale_factor: float = 0.0,
    stroke_combine_with_glyphs: bool = False,
    vision_combine_type: str = "average",
    speech_corpus_sample_rate: int = 24_000,  # Sample rate of original audio.
    speech_frame_length_ms: float = 50.0,
    speech_frame_step_ms: float = 12.5,
    speech_num_mel_channels: int = 128,
    speech_frame_normalization: str = "none",
    speech_spectrum_patch_size: int = 16,
    speech_spectrum_patch_overlap: int = 6,
    speech_spectrum_augmentation: bool = False,
    speech_framework_type: str = "dmvr",
    speech_normalize_waveform: bool = False,
    speech_keep_waveform: bool = False,
    speech_tokenizer_name_or_path: str | None = None,
    speech_normalize_embeddings: bool = False,
    is_training: bool = True,
) -> str:
  """Registers task from gin scaffolding."""

  config = {
      "concept_embedding_type": concept_embedding_type,
      "glyph_only_targets": glyph_only_targets,
      "glyph_recognition": glyph_recognition,
      "manual_padding": manual_padding,
      "max_stroke_sequence_length": max_stroke_sequence_length,
      "max_glyph_sequence_length": max_glyph_sequence_length,
      "max_phonetic_sequence_length": max_phonetic_sequence_length,
      "max_speech_frame_sequence_length": max_speech_frame_sequence_length,
      "noisify_embeddings": noisify_embeddings,
      "noisify_order": noisify_order,
      "noisify_neftune_alphas": noisify_neftune_alphas,
      "stroke_normalization_type": stroke_normalization_type,
      "stroke_random_scale_factor": stroke_random_scale_factor,
      "stroke_combine_with_glyphs": stroke_combine_with_glyphs,
      "stroke_token_vocab_filename": stroke_token_vocab_filename,
      "vision_combine_type": vision_combine_type,
      "speech_corpus_sample_rate": speech_corpus_sample_rate,
      "speech_frame_length_ms": speech_frame_length_ms,
      "speech_frame_step_ms": speech_frame_step_ms,
      "speech_num_mel_channels": speech_num_mel_channels,
      "speech_frame_normalization": speech_frame_normalization,
      "speech_spectrum_patch_size": speech_spectrum_patch_size,
      "speech_spectrum_patch_overlap": speech_spectrum_patch_overlap,
      "speech_spectrum_augmentation": speech_spectrum_augmentation,
      "speech_framework_type": speech_framework_type,
      "speech_normalize_waveform": speech_normalize_waveform,
      "speech_keep_waveform": speech_keep_waveform,
      "speech_tokenizer_name_or_path": speech_tokenizer_name_or_path,
  }
  if stroke_token_vocab_filename:
    config.update({
        "stroke_tokenizer": {
            "vocab_filename": stroke_token_vocab_filename,
        },
    })
  config = ml_collections.FrozenConfigDict(config)
  logging.info("[%s] Task configuration: %s", task_name, config)

  if not dataset_dir:
    dataset_dir = ds_lib.DATASET_DIR.value
    logging.info("Using value of `dataset_dir` flag: %s", dataset_dir)
  else:
    # This seems to be the only sane way to override `ds_lib.DATASET_DIR`.
    logging.info("Setting dataset dir to `%s`", dataset_dir)
    FLAGS.dataset_dir = dataset_dir

  sketch_stroke_stats = ds_lib.get_sketch_stroke_stats(config)
  stroke_tokenizer = ds_lib.get_stroke_tokenizer(config)

  if is_training and speech_spectrum_augmentation:
    # Initialize spectrum augmentation.
    augmentation.tf_spec_augment_init()

  speech_tokenizer = None
  if speech_tokenizer_name_or_path:
    speech_tokenizer = audio_tokenizer.get_tokenizer(
        model_config_name_or_path=speech_tokenizer_name_or_path,
        sample_rate=speech_corpus_sample_rate,
        normalize_embeddings=speech_normalize_embeddings,
    )

  task_name = f"{_TASK_NAME_PREFIX}_{task_name}"
  seqio.TaskRegistry.remove(task_name)
  seqio.TaskRegistry.add(
      task_name,
      source=seqio.TFExampleDataSource(
          split_to_filepattern=_get_file_patterns_by_split(
              dataset_dir=ds_lib.DATASET_DIR.value,
              dataset_format=ds_lib.DATASET_FORMAT.value
          ),
          reader_cls=_get_reader_cls(
              dataset_format=ds_lib.DATASET_FORMAT.value
          ),
          num_input_examples=None,
          feature_description=parser_lib.feature_specification(config)
      ),
      preprocessors=[
          functools.partial(
              _dataset_preprocessor,
              config=config,
              sketch_stroke_stats=sketch_stroke_stats,
              stroke_tokenizer=stroke_tokenizer,
              speech_tokenizer=speech_tokenizer,
              is_training=is_training,
          ),
          seqio.CacheDatasetPlaceholder(),
      ],
      output_features=_output_features(config),
      postprocess_fn=None,
      metric_fns=(  # pylint: disable=g-long-ternary
          [
              metrics.wer,
              metrics.sequence_accuracy,
          ] if config.glyph_recognition else []
      ),
  )
  logging.info("Registering task `%s` ...", task_name)
  return task_name
