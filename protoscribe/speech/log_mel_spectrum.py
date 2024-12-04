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

"""Utility for computing log-mel spectrograms."""

from protoscribe.speech import log_mel_spectrum_dmvr as dmvr_lib
from protoscribe.speech import resampler_dmvr as resample_lib
import tensorflow as tf

# Frame duration in milliseconds.
_FRAME_LENGTH_MS = 50.0

# Frame step in milliseconds.
_FRAME_STEP_MS = 12.5

# Number of mel filterbanks.
_NUM_MEL_CHANNELS = 128

# Waveform pre-emphasis filter coefficient.
_PRE_EMPHASIS = 0.97

# Lower bound on frequencies to be included in a mel filterbank.
_MEL_LOWER_EDGE_HERTZ = 125.0

# Upper bound on frequencies to be included in a mel filterbank.
_MEL_UPPER_EDGE_HERTZ = 7600.0

# Sample rate for computing the log-mel spectra. We currently keep this fixed.
_FIXED_SAMPLE_RATE = 16_000


def z_normalize_spectrogram(spectrogram: tf.Tensor) -> tf.Tensor:
  """Normalizes spectrogram to zero mean and unity std.

  Args:
    spectrogram: Tensor with shape [n_time_bins, n_freq_bins].

  Returns:
    Normalized spectrogram.
  """
  spectrogram = tf.expand_dims(spectrogram, axis=-1)
  spectrogram = tf.image.per_image_standardization(spectrogram)
  spectrogram = tf.squeeze(spectrogram, axis=-1)
  return spectrogram


# Empirically established values so that spectrum logs lie between
# -1 and 1 after rescaling.
_EPSILON = 1e-4
_MEAN_LOG = 1.6
_STD_LOG = 10.9


def normalize_spectrogram(
    spectrogram: tf.Tensor, log_domain: bool = True, clip: bool = False
) -> tf.Tensor:
  """Rescales the spectrogram using empirical mean and std.

  Args:
    spectrogram: Tensor with shape [n_time_bins, n_freq_bins].
    log_domain: If true, the spectrogram is in log-domain.
    clip: Clip values to [-1.0, 1.0].

  Returns:
    Normalized spectrogram in [-1.0, 1.0].
  """
  if not log_domain:
    spectrogram = tf.math.log(spectrogram + _EPSILON)
  spectrogram = (spectrogram - _MEAN_LOG) / _STD_LOG
  return tf.clip_by_value(spectrogram, -1.0, 1.0) if clip else spectrogram


def _msec_to_samples(duration_ms: float, sample_rate: float) -> int:
  return int(round(sample_rate * duration_ms / 1000.0))


def log_mel_spectrogram_dmvr(
    samples: tf.Tensor,
    sample_rate: int,
    normalize_waveform: bool,
    frame_length_ms: float = _FRAME_LENGTH_MS,
    frame_step_ms: float = _FRAME_STEP_MS,
    num_mel_channels: int = _NUM_MEL_CHANNELS
) -> tf.Tensor:
  """Computes a log-mel spectrogram for the given audio samples.

  This uses DMVR backend.

  Args:
    samples: waveform samples of dimension (N,).
    sample_rate: sample rate
    normalize_waveform: Normalize the waveform before processing.
    frame_length_ms: frame duration in milliseconds.
    frame_step_ms: frame step (stride) in milliseconds.
    num_mel_channels: number of log-mel features.

  Returns:
    Log mel-spectral features of dimension (L, D), where the first dimension
    is the number of analysis frames L  and D is the number of mel-channels.
  """
  samples = tf.cast(samples, dtype=tf.float32)
  return dmvr_lib.compute_audio_spectrogram(
      raw_audio=samples,
      num_subclips=1,
      sample_rate=sample_rate,
      spectrogram_type="logmf",
      frame_length=_msec_to_samples(frame_length_ms, sample_rate),
      frame_step=_msec_to_samples(frame_step_ms, sample_rate),
      num_features=num_mel_channels,
      lower_edge_hertz=_MEL_LOWER_EDGE_HERTZ,
      upper_edge_hertz=_MEL_UPPER_EDGE_HERTZ,
      preemphasis=_PRE_EMPHASIS,
      normalize=normalize_waveform
  )


def log_mel_spectrogram(
    backend: str,
    samples: tf.Tensor,
    sample_rate: int,
    normalize_waveform: bool,
    frame_length_ms: float = _FRAME_LENGTH_MS,
    frame_step_ms: float = _FRAME_STEP_MS,
    num_mel_channels: int = _NUM_MEL_CHANNELS
) -> tf.Tensor:
  """Computes a log-mel spectrogram for the given audio samples.

  Args:
    backend: Type of the acoustic backend to use.
    samples: Waveform samples of dimension (N,).
    sample_rate: Sample rate in Hz.
    normalize_waveform: Normalize the waveform before processing.
    frame_length_ms: Frame duration in milliseconds.
    frame_step_ms: Frame step in milliseconds.
    num_mel_channels: Number of log-mel features.

  Returns:
    Log mel-spectral features of dimension (L, D), where the first dimension
    is the number of analysis frames L  and D is the number of mel-channels.

  Raises:
    ValueError if backend is unsupported or upsampling is requested.
  """

  if sample_rate < _FIXED_SAMPLE_RATE:
    raise ValueError(f"Upsampling is requested from {sample_rate} Hz!")

  resampled_samples = resample_lib.resample_audio(
      samples,
      in_sample_rate=sample_rate,
      out_sample_rate=_FIXED_SAMPLE_RATE
  )
  if backend == "dmvr":
    return log_mel_spectrogram_dmvr(
        resampled_samples,
        _FIXED_SAMPLE_RATE,
        normalize_waveform,
        frame_length_ms=frame_length_ms,
        frame_step_ms=frame_step_ms,
        num_mel_channels=num_mel_channels
    )
  else:
    raise ValueError(f"Unsupported backend {backend}")
