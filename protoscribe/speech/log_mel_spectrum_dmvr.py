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

"""Spectrogram utilities from DMVR project.

See https://github.com/google-deepmind/dmvr.
"""

import tensorflow as tf


def _preemphasis(audio: tf.Tensor, coef: float = 0.97) -> tf.Tensor:
  """Scale up the high frequency components in the waveform.

  Args:
    audio: Input waveform.
    coef: Pre-emphasis coefficient.

  Returns:
    Pre-emphasized audio.
  """
  return tf.concat([audio[:1], audio[1:] - coef * audio[:-1]], axis=0)


def compute_audio_spectrogram(
    raw_audio: tf.Tensor,
    num_subclips: int = 1,
    sample_rate: int = 48000,
    spectrogram_type: str = 'logmf',
    frame_length: int = 2048,
    frame_step: int = 1024,
    num_features: int = 80,
    lower_edge_hertz: float = 80.0,
    upper_edge_hertz: float = 7600.0,
    preemphasis: float | None = None,
    normalize: bool = False,
    fft_output_conversion: str = 'magnitude',
) -> tf.Tensor:
  """Computes audio spectrograms.

  Args:
    raw_audio: Tensor representing audio.
    num_subclips: Number of test clips (1 by default). If more than 1, this will
      sample multiple linearly spaced clips within each audio at test time.
      If 1, then a single clip in the middle of the audio is sampled. The clips
      are aggreagated in the batch dimension.
    sample_rate: The sample rate of the input audio.
    spectrogram_type: The type of the spectrogram to be extracted from the
      waveform. Can be either `spectrogram`, `logmf`, and `mfcc`.
    frame_length: The length of each spectrogram frame.
    frame_step: The stride of spectrogram frames.
    num_features: The number of spectrogram features.
    lower_edge_hertz: Lowest frequency to consider.
    upper_edge_hertz: Highest frequency to consider.
    preemphasis: The strength of pre-emphasis on the waveform. If None, no
      pre-emphasis will be applied.
    normalize: Whether to normalize the waveform or not.
    fft_output_conversion: The string indicating the output conversion function.
      Currently, only `magnitude` and `magnitude_squared` are supported.

  Returns:
    Tensor containing the extracted spectrograms.

  Raises:
    ValueError: if `spectrogram_type` is one of `spectrogram`, `logmf`, or
      `mfcc`.
  """
  if spectrogram_type not in ['spectrogram', 'logmf', 'mfcc']:
    raise ValueError('Spectrogram type should be one of `spectrogram`, '
                     '`logmf`, or `mfcc`, got {}'.format(spectrogram_type))

  if fft_output_conversion not in ['magnitude', 'magnitude_squared']:
    raise ValueError(
        'FFT output conversion should be one of `magnitude` or '
        '`magnitude_squared, god {}`'.format(fft_output_conversion))

  if normalize:
    raw_audio /= (
        tf.reduce_max(tf.abs(raw_audio), axis=-1, keepdims=True) + 1e-8)
  if num_subclips > 1:
    raw_audio = tf.reshape(raw_audio, [num_subclips, -1])
  if preemphasis is not None:
    raw_audio = _preemphasis(raw_audio, preemphasis)

  def _extract_spectrogram(
      waveform: tf.Tensor,
      spectrogram_type: str) -> tf.Tensor:
    stfts = tf.signal.stft(waveform,
                           frame_length=frame_length,
                           frame_step=frame_step,
                           fft_length=frame_length,
                           window_fn=tf.signal.hann_window,
                           pad_end=True)
    if fft_output_conversion == 'magnitude_squared':
      stfts = tf.square(stfts)
    spectrograms = tf.abs(stfts)

    if spectrogram_type == 'spectrogram':
      return spectrograms[..., :num_features]

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_features, num_spectrogram_bins, sample_rate, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    if spectrogram_type == 'logmf':
      return log_mel_spectrograms

    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrograms)[..., :13]
    return mfccs

  return _extract_spectrogram(raw_audio, spectrogram_type)
