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

"""Speech resampler based on Fast Fourier Transforms (FFTs) from DMVR.

See: https://github.com/google-deepmind/dmvr
"""

import tensorflow as tf


def _resample_audio_fft(
    x: tf.Tensor,
    in_sample_rate: int,
    out_sample_rate: int,
    resolution_bits: float | None = None
) -> tf.Tensor:
  """Resamples audio using FFTs.

  Args:
    x: Input audio signal.
    in_sample_rate: The original sample rate of the input audio in Hz..
    out_sample_rate: The target sample rate in Hz.
    resolution_bits: Resolution bits used to scale the FFTs. If None no scaling
      is used.

  Returns:
    The resampled audio signal.
  """
  axis = -1  # tf.signal.fft operates on the innermost dimension of x
  if in_sample_rate == out_sample_rate:
    return x

  scale = 2**(resolution_bits - 1) if resolution_bits else None

  if scale:
    x /= scale

  factor = float(out_sample_rate) / in_sample_rate
  original_size = tf.shape(x)[axis]
  resampled_size = tf.cast(
      tf.cast(original_size, dtype=tf.float32) * factor, dtype=tf.int32)

  x_ = tf.signal.fft(tf.cast(x, dtype=tf.complex64))

  shape = x.get_shape().as_list()
  rank = len(shape)
  sl_beg = [slice(None)] * rank
  sl_end = [slice(None)] * rank

  min_size = tf.minimum(resampled_size, original_size)
  sl_beg[axis] = slice(0, (min_size + 1) // 2)
  sl_end[axis] = slice(-(min_size - 1) // 2, None)

  # Compute padding: empty unless upsampling (resampled_size > original_size).
  pad_shape = list(shape)
  pad_shape[axis] = tf.maximum(0, resampled_size - original_size)
  padding = tf.zeros(pad_shape, dtype=x_.dtype)

  y_ = tf.concat([x_[sl_beg], padding, x_[sl_end]], axis=axis)
  y = tf.signal.ifft(y_)
  y = tf.math.real(y) * factor

  # Deliberately subtract 1 to prevent clipped values from going out of range.
  y = tf.clip_by_value(y, -1, 1)
  if scale:
    y *= scale - 1
  if shape[axis] is not None:
    shape[axis] = int(shape[axis] * factor)
  y.set_shape(shape)

  return y


def resample_audio(
    audio: tf.Tensor, in_sample_rate: int, out_sample_rate: int
) -> tf.Tensor:
  """Resamples raw audio.

  Args:
    audio: Input audio signal.
    in_sample_rate: The original sample rate of the input audio in Hz.
    out_sample_rate: The target sample rate in Hz.

  Returns:
    The resampled audio signal.
  """
  return _resample_audio_fft(audio, in_sample_rate, out_sample_rate)
