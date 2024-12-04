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

"""Simple test for DMVR implemenentation of spectral features."""

from absl.testing import absltest
import numpy as np
from protoscribe.speech import log_mel_spectrum_dmvr as lib
import tensorflow as tf


def _get_num_frames_output(
    num_frames: int,
    frame_length: int,
    frame_step: int,
    pad_end: bool,
) -> int:
  """Get the number of frames in the output spectrogram.

  See https://www.tensorflow.org/api_docs/python/tf/signal/frame

  Args:
    num_frames: number of frames in the raw audio input.
    frame_length: range of a spectrogram feature.
    frame_step: number of frames between each spectrogram feature.
    pad_end: do we pad the input audio with zeros to not discard any input data.

  Returns:
    The number of frames in the output spectrogram.
  """
  if pad_end:
    val = num_frames // frame_step
    if num_frames % frame_step != 0:
      val += 1
    return val
  else:
    effective_frames = num_frames - frame_length
    return 1 + effective_frames // frame_step


class LogMelSpectrumDmvrTest(absltest.TestCase):

  def test_shape(self):
    """Checks the shape of the spectrogram."""
    sample_rate = 16_000
    frame_length = 1200
    frame_step = 480
    num_features = 128
    padding = True  # Padding is always true for DMVR spectrograms.

    num_frames_list = [
        sample_rate * 30,
        sample_rate * 20 + 5,
        frame_step * frame_step * 8,
    ]
    for num_frames in num_frames_list:
      audio_sample = tf.convert_to_tensor(
          np.random.randn(num_frames), dtype=tf.float32
      )
      spectrum = lib.compute_audio_spectrogram(
          audio_sample,
          sample_rate=sample_rate,
          frame_length=frame_length,
          frame_step=frame_step,
          num_features=num_features,
      )
      out_num_frames = _get_num_frames_output(
          num_frames, frame_length, frame_step, padding
      )
      expected_shape = (out_num_frames, num_features)
      self.assertEqual(expected_shape, spectrum.shape)


if __name__ == "__main__":
  absltest.main()
