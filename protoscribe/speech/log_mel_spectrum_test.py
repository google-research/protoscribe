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

"""Unit tests for log mel-spectral utilities."""

import os

from absl import flags
from absl.testing import absltest
from protoscribe.speech import log_mel_spectrum as lib
from scipy.io import wavfile
import tensorflow as tf

FLAGS = flags.FLAGS

_TESTDATA_DIR = "protoscribe/speech/testdata"
_WAVEFORM_NAME = "bur_0366_0096392289_16kHz.wav"
_NUM_SAMPLES = 65536
_NUM_FRAMES_DMVR = 328
_SAMPLE_RATE = 16_000
_NUM_CHANNELS = 128
_LONG_FRAME_DUR_MS = 70.  # Longer than default 50ms.
_LONG_FRAME_STEP_MS = 14.  # Longer than default 12.5ms.
_SHORT_FRAME_DUR_MS = 25.  # Shorter than default 50ms.
_SHORT_FRAME_STEP_MS = 10.  # Shorter than default 12.5ms.


class LogMelSpectrumTest(absltest.TestCase):

  def _read_waveform(self) -> tf.Tensor:
    path = os.path.join(FLAGS.test_srcdir, _TESTDATA_DIR, _WAVEFORM_NAME)
    sample_rate, samples = wavfile.read(path)
    self.assertEqual(_SAMPLE_RATE, sample_rate)
    self.assertLen(samples, _NUM_SAMPLES)
    return tf.constant(samples, dtype=tf.float32)

  def test_log_mel_spectrogram_dmvr(self):
    samples = self._read_waveform()
    features = lib.log_mel_spectrogram_dmvr(
        samples, _SAMPLE_RATE, normalize_waveform=False
    )
    self.assertEqual(features.shape, (_NUM_FRAMES_DMVR, _NUM_CHANNELS))
    features = lib.log_mel_spectrogram_dmvr(
        samples,
        _SAMPLE_RATE,
        normalize_waveform=False,
        frame_length_ms=_LONG_FRAME_DUR_MS,
        frame_step_ms=_LONG_FRAME_STEP_MS
    )
    self.assertGreater(_NUM_FRAMES_DMVR, features.shape[0])
    features = lib.log_mel_spectrogram_dmvr(
        samples,
        _SAMPLE_RATE,
        normalize_waveform=False,
        frame_length_ms=_SHORT_FRAME_DUR_MS,
        frame_step_ms=_SHORT_FRAME_STEP_MS
    )
    self.assertLess(_NUM_FRAMES_DMVR, features.shape[0])

  def test_unknown_backend(self):
    samples = self._read_waveform()
    with self.assertRaises(ValueError):
      _ = lib.log_mel_spectrogram(
          backend="bogus",
          samples=samples,
          sample_rate=_SAMPLE_RATE,
          normalize_waveform=True
      )

  def test_bad_upsampling(self):
    # The following snippet requests upsampling from 8 kHz to 16 kHz currently
    # required for computing the log-mel spectral features.
    with self.assertRaises(ValueError):
      _ = lib.log_mel_spectrogram(
          backend="dmvr",
          samples=tf.ones((100,), dtype=tf.float32),
          sample_rate=8_000,
          normalize_waveform=False
      )


if __name__ == "__main__":
  absltest.main()
