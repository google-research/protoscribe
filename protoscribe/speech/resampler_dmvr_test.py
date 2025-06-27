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

"""Simple tests for audio resampler."""

from absl.testing import absltest
from absl.testing import parameterized
from protoscribe.speech import resampler_dmvr as lib
import tensorflow as tf


class ResamplerDmvrTest(parameterized.TestCase):

  @parameterized.parameters((16_000, 32_000), (32_000, 16_000))
  def test_resample_audio_const(self, input_sample_rate, output_sample_rate):
    audio = tf.ones(shape=(1, input_sample_rate))
    expected_out = tf.ones(shape=(1, output_sample_rate))

    resampled_audio = lib.resample_audio(
        audio=audio,
        in_sample_rate=input_sample_rate,
        out_sample_rate=output_sample_rate,
    )

    # Larger diffs at the boundaries, so we consider average across whole seq.
    mean_abs_diff = abs(expected_out.numpy() - resampled_audio.numpy()).mean()
    self.assertAlmostEqual(mean_abs_diff, 0, places=2)

  @parameterized.parameters((16_000, 32_000), (32_000, 16_000))
  def test_resample_audio_sin(self, input_sample_rate, output_sample_rate):
    audio = tf.math.sin(tf.linspace(0, 10, input_sample_rate))[None, :]
    expected_out = tf.math.sin(tf.linspace(0, 10, output_sample_rate))[None, :]

    resampled_audio = lib.resample_audio(
        audio=audio,
        in_sample_rate=input_sample_rate,
        out_sample_rate=output_sample_rate,
    )

    # Larger diffs at the boundaries, so we consider average across whole seq.
    mean_abs_diff = abs(expected_out.numpy() - resampled_audio.numpy()).mean()
    self.assertAlmostEqual(mean_abs_diff, 0, places=2)


if __name__ == "__main__":
  absltest.main()
