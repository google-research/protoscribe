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

from absl.testing import absltest
from protoscribe.corpus.builder import speech_pipeline as lib
import tensorflow as tf


class SpeechPipelineTest(absltest.TestCase):

  def test_dummy_pipeline(self):
    pipeline = lib.DummySpeechPipeline()
    example = tf.train.Example()
    doc_id = 12345

    # Checks that no processing of empty documents is allowed.
    with self.assertRaises(ValueError):
      pipeline.process_example(doc_id=doc_id, input_example=example)

    # Fills in the required fields (and more) and tests again that the input
    # document is unchanged with two extra audio-related features added
    # (sample rate and the audio samples).
    pron = "o\" m # i\" p"
    example.features.feature["text/sampa"].bytes_list.value.append(
        pron.encode("utf-8")
    )
    example.features.feature["other"].int64_list.value.append(doc_id)
    new_id, new_example = pipeline.process_example(
        doc_id=doc_id, input_example=example
    )
    self.assertEqual(new_id, doc_id)
    self.assertLen(new_example.features.feature, 4)
    self.assertIn("text/sampa", new_example.features.feature)
    pron_feature = new_example.features.feature["text/sampa"]
    self.assertLen(pron_feature.bytes_list.value, 1)
    self.assertEqual(pron_feature.bytes_list.value[0].decode("utf-8"), pron)
    self.assertIn("other", new_example.features.feature)
    other_feature = new_example.features.feature["other"]
    self.assertLen(other_feature.int64_list.value, 1)
    self.assertEqual(other_feature.int64_list.value[0], doc_id)
    self.assertIn("audio/sample_rate", new_example.features.feature)
    sample_rate_feat = new_example.features.feature["audio/sample_rate"]
    self.assertLen(sample_rate_feat.int64_list.value, 1)
    self.assertEqual(
        sample_rate_feat.int64_list.value[0], lib.DUMMY_SAMPLE_RATE_HZ
    )
    self.assertIn("audio/waveform", new_example.features.feature)
    waveform_feat = new_example.features.feature["audio/waveform"]
    self.assertLen(waveform_feat.float_list.value, lib.DUMMY_NUM_SAMPLES)


if __name__ == "__main__":
  absltest.main()
