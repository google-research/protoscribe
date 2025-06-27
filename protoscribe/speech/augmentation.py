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

"""Miscellaneous speech augmentation utilities."""

import ml_collections
from protoscribe.speech import augmentation_lingvo as impl_lib
import tensorflow as tf


def _default_spec_augment_config() -> ml_collections.FrozenConfigDict:
  """Returns default configuration."""
  return ml_collections.FrozenConfigDict({
      "freq_mask_max_bins": 15,
      "freq_mask_count": 1,
      "time_mask_max_frames": 10,
      "time_mask_count": 1,
      "time_mask_max_ratio": 1.,
      "time_warp_max_frames": 0,
      "time_warp_max_ratio": 0.,
  })


def tf_spec_augment_init() -> None:
  """Global initialization for spectrum augmenter.

  Should be called once ideally to create a global step variable.
  """
  impl_lib.tf_spec_augment_init()


def tf_spec_augment(
    spectrum: tf.Tensor,
    config: ml_collections.FrozenConfigDict | None = None,
) -> tf.Tensor:
  """Performs spectrum augmentation on the inputs.

  Aka, SpecAugment:
    Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D.
    and Le, Q.V., 2019. Specaugment: A simple data augmentation method
    for automatic speech recognition. arXiv preprint arXiv:1904.08779.

  Combines three transformations:
   - a time warping of up to max(time_warp_max_frames,
   time_warp_max_ratio*input_length) frames.
   - a masking of sampled frequencies with zeros along the entire time axis
   (freq_mask)
   - a masking of sampled timesteps with zeros along the entire frequency axis
   (time_mask)

  Args:
    spectrum: input mel spectrogram of shape [time, freq].
    config: dictionary containing the following
      - freq_mask_max_bins (int), max number of consecutive mel bins to mask in
        a band.
      - freq_mask_count (int), number of frequency bands to mask.
      - time_mask_max_frames (int), max number of consecutive time frames to
        mask.
      - time_mask_count (int), number of time bands to mask.
      - time_mask_max_ratio (float), max time mask ratio.
      - time_warp_max_frames (int), max numer of time frames to warp.
      - time_warp_max_ratio (int), max ratio of the time warp.
      Optionally, the dictionary may contain the following params
      - use_dynamic_time_mask_max_frames (bool), whether to determine the
        time_mask_max_frames dynamically.
      - time_masks_per_frame (float)

  Returns:
    Augmented mel spectrogram of shape (num_time_bins, num_freq_bins).
  """
  if not config:
    config = _default_spec_augment_config()

  aug_config = impl_lib.AugmenterConfig(
      freq_mask_max_bins=config.freq_mask_max_bins,
      freq_mask_count=config.freq_mask_count,
      time_mask_max_frames=config.time_mask_max_frames,
      time_mask_count=config.time_mask_count,
      time_warp_max_frames=config.time_warp_max_frames,
      time_warp_max_ratio=config.time_warp_max_ratio,
      time_mask_max_ratio=config.time_mask_max_ratio,
      use_dynamic_time_mask_max_frames=config.get(
          "use_dynamic_time_mask_max_frames", False
      ),
      time_masks_per_frame=config.get("time_masks_per_frame", 0.0),
      time_warp_bound=config.get("time_warp_bound", "static")
  )
  spectrum = spectrum[None, ...]  # Batch dimension.
  spectrum = spectrum[..., None]  # Channel dimension.
  spec_shape = tf.shape(spectrum)
  paddings = tf.zeros(spec_shape[:2])

  outputs, _ = impl_lib.tf_spec_augment_lingvo(aug_config, spectrum, paddings)
  outputs = tf.squeeze(outputs, axis=[0, -1])
  return outputs
