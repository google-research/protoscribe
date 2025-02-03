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

"""Sanity check for miscellaneous model configurations in gin."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import gin

# Core PMMX configurations. These are copied from
#   protoscribe/pmmx/config/runs
# and modified to work in test environment.
_CORE_CONFIG_BASE_DIR = (
    "protoscribe/models/pmmx/testdata"
)

# Configurations for individual models.
_MODEL_CONFIG_BASE_DIR = (
    "protoscribe/models/pmmx/configs"
)


def _config_path(
    filename: str,
    config_dir: str = _MODEL_CONFIG_BASE_DIR
) -> str:
  """Returns full path of the specified file name."""
  return os.path.join(
      absltest.get_default_test_srcdir(), config_dir, filename
  )


class ModelConfigGinTest(parameterized.TestCase):

  def tearDown(self):
    super().tearDown()
    gin.clear_config()

  @parameterized.parameters(
      "glyph_concepts",
      "glyph_logmel-spectrum",
      "glyph_phonemes",
      "sketch-token_concepts",
  )
  def test_model_train(self, model_dir: str) -> None:
    """Tests tiny model configuration for training."""
    tmp_dir = absltest.get_default_test_tmpdir()
    gin.parse_config_files_and_bindings(
        config_files=[
            _config_path("pretrain_test.gin", config_dir=_CORE_CONFIG_BASE_DIR),
            _config_path(os.path.join(model_dir, "model_tiny.gin")),
            _config_path(os.path.join(model_dir, "dataset.gin")),
        ],
        bindings=[
            f"DATA_DIR=\"{tmp_dir}\"",
            f"MODEL_DIR=\"{tmp_dir}\"",
            "TRAIN_STEPS=1",
            "BATCH_SIZE=8",
            "EVAL_BATCH_SIZE=8",
        ],
        finalize_config=True,
        skip_unknown=False
    )

  @parameterized.parameters(
      "glyph_concepts",
      "glyph_logmel-spectrum",
      "glyph_phonemes",
      "sketch-token_concepts",
  )
  def test_model_infer(self, model_dir: str) -> None:
    """Tests tiny model configuration in inference mode."""
    tmp_dir = absltest.get_default_test_tmpdir()
    gin.parse_config_files_and_bindings(
        config_files=[
            _config_path("infer_test.gin", config_dir=_CORE_CONFIG_BASE_DIR),
            _config_path(os.path.join(model_dir, "model_tiny.gin")),
            _config_path(os.path.join(model_dir, "dataset.gin")),
        ],
        bindings=[
            f"DATA_DIR=\"{tmp_dir}\"",
            f"CHECKPOINT_PATH=\"{tmp_dir}\"",
            f"INFER_OUTPUT_DIR=\"{tmp_dir}\"",
            "BATCH_SIZE=8",
        ],
        finalize_config=True,
        skip_unknown=False
    )


if __name__ == "__main__":
  absltest.main()
