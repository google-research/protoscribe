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

"""Main file for running the training, evaluation or prediction pipeline.

This file is intentionally kept short. The majority for logic is in libraries
that can be easily tested and imported in Colab.
"""

from absl import app
from absl import flags
from absl import logging
from clu import platform
import jax
from ml_collections import config_flags
from protoscribe.models.flax import predict
from protoscribe.models.flax import train
import tensorflow as tf

_WORK_DIR = flags.DEFINE_string(
    "workdir", None,
    "Directory to store model data when training or to get the checkpoints "
    "for inference.")

_CONFIG = config_flags.DEFINE_config_file(
    "config",
    "configs/default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)

_PREDICT = flags.DEFINE_boolean(
    "predict", False,
    "Run inference given the trained model."
)

_CHECKPOINT_DIR = flags.DEFINE_string(
    "checkpoint_dir", None,
    "Directory containing the checkpoints to run the inference from."
)

flags.mark_flags_as_required(["config", "workdir"])


def _train_and_evaluate() -> None:
  """Runs training and evaluation loops."""
  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], "GPU")

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(
      f"process_index: {jax.process_index()}, "
      f"process_count: {jax.process_count()}"
  )
  platform.work_unit().create_artifact(
      platform.ArtifactType.DIRECTORY, _WORK_DIR.value, "workdir"
  )

  train.train_and_evaluate(_CONFIG.value, _WORK_DIR.value)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
  logging.info("JAX local devices: %r", jax.local_devices())

  if _PREDICT.value:
    if not _CHECKPOINT_DIR.value:
      raise app.UsageError("Specify checkpoint directory!")
    predict.predict(_CONFIG.value, _CHECKPOINT_DIR.value, _WORK_DIR.value)
  else:
    _train_and_evaluate()


if __name__ == "__main__":
  jax.config.config_with_absl()
  app.run(main)
