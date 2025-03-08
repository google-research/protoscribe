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

r"""Basic quantizer for corpus sketches.

Please run with `--override_dataset_dir` having copied the dataset locally.
Otherwise the run will take forever.

Example:
--------
python protoscribe/corpus/tools/quantize_sketches_simple_main.py \
  --dataset_dir /usr/local/data/protoscribe \
  --sample_num_sketches 10000 \
  --vocab_size 1024 \
  --output_npy_file /tmp/vocab.npy \
  --logtostderr
"""

import itertools
import logging
from typing import Sequence

from absl import app
from absl import flags
import ml_collections
import numpy as np
from protoscribe.corpus.reader import tasks as tasks_lib
from protoscribe.corpus.tools import quantize_sketches_simple
import seqio
from sklearn import cluster

import glob
import os

Array = np.ndarray

_RANDOM_SEED = 42

_MAX_STROKE_SEQUENCE_LENGTH = flags.DEFINE_integer(
    "max_stroke_sequence_length", 250,
    "Maximum number of the points in the stroke."
)

_STROKE_NORMALIZATION_TYPE = flags.DEFINE_enum(
    "stroke_normalization_type", "sketch-rnn",
    [
        "none",
        "z-standardize",
        "min-max",
        "mean-norm",
        "sketch-rnn",
        "det-covar"
    ],
    "Stroke normalization type."
)

_STROKE_RANDOM_SCALE_FACTOR = flags.DEFINE_float(
    "stroke_random_scale_factor", 0.,
    "Random stretch factor for sketch. By default do not stretch the sketch. "
    "This corresponds to the current values in model configurations."
)

_SAMPLE_NUM_SKETCHES = flags.DEFINE_integer(
    "sample_num_sketches", 10,
    "Number of sketches to randomly sample from the dataset."
)

_SAMPLE_NUM_POINTS = flags.DEFINE_integer(
    "sample_num_points", 20_000_000,
    "Number of points to sample from the sketches above."
)

_LIFT_TO_TOUCH_RATIO = flags.DEFINE_float(
    "lift_to_touch_ratio", 0.85,
    "Percentage of movements with pen lifted (minority) to pen touching the "
    "paper (majority). Set to 0 to disable."
)

_OUTPUT_NPY_FILE = flags.DEFINE_string(
    "output_npy_file", None,
    "Output numpy binary file (.npy) containing the resulting quantization.",
    required=True
)

_SPLIT = flags.DEFINE_string(
    "split", "train",
    "Dataset split."
)

_VOCAB_SIZE = flags.DEFINE_integer(
    "vocab_size", 1024,
    "Size of the vocabulary."
)

_ALGORITHM_TYPE = flags.DEFINE_enum(
    "algorithm_type", "regular", ["regular", "mini-batch"],
    "Type of the K-Means to run. Regular K-Means by default. For large "
    "amounts of data, prefer `mini-batch`."
)

_KMEANS_MAX_ITER = flags.DEFINE_integer(
    "kmeans_max_iter", 1_000,
    "Maximum number of iterations."
)

_TOLERANCE = flags.DEFINE_float(
    "tolerance", 1e-6,
    "Relative tolerance with regards to Frobenius norm of the difference "
    "in the cluster centers of two consecutive iterations to declare "
    "convergence."
)

_TASK_NAME = "quantizer"


def _get_config() -> ml_collections.FrozenConfigDict:
  """Loads the configuration for the model."""
  config = ml_collections.FrozenConfigDict({
      "max_stroke_sequence_length": _MAX_STROKE_SEQUENCE_LENGTH.value,
      "stroke_normalization_type": _STROKE_NORMALIZATION_TYPE.value,
      "stroke_random_scale_factor": _STROKE_RANDOM_SCALE_FACTOR.value,
  })
  return config


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  logging.info("Initializing ...")
  config = _get_config()
  task_name = tasks_lib.register(
      task_name=_TASK_NAME,
      max_stroke_sequence_length=config.max_stroke_sequence_length,
      stroke_normalization_type=config.stroke_normalization_type,
      stroke_random_scale_factor=config.stroke_random_scale_factor
  )
  task = seqio.TaskRegistry.get(task_name)
  ds = task.get_dataset(split=_SPLIT.value)

  logging.info("Collecting points ...")
  examples = itertools.islice(
      ds.as_numpy_iterator(), _SAMPLE_NUM_SKETCHES.value
  )
  points = quantize_sketches_simple.collect_points(
      examples,
      random_seed=_RANDOM_SEED,
      total_num_points=_SAMPLE_NUM_POINTS.value,
      lift_to_touch_ratio=_LIFT_TO_TOUCH_RATIO.value
  )

  logging.info(
      "Collected %d points. Clustering with %s K-Means ...",
      points.shape[0], _ALGORITHM_TYPE.value
  )
  if _ALGORITHM_TYPE.value == "regular":
    result = cluster.KMeans(
        n_clusters=_VOCAB_SIZE.value,
        max_iter=_KMEANS_MAX_ITER.value,
        algorithm="auto",
        tol=_TOLERANCE.value,
        verbose=0
    ).fit(points)
  else:
    result = cluster.MiniBatchKMeans(
        n_clusters=_VOCAB_SIZE.value,
        compute_labels=False,
        max_iter=_KMEANS_MAX_ITER.value,
        max_no_improvement=5_000,
        batch_size=4096,
        tol=_TOLERANCE.value,
        verbose=1
    ).fit(points)
  logging.info(
      "Finished %d points in %s iterations. Inertia: %f",
      points.shape[0], result.n_iter_, result.inertia_
  )

  logging.info("Saving clusters to %s ...", _OUTPUT_NPY_FILE.value)
  with open(_OUTPUT_NPY_FILE.value, mode="wb") as f:
    np.save(f, result.cluster_centers_.astype(dtype=np.float64))


if __name__ == "__main__":
  app.run(main)
