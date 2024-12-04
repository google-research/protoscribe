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
from typing import Any, Iterable, Sequence

from absl import app
from absl import flags
import ml_collections
import numpy as np
from protoscribe.corpus.reader import tasks as tasks_lib
from protoscribe.sketches.utils import stroke_utils as strokes_lib
from sklearn import cluster
import t5

import glob
import os

Array = np.ndarray

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
    "stroke_random_scale_factor", 0.15,
    "Random stretch factor for sketch."
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

_TASK_NAME = "quantizer"


def _get_config() -> ml_collections.ConfigDict:
  """Loads the configuration for the model."""
  config = ml_collections.ConfigDict()
  config.max_stroke_sequence_length = _MAX_STROKE_SEQUENCE_LENGTH.value
  config.stroke_normalization_type = _STROKE_NORMALIZATION_TYPE.value
  config.stroke_random_scale_factor = _STROKE_RANDOM_SCALE_FACTOR.value
  return config


def _collect_points(examples: Iterable[dict[str, Any]]) -> Array:
  """Collects the points where the pen touches and lifts from the paper.

  Args:
    examples: Documents read from the corpus. Each document is a dictionary
      of features.

  Returns:
    Array of touch and lift points.

  Raises:
    ValueError if documents are invalid.
  """

  # Collect lists of tuples representing all touch and lift points.
  logging.info("Collecting all touch and lift points ...")
  touch_points, lift_points = [], []
  for i, features in enumerate(examples):
    if "strokes" not in features:
      raise ValueError(f"[{i}] Bad dataset: strokes expected!")
    sketch = features["strokes"]
    # Stroke-3: The third element indicates whether the pen is lifted away
    # from the paper.
    sketch = strokes_lib.stroke5_to_stroke3(sketch)
    # Offset to next touch point.
    pen_lift_ids = np.where(sketch[:, 2] == 1)[0] + 1
    pen_lift_ids = pen_lift_ids[:-1]  # Exclude EOS.
    # Pen touching the paper.
    pen_touch_ids = set(range(len(sketch))) - set(pen_lift_ids)

    touch_points.append(sketch[list(pen_touch_ids), :2])
    lift_points.append(sketch[list(pen_lift_ids), :2])

  touch_points = np.concatenate(touch_points, dtype=np.float32)
  lift_points = np.concatenate(lift_points, dtype=np.float32)
  num_touch = touch_points.shape[0]
  num_lift = lift_points.shape[0]
  logging.info(
      "Lift points: %d, touch points: %d, lift/touch data ratio: %f",
      num_lift, num_touch, num_lift / num_touch
  )

  # Sample the points given the required lift-to-touch ration.
  logging.info("Sampling points ...")
  if _LIFT_TO_TOUCH_RATIO.value > 0.:
    num_sample_lift = int(
        _LIFT_TO_TOUCH_RATIO.value * _SAMPLE_NUM_POINTS.value
    )
    num_sample_touch = _SAMPLE_NUM_POINTS.value - num_sample_lift
    if num_sample_lift < num_lift:
      logging.info("Sampling %d lift points ...", num_sample_lift)
      lift_ids = np.random.choice(num_lift, num_sample_lift, replace=False)
      lift_points = lift_points[lift_ids]
    if num_sample_touch < num_touch:
      logging.info("Sampling %d touch points ...", num_sample_touch)
      touch_ids = np.random.choice(num_touch, num_sample_touch, replace=False)
      touch_points = touch_points[touch_ids]

  # Translates slice objects to concatenation along the first axis.
  return np.r_[touch_points, lift_points].astype(dtype=np.float32)


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
  task = t5.data.TaskRegistry.get(task_name)
  ds = task.get_dataset(split=_SPLIT.value)

  logging.info("Collecting points ...")
  examples = itertools.islice(
      ds.as_numpy_iterator(), _SAMPLE_NUM_SKETCHES.value
  )
  points = _collect_points(examples)

  logging.info(
      "Collected %d points. Clustering with %s K-Means ...",
      points.shape[0], _ALGORITHM_TYPE.value
  )
  if _ALGORITHM_TYPE.value == "regular":
    result = cluster.KMeans(
        n_clusters=_VOCAB_SIZE.value,
        max_iter=_KMEANS_MAX_ITER.value,
        algorithm="auto",
        tol=1e-6,
        verbose=0
    ).fit(points)
  else:
    result = cluster.MiniBatchKMeans(
        n_clusters=_VOCAB_SIZE.value,
        compute_labels=False,
        max_iter=_KMEANS_MAX_ITER.value,
        max_no_improvement=5_000,
        batch_size=4096,
        tol=0.0,
        verbose=1
    ).fit(points)
  logging.info(
      "Finished %d points in %s iterations. Inertia: %f",
      points.shape[0], result.n_iter_, result.inertia_
  )

  logging.info("Saving clusters to %s ...", _OUTPUT_NPY_FILE.value)
  with open(_OUTPUT_NPY_FILE.value, mode="wb") as f:
    np.save(f, result.cluster_centers_)


if __name__ == "__main__":
  app.run(main)
