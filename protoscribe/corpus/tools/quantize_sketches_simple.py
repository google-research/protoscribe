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

"""Simple 2D point quantization."""

import logging
from typing import Any, Iterable

import numpy as np
from protoscribe.sketches.utils import stroke_utils as strokes_lib

Array = np.ndarray


def _collect_points_basic(
    examples: Iterable[dict[str, Any]],
    rng: np.random.Generator,
    total_num_points: int
) -> Array:
  """Collects the stroke points.

  Args:
    examples: Documents read from the corpus. Each document is a dictionary
      of features.
    rng: Random number generator.
    total_num_points: Number of points to sample.

  Returns:
    Array of points.

  Raises:
    ValueError if documents are invalid.
  """
  logging.info("Collecting points ...")
  sketches = []
  for i, features in enumerate(examples):
    if "strokes" not in features:
      raise ValueError(f"[{i}] Bad dataset: strokes expected!")
    sketch = features["strokes"]
    sketch3 = strokes_lib.stroke5_to_stroke3(sketch)
    sketches.append(sketch3[:, :2].astype(dtype=np.float32))

  points = np.concatenate(sketches)
  logging.info("Collected %d points.", points.shape[0])

  # Sample the requested number of points.
  if total_num_points > points.shape[0]:
    logging.info("Shuffling the points ...")
    rng.shuffle(points, axis=0)
    return points

  logging.info("Sampling %d points ...", total_num_points)
  sampled_points = rng.choice(points, total_num_points, replace=False)

  return sampled_points


def _collect_points_lift_to_touch(
    examples: Iterable[dict[str, Any]],
    rng: np.random.Generator,
    total_num_points: int,
    lift_to_touch_ratio: float
) -> Array:
  """Collects the points where the pen touches and lifts from the paper.

  This tokenizer is based on Leo Sampaio Ferraz Ribeiro, Tu Bui,
  John Collomosse, Moacir Ponti "Sketchformer: Transformer-based Representation
  for Sketched Structure" (https://arxiv.org/abs/2002.10381).

  Args:
    examples: Documents read from the corpus. Each document is a dictionary
      of features.
    rng: Random number generator.
    total_num_points: Number of points to sample.
    lift_to_touch_ratio: Percentage of movements with pen lifted (minority) to
      pen touching the ""paper (majority). Set to 0 to disable.

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

  touch_points = np.concatenate(touch_points).astype(dtype=np.float32)
  lift_points = np.concatenate(lift_points).astype(dtype=np.float32)
  num_touch = touch_points.shape[0]
  num_lift = lift_points.shape[0]
  logging.info(
      "Lift points: %d, touch points: %d, lift/touch data ratio: %f",
      num_lift, num_touch, num_lift / num_touch
  )

  # Sample the points given the required lift-to-touch ratio.
  logging.info("Sampling points ...")
  if lift_to_touch_ratio > 0.:
    num_sample_lift = int(
        lift_to_touch_ratio * total_num_points
    )
    num_sample_touch = total_num_points - num_sample_lift
    if num_sample_lift < num_lift:
      logging.info("Sampling %d lift points ...", num_sample_lift)
      lift_ids = rng.choice(num_lift, num_sample_lift, replace=False)
      lift_points = lift_points[lift_ids]
    if num_sample_touch < num_touch:
      logging.info("Sampling %d touch points ...", num_sample_touch)
      touch_ids = rng.choice(num_touch, num_sample_touch, replace=False)
      touch_points = touch_points[touch_ids]

  # Translates slice objects to concatenation along the first axis.
  return np.r_[touch_points, lift_points].astype(dtype=np.float32)


def collect_points(
    examples: Iterable[dict[str, Any]],
    random_seed: int,
    total_num_points: int,
    lift_to_touch_ratio: float
) -> Array:
  """Collects the points where the pen touches and lifts from the paper.

  Args:
    examples: Documents read from the corpus. Each document is a dictionary
      of features.
    random_seed: Integer random number generator seed.
    total_num_points: Number of points to sample.
    lift_to_touch_ratio: Percentage of movements with pen lifted (minority) to
      pen touching the ""paper (majority). Set to 0 to disable.

  Returns:
    Array of touch and lift points.

  Raises:
    ValueError if documents are invalid.
  """
  rng = np.random.default_rng(random_seed)
  if lift_to_touch_ratio != 0.:
    return _collect_points_lift_to_touch(
        examples, rng, total_num_points, lift_to_touch_ratio
    )
  else:
    return _collect_points_basic(examples, rng, total_num_points)
