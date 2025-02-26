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

"""Utilities for stroke statistics and transformations."""

import dataclasses
import json
import logging
import math
from typing import Union

import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow as tf

import glob
import os

FinalStrokeStats = dict[str, Union[int, float, tuple[float, float]]]
Tensor = Union[jnp.ndarray, np.ndarray, tf.Tensor]


@dataclasses.dataclass
class StrokeStats:
  """Helper class for computing the stroke stats for normalization.

  The helper variables in this class can be used for
    - Data standardization (Z-score normalization): Bring all dimensions
      to have zero mean and unit variance.
    - Rescaling (Min-Max normalization): Bring all features into the
      same range, e.g. [0, 1].

  Note: The naive (co)variance accumulation algorithm is numerically
  unstable. It's preferrable to use the online algorithms provided in

     https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

  for this but it may be tricky as accumulation happens per-document in
  parallel. Instead we implement shifted (co)variance as described by
  the above document.
  """

  count: int = 0
  sum_x: float = 0.
  sum_y: float = 0.
  sum_xy: float = 0.
  sumsq_x: float = 0.
  sumsq_y: float = 0.
  min_x: float = 1_000_000.
  max_x: float = -1.
  min_y: float = 1_000_000.
  max_y: float = -1.
  max_sample_size: int = -1

  # Constant shift.
  k_x: float = 0.
  k_y: float = 0.

  def update_point(self, x, y) -> None:
    """Update the stats from observing a single point."""
    if self.count == 0:
      self.k_x = x
      self.k_y = y

    self.count += 1
    self.sum_x += (x - self.k_x)
    self.sum_y += (y - self.k_y)
    self.sum_xy += ((x - self.k_x) * (y - self.k_y))
    self.sumsq_x += ((x - self.k_x) ** 2)
    self.sumsq_y += ((y - self.k_y) ** 2)
    self.min_x = min(self.min_x, x)
    self.max_x = max(self.max_x, x)
    self.min_y = min(self.min_y, y)
    self.max_y = max(self.max_y, y)

  def accumulate(self, stats) -> None:
    """Accumulates the data from the other stats instance."""
    self.count += stats.count
    self.sum_x += stats.sum_x
    self.sum_y += stats.sum_y
    self.sum_xy += stats.sum_xy
    self.sumsq_x += stats.sumsq_x
    self.sumsq_y += stats.sumsq_y

    self.min_x = min(stats.min_x, self.min_x)
    self.min_y = min(stats.min_y, self.min_y)
    self.max_x = max(stats.max_x, self.max_x)
    self.max_y = max(stats.max_y, self.max_y)

    # Compute maximum among all the observed samples.
    if stats.max_sample_size > self.max_sample_size:
      self.max_sample_size = stats.max_sample_size

  def finalize(self) -> FinalStrokeStats:
    """Finalizes the accumulated statistics."""

    # Compute Var(x), Var(y) and Covar(x, y).
    var_x = (self.sumsq_x - (self.sum_x ** 2) / self.count) / self.count
    var_y = (self.sumsq_y - (self.sum_y ** 2) / self.count) / self.count
    covar = ((self.sum_xy - (self.sum_x * self.sum_y) / self.count) /
             self.count)
    # Determinant of the full covariance matrix M = ((Var(x), Covar(x, y)),
    # (Covar(y, x), Var(y))): det(M) = Var(x)Var(y) - Covar(x, y)^2.
    det_covar = var_x * var_y - (covar ** 2)
    # ``Naive'' stddev - similar to Sketch-RNN, flatten into one dimension and
    # compute variance. Prefix "f_" stands for "flattened".
    f_count = 2 * self.count
    f_sum = self.sum_x + self.sum_y
    f_sumsq = self.sumsq_x + self.sumsq_y
    f_var = (f_sumsq  - (f_sum ** 2) / f_count) / f_count

    def unshift_mean(total_sum: float, k: float) -> float:
      return (total_sum + self.count * k) / self.count

    final_stats = {
        "mean": (
            unshift_mean(self.sum_x, self.k_x),
            unshift_mean(self.sum_y, self.k_y)),
        "stddev": (math.sqrt(var_x), math.sqrt(var_y)),
        "min": (self.min_x, self.min_y),
        "max": (self.max_x, self.max_y),
        "covar": math.sqrt(det_covar),
        "f_stddev": math.sqrt(f_var),
        "max_sequence_length": self.max_sample_size,
    }
    return final_stats

  def save(self, file_path: str) -> None:
    """Finalizes and saves stroke statistics."""
    logging.info("Saving stroke stats to %s ...", file_path)
    with open(file_path, mode="w") as f:
      json.dump(self.finalize(), f)


def should_normalize_strokes(config: ml_collections.FrozenConfigDict) -> bool:
  """Returns true if stroke normalization should be applied."""
  norm_type = config.get("stroke_normalization_type")
  if not norm_type:
    return False
  match norm_type:
    case "z-standardize" | "min-max" | "mean-norm" | "sketch-rnn" | "det-covar":
      return True
    case _:
      if norm_type.lower() == "none":
        return False
  raise ValueError(f"Unsupported stroke normalization type: {norm_type}")


def load_stroke_stats(
    config: ml_collections.FrozenConfigDict, stats_file: str
) -> FinalStrokeStats:
  """Loads stroke stats from a JSON file."""
  if should_normalize_strokes(config):
    logging.info("Loading stroke statistics for data scaling from %s ...",
                 stats_file)
    with open(stats_file, mode="r") as f:
      stats = json.load(f)
    if "mean" not in stats:
      raise ValueError("Statistics should contain means")
    if "stddev" not in stats:
      raise ValueError("Statistics should contain standard deviation")
  else:
    logging.info("Not scaling the data.")
    stats = {
        "mean": (0., 0.),
        "stddev": (1., 1.),
        "min": (0., 0.),
        "max": (1., 1.),
        "f_stddev": 1.,
    }
  return stats


def normalize_strokes(
    config: ml_collections.FrozenConfigDict,
    stats: FinalStrokeStats,
    x: Tensor,
    y: Tensor
) -> tuple[Tensor, Tensor]:
  """Applies various types of scaling/normalization."""
  norm_type = config.get("stroke_normalization_type")
  if norm_type == "z-standardize":
    x = (x - stats["mean"][0]) / stats["stddev"][0]
    y = (y - stats["mean"][1]) / stats["stddev"][1]
  elif norm_type == "min-max":
    x = (x - stats["min"][0]) / (stats["max"][0] - stats["min"][0])
    y = (y - stats["min"][1]) / (stats["max"][1] - stats["min"][1])
  elif norm_type == "mean-norm":
    x = (x - stats["mean"][0]) / (stats["max"][0] - stats["min"][0])
    y = (y - stats["mean"][1]) / (stats["max"][1] - stats["min"][1])
  elif norm_type == "sketch-rnn":
    # Simply divide by `flattened` standard deviation, no translation.
    x = x / stats["f_stddev"]
    y = y / stats["f_stddev"]
  elif norm_type == "det-covar":
    # Normalize by square root of determinant of the covariance matrix.
    x = (x - stats["mean"][0]) / stats["covar"]
    y = (y - stats["mean"][1]) / stats["covar"]
  return x, y


def denormalize_strokes(
    config: ml_collections.FrozenConfigDict,
    stats: FinalStrokeStats,
    x: Tensor, y: Tensor
) -> tuple[Tensor, Tensor]:
  """Applies inverse scaling/normalization to x and y coordinates."""
  norm_type = config.get("stroke_normalization_type")
  if norm_type == "z-standardize":
    x = x * stats["stddev"][0] + stats["mean"][0]
    y = y * stats["stddev"][1] + stats["mean"][1]
  elif norm_type == "min-max":
    x = x * (stats["max"][0] - stats["min"][0]) + stats["min"][0]
    y = y * (stats["max"][1] - stats["min"][1]) + stats["min"][1]
  elif norm_type == "mean-norm":
    x = x * (stats["max"][0] - stats["min"][0]) + stats["mean"][0]
    y = y * (stats["max"][1] - stats["min"][1]) + stats["mean"][1]
  elif norm_type == "sketch-rnn":
    x = x * stats["f_stddev"]
    y = y * stats["f_stddev"]
  elif norm_type == "det-covar":
    x = x * stats["covar"] + stats["mean"][0]
    y = y * stats["covar"] + stats["mean"][1]
  return x, y


def denormalize_strokes_array(
    config: ml_collections.FrozenConfigDict,
    stats: FinalStrokeStats,
    strokes: Tensor,
) -> np.ndarray:
  """Scaling/normalization to array in stroke-5 or stroke-3 format."""
  x, y = denormalize_strokes(
      config, stats, strokes[:, 0], strokes[:, 1]
  )
  if strokes.shape[1] == 5:
    strokes = np.stack(
        [x, y, strokes[:, 2], strokes[:, 3], strokes[:, 4]], axis=1
    )
  elif strokes.shape[1] == 3:  # sketch-3 format.
    strokes = np.stack([x, y, strokes[:, 2]], axis=1)
  else:
    raise RuntimeError("Unknown sketch format!")

  # Keep BOS and EOS.
  strokes[0, 0:2] = 0.
  strokes[-1, 0:2] = 0.
  return strokes
