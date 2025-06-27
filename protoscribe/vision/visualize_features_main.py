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

"""Tool for visualizing the vision features."""

import logging
import os
import random
from typing import Sequence

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

import glob
import os

_INPUT_FEATURES_NPZ_FILE = flags.DEFINE_string(
    "input_features_npz_file", None,
    "Input file in .npz format containing the features.",
    required=True)

_OUTPUT_PLOT_FILE = flags.DEFINE_string(
    "output_plot_file", None,
    "Path to the output plot.",
    required=True)

_NUM_RANDOM_LABELS = flags.DEFINE_integer(
    "num_random_labels", -1,
    "Number of random labels to display. By default selects all, which "
    "makes the plot too dense and unreadable.")


class TSNEReductionPlot(object):
  """Performs t-SNE dimensionality reduction and plots against given labels.
  """

  def __init__(self, title_suffix: str):
    """Initialize TSNE plot.

    Args:
      title_suffix: suffix appended to title.
    """
    self.title = f"t-SNE for `{title_suffix}`"
    self._set_styling()

  def _set_styling(self):
    """Sets figure styling."""

    _ = plt.figure(figsize=(400, 400))
    self.fig, self.ax = plt.subplots()
    self.ax.set_xlabel("x1", fontdict={"fontsize": 12, "fontweight": "medium"})
    self.ax.set_ylabel("x2", fontdict={"fontsize": 12, "fontweight": "medium"})
    self.ax.set_title(
        self.title, fontdict={
            "fontsize": 16,
            "fontweight": "medium"
        })

    plt.legend(loc="upper left", bbox_to_anchor=[1.01, 1.01])

  def plot(self, x: np.ndarray, labels: Sequence[str]):
    """Reduce given data x to 2D via t-SNE reduction and plot.

    Args:
      x: numpy array of data to reduce and plot.
      labels: string labels that correspond to x. Must have same length as x.

    Returns:
      numpy array of tsne reduced input x.
    """
    if len(x) != len(labels):
      raise ValueError("Must have a label for each input in x.")

    tsne = TSNE(n_components=2).fit_transform(x)

    tsne_data = pd.DataFrame({
        "x1": tsne[:, 0],
        "x2": tsne[:, 1],
        "labels": labels
    })

    axes = sns.scatterplot(
        x="x1",
        y="x2",
        data=tsne_data,
        hue="labels",
        alpha=0.75,
        legend=False,
        ax=self.ax)

    if _NUM_RANDOM_LABELS.value < 0:
      ids = list(range(tsne_data.shape[0]))
    else:
      ids = [
          random.randint(0, tsne_data.shape[0] - 1)
          for _ in range(_NUM_RANDOM_LABELS.value)
      ]
    for idx in ids:
      text = tsne_data.labels[idx]
      axes.text(
          tsne_data.x1[idx] + 0.01, tsne_data.x2[idx],
          text.split("_")[0],  # Remove POS.
          horizontalalignment="left",
          size="medium", color="black", weight="semibold"
      )

    return tsne

  def save_plot(self, save_to_path: str) -> None:
    """Save the plot as png.

    Args:
      save_to_path: path of file where to save a plot
    """
    with open(save_to_path, "wb") as f:
      self.fig.savefig(f, format="png", dpi=600, bbox_inches="tight")


def _path_to_title(path: str) -> str:
  """Converts filename to the title of the plot."""
  filename = os.path.basename(path)
  return os.path.splitext(filename)[0]


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  logging.info("Reading features from %s ...", _INPUT_FEATURES_NPZ_FILE.value)
  labels = []
  x = []
  with open(_INPUT_FEATURES_NPZ_FILE.value, "rb") as npz_file:
    data = np.load(npz_file)
    for key in data.keys():
      logging.info("Processing %s ...", key)
      features = data[key]
      if len(features.shape) != 2:
        raise ValueError(f"{key}: Expected two-dimension features!")
      logging.info("%s: shape: %s", key, features.shape)
      mean_features = np.mean(features, axis=0)
      labels.append(key)
      x.append(mean_features[np.newaxis, :])

  # Compute t-SNE.
  plot_title = _path_to_title(_INPUT_FEATURES_NPZ_FILE.value)
  tsne = TSNEReductionPlot(plot_title)
  tsne.plot(np.concatenate(x, axis=0), labels)
  logging.info("Saving plot to %s ...", _OUTPUT_PLOT_FILE.value)
  tsne.save_plot(_OUTPUT_PLOT_FILE.value)


if __name__ == "__main__":
  app.run(main)
