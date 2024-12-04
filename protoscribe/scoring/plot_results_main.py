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

r"""Plot TSVs produced by `glyph_infer.sh`.

Please note: There is a reason why we don't use internal SeaBorn. It is not
the latest version and does not display the results as nicely as the latest
version. Please install the latest seaborn and pandas in a local Python
environment with `pip install seaborn --upgrade`.

Example:
--------
  python protoscribe/scoring/plot_results_main.py \
    --input_tsv_file /tmp/results/full_unseen.tsv \
    --output_plot_file /tmp/results/full_unseen.png \
    --title "All Unseen (8990 examples)" \
    --logtostderr
"""

import logging
from typing import Sequence

from absl import app
from absl import flags
import pandas as pd
import seaborn as sns

_INPUT_TSV_FILE = flags.DEFINE_string(
    "input_tsv_file", None,
    "Input TSV file with the results.")

_OUTPUT_PLOT_FILE = flags.DEFINE_string(
    "output_plot_file", None,
    "Output plot file name.")

_TITLE = flags.DEFINE_string(
    "title", None,
    "Title of the splot.")


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  logging.info("Reading results from %s ...", _INPUT_TSV_FILE.value)
  df = pd.read_csv(_INPUT_TSV_FILE.value, sep="\t")
  systems = sorted(set(df["Name"].tolist()))

  logging.info("Plotting ...")
  plot = sns.catplot(data=df, y="Name", x="Mean", kind="bar",
                     col="Type", hue="Name", legend=False,
                     order=systems, sharex=False)
  for ax in plot.axes.flat:
    ax.grid(True, axis="both")
    ax.set_axisbelow(True)
  if _TITLE.value:
    plot.fig.suptitle(_TITLE.value)
    plot.fig.tight_layout()

  logging.info("Saving %s ...", _OUTPUT_PLOT_FILE.value)
  plot.savefig(_OUTPUT_PLOT_FILE.value, dpi=800)


if __name__ == "__main__":
  flags.mark_flag_as_required("input_tsv_file")
  flags.mark_flag_as_required("output_plot_file")
  app.run(main)
