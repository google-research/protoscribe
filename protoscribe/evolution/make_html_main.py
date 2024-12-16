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

r"""Constructs an HTML page to view the proposed spellings with glyph images.

See `make_html.py` for the definitions of relevant flags.

Example:
--------
  EXPERIMENT_DIR=...
  python protoscribe/evolution/make_html_main.py \
    --extensions_file=${EXPERIMENT_DIR}/inference_extensions/extensions.tsv \
    --svg_src_dir=${EXPERIMENT_DIR}/extensions_svg \
    --output_html_dir=/tmp/html \
    --logtostderr
"""

from absl import app
from protoscribe.evolution import make_html


def main(unused_argv):
  make_html.make_html()


if __name__ == "__main__":
  app.run(main)
