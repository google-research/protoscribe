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

"""Translates offsets in the matrices output for paths by SketchPad.

For how the path matrices work, see

https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/transform
"""

import re

from absl import app
from absl import flags

import glob
import os

flags.DEFINE_integer("pad", 10, "Final padding from top left corner.")
flags.DEFINE_string("input", None, "Input SVG.")
flags.DEFINE_string("output", None, "Output SVG.")

FLAGS = flags.FLAGS


MAT = re.compile(r"matrix\([\d,\.]*\)")


def main(unused_argv):
  lines = []
  offsets = []

  def extract_offsets(expr):
    expr = expr.replace("matrix", "").replace("(", "").replace(")", "")
    expr = expr.split(",")
    return [float(c) for c in expr]

  with open(FLAGS.input) as stream:
    for line in stream:
      line = line.strip()
      lines.append(line)
      matched = MAT.search(line)
      if matched:
        offsets.append(extract_offsets(matched.group(0)))

  min_e = 1_000_000
  min_f = 1_000_000

  for offset in offsets:
    _, _, _, _, e, f = offset
    min_e = e if e < min_e else min_e
    min_f = f if f < min_f else min_f

  with open(FLAGS.output, "w") as stream:
    for line in lines:
      matched = MAT.search(line)
      if matched:
        a, b, c, d, e, f = extract_offsets(matched.group(0))
        e, f = e - min_e + FLAGS.pad, f - min_f + FLAGS.pad
        new_mat = f"matrix({a},{b},{c},{d},{e},{f})"
        line = line.replace(matched.group(0), new_mat)
      stream.write(f"{line}\n")


if __name__ == "__main__":
  flags.mark_flag_as_required("input")
  flags.mark_flag_as_required("output")
  app.run(main)
