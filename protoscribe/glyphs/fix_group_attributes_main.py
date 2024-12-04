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

"""Moves attributes on groups to contained paths.

The package `svgpathtools` does not deal with attributes that are not on paths.
"""

import  xml.etree.ElementTree as ET

from absl import app
from absl import flags

_INPUT_SVG = flags.DEFINE_string(
    "input_svg", None,
    "Input SVG file.",
    required=True
)
_OUTPUT_SVG = flags.DEFINE_string(
    "output_svg", None,
    "Path to an output SVG.",
    required=True
)


def _copy_group_attributes(
    node: ET.Element, namespace: str, transform: str = ""
) -> None:
  """Copies the group attributes to paths.

  Also, if the group has "visibility: hidden" in its style, remove it.

  Args:
    node: a tree node
    namespace: A namespace to remove
    transform: accumulated transform
  """
  tag = node.tag.replace(namespace, "")
  if tag == "path":
    node_transform = (
        node.attrib["transform"] if "transform" in node.attrib else ""
    )
    node_transform = f"{node_transform} {transform}".strip()
    node.attrib["transform"] = node_transform
    return
  if tag == "g":
    if "style" in node.attrib and "visibility: hidden" in node.attrib["style"]:
      for child in node:
        node.remove(child)
    elif "transform" in node.attrib:
      transform = f"{transform} {node.attrib['transform']}".strip()
  for child in node:
    _copy_group_attributes(child, namespace, transform)


def copy_group_attributes(svg_tree: ET.ElementTree, namespace: str) -> None:
  """Copy the group attributes to paths.

  Args:
    svg_tree: an SVG tree
    namespace: A namespace to remove because XML is ...
  """
  root = svg_tree.getroot()
  _copy_group_attributes(root, f"{{{namespace}}}")


def main(unused_argv):
  ns = "http://www.w3.org/2000/svg"
  ET.register_namespace("", ns)
  svg_tree = ET.parse(_INPUT_SVG.value)
  copy_group_attributes(svg_tree, ns)
  svg_tree.write(_OUTPUT_SVG.value)


if __name__ == "__main__":
  app.run(main)
