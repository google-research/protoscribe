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

"""Constructs an SVG `text` out of a set of SVGs of glyphs.

Allows for random rotations and scaling of glyphs.
"""

import copy
import random
import xml.etree.ElementTree as ET

from absl import flags
from protoscribe.glyphs import svg_simplify

import glob
import os
# Internal resources dependency

_RANDOM_RESIZE = flags.DEFINE_bool(
    "random_resize", False,
    "Randomly resize between 0.75 and 1."
)

_RANDOM_ROTATE = flags.DEFINE_bool(
    "random_rotate", False,
    "Randomly rotate glyphs between -10 and 10 degrees."
)

_RANDOM_PAD = flags.DEFINE_bool(
    "random_pad", False,
    "Randomly pad from 10 to 40 between glyphs."
)

_EXTRA_PAD = flags.DEFINE_integer(
    "extra_pad",
    20,
    "Put this much space at the bottom and right to make sure the glyphs don't "
    "get cut off."
)

_UNIFORM_HEIGHT = flags.DEFINE_integer(
    "uniform_height", 500,
    "Uniform height to scale everything to."
)

_VERTICAL_TRANSLATION = flags.DEFINE_integer(
    "vertical_translation",
    100,
    "Uniform vertical translation to avoid things getting cut off at the top",
)

_SIMPLIFY_SVG_TREES = flags.DEFINE_boolean(
    "simplify_svg_trees", False,
    "Converts the tree to equivalent representation by flattening all the "
    "paths by removing transforms and adjusting the viewbox to fit all the "
    "glyphs."
)

FLAGS = flags.FLAGS


def concat_xml_svgs(
    trees: list[ET.ElementTree],
    glyphs: list[str] | None = None,
    clone_trees: bool = False
) -> tuple[ET.ElementTree, float, float]:
  """Concatenates SVGs.

  Args:
    trees: A list of SVG element trees.
    glyphs: Names of glyphs. If provided, must be the same length as SVGs
    clone_trees: Perform deep copy on the input trees if the same tree
      instances are passed in repeatedly.

  Returns:
    Tuple of concatenated SVG, width, and height.
  """

  if glyphs:
    assert len(glyphs) == len(trees)
    glyphs = list(enumerate(glyphs))

  # Since the element trees are modified by this call, potentially reusing
  # thw `concat_xml_svg` calls will lead to unexpected results. Perform deep
  # copy of the passed trees.
  if clone_trees:
    trees = [copy.deepcopy(tree) for tree in trees]

  # Some of the SVGs lack `width` and `height attributes, in which
  # case we try the `viewBox`.
  def get_dimensions(svg_elt):
    try:
      w = float(svg_elt.attrib["width"].replace("px", ""))
      h = float(svg_elt.attrib["height"].replace("px", ""))
    except KeyError:
      viewbox = svg_elt.attrib["viewBox"].split()
      w = float(viewbox[-2])
      h = float(viewbox[-1])
    return w, h

  # TODO: I'm not entirely sure how many of these are necessary.
  def set_dimensions(svg_elt, w, h):
    svg_elt.attrib["width"] = f"{int(w)}px"
    svg_elt.attrib["height"] = f"{int(h)}px"
    viewbox = [e.strip() for e in svg_elt.attrib["viewBox"].split()]
    if len(viewbox) != 4:
      raise ValueError("Invalid viewbox: Expected 4 elements!")
    viewbox[-2] = str(w)
    viewbox[-1] = str(h)
    svg_elt.attrib["viewBox"] = " ".join(viewbox)

  def shift_scale_rotate(shift, scale, rotation):
    shift_scale_rotation = (
        f"translate({shift} {_VERTICAL_TRANSLATION.value}) "
        f"rotate({rotation}) scale({scale})"
    )
    return shift_scale_rotation

  def add_transform(elt, transform):
    if "transform" in elt.attrib:
      transform = f"{elt.attrib['transform']} {transform}"
    elt.attrib["transform"] = transform

  def random_pad() -> int:
    if not _RANDOM_PAD.value:
      return 0
    return random.randint(1, 5)

  def random_scale() -> float:
    if not _RANDOM_RESIZE.value:
      return 1
    return 0.75 + 0.25 * random.random()

  def random_rotation() -> float:
    if not _RANDOM_ROTATE.value:
      return 0
    return random.uniform(-10.0, 10.0)

  def regroup_elements_under_new_root(from_root, to_root, transform):
    g = ET.Element("g")
    add_transform(g, transform)
    for child in from_root:
      g.append(child)
    if to_root == from_root:
      for child in g:
        to_root.remove(child)
    to_root.append(g)

  def find_paths(root):
    prefix = "{%s}" % svg_simplify.XML_SVG_NAMESPACE
    path_types = [
        "path",
        "polygon",
        "line",
        "ellipse",
        "circle",
        "rect",
        "polyline",
    ]
    sub_paths = []
    for path_type in path_types:
      sub_paths.extend(root.findall(f".//{prefix}{path_type}"))
    return sub_paths

  def propagate_transforms_to_paths(root):
    for g in root:
      if "transform" in g.attrib:
        transform = g.attrib["transform"]
        for path in find_paths(g):
          add_transform(path, transform)
        del g.attrib["transform"]

  def add_glyph_to_paths(root, glyph):
    for path in find_paths(root):
      path.attrib[svg_simplify.XML_SVG_POSITION_AND_GLYPH] = (
          f"{glyph[0]},{glyph[1]}"
      )

  ET.register_namespace("", svg_simplify.XML_SVG_NAMESPACE)
  main_tree = trees[0]
  main_root = main_tree.getroot()
  # Scale everything to be uniform initially.
  uniform_height = _UNIFORM_HEIGHT.value
  max_height = uniform_height
  w, h = get_dimensions(main_root)
  scale = uniform_height / h * random_scale()
  w, h = w * scale, h * scale
  if h > max_height:
    max_height = h
  shift = 0
  for i in range(len(trees)):
    tree = trees[i]
    glyph = glyphs[i] if glyphs else None
    root = tree.getroot()
    w, h = get_dimensions(root)
    scale = uniform_height / h * random_scale()
    w, h = w * scale, h * scale
    if h > max_height:
      max_height = h
    rotation = random_rotation()
    transform = shift_scale_rotate(shift, scale, rotation)
    if glyph:
      add_glyph_to_paths(root, glyph)
    regroup_elements_under_new_root(root, main_root, transform)
    shift += w + random_pad()
  w = shift + _EXTRA_PAD.value
  h = max_height + _VERTICAL_TRANSLATION.value + _EXTRA_PAD.value
  set_dimensions(main_root, w, h)

  if _SIMPLIFY_SVG_TREES.value:
    main_tree, _, _ = svg_simplify.simplify_svg_tree(main_tree)
    w, h = get_dimensions(main_root)
  else:
    propagate_transforms_to_paths(main_tree.getroot())

  return main_tree, w, h


def concat_svgs(
    svgs: list[str], glyphs: list[str] | None = None,
) -> tuple[ET.ElementTree, float, float]:
  """Concatenates SVGs.

  Args:
    svgs: A list of svgs file paths.
    glyphs: Names of glyphs. If provided, must be the same length as svgs

  Returns:
    Tuple of concatenated SVG, width, and height.
  """
  ET.register_namespace("", svg_simplify.XML_SVG_NAMESPACE)

  trees = []
  for file_path in svgs:
    f = open(file_path, mode="rt")
    svg_tree = ET.parse(f)
    trees.append(svg_tree)
    f.close()

  return concat_xml_svgs(trees, glyphs=glyphs, clone_trees=False)
