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

"""Computes shortest distance path for a pair of sequences."""

from __future__ import annotations

from typing import Any, Callable


class Cell(object):
  """Matrix cell."""

  def __init__(
      self,
      cost: float,
      back_pointer: Cell | None,
      elt1: str | None,
      elt2: str | None
  ) -> None:
    self.cost_ = cost
    self.back_pointer_ = back_pointer
    self.elt1_ = elt1
    self.elt2_ = elt2

  @property
  def cost(self) -> float:
    return self.cost_

  @property
  def back_pointer(self) -> Cell | None:
    return self.back_pointer_

  @property
  def pair(self) -> tuple[str | None, str | None]:
    return (self.elt1_, self.elt2_)


def default_ins(unused_item: Any) -> float:
  return 1.


def default_del(unused_item: Any) -> float:
  return 1.


def default_sub(item1: Any, item2: Any) -> float:
  if item1 == item2:
    return 0.
  return 1.


class Matrix(object):
  """Matrix for dynamic programming computation and storing back pointers."""

  def __init__(
      self,
      seq1: str | list[str],
      seq2: str | list[str],
      insert: Callable[[Any], float] = default_ins,
      delete: Callable[[Any], float] = default_del,
      substitute: Callable[[Any, Any], float] = default_sub,
  ):
    if isinstance(seq1, str):
      seq1 = seq1.split()
    if isinstance(seq2, str):
      seq2 = seq2.split()
    seq1 = [None] + list(seq1)
    seq2 = [None] + list(seq2)
    self.max1_ = len(seq1)
    self.max2_ = len(seq2)
    self.data_ = {}
    self.data_[0, 0] = Cell(0, None, None, None)
    cum = 0
    for i in range(1, self.max1_):
      cum += insert(seq1[i])
      self.data_[i, 0] = Cell(cum, self.data_[i - 1, 0], seq1[i], None)
    cum = 0
    for i in range(1, self.max2_):
      cum += delete(seq2[i])
      self.data_[0, i] = Cell(cum, self.data_[0, i - 1], None, seq2[i])
    for i in range(1, self.max1_):
      for j in range(1, self.max2_):
        l1el = seq1[i]
        l2el = seq2[j]
        c1 = self.data_[i, j - 1].cost + insert(l1el)
        c2 = self.data_[i - 1, j].cost + delete(l2el)
        c3 = self.data_[i - 1, j - 1].cost + substitute(l1el, l2el)
        if c1 <= c2 and c1 <= c3:
          self.data_[i, j] = Cell(c1, self.data_[i, j - 1], None, l2el)
        elif c2 <= c1 and c2 <= c3:
          self.data_[i, j] = Cell(c2, self.data_[i - 1, j], l1el, None)
        else:
          self.data_[i, j] = Cell(c3, self.data_[i - 1, j - 1], l1el, l2el)
    c = self.data_[self.max1_ - 1, self.max2_ - 1]
    path = []
    while c:
      if c.pair != (None, None):
        path.append(c.pair)
      c = c.back_pointer
    path.reverse()
    self._path = path

  @property
  def path(self) -> list[tuple[str | None, str | None]]:
    return self._path

  @property
  def cost(self) -> float:
    return self.data_[self.max1_ - 1, self.max2_ - 1].cost_


def best_match(
    seq1: str,
    seq2: str,
    insert: Callable[[Any], float] = default_ins,
    delete: Callable[[Any], float] = default_del,
    substitute: Callable[[Any, Any], float] = default_sub
) -> tuple[float, list[tuple[str | None, str | None]]]:
  """Computes best match for seq1, seq2.

  Args:
    seq1: A space-delimited string or list.
    seq2: A space-delimited string or list.
    insert: Insertion cost function.
    delete: Deletion cost function.
    substitute: Substitution cost function.

  Returns:
    A pair of a cost and a path
  """
  m = Matrix(seq1, seq2, insert=insert, delete=delete, substitute=substitute)
  return m.cost, m.path
