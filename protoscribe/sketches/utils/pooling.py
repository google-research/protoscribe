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

"""Pooling utilities."""

import functools

import jax
import jax.numpy as jnp

JTensor = jnp.ndarray


@functools.partial(jax.vmap, in_axes=[0, 0], out_axes=0)
def _batch_gather(x: JTensor, idx: JTensor) -> JTensor:
  """Performs a batched gather of the data.

  Args:
    x: A [batch, num_in, ...] JTensor of data to gather from.
    idx: A [batch, num_out] JTensor of dtype int32 or int64 specifying which
      elements to gather. Every value is expected to be in the range of [0,
      num_in].

  Returns:
    A [batch, num_out, ...] JTensor of gathered data.
  """
  return x[idx]


def get_pooling(pooling: str, enc: JTensor, mask: JTensor) -> JTensor:
  """Pools embeddings.

  Args:
    pooling: Type of pooling to use.
    enc: Embeddings <float>[batch, len, dim].
    mask: Mask of inputs Union(<float>, <int>, <bool>)[batch, len].

  Returns:
    Pooled embeddings <float>[batch, dim].
  """
  if pooling == "max":
    return _max_pool(enc, mask)
  elif pooling == "mean":
    return _mean_pool(enc, mask)
  elif pooling == "first":
    return _first_pool(enc)
  elif pooling == "last":
    return _last_pool(enc, mask)
  else:
    raise ValueError(f"Unsupported pooling operation: {pooling}")


def _max_pool(enc: JTensor, mask: JTensor) -> JTensor:
  """Max pooling.

  Args:
    enc: Embeddings <float>[batch, len, dim].
    mask: Mask of inputs Union(<float>, <int>, <bool>)[batch, len].

  Returns:
    Pooled embeddings <float>[batch, dim].
  """
  mask = jnp.expand_dims(mask, -1)
  enc = jnp.where(mask, enc, jnp.nan)
  return jnp.nanmax(enc, axis=1)


def _mean_pool(enc: JTensor, mask: JTensor) -> JTensor:
  """Mean pooling.

  Args:
    enc: Embeddings <float>[batch, len, dim].
    mask: Mask of inputs Union(<float>, <int>, <bool>)[batch, len].

  Returns:
    Pooled embeddings <float>[batch, dim].
  """
  enc = enc * mask[:, :, None]
  enc = enc.sum(axis=1)
  return enc / mask.sum(axis=1)[:, None]


def _first_pool(enc: JTensor) -> JTensor:
  """Refurns first embedding.

  Args:
    enc: Embeddings <float>[batch, len, dim].

  Returns:
    First embedding <float>[batch, dim].
  """
  bsize, _, dim = enc.shape
  enc = enc[:bsize, :1, :dim]
  return jax.lax.squeeze(enc, [1])


def _last_pool(enc: JTensor, mask: JTensor) -> JTensor:
  """Returns last (before padding) embedding.

  Args:
    enc: Embeddings <float>[batch, len, dim].
    mask: Mask of inputs Union(<float>, <int>, <bool>)[batch, len].

  Returns:
    Pooled embeddings <float>[batch, dim].
  """
  # Compute the length of each sequence by counting the indicator tokens.
  lengths = jnp.sum(mask, axis=1, dtype=jnp.int32)
  # Find the position of the last token in each sequence.
  last_idx = jnp.asarray(jnp.maximum(lengths - 1, 0), dtype=jnp.int32)
  # Get the embeddings from the last token
  return _batch_gather(enc, last_idx)
