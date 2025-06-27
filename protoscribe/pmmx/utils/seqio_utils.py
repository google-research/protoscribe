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

"""General utility functions for t5x."""

import time
from typing import Any, Callable, Optional, Type

from absl import logging
from jax.experimental import multihost_utils
import numpy as np
import seqio
from t5x import utils
import tensorflow as tf


def get_dataset(
    cfg: utils.DatasetConfig,
    shard_id: int,
    num_shards: int,
    feature_converter_cls: Type[seqio.FeatureConverter],
    num_epochs: Optional[int] = None,
    num_seeds: int = 1,
    continue_from_last_checkpoint: bool = False,
    define_task_fn: Optional[Callable[[str], Any]] = None,
) -> tf.data.Dataset:
  """Returns a dataset from SeqIO based on a `DatasetConfig`.

  Use this in conjunction with preprocessors that are decorated with
  `@seqio.map_over_dataset` to create an augmented dataset, where each pass over
  the data will use a new random seed from a cycle of size `num_seeds`.
  If you are using `seqio.CacheDatasetPlaceholder`, your nondeterministic
  preprocessors should come after that in the preprocessor list.

  Set `num_seeds` to the number of distinct random seeds to use. To avoid
  training on the same `(example, seed)` pair twice, set the number of
  random seeds to the number of passes that will be made over the dataset. For
  example, if your dataset has 100K batches worth of data and you are training
  for 1M steps, `num_seeds` should be at least 10 to avoid training on the
  same pair twice.

  Args:
    cfg: seqio.DatasetConfig
    shard_id: int. data parallelism shard id
    num_shards: int. data parallelism num shards
    feature_converter_cls: class of the `seqio.FeatureConverter` to use
    num_epochs: int. number of epochs over the augmented dataset, where the
      the augmented dataset is `num_seeds` times bigger than the original. it is
      recommended to set this to None, so that the dataset will be repeated
      until the end of training (usually governed by `train.total_steps`)
    num_seeds: int. number of different seeds to use
    continue_from_last_checkpoint: bool. whether to resume from checkpoint
    define_task_fn: Callable. If task not found in registry, calls this function
      to define the task.

  Returns:
    tf.data.Dataset
  """
  if continue_from_last_checkpoint:
    raise ValueError(
        '`continue_from_last_checkpoint` must be set to False as this is not '
        'supported by this dataset fn.')
  del continue_from_last_checkpoint

  if cfg.module:
    utils.import_module(cfg.module)

  if cfg.batch_size % num_shards:
    raise ValueError(
        f'Batch size ({cfg.batch_size}) must be divisible by number of '
        f'shards ({num_shards}).')

  if define_task_fn:
    tasks = seqio.TaskRegistry.names()
    mixtures = seqio.MixtureRegistry.names()
    if (cfg.mixture_or_task_name not in tasks) and (
        cfg.mixture_or_task_name not in mixtures):
      define_task_fn(cfg.mixture_or_task_name)

  if isinstance(cfg.mixture_or_task_name, seqio.DatasetProviderBase):
    mixture_or_task = cfg.mixture_or_task_name
  else:
    mixture_or_task = seqio.get_mixture_or_task(cfg.mixture_or_task_name)

  shard_info = seqio.ShardInfo(index=shard_id, num_shards=num_shards)

  if cfg.seed is None:
    # Use a shared timestamp across devices as the seed.
    seed = multihost_utils.broadcast_one_to_all(np.int32(time.time()))
  else:
    seed = cfg.seed

  if num_seeds is None or num_seeds < 1:
    raise ValueError(
        '`num_seeds` is required for this version of `get_dataset`. Either '
        'set `num_seeds` or switch to `t5x.utils.get_dataset`.')

  if num_seeds > 1 and isinstance(mixture_or_task, seqio.Mixture):
    raise ValueError(
        'Due to a bug, data augmentation is disabled when using Mixtures. '
        'Please set seqio_utils.get_dataset.num_seeds==1 until we address '
        'b/232013238. Note that setting num_seeds to 1 effectively disables '
        'nondeterministic data augmentation, such as random image cropping and '
        'flipping.')

  if num_seeds == 1:
    logging.info('Falling back to default seqio behavior since num_seeds==1')
    return utils.get_dataset_inner(
        cfg, shard_info, feature_converter_cls, seed=0, num_epochs=num_epochs)

  ds = None
  for i in range(num_seeds):
    seed_i = seed + i
    logging.info('Setting up epoch=%d with seed=%s', i, seed_i)
    next_ds = utils.get_dataset_inner(
        cfg, shard_info, feature_converter_cls, seed=seed_i, num_epochs=1)
    if ds is None:
      ds = next_ds
    else:
      ds = ds.concatenate(next_ds)

  assert ds is not None
  ds: tf.data.Dataset
  return ds.repeat(num_epochs)  # Repeat over the augmented dataset.
