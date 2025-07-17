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

"""Training and evaluation loop implementation.

This implementation is based on the WMT example that ships with Flax.
"""

import dataclasses
import functools
from typing import Callable

from absl import flags
from absl import logging
from clu import metric_writers
from clu import periodic_actions
import flax
from flax import jax_utils
from flax import linen as nn
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import orbax.checkpoint as ocp
from protoscribe.models.flax import input_pipeline
from protoscribe.models.flax import utils
from protoscribe.models.flax import vanilla_network
from protoscribe.models.flax import variational_network
from protoscribe.sketches.utils import continuous_bernoulli
from protoscribe.sketches.utils import latents
import tensorflow as tf

FLAGS = flags.FLAGS

Array = flax.typing.Array
PRNGKey = flax.typing.PRNGKey
TrainState = train_state.TrainState
ModelConfig = (
    vanilla_network.TransformerConfig | variational_network.TransformerConfig
)
Model = (
    vanilla_network.Transformer | variational_network.VariationalTransformer
)


def _get_model(
    backend_type: str, config: ModelConfig
) -> Model:
  """Returns an instance of transformer model given the config.

  Args:
     backend_type: Type of the model (string).
     config: Model configuration.

  Returns:
     A model which may be of several supported model types.
  """
  if backend_type == "vanilla":
    model = vanilla_network.Transformer(config)
  elif backend_type == "variational":
    model = variational_network.VariationalTransformer(config)
  else:
    raise ValueError(f"Unsupported backend: {backend_type}")

  return model


def _create_optimizer(
    config: ml_collections.ConfigDict, learning_rate_fn
) -> optax.GradientTransformation:
  """Creates optimizer gradient transformation."""
  if config.optimizer.name == "adam":
    return optax.adamw(
        learning_rate=learning_rate_fn,
        b1=0.9,
        b2=0.98,
        eps=1e-9,
        weight_decay=config.optimizer.weight_decay,
    )
  if config.optimizer.name == "adafactor":
    return optax.adafactor(
        learning_rate=learning_rate_fn,
        weight_decay_rate=config.optimizer.weight_decay,
        # Defaults to True
        # Setting to False based on
        # https://discuss.huggingface.co/t/t5-finetuning-tips/684/3
        # Tested that removing the scaling helps with t2im models.
        multiply_by_parameter_scale=False,
    )
  raise ValueError(f"Unknown optimizer: {config.optimizer.name}")


def _rsqrt_schedule(
    init_value: float, shift: int = 0,
) -> Callable[[int], float]:
  """Applies a reverse square-root schedule.

  The reverse square root schedule is simply `lr = init_value / sqrt(step)`.

  Args:
    init_value: Base learning rate (before applying the rsqrt schedule).
    shift: How many steps the rsqrt should be shifted. Shifting the rsqrt
      schedule makes it less steep in the beginning (close to 0).

  Returns:
    A schedule `count -> learning_rate`.
  """

  def schedule(count: int) -> float:
    return init_value * (count + shift) ** -0.5 * shift**0.5

  return schedule


def _create_learning_rate_schedule(learning_rate: float, warmup_steps: int):
  """Creates a rsqrt schedule with linear warmup."""
  return optax.join_schedules(
      [
          optax.linear_schedule(
              init_value=0,
              end_value=learning_rate,
              transition_steps=warmup_steps,
          ),
          _rsqrt_schedule(init_value=learning_rate, shift=warmup_steps),
      ],
      boundaries=[warmup_steps],
  )


def _compute_weighted_cross_entropy(
    logits: Array,
    targets: Array,
    weights: Array | None = None,
    label_smoothing: float = 0.0
) -> tuple[Array, Array]:
  """Compute weighted cross entropy and entropy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch, length].
   label_smoothing: label smoothing constant, used to determine the on and off
     values.

  Returns:
    Tuple of scalar loss and batch normalizing factor.

  Raises:
    Value error for incorrect shapes.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError(
        "Incorrect shapes. Got shape %s logits and %s targets"
        % (str(logits.shape), str(targets.shape))
    )

  vocab_size = logits.shape[-1]
  confidence = 1.0 - label_smoothing
  low_confidence = (1.0 - confidence) / (vocab_size - 1)
  normalizing_constant = -(
      confidence * jnp.log(confidence)
      + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
  )
  soft_targets = common_utils.onehot(
      targets, vocab_size, on_value=confidence, off_value=low_confidence
  )

  loss = -jnp.sum(soft_targets * nn.log_softmax(logits), axis=-1)
  loss = loss - normalizing_constant

  normalizing_factor = np.prod(targets.shape)
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


def _compute_weighted_accuracy(
    logits: Array,
    targets: Array,
    weights: Array | None = None
) -> tuple[Array, Array]:
  """Compute weighted accuracy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch, length]

  Returns:
    Tuple of scalar loss and batch normalizing factor.

  Raises:
    ValueError for incompatible shapes.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError(
        "Incorrect shapes. Got shape %s logits and %s targets"
        % (str(logits.shape), str(targets.shape))
    )
  loss = jnp.equal(jnp.argmax(logits, axis=-1), targets)
  normalizing_factor = np.prod(logits.shape[:-1])
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


def _cb_log_constant_from_logits(
    logits: Array, weights: Array | None = None
) -> Array:
  """Continuous Bernoulli normalizing factor.

  Computes log(C(lambda)) for the Continuous Bernoulli distribution's
  normalizing constant, where lambda is derived from input logits.

  Args:
    logits: Floating point array of dimension (B, L, D).
    weights: None or array of shape [batch, length]

  Returns:
    Normalizing constant and normalizing factor.
  """
  lambdas = jax.nn.sigmoid(logits)
  lambdas = continuous_bernoulli.clamp_probs(lambdas)
  cb_const = -continuous_bernoulli.cb_log_norm_const(lambdas)

  normalizing_factor = np.prod(logits.shape[:-1])
  if weights is not None:
    cb_const = cb_const * weights[..., None]
    normalizing_factor = weights.sum()

  return jnp.sum(cb_const), normalizing_factor


def _total_loss(
    model_outputs: dict[str, Array],
    targets: Array,
    weights: Array,
    label_smoothing: float = 0.0,
    kl_weight: float = 1.,
    kl_multiplier: float = 1.,
    cb_log_norm_const: bool = False
) -> Array:
  """Computes total loss."""
  logits = model_outputs["logits"]

  ce_loss, ce_weight_sum = _compute_weighted_cross_entropy(
      logits, targets, weights, label_smoothing
  )
  kld_loss = 0.
  if "z_mean" in model_outputs:
    kld_loss, _ = latents.kl_regularization_loss(
        model_outputs["z_mean"], model_outputs["z_log_var"]
    )
    kld_loss = kld_loss * kl_weight * kl_multiplier

  cb_const, cb_norm_factor = (
      _cb_log_constant_from_logits(logits, weights=weights)
      if cb_log_norm_const and "z_mean" in model_outputs
      else (0., 1.)
  )
  loss = (
      ce_loss / ce_weight_sum + kld_loss + cb_const / cb_norm_factor
  )
  return loss


def _compute_metrics(
    model_outputs: dict[str, Array],
    labels: Array,
    weights: Array,
    label_smoothing: float = 0.0,
    kl_weight: float = 1.,
    kl_multiplier: float = 1.,
    cb_log_norm_const: bool = False
) -> dict[str, Array]:
  """Computes summary metrics."""
  total_loss = _total_loss(
      model_outputs,
      labels,
      weights,
      label_smoothing=label_smoothing,
      kl_weight=kl_weight,
      kl_multiplier=kl_multiplier,
      cb_log_norm_const=cb_log_norm_const
  )

  logits = model_outputs["logits"]
  ce_loss, ce_weight_sum = _compute_weighted_cross_entropy(
      logits, labels, weights, label_smoothing
  )
  acc, _ = _compute_weighted_accuracy(logits, labels, weights)
  metrics = {
      "total_loss": total_loss,
      "ce_loss": ce_loss,
      "accuracy": acc,
      "num_valid_tokens": ce_weight_sum,
  }
  if "z_mean" in model_outputs:
    metrics["kld"], _ = latents.kl_regularization_loss(
        model_outputs["z_mean"], model_outputs["z_log_var"]
    )
  metrics = jax.lax.psum(metrics, axis_name="batch")
  return metrics


# Primary training and evaluation step functions.
# -----------------------------------------------------------------------------


def _train_step(
    state: TrainState,
    batch: dict[str, Array],
    config: ModelConfig,
    backend_type: str,
    learning_rate_fn,
    label_smoothing: float = 0.,
    dropout_rng: PRNGKey | None = None,
    kl_weight: float = 1.,
    kl_multiplier: float = 1.,
    cb_log_norm_const: bool = False
) -> tuple[TrainState, dict[str, Array]]:
  """Perform a single training step."""
  targets = batch["targets"]
  weights = jnp.where(targets > 0, 1, 0).astype(jnp.float32)
  dropout_rng = jax.random.fold_in(dropout_rng, state.step)
  dropout_rng, latents_rng = jax.random.split(dropout_rng, 2)

  def loss_fn(params):
    """loss function used for training."""
    model_outputs = _get_model(backend_type, config).apply(
        {"params": params},
        features=batch,
        rngs={
            "dropout": dropout_rng,
            "latents": latents_rng,
        },
    )
    loss = _total_loss(
        model_outputs,
        targets,
        weights,
        label_smoothing=label_smoothing,
        kl_weight=kl_weight,
        kl_multiplier=kl_multiplier,
        cb_log_norm_const=cb_log_norm_const
    )
    return loss, model_outputs

  step = state.step

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, model_outputs), grads = grad_fn(state.params)
  grads = jax.lax.pmean(grads, axis_name="batch")
  new_state = state.apply_gradients(grads=grads)

  metrics = _compute_metrics(
      model_outputs,
      targets,
      weights,
      label_smoothing=label_smoothing,
      kl_weight=kl_weight,
      kl_multiplier=kl_multiplier,
      cb_log_norm_const=cb_log_norm_const
  )
  metrics["learning_rate"] = learning_rate_fn(step)
  if "kld" in metrics:
    metrics["kld_weight"] = kl_weight

  return new_state, metrics


def _eval_step(
    params: dict[str, Array],
    batch: dict[str, Array],
    config: ModelConfig,
    backend_type: str,
    label_smoothing: float = 0.0,
    kl_weight: float = 1.,
    kl_multiplier: float = 1.,
    cb_log_norm_const: bool = False
) -> dict[str, Array]:
  """Calculates evaluation metrics on a batch."""
  targets = batch["targets"]
  weights = jnp.where(targets > 0, 1.0, 0.0)
  model_outputs = _get_model(backend_type, config).apply(
      {"params": params}, features=batch
  )
  return _compute_metrics(
      model_outputs,
      targets,
      weights,
      label_smoothing=label_smoothing,
      kl_weight=kl_weight,
      kl_multiplier=kl_multiplier,
      cb_log_norm_const=cb_log_norm_const
  )


def evaluate(
    *, p_eval_step, params, eval_ds: tf.data.Dataset, num_eval_steps: int
):
  """Evaluates the params an return a dictionary with the metrics."""
  logging.info("Gathering evaluation metrics.")
  eval_metrics = []
  eval_iter = iter(eval_ds)
  for _, eval_batch in zip(range(num_eval_steps), eval_iter):
    eval_batch = jax.tree_util.tree_map(lambda x: x._numpy(), eval_batch)  # pylint: disable=protected-access
    eval_batch = common_utils.shard(eval_batch)
    metrics = p_eval_step(params, eval_batch)
    eval_metrics.append(metrics)
  eval_metrics = common_utils.get_metrics(eval_metrics)
  eval_metrics_sums = jax.tree_util.tree_map(jnp.sum, eval_metrics)
  num_valid_tokens = eval_metrics_sums.pop("num_valid_tokens")
  eval_summary = jax.tree_util.tree_map(
      lambda x: x / num_valid_tokens,  # pylint: disable=cell-var-from-loop
      eval_metrics_sums,
  )
  return eval_summary


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str) -> None:
  """Runs a training and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  tf.io.gfile.makedirs(workdir)

  # Load Datasets
  # ---------------------------------------------------------------------------
  logging.info("Initializing dataset.")
  if not FLAGS.dataset_dir:
    raise ValueError("Specify dataset directory --dataset_dir!")
  train_ds, eval_ds = input_pipeline.get_train_and_eval_datasets(
      num_devices=jax.local_device_count(),
      config=config,
      dataset_dir=FLAGS.dataset_dir
  )

  train_iter = iter(train_ds)

  logging.info("Initializing model, optimizer, and step functions.")

  # Build Model and Optimizer
  # ---------------------------------------------------------------------------
  if config.backend_type == "vanilla":
    train_config = vanilla_network.get_config(config, utils.RunType.TRAIN)
    eval_config = vanilla_network.get_config(config, utils.RunType.EVAL)
  elif config.backend_type == "variational":
    train_config = variational_network.get_config(config, utils.RunType.TRAIN)
    eval_config = variational_network.get_config(config, utils.RunType.EVAL)
  else:
    raise ValueError(f"Unknown backend: {config.backend_type}")

  start_step = 0
  rng = jax.random.key(config.seed)
  rng, init_rng = jax.random.split(rng)
  first_batch = jax.tree_util.tree_map(np.asarray, next(train_iter))
  all_shapes = dict(
      [(name, first_batch[name].shape) for name in first_batch.keys()]
  )
  logging.info("Shapes: %s", all_shapes)

  model = _get_model(config.backend_type, eval_config)
  initial_variables = jax.jit(model.init)(init_rng, features=first_batch)

  # Create train state with an optimizer and weight decay.
  learning_rate_fn = _create_learning_rate_schedule(
      learning_rate=config.optimizer.learning_rate,
      warmup_steps=config.optimizer.warmup_steps
  )
  state = TrainState.create(
      apply_fn=model.apply,
      params=initial_variables["params"],
      tx=_create_optimizer(config=config, learning_rate_fn=learning_rate_fn),
  )

  # We access model params only via state.params
  del initial_variables

  if config.restore_checkpoints:
    # Restore unreplicated optimizer + model state from last checkpoint.
    state = checkpoints.restore_checkpoint(workdir, state)
    # Grab last step.
    start_step = int(state.step)

  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0
  )
  if start_step == 0:
    writer.write_hparams(dict(config))

  # Replicate state.
  state = jax_utils.replicate(state)

  # compile multidevice versions of train/eval step and cache init fn.
  p_train_step = jax.pmap(
      functools.partial(
          _train_step,
          config=train_config,
          backend_type=config.backend_type,
          learning_rate_fn=learning_rate_fn,
          label_smoothing=config.label_smoothing,
          kl_multiplier=config.latents.kl_multiplier,
          cb_log_norm_const=config.latents.continuous_bernoulli_log_norm_const
      ),
      axis_name="batch",
      donate_argnums=(0,),
  )
  p_eval_step = jax.pmap(
      functools.partial(
          _eval_step,
          config=eval_config,
          backend_type=config.backend_type,
      ),
      axis_name="batch"
  )

  # Main Train Loop
  # ---------------------------------------------------------------------------

  # We init the first set of dropout PRNG keys, but update it afterwards inside
  # the main pmap"d training update for performance.
  dropout_rngs = jax.random.split(rng, jax.local_device_count())
  del rng

  # Annealer for \beta parameter, which is KL multiplier from the original
  # beta-VAE architecture (Higgins et al. "beta-VAE: Learning Basic Visual
  # Concepts with a Constrained Variational Framework").
  kl_annealer = latents.KLAnnealing(
      annealing_type=config.latents.kl_annealing,
      total_num_steps=(
          config.num_train_steps if not config.latents.kl_cyclical
          else config.num_train_steps // config.latents.kl_num_cycles
      ),
      cyclical=config.latents.kl_cyclical
  )
  logging.info("KL annealer: %s", dataclasses.asdict(kl_annealer))

  logging.info("Starting training loop.")
  hooks = []
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=config.num_train_steps, writer=writer
  )
  if jax.process_index() == 0:
    hooks += [
        report_progress,
        periodic_actions.Profile(logdir=workdir, num_profile_steps=5),
    ]
  train_metrics = []
  with metric_writers.ensure_flushes(writer):
    for step in range(start_step, config.num_train_steps):
      is_last_step = step == config.num_train_steps - 1

      # Shard data to devices and do a training step.
      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        batch = next(train_iter)
        kl_weight = np.array([kl_annealer.step()] * jax.local_device_count())
        batch = common_utils.shard(jax.tree_util.tree_map(np.asarray, batch))
        state, metrics = p_train_step(
            state,
            batch,
            dropout_rng=dropout_rngs,
            kl_weight=kl_weight
        )
        train_metrics.append(metrics)

      # Quick indication that training is happening.
      logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)
      for h in hooks:
        h(step)

      # Periodic metric handling.
      if step % config.eval_every_steps == 0 or is_last_step:
        with report_progress.timed("training_metrics"):
          logging.info("Gathering training metrics.")
          train_metrics = common_utils.get_metrics(train_metrics)
          lr = train_metrics.pop("learning_rate").mean()
          if "kld_weight" in train_metrics:
            kld_weight = train_metrics.pop("kld_weight").mean()
          metrics_sums = jax.tree_util.tree_map(jnp.sum, train_metrics)
          num_valid_tokens = metrics_sums.pop("num_valid_tokens")
          summary = jax.tree_util.tree_map(
              lambda x: x / num_valid_tokens,  # pylint: disable=cell-var-from-loop
              metrics_sums
          )
          summary["learning_rate"] = lr
          if "kld" in train_metrics:
            summary["kld_weight"] = kld_weight
          summary = {"train_" + k: v for k, v in summary.items()}
          writer.write_scalars(step, summary)
          train_metrics = []

        with report_progress.timed("eval"):
          eval_results = evaluate(
              p_eval_step=p_eval_step,
              params=state.params,
              eval_ds=eval_ds,
              num_eval_steps=config.num_eval_steps,
          )
          writer.write_scalars(
              step, {"eval_" + k: v for k, v in eval_results.items()}
          )

      # Save a checkpoint on one host after every checkpoint_freq steps.
      save_checkpoint = (
          step % config.checkpoint_every_steps == 0 or is_last_step
      )
      if config.save_checkpoints and save_checkpoint:
        logging.info("Saving checkpoint step %d.", step)

        # Orbax can not handle host local arrays from pmap.
        replicated_state = jax.tree_util.tree_map(
            ocp.utils.fully_replicated_host_local_array_to_global_array,
            state,
        )
        with report_progress.timed("checkpoint"):
          checkpoints.save_checkpoint_multiprocess(
              workdir, replicated_state, step
          )
