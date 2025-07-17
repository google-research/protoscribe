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

"""Inference loop.

Note: There are some uses of Numpy APIs below. These are used to handle batches
that include pass-through features some of which are strings which JAX does not
support.
"""

import functools
import json
import os
from typing import Any

from absl import flags
from absl import logging
import flax
from flax import jax_utils
from flax.training import checkpoints
from flax.training import common_utils
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import ml_collections
import numpy as np
from protoscribe.models.flax import decoding
from protoscribe.models.flax import input_pipeline
from protoscribe.models.flax import utils
from protoscribe.models.flax import vanilla_network
from protoscribe.models.flax import variational_network
from protoscribe.sketches.utils import stroke_tokenizer
import tensorflow as tf

Array = flax.typing.Array
PRNGKey = flax.typing.PRNGKey

ModelConfig = (
    vanilla_network.TransformerConfig | variational_network.TransformerConfig
)
Model = (
    vanilla_network.Transformer | variational_network.VariationalTransformer
)

# End-of-sketch token.
_EOS_ID = stroke_tokenizer.Token.END_OF_SKETCH

FLAGS = flags.FLAGS


def _to_jax(
    batch: dict[str, tf.Tensor],
) -> dict[str, Array]:
  """Converts mapping of TF tensors to JAX arrays."""
  return jax.tree_util.tree_map(lambda x: x._numpy(), batch)  # pylint: disable=protected-access


def _get_model(
    backend_type: str, config: ModelConfig
) -> Model:
  """Returns an instance of transformer model given the config."""
  if backend_type == "vanilla":
    return vanilla_network.Transformer(config)
  elif backend_type == "variational":
    return variational_network.VariationalTransformer(config)
  else:
    raise ValueError(f"Unsupported backend: {backend_type}")


def _pad_examples(x: Array, desired_batch_size: int) -> Array:
  """Expands batch to desired size by repeating last slice."""
  batch_pad = desired_batch_size - x.shape[0]
  return jnp.concatenate([x, jnp.tile(x[-1], (batch_pad, 1))], axis=0)


def _to_host(x: Array) -> Array:
  """Collect batches from all devices to host and flatten batch dimensions."""
  n_device, n_batch, *remaining_dims = x.shape
  return np.array(x).reshape((n_device * n_batch,) + tuple(remaining_dims))


def _initialize_cache(
    params: dict[str, Array],
    inputs: Array,
    targets: Array,
    backend_type: str,
    model_config: ModelConfig,
) -> dict[str, Array]:
  """Initializes a cache for a given input shape and max decode length.

  Args:
    params: Model parameters.
    inputs: Input features.
    targets: Target features.
    backend_type: Type of the model.
    model_config: Model-specific configuration dictionary.

  Returns:
    Initial cache parameters.
  """
  model = _get_model(backend_type, model_config)
  _, initial_variables = model.apply(
      {
          "params": params
      },
      features=dict(
          inputs=jnp.ones_like(inputs),
          targets=jnp.ones_like(targets)
      ),
      mutable=["cache"]
  )
  return initial_variables["cache"]


def _prediction_step(
    config: ml_collections.ConfigDict,
    model_config: ModelConfig,
    inputs: Array,
    params: dict[str, Array],
    cache: dict[str, Array],
    device_rngs: PRNGKey,
) -> tuple[Array, Array]:
  """Predicts sketch tokens given a batch of features with fast decoding.

  Args:
    config: Global configuration dictionary.
    model_config: Model configuration dictionary.
    inputs: Input features array. These are multimodal embeddings
      (semantics/speech/vision and so on).
    params: Dictionary of arrays of model parameters.
    cache: Decoding cache.
    device_rngs: Random number generator keys.

  Returns:
    Tuple consisting of:
      - Array [n_batch, n_best, n_length] of `n_best` hypotheses sorted in
        increasing order of log-probability.
      - Array of log-probability scores for hypotheses above.

  Raises:
    ValueError if decoding algorithm is not supported.
  """

  # Fetch global configuration options.
  backend_type = config.backend_type
  num_hypotheses = config.decoder.num_hypotheses
  max_decode_len = config.decoder.max_predict_length

  # Prepare transformer fast-decoder call for decoding (beam search or
  # stochastic sampling): for search, we need to set up our decoder model
  # to handle a batch size equal to batch_size * num_hypotheses, where
  # each batch item's data is expanded in-place rather than tiled, i.e., if
  # we denote each batch element subtensor as el[n]:
  # [el0, el1, el2] --> num_hypotheses=2 --> [el0,el0,el1,el1,el2,el2]
  model = _get_model(backend_type, model_config)
  encoded_inputs = model.apply(
      {"params": params},
      inputs,
      method=model.encode
  )
  if backend_type == "variational":
    # Ignore dummy posterior parameters mean(z) and log_var(z).
    z_mean, _, encoded_inputs = encoded_inputs
    # Sample the latent code.
    z = jax.random.normal(device_rngs, shape=z_mean.shape, dtype=jnp.float32)
    z = decoding.flat_batch_beam_expand(z, num_hypotheses)

  encoded_inputs = decoding.flat_batch_beam_expand(
      encoded_inputs, num_hypotheses
  )

  # For variational model, decode from the sampled latent code.
  if backend_type == "variational":
    encoded_inputs = (z, encoded_inputs)
  raw_inputs = decoding.flat_batch_beam_expand(inputs, num_hypotheses)

  def token_ids_to_logits(
      state: decoding.DecodingState
  ) -> tuple[Array, dict[str, Array]]:
    """Token slice to logits from decoder model."""
    # Below, beam=num_hypotheses:
    #   flat_ids: [batch * beam, seq_len]
    #   cache is expanded inside search implementation to become `flat_cache`
    #   flat_cache: [batch * beam, num_heads, depth_per_head, max_decode_len]
    #   flat_logits: [batch * beam, seq_len, vocab]
    flat_cache = state.cache
    flat_ids = state.cur_token
    flat_logits, new_vars = model.apply(
        {
            "params": params,
            "cache": flat_cache,
        },
        encoded_inputs,
        raw_inputs,  # only needed for input padding mask
        flat_ids,
        mutable=["cache"],
        method=model.decode,
        rngs={
            "latents": device_rngs,
        }
    )
    new_flat_cache = new_vars["cache"]
    # Remove singleton sequence-length dimension:
    #   [batch * num_hypotheses, 1, vocab] --> [batch * num_hypotheses, vocab]
    flat_logits = flat_logits.squeeze(axis=1)
    return flat_logits, new_flat_cache

  # Set up the decoding function.
  if config.decoder.algorithm == "beam":  # beam search.
    decoder_fn = decoding.beam_search
    decoder_params = dict(
        alpha=config.decoder.brevity_alpha,
        max_decode_len=max_decode_len
    )
  elif config.decoder.algorithm == "sampling":  # stochastic sampling.
    decoder_fn = decoding.temperature_sample
    decoder_params = dict(
        max_decode_steps=max_decode_len,
        temperature=config.decoder.temperature,
        topk=config.decoder.top_k,
        topp=config.decoder.top_p
    )
  else:
    raise ValueError(f"Unsupported algorithm: {config.decoder.algorithm}")

  # Using the above-defined single-step decoder function, run specified decoding
  # algorithm over possible sequences given input encoding.
  decoder_input_tokens = jnp.zeros(
      (inputs.shape[0], max_decode_len), dtype=jnp.int32
  )
  hypotheses, scores = decoder_fn(
      inputs=decoder_input_tokens,
      cache=cache,
      tokens_to_logits=token_ids_to_logits,
      eos_id=_EOS_ID,
      decode_rng=device_rngs,
      num_decodes=num_hypotheses,
      **decoder_params
  )

  # Search returns [n_batch, n_beam, n_length] with beam dimension
  # sorted in increasing order of log-probability.
  return hypotheses, scores


def _batch_predictions_to_dicts(
    config: ml_collections.ConfigDict,
    input_batch: dict[str, Array],
    predictions: Array,
    scores: Array
) -> list[dict[str, Any]]:
  """Processes the predictions for the current batch.

  Args:
    config: Configuration dictionary.
    input_batch: Batch of inputs features corresponding to predictions.
    predictions: Array of predictions (B, W, L) where B is batch size, W is the
      beam width and L is the decoding length. The beam dimension is sorted in
      the increasing order of log-probability.
    scores: An array of W scores. Each score is a log-probability corresponding
      to a single hypothesis.

  Returns:
    A list of W (the beam width) hypothesis dictionaries where each dictionary
    contains features and the corresponding predictions. The hypothesis are
    sorted in the increasing order of log-likelihood.
  """
  batch = jax.tree_util.tree_map(
      lambda x: _to_host(x).tolist(), input_batch
  )
  batch_size = predictions.shape[0]
  prediction_dicts = []
  for i in range(batch_size):
    prediction_dict = {
        "inputs": dict([
            (name, batch[name][i].decode("utf-8"))
            if name != "doc.id" else (name, batch[name][i])
            for name in config.protoscribe.features.passthrough
        ]),
        "prediction": [
            # Prune out padding for each individual hypothesis.
            hypothesis[:hypothesis.index(0)]
            if 0 in hypothesis else hypothesis
            for hypothesis in predictions[i].tolist()
        ],
        "aux": {
            "scores": scores[i].tolist(),
        },
    }
    prediction_dicts.append(prediction_dict)

  return prediction_dicts


def predict(
    config: ml_collections.ConfigDict, checkpoint_dir: str, work_dir: str
) -> None:
  """Runs prediction.

  Args:
    config: Configuration to use.
    checkpoint_dir: Model directory containing the checkpoints.
    work_dir: Working directory for the job where the prediction will be stored.
  """
  # Load dataset.
  # ---------------------------------------------------------------------------
  logging.info("Initializing dataset ...")
  if not FLAGS.dataset_dir:
    raise ValueError("Specify dataset directory --dataset_dir!")
  num_devices = jax.local_device_count()
  predict_ds = input_pipeline.get_test_dataset(
      config=config,
      dataset_dir=FLAGS.dataset_dir,
      num_devices=num_devices
  )
  if config.decoder.num_predict_steps > 0:
    predict_ds = predict_ds.take(config.decoder.num_predict_steps)
  predict_iter = iter(predict_ds)

  # Initialize model.
  # ---------------------------------------------------------------------------
  logging.info("Restoring model ...")
  if config.backend_type == "vanilla":
    predict_config = vanilla_network.get_config(config, utils.RunType.PREDICT)
  elif config.backend_type == "variational":
    predict_config = variational_network.get_config(
        config, utils.RunType.PREDICT
    )
  else:
    raise ValueError(f"Unknown backend: {config.backend_type}")

  rng = jax.random.key(config.seed)
  rng, init_rng = jax.random.split(rng)
  first_batch = _to_jax(next(predict_iter))
  all_shapes = dict(
      [(name, first_batch[name].shape) for name in first_batch.keys()]
  )
  logging.info("Shapes: %s", all_shapes)

  model = _get_model(config.backend_type, predict_config)
  initial_variables = jax.jit(model.init)(
      init_rng,
      features=dict(
          inputs=jnp.ones_like(first_batch["inputs"]),
          targets=jnp.ones_like(first_batch["targets"])
      )
  )
  del initial_variables["cache"]
  initial_variables = flax.core.freeze(initial_variables)
  state = checkpoints.restore_checkpoint(
      checkpoint_dir, target=initial_variables
  )["params"]

  # Create one copy of the model parameters for each local device.
  state = jax_utils.replicate(state)

  # Run inference.
  # ---------------------------------------------------------------------------
  logging.info("Compiling cache initializer ...")
  p_init_cache = jax.pmap(
      functools.partial(
          _initialize_cache,
          backend_type=config.backend_type,
          model_config=predict_config,
      ),
      axis_name="batch",
  )

  logging.info("Compling predictor ...")
  rng, prediction_rng = jax.random.split(rng)
  p_prediction_step = jax.pmap(
      functools.partial(
          _prediction_step,
          config=config,
          model_config=predict_config,
      ),
      axis_name="batch",
  )

  if not tf.io.gfile.exists(work_dir) and jax.process_index() == 0:
    logging.info("Making %s ...", work_dir)
    tf.io.gfile.makedirs(work_dir)
  output_jsonl_path = os.path.join(work_dir, "inference_results.jsonl")
  logging.info("Writing predictions to %s ...", output_jsonl_path)
  if jax.process_index() == 0:
    output_file = tf.io.gfile.GFile(output_jsonl_path, "w")

  logging.info("Running the inference ...")
  batch_id = 0
  predict_iter = iter(predict_ds.prefetch(tf.data.experimental.AUTOTUNE))
  for batch in predict_iter:
    logging.info("Processing batch %d ...", batch_id)
    batch = _to_jax(batch)
    # Handle final odd-sized batch by padding instead of dropping it.
    cur_batch_size = batch["inputs"].shape[0]
    if cur_batch_size % num_devices:
      padded_size = int(
          jnp.ceil(cur_batch_size / num_devices) * num_devices
      )
      batch = jax.tree_util.tree_map(
          lambda x: _pad_examples(x, padded_size),  # pylint: disable=cell-var-from-loop
          batch,
      )
    batch = common_utils.shard(batch)
    cache = p_init_cache(
        params=state,
        inputs=batch["inputs"],
        targets=batch["targets"],
    )
    prediction_rng = jax.random.fold_in(prediction_rng, batch_id)
    prediction_rngs = jax.random.split(prediction_rng, num_devices)
    predicted, scores = p_prediction_step(
        inputs=batch["inputs"],
        params=state,
        cache=cache,
        device_rngs=prediction_rngs,
    )
    predicted = _to_host(predicted)
    scores = _to_host(scores)

    # Process the predictions for the current batch and dump them to the
    # predictions file in JSONL format.
    prediction_dicts = _batch_predictions_to_dicts(
        config, batch, predicted, scores
    )
    for prediction_dict in prediction_dicts:
      output_file.write(
          json.dumps(prediction_dict, sort_keys=True, ensure_ascii=False) + "\n"
      )
      output_file.flush()

    batch_id += 1

  # Clean up and mark the inference process as complete.
  output_file.close()
  with tf.io.gfile.GFile(f"{output_jsonl_path}.COMPLETED", "w") as f:
    f.write("")

  multihost_utils.sync_global_devices("protoscribe:complete")
  logging.info("Inference complete.")
