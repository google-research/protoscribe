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

from absl.testing import absltest
from absl.testing import parameterized
import flax
from flax.training import train_state
import jax
import numpy as np
import optax
from protoscribe.models.flax import utils
from protoscribe.models.flax import vanilla_network as lib

# Model parameters.
_OUTPUT_VOCAB_SIZE = 64
_TOKEN_EMBEDDING_DIM = 32
_NUM_HEADS = 1
_NUM_ENCODER_LAYERS = 1
_NUM_DECODER_LAYERS = 1
_MLP_DIM = 8
_QKV_DIM = 4
_MAX_STROKE_TOKENS = 5

# Batch parameters.
_BATCH_SIZE = 3
_INPUT_EMBEDDING_DIM = 16
_INPUT_EMBEDDING_LEN = 7


def _mock_config(run_type: utils.RunType) -> lib.TransformerConfig:
  """Creates simple configuration for the test.

  Args:
    run_type: Can be one of the run modes: `TRAIN`, `EVAL` or `PREDICT`.

  Returns:
    Simple configuration.
  """
  config = lib.TransformerConfig(
      output_vocab_size=_OUTPUT_VOCAB_SIZE,
      emb_dim=_TOKEN_EMBEDDING_DIM,
      num_heads=_NUM_HEADS,
      num_encoder_layers=_NUM_ENCODER_LAYERS,
      num_decoder_layers=_NUM_DECODER_LAYERS,
      qkv_dim=_QKV_DIM,
      mlp_dim=_MLP_DIM,
      deterministic=False if run_type == utils.RunType.TRAIN else True,
      decode=True if run_type == utils.RunType.PREDICT else False,
  )
  return config


def _mock_batch(
    config: lib.TransformerConfig, rng: np.random.Generator
) -> dict[str, np.ndarray]:
  """Creates a mock batch of input features.

  Args:
    config: Model configuration.
    rng: Random number generator.

  Returns:
    A dictionary of arrays.
  """
  return dict(
      inputs=rng.standard_normal(
          size=(_BATCH_SIZE, _INPUT_EMBEDDING_LEN, _INPUT_EMBEDDING_DIM),
          dtype=np.float32
      ),
      targets=rng.integers(
          low=0,
          high=config.output_vocab_size,
          size=(_BATCH_SIZE, _MAX_STROKE_TOKENS),
          dtype=np.int32
      ),
  )


class VanillaNetworkTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._rng = np.random.default_rng()
    self._jax_rng = jax.random.key(0)

  @parameterized.named_parameters(
      dict(testcase_name="train_model", run_type=utils.RunType.TRAIN),
      dict(testcase_name="evaluate_model", run_type=utils.RunType.EVAL),
  )
  def test_train_or_eval(self, run_type: utils.RunType):
    config = _mock_config(run_type)
    model = lib.Transformer(config)
    batch = _mock_batch(config, self._rng)

    # Init model and create initial parameters.
    rng, init_rng = jax.random.split(self._jax_rng)
    initial_variables = model.init(init_rng, features=batch)
    self.assertIn("params", initial_variables)

    params = initial_variables["params"]
    if run_type == utils.RunType.TRAIN:
      state = train_state.TrainState.create(
          apply_fn=model.apply,
          params=initial_variables["params"],
          tx=optax.identity()
      )
      self.assertNotEmpty(state.params)
      params = state.params

    # Perform one training/development step.
    next_batch = _mock_batch(config, self._rng)
    _, dropout_rng = jax.random.split(rng)
    model_outputs = model.apply(
        {"params": params},
        features=next_batch,
        rngs=(
            {"dropout": dropout_rng}
            if run_type == utils.RunType.TRAIN else None
        )
    )
    self.assertIn("logits", model_outputs)
    self.assertEqual(
        model_outputs["logits"].shape,
        (
            _BATCH_SIZE,
            _MAX_STROKE_TOKENS,
            _OUTPUT_VOCAB_SIZE,
        )
    )

  def test_predictions(self):
    config = _mock_config(utils.RunType.PREDICT)
    model = lib.Transformer(config)
    batch = _mock_batch(config, self._rng)

    # Init model and create initial parameters.
    init_rng, _ = jax.random.split(self._jax_rng)
    initial_variables = model.init(init_rng, features=batch)
    self.assertIn("params", initial_variables)
    params = initial_variables["params"]
    self.assertNotEmpty(params)

    # Initialize cache.
    _, initial_variables = model.apply(
        {"params": params},
        features=batch,
        mutable=["cache"]
    )
    self.assertIn("cache", initial_variables)
    initial_variables = flax.core.freeze(initial_variables)
    cache = initial_variables["cache"]
    self.assertNotEmpty(cache)

    # Run encoder.
    encoded_inputs = model.apply(
        {"params": params},
        batch["inputs"],
        method=model.encode
    )
    self.assertEqual(
        encoded_inputs.shape,
        (
            _BATCH_SIZE,
            _INPUT_EMBEDDING_LEN,
            _TOKEN_EMBEDDING_DIM,
        )
    )

    # Run the decoder auto-regressively.
    for step in range(_MAX_STROKE_TOKENS):
      targets = batch["targets"][:, step]
      logits, new_variables = model.apply(
          {
              "params": params,
              "cache": cache,
          },
          encoded=encoded_inputs,
          inputs=batch["inputs"],
          targets=targets[:, None],  # (B, 1).
          mutable=["cache"],
          method=model.decode,
      )
      self.assertEqual(
          logits.shape,
          (
              _BATCH_SIZE,
              1,
              _OUTPUT_VOCAB_SIZE,
          )
      )
      self.assertIn("cache", new_variables)
      cache = new_variables["cache"]
      self.assertNotEmpty(cache)

  def test_model_config(self):
    config = _mock_config(utils.RunType.TRAIN)
    self.assertFalse(config.deterministic)
    self.assertFalse(config.decode)

    config = _mock_config(utils.RunType.EVAL)
    self.assertTrue(config.deterministic)
    self.assertFalse(config.decode)

    config = _mock_config(utils.RunType.PREDICT)
    self.assertTrue(config.deterministic)
    self.assertTrue(config.decode)


if __name__ == "__main__":
  absltest.main()
