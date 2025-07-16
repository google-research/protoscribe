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
import chex
import flax
from flax.training import train_state
import jax
import numpy as np
import optax
from protoscribe.glyphs import glyph_vocab as glyph_lib
from protoscribe.models.flax import utils
from protoscribe.models.flax import variational_network as lib

# Core model parameters.
_OUTPUT_VOCAB_SIZE = 64
_TOKEN_EMBEDDING_DIM = 32
_NUM_HEADS = 1
_NUM_ENCODER_LAYERS = 1
_NUM_DECODER_LAYERS = 1
_MLP_DIM = 8
_QKV_DIM = 4
_MAX_STROKE_TOKENS = 5
_MAX_GLYPH_TOKENS = 3
_GLYPH_VOCAB_SIZE = 16

# Latent parameters.
_LATENT_SIZE = 6
_LATENT_NUM_ENCODER_LAYERS = 1

# Batch parameters.
_BATCH_SIZE = 3
_INPUT_EMBEDDING_DIM = 16
_INPUT_EMBEDDING_LEN = 7


def _mock_config(
    run_type: utils.RunType,
    encoder_pooling: str = "mean",
    encoder_mask_non_concept_tokens: bool = True,
    latent_blend_strategy: str = "encoder_concat",
    latent_share_token_embeddings: bool = False
) -> lib.TransformerConfig:
  """Creates simple configuration for the test.

  Args:
    run_type: Can be one of the run modes: `TRAIN`, `EVAL` or `PREDICT`.
    encoder_pooling: Pooling algorithm. One of: `max`, `mean`, `first`, `last`.
    encoder_mask_non_concept_tokens: Mask out non-concept tokens.
    latent_blend_strategy: How to blend the latents.
    latent_share_token_embeddings: Share the token embeddings between the
      encoder and decoder.

  Returns:
    Simple configuration.
  """
  config = lib.TransformerConfig(
      # Core transformer.
      output_vocab_size=_OUTPUT_VOCAB_SIZE,
      emb_dim=_TOKEN_EMBEDDING_DIM,
      num_heads=_NUM_HEADS,
      num_encoder_layers=_NUM_ENCODER_LAYERS,
      num_decoder_layers=_NUM_DECODER_LAYERS,
      qkv_dim=_QKV_DIM,
      mlp_dim=_MLP_DIM,
      deterministic=False if run_type == utils.RunType.TRAIN else True,
      decode=True if run_type == utils.RunType.PREDICT else False,
      # Latents.
      latent_size=_LATENT_SIZE,
      latent_num_encoder_layers=_LATENT_NUM_ENCODER_LAYERS,
      encoder_pooling=encoder_pooling,
      encoder_mask_non_concept_tokens=encoder_mask_non_concept_tokens,
      latent_blend_strategy=latent_blend_strategy,
      latent_share_token_embeddings=latent_share_token_embeddings
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
  return {
      "inputs": rng.standard_normal(
          size=(_BATCH_SIZE, _INPUT_EMBEDDING_LEN, _INPUT_EMBEDDING_DIM),
          dtype=np.float32
      ),
      "targets": rng.integers(
          low=0,
          high=config.output_vocab_size,
          size=(_BATCH_SIZE, _MAX_STROKE_TOKENS),
          dtype=np.int32
      ),
      "text.glyph.tokens": rng.integers(
          low=glyph_lib.GLYPH_PAD,
          high=_GLYPH_VOCAB_SIZE,
          size=(_BATCH_SIZE, _MAX_GLYPH_TOKENS),
          dtype=np.int32,
      ),
      "text.glyph.types": rng.integers(
          low=glyph_lib.GLYPH_TYPE_MASK_NUMBER,
          high=glyph_lib.GLYPH_TYPE_MASK_CONCEPT,
          size=(_BATCH_SIZE, _MAX_GLYPH_TOKENS),
          dtype=np.int32,
      ),
      "sketch.glyph_affiliations.ids": rng.integers(
          low=glyph_lib.GLYPH_TYPE_MASK_NUMBER,
          high=glyph_lib.GLYPH_TYPE_MASK_CONCEPT,
          size=(_BATCH_SIZE, _MAX_STROKE_TOKENS),
          dtype=np.int32,
      ),
  }


class VariationalNetworkTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._rng = np.random.default_rng()
    self._jax_rng = jax.random.key(0)

  @parameterized.named_parameters(
      dict(
          testcase_name="train_model",
          run_type=utils.RunType.TRAIN,
      ),
      dict(
          testcase_name="evaluate_model",
          run_type=utils.RunType.EVAL,
      ),
      dict(
          testcase_name="train_model_share-embeddings",
          run_type=utils.RunType.TRAIN,
          latent_share_token_embeddings=True,
      ),
      dict(
          testcase_name="evaluate_model_share-embeddings",
          run_type=utils.RunType.EVAL,
          latent_share_token_embeddings=True,
      ),
      dict(
          testcase_name="train_model_mask-numbers",
          run_type=utils.RunType.TRAIN,
          encoder_mask_non_concept_tokens=True,
      ),
      dict(
          testcase_name="evaluate_model_mask-numbers",
          run_type=utils.RunType.EVAL,
          encoder_mask_non_concept_tokens=True,
      ),
      dict(
          testcase_name="train_model_pool-max",
          run_type=utils.RunType.TRAIN,
          encoder_pooling="max",
      ),
      dict(
          testcase_name="evaluate_model_pool-max",
          run_type=utils.RunType.EVAL,
          encoder_pooling="max",
      ),
      dict(
          testcase_name="train_model_pool-first",
          run_type=utils.RunType.TRAIN,
          encoder_pooling="first",
      ),
      dict(
          testcase_name="evaluate_model_pool-first",
          run_type=utils.RunType.EVAL,
          encoder_pooling="first",
      ),
      dict(
          testcase_name="train_model_pool-last",
          run_type=utils.RunType.TRAIN,
          encoder_pooling="last",
      ),
      dict(
          testcase_name="evaluate_model_pool-last",
          run_type=utils.RunType.EVAL,
          encoder_pooling="last",
      ),
      dict(
          testcase_name="train_model_blend-enc-prepend",
          run_type=utils.RunType.TRAIN,
          latent_blend_strategy="encoder_prepend",
      ),
      dict(
          testcase_name="evaluate_model_blend-enc-prepend",
          run_type=utils.RunType.EVAL,
          latent_blend_strategy="encoder_prepend",
      ),
      dict(
          testcase_name="train_model_blend-dec-concat",
          run_type=utils.RunType.TRAIN,
          latent_blend_strategy="decoder_concat",
      ),
      dict(
          testcase_name="evaluate_model_blend-dec-concat",
          run_type=utils.RunType.EVAL,
          latent_blend_strategy="decoder_concat",
      ),
      dict(
          testcase_name="train_model_blend-dec-add",
          run_type=utils.RunType.TRAIN,
          latent_blend_strategy="decoder_add",
      ),
      dict(
          testcase_name="evaluate_model_blend-dec-add",
          run_type=utils.RunType.EVAL,
          latent_blend_strategy="decoder_add",
      ),
      dict(
          testcase_name="train_model_blend-none",
          run_type=utils.RunType.TRAIN,
          latent_blend_strategy="no_latents",
      ),
      dict(
          testcase_name="evaluate_model_blend-none",
          run_type=utils.RunType.EVAL,
          latent_blend_strategy="no_latents",
      ),
      dict(
          testcase_name="train_model_blend-latents-only",
          run_type=utils.RunType.TRAIN,
          latent_blend_strategy="no_conditional",
      ),
      dict(
          testcase_name="evaluate_model_blend-latents-only",
          run_type=utils.RunType.EVAL,
          latent_blend_strategy="no_conditional",
      ),
  )
  def test_train_or_eval(
      self,
      run_type: utils.RunType,
      encoder_pooling: str = "mean",
      encoder_mask_non_concept_tokens: bool = False,
      latent_blend_strategy: str = "encoder_concat",
      latent_share_token_embeddings: bool = False
  ):
    config = _mock_config(
        run_type,
        encoder_pooling,
        encoder_mask_non_concept_tokens,
        latent_blend_strategy,
        latent_share_token_embeddings
    )
    model = lib.VariationalTransformer(config)
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

    # Setup the RNGs.
    dropout_rng, latents_rng = jax.random.split(rng)
    rngs = {"latents": latents_rng}
    if run_type == utils.RunType.TRAIN:
      rngs.update({"dropout": dropout_rng})

    # Perform one training/development step.
    next_batch = _mock_batch(config, self._rng)
    model_outputs = model.apply(
        {"params": params},
        features=next_batch,
        rngs=rngs,
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
    if latent_blend_strategy == "no_latents":
      self.assertNotIn("z_mean", model_outputs)
      self.assertNotIn("z_log_var", model_outputs)
      return

    self.assertIn("z_mean", model_outputs)
    self.assertIn("z_log_var", model_outputs)
    self.assertEqual(
        model_outputs["z_mean"].shape,
        model_outputs["z_log_var"].shape
    )
    self.assertEqual(
        model_outputs["z_mean"].shape,
        (
            _BATCH_SIZE,
            _LATENT_SIZE,
        )
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="predict_model",
      ),
      dict(
          testcase_name="predict_model_share-embeddings",
          latent_share_token_embeddings=True,
      ),
      dict(
          testcase_name="predict_model_mask-numbers",
          encoder_mask_non_concept_tokens=True,
      ),
      dict(
          testcase_name="predict_model_pool-max",
          encoder_pooling="max",
      ),
      dict(
          testcase_name="predict_model_pool-first",
          encoder_pooling="first",
      ),
      dict(
          testcase_name="predict_model_pool-last",
          encoder_pooling="last",
      ),
      dict(
          testcase_name="predict_model_blend-enc-prepend",
          latent_blend_strategy="encoder_prepend",
      ),
      dict(
          testcase_name="predict_model_blend-dec-concat",
          latent_blend_strategy="decoder_concat",
      ),
      dict(
          testcase_name="predict_model_blend-dec-add",
          latent_blend_strategy="decoder_add",
      ),
  )
  def test_predictions(
      self,
      encoder_pooling: str = "mean",
      encoder_mask_non_concept_tokens: bool = False,
      latent_blend_strategy: str = "encoder_concat",
      latent_share_token_embeddings: bool = False
  ):
    config = _mock_config(
        utils.RunType.PREDICT,
        encoder_pooling=encoder_pooling,
        encoder_mask_non_concept_tokens=encoder_mask_non_concept_tokens,
        latent_blend_strategy=latent_blend_strategy,
        latent_share_token_embeddings=latent_share_token_embeddings
    )
    model = lib.VariationalTransformer(config)
    batch = _mock_batch(config, self._rng)

    # Init model and create initial parameters.
    init_rng, rng = jax.random.split(self._jax_rng)
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

    # Run encoder, sample z from the posterior distribution and prepare
    # decoder inputs.
    z_mean, z_log_var, encoded_inputs = model.apply(
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
    self.assertEqual(
        z_mean.shape,
        (
            _BATCH_SIZE,
            _LATENT_SIZE,
        )
    )
    self.assertEqual(z_mean.shape, z_log_var.shape)
    z = self._rng.standard_normal(size=z_mean.shape, dtype=np.float32)
    encoded_inputs = (z, encoded_inputs)

    # Run the decoder auto-regressively.
    latent_rng, _ = jax.random.split(rng)
    for step in range(_MAX_STROKE_TOKENS):
      targets = batch["targets"][:, step]
      logits, new_variables = model.apply(
          {
              "params": params,
              "cache": cache,
          },
          encoded=encoded_inputs,
          cond_inputs=batch["inputs"],
          targets=targets[:, None],  # (B, 1).
          mutable=["cache"],
          method=model.decode,
          rngs={
              "latents": latent_rng,
          },
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


class LatentBlenderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._rng = np.random.default_rng()
    self._z = self._rng.standard_normal(
        size=(_BATCH_SIZE, _LATENT_SIZE), dtype=np.float32
    )
    self._encoded_cond_inputs = self._rng.standard_normal(
        size=(_BATCH_SIZE, _INPUT_EMBEDDING_LEN, _TOKEN_EMBEDDING_DIM),
        dtype=np.float32
    )
    self._decoder_inputs = self._rng.standard_normal(
        size=(_BATCH_SIZE, _MAX_STROKE_TOKENS, _TOKEN_EMBEDDING_DIM),
    )

  def _run_blender(self, blend_strategy: str) -> tuple[chex.Array, chex.Array]:
    """Runs blender module given the blending strategy.

    Args:
      blend_strategy: Type of blending to run.

    Returns:
      A tuple of either encoder outputs with possibly blended in latents and
      embedded decoder input tokens with latents blended in.
    """
    config = _mock_config(
        run_type=utils.RunType.TRAIN, latent_blend_strategy=blend_strategy
    )
    (encoded, y), _ = lib.LatentBlender(config).init_with_output(
        jax.random.PRNGKey(0),
        z=self._z,
        encoded_cond_inputs=self._encoded_cond_inputs,
        decoder_inputs=self._decoder_inputs
    )
    return encoded, y

  @parameterized.named_parameters(
      dict(testcase_name="encoder_concat", blend_strategy="encoder_concat"),
      dict(testcase_name="decoder_concat", blend_strategy="decoder_concat"),
      dict(testcase_name="decoder_add", blend_strategy="decoder_add"),
  )
  def test_concat_or_add(self, blend_strategy: str):
    encoded, y = self._run_blender(blend_strategy)
    self.assertEqual(encoded.shape, self._encoded_cond_inputs.shape)
    self.assertEqual(y.shape, self._decoder_inputs.shape)
    if blend_strategy == "encoder_concat":
      self.assertEqual(y.tolist(), self._decoder_inputs.tolist())
    else:
      self.assertEqual(encoded.tolist(), self._encoded_cond_inputs.tolist())

  def test_encoder_prepend(self):
    encoded, y = self._run_blender("encoder_prepend")
    self.assertEqual(encoded.shape, (
        _BATCH_SIZE,
        self._encoded_cond_inputs.shape[1] + 1,
        self._encoded_cond_inputs.shape[2],
    ))
    self.assertEqual(
        encoded[:, 1:, :].tolist(), self._encoded_cond_inputs.tolist()
    )
    self.assertEqual(y.shape, self._decoder_inputs.shape)
    self.assertEqual(y.tolist(), self._decoder_inputs.tolist())

  def test_latents_only_no_conditional(self):
    """No conditional features, project latents instead as encoder outputs."""
    encoded, y = self._run_blender("no_conditional")
    self.assertEqual(encoded.shape, (_BATCH_SIZE, 1, y.shape[2]))
    self.assertEqual(y.shape, self._decoder_inputs.shape)
    self.assertEqual(y.tolist(), self._decoder_inputs.tolist())

  def test_no_latents(self):
    encoded, y = self._run_blender("no_latents")
    self.assertEqual(encoded.shape, self._encoded_cond_inputs.shape)
    self.assertEqual(y.shape, self._decoder_inputs.shape)
    self.assertEqual(encoded.tolist(), self._encoded_cond_inputs.tolist())
    self.assertEqual(y.tolist(), self._decoder_inputs.tolist())


if __name__ == "__main__":
  absltest.main()
