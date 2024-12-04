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

"""Simple utilities for parsing and writing inference JSONs."""

import json
import logging
from typing import Any, Mapping, Sequence, Type

import jax
import jax.numpy as jnp
import numpy as np
import seqio
import tensorflow as tf
from tensorflow.io import gfile

_MANDATORY_FEATURES = [
    'doc.id', 'concept.name', 'number.name', 'text.sampa', 'text.words',
]


def get_scorer_dict(
    pred_dict: dict[str, Any],
    pronunciation_lexicon: dict[str, list[str]]
) -> dict[str, Any]:
  """Constructs scorer-specific dictionary from the given predictions."""
  if 'inputs' not in pred_dict:
    raise ValueError('Input features not found in JSON!')

  aux_dict = pred_dict['inputs']
  scorer_dict: dict[str, Any] = {
      key: aux_dict[key] for key in _MANDATORY_FEATURES
  }
  scorer_dict['text.inputs'] = (
      scorer_dict['number.name'] + ' ' + scorer_dict['concept.name']
  )
  # Document ID is stored as float for some reason.
  scorer_dict['doc.id'] = int(scorer_dict['doc.id'])

  # Fills in pronunciation for the concept.
  concept_name = scorer_dict['concept.name'].split('_')[0]  # POS tag.
  if concept_name not in pronunciation_lexicon:
    raise ValueError(f'No pronunciation for concept `{concept_name}`!')
  scorer_dict['concept.pron'] = pronunciation_lexicon[concept_name]

  return scorer_dict


def glyph_pron(score_dict: dict[str, Any], k: int) -> str:
  """Retrieves k-th pronunciation among n-best from the dictionary."""
  kth_prons = score_dict['glyph.prons'][k]
  prons = ' # '.join(
      [' '.join(pron) for pron in filter(None, kth_prons)]
  )
  return prons


def get_confidence(json_dict: dict[str, Any]) -> float:
  """Computes confidence measure between best and second-best hypothesis."""
  if 'aux' not in json_dict or 'scores' not in json_dict['aux']:
    raise ValueError('Expected generation scores under `scores.aux!')
  scores = json_dict['aux']['scores']
  confidence = 0.  # When no beam is available, use dummy confidence.
  if isinstance(scores, list):
    if len(scores) < 2:
      raise ValueError('Expected beam with at least two paths!')
    confidence = scores[-1] - scores[-2]
  return confidence


def write_inferences_to_file(
    path: str,
    inferences: tuple[Sequence[Any], Mapping[str, Any]],
    task_ds: tf.data.Dataset,
    mode: str,
    vocabulary: seqio.Vocabulary | None = None,
    json_encoder_cls: Type[json.JSONEncoder] = seqio.TensorAndNumpyEncoder,
    include_all_inputs: bool = False,
    input_fields_to_include: Sequence[str] | None = None,
    output_ids: bool = False,
) -> None:
  """Write model predictions, along with pretokenized inputs, to JSONL file.

  Args:
    path: File path to write to.
    inferences: A tuple containing (predictions, aux_values). If mode is
      'predict' then the `predictions` will be token IDs. If it's 'score' then
      it'll be a collection of scores. `aux_values` will be an empty dictionary
      unless mode is 'predict_with_aux', in which case it'll contain the model's
      auxiliary outputs.
    task_ds: Original task dataset. Features from task with suffix
      `_pretokenized` are added to the outputs.
    mode: Prediction mode, either 'predict', 'score' or 'predict_with_aux'.
    vocabulary: Task output vocabulary. Only used in `predict` mode in order to
      decode predicted outputs into string.
    json_encoder_cls: a JSON encoder class used to customize JSON serialization
      via json.dumps.
    include_all_inputs: if True, will include all model inputs in the output
      JSONL file (including raw tokens) in addition to the pretokenized inputs.
    input_fields_to_include: List of input fields to include in the output JSONL
      file. This list should be None if `include_all_inputs` is set to True.
    output_ids: if True, will output the token ID sequence for the output, in
      addition to the decoded text.
  """
  all_predictions, all_aux_values = inferences

  if mode in ('predict', 'predict_with_aux') and vocabulary is None:
    raise ValueError(
        'The `vocabulary` parameter is required in `predict` and '
        '`predict_with_aux` modes'
    )

  def _json_compat(value):
    if isinstance(value, bytes):
      return value.decode('utf-8')
    elif isinstance(value, (jnp.bfloat16, jnp.floating)):
      return float(value)
    elif isinstance(value, jnp.integer):
      return float(value)
    elif isinstance(value, (jnp.ndarray, np.ndarray)):
      # Flatten array features.
      return value.tolist()
    else:
      return value

  if include_all_inputs and input_fields_to_include is not None:
    raise ValueError(
        'include_all_inputs and input_fields_to_include should not be set'
        ' simultaneously.'
    )
  with gfile.GFile(path, 'w') as f:
    for i, inp in task_ds.enumerate().as_numpy_iterator():
      predictions = all_predictions[i]
      aux_values = jax.tree.map(
          f=lambda v, i=i: v[i],
          tree=all_aux_values,
          is_leaf=lambda v: isinstance(v, (np.ndarray, list)),
      )

      if include_all_inputs:
        inputs = inp
      elif input_fields_to_include is not None:
        inputs = {
            k: v
            for k, v in inp.items()
            if k in input_fields_to_include
            or (
                k.endswith('_pretokenized')
                and k[: -len('_pretokenized')] in input_fields_to_include
            )
        }
      else:
        inputs = {k: v for k, v in inp.items() if k.endswith('_pretokenized')}

      json_dict = {}
      json_dict['inputs'] = {k: _json_compat(v) for k, v in inputs.items()}

      if mode == 'predict':
        assert vocabulary is not None
        json_dict['prediction'] = _json_compat(
            vocabulary.decode_tf(tf.constant(predictions)).numpy()
        )
        if output_ids:
          pred = _json_compat(tf.constant(predictions).numpy())
          # Truncate padding tokens.
          assert isinstance(pred, list)
          pred = pred[: pred.index(0)] if 0 in pred else pred
          json_dict['prediction_tokens'] = pred
      elif mode == 'score':
        json_dict['score'] = _json_compat(predictions)
        if aux_values:
          json_dict['aux'] = jax.tree.map(_json_compat, aux_values)
      elif mode == 'predict_with_aux':
        assert vocabulary is not None
        json_dict['prediction'] = _json_compat(
            vocabulary.decode_tf(tf.constant(predictions)).numpy()
        )
        if (
            isinstance(json_dict['prediction'], list) and
            json_dict['prediction'] and
            isinstance(json_dict['prediction'][0], list) and
            json_dict['prediction'][0] and
            isinstance(json_dict['prediction'][0][0], int)
        ):
          # Truncate padding tokens if predictions are integers.
          preds = []
          for pred in json_dict['prediction']:
            pred = pred[: pred.index(0)] if 0 in pred else pred
            preds.append(pred)
          json_dict['prediction'] = preds
        if output_ids:
          pred = _json_compat(tf.constant(predictions).numpy())
          # Truncate padding tokens.
          pred = pred[: pred.index(0)] if 0 in pred else pred
          json_dict['prediction_tokens'] = pred
        json_dict['aux'] = jax.tree.map(_json_compat, aux_values)
      else:
        raise ValueError(f'Invalid mode: {mode}')
      json_str = json.dumps(json_dict, cls=json_encoder_cls)
      f.write(json_str + '\n')


def load_jsonl(path: str) -> list[dict[str, Any]]:
  """Reads JSON dictionaries from a file.

  Args:
    path: Path to a JSONL file.

  Returns:
    A list of individual dictionaries.
  """
  logging.info('Reading JSONL from %s ...', path)
  dicts = []
  with gfile.GFile(path, mode='r') as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      dicts.append(json.loads(line))
  logging.info('Read %d dictionaries.', len(dicts))

  return dicts
