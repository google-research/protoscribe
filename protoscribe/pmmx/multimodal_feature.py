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

"""Helper class for juggling input features."""

import dataclasses
from typing import Any, Mapping, Optional, Sequence, Type

from absl import logging
import flax
import gin
import jax.numpy as jnp
import numpy as np

from flaxformer.components.attention import dense_attention

# Type stubs
DType = jnp.dtype | np.dtype
Array = jnp.ndarray | np.ndarray


@dataclasses.dataclass
class MultimodalFeature:
  """Helper class for juggling input features.

  Unlike T5/P5X, which only has two features (`inputs` and `targets`), PMMX
  supports many combinations of input features. This presents a coding challenge
  since every feature needs to be plumbed through the entire nn.Module stack.

  The `MultimodalFeature` class allows the features to be tracked as a
  collection. It keeps relevant information together, such as `positions` and
  `segment_ids`, and has helpful utility methods for constructing attention
  masks and segment_id spans.

  Attributes:
    name: original feature name, as passed by the FeatureConverter
    values: actual feature values
    modality_name: name of the modality that this feature maps to
    modality_id: int identifier of the modality
    positions: explicit position ids (None except when packing is used)
    segment_ids: segment ids (None except when packing is used)
    subfeatures: sequence of subfeatures names.
    embedder_name: name of the embedder in the model
  """
  name: str
  values: Array
  modality_name: str
  modality_id: int
  positions: Optional[Array]  # packing-only
  segment_ids: Optional[Array]  # packing-only
  subfeatures: Sequence[Type['MultimodalFeature']] = tuple()

  @property
  def embedder_name(self) -> str:
    if self.name.endswith('text_tokens') or self.modality_name == 'text_tokens':
      embedder_name = 'embedder'  # special case for ckpt compatibility
    else:
      embedder_name = f'{self.modality_name}_embedder'
    logging.info('Returning embedder %s for feature %s', embedder_name,
                 self.name)
    return embedder_name

  def make_modality_segment_ids(self, dtype: DType) -> Array:
    """Returns modality segment ids for this feature.

    Args:
      dtype: dtype to use

    Returns:
      Array of shape [seq_length] containing the modality id for this feature
    """
    return np.full([self.values.shape[1]], self.modality_id, dtype=dtype)

  def make_attention_mask(self, batch: Sequence[Any], dtype: DType) -> Array:
    """Creates a self-attention mask for this MultimodalFeature.

    Args:
      batch: all of the MultimodalFeatures to attend over
      dtype: type of values in the returned mask

    Returns:
      Array, the attention mask for this MultimodalFeature
    """
    return dense_attention.make_attention_mask(
        query_input=attention_mask_for_zeros([self.values]),
        key_input=attention_mask_for_zeros([other.values for other in batch]),
        dtype=dtype)


def attention_mask_for_zeros(inputs: Sequence[Array]) -> Array:
  """Creates a mask from the encoder inputs.

  This assumes the mask is a function of the value. In particular:
    For 2D inputs, zero-valued tokens are masked.
    For 3D inputs, zero vectors are masked (all elements must be exactly zero)

  Args:
    inputs: input values (2D for tokens or 3D for vectors)

  Returns:
    an Array of shape `[batch_size, seq_len]`
  """
  value_list = []
  for values in inputs:
    if values.ndim == 2:
      value_list.append(values > 0)
    elif values.ndim == 3:
      value_list.append(jnp.any(values != 0, axis=2))
    else:
      raise ValueError('rank must be 2 or 3')
  return jnp.concatenate(value_list, axis=1)


@gin.configurable
def linearize_encoder_features(
    batch: Mapping[str, Array],
    feature_spec: Sequence[tuple[str, str]],
    modality_spec: Sequence[str],
    sub_feature_spec: Optional[Mapping[str, Sequence[str]]] = None,
    passthrough_features: Sequence[str] = tuple(),
) -> Sequence[MultimodalFeature]:
  """Returns a sorted list of encoder features.

  The order of the returned features (i.e. the linearization) is determined by
  the order of tuples in `feature_spec`. The tuples in `feature_spec`
  specify the name of each feature's modality. Sharing modalities across
  features is possible.

  The name of the modality is used in two ways:
    1) It acts as the prefix of an embedder in `MultimodalEncoder`. For
       example, the `image_dense` modality maps to
       `image_dense_embedder`. This allows modality embeddings to be
       Gin-configurable. A special case is `text_token`, which just maps to
       `embedder`.
    2) It is used to look up the modality id in `modality_spec`, which is used
       by `multimodal_relative_position_biases.py`.

  Example:
    batch = {
        'page_text': [...],
        'title_text': [...],
        'image_dense': [...]
    }

    feature_spec = (
        ('image_dense', 'image_dense'),
        ('page_text', 'text_tokens'),
        ('title_text', 'text_tokens')
    )

    modality_spec = ('image_dense', 'text_tokens')

    The features would be returned in the following order:
      `image_dense`, then `page_text`, then `title_text`

    The `page_text` and `title_text` tokens would have
    `modality_name=text_tokens` (`modality_id=1`) and the `image_dense`
    feature would have `modality_name=image_tokens` (`modality_id=0`).

  Args:
    batch: feature names to values
    feature_spec: list of `(feature_name, modality_name)` pairs, in implicit
      sequence order
    modality_spec: list of `modality_name`s, in implicit id order
    sub_feature_spec: an optional mapping describing relations between primary
      and sub features.
    passthrough_features: list of features that are not linearized.

  Returns:
    a list of MultimodalFeatures, in a stable sequence order

  Raises:
    ValueError: if the provided features included decoder features or if
      the features could not be found in the `feature_modalities`
  """
  # Copy the batch so we can check that all features were ingested at the end.
  batch = dict(batch)
  packing_features = {}
  for k in list(batch):
    if k in passthrough_features:
      batch.pop(k)
    if k.endswith('_positions') or k.endswith('_segment_ids'):
      packing_features[k] = batch.pop(k)
    if k.startswith('decoder'):
      raise ValueError(f'found decoder feature={k} during linearization')

  # Create a dict for looking up the sort index for each feature.
  ordinals = {k: i for (i, (k, _)) in enumerate(feature_spec)}
  # Create a dict for looking up modality names from feature names.
  feature_to_modality = dict(feature_spec)
  # Create a dict for looking up modality ids from modality names.
  modality_name_to_id = {k: i for (i, k) in enumerate(modality_spec)}

  for k in batch:
    if k not in ordinals:
      raise ValueError(
          f'feature={k} not defined in feature_spec={feature_spec}')
    if k not in feature_to_modality:
      raise ValueError(
          f'Failed to detect the modality of feature={k} in '
          f'`feature_spec={feature_spec}`. Please add your feature '
          f'to the spec (in the gin config).')
    if feature_to_modality[k] not in modality_name_to_id:
      raise ValueError(
          f'Failed to find a modality id for feature={k} with '
          f'modality={feature_to_modality[k]} in '
          f'modality_ids={modality_name_to_id}.')
  ordered_features = []
  ordered_feature_names = sorted(batch, key=ordinals.__getitem__)
  multimodal_features = {}
  for name in ordered_feature_names:
    if packing_features:
      positions = packing_features[f'{name}_positions']
      segment_ids = packing_features[f'{name}_segment_ids']
    else:
      positions = None
      segment_ids = None
    modality_name = feature_to_modality[name]
    modality_id = modality_name_to_id[modality_name]
    values = batch.pop(name)
    multimodal_features[name] = MultimodalFeature(
        name=name,
        values=values,
        modality_name=modality_name,
        modality_id=modality_id,
        positions=positions,
        segment_ids=segment_ids)

  # Validates that all features were consumed.
  if batch:
    raise ValueError(f'Leftover features in batch={batch}')
  all_sub_feature_names = set()
  for name in ordered_feature_names:
    # If this feature has sub-features, add them to the `subfeatures` attribute.
    if sub_feature_spec and name in sub_feature_spec:
      sf_names = sub_feature_spec[name]
      all_sub_feature_names.update(sf_names)
      subfeatures = tuple(multimodal_features[sf_name] for sf_name in sf_names)
      multimodal_features[name] = dataclasses.replace(  # pytype: disable=wrong-arg-types  # dataclasses-replace-types
          multimodal_features[name], subfeatures=subfeatures)
  # Collect all of the top-level features into an ordered sequence.
  for name in ordered_feature_names:
    if name not in all_sub_feature_names:
      ordered_features.append(multimodal_features[name])
  return ordered_features


def make_encoder_mask(encoder_features: Sequence[MultimodalFeature],
                      dtype: DType) -> tuple[Array, Optional[Array]]:
  """Constructs a mask and optional segment ids, if packing."""
  # Prepare the self-attention masks.
  encoder_masks = [
      ef.make_attention_mask(encoder_features, dtype) for ef in encoder_features
  ]

  packing = encoder_features[0].segment_ids is not None

  # Add segmentation block-diagonal attention mask if using packing.
  if packing:
    encoder_segment_ids = jnp.concatenate(
        [ef.segment_ids for ef in encoder_features], axis=-1)
    for i, ef in enumerate(list(encoder_features)):
      encoder_segment_mask = dense_attention.make_attention_mask(
          ef.segment_ids,
          encoder_segment_ids,
          pairwise_fn=jnp.equal,
          dtype=dtype)
      encoder_masks[i] = dense_attention.combine_masks(encoder_masks[i],
                                                       encoder_segment_mask)
  else:
    encoder_segment_ids = None

  encoder_mask = jnp.concatenate(encoder_masks, axis=-2)

  return encoder_mask, encoder_segment_ids


@gin.configurable
def make_encoder_mask_fewshot(
    encoder_features: Sequence[MultimodalFeature],
    dtype: DType,
    fewshot_feature_visibilities: str = '',
    fewshot_segment_ids_const: int = 10000,
    ) -> tuple[Array, Optional[Array]]:
  """Constructs a mask and optional segment ids for fewshot prompting.

  Building encoder masks to enable fewshot examples to attend to each other
  under packing settings.

  Args:
    encoder_features: Sequence of linearized encoder features.
    dtype: encoder features' data type.
    fewshot_feature_visibilities: encoder features visibility strategy,
      a string in 'modality_name1/modality_name2:visibility,...' format.
      E.g. 'text_tokens/text_tokens:all' meaning text_tokens modality can attend
      to all other fewshot examples' text_tokens field.
    fewshot_segment_ids_const: A const in segment_ids to differentiate the
      few-shot examples from packed examples. E.g. examples with segment_ids in
      [0, 10000] are from the same seq in the few-shot learning settings and
      examples with segment_ids 20000 are packed examples.

  Returns:
    encoder_mask: encoder masks that enable fewshot examples to attend to
      each other.
    encoder_segment_ids: concatenated encoder_segment_ids in the packing
      setting.
  """
  # Prepare the self-attention masks.
  encoder_masks = [
      ef.make_attention_mask(encoder_features, dtype) for ef in encoder_features
  ]

  packing = encoder_features[0].segment_ids is not None
  fewshot_feature_visibilities = fewshot_feature_visibilities.split(',')
  visibility_dict = {}
  for fewshot_feature_visibility in fewshot_feature_visibilities:
    modality_names, visibility = fewshot_feature_visibility.split(':')
    visibility_dict[modality_names] = visibility
  if packing:
    encoder_segment_ids = jnp.concatenate(
        [ef.segment_ids for ef in encoder_features], axis=-1)
    for i, ef in enumerate(list(encoder_features)):
      encoder_mask_list = []
      modality_name = encoder_features[i].modality_name
      for j in range(len(list(encoder_features))):
        key_modality_name = encoder_features[j].modality_name
        modality_names = '/'.join([modality_name, key_modality_name])
        visibility = visibility_dict.get(modality_names, 'self')
        if visibility == 'self':
          encoder_segment_mask = dense_attention.make_attention_mask(
              ef.segment_ids,
              encoder_features[j].segment_ids,
              pairwise_fn=jnp.equal,
              dtype=dtype)
        elif visibility == 'all':
          encoder_segment_mask = jnp.ones([
              ef.segment_ids.shape[0], 1, ef.segment_ids.shape[1],
              encoder_features[j].segment_ids.shape[1]
          ]).astype('int')
          encoder_segment_mask_packing = dense_attention.make_attention_mask(
              ef.segment_ids // fewshot_segment_ids_const,
              encoder_features[j].segment_ids // fewshot_segment_ids_const,
              pairwise_fn=jnp.equal,
              dtype=dtype,
          ).astype('int')
          encoder_segment_mask = (
              encoder_segment_mask * encoder_segment_mask_packing
          )
          encoder_segment_mask = encoder_segment_mask.astype('bool')

        else:
          raise ValueError(f'visibility {visibility} for features modality'
                           f'{modality_names} not implemented')
        encoder_mask_list.append(encoder_segment_mask)
      encoder_segment_mask = jnp.concatenate(encoder_mask_list, axis=-1)
      encoder_masks[i] = dense_attention.combine_masks(encoder_masks[i],
                                                       encoder_segment_mask)
  else:
    encoder_segment_ids = None

  encoder_mask = jnp.concatenate(encoder_masks, axis=-2)
  return encoder_mask, encoder_segment_ids


@flax.struct.dataclass
class SequenceMetadata(object):
  modality_segment_ids: Array
  feature_name_to_segment_id_map: Mapping[str, int]
  feature_name_to_bounds_map: Mapping[str, tuple[int, int]]


def make_sequence_metadata(
    encoder_features: Sequence[MultimodalFeature],
    dtype: DType) -> SequenceMetadata:
  """Provided a sequence of MultimodalFeatures, returns a SequenceMetadata.

  Args:
    encoder_features: the features being encoded by e.g. a MultimodalEncoder
    dtype: the data type to use for the modality_segment_ids array

  Returns:
    a SequenceMetadata object for passing useful information about the original
      feature layer to other parts of the MultimodalEncoder, such as feature
      ranges, the original dense representation of each feature, and the
      "modality ids" assigned to each feature
  """
  segment_id_map = {}
  bounds_map = {}
  modality_segment_ids_list = []
  start = 0
  end = 0
  for ef in encoder_features:
    sub_array = ef.make_modality_segment_ids(dtype)
    if ef.name in segment_id_map:
      raise ValueError(f'duplicate feature name={ef.name}')
    segment_id_map[ef.name] = ef.modality_id
    modality_segment_ids_list.append(sub_array)
    end += ef.values.shape[1]  # feature values have shape [batch, seqlen, ...]
    bounds_map[ef.name] = (start, end)
    start = end
  modality_segment_ids = np.concatenate(modality_segment_ids_list, axis=0)
  return SequenceMetadata(
      modality_segment_ids=modality_segment_ids,
      feature_name_to_segment_id_map=segment_id_map,
      feature_name_to_bounds_map=bounds_map)
