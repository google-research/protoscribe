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

"""Partitioning rules in support of PjitPartitioner.
"""

from typing import Callable, Optional, Sequence, Tuple


def additional_axis_rules() -> Sequence[Tuple[str, Optional[str]]]:
  """Axis rules to help with sharding and/or replication of T5 models.

  These rules are added to the standard T5 rules.

  Returns:
    a sequence of (logical_axis_name, data_or_model_or_none) pairs
  """
  return (
      ('vit_batch_tiled', None),  # Batch dim but replicated not parallelized
      ('vit_row', None),  # 2d image patch dim (height)
      ('vit_col', None),  # 2d image patch dim (width)
      ('vit_seq', None),  # Flattened image patch sequence (width * height)
      ('vit_channel', None),  # Num. of color channels (usually 3)
      ('vit_embed', None),  # ViT embed dim
      ('vit_head', None),  # ViT attention head dim
      ('vit_kv', None),  # ViT attention KV dim
      ('vit_joined_kv', None),  # ViT attention head+KV dims.
      ('vit_mlp', None),  # ViT MLP dim
      ('vit_out', None),  # ViT representation dim
      ('vit_expert', 'expert'),  # ViT expert dim.
      ('vit_expert_replicas', None),  # ViT expert replicas.
      ('vit_expert_group', None),  # ViT expert groups.
      ('vit_expert_embed', None),  # ViT embed dim in MoE layers.
      ('vit_expert_mlp', None),  # ViT MLP dim in MoE layers.
      ('vit_router', None),
      ('vit_stack', None),
      ('vit_layers', None),
      ('vit_raw', None),
      ('dense_embed', None),  # Dense Transformer input embedding
      ('head_mlp', None),  # Head mlp dim.
      # Below are MAX ViT rules.
      ('instance', None),
      ('raw', 'model'),
      ('stack', None),
      ('layers', None),
      ('generic', None),
      ('unmodeled', None),
      ('expert', 'data'),
      ('expert_replicas', None),
      ('expert_group', None),
      ('expert_embed', 'model'),
      # Below are Adaptation rules.
      ('prompt_length', None),
      ('prompt_embed', None),
  )


ParamAxesNamesOverrides = Sequence[Tuple[str, Tuple[str, ...]]]
ParamAxesNamesOverrideFn = Callable[[], ParamAxesNamesOverrides]


def no_override() -> ParamAxesNamesOverrides:
  """An empty mapping that leaves everything up to logical named axes."""
  return ()


def legacy_vit_names_override() -> ParamAxesNamesOverrides:
  """Mapping of regex to named axes.

  This assigns parameters to named axes for architectures that do not use
  `param_with_axes` to define their variables, such as Scenic ViT.

  Returns:
    a sequence of (regex, named_axes) pairs
  """
  # pylint: disable=line-too-long
  # The regex r'^(*/)/' matches the potential left_encoder/ or right_encoder/
  # prefix for Dual Encoder use cases.
  return (
      (r'^(.*[/])?vit_encoder/Transformer/encoderblock_(\d+)/MultiHeadDotProductAttention_\d+/(key|query|value)/kernel', ('vit_embed', 'vit_head', 'vit_kv')),
      (r'^(.*[/])?vit_encoder/Transformer/encoderblock_(\d+)/MultiHeadDotProductAttention_\d+/(key|query|value)/bias', ('vit_head', 'vit_kv')),
      (r'^(.*[/])?vit_encoder/Transformer/encoderblock_(\d+)/MultiHeadDotProductAttention_\d+/out/kernel', ('vit_head', 'vit_kv', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/Transformer/encoderblock_(\d+)/MultiHeadDotProductAttention_\d+/out/bias', ('vit_embed',)),
      (r'^(.*[/])?vit_encoder/Transformer/encoderblock_(\d+)/LayerNorm_(\d+)/(bias|scale)', ('vit_embed',)),
      (r'^(.*[/])?vit_encoder/Transformer/encoderblock_(\d+)/MlpBlock_(\d+)/Dense_0/bias', ('vit_mlp',)),
      (r'^(.*[/])?vit_encoder/Transformer/encoderblock_(\d+)/MlpBlock_(\d+)/Dense_0/kernel', ('vit_embed', 'vit_mlp')),
      (r'^(.*[/])?vit_encoder/Transformer/encoderblock_(\d+)/MlpBlock_(\d+)/Dense_1/bias', ('vit_embed',)),
      (r'^(.*[/])?vit_encoder/Transformer/encoderblock_(\d+)/MlpBlock_(\d+)/Dense_1/kernel', ('vit_mlp', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/MAPHead_0/probe', ('vit_batch_tiled', 'vit_seq', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/MAPHead_0/LayerNorm_0/(scale|bias)', ('vit_embed',)),
      (r'^(.*[/])?vit_encoder/MAPHead_0/MlpBlock_0/Dense_0/bias', ('vit_mlp',)),
      (r'^(.*[/])?vit_encoder/MAPHead_0/MlpBlock_0/Dense_0/kernel', ('vit_embed', 'vit_mlp')),
      (r'^(.*[/])?vit_encoder/MAPHead_0/MlpBlock_0/Dense_1/bias', ('vit_embed',)),
      (r'^(.*[/])?vit_encoder/MAPHead_0/MlpBlock_0/Dense_1/kernel', ('vit_mlp', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/MAPHead_0/MultiHeadDotProductAttention_0/(key|query|value)/bias', ('vit_head', 'vit_kv')),
      (r'^(.*[/])?vit_encoder/MAPHead_0/MultiHeadDotProductAttention_0/(key|query|value)/kernel', ('vit_embed', 'vit_head', 'vit_kv')),
      (r'^(.*[/])?vit_encoder/MAPHead_0/MultiHeadDotProductAttention_0/out/bias', ('vit_embed',)),
      (r'^(.*[/])?vit_encoder/MAPHead_0/MultiHeadDotProductAttention_0/out/kernel', ('vit_head', 'vit_kv', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/Transformer/encoder_norm/(bias|scale)', ('vit_embed',)),
      (r'^(.*[/])?vit_encoder/Transformer/posembed_input/pos_embedding', ('vit_batch_tiled', 'vit_seq', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/pos_embedding', ('vit_batch_tiled', 'vit_seq', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/embedding/kernel', ('vit_row', 'vit_col', 'vit_channel', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/embedding/bias', ('vit_embed',)),
      (r'^(.*[/])?vit_encoder/cls', ('vit_batch_tiled', 'vit_seq', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/TransformerTokenLearner/DynamicTokenizer_(\d+)/Conv_(\d+)/kernel', ('vit_row', 'vit_col', 'vit_channel', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/TransformerTokenLearner/encoderblock_(\d+)/MultiHeadDotProductAttention_\d+/(key|query|value|out)/kernel', ('vit_embed', 'vit_head', 'vit_kv')),
      (r'^(.*[/])?vit_encoder/TransformerTokenLearner/posembed_input/pos_embedding', ('vit_batch_tiled', 'vit_seq', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/pre_logits/kernel', ('vit_embed', 'vit_out')),
      (r'^(.*[/])?vit_encoder/pre_logits/bias', ('vit_out',)),
  )
  # pylint: enable=line-too-long


def vmoe_names_override() -> ParamAxesNamesOverrides:
  # pylint: disable=line-too-long
  # The regex r'^(*/)/' matches the potential left_encoder/ or right_encoder/
  # prefix for Dual Encoder use cases.
  return (
      (r'^(.*[/])?vit_encoder/embedding/kernel', ('vit_row', 'vit_col', 'vit_channel', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/embedding/bias', ('vit_embed',)),
      (r'^(.*[/])?vit_encoder/cls', ('vit_batch_tiled', 'vit_seq', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/Encoder/posembed_input/pos_embedding', ('vit_batch_tiled', 'vit_seq', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/Encoder/encoderblock_(\d+)/LayerNorm_(\d+)/(bias|scale)', ('vit_embed',)),
      (r'^(.*[/])?vit_encoder/Encoder/encoderblock_(\d+)/SelfAttention/(key|query|value)/kernel', ('vit_embed', 'vit_joined_kv')),
      (r'^(.*[/])?vit_encoder/Encoder/encoderblock_(\d+)/SelfAttention/(key|query|value)/bias', ('vit_joined_kv',)),
      (r'^(.*[/])?vit_encoder/Encoder/encoderblock_(\d+)/SelfAttention/out/kernel', ('vit_joined_kv', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/Encoder/encoderblock_(\d+)/SelfAttention/out/bias', ('vit_embed',)),
      (r'^(.*[/])?vit_encoder/Encoder/encoderblock_(\d+)/Mlp/wi/bias', ('vit_mlp',)),
      (r'^(.*[/])?vit_encoder/Encoder/encoderblock_(\d+)/Mlp/wi/kernel', ('vit_embed', 'vit_mlp')),
      (r'^(.*[/])?vit_encoder/Encoder/encoderblock_(\d+)/Mlp/wo/bias', ('vit_embed',)),
      (r'^(.*[/])?vit_encoder/Encoder/encoderblock_(\d+)/Mlp/wo/kernel', ('vit_mlp', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/Encoder/encoderblock_(\d+)/Moe/Mlp/wi/bias', ('vit_expert', 'vit_expert_mlp',)),
      (r'^(.*[/])?vit_encoder/Encoder/encoderblock_(\d+)/Moe/Mlp/wi/kernel', ('vit_expert', 'vit_embed', 'vit_expert_mlp')),
      (r'^(.*[/])?vit_encoder/Encoder/encoderblock_(\d+)/Moe/Mlp/wo/bias', ('vit_expert', 'vit_embed',)),
      (r'^(.*[/])?vit_encoder/Encoder/encoderblock_(\d+)/Moe/Mlp/wo/kernel', ('vit_expert', 'vit_expert_mlp', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/Encoder/encoder_norm/(bias|scale)', ('vit_embed',)),
      (r'^(.*[/])?vit_encoder/MAPHead/probe', ('vit_batch_tiled', 'vit_seq', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/MAPHead/LayerNorm/(scale|bias)', ('vit_embed',)),
      (r'^(.*[/])?vit_encoder/MAPHead/Mlp/(Dense_0|wi)/bias', ('vit_mlp',)),
      (r'^(.*[/])?vit_encoder/MAPHead/Mlp/(Dense_0|wi)/kernel', ('vit_embed', 'vit_mlp')),
      (r'^(.*[/])?vit_encoder/MAPHead/Mlp/(Dense_1|wo)/bias', ('vit_embed',)),
      (r'^(.*[/])?vit_encoder/MAPHead/Mlp/(Dense_1|wo)/kernel', ('vit_mlp', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/MAPHead/MultiHeadDotProductAttention/(key|query|value)/bias', ('vit_joined_kv',)),
      (r'^(.*[/])?vit_encoder/MAPHead/MultiHeadDotProductAttention/(key|query|value)/kernel', ('vit_embed', 'vit_joined_kv')),
      (r'^(.*[/])?vit_encoder/MAPHead/MultiHeadDotProductAttention/out/bias', ('vit_embed',)),
      (r'^(.*[/])?vit_encoder/MAPHead/MultiHeadDotProductAttention/out/kernel', ('vit_joined_kv', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/pre_logits/kernel', ('vit_embed', 'vit_out')),
      (r'^(.*[/])?vit_encoder/pre_logits/bias', ('vit_out',)),
  )
  # pylint: enable=line-too-long


def vmoe_max_names_override() -> ParamAxesNamesOverrides:
  # pylint: disable=line-too-long
  # The regex r'^(*/)/' matches the potential left_encoder/ or right_encoder/
  # prefix for Dual Encoder use cases.
  return (
      (r'^(.*[/])?vit_encoder/max_vmoe/cls_token', ('vit_batch_tiled', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/max_vmoe/vision_raw_to_embed/rgb_to_embedding/bias', ('vit_embed',)),
      (r'^(.*[/])?vit_encoder/max_vmoe/vision_raw_to_embed/rgb_to_embedding/kernel', ('vit_raw', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/max_vmoe/vision_raw_to_embed/rgb_pos_encoding/flattened_position_embeddings/embedding', ('vit_seq', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/max_vmoe/post_encoder/MAPHead_0/MultiHeadDotProductAttention_0/(q|k|v)/bias', ('vit_head', 'vit_kv')),
      (r'^(.*[/])?vit_encoder/max_vmoe/post_encoder/MAPHead_0/MultiHeadDotProductAttention_0/(q|k|v)/kernel', ('vit_embed', 'vit_head', 'vit_kv')),
      (r'^(.*[/])?vit_encoder/max_vmoe/post_encoder/MAPHead_0/MultiHeadDotProductAttention_0/o/bias', ('vit_embed',)),
      (r'^(.*[/])?vit_encoder/max_vmoe/post_encoder/MAPHead_0/MultiHeadDotProductAttention_0/o/kernel', ('vit_head', 'vit_kv', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/max_vmoe/post_encoder/MAPHead_0/FeedForward_0/wi/bias', ('vit_mlp',)),
      (r'^(.*[/])?vit_encoder/max_vmoe/post_encoder/MAPHead_0/FeedForward_0/wi/kernel', ('vit_embed', 'vit_mlp')),
      (r'^(.*[/])?vit_encoder/max_vmoe/post_encoder/MAPHead_0/FeedForward_0/wo/bias', ('vit_embed',)),
      (r'^(.*[/])?vit_encoder/max_vmoe/post_encoder/MAPHead_0/FeedForward_0/wo/kernel', ('vit_mlp', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/max_vmoe/post_encoder/MAPHead_0/LayerNorm_0/(scale|bias)', ('vit_embed',)),
      (r'^(.*[/])?vit_encoder/max_vmoe/post_encoder/MAPHead_0/probe', ('vit_batch_tiled', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/max_vmoe/post_encoder/pre_logits/bias', ('vit_out',)),
      (r'^(.*[/])?vit_encoder/max_vmoe/post_encoder/pre_logits/kernel', ('vit_embed', 'vit_out')),
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/final_layer_norm/(bias|scale)', ('vit_embed',)),
      # Encoder blocks without scan.
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_\d+/layer_norm_(sa|ffn)/(bias|scale)', ('vit_embed',)),
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_\d+/self_attention/(q|k|v)/bias', ('vit_head', 'vit_kv')),
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_\d+/self_attention/(q|k|v)/kernel', ('vit_embed', 'vit_head', 'vit_kv')),
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_\d+/self_attention/o/bias', ('vit_embed',)),
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_\d+/self_attention/o/kernel', ('vit_head', 'vit_kv', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_\d+/feed_forward/wi/bias', ('vit_mlp',)),
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_\d+/feed_forward/wi/kernel', ('vit_embed', 'vit_mlp')),
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_\d+/feed_forward/wo/bias', ('vit_embed',)),
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_\d+/feed_forward/wo/kernel', ('vit_mlp', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_\d+/feed_forward/experts/wi/bias', ('vit_expert', 'vit_mlp',)),
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_\d+/feed_forward/experts/wi/kernel', ('vit_expert', 'vit_embed', 'vit_expert_mlp')),
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_\d+/feed_forward/experts/wo/bias', ('vit_expert', 'vit_embed',)),
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_\d+/feed_forward/experts/wo/kernel', ('vit_expert', 'vit_expert_mlp', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_\d+/feed_forward/router/w/kernel', ('vit_embed', 'vit_router',)),
      # Encoder blocks with scan.
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_scan_(sparse|dense)/layer_norm_(sa|ffn)/(bias|scale)', ('vit_embed', 'vit_stack')),
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_scan_(sparse|dense)/self_attention/(q|k|v)/bias', ('vit_head', 'vit_stack', 'vit_kv')),
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_scan_(sparse|dense)/self_attention/(q|k|v)/kernel', ('vit_embed', 'vit_stack', 'vit_head', 'vit_kv')),
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_scan_(sparse|dense)/self_attention/o/bias', ('vit_embed', 'vit_stack')),
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_scan_(sparse|dense)/self_attention/o/kernel', ('vit_head', 'vit_stack', 'vit_kv', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_scan_dense/feed_forward/wi/bias', ('vit_mlp', 'vit_stack')),
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_scan_dense/feed_forward/wi/kernel', ('vit_embed', 'vit_stack', 'vit_mlp')),
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_scan_dense/feed_forward/wo/bias', ('vit_embed', 'vit_stack')),
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_scan_dense/feed_forward/wo/kernel', ('vit_mlp', 'vit_stack', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_scan_sparse/feed_forward/experts/wi/bias', ('vit_expert', 'vit_stack', 'vit_expert_mlp',)),
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_scan_sparse/feed_forward/experts/wi/kernel', ('vit_expert', 'vit_stack', 'vit_embed', 'vit_expert_mlp')),
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_scan_sparse/feed_forward/experts/wo/bias', ('vit_expert', 'vit_stack', 'vit_embed',)),
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_scan_sparse/feed_forward/experts/wo/kernel', ('vit_expert', 'vit_stack', 'vit_expert_mlp', 'vit_embed')),
      (r'^(.*[/])?vit_encoder/max_vmoe/moe_transformer_encoder/layer_scan_sparse/feed_forward/router/w/kernel', ('vit_embed', 'vit_stack', 'vit_router',)),
  )
  # pylint: enable=line-too-long
