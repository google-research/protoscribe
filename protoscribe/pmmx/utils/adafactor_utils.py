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

"""Adafactor logical rules for PMMX."""

from flax.core import freeze
from flax.core import unfreeze
from t5x import adafactor


def vit_factor_rules(scan_layers: bool = False):
  """Additional factor rules for ViT.

  These are the Legacy rules, which are still required by scan_vit.gin.

  Any parameters that are larger than 2-dim need a factor rule in adafactor.
  The t5 ones were included defined in standard_factor_rules. Here we add the
  ones for ViT modules. If the new param have something similar to the ones in:
  t5/experimental/p5x/adafactor.py, we just follow similar
  routines. If not, we put half to c and the other half to r.

  Args:
    scan_layers: bool. whether scan-over-layers is enabled

  Returns:
    rules: rules for ViT.
  """
  standard_rules = adafactor.standard_factor_rules(scan_layers=scan_layers)

  # pylint: disable=line-too-long
  new_rules = (
      (r'vit_encoder/MAPHead_0/probe', ('c', 'c', 'r')),
      (r'vit_encoder/MAPHead_0/MultiHeadDotProductAttention_0/(key|query|value)/kernel', ('r', 'c', 'c')),
      (r'vit_encoder/MAPHead_0/MultiHeadDotProductAttention_0/out/kernel', ('c', 'c', 'r')),
      (r'vit_encoder/Transformer/encoderblock_(\d+)/MultiHeadDotProductAttention_\d/(key|query|value)/kernel', ('r', 'c', 'c')),
      (r'vit_encoder/Transformer/encoderblock_(\d+)/MultiHeadDotProductAttention_\d/out/kernel', ('c', 'c', 'r')),
      (r'vit_encoder/Transformer/posembed_input/pos_embedding', ('c', 'c', 'r')),
      (r'vit_encoder/embedding/kernel', ('c', 'c', 'r', 'r')),
      (r'vit_encoder/cls', ('c', 'c', 'r')),
      (r'vit_encoder/TransformerTokenLearner/DynamicTokenizer_(\d+)/Conv_(\d+)/kernel', ('c', 'c', 'r', 'r')),
      (r'vit_encoder/TransformerTokenLearner/encoderblock_(\d+)/MultiHeadDotProductAttention_\d/(key|query|value)/kernel', ('r', 'c', 'c')),
      (r'vit_encoder/TransformerTokenLearner/encoderblock_(\d+)/MultiHeadDotProductAttention_\d/out/kernel', ('c', 'c', 'r')),
      (r'vit_encoder/TransformerTokenLearner/posembed_input/pos_embedding', ('c', 'c', 'r')),
      (r'vit_encoder/pos_embedding', ('c', 'c', 'r')),
  )
  # pylint: enable=line-too-long

  rules = new_rules + standard_rules
  return rules


def vmoe_factor_rules(scan_layers: bool = False):
  """Additional factor rules for V-MoE.

  Any parameters that are larger than 2-dim need a factor rule in adafactor.
  The t5 ones were included defined in standard_factor_rules. Here we add the
  ones for V-MoE modules. If the new param have something similar to the ones
  in t5/experimental/p5x/adafactor.py, we just follow similar
  routines. If not, we put half to c and the other half to r.

  Args:
    scan_layers: bool. whether scan-over-layers is enabled

  Returns:
    rules: rules for V-MoE.
  """
  standard_rules = adafactor.standard_factor_rules(scan_layers=scan_layers)
  # pylint: disable=line-too-long
  # TODO: Check if expert's bias factorization is equivalent to what we did.
  # When we trained V-MoE-G, we implicitly used something ('b', 'n'), but this isn't
  # allowed by adafactor.py, so I just used ('r', 'c') which is the default.
  new_rules = (
      (r'vit_encoder/embedding/kernel', ('c', 'c', 'r', 'r')),
      (r'vit_encoder/cls', ('c', 'c', 'r')),
      (r'vit_encoder/Encoder/posembed_input/pos_embedding', ('c', 'c', 'r')),
      (r'vit_encoder/Encoder/encoderblock_(\d+)/Moe/Mlp/wi/bias', ('b', 'r',)),
      (r'vit_encoder/Encoder/encoderblock_(\d+)/Moe/Mlp/wi/kernel', ('b', 'r', 'c')),
      (r'vit_encoder/Encoder/encoderblock_(\d+)/Moe/Mlp/wo/bias', ('b', 'r',)),
      (r'vit_encoder/Encoder/encoderblock_(\d+)/Moe/Mlp/wo/kernel', ('b', 'r', 'c')),
  )
  # pylint: enable=line-too-long
  rules = new_rules + standard_rules
  return rules


def logical_factor_rules():
  """Logical factor rules for PMMX (T5X plus multimodal models like ViT)."""
  rules = unfreeze(adafactor.standard_logical_factor_rules())
  rules.update({
      'vit_batch_tiled': adafactor.FactorDim.COLUMN,
      'vit_row': adafactor.FactorDim.COLUMN,
      'vit_col': adafactor.FactorDim.COLUMN,
      'vit_seq': adafactor.FactorDim.COLUMN,
      'vit_channel': adafactor.FactorDim.ROW,
      'vit_embed': adafactor.FactorDim.ROW,
      'vit_head': adafactor.FactorDim.COLUMN,
      'vit_kv': adafactor.FactorDim.COLUMN,
      'vit_mlp': adafactor.FactorDim.ROW,
      'dense_embed': adafactor.FactorDim.COLUMN,
      'head_mlp': adafactor.FactorDim.COLUMN,
      # MAX ViT below
      'generic': adafactor.FactorDim.COLUMN,
      'abspos_buckets': adafactor.FactorDim.NONE,
      'raw': adafactor.FactorDim.ROW,
      # V-MoE below.
      'vit_heads': adafactor.FactorDim.COLUMN,
      'vit_joined_kv': adafactor.FactorDim.COLUMN,

      'vit_raw': adafactor.FactorDim.ROW,
      'vit_out': adafactor.FactorDim.ROW,
      'vit_expert': adafactor.FactorDim.BATCH,
      'vit_expert_mlp': adafactor.FactorDim.ROW,
      'vit_router': adafactor.FactorDim.COLUMN,
      # 'batch', 'length' should not occur in parameters
      'unmodeled': adafactor.FactorDim.NONE,

      # Adaptation below.
      'prompt_length': adafactor.FactorDim.NONE,
      'prompt_embed': adafactor.FactorDim.NONE,
  })
  return freeze(rules)
