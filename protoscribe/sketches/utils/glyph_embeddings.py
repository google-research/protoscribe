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

"""Computes semantic and phonetic embeddings [V, embedding_length] tensors."""

from absl import logging
import jax
import jax.numpy as jnp
import ml_collections
from protoscribe.glyphs import glyph_vocab as glyph_lib
from protoscribe.language.embeddings import embedder as embedder_lib
from protoscribe.language.morphology import morphemes
from protoscribe.language.phonology import phonetic_embeddings as phonetic_lib
from protoscribe.sketches.utils import stroke_utils

JTensor = jnp.ndarray

_EPSILON = 1E-12
_NEG_INF_VALUE = jnp.finfo(jnp.float32).min


def _l2_normalize(
    x: JTensor, axis: int = -1, epsilon: float = _EPSILON
) -> JTensor:
  """Numerically stable L2 normalizer along a given axis."""
  square_sum = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
  x_inv_norm = jax.lax.rsqrt(jnp.maximum(square_sum, epsilon))
  return jnp.multiply(x, x_inv_norm)

# TODO: In both of these routines we assign 1. to the first position of the
# empty embedding so that the resulting embedding is not at the origin, since
# this makes no sense for a cosine distance. At some point we should perhaps
# migrate this logic into the code that generates empty embeddings in the first
# place, though it only matters for the cosine loss.


def glyphs_to_embeddings(
    config: ml_collections.ConfigDict,
    glyph_vocab: glyph_lib.GlyphVocab,
) -> JTensor:
  """Computes semantic and phonetic embeddings [V, embedding_length] tensors.

  This should allow one to look up any glyph via its index and get its embedding
  (either semantic or phonetic) by matrix multiplying the one-hot vector for the
  glyph with the embedding.

  TODO: Even though we load the number lexicon there are actually no
  embeddings to be had for number glyphs. The problem is that the input concepts
  are based on the number value --- e.g. 45 --- whereas the number is written
  with glyphs --- e.g. X X X X V --- which don't have values that correspond to
  the original number: presumably if anything in this case it would be "ten,
  ten, ten, ten, five"  For now we just disable completely so that they should
  get zeroed embeddings.

  Args:
    config: Configuration specification.
    glyph_vocab: Glyph vocabulary.

  Returns:
    A [V, 2, embedding_length] tensor.
  """
  # TODO: Not clear this logic is needed since in any case we presume that
  # the passed configuration must contain this path and other paths below.
  # TODO: I don't think this is the right way to ago about this
  # (passing paths through config). Will need to re-think this later.

  # Semantic embeddings:
  concept_embedding_type = config.concept_embedding_type
  logging.info("Loading `%s` concept embeddings ...", concept_embedding_type)
  embedder = embedder_lib.load_embeddings_from_type(concept_embedding_type)

  semantic_embeddings = []
  num_glyphs = len(glyph_vocab)
  number_and_special_ids = glyph_vocab.special_glyph_ids()
  for idx in range(num_glyphs):
    glyph_name = glyph_vocab.id_to_name(idx)
    semantic_embedding = None
    for tag in ["NOUN", "VERB", "ADJ"]:
      emb = embedder.embed(f"{glyph_name}_{tag}", embedder.embedding_type)
      # pylint: disable=g-explicit-length-test
      if len(emb):
        semantic_embedding = emb[0]
        break
      # pylint: enable=g-explicit-length-test
    if semantic_embedding is None:
      # A bit kludgy, but this will get the null embedding if the glyph + tag
      # doesn't work:
      emb = embedder.embed(
          glyph_name,
          embedder.embedding_type,
          add_null_embedding=True,
      )
      semantic_embedding = emb[0]
    if idx in number_and_special_ids:  # Ignore the number and special glyphs.
      semantic_embedding = embedder.null_embedding
    if (semantic_embedding == embedder.null_embedding).all():
      semantic_embedding[0] = 1.0
    semantic_embeddings.append(semantic_embedding)
  semantic_embeddings = jnp.array(semantic_embeddings)

  # Phonetic embeddings
  lexicon = morphemes.Lexicon()
  lexicon.load_lexicon(config.main_lexicon)
  # TODO: Reminder to myself that at some point we want the
  # meanings_to_morphemes to allow for more than one morpheme per meaning.
  terms = {}
  for term in lexicon.meanings_to_morphemes.keys():
    terms[term.split("_")[0]] = lexicon.meanings_to_morphemes[term].form

  ph_embedder = phonetic_lib.load_phonetic_embedder(config.phonetic_embeddings)
  empty_embedding = ph_embedder.empty_embedding
  empty_embedding[0] = 1.0
  phonetic_embeddings = []
  for idx in range(num_glyphs):
    glyph_name = glyph_vocab.id_to_name(idx)
    if glyph_name in terms:
      form = terms[glyph_name]
      try:
        ph_emb = ph_embedder.embedding(form)
      # TODO: This is to protect the system from dying of an index error if
      # there is a mismatch between the lexicon and the lexicon that the (new)
      # phonetic embeddings know about. However, we really need to come up with
      # a more robust approach.
      except IndexError:
        msg = f"Error: {form} not found in phonetic embeddings."
        logging.error(msg)
        ph_emb = empty_embedding
      phonetic_embeddings.append(ph_emb)
    else:
      phonetic_embeddings.append(empty_embedding)
  phonetic_embeddings = jnp.array(phonetic_embeddings)
  joint_embeddings = jnp.reshape(
      jnp.concatenate(
          [semantic_embeddings, phonetic_embeddings],
          axis=1,
      ),
      (semantic_embeddings.shape[0], 2, semantic_embeddings.shape[1]),
  )
  return joint_embeddings


def construct_batch_glyph_embeddings(
    batch_glyph_embeddings: JTensor,
    glyph_types: JTensor,
) -> JTensor:
  """Constructs the `targets` tensor to be passed to `joint_embeddings_loss`.

  Args:
    batch_glyph_embeddings: a [batch, 2, emb_len] tensor of semantic and
      phonetic embeddings for the target glyphs.
    glyph_types: a [batch, n] tensor of indices from `text/glyph/types` or the
      output of construct_loss_mask_for_stroke_glyph_affiliations.

  Returns:
    A [batch, n, 2, emb_len] tensor of target embeddings.
  """
  # TODO: This is used for all paddings and number glyphs, since we do not
  # want to count either of these in the loss computation, and zero is what the
  # number glyphs map to in the embeddings tensor passed to
  # `joint_embeddings_loss`. If the latter changes, then this will need to
  # change also.
  assert batch_glyph_embeddings.shape[0] == glyph_types.shape[0]
  empty_embeddings = jnp.zeros_like(batch_glyph_embeddings)
  # Sets the first element of each to one so it's not at the origin.
  empty_embeddings = empty_embeddings.at[:, :, 0].set(1.0)
  # Shape [3, batch, 2, emb_len]
  indexed_embeddings = jnp.array(
      [
          empty_embeddings,  # glyph_type 0
          empty_embeddings,  # glyph_type 1
          batch_glyph_embeddings,  # glyph_type 2
      ],
  )
  # Transpose to shape [batch, 3, 2, emb_len]
  indexed_embeddings = jnp.transpose(indexed_embeddings, (1, 0, 2, 3))
  targets = jnp.take(indexed_embeddings, glyph_types, axis=1)
  # Pick the 0th, 1st diagonal...Seems there should be a more straightforward
  # way to do this but I haven't figured it out yet.
  targets = jnp.transpose(
      jnp.diagonal(targets, axis1=0, axis2=1),
      (3, 0, 1, 2),
  )
  return targets


def construct_loss_mask(
    glyph_types: JTensor,
    concepts: bool = True
) -> JTensor:
  """Constructs a `mask` tensor to be passed to `joint_embeddings_loss`.

  As defined here, we predict losses only for `concept` glyphs, not digits,
  which in any case currently have null embeddings.

  Args:
    glyph_types: a [batch, n] tensor of indices from `text/glyph/types`.
    concepts: Compute the mask for concepts: 1 for concepts, 0 for everything
      else. If set to False, the mask is reversed.

  Returns:
    A [batch, n] tensor of 1's and 0's.
  """
  mask = (glyph_types == glyph_lib.GLYPH_TYPE_MASK_CONCEPT).astype(
      dtype=jnp.int32
  )
  return mask if concepts else (1 - mask)


def construct_loss_mask_for_stroke_glyph_affiliations(
    glyphs: JTensor,
    glyph_types: JTensor,
    stroke_glyph_affiliations: JTensor,
    concepts: bool = True,
) -> JTensor:
  """Constructs a mask for stroke portions affiliated with concept glyphs.

  In the below, `n` is the length of the glyphs, whereas `N` is the length of
    the (tokenized) stroke data.

  Args:
    glyphs: a [batch, n] tensor of glyphs.
    glyph_types: a [batch, n] tensor of indices from `text/glyph/types`.
    stroke_glyph_affiliations: a [batch, N] tensor of glyph-stroke affiliation
      information originally from `strokes/glyph_affiliations/ids`
    concepts: Compute the mask for concepts: 1 for concepts, 0 for everything
      else. If set to False, the mask is reversed.

  Returns:
    A [batch, N] tensor of glyph_lib.GLYPH_TYPE_MASK_PAD or
     glyph_lib.GLYPH_TYPE_MASK_CONCEPT.
  """

  def extend_with_eos(arr, eos):
    eos = jnp.tile(eos, (arr.shape[0], 1))
    return jnp.concatenate([arr, eos], axis=1)

  glyphs_plus_eos = extend_with_eos(glyphs, stroke_utils.END_OF_STROKE)
  glyph_types_plus_eos = extend_with_eos(
      glyph_types,
      glyph_lib.GLYPH_TYPE_MASK_PAD,
  )
  # Below, -1 changes the zeros to -1, so that zeros in
  # stroke_glyph_affiliations will not count as being in idx.
  idx = jnp.where(
      glyph_types_plus_eos == glyph_lib.GLYPH_TYPE_MASK_CONCEPT,
      glyphs_plus_eos,
      jnp.zeros_like(glyphs_plus_eos) - 1,
  )
  mask = jnp.where(
      jnp.isin(stroke_glyph_affiliations, idx),
      jnp.ones_like(stroke_glyph_affiliations),
      jnp.zeros_like(stroke_glyph_affiliations),
  )
  return mask if concepts else (1 - mask)


def construct_targeted_sem_phon_loss_mask(
    mask: JTensor,
) -> tuple[JTensor, JTensor]:
  """Constructs `mask` tensor pair to be passed to `joint_embeddings_loss`.

  This corresponds to the semantic-phonetic mask pair argument.

  Args:
    mask: a [batch, n] mask tensor, typically the output of construct_loss_mask.

  Returns:
    A pair of [batch, n] tensors of 1's and 0's. The first is the semantic mask,
    which masks all but the first position, and the second is the phonetic mask,
    which masks all but the last position.
  """
  first = jnp.argmax(mask, axis=1)
  last = mask.shape[1] - 1 - jnp.argmax(mask[:, -1::-1], axis=1)
  sem_mask = jax.nn.one_hot(first, mask.shape[1], axis=1, dtype=jnp.int32)
  phon_mask = jax.nn.one_hot(last, mask.shape[1], axis=1, dtype=jnp.int32)
  return sem_mask, phon_mask


def pairwise_cosine_similarity(
    embeddings: JTensor,
    glyph_vocab: glyph_lib.GlyphVocab,
    temperature: float = 1.0,
    mask_diagonal: bool = True,
    similarity_to_self: float = 1.0
) -> JTensor:
  """Computes cosine similarity matrix.

  Args:
    embeddings: [V, D] tensor, where D is the embedding dimension.
    glyph_vocab: Glyph vocabulary.
    temperature: Temperature factor for controlling probabilities.
    mask_diagonal: Whether to include similarity to self.
    similarity_to_self: The value to put on the diagonal that signifies
      the "soft" self-similarity if `mask_diagonal` is disabled.

  Returns:
    Array with shape [V, V] with probabities of picking the closest glyphs.
  """

  # Compute point-wise similarities.
  norm_embeddings = _l2_normalize(embeddings, axis=-1)
  sim = jnp.matmul(norm_embeddings, jnp.transpose(norm_embeddings))

  # Optionally mask out the diagonal and special glyphs. The similarities to
  # special glyphs will be masked out completely.
  mask_ids = glyph_vocab.special_glyph_ids()
  sim = sim.at[mask_ids, :].set(_NEG_INF_VALUE)
  sim = sim.at[:, mask_ids].set(_NEG_INF_VALUE)
  self_sim_value = _NEG_INF_VALUE if mask_diagonal else similarity_to_self
  sim = jnp.fill_diagonal(sim, self_sim_value, inplace=False)

  # Normalize to probabilities.
  norm_sim = jax.nn.softmax(sim / temperature, axis=-1)
  return norm_sim


def build_closest_glyphs(
    embeddings: JTensor,
    glyph_vocab: glyph_lib.GlyphVocab,
    temperature: float = 1.0,
    mask_diagonal: bool = True,
    similarity_to_self: float = 1.0
) -> JTensor:
  """Computes closest (according to semantics and phonetics) glyphs.

  Args:
    embeddings: [V, 2, D] tensor of glyph embeddings.
    glyph_vocab: Glyph vocabulary.
    temperature: Temperature factor for controlling probabilities.
    mask_diagonal: Whether to include similarity to self.
    similarity_to_self: The value to put on the diagonal that signifies
      the "soft" self-similarity if `mask_diagonal` is enabled.

  Returns:
    Returns an array with shape [V, 2, V] representing a pairwise matrix
    representing the most probable closest glyphs to the given one according
    to semantics ([:, 0, :]) or phonetics ([:, 1, :]).
  """
  assert len(embeddings.shape) == 3
  assert embeddings.shape[1] == 2

  kw_args = dict(
      temperature=temperature, mask_diagonal=mask_diagonal,
      similarity_to_self=similarity_to_self
  )
  semantic_sim = pairwise_cosine_similarity(
      embeddings[:, 0, :], glyph_vocab, **kw_args
  )
  phonetic_sim = pairwise_cosine_similarity(
      embeddings[:, 1, :], glyph_vocab, **kw_args
  )
  return jnp.stack([semantic_sim, phonetic_sim], axis=1)


def closest_glyphs_soft_targets(
    targets: JTensor, closest_targets: JTensor
) -> JTensor:
  """Constructs closest glyphs soft targets for supplied targets.

  Args:
    targets: Tensor of integer target tokens [B, N].
    closest_targets: [V, 2, V] tensor of pair-wise closest glyphs according to
      semantics or phonetics.

  Returns:
    A tensor [B, N, 2, V] of soft targets.
  """
  num_classes = closest_targets.shape[0]
  one_hot_targets = jax.nn.one_hot(
      targets, num_classes=num_classes, axis=-1, dtype=jnp.float32
  )
  # (B, N, 2, V].
  closest_glyphs = jnp.einsum("ijk,klm->ijlm", one_hot_targets, closest_targets)
  return closest_glyphs
