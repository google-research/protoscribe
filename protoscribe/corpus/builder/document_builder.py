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

"""Utilities for creating a single dataset document."""

import dataclasses
import json
import logging
import random

from absl import flags
from protoscribe.corpus.builder import prepare_utils as utils
from protoscribe.corpus.builder import vision_features_reader as vision
from protoscribe.glyphs import glyph_vocab as glyph_lib
from protoscribe.glyphs import make_text
from protoscribe.glyphs import numbers_to_glyphs
from protoscribe.glyphs import svg_to_strokes
from protoscribe.language.embeddings import embedder
from protoscribe.language.phonology import phonetic_embeddings as phon_embed_lib
from protoscribe.language.syntax import phrase as phrase_lib
from protoscribe.semantics import concept_glosses as gloss_lib
from protoscribe.sketches.utils import stroke_stats as stroke_stats_lib
from protoscribe.sketches.utils import stroke_utils
from protoscribe.texts import generate as generate_lib
import tensorflow as tf

import glob
import os

StrokeStats = stroke_stats_lib.StrokeStats

_PREFER_CONCEPT_SVG = flags.DEFINE_bool(
    "prefer_concept_svg", False,
    "The actual SVGs are looked up by their corresponding glyph names. If "
    "enabled, first try to look up the SVG by the corresponding concept name "
    "ignoring the glyph names if found. This is useful for selecting new "
    "representations produced by the previous round during system evolution. "
    "This option should be enabled for evolution rounds > 0."
)

# Set the default flags for `make_text` and `svg_to_strokes` here. Please
# note, the Ramer-Douglas-Peucker (RDP) tolerance for stroke pruning is set
# to something similar to Sketch-RNN value for QuickDraw and similar datasets
# (aggressive pruning).

make_text.FLAGS.random_resize = True
make_text.FLAGS.random_rotate = True
make_text.FLAGS.random_pad = True
make_text.FLAGS.simplify_svg_trees = False
svg_to_strokes.FLAGS.points_per_segment = 100
svg_to_strokes.FLAGS.flip_vertical = False
svg_to_strokes.FLAGS.deltas = True
svg_to_strokes.FLAGS.apply_rdp = True
svg_to_strokes.FLAGS.rdp_tolerance = 2.0
svg_to_strokes.FLAGS.path_is_stroke = True


@dataclasses.dataclass
class Params:
  """Parameters required to build a single ProtoScribe document.

  These are the global parameters, easily serializable between workers.
  """

  supercategories: dict[str, str]
  concepts: list[str]
  unseen_concepts: list[str]
  all_concepts: list[str]
  concepts_vocab: dict[str, tuple[int, bool]]
  glyph_vocab: glyph_lib.GlyphVocab
  concept_glosses: dict[str, dict[str, str]]
  all_embeddings: dict[str, embedder.Embeddings]
  text_generator: generate_lib.TextGenerator
  vision_features: vision.VisionFeaturesReader
  phonetic_embeddings: phon_embed_lib.PhoneticEmbeddings


def _build_concepts_vocab(
    concepts: list[str], new_concepts: list[str], concepts_vocab_file: str
) -> dict[str, tuple[int, bool]]:
  """Constructs vocabulary for all the concepts."""
  # Each concept name maps to a tuple consisting of unique integer ID and a
  # boolean flag indicating whether the concept is `new/unseen`.
  concepts_dict = {name: (idx, False) for idx, name in enumerate(concepts)}
  offset = len(concepts_dict)
  concepts_dict.update({
      name: (offset + idx, True) for idx, name in enumerate(new_concepts)
  })
  logging.info("Saving concepts vocab in %s ...", concepts_vocab_file)
  with open(concepts_vocab_file, mode="wt") as f:
    json.dump(concepts_dict, f)
  logging.info("Concept vocab with %d entries", len(concepts_dict))
  return concepts_dict


def _build_strokes(
    concept: str,
    glyph_vocab: glyph_lib.GlyphVocab,
    glyph_names: list[str],
    num_concept_glyphs: int,
) -> tuple[
    list[list[tuple[float, float]]] | None,
    list[tuple[int, str]],
    int,
]:
  """Builds accounting document consisting of (possibly multiple) glyphs.

  Args:
    concept: Name of the concept (includes the POS tag).
    glyph_vocab: Glyph vocabulary object.
    glyph_names: Discrete representation of spellings for the text.
    num_concept_glyphs: Number of `concept` glyphs originally given.

  Returns:
    Strokes and their corresponding discrete glyph affiliations, along with the
    final number of concept glyphs found.
  """
  svgs = []
  # First collect the SVGs for the numbers.
  for glyph_name in glyph_names[:-num_concept_glyphs]:
    path_and_variant_id = glyph_vocab.find_svg_from_name(glyph_name)
    if not path_and_variant_id:
      raise ValueError(f"SVG for glyph `{glyph_name}` not found!")
    svgs.append(path_and_variant_id[0])

  # Generally, the glyph names come from the discrete mapping between concepts
  # and the corresponding spellings, where the spellings denote the original
  # meanings of the constituent glyphs. As the system evolves beyond its initial
  # inception (`round 0`) both graphical and discrete representation for the new
  # concepts appears (`round > 0`). For graphics, we want to pick this new
  # representation if it is available, hence the following logic tries to locate
  # the SVG based on concept name first, if configured.
  path_and_variant_id = glyph_vocab.find_svg_from_name(concept)
  if _PREFER_CONCEPT_SVG.value and path_and_variant_id:
    svgs.append(path_and_variant_id[0])
    glyph_names[-num_concept_glyphs:] = [concept.split("_")[0]]
    num_concept_glyphs = 1
  else:
    for glyph_name in glyph_names[-num_concept_glyphs:]:
      path_and_variant_id = glyph_vocab.find_svg_from_name(glyph_name)
      if not path_and_variant_id:
        raise ValueError(f"SVG for glyph `{glyph_name}` not found!")
      svgs.append(path_and_variant_id[0])

  # Combine SVGs (for multiglyphs) and generate stroke data.
  svg_tree, _, _ = make_text.concat_svgs(svgs, glyph_names)
  strokes, stroke_glyph_affiliations = svg_to_strokes.svg_tree_to_strokes(
      svg_tree
  )
  return strokes, stroke_glyph_affiliations, num_concept_glyphs


def build_tf_example(
    doc_id: int,
    concept: str,
    params: Params,
) -> tuple[int, tf.train.Example, StrokeStats | None]:
  """Constructs a single document represented by TensorFlow example.

  Args:
    doc_id: Unique document ID. An integer between 1 and `num_texts`.
    concept: Concept name (includes the POS tag).
    params: Document building parameters.

  Returns:
    A tuple with:
      - Document ID (same as the input one).
      - Corpus document in `tf.train.Example` format.
      - Stroke statistics (if present).
  """
  # Randomly sample number and concept.
  number = random.randint(1, generate_lib.FLAGS.max_commodity)
  unseen_concept = False
  if params.all_concepts.index(concept) >= len(params.concepts):
    unseen_concept = True
  concept_gloss = gloss_lib.find_gloss(concept, params.concept_glosses)
  if not concept_gloss:
    concept_gloss = concept.split("_")[0]

  # Prepare input features.
  sep = " "
  input_text = f"{number}{sep}{concept}"
  verbalized, sampa = params.text_generator.generate_transcriptions(input_text)

  all_input_emb = {}
  for embedding_type in embedder.EMBEDDING_TYPES:
    input_emb = params.all_embeddings[embedding_type].embed(
        input_text, embedding_type)
    if len(input_emb) != 2:
      raise ValueError(
          f"Suspicious \"{embedding_type}\" input embeddings with "
          f"length {len(input_emb)}!")
    all_input_emb[embedding_type] = input_emb

  # Prepare accounting document.
  if unseen_concept:
    concept_glyphs = [glyph_lib.DUMMY_GLYPH_NAME]
  else:
    concept_glyphs = params.text_generator.generate_glyphs(
        concept, strip_glyph_suffix=True
    )
  if not concept_glyphs:
    raise ValueError("Invalid glyph names!")
  doc_numbers = list(numbers_to_glyphs.pseudo_roman(number))
  all_glyphs = doc_numbers
  glyphs_number_mask = [True] * len(doc_numbers)
  all_glyphs.extend(concept_glyphs)
  num_concept_glyphs = len(concept_glyphs)
  strokes, stroke_glyph_affiliations, num_concept_glyphs = _build_strokes(
      concept, params.glyph_vocab, all_glyphs, num_concept_glyphs,
  )
  glyphs_number_mask.extend([False] * num_concept_glyphs)

  # Prepare glyph tokens and concept ID.
  glyph_tokens, glyph_types_mask = params.glyph_vocab.tokenize(
      all_glyphs, glyphs_number_mask,
  )
  concept_id = params.concepts_vocab[concept][0]

  # Lookup vision embeddings.
  vision_features = params.vision_features.features_for_concept(concept)

  # Create single TF record.
  features_dict = {
      "doc/id": utils.int64_feature([doc_id]),  # ID of this record.
      "concept/name": utils.bytes_feature([str.encode(concept)]),
      "concept/unseen": utils.int64_feature([unseen_concept]),
      "concept/id": utils.int64_feature([concept_id]),
      "number/name": utils.bytes_feature([str.encode(str(number))]),
      "text/glyph/tokens": utils.int64_feature(glyph_tokens),
      "text/glyph/types": utils.int64_feature(glyph_types_mask),
      "text/sampa": utils.bytes_feature([str.encode(sampa)]),
      "text/words": utils.bytes_feature([str.encode(verbalized)]),
      "text/gloss": utils.bytes_feature([str.encode(concept_gloss)]),
      **vision_features,
  }

  for embedding_type in embedder.EMBEDDING_TYPES:
    emb = all_input_emb[embedding_type]
    features_dict.update({
        f"text/{embedding_type}/emb_dim": utils.int64_feature([emb[0].size]),
        f"text/{embedding_type}/number_emb": utils.float_feature(emb[0]),
        f"text/{embedding_type}/concept_emb": utils.float_feature(emb[1]),
    })

  phon_emb = []
  for word in phrase_lib.phrase_to_words(verbalized):
    phon_emb.append(params.phonetic_embeddings.embedding(word))
  phon_emb_dims, phon_emb = utils.flatten_embedding(phon_emb)
  features_dict.update({
      "text/phonetic_embedding/emb_dim": utils.int64_feature(phon_emb_dims),
      "text/phonetic_embedding/emb": utils.float_feature(phon_emb),
  })

  # Construct the x and y stroke vectors, interspersing each stroke
  # with a end-of-stroke token. Record glyph affiliation for each
  # resulting point as well.
  stroke_glyph_affiliations_ids = []
  for text_pos, glyph_name in stroke_glyph_affiliations:
    stroke_glyph_affiliations_ids.append((
        text_pos, params.glyph_vocab.name_to_id(glyph_name)
    ))
  (
      x_stroke_points, y_stroke_points, stroke_stats,
      glyph_affiliations_text_pos, glyph_affiliations_ids
  ) = stroke_utils.stroke_points(strokes, stroke_glyph_affiliations_ids)
  npoints = len(x_stroke_points)
  features_dict.update({
      "strokes/npoints": utils.int64_feature([npoints]),
      "strokes/x_stroke_points": utils.float_feature(x_stroke_points),
      "strokes/y_stroke_points": utils.float_feature(y_stroke_points),
      "strokes/glyph_affiliations/text_pos": utils.int64_feature(
          glyph_affiliations_text_pos
      ),
      "strokes/glyph_affiliations/ids": utils.int64_feature(
          glyph_affiliations_ids
      ),
  })

  example = tf.train.Example(features=tf.train.Features(
      feature=features_dict))
  return doc_id, example, stroke_stats


def init_params(
    phonetic_embeddings_path: str,
    concepts_vocab_file: str
) -> Params:
  """Initializes all the bits required for generating documents.

  Args:
    phonetic_embeddings_path: Path to phonetic embeddings.
    concepts_vocab_file: Concept vocabulary file in JSON format.

  Returns:
    Initialized document builder params.

  Raises:
    ValueError if unrecoverable errors are encountered.
  """
  supercategories, concepts = generate_lib.load_concepts(
      generate_lib.FLAGS.concepts)

  _, unseen_concepts = generate_lib.load_concepts(
      generate_lib.FLAGS.unseen_concepts)
  common_concepts = set(concepts).intersection(set(unseen_concepts))
  if common_concepts:
    raise ValueError("Concept duplication between seen and unseen sets: "
                     f"{common_concepts}")
  all_concepts = concepts + unseen_concepts
  concepts_vocab = _build_concepts_vocab(
      concepts, unseen_concepts, concepts_vocab_file
  )
  glyph_vocab = glyph_lib.load_or_build_glyph_vocab()
  concept_glosses = gloss_lib.read_glosses()

  # Load the various embeddings.
  logging.info("Loading embeddings ...")
  all_embeddings = {}
  for embedding_type in embedder.EMBEDDING_TYPES:
    all_embeddings[embedding_type] = embedder.load_embeddings_from_type(
        embedding_type
    )
  logging.info("Done.")

  logging.info("Loading the language parameters ...")
  text_generator = generate_lib.TextGenerator()
  text_generator.initialize_for_generation()
  phonetic_embeddings = phon_embed_lib.load_phonetic_embedder(
      phonetic_embeddings_path
  )

  vision_features = vision.VisionFeaturesReader()
  vision_features.init()

  return Params(
      supercategories=supercategories,
      concepts=concepts,
      unseen_concepts=unseen_concepts,
      all_concepts=all_concepts,
      concepts_vocab=concepts_vocab,
      glyph_vocab=glyph_vocab,
      concept_glosses=concept_glosses,
      all_embeddings=all_embeddings,
      text_generator=text_generator,
      vision_features=vision_features,
      phonetic_embeddings=phonetic_embeddings,
  )
