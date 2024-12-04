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

r"""Poincare embedding model for WordNet relations.

For now the implementation focuses on hypernymy/hyponymy (class/subsumption)
relations.

TODO: This does NOT cope with WordNet adjectives.

Example:
--------
python protoscribe/semantics/wordnet/poincare_embeddings_main.py \
  --output_embeddings_file ~/tmp/poincare_embeddings.txt --logtostderr
"""

from collections.abc import Sequence
import logging

from absl import app
from absl import flags
from gensim.models.poincare import PoincareModel
from nltk.corpus import wordnet as wn

from protoscribe.semantics.wordnet import utils

Synset = wn.synset

_EMBEDDING_DIM = flags.DEFINE_integer(
    "embedding_dim", 300,
    "Dimension for embedding vectors.",
)

_NUM_EPOCHS = flags.DEFINE_integer(
    "num_epochs", 20,
    "Number of epochs for training."
)

_NUM_BURN_IN_EPOCHS = flags.DEFINE_integer(
    "num_burn_in_epochs", 10,
    "Number of burn-in epochs."
)

_OUTPUT_EMBEDDINGS_FILE = flags.DEFINE_string(
    "output_embeddings_file", None,
    "Path to the output model.",
    required=True
)

_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size", 10,
    "Number of training examples in a batch."
)

_NUM_WORKERS = flags.DEFINE_integer(
    "num_workers", 1,
    "Number of threads for training the model. Currently only one worker "
    "is supported by the model implementation."
)

_USE_CLOSURE = flags.DEFINE_bool(
    "use_closure", True,
    "Use closure of all instance hyponyms when computing the relations."
)

_BUGGY_SYNSETS = set([
    "restrain.v.01"
])


def _add_relations_for_node(
    node: Synset, pairs: set[tuple[str, str]]
) -> set[tuple[str, str]]:
  """Amends all the relation pairs for the given root."""
  for w in node.hyponyms():
    if w.name() in _BUGGY_SYNSETS:
      continue
    pairs.add((w.name(), node.name()))
    _add_relations_for_node(w, pairs)

  for w in node.lemmas():
    if w.name() in _BUGGY_SYNSETS:
      continue
    if w is not node:
      pairs.add((w.name(), node.name()))

  return pairs


def _add_relations_default() -> set[tuple[str, str]]:
  """Simple retrieval of relations."""
  logging.info("Retrieving all synsets ...")
  roots = list(wn.all_synsets())
  logging.info("Collecting relations from %d synsets ...", len(roots))
  relations = set()
  for root in roots:
    relations_for_root = _add_relations_for_node(root, set())
    if relations_for_root:
      relations.update(relations_for_root)
  return relations


def _add_relations_closure() -> set[tuple[str, str]]:
  """Relations retrieval using closure operations."""
  relations = set()
  for synset in wn.all_synsets():
    # Compute transitive  closure of all hypernyms of a synset.
    for hyper in synset.closure(lambda s: s.hypernyms()):
      relations.add((synset.name(), hyper.name()))

    # Compute transitive closure for all instances of a synset.
    for instance in synset.instance_hyponyms():
      for hyper in instance.closure(lambda s: s.instance_hypernyms()):
        relations.add((instance.name(), hyper.name()))
        for h in hyper.closure(lambda s: s.hypernyms()):
          relations.add((instance.name(), h.name()))
  return relations


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if _USE_CLOSURE.value:
    relations = _add_relations_closure()
  else:
    relations = _add_relations_default()
  logging.info("Assembled %d relations", len(relations))

  logging.info("Training the embeddings for %d epochs ...", _NUM_EPOCHS.value)
  model = PoincareModel(
      relations,
      size=_EMBEDDING_DIM.value,
      negative=10,
      burn_in=_NUM_BURN_IN_EPOCHS.value,
      workers=_NUM_WORKERS.value
  )
  model.train(epochs=_NUM_EPOCHS.value, batch_size=_BATCH_SIZE.value)

  logging.info("Saving embeddings to %s ...", _OUTPUT_EMBEDDINGS_FILE.value)
  utils.save_keyvec_embeddings(
      model.kv,
      output_file=_OUTPUT_EMBEDDINGS_FILE.value
  )


if __name__ == "__main__":
  app.run(main)
