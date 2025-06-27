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

r"""Node2Vec embeddings for WordNet.

An option to construct an undirected or directed acyclic graph (DAG) is
provided. According to "Learning Linguistic Tree Structures with Text and Graph
Methods" (https://www.skoltech.ru/app/data/uploads/2022/09/thesis6.pdf)
undirected graphs may yield better results on some tasks.

TODO: This does NOT cope with WordNet adjectives.

Example:
--------
python protoscribe/semantics/wordnet/node2vec_embeddings_main.py \
  --output_embeddings_file /tmp/node2vec_nodag_embeddings.txt \
  --noconstruct_dag \
  --logtostderr
"""

from collections.abc import Sequence
import logging
import tempfile

from absl import app
from absl import flags
import node2vec

from protoscribe.semantics.wordnet import utils
from protoscribe.semantics.wordnet import wordnet_utils

_EMBEDDING_DIM = flags.DEFINE_integer(
    "embedding_dim", 300,
    "Dimension for embedding vectors.",
)

_WALK_LENGTH = flags.DEFINE_integer(
    "walk_length", 30,
    "Length of a random walk."
)

_NUM_WALKS = flags.DEFINE_integer(
    "num_walks", 200,
    "Number of random walks."
)

_NUM_WORKERS = flags.DEFINE_integer(
    "num_workers", 1,
    "Number of threads for training the model."
)

_NUM_EPOCHS = flags.DEFINE_integer(
    "num_epochs", 20,
    "Number of epochs for training."
)

_CONSTRUCT_DAG = flags.DEFINE_boolean(
    "construct_dag", True,
    "Construct directed acyclic graph of WordNet relations."
)

_OUTPUT_EMBEDDINGS_FILE = flags.DEFINE_string(
    "output_embeddings_file", None,
    "Path to the output model.",
    required=True
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  logging.info("Building WordNet graph ...")
  dag = wordnet_utils.build_graph(construct_dag=_CONSTRUCT_DAG.value)
  with tempfile.TemporaryDirectory() as temp_dir_name:
    logging.info("Constructing model ...")
    model = node2vec.Node2Vec(
        dag,
        dimensions=_EMBEDDING_DIM.value,
        walk_length=_WALK_LENGTH.value,
        num_walks=_NUM_WALKS.value,
        workers=_NUM_WORKERS.value,
        temp_folder=temp_dir_name,
    )
    logging.info("Training model ...")
    word2vec = model.fit(
        iter=_NUM_EPOCHS.value, workers=_NUM_WORKERS.value
    )

  logging.info("Saving embeddings to %s ...", _OUTPUT_EMBEDDINGS_FILE.value)
  utils.save_keyvec_embeddings(
      word2vec.wv,
      output_file=_OUTPUT_EMBEDDINGS_FILE.value
  )


if __name__ == "__main__":
  app.run(main)
