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

"""Category generation.

Creates three sets of categories:

  1. Administrative categories, possibly with supercategories.
  2. Non-administrative categories, which are still assumed to be
     readily depictable.
  3. Other categories.
"""

import collections
import csv

from absl import app
from absl import flags

import glob
import os

BASE_ = "protoscribe/data"

flags.DEFINE_string("gardiner", f"{BASE_}/concepts/gardiner_concepts.tsv",
                    "TSV with concepts based on Gardiner's hieroglyph list.")
flags.DEFINE_string("categories", f"{BASE_}/concepts/concept_categories.tsv",
                    "TSV with hierarchical categories.")
flags.DEFINE_string(
    "additional_non_administrative_concepts",
    f"{BASE_}/concepts/additional_non_administrative_concepts.txt",
    (
        "Additional hand-created set of "
        "`non-administrative` but depictable concepts."
    ),
)
flags.DEFINE_string("embeddings", f"{BASE_}/semantics/bnc/embeddings.txt",
                    "Path to embeddings with frequencies.")
flags.DEFINE_string("administrative",
                    f"{BASE_}/concepts/administrative_categories.txt",
                    "Path to administrative categories.")
flags.DEFINE_string("non_administrative",
                    f"{BASE_}/concepts/non_administrative_categories.txt",
                    "Path to non-administrative categories.")
flags.DEFINE_string("other", f"{BASE_}/concepts/other_categories.txt",
                    "Path to other categories.")

FLAGS = flags.FLAGS


def load_gardiner():
  """Loads TSV derived from the Gardiner hieroglyph list.

  Returns:
    Two sets, one of non-administrative and the other of administrative
    categories.
  """
  administrative = set()
  non_administrative = set()
  with open(FLAGS.gardiner) as stream:
    reader = csv.reader(stream, delimiter="\t", quotechar='"')
    rows = [row for row in reader]
    for row in rows[1:]:
      concepts, administrative_q = row[2], row[3]
      if administrative_q == "Yes":
        category = administrative
      else:
        category = non_administrative
      for concept in concepts.split(","):
        concept = concept.strip()
        if concept:
          category.add(concept)
  return administrative, non_administrative


def load_categories():
  """Loads TSV of hierarchical categories.

  Returns:
    A defaultdict mapping categories to supercategories.
  """
  supercategories = collections.defaultdict(str)
  with open(FLAGS.categories) as stream:
    reader = csv.reader(stream, delimiter="\t", quotechar='"')
    rows = [row for row in reader][1:]
    for row in rows:
      categories, instances = "", ""
      try:
        categories, instances = row[1:]
      except ValueError:
        continue
      categories = [c.strip() for c in categories.split(",")]
      instances = [i.strip() for i in instances.split(",")]
      for category in categories:
        for instance in instances:
          supercategories[instance] = category
  return supercategories


def load_embeddings():
  """Loads embeddings.

  Returns:
    A set of terms in the embeddings.
  """
  embeddings = set()
  with open(FLAGS.embeddings) as stream:
    for line in stream:
      line = line.split()
      embeddings.add(line[1])
  return embeddings


def load_additional_non_administrative_concepts(concepts):
  """Loads a set of additional non-administrative concepts.

  Args:
    concepts: a set

  Returns:
    A set
  """
  with open(FLAGS.additional_non_administrative_concepts) as stream:
    for line in stream:
      concepts.add(line.strip())
  return concepts


def find_embedding_entry(category, embeddings):
  """Finds a category in the embeddings.

  Args:
    category: a category
    embeddings: set of embedding terms

  Returns:
    A string or None.
  """
  if category in embeddings:
    return category
  elif f"{category}_NOUN" in embeddings:
    return f"{category}_NOUN"
  elif f"{category}_ADJ" in embeddings:
    return f"{category}_ADJ"
  elif f"{category}_VERB" in embeddings:
    return f"{category}_VERB"
  return None


def output_categories(categories,
                      supercategories,
                      embeddings,
                      stream,
                      find_supercategories=False):
  """Writes category set to a stream.

  Args:
    categories: set of categories
    supercategories: defaultdict of supercategories
    embeddings: set of embedding terms
    stream: a stream
    find_supercategories: boolean
  Returns:
    set of categories that have been used
  """
  used_categories = set()
  for category in categories:
    embedding_entry = find_embedding_entry(category, embeddings)
    if not embedding_entry:
      continue
    if find_supercategories:
      supercategory = supercategories[category]
      supercategory_embedding_entry = find_embedding_entry(
          supercategory, embeddings)
      if supercategory_embedding_entry:
        if embedding_entry not in used_categories:
          stream.write(f"{supercategory_embedding_entry}\t{embedding_entry}\n")
        if supercategory_embedding_entry not in used_categories:
          stream.write(f"{supercategory_embedding_entry}\n")
        used_categories.add(embedding_entry)
        used_categories.add(supercategory_embedding_entry)
      else:
        if embedding_entry not in used_categories:
          stream.write(f"{embedding_entry}\n")
        used_categories.add(embedding_entry)
    else:
      if embedding_entry not in used_categories:
        stream.write(f"{embedding_entry}\n")
      used_categories.add(embedding_entry)
  return used_categories


def main(unused_argv):
  administrative, non_administrative = load_gardiner()
  non_administrative = load_additional_non_administrative_concepts(
      non_administrative
  )
  supercategories = load_categories()
  embeddings = load_embeddings()
  # Assume that anything in the set of listed supercategories can also
  # be an administrative category.
  for category in supercategories:
    administrative.add(category)
  with open(FLAGS.administrative, "w") as stream:
    used_categories = output_categories(
        administrative,
        supercategories,
        embeddings,
        stream,
        find_supercategories=True)
  with open(FLAGS.non_administrative, "w") as stream:
    used_categories = used_categories.union(
        output_categories(
            non_administrative,
            supercategories,
            embeddings,
            stream,
            find_supercategories=False))
  with open(FLAGS.other, "w") as stream:
    for embedding in embeddings:
      if embedding not in used_categories:
        stream.write(f"{embedding}\n")


if __name__ == "__main__":
  app.run(main)
