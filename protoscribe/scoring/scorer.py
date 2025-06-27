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

r"""Tools for scoring inference outputs.

An output from t5x/seqIO-based pipelines in JSONL format, where a single
line is a dictionary:

    {
      "doc.id", 1234,
      "number.name": "8",
      "concept.name": "bride_NOUN",
      "text.words": "k e j # p e k",
      "glyph.names": [["V", "I", "I", "I", "cloth", "trousers"]],
      "glyph.prons": [[[[], [], [], [], ["a", "k"], ["j", "u"]]
    }

Statistics collected:
---------------------
- `total_num_match↑`:
    Cases where the number glyphs correctly match, total and mean.

- `total_multi_glyph↑`:
    Cases where a multiple glyph is predicted for the concept, total and mean.

- `sem_top_k_score↓`:
    For predicted concept glyph g, let i be the index of g in the top k glyphs
    closest semantically to the target concept, or 10 if g is not in the top k
    glyphs. For each text, find the minimum i over the concept glyphs. Compute
    the total and mean over all texts.

- `sem_top_k_accuracy↑`:
    Predicted glyph is counted as correct if its semantics is within the top-k
    closest concepts. Note, similar to above, this is a text-level rather than
    glyph-level metric.

- `phon_top_k_score↓`:
    Same as `sem_top_k_score↓`, but computed over phonetic closeness.

- `phon_top_k_accuracy↑`:
    Same as `sem_top_k_accuracy↑`, but computed over phonetic closeness.

- `sem_phon_top_k_accuracy↑`:
    Prediction is correct if within both top-k semantically and phonetically
    closest concepts.

- 'phone_error_rate↓': Basic phone error rate computed between the truth and
    hypothesis pronunciations.
"""

import collections
import dataclasses
import json
import os
from typing import Any

from absl import flags
import jiwer
from protoscribe.glyphs import numbers_to_glyphs
from protoscribe.language.embeddings import embedder as embedder_lib
from protoscribe.language.phonology import phonetic_embeddings as phonetic_embedder_lib
from protoscribe.language.phonology import sampa as sampa_lib
from protoscribe.texts import generate as generate_lib

import glob
import os

_K_NEAREST_CONCEPTS = flags.DEFINE_integer(
    "k_nearest_concepts",
    5,
    "Number of nearest concepts to retrieve."
)

_K_NEAREST_PRONS = flags.DEFINE_integer(
    "k_nearest_prons",
    5,
    "Number of nearest pronunciations to retrieve."
)

_CONCEPT_EMBEDDING_TYPE = flags.DEFINE_string(
    "concept_embedding_type",
    "bnc",
    "Type of semantic embeddings. Note, it's currently not possible to score "
    "multiple systems produced with different semantic embedding types."
)

SCORE_TYPES = [
    "total_num_match↑",
    "total_multi_glyph↑",
    "sem_top_k_score↓",
    "phon_top_k_score↓",
    "sem_top_k_accuracy↑",
    "phon_top_k_accuracy↑",
    "sem_phon_top_k_accuracy↑",
    "phone_error_rate↓",
]


@dataclasses.dataclass(frozen=True)
class Score:
  num_match: bool
  multi_glyph: bool
  sem_top_k_score: int
  sem_top_k_accuracy: int
  phon_top_k_score: int
  phon_top_k_accuracy: int
  sem_phon_top_k_accuracy: int


class Scorer:
  """Container for scoring methods."""

  def __init__(self):
    self.cumulated_scores = dict(map(lambda x: (x, 0), SCORE_TYPES))
    self.cumulated_scores["total"] = 0

  def reset(self) -> None:
    for k in self.cumulated_scores:
      self.cumulated_scores[k] = 0

  def find_score(self, which: str) -> tuple[int, int, float]:
    """Finds the total count and mean for a given score.

    Args:
      which: which score to access.

    Returns:
      A tuple of sample counts (number of inferred documents), total cumulated
      score and its mean.
    Raises:
      ValueError if total is 0 or `which` is not a valid score name.
    """
    if self.cumulated_scores["total"] == 0:
      raise ValueError("Total must not be 0")
    if which not in self.cumulated_scores.keys():
      raise ValueError(f"{which=} is not found in cumulated_scores")
    num_examples = self.cumulated_scores["total"]
    total = self.cumulated_scores[which]
    mean = total / num_examples
    return num_examples, total, mean

  def eval_single_prediction(
      self,
      pred_dict: dict[str, Any],
      cumulate_scores: bool = False,
  ) -> Score:
    """Evaluates a single prediction represented as a dictionary.

    Args:
      pred_dict: Prediction context dictionary.
      cumulate_scores: If True, cumulate the scores.

    Returns:
      A Score object.
    """
    inp_sem = pred_dict["concept.name"].split("_")[0]
    sem_top_k = [c.split("_")[0] for c in pred_dict["semantics_knn"]]
    if inp_sem not in sem_top_k:
      sem_top_k = [inp_sem] + sem_top_k

    inp_phon = _last_word_pron(pred_dict["text.words"]).replace(" ", "")
    phon_top_k = pred_dict["phonetics_knn"]
    if inp_phon not in phon_top_k:
      phon_top_k = [inp_phon] + phon_top_k

    nominal_target_num = numbers_to_glyphs.pseudo_roman(
        int(pred_dict["number.name"]))
    pred_num, pred_sem = numbers_to_glyphs.number_and_other_glyphs(
        pred_dict["glyph.names"])
    num_of_number_glyphs = len(pred_num)
    pred_num = "".join(pred_num)
    num_match = True if nominal_target_num == pred_num else False
    multi_glyph = True if len(pred_sem) > 1 else False

    pred_phon = pred_dict["glyph.prons"][num_of_number_glyphs:]
    sem_mismatch_score = _K_NEAREST_CONCEPTS.value * 2
    sem_top_k_score = sem_mismatch_score
    for x in pred_sem:
      try:
        i = sem_top_k.index(x)
        if i < sem_top_k_score:
          sem_top_k_score = i
      except ValueError:
        pass
    sem_top_k_accuracy = 1 if sem_top_k_score != sem_mismatch_score else 0

    phon_mismatch_score = _K_NEAREST_PRONS.value * 2
    phon_top_k_score = phon_mismatch_score
    for x in pred_phon:
      try:
        i = phon_top_k.index(x)
        if i < phon_top_k_score:
          phon_top_k_score = i
      except ValueError:
        pass
    phon_top_k_accuracy = 1 if phon_top_k_score != phon_mismatch_score else 0
    sem_phon_top_k_accuracy = 0
    if sem_top_k_accuracy == 1 and phon_top_k_accuracy == 1:
      sem_phon_top_k_accuracy = 1

    score = Score(
        num_match=num_match,
        multi_glyph=multi_glyph,
        sem_top_k_score=sem_top_k_score,
        sem_top_k_accuracy=sem_top_k_accuracy,
        phon_top_k_score=phon_top_k_score,
        phon_top_k_accuracy=phon_top_k_accuracy,
        sem_phon_top_k_accuracy=sem_phon_top_k_accuracy,
    )
    if cumulate_scores:
      self.cumulated_scores["total"] += 1
      self.cumulated_scores["total_num_match↑"] += 1 if score.num_match else 0
      self.cumulated_scores["total_multi_glyph↑"] += (
          1 if score.multi_glyph else 0
      )
      self.cumulated_scores["sem_top_k_score↓"] += score.sem_top_k_score
      self.cumulated_scores["sem_top_k_accuracy↑"] += score.sem_top_k_accuracy
      self.cumulated_scores["phon_top_k_score↓"] += score.phon_top_k_score
      self.cumulated_scores["phon_top_k_accuracy↑"] += score.phon_top_k_accuracy
      self.cumulated_scores["sem_phon_top_k_accuracy↑"] += (
          score.sem_phon_top_k_accuracy
      )
    return score


class ComparativeScorer:
  """Provide scores for a set of directories or files."""

  def __init__(
      self,
      phoible_phonemes_path: str,
      phoible_features_path: str,
      phonetic_embeddings_path: str,
      main_lexicon_path: str,
      number_lexicon_path: str
  ) -> None:
    self.store = collections.defaultdict(lambda: collections.defaultdict(dict))
    self.scorers = {}

    # Caches for semantic and phonetic KNN lookups.
    self._semantics_knn_cache = {}
    self._phonetics_knn_cache = {}

    # Load semantic and phonetic embeddings.
    self._semantic_embedder = embedder_lib.load_embeddings_from_type(
        _CONCEPT_EMBEDDING_TYPE.value
    )
    self._phonetic_embedder = phonetic_embedder_lib.load_phonetic_embedder(
        phonetic_embeddings_path,
        phoible_phonemes_path=phoible_phonemes_path,
        phoible_features_path=phoible_features_path,
    )

    # Loading seen concepts (these are administrative concepts initially).
    seen_concepts_files = generate_lib.FLAGS.concepts
    if not seen_concepts_files:
      raise ValueError("Specify --concepts flag to load the seen concepts!")
    _, self._reference_concepts = generate_lib.load_concepts(
        seen_concepts_files
    )

    # Load all the seen phonetic forms used in the training.
    _, self._seen_phonetic_forms = generate_lib.load_phonetic_forms(
        seen_concepts=self._reference_concepts,
        main_lexicon_file=main_lexicon_path,
        number_lexicon_file=number_lexicon_path,
    )
    self._reference_concepts = set(self._reference_concepts)

  def reset(self) -> None:
    self.store = collections.defaultdict(lambda: collections.defaultdict(dict))
    self.scorers = {}

  def score_dirs(
      self,
      directories: list[str],
      pretty_names: list[str] | None = None,
      file_suffix: str = "json"
  ) -> None:
    """Scores all files in all directories, cumulating the scores for each.

    Args:
      directories: A list of directories.
      pretty_names: An optional list of more mnemonic names for each condition.
      file_suffix: Suffix for the input files.
    """
    if pretty_names:
      assert len(directories) == len(pretty_names)
    pnames = []
    for i in range(len(directories)):
      dpath = directories[i]
      pname = pretty_names[i] if pretty_names else dpath
      pnames.append(pname)
      files = glob.glob(os.path.join(dpath, f"*.{file_suffix}"))
      for path in files:
        with open(path) as stream:
          pred_dict = json.load(stream)
          self.store[pname][os.path.basename(path)] = pred_dict

    self._sanity_check_store(pnames)
    self._score_predictions(pnames)

  def score_jsonl_files(
      self,
      jsonl_files: list[str],
      pretty_names: list[str] | None = None,
  ) -> None:
    """Scores all entries in given files, cumulating the scores for each.

    Args:
      jsonl_files: A list of files in JSONL format.
      pretty_names: An optional list of more mnemonic names for each condition.
    """
    if pretty_names:
      assert len(jsonl_files) == len(pretty_names)
    pnames = []
    for i in range(len(jsonl_files)):
      file_path = jsonl_files[i]
      pname = pretty_names[i] if pretty_names else file_path
      pnames.append(pname)
      with open(file_path, "rt") as f:
        for idx, line in enumerate(f):
          line = line.strip()
          if not line:  # Skip blank lines.
            continue
          self.store[pname][str(idx)] = json.loads(line)

    self._sanity_check_store(pnames)
    self._score_predictions(pnames)

  def _sanity_check_store(self, pretty_names: list[str]) -> None:
    """Sanity check to make sure that the stores have the same entries."""
    valid_names = set(self.store[pretty_names[0]].keys())
    for pretty_name in pretty_names[1:]:
      new_names = set(self.store[pretty_name].keys())
      if valid_names.union(new_names) != valid_names:
        diff = valid_names.union(new_names).difference(valid_names)
        diff = "\n\t".join(diff)
        raise ValueError(
            f"System {pretty_name} has different entries from the "
            f"rest:\n{diff}"
        )

  def _score_predictions(self, pretty_names: list[str]) -> None:
    """Scores all currently stored predictions."""
    for pretty_name in pretty_names:
      scorer = Scorer()
      truth_prons = []
      hypothesis_prons = []
      for path in self.store[pretty_name]:
        pred_dict = self.store[pretty_name][path]
        pred_dict = self.prepare_for_scoring(pred_dict)
        scorer.eval_single_prediction(pred_dict, cumulate_scores=True)

        # Accumulate PER separately.
        truth_prons.append(_last_word_pron(pred_dict["text.words"]))
        if pred_dict["glyph.prons"]:
          hypothesis_prons.append(
              " ".join([c for c in pred_dict["glyph.prons"][-1]])
          )
        else:
          hypothesis_prons.append("<EMPTY>")

      # This is a hack: we multiply the real phone error rate (PER) by total
      # number of examples so that the final mean value which we are after will
      # be accurate.
      scorer.cumulated_scores["phone_error_rate↓"] = (
          scorer.cumulated_scores["total"] *
          jiwer.wer(truth_prons, hypothesis_prons)
      )
      self.scorers[pretty_name] = scorer

  def prepare_for_scoring(self, pred_dict: dict[str, Any]) -> dict[str, Any]:
    """Prepares the predictions for scoring by filling the missing bits."""

    # Select glyph names and pronunciations corresponding to best path.
    pred_dict["glyph.names"] = pred_dict["glyph.names"][-1]
    pred_dict["glyph.prons"] = pred_dict["glyph.prons"][-1]
    if len(pred_dict["glyph.names"]) != len(pred_dict["glyph.prons"]):
      raise ValueError("Lengths of glyph names and prons must match")

    # Compute k-nearest neighbors.
    semantics_knn = self._semantics_knn(pred_dict["concept.name"])
    phonetics_knn = self._phonetics_knn(pred_dict["text.words"])
    pred_dict.update({
        "semantics_knn": semantics_knn,
        "phonetics_knn": phonetics_knn,
    })

    # Massage pronunciations for individual predicted glyphs.
    glyph_prons = []
    for i, glyph_name in enumerate(pred_dict["glyph.names"]):
      pron = pred_dict["glyph.prons"][i]
      glyph_pron = "".join(pron) if pron else glyph_name
      glyph_prons.append(glyph_pron)
    pred_dict["glyph.prons"] = glyph_prons

    return pred_dict

  def _semantics_knn(self, concept_name: str) -> list[str]:
    """Computes k-nearest neighbors for semantics.

    Args:
      concept_name: Name of the concept.

    Returns:
      A list of nearest concepts sorted in increasing distance order.
    """
    if concept_name in self._semantics_knn_cache:
      return self._semantics_knn_cache[concept_name]

    concept_embedding = self._semantic_embedder[concept_name]
    nearest_concepts = self._semantic_embedder.get_k_nearest_neighbours(
        concept_embedding,
        _K_NEAREST_CONCEPTS.value,
        allowed_entries=self._reference_concepts
    )
    nearest_concepts = [concept for concept, dist in nearest_concepts]
    self._semantics_knn_cache[concept_name] = nearest_concepts
    return nearest_concepts

  def _phonetics_knn(self, word_prons: str) -> list[str]:
    """Computes k-nearest neighbors for pronunciations.

    Args:
      word_prons: Word transcription as strings. Individual syllables are
        separated by `#`.

    Returns:
      A list of nearest words sorted in increasing distance order.
    """
    if word_prons in self._phonetics_knn_cache:
      return self._phonetics_knn_cache[word_prons]

    last_word_pron = _last_word_pron(word_prons)
    closest_prons = self._phonetic_embedder.get_k_nearest_neighbors(
        last_word_pron,
        _K_NEAREST_PRONS.value,
        allowed_terms=self._seen_phonetic_forms
    )
    closest_prons = [pron.replace(" ", "") for pron, dist in closest_prons]
    self._phonetics_knn_cache[word_prons] = closest_prons
    return closest_prons


def _last_word_pron(word_prons: str) -> str:
  """Returns pronunciation of last word given the full pronunciation."""
  # TODO: This just gets the last word for looking up the closest
  # phonetic embeddings to the word of interest, but this is not general.
  return word_prons.split(sampa_lib.WORD_BOUNDARY)[-1].strip()
