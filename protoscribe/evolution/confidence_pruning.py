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

"""Utilities for pruning the result dictionaries based on their confidence."""

import enum
import logging
import math
from typing import Any

import numpy as np
import scipy

_EPSILON = np.finfo(np.float32).eps

# Default parameters for various pruning methods.
DEFAULT_MIN_CONFIDENCE_THRESHOLD = 0.
DEFAULT_MIN_PROBABILITY = float(_EPSILON)
DEFAULT_TOP_K = 400
DEFAULT_TOP_PERCENTAGE = 70
DEFAULT_MAX_CUMULATIVE_PROBABILITY = 1.


class Method(enum.StrEnum):
  """Method for pruning the list of best hypotheses for the test set.

  Needs to be derived from `StrEnum` because we use this in flags which are
  passed around as strings both parsed and unparsed.
  """
  # No pruning. Pass through all the results.
  NONE = "none"

  # Prune all the candidates below a certain confidence threshold, which is
  # interpreted as an absolute value.
  THRESHOLD = "threshold"

  # Convert the absolute confidence values to distribution over the test set
  # and prune by probability threshold.
  PROBABILITY = "probability"

  # Keep top-K best results according to absolute value of the confidence.
  TOP_K = "top_k"

  # Keep given percentage of the best results.
  TOP_PERCENTAGE = "top_percentage"

  # Finds the top results with cumulative probability mass smaller than a
  # specified probability threshold (technique from nucleus sampling).
  TOP_P = "top_p"


def confidence(results: dict[str, Any]) -> float:
  """Retrieves confidence from the results."""
  conf = results["glyph.confidence"]
  if not isinstance(conf, float):
    raise ValueError("Invalid variable type for confidence!")
  return conf


def confidence_distribution(
    all_docs: list[dict[str, Any]],
    log_domain: bool,
    use_softmax: bool
) -> list[float]:
  """Converts the space of all best test confidences to distribution.

  Args:
    all_docs: List of accounting documents from the test set represented as
      dictionaries.
    log_domain: Whether to return log probabilities or regular probabilities.
    use_softmax: Use `softmax` instead of simple normalization to [0, 1] range.

  Returns:
    A list of (possibly log) probabilities.
  """
  confidences = [confidence(cur_dict) for cur_dict in all_docs]
  confidences = np.array(confidences, dtype=np.float64)
  if use_softmax:
    if log_domain:
      vals = scipy.special.log_softmax(confidences)
    else:
      vals = scipy.special.softmax(confidences)
  else:
    vals = confidences / np.sum(confidences)
    if log_domain:
      vals = np.log(vals + _EPSILON)

  return vals.tolist()


def _renormalize_probs(pruned_dicts: dict[str, dict[str, Any]]) -> None:
  """Renormalizes probability mass of the pruned dictionaries.

  Args:
    pruned_dicts: Result dictionaries containing log probabilities, among
      other things.

  Raises:
    ValueError if log probability is not present or renormalization is not
    possible.
  """
  sum_prob = 0.
  for key in sorted(pruned_dicts.keys()):
    if "log_prob" not in pruned_dicts[key]:
      raise ValueError(f"{key}: Log probability not present!")
    sum_prob += np.exp(pruned_dicts[key]["log_prob"])

  if sum_prob > 1.:
    raise ValueError(f"Invalid initial probability mass: {sum_prob}")
  for key in sorted(pruned_dicts.keys()):
    prob = np.exp(pruned_dicts[key]["log_prob"])
    pruned_dicts[key]["log_prob"] = np.log(
        prob + (1. - sum_prob) / len(pruned_dicts)
    )


def _prune_by_threshold(
    all_docs: list[dict[str, Any]], min_confidence_threshold: float
) -> dict[str, dict[str, Any]]:
  """Prunes the list of results using a fixed confidence threshold.

  Args:
    all_docs: List of accounting documents from the test set represented as
      dictionaries.
    min_confidence_threshold: Prune all the hypotheses that are below this
      value.

  Returns:
    Mapping between concepts and the corresponding best (according to the
    specified pruning method) accounting document for that concept from the
    test set.

  Raises:
    ValueError if the input docs are not deduped by concept name.
  """
  best_docs = {}
  num_dropped = 0
  for cur_dict in all_docs:
    current_confidence = confidence(cur_dict)
    if current_confidence < min_confidence_threshold:
      num_dropped += 1
      continue
    concept = cur_dict["concept.name"]
    if concept in best_docs:
      raise ValueError(f"Duplicate concept: {concept}!")
    else:
      best_docs[concept] = cur_dict

  logging.info(
      "Proposing %d extensions. Pruned %d low-confidence ones.",
      len(best_docs), num_dropped
  )
  return best_docs


def _prune_by_probability(
    all_docs: list[dict[str, Any]], min_prob: float, use_softmax: bool
) -> dict[str, dict[str, Any]]:
  """Prunes the list of results using a minimal probability threshold.

  Args:
    all_docs: List of accounting documents from the test set represented as
      dictionaries.
    min_prob: Weed out all the entries whose probability as below this
      threshold.
    use_softmax: Use `softmax` instead of simple normalization to [0, 1] range.

  Returns:
    Mapping between concepts and the corresponding best (according to the
    specified pruning method) accounting document for that concept from the
    test set.

  Raises:
    ValueError if the input docs are not deduped by concept name.
  """
  log_probs = confidence_distribution(
      all_docs, log_domain=True, use_softmax=use_softmax
  )
  log_min_prob = np.log(min_prob)
  best_docs = {}
  num_dropped = 0
  for i, cur_dict in enumerate(all_docs):
    if log_probs[i] < log_min_prob:
      num_dropped += 1
      continue
    concept = cur_dict["concept.name"]
    if concept in best_docs:
      raise ValueError(f"Duplicate concept: {concept}!")
    else:
      best_docs[concept] = cur_dict
      best_docs[concept]["log_prob"] = log_probs[i]

  _renormalize_probs(best_docs)
  logging.info(
      "Proposing %d extensions. Pruned %d low-probability ones.",
      len(best_docs), num_dropped
  )
  return best_docs


def _prune_by_top_k(
    all_docs: list[dict[str, Any]], top_k: int
) -> dict[str, dict[str, Any]]:
  """Prunes the list of results keeping the top K best.

  Args:
    all_docs: List of accounting documents from the test set represented as
      dictionaries.
    top_k: An integer specifying the top-K results to retain.

  Returns:
    Mapping between concepts and the corresponding best (according to the
    specified pruning method) accounting document for that concept from the
    test set.

  Raises:
    ValueError if the input docs are not deduped by concept name.
  """
  # Construct a mapping between (best) concepts and result dictionaries.
  unique_dicts = {}
  for cur_dict in all_docs:
    concept = cur_dict["concept.name"]
    if concept in unique_dicts:
      raise ValueError(f"Duplicate concept: {concept}!")
    else:
      unique_dicts[concept] = cur_dict

  # Sort the results by absolute values of the confidence and select top-K.
  concept_confs = [(conc, confidence(d)) for conc, d in unique_dicts.items()]
  concept_confs.sort(key=lambda x: x[1], reverse=True)
  top_k = len(concept_confs) if top_k > len(concept_confs) else top_k
  num_dropped = len(concept_confs) - top_k
  best_docs = {}
  for concept, _ in concept_confs[:top_k]:
    best_docs[concept] = unique_dicts[concept]

  logging.info(
      "Proposing %d extensions. Pruned %d low-confidence ones.",
      len(best_docs), num_dropped
  )
  return best_docs


def _prune_by_top_percentage(
    all_docs: list[dict[str, Any]], top_percentage: int
) -> dict[str, dict[str, Any]]:
  """Prunes the list of results keeping the top K best.

  Args:
    all_docs: List of accounting documents from the test set represented as
      dictionaries.
    top_percentage: A number between 0 and 100 specifying the percentage of
      best results to keep.

  Returns:
    Mapping between concepts and the corresponding best (according to the
    specified pruning method) accounting document for that concept from the
    test set.

  Raises:
    ValueError if the input docs are not deduped by concept name.
  """
  concepts = set([d["concept.name"] for d in all_docs])
  top_k = math.ceil(len(concepts) * float(top_percentage) / 100.)
  return _prune_by_top_k(all_docs, top_k=top_k)


def _prune_by_top_p(
    all_docs: list[dict[str, Any]],
    max_cumulative_prob: float,
    use_softmax: bool
) -> dict[str, dict[str, Any]]:
  """Prunes the list of results using a minimal cumulative probability.

  This is part of Nucleus Sampling introduced in "The Curious Case of Neural
  Text Degeneration" (2019): https://arxiv.org/abs/1904.09751

  Args:
    all_docs: List of accounting documents from the test set represented as
      dictionaries.
    max_cumulative_prob: Select results whose cumulative probability is below
      this threshold. Lower values mean sampling from a smaller, more
      top-weighted nucleus.
    use_softmax: Use `softmax` instead of simple normalization to [0, 1] range.

  Returns:
    Mapping between concepts and the corresponding best (according to the
    specified pruning method) accounting document for that concept from the
    test set.

  Raises:
    ValueError if the input docs are not deduped by concept name.
  """

  # Determine the nucleus which should be a set of all results with sorted
  # cumulative probabilities below `p` given by `max_cumulative_prob`. Elements
  # outside of nucleus are discarded.
  probs = confidence_distribution(
      all_docs, log_domain=False, use_softmax=use_softmax
  )
  ids_and_probs = [(i, p) for i, p in enumerate(probs)]
  ids_and_probs.sort(key=lambda x: x[1], reverse=True)  # Sort descending.
  sorted_probs = [p for _, p in ids_and_probs]
  sorted_cum_probs = np.cumsum(sorted_probs)
  cutoff_index = np.sum(sorted_cum_probs <= max_cumulative_prob)
  nucleus_idx = [idx for idx, _ in ids_and_probs[:cutoff_index]]

  # Assemble best results dict.
  best_docs = {}
  num_dropped = 0
  for i, cur_dict in enumerate(all_docs):
    if i not in nucleus_idx:
      # Drop non-nucleus hypotheses.
      num_dropped += 1
      continue

    log_prob = np.log(probs[i])
    concept = cur_dict["concept.name"]
    if concept in best_docs:
      raise ValueError(f"Duplicate concept: {concept}!")
    else:
      best_docs[concept] = cur_dict
      best_docs[concept]["log_prob"] = log_prob

  _renormalize_probs(best_docs)
  logging.info(
      "Proposing %d extensions. Pruned %d low-confidence ones.",
      len(best_docs), num_dropped
  )
  return best_docs


def dedup_by_concepts(
    all_docs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
  """Removes all documents with the same concept keeping the best one.

  Args:
    all_docs: List of *all* the accounting documents from the test set
      represented as dictionaries. These have documents with duplicate
      concepts.

  Returns:
    A list of accounting documents corresponding to unique concepts.
  """
  best_docs = {}
  for cur_doc in all_docs:
    concept = cur_doc["concept.name"]
    if concept not in best_docs:
      best_docs[concept] = cur_doc
      continue

    # Promote the results to best hypothesis if their confidence is higher
    # than the current best.
    cand_confidence = confidence(cur_doc)
    cur_confidence = confidence(best_docs[concept])
    if cand_confidence > cur_confidence:
      best_docs[concept] = cur_doc

  results = [best_docs[concept] for concept in sorted(best_docs.keys())]
  logging.info(
      "Deduped %d docs into %d (dropped %d).",
      len(all_docs), len(results), len(all_docs) - len(results)
  )
  return results


def prune_docs(
    all_docs: list[dict[str, Any]],
    pruning_method: Method,
    use_softmax: bool = True,
    min_confidence_threshold: float = DEFAULT_MIN_CONFIDENCE_THRESHOLD,
    min_prob: float = DEFAULT_MIN_PROBABILITY,
    top_k: int = DEFAULT_TOP_K,
    top_percentage: int = DEFAULT_TOP_PERCENTAGE,
    max_cumulative_prob: float = DEFAULT_MAX_CUMULATIVE_PROBABILITY
) -> dict[str, dict[str, Any]]:
  """Given the list of parsed results prunes them according to some criterion.

  Args:
    all_docs: List of accounting documents from the test set represented as
      dictionaries.
    pruning_method: An enum specifying the pruning method.
    use_softmax: For pruning methods that use probabilities, when converting the
      confidences to distribution use `softmax` function instead of plain
      normalization to [0, 1] interval.
    min_confidence_threshold: When pruning by confidence threshold, prune all
      the hypotheses that are below this value (used in `THRESHOLD` pruning).
    min_prob: Prune all the hypotheses that are below this probability (used
      in `PROBABILITY` pruning).
    top_k: An integer specifying the top-K results to retain (used in `TOP_K`
      pruning).
    top_percentage: An integer between 0 and 100 specifying the percentage of
      best results to keep.
    max_cumulative_prob: Select results whose cumulative probability is below
      this threshold.

  Returns:
    Mapping between concepts and the corresponding best (according to the
    specified pruning method) accounting document for that concept from the
    test set.

  Raises:
    ValueError if unsupported pruning is requested.
  """
  if pruning_method == Method.NONE:
    # Ignore confidences altogether. Return the mapping between concept names
    # and results where non-unique entries are simply ignored.
    best_docs = {}
    for d in all_docs:
      concept_name = d["concept.name"]
      if concept_name not in best_docs:
        best_docs[concept_name] = d
    return best_docs

  best_docs = {}
  deduped_docs = dedup_by_concepts(all_docs)
  if pruning_method == Method.THRESHOLD:
    best_docs = _prune_by_threshold(
        deduped_docs, min_confidence_threshold=min_confidence_threshold
    )
  elif pruning_method == Method.PROBABILITY:
    best_docs = _prune_by_probability(
        deduped_docs, min_prob=min_prob, use_softmax=use_softmax
    )
  elif pruning_method == Method.TOP_K:
    best_docs = _prune_by_top_k(deduped_docs, top_k=top_k)
  elif pruning_method == Method.TOP_PERCENTAGE:
    best_docs = _prune_by_top_percentage(
        deduped_docs, top_percentage=top_percentage
    )
  elif pruning_method == Method.TOP_P:
    best_docs = _prune_by_top_p(
        deduped_docs,
        max_cumulative_prob=max_cumulative_prob,
        use_softmax=use_softmax
    )
  else:
    raise ValueError(f"Unsupported pruning method: {pruning_method}")

  return best_docs
