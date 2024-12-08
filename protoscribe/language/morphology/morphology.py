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

"""Morphology definition.

Uses lexicon defined in morphemes module and Pynini's features and
paradigms models.
"""

import collections
import logging

from protoscribe.language.morphology import morphemes
from protoscribe.language.morphology import morphology_parameters_pb2

from google.protobuf import text_format
import pynini as py
from pynini.lib import byte
from pynini.lib import features
from pynini.lib import paradigms
import glob
import os

_BYTE_STAR = py.closure(byte.BYTE)
_DEFAULT_FAR_TYPE = "sttable"

Feature = features.Feature
MorphologyParameters = morphology_parameters_pb2.MorphologyParameters


def _featval(f: str, v: str) -> str:
  """Helper function to combine feature and value names.

  Args:
    f: feature name
    v: value name

  Returns:
    A string of the form `f=v`
  """
  return f"{f}={v}"


def _make_feature_settings(
    feats: list[Feature], setting: list[str], settings: list[list[str]]
) -> None:
  """Constructs all feature-value settings.

  Args:
    feats: list of Feature instances.
    setting: a list, the setting currently being constructed.
    settings: a list to accumulate the completed settings.
  """
  if not feats:
    settings.append(setting)
  else:
    feature = feats[0]
    for v in feature.values:
      fv = _featval(feature.name, v)
      _make_feature_settings(feats[1:], setting + [fv], settings)


class MorphologyError(Exception):  # pylint: disable=g-bad-exception-name
  pass


class BasicMorphologyBuilder:
  """Base class to control morphology construction.

  Main work is in the setup function.
  """

  def __init__(self) -> None:
    self._feats = []
    self._lemma_vector = None
    self._cat = None
    self._nogen = False
    self._zero_feat_vals = []
    self._bound = ""
    self._lexicon = None
    self._paradigm = None
    self._affixes = None
    self._all_forms_to_meanings = None

  def setup(
      self,
      lexicon: morphemes.Lexicon,
      affix_specification: str | list[str],
      bound: str,
      zero_feat_vals: list[tuple[str, str]],
      rule_templates: list[tuple[py.FstLike, py.FstLike, py.FstLike]]
  ) -> None:
    """Sets up the Paradigm of inflected forms.

    The system is currently set up to allow only suffixal morphology
    of the agglutinative type, with one suffix per feature/value
    specification (i.e. no portmanteau morphs). A traditional
    `inflectional` type morphology system can however be mimicked by
    passing appropriate rules.

    Args:
      lexicon: A lexicon as defined in the morpheme module.
      affix_specification: A list of templates or a path to saved affixes
      bound: boundary symbol. This defaults to "+" and should be disjoint from
        the set of phonemes used in the phonology.
      zero_feat_vals: optional list of feature-value pairs that have zero morphs
      rule_templates: list of tuples (tau (=phi X psi), lambda, rho) for
        context-dependent rewrite rules to be applied to word forms constructed
        by the Paradigm. The actual rules are built in this function since they
        depend upon the Category's sigma_star FST.
    """
    self._lexicon = lexicon
    self._stems = list(lexicon.forms_to_morphemes.keys())
    self._bound = bound
    self._zero_feat_vals = set(zero_feat_vals) if zero_feat_vals else set()
    self._rules = []
    assert self._cat is not None
    if rule_templates:
      for tau, lamda, rho in rule_templates:
        self._rules.append(py.cdrewrite(tau, lamda, rho, self._cat.sigma_star))
    ns = byte.NOT_SPACE
    # NB: this should not be cyclic, so allow a reasonable maximum
    delspace = py.closure(py.cross(" ", ""), 0, 5)
    multispace = delspace + " "
    normalize_whitespace = (
        py.cdrewrite(multispace, ns, ns, self._cat.sigma_star)
        @ py.cdrewrite(delspace, "[BOS]", ns, self._cat.sigma_star)
        @ py.cdrewrite(delspace, ns, "[EOS]", self._cat.sigma_star)
        # Clean up trailing space before boundary:
        @ py.cdrewrite(delspace, ns, self._bound, self._cat.sigma_star)
    ).optimize()
    self._rules.append(normalize_whitespace)
    self._affix_specification = affix_specification
    self._affixes = {}
    # If affix_specification is a string, then assume it's a path from which to
    # load saved affixes.
    if isinstance(self._affix_specification, str):
      self._nogen = True
      with open(self._affix_specification) as stream:
        for line in stream:
          line = line.strip("\n").split("\t")
          self._affixes[line[0]] = line[1]
    else:
      self._nogen = False
      # This presumes that the lexicon's phoible_templates has a phoneme set,
      # but this should be the case assuming that the lexicon was created with
      # morphemes.make_lexicon, otherwise by default even the lexicon's
      # phoible_templates would be none.
      max_affix_syllables = -1
      for spec in self._affix_specification:
        spec = spec.split(".")
        if len(spec) > max_affix_syllables:
          max_affix_syllables = len(spec)
      assert self._lexicon.phoible_templates is not None
      self._affix_fst = self._lexicon.phoible_templates.populate_syllables(
          self._lexicon.phoible_templates.phonemes,
          self._affix_specification,
          max_affix_syllables,
      )
      for feat in self._feats:
        for v in feat.values:
          self._affixes[_featval(feat.name, v)] = self.make_morph(
              (feat.name, v))
    stem = paradigms.make_byte_star_except_boundary()
    slots = []
    self._lemma_vector = features.FeatureVector(self._cat, *self._lemma_vector)
    settings = []
    _make_feature_settings(self._feats, [], settings)
    for setting in settings:
      suffixes = "".join([self._affixes[s] for s in setting])
      fv = features.FeatureVector(self._cat, *setting)
      slots.append((paradigms.suffix(suffixes, stem), fv))
    self._paradigm = paradigms.Paradigm(
        category=self._cat,
        slots=slots,
        lemma_feature_vector=self._lemma_vector,
        stems=self._stems,
        rules=self._rules)
    self._all_forms_to_meanings = None

  def make_morph(self, feat_val: tuple[str, str]) -> str:
    """Constructs morph for a given f-v pair given the affix_specification.

    If feature-value is in the set of zero feature values, returns
    just a boundary (empty morph). Note that we assume that inflected forms will
    look like the following:

    b a s+ s u f+ s u f

    Args:
      feat_val: a feature-value pair

    Returns:
      An FST representing the morph.
    Raises:
      MorphologyError: Fails if we are loading affixes from a lexicon.
    """
    if self._nogen:
      raise MorphologyError("Affix lexicon is fixed, cannot generate.")
    if feat_val in self._zero_feat_vals:
      return self._bound
    assert self._lexicon is not None
    assert self._lexicon.phoible_templates is not None
    form = self._lexicon.phoible_templates.randgen(self._affix_fst, 1)[0]
    return self._bound + " " + form

  @property
  def stems(self) -> list[list[str]]:
    return self._stems

  @property
  def cat(self) -> features.Category | None:
    return self._cat

  @property
  def lemma_vector(self) -> features.FeatureVector | None:
    return self._lemma_vector

  def analyze(self, word) -> list[paradigms.Analysis]:
    assert self._paradigm is not None
    return self._paradigm.analyze(word)

  def lemmatize(self, word) -> py.Fst | None:
    assert self._paradigm is not None
    return self._paradigm.lemmatize(word)

  def dump_affixes(self, path: str) -> None:
    assert self._affixes is not None
    with open(path, "w") as stream:
      for fv in self._affixes:
        stream.write(f"{fv}\t{self._affixes[fv]}\n")

  def lookup_meanings(self, form: str) -> set[str]:
    if not self._all_forms_to_meanings:
      self._generate_all_forms()
    assert self._all_forms_to_meanings is not None
    return self._all_forms_to_meanings[form]

  def inflected_form(
      self, meaning: str, fv: features.FeatureVector
  ) -> tuple[str, str]:
    """Return the inflected form for a given meaning and FeatureVector.

    The construction is a bit complicated. We derive the stem by
    looking up the meaning in the lexicon and finding the
    corresponding morpheme. Then that is concatenated with the
    boundary symbol (so that only that stem will match), byte*, and
    the FeatureVector's acceptor. The rules, if any are then
    applied. This gives us the derived form, with boundaries and
    features. We want the *output* of this, so we project to the
    output. This is then right-composed with the paradigm's
    stems_to_forms. At that point two outputs are produced, one where
    the features are rewritten as strings of bytes so they are
    human-readable, and one where the boundaries and symbols are
    removed so we get the actual surface form.

    Args:
      meaning: a string corresponding to one of the meanings in the lexicon.
      fv: a FeatureVector.

    Returns:
      A pair consisting of the fully annotated inflected form, and the
      surface form, or None if no such meaning.
    """
    assert self._lexicon is not None
    try:
      stem = self._lexicon.meanings_to_morphemes[meaning].form
      stem += self._bound + _BYTE_STAR + fv.acceptor
      for rule in self._paradigm.rules:
        stem @= rule
      stem = stem.project("output")
      base = self._paradigm.stems_to_forms @ stem
      generator0 = (
          base @ self._paradigm.feature_label_rewriter).paths().ostrings()
      generator1 = (base @ self._paradigm.deleter).paths().ostrings()
      return list(generator0)[0], list(generator1)[0]
    except KeyError:
      return None

  def _generate_all_forms(self) -> None:
    """Generates all forms in the lexicon."""
    assert self._lexicon is not None
    self._all_forms_to_meanings = collections.defaultdict(set)
    for meaning in self._lexicon.meanings_to_morphemes.keys():
      forms = self.forms(meaning)
      for _, form in forms:
        self._all_forms_to_meanings[form].add(meaning)

  def forms(self, meaning: str) -> list[tuple[str, str]] | None:
    """Constructs all forms for a given meaning.

    Args:
      meaning: a string corresponding to one of the meanings in the lexicon.

    Returns:
      A list of pairs consisting of the fully annotated inflected form,
      and the surface form, or [] if no such meaning.
    """
    assert self._lexicon is not None
    try:
      stem = self._lexicon.meanings_to_morphemes[meaning].form
      generator = stem @ self._paradigm.stems_to_forms
      generator0 = generator @ self._paradigm.feature_label_rewriter
      generator1 = generator @ self._paradigm.deleter
      ostrings0 = list(generator0.paths().ostrings())
      ostrings1 = list(generator1.paths().ostrings())
      assert len(ostrings0) == len(ostrings1)
      forms = []
      # TODO: Check that the two string sets always come out in the same
      # order. If not then I will have to do something differently. So
      # far though the test always passes, so this seems to be true.
      for i in range(len(ostrings0)):
        forms.append((ostrings0[i], ostrings1[i]))
      return forms
    except KeyError:
      return []

  def dump_parameters(self, unused_path, unused_far_path):
    raise NotImplementedError


# Need a version to read and write templates...
def write_rules_to_far(
    rules: list[py.Fst] | list[tuple[py.Fst, py.Fst, py.Fst]] | None,
    far_path: str,
    far_type: py.FarType = _DEFAULT_FAR_TYPE,
    is_templates: bool = False
) -> None:
  """Writes rules to FST archive (FAR).

  Args:
    rules: list of rules.
    far_path: path to FAR.
    far_type: type of FAR.
    is_templates: if true, rules are tuples of tau, lambda, rho.
  """
  if not rules:
    return
  if not far_path:
    return
  if is_templates and isinstance(rules[0], tuple):
    arc_type = rules[0][0].arc_type()
  elif isinstance(rules[0], py.Fst):
    arc_type = rules[0].arc_type()
  flattened_rules = []
  if is_templates and isinstance(rules[0], tuple):
    for t, l, r in rules:
      flattened_rules.extend([t, l, r])
  rules = flattened_rules
  logging.info("Writing FAR to %s ...", far_path)
  with py.Far(far_path, "w", arc_type, far_type=far_type) as sink:
    i = 0
    for rule in rules:
      sink[f"{i:04d}"] = rule
      i += 1


def read_rules_from_far(
    far_path: str,
    far_type: py.FarType = _DEFAULT_FAR_TYPE,
    is_templates: bool = False
) -> list[py.FstLike] | list[tuple[py.FstLike, ...]]:
  """Reads rule_templates from an FST archive (FAR).

  Args:
    far_path: path to FAR.
    far_type: type of FAR.
    is_templates: if true, rules are tuples of tau, lambda, rho.

  Returns:
    List of FSTs or a list of tuples of FSTs.
  """
  rules = []
  if not far_path:
    return rules
  try:
    logging.info("Reading FAR from %s ...", far_path)
    with py.Far(far_path, "r") as source:
      assert source.far_type() == far_type
      for k, v in source:
        rules.append((k, v))
    rules = [v for (_, v) in sorted(rules)]
    if is_templates:
      assert len(rules) % 3 == 0
      templates = []
      for i in range(0, len(rules), 3):
        template = rules[i], rules[i + 1], rules[i + 2]
        templates.append(template)
      rules = templates
  except py.FstIOError:
    pass
  return rules


class BasicNounMorphologyBuilder(BasicMorphologyBuilder):
  """Basic noun morphology supporting gender, number and case."""

  def __init__(
      self,
      lexicon: morphemes.Lexicon,
      affix_specification: str | list[str],
      gen_feats: list[str] | None = None,
      num_feats: list[str] | None = None,
      cas_feats: list[str] | None = None,
      bound: str = "+",
      zero_feat_vals: list[tuple[str, str]] | None = None,
      rule_templates: list[tuple[
          py.FstLike, py.FstLike, py.FstLike]] | None = None
  ) -> None:
    super().__init__()
    self._feats = []
    # Lazily make this the first mentioned value for each feature:
    self._lemma_vector = []
    self._gen_feats = gen_feats
    self._num_feats = num_feats
    self._cas_feats = cas_feats
    self._zero_feat_vals = zero_feat_vals
    self._bound = bound
    self._rule_templates = rule_templates
    if self._gen_feats:
      self._feats.append(features.Feature("gen", *self._gen_feats))
      self._lemma_vector.append(_featval("gen", self._gen_feats[0]))
    if self._num_feats:
      self._feats.append(features.Feature("num", *self._num_feats))
      self._lemma_vector.append(_featval("num", self._num_feats[0]))
    if self._cas_feats:
      self._feats.append(features.Feature("cas", *self._cas_feats))
      self._lemma_vector.append(_featval("cas", self._cas_feats[0]))
    self._cat = features.Category(*self._feats)
    self.setup(lexicon, affix_specification, self._bound, self._zero_feat_vals,
               self._rule_templates)

  def _parameters_to_proto(self) -> MorphologyParameters:
    """Creates morphology parameters from data members."""
    params = MorphologyParameters()
    if self._gen_feats:
      params.gender_features.extend(self._gen_feats)
    if self._num_feats:
      params.number_features.extend(self._num_feats)
    if self._cas_feats:
      params.case_features.extend(self._cas_feats)
    if self._zero_feat_vals:
      feat_vals = ["{f}={v}" for (f, v) in self._zero_feat_vals]
      params.zero_feature_values.extend(feat_vals)
    if self._bound:
      params.boundary = self._bound
    return params

  def dump_parameters(self, path: str, far_path: str) -> None:
    """Saves morphology parameters and rules."""
    logging.info("Writing parameters to %s ...", path)
    with open(path, "wt") as f:
      params = self._parameters_to_proto()
      f.write(text_format.MessageToString(params))
    write_rules_to_far(self._rule_templates, far_path, is_templates=True)

  @staticmethod
  def load_parameters(path: str, far_path: str) -> collections.defaultdict:
    """Loads parameters and FAR.

    Args:
      path: path to parameters.
      far_path: path to FAR.

    Returns:
      collections.defaultdict containing the parameters and the list
      of FSTs, if any, from the FAR.
    """
    params = collections.defaultdict(list)
    logging.info("Loading morphology parameters from %s ...", path)
    with open(path, mode="rt") as f:
      params_proto = text_format.Parse(f.read(), MorphologyParameters())
      if params_proto.gender_features:
        params["gen_feats"] = list(params_proto.gender_features)
      if params_proto.number_features:
        params["num_feats"] = list(params_proto.number_features)
      if params_proto.case_features:
        params["cas_feats"] = list(params_proto.case_features)
      if params_proto.zero_feature_values:
        fv = [tuple(fv.split("=")) for fv in params_proto.zero_feature_values]
        params["zero_feat_vals"] = fv
      if params_proto.boundary:
        params["bound"] = params_proto.boundary
    params["rules"] = read_rules_from_far(far_path, is_templates=True)
    return params
