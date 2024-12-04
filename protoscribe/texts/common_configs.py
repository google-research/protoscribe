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

"""Common configuration setup for text generation."""

import collections

from protoscribe.language.morphology import morphemes
from protoscribe.language.phonology import phoible_templates as pt
from protoscribe.texts import number_config_pb2

from google.protobuf import text_format
import pynini as py
from pynini.lib import byte
from pynini.lib import features
import glob
import os

NumberConfig = number_config_pb2.NumberConfig

# Boundary specification
BOUND = "+"

# Affix shapes
AFFIX_SHAPES = {
    "AFFIX_VC": ["VC"],
    "AFFIX_CVC": ["CVC", "VC"],
}

_PROTO_PREFIX = "protoscribe/language/phonology/config"
MORPHEME_CORE = f"{_PROTO_PREFIX}/mono.textproto"
MORPHEME_CORE_SESQUI = f"{_PROTO_PREFIX}/sesqui.textproto"
MORPHEME_CORE_DI = f"{_PROTO_PREFIX}/disyllabic.textproto"
MORPHEME_CORE_TEMPLATIC = f"{_PROTO_PREFIX}/templatic.textproto"
MORPHEME_CVCC_MONO = f"{_PROTO_PREFIX}/cvcc_mono.textproto"

MORPHEME_SHAPES = {
    "MORPHEME_CORE": MORPHEME_CORE,
    "MORPHEME_CORE_SESQUI": MORPHEME_CORE_SESQUI,
    "MORPHEME_CORE_DI": MORPHEME_CORE_DI,
    "MORPHEME_CORE_TEMPLATIC": MORPHEME_CORE_TEMPLATIC,
    "MORPHEME_CVCC_MONO": MORPHEME_CVCC_MONO,
}

# TODO: These are placeholders for now, since obviously if we vary the set
# of phonemes a lot then we would need to generalize these rules. Ideally all of
# should be configurable via protos somehow.

# Common rule templates and rules
SIGSTAR = py.closure(byte.BYTE)
EMPTY = py.accep("")

VOICING_TAU = (
    py.cross("p", "b")
    | py.cross("t", "d")
    | py.cross("k", "ɡ")
    | py.cross("f", "v")
    | py.cross("s", "z")
    # Since we are restricted for the nonce to English phonemes, we replace the
    # velar fricatives with interdentals.
    | py.cross("θ", "ð")
    #    | py.cross("x", "ɣ")
)
V = py.accep("a") | "a̟" | "e" | "i" | "o" | "o̞" | "ɔ" | "u" | "ə" | "ɛ"
VOICING_TEMPLATE = VOICING_TAU, V + " ", py.closure(BOUND, 1) + " " + V
VOICING = py.cdrewrite(*VOICING_TEMPLATE, SIGSTAR)

NASAL = py.accep("m") | "n" | "ŋ"
NASALIZATION_TAU = (
    py.cross("p", "m")
    | py.cross("t", "n")
    | py.cross("k", "ŋ")
    | py.cross("b", "m")
    | py.cross("d", "n")
    | py.cross("ɡ", "ŋ")
)
NASALIZATION_TEMPLATE = NASALIZATION_TAU, NASAL + " # ", ""
NASALIZATION = py.cdrewrite(*NASALIZATION_TEMPLATE, SIGSTAR)

RULES = {
    "VOICING": VOICING,
    "NASALIZATION": NASALIZATION,
}

RULE_TEMPLATES = {
    "VOICING_TEMPLATE": VOICING_TEMPLATE,
    "NASALIZATION_TEMPLATE": NASALIZATION_TEMPLATE,
}

# Simple baseline crib lexicon for numbers
NUMBER_CRIB_LEXICON = morphemes.Lexicon()
NUMBER_CRIB_LEXICON.add_morpheme(morphemes.Morpheme("1", "o n e"))
NUMBER_CRIB_LEXICON.add_morpheme(morphemes.Morpheme("2", "t w o"))
NUMBER_CRIB_LEXICON.add_morpheme(morphemes.Morpheme("3", "t h r e e"))
NUMBER_CRIB_LEXICON.add_morpheme(morphemes.Morpheme("4", "f o u r"))
NUMBER_CRIB_LEXICON.add_morpheme(morphemes.Morpheme("5", "f i v e"))
NUMBER_CRIB_LEXICON.add_morpheme(morphemes.Morpheme("6", "s i x"))
NUMBER_CRIB_LEXICON.add_morpheme(morphemes.Morpheme("7", "s e v e n"))
NUMBER_CRIB_LEXICON.add_morpheme(morphemes.Morpheme("8", "e i g h t"))
NUMBER_CRIB_LEXICON.add_morpheme(morphemes.Morpheme("9", "n i n e"))
NUMBER_CRIB_LEXICON.add_morpheme(morphemes.Morpheme("[E1]", "t e n"))
NUMBER_CRIB_LEXICON.add_morpheme(morphemes.Morpheme("[E2]", "h u n d r e d"))
NUMBER_CRIB_LEXICON.add_morpheme(morphemes.Morpheme("[E3]", "t h o u s a n d"))

POWER_LIST = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "[E1]", "[E2]",
              "[E3]")

# Number configuration


def parse_number_config_file(path) -> collections.defaultdict:
  """Parses number grammar configuration file.

  Args:
    path: path to number configuration file.

  Returns:
    default dict containing number feature settings.
  """
  values = collections.defaultdict(list)
  with open(path, mode="rt") as f:
    config = text_format.Parse(f.read(), NumberConfig())
  if config.core.number_features:
    values["NUM_FEATS"] = list(config.core.number_features)
  if config.core.case_features:
    values["CAS_FEATS"] = list(config.core.case_features)
  if config.core.gender_features:
    values["GEN_FEATS"] = list(config.core.gender_features)
  if config.core.zero_feature_values:
    values["ZERO_FEAT_VALS"] = tuple(
        [tuple(c.split("=")) for c in config.core.zero_feature_values]
    )

  values["ONE_DELETION"] = config.one_deletion
  if config.one_config_features:
    values["ONE_CONFIG"] = list(config.one_config_features)
  if config.two_config_features:
    values["TWO_CONFIG"] = list(config.two_config_features)
  if config.many_config_features:
    values["MANY_CONFIG"] = list(config.many_config_features)

  return values


class TemplaticRules:
  """Provides an interface to make it easier to write features-sensitive rules.

  Initializer requires a PhoibleTemplates specification, a path to a
  ProsodyTemplate text proto, and a Pynini Category specification.
  """

  def __init__(self, phoible_templates, proto_path, cat):
    self._template_proto = pt.load_proto(proto_path)
    self._cat = cat
    self._max_phonemes = self._template_proto.number_of_phonemes + 10
    self._phoneme_superset = phoible_templates.phoible.top_k_segments(
        self._max_phonemes
    )
    self._sigstar = cat.sigma_star
    self._v = py.union(
        *phoible_templates.phoible.matching_phonemes(
            ["+syllabic"], self._phoneme_superset
        )
    )
    self._c = py.union(
        *phoible_templates.phoible.matching_phonemes(
            ["-syllabic"], self._phoneme_superset
        )
    )
    self._space = py.accep(" ")
    self._slots = {
        "C": self._c,
        "C+": self._c + py.closure(" " + self._c),
        "C?": self._c.ques,
        "V": self._v,
        "V+": self._v + py.closure(" " + self._v),
        "V?": self._v.ques,
    }
    self._label_rewriter = py.cdrewrite(
        self._cat.feature_mapper, "", "", self._sigstar
    )
    delspace = py.closure(py.cross(" ", ""), 0, 5)
    multispace = delspace + " "
    ns = byte.NOT_SPACE
    self._normalize_whitespace = (
        py.cdrewrite(multispace, ns, ns, self._sigstar)
        @ py.cdrewrite(delspace, "[BOS]", ns, self._sigstar)
        @ py.cdrewrite(delspace, ns, "[EOS]", self._sigstar)
        # Clean up trailing space before boundary:
        @ py.cdrewrite(delspace, ns, BOUND, self._sigstar)
    ).optimize()

  @property
  def label_rewriter(self):
    return self._label_rewriter

  def compile_rule(self, template, feats):
    """Compile a template to apply if a word matches the specification in feats.

    The template should be a list of strings or pairs, where the string elements
    are one of "C", "C*", "C?", "V", "V*", "V?", and the pairs consist of an
    input and an output, which is one of the above, or a single symbol or
    tuple/list of phonemes.

    An example is as follows:

    ["C", ("V", "i"), "C+",...]

    Args:
      template: A template specification as described above.
      feats: A list of features such as ["num=pl", "cas=nom"]

    Returns:
      A compiled rule FST.
    """
    rule = None
    for slot in template:
      if not rule:
        rule = py.accep("")
      else:
        rule += self._space
      try:
        rule += self._slots[slot]
      except KeyError:
        assert type(slot) in [tuple, list]
        assert len(slot) == 2
        pair = []
        for pos in slot:
          try:
            pos = self._slots[pos]
          except KeyError:
            if type(pos) in [tuple, list]:
              pos = py.union(*pos)
            else:
              pos = py.accep(pos)
          pair.append(pos)
        rule += py.cross(*pair)
    rule = py.invert((py.invert(rule) @ self._normalize_whitespace))

    def feature_acceptor(feats):
      return (
          self._sigstar
          + features.FeatureVector(self._cat, *feats).acceptor
          + self._sigstar
      ).optimize()

    rule += feature_acceptor(feats)
    rule @= self._normalize_whitespace
    return rule.optimize()


NOUN_TEMPLATE_RULE_EXAMPLES = (
    (("C", ("V", "a"), "C+", ("V", "a"), "C?"), ["num=sg"]),
    (("C", ("V", ""), "C+", ("V", "u"), "C?"), ["num=du"]),
    (("C", ("V", "i"), "C+", ("V", ""), "C?"), ["num=pl"]),
)
