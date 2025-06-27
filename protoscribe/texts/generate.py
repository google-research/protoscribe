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

"""Generates a new language, saving out the parameters.

One can then use this to generate texts in the language.
"""

import collections
import csv
import logging
import random

from absl import flags
from protoscribe.glyphs import numbers_to_glyphs
from protoscribe.language.morphology import morphemes
from protoscribe.language.morphology import morphology
from protoscribe.language.morphology import numbers
from protoscribe.language.phonology import phoible_segments
from protoscribe.language.phonology import phoible_templates as pt
from protoscribe.language.phonology import sampa
from protoscribe.language.syntax import phrase
from protoscribe.texts import common_configs

from pynini.lib import features
import glob
import os

_CONCEPTS = flags.DEFINE_list(
    "concepts", None,
    "Paths to lists of concepts."
)

_UNSEEN_CONCEPTS = flags.DEFINE_list(
    "unseen_concepts", None,
    "Paths to lists of unseen concepts."
)

_EXCLUDE_CONCEPTS_FILE = flags.DEFINE_string(
    "exclude_concepts_file", None,
    "Text file containing new-line separated lists of concepts one needs to "
    "exclude from the files provided by `--concepts` and `--unseen_concepts`."
)

_CONCEPT_SPELLINGS = flags.DEFINE_string(
    "concept_spellings", None,
    "If provided, this is a text file in TSV format containing mapping from "
    "concepts to their glyph spellings, overriding the default glyph "
    "generation for a concept."
)

_AFFIX_LEXICON = flags.DEFINE_string(
    "affix_lexicon", None,
    "Path to affix_lexicon."
)

_MAIN_LEXICON = flags.DEFINE_string(
    "main_lexicon", None,
    "Path to main lexicon."
)

_NUMBER_LEXICON = flags.DEFINE_string(
    "number_lexicon", None,
    "Path to number lexicon."
)

_MORPHOLOGY_PARAMS = flags.DEFINE_string(
    "morphology_params", None,
    "Path to the morphology parameters in protocol buffer text format. "
    "See `protoscribe.MorphologyParameters` protocol buffer."
)

_NUMBER_PHON_RULES = flags.DEFINE_string(
    "number_phon_rules", None,
    "Path to an FST archive (FAR) of number phonological rules."
)

_PHON_RULES = flags.DEFINE_string(
    "phon_rules", None,
    "Path to FST archive (FAR) of phonological rules."
)

_NUMBER_CONFIG_FILE = flags.DEFINE_string(
    "number_config_file",
    (
        "protoscribe/texts/"
        "configs/number_config_sg_du_pl.textproto"
    ),
    "Path to number configuration in textual protocol buffer format. "
    "Please see `protoscribe.NumberConfig` protocol buffer definition."
)

_OUTPUT_TEXTS = flags.DEFINE_string(
    "output_texts", None,
    "Output text file containing generated `texts`."
)

_AFFIX_SHAPE = flags.DEFINE_string(
    "affix_shape", "AFFIX_VC",
    "Shape of affix as defined in `common_configs`."
)

_MORPHEME_SHAPE = flags.DEFINE_string(
    "morpheme_shape",
    "MORPHEME_CORE_SESQUI",
    "Shape of morpheme as defined in `common_configs`.",
)

_WORD_PHONOLOGICAL_RULE_TEMPLATES = flags.DEFINE_list(
    "word_phonological_rule_templates",
    "VOICING_TEMPLATE",
    "List of phonological rule templates to be used in morphological paradigm."
)

_NUMBER_PHONOLOGICAL_RULES = flags.DEFINE_list(
    "number_phonological_rules",
    "NASALIZATION",
    "List of phonological rule templates to be used in numbers.",
)

_MAX_HOMOPHONY = flags.DEFINE_integer(
    "max_homophony", 10,
    "Attempt to limit the maximum number of times a form can be reused to "
    "this amount."
)

_MAX_NUMBER_OF_SINGLE_SEGMENT_FORMS = flags.DEFINE_integer(
    "max_number_of_single_segment_forms", 10,
    "Limit how many morphemes can be a single (vowel) segment.",
)

_NUM_TEXTS = flags.DEFINE_integer(
    "num_texts", 1000,
    "Number of texts to generate."
)

_MAX_COMMODITY = flags.DEFINE_integer(
    "max_commodity", 99,
    "Maximum number for a single commodity (e.g., `22 sheep`)."
)

_PROBABILITY_OF_SUPERCATEGORY_GLYPH = flags.DEFINE_float(
    "probability_of_supercategory_glyph", 0.25,
    "Probability of generating a supercategory glyph if one is available.",
)

FLAGS = flags.FLAGS


def load_concepts(concept_lists: list[str]) -> (
    tuple[dict[str, str], list[str]]
):
  """Loads categories from files."""
  exclude_concepts = set()
  if _EXCLUDE_CONCEPTS_FILE.value:
    logging.info("Concept exclusion in %s ...", _EXCLUDE_CONCEPTS_FILE.value)
    with open(_EXCLUDE_CONCEPTS_FILE.value) as f:
      for line in f:
        line = line.strip()
        if line:
          exclude_concepts.add(line)
    logging.info("Will exclude %d concepts.", len(exclude_concepts))

  logging.info("Loading concepts from %s ...", concept_lists)
  supercategories = collections.defaultdict(str)
  concepts = set()
  for path in concept_lists:
    with open(path) as stream:
      for line in stream:
        toks = line.strip().split()
        if not toks:
          continue
        if len(toks) > 1:
          if toks[1] not in exclude_concepts:
            concepts.add(toks[1])
            if toks[0] not in exclude_concepts:
              supercategories[toks[1]] = toks[0]
        else:
          if toks[0] not in exclude_concepts:
            concepts.add(toks[0])
  logging.info("Loaded %d concepts and %d super-categories.", len(concepts),
               len(supercategories))
  return supercategories, list(concepts)


def _load_concept_spellings(path: str) -> dict[str, list[str]]:
  """Loads a mapping between concepts and sequences of glyphs."""
  spellings = {}
  if not path:
    return spellings
  with open(path) as stream:
    reader = csv.reader(stream, delimiter="\t", quotechar='"')
    for row in reader:
      if len(row) != 2:
        raise ValueError(f"Expected two columns in {path}")
      concept = row[0].strip()
      glyphs = row[1].strip()
      spellings[concept] = glyphs.split()
  return spellings


def load_phonetic_forms(
    main_lexicon_file: str | None = None,
    number_lexicon_file: str | None = None,
    seen_concepts: list[str] | None = None,
) -> tuple[dict[str, list[str]], list[str]]:
  # TODO: This does NOT take any phonological variation into account. Also,
  # it is not entirely correct in that the argument is not really
  # `seen_concepts` but administrative concepts, some of which may not have been
  # seen.
  """Returns phonetic forms from the training data, and prons for all words.

  Args:
    main_lexicon_file: Main lexicon.
    number_lexicon_file: Number lexicon.
    seen_concepts: A list of seen concepts. If not provided, no filtering on
      of phonetic forms based on the seen concepts is performed.

  Returns:
    A tuple of a mapping from (all) concepts/words to their readings, and a list
    of seen phonetic forms (which will be empty if no `seen_concepts` is
    supplied).
  """
  seen_phonetic_forms = set()
  pronunciation_lexicon = {}

  if not main_lexicon_file:
    main_lexicon_file = _NUMBER_LEXICON.value
    if not main_lexicon_file:
      raise ValueError("Main lexicon file not specified!")

  logging.info("Loading main lexicon from %s ...", main_lexicon_file)
  pos_collisions = set()
  with open(main_lexicon_file) as s:
    for line in s:
      conc, phon = line.strip("\n").split("\t")
      # TODO: This will fail if we have the same term with two different
      # parts of speech.
      word = conc.split("_")[0]
      if word in pronunciation_lexicon:
        pos_collisions.add(word)
      pronunciation_lexicon[word] = phon.split()
      if seen_concepts and conc in seen_concepts:
        seen_phonetic_forms.add(phon)

  # Removing parts-of-speech results in key collisions. Print these words,
  # if any.
  if pos_collisions:
    logging.warning(
        "Removing POS from concepts results in pronunciation "
        "collisions for: %s", pos_collisions
    )

  if not number_lexicon_file:
    number_lexicon_file = _NUMBER_LEXICON.value
    if not number_lexicon_file:
      raise ValueError("Number lexicon file not specified!")

  logging.info("Loading number lexicon from %s ...", number_lexicon_file)
  with open(number_lexicon_file) as s:
    for line in s:
      _, phon = line.strip("\n").split("\t")
      if seen_concepts:
        seen_phonetic_forms.add(phon)

  if seen_concepts:
    logging.info("Loaded %d seen phonetic forms.", len(seen_phonetic_forms))
  return pronunciation_lexicon, list(seen_phonetic_forms)


class TextGenerator:
  """Class to handle scribal text generation."""

  def __init__(self):
    all_concept_lists = _CONCEPTS.value
    if not all_concept_lists:
      raise ValueError("No mandatory concept lists have been provided!")
    if _UNSEEN_CONCEPTS.value:
      all_concept_lists.extend(_UNSEEN_CONCEPTS.value)
    self._supercategories, self._concepts = load_concepts(all_concept_lists)
    self._concepts = list(self._concepts)
    self._concept_spellings = _load_concept_spellings(_CONCEPT_SPELLINGS.value)
    assert _MORPHEME_SHAPE.value in common_configs.MORPHEME_SHAPES
    self._shape_proto = common_configs.MORPHEME_SHAPES[_MORPHEME_SHAPE.value]
    self._number_features = common_configs.parse_number_config_file(
        _NUMBER_CONFIG_FILE.value
    )
    self._phoible_templates = pt.PhoibleTemplates(
        phoible_segments.PHOIBLE,
        phoible_segments.PHOIBLE_FEATURES,
        pt.SYLLABLE_TEMPLATES,
    )
    feats = []
    if self._number_features["GEN_FEATS"]:
      feats.append(features.Feature("gen", *self._number_features["GEN_FEATS"]))
    if self._number_features["NUM_FEATS"]:
      feats.append(features.Feature("num", *self._number_features["NUM_FEATS"]))
    if self._number_features["CAS_FEATS"]:
      feats.append(features.Feature("cas", *self._number_features["CAS_FEATS"]))
    self._noun_cat = features.Category(*feats)
    assert _AFFIX_SHAPE.value in common_configs.AFFIX_SHAPES
    self._affix_shape = common_configs.AFFIX_SHAPES[_AFFIX_SHAPE.value]
    self._nouns_rule_templates = []
    if _MORPHEME_SHAPE.value == "MORPHEME_CORE_TEMPLATIC":
      self._templatic_rules = common_configs.TemplaticRules(
          self._phoible_templates, self._shape_proto, self._noun_cat
      )
      for template, feat in common_configs.NOUN_TEMPLATE_RULE_EXAMPLES:
        self._nouns_rule_templates.append((
            self._templatic_rules.compile_rule(template, feat),
            common_configs.EMPTY,
            common_configs.EMPTY,
        ))
    for r in _WORD_PHONOLOGICAL_RULE_TEMPLATES.value:
      assert r in common_configs.RULE_TEMPLATES
      self._nouns_rule_templates.append(common_configs.RULE_TEMPLATES[r])
    self._number_rules = []
    for r in _NUMBER_PHONOLOGICAL_RULES.value:
      assert r in common_configs.RULES
      self._number_rules.append(common_configs.RULES[r])
    self._initialized_for_generation = False

  def generate_lexical_resources(self):
    """Generation and saving of lexical resources for language."""
    crib_lexicon = morphemes.Lexicon()
    # Initial hack: base the lengths on the English word lengths
    for concept in self._concepts:
      pseudo_pron = " ".join(list(concept))
      crib_lexicon.add_morpheme(morphemes.Morpheme(concept, pseudo_pron))
    main_lexicon = morphemes.make_lexicon(
        self._shape_proto,
        self._concepts,
        crib_lexicon=crib_lexicon,
        max_homophony=_MAX_HOMOPHONY.value,
        max_number_of_single_segment_forms=(
            _MAX_NUMBER_OF_SINGLE_SEGMENT_FORMS.value
        ),
        phoible=self._phoible_templates,
    )
    nouns = morphology.BasicNounMorphologyBuilder(
        main_lexicon,
        self._affix_shape,
        bound=common_configs.BOUND,
        num_feats=self._number_features["NUM_FEATS"],
        gen_feats=self._number_features["GEN_FEATS"],
        cas_feats=self._number_features["CAS_FEATS"],
        zero_feat_vals=self._number_features["ZERO_FEAT_VALS"],
        rule_templates=self._nouns_rule_templates,
    )
    main_lexicon.dump_lexicon(_MAIN_LEXICON.value)
    nouns.dump_parameters(_MORPHOLOGY_PARAMS.value, _PHON_RULES.value)
    nouns.dump_affixes(_AFFIX_LEXICON.value)
    morphology.write_rules_to_far(self._number_rules, _NUMBER_PHON_RULES.value)
    number_lexicon = morphemes.make_lexicon(
        self._shape_proto,
        common_configs.POWER_LIST,
        crib_lexicon=common_configs.NUMBER_CRIB_LEXICON,
        max_homophony=1,  # Disallow homophony for numbers
        # Be draconian about the number of single-segment numbers there can be.
        max_number_of_single_segment_forms=1,
        phoible=self._phoible_templates,
    )
    number_lexicon.dump_lexicon(_NUMBER_LEXICON.value)

  def initialize_for_generation(self, force=False):
    """Initialize resources for generation."""
    if self._initialized_for_generation and not force:
      return
    self._lexicon = morphemes.Lexicon()
    lexicon = self._lexicon
    lexicon.load_lexicon(_MAIN_LEXICON.value)
    params = morphology.BasicNounMorphologyBuilder.load_parameters(
        _MORPHOLOGY_PARAMS.value, _PHON_RULES.value
    )
    nouns = morphology.BasicNounMorphologyBuilder(
        lexicon,
        _AFFIX_LEXICON.value,
        bound=params["bound"],
        num_feats=params["num_feats"],
        cas_feats=params["cas_feats"],
        gen_feats=params["gen_feats"],
        zero_feat_vals=params["zero_feat_vals"],
        rule_templates=params["rules"],
    )
    self._number_lexicon = morphemes.Lexicon()
    number_lexicon = self._number_lexicon
    number_lexicon.load_lexicon(_NUMBER_LEXICON.value)
    number_phon_rules = morphology.read_rules_from_far(_NUMBER_PHON_RULES.value)
    number_grammar = numbers.Numbers(
        number_lexicon,
        number_phon_rules,
        one_deletion=self._number_features["ONE_DELETION"],
    )
    self._np = phrase.CountedNoun(
        nouns,
        number_grammar,
        one=self._number_features["ONE_CONFIG"],
        two=self._number_features["TWO_CONFIG"],
        many=self._number_features["MANY_CONFIG"],
    )
    self._ipa_to_sampa = sampa.IpaToSampaConverter(
        phoible_path=phoible_segments.PHOIBLE,
        phoible_features_path=phoible_segments.PHOIBLE_FEATURES,
    )
    self._initialized_for_generation = True

  def generate_initial_frequency_ordered_list(self):
    """Generates a frequency ordered list of basic forms for phonetic embedding.

    Based originally on the basic lexicon and number lexicon.

    TODO: This does not take into account possible morphological
    combinations and phonological rules.

    Returns:
      A list of forms, frequency ordered.
    """
    if not self._initialized_for_generation:
      self.initialize_for_generation()
    counts = collections.defaultdict(int)
    for form in self._lexicon.forms_to_morphemes:
      counts[form] += len(self._lexicon.forms_to_morphemes[form])
    for form in self._number_lexicon.forms_to_morphemes:
      counts[form] += len(self._lexicon.forms_to_morphemes[form])
    counts_list = []
    for form in counts:
      counts_list.append((counts[form], form))
    counts_list.sort(reverse=True)
    return [c[1] for c in counts_list]

  @staticmethod
  def generate_initial_frequency_ordered_list_lite(
      main_lexicon=None,
      number_lexicon_path=None,
  ):
    """Generates a frequency ordered list of basic forms for phonetic embedding.

    Lite version of the above that does not require initialization but simply
    loads a lexicon and number lexicon.

    Args:
      main_lexicon: path to main lexicon
      number_lexicon_path: path to number lexicon

    Returns:
      A list of forms, frequency ordered.
    """
    lexicon = morphemes.Lexicon()
    if not main_lexicon:
      main_lexicon = _MAIN_LEXICON.value
    lexicon.load_lexicon(main_lexicon)
    if not number_lexicon_path:
      number_lexicon_path = _NUMBER_LEXICON.value
    number_lexicon = morphemes.Lexicon()
    number_lexicon.load_lexicon(number_lexicon_path)

    counts = collections.defaultdict(int)
    for form in lexicon.forms_to_morphemes:
      counts[form] += len(lexicon.forms_to_morphemes[form])
    for form in number_lexicon.forms_to_morphemes:
      counts[form] += len(lexicon.forms_to_morphemes[form])
    counts_list = []
    for form in counts:
      counts_list.append((counts[form], form))
    counts_list.sort(reverse=True)
    return [c[1] for c in counts_list]

  def generate_transcriptions(self, text):
    """Generate verbalized_text and SAMPA.

    Args:
      text: input text
    Returns:
      Tuple of text and SAMPA.
    Raises:
      Exception: if the class instance is not initialized for generation.
    """
    if not self._initialized_for_generation:
      raise ValueError("Generator is not initialized for generation")
    verbalized_text = self._np.verbalize(text, " ")
    sampa_text = []
    for word in phrase.phrase_to_words(verbalized_text):
      sampa_word = self._ipa_to_sampa.convert(word)
      sampa_text.append(sampa_word)
    sampa_text = " # ".join(sampa_text)
    return verbalized_text, sampa_text

  def generate_texts(self):
    """Generates random texts."""
    self.initialize_for_generation()

    # Double-check that the number of requested texts is sane. Because we are
    # sampling the unique texts we want to make sure that the cardinality of
    # Cartesian product between number of commodities and concepts is greater
    # or equal than requested number of texts.
    num_concepts = len(self._concepts)
    num_texts = _NUM_TEXTS.value
    num_possible_texts = _MAX_COMMODITY.value * num_concepts
    if num_texts > num_possible_texts:
      # Stop raising an Error for this.
      msg = (
          f"Too many ({num_texts}) unique texts requested. The maximum "
          "number of unique texts in this configuration is "
          f"{num_possible_texts}. Reducing number of texts to that."
      )
      logging.warning(msg)
      num_texts = num_possible_texts

    all_texts = set()
    with open(_OUTPUT_TEXTS.value, "wt") as stream:
      while len(all_texts) < num_texts:
        number = random.randint(1, _MAX_COMMODITY.value)
        concept = random.choice(self._concepts)
        concept_text = f"{number} {concept}"
        verbalized_text, sampa_text = self.generate_transcriptions(concept_text)
        number_text = " ".join(list(numbers_to_glyphs.pseudo_roman(number)))
        concept_glyphs = self.generate_glyphs(concept)
        glyph_text = f"{number_text} {concept_glyphs}"
        text_contents = (
            f"{concept_text}\t{verbalized_text}\t{sampa_text}\t{glyph_text}"
        )
        if text_contents not in all_texts:
          stream.write(f"{text_contents}\n")
          all_texts.add(text_contents)

  def generate_glyphs(
      self, concept: str, strip_glyph_suffix: bool = False
  ) -> list[str]:
    """Generates a glyph for a concept.

    May generate an initial (prefix) glyph for a supercategory. First checks to
    see if the concept has a spelling in self._concept_spellings, assuming that
    has been filled out.

    Note, the returned glyph names don't have "_GLYPH" suffixes.

    Args:
      concept: a concept.
      strip_glyph_suffix: strip "_GLYPH" suffix from the name.

    Returns:
      A string for the glyph(s).
    """
    if concept in self._concept_spellings:
      spelling = self._concept_spellings[concept]
      if strip_glyph_suffix:
        spelling = [g.split("_")[0] for g in spelling]
      return spelling

    supercategory = (self._supercategories[concept]
                     if concept in self._supercategories else None)
    concept = concept.split("_")[0]  # Remove POS tag.
    glyph_seq = [concept] if strip_glyph_suffix else [f"{concept}_GLYPH"]
    if (
        supercategory
        and random.random()
        < _PROBABILITY_OF_SUPERCATEGORY_GLYPH.value
    ):
      supercategory = supercategory.split("_")[0]
      glyph_seq = [supercategory, concept]
      if not strip_glyph_suffix:
        glyph_seq = [f"{g}_GLYPH" for g in glyph_seq]

    return glyph_seq
