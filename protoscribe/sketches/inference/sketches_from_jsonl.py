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

"""Processor for sketch generation results in JSONL format."""

import logging
import os
from typing import Any

from absl import flags
from matplotlib import pyplot as plt
import ml_collections
import numpy as np
from protoscribe.corpus.reader import corpus_reader as ds_lib
from protoscribe.glyphs import glyph_vocab as glyph_lib
from protoscribe.sketches.inference import json_utils
from protoscribe.sketches.utils import stroke_stats as stats_lib
from protoscribe.sketches.utils import stroke_tokenizer as tokenizer_lib
from protoscribe.sketches.utils import stroke_utils as stroke_lib

import glob
import os

SketchToken = tokenizer_lib.Token

RECOGNIZER_JSON = flags.DEFINE_bool(
    "recognizer_json", False,
    "If enabled, the JSON predictions are assumed to come from the recognizer. "
    "This is a different format where the actual predictions are the discrete "
    "glyph IDs and the sketch tokens are found as sub-features under `inputs`."
)

COMBINED_GLYPHS_AND_STROKES = flags.DEFINE_bool(
    "combined_glyphs_and_strokes", False,
    "If enabled, the JSON predictions are assumed to contain both the glyphs "
    "and the sketch tokens in each output sequence. The glyphs are prefix of "
    "the sequence."
)

_IGNORE_ERRORS = flags.DEFINE_bool(
    "ignore_errors", False,
    "If False, raise an exception if various delimeters are missing, "
    "else just log an error."
)

_SAVE_STROKES_IN_JSONL = flags.DEFINE_bool(
    "save_strokes_in_jsonl", True,
    "If enabled, adds the generated strokes (in Stroke-3 format) to the final "
    "predictions file in JSONL format."
)

_PRUNE_NUMBERS = flags.DEFINE_bool(
    "prune_numbers", True,
    "If enabled (default) omits the strokes corresponding to number glyphs "
    "from the sketch."
)

_SKIP_PLOTS = flags.DEFINE_bool(
    "skip_plots", False,
    "Do not generate the images for the sketches."
)

_PLOT_BEST_HYPOTHESIS_ONLY = flags.DEFINE_bool(
    "plot_best_hypothesis_only", False,
    "In case when multiple hypotheses are present, only plot the best one."
)

_PLOT_COLOR = flags.DEFINE_string(
    "plot_color", None,
    "Color for the plot. This is a matplotlib abbreviation, e.g., `r` for red. "
    "By default each stroke gets its own color."
)

_TRANSPARENT = flags.DEFINE_bool(
    "transparent", False,
    "Make the plot transparent."
)

_SHOW_TITLE = flags.DEFINE_bool(
    "show_title", True,
    "If disabled, the title is saved in a separate text file."
)

_DPI = flags.DEFINE_integer(
    "dpi", 150,
    "Resolution for output images (dots per ink)."
)

_SAVE_SVG = flags.DEFINE_bool(
    "save_svg", False,
    "Save SVGs in addition to PNGs. The SVGs will have no annotations."
)


def json_to_sketch(
    config: ml_collections.FrozenConfigDict,
    sketch_dict: dict[str, Any],
    stroke_stats: stats_lib.FinalStrokeStats,
    stroke_tokenizer: tokenizer_lib.StrokeTokenizer,
    glyph_vocab: glyph_lib.GlyphVocab,
    pronunciation_lexicon: dict[str, list[str]],
    output_dir: str
) -> dict[str, Any]:
  """Generates sketch from the decoding results."""
  scorer_dict = json_utils.get_scorer_dict(sketch_dict, pronunciation_lexicon)
  input_text, title = _title_from_inputs(scorer_dict)

  # Check if glyph predictions are available. These should be present in the
  # output of the recognizer (in recognizer's JSONLs) or when the sketch
  # generator runs in the combined mode generating both strokes and discrete
  # glyphs.
  if RECOGNIZER_JSON.value or COMBINED_GLYPHS_AND_STROKES.value:
    names, pronunciations, scores = _glyphs_from_json(
        sketch_dict, input_text, glyph_vocab, pronunciation_lexicon
    )
    scorer_dict["glyph.names"] = names
    scorer_dict["glyph.scores"] = scores
    scorer_dict["glyph.prons"] = pronunciations
    if len(scores) >= 2:
      scorer_dict["glyph.confidence"] = scores[-1] - scores[-2]

    # Fancy title for the glyphs of the form: NUM, GLYPH/PRON.
    best_names = names[-1]
    best_prons = pronunciations[-1]
    glyphs = []
    for i, name in enumerate(best_names):
      pron = "".join(best_prons[i]) if best_prons[i] else ""
      glyphs.append(f"{name}/{pron}" if pron else name)
    glyphs = " ".join(glyphs)
    title = f"{title}\nGlyphs: {glyphs}"

    # Fill in best hypotheses in human-readable form.
    scorer_dict["glyph.names.best"] = " ".join(best_names)
    scorer_dict["glyph.prons.best"] = json_utils.glyph_pron(scorer_dict, k=-1)

  # Process the actual strokes.
  nbest_strokes, nbest_polylines = _strokes_from_json(
      config, sketch_dict, input_text, stroke_stats, stroke_tokenizer
  )

  if _SAVE_STROKES_IN_JSONL.value:
    scorer_dict["strokes.nbest.deltas"] = [s.tolist() for s in nbest_strokes]

  # Set the sketch confidence. This may have been propagated by the recognizer
  # already or needs to be computed from the JSONL.
  if "sketch.confidence" in sketch_dict["inputs"]:
    scorer_dict["sketch.confidence"] = (
        sketch_dict["inputs"]["sketch.confidence"]
    )
  else:
    confidence = 0.
    if COMBINED_GLYPHS_AND_STROKES.value:
      confidence = json_utils.get_confidence(sketch_dict)
    scorer_dict["sketch.confidence"] = confidence

  # Save vector graphics.
  if _SAVE_SVG.value:
    output_file = os.path.join(output_dir, f"{input_text}.svg")
    output_file = output_file.replace(" ", "_")
    logging.info("%s: Saving %s ...", input_text, output_file)
    nbest_strokes = nbest_strokes[-1]  # Plot best hypothesis only.
    stroke_lib.stroke3_strokes_to_svg_file(nbest_strokes, output_file)

  # Save detailed plots in raster format.
  if not _SKIP_PLOTS.value:
    _plot_hypotheses(
        scorer_dict, input_text, title, nbest_polylines, output_dir
    )

  return scorer_dict


def _glyphs_from_json(
    sketch_dict: dict[str, Any],
    input_text: str,
    glyph_vocab: glyph_lib.GlyphVocab,
    pronunciation_lexicon: dict[str, list[str]],
) -> tuple[list[list[str]], list[list[list[str]]], list[float]]:
  """Returns top glyph sequence prediction along with scores and prons."""

  # Fetch the glyph tokens. Hypotheses are sorted in ascending order by log
  # probabilitity.
  hypotheses = sketch_dict["prediction"]
  if isinstance(hypotheses[0], float):
    hypotheses = [hypotheses]

  # Fetch the beam scores. These have to match the number of hypotheses.
  if "aux" not in sketch_dict:
    raise ValueError(f"{input_text}: Expecting `aux` dict!")
  if "scores" not in sketch_dict["aux"]:
    raise ValueError(f"{input_text}: Decoding scores not found")
  scores = sketch_dict["aux"]["scores"]
  if isinstance(scores, float):
    scores = [scores]
  if len(scores) != len(hypotheses):
    raise ValueError(
        f"{input_text}: Mismatching scores ({len(scores)}) and hypotheses "
        f"({len(hypotheses)})"
    )

  # Detokenize and retrieve pronunciations.
  names = []
  pronunciations = []
  for i, tokens in enumerate(hypotheses):
    # When glyphs and sketch tokens are combined remove everything after the
    # glyph prefix. When glyph prefix is extracted we prepend/append the control
    # tokens because these are omitted during the training on purpose.
    if COMBINED_GLYPHS_AND_STROKES.value:
      glyph_tokens = []
      is_sketch_token = False
      for token in tokens:
        if token >= ds_lib.STROKE_OFFSET_FOR_GLYPH_IDS:
          if is_sketch_token:
            raise ValueError(
                f"{input_text}: [hypothesis #{i}] "
                "Glyph token outside the prefix!"
            )
          glyph_tokens.append(token - ds_lib.STROKE_OFFSET_FOR_GLYPH_IDS)
          is_sketch_token = False
        else:
          is_sketch_token = True
      tokens = [glyph_lib.GLYPH_BOS] + glyph_tokens + [glyph_lib.GLYPH_EOS]

    # Sanity checks and detokenization.
    best_hypothesis = (i == (len(hypotheses) - 1))
    if tokens[0] != glyph_lib.GLYPH_BOS:
      error = f"{input_text}: [hypothesis #{i}] BOS glyph token missing!"
      # Only throw for the best hypothesis.
      if best_hypothesis:
        raise ValueError(error)
      else:
        logging.warning(error)
    else:
      tokens = tokens[1:]

    if glyph_lib.GLYPH_EOS not in tokens:
      error = f"{input_text}: [hypothesis #{i}] EOS glyph token missing!"
      if best_hypothesis:
        raise ValueError(error)
      else:
        logging.warning(error)
        eos_pos = len(tokens)
    else:
      eos_pos = tokens.index(glyph_lib.GLYPH_EOS)
    seq_names = glyph_vocab.detokenize(tokens[0:eos_pos])
    names.append(seq_names)

    seq_pron = []
    for name in seq_names:
      if name in pronunciation_lexicon:
        seq_pron.append(pronunciation_lexicon[name])
      else:
        seq_pron.append([])  # Number.
    pronunciations.append(seq_pron)

  return names, pronunciations, scores


def _title_from_inputs(scorer_dict: Any) -> tuple[str, str]:
  """Builds title from individual input features."""
  input_text = scorer_dict["text.inputs"]
  title = [f"Input concept: {input_text}"]
  pron = scorer_dict["text.sampa"]
  title.append(f"Phonology: {pron}")
  words = scorer_dict["text.words"]
  title.append(f"Words: {words}")

  return input_text, "\n".join(title)


def _strokes_from_json(
    config: ml_collections.FrozenConfigDict,
    sketch_dict: dict[str, Any],
    input_text: str,
    stroke_stats: stats_lib.FinalStrokeStats,
    stroke_tokenizer: tokenizer_lib.StrokeTokenizer,
) -> tuple[list[np.ndarray], list[list[stroke_lib.Array]]]:
  """Generate plottable strokes from JSON dictionary for all the hypotheses.

  Args:
    config: Configuration dictionary.
    sketch_dict: Input JSON dictionary from the inference script.
    input_text: Inputs corresponding to this prediction.
    stroke_stats: Stroke statistics for denormalization.
    stroke_tokenizer: Tokenizer/detokenizer for strokes.

  Returns:
    Tuple that contains n-best strokes in stroke-3 format and n-best polylines
    for plotting the glyph.
  """

  if RECOGNIZER_JSON.value:
    if "sketch_tokens" not in sketch_dict["inputs"]:
      raise ValueError("Expecting `sketch_tokens` under `inputs`")
    tokens = sketch_dict["inputs"]["sketch_tokens"]
  else:
    tokens = sketch_dict["prediction"]
  nbest_token_seqs = [tokens] if isinstance(tokens[0], int) else tokens

  nbest_polylines = []
  nbest_strokes = []
  for idx, tokens in enumerate(nbest_token_seqs):
    hyp_str = f"[hypothesis #{idx}]: " if len(nbest_token_seqs) > 1 else ""

    # When glyphs and sketch tokens are combined remove the glyph sequence
    # prefix.
    if COMBINED_GLYPHS_AND_STROKES.value:
      sketch_tokens = []
      token_is_glyph = True
      for token in tokens:
        if token >= ds_lib.STROKE_OFFSET_FOR_GLYPH_IDS:
          if not token_is_glyph:
            raise ValueError(
                f"{input_text}: {hyp_str}Found glyph token outside the prefix!"
            )
          token_is_glyph = True
          continue
        else:
          sketch_tokens.append(token)
          token_is_glyph = False
      tokens = sketch_tokens

    # There should be a delimiter between numeric and concept subsequences.
    if SketchToken.END_OF_NUMBERS not in tokens:
      msg = f"{input_text}: {hyp_str}Number-concept delimiter missing!"
      if _IGNORE_ERRORS.value:
        logging.error(msg)
        continue
      raise ValueError(msg)
    end_of_numbers_pos = tokens.index(SketchToken.END_OF_NUMBERS)

    # Check for BOS/EOS.
    if tokens[0] != SketchToken.START_OF_SKETCH:
      msg = f"{input_text}: {hyp_str}BOS sketch token missing!"
      if _IGNORE_ERRORS.value:
        logging.error(msg)
        continue
      raise ValueError(msg)

    tokens = tokens[1:]
    if SketchToken.END_OF_SKETCH in tokens:
      eos_pos = tokens.index(SketchToken.END_OF_SKETCH)
    else:
      logging.warning(
          "%s: %s EOS missing in sketch tokens: %s", input_text, hyp_str, tokens
      )
      eos_pos = len(tokens)
    if eos_pos <= end_of_numbers_pos:
      msg = f"{input_text}: {hyp_str}Invalid end-of-numbers position!"
      if _IGNORE_ERRORS.value:
        logging.error(msg)
        continue
      raise ValueError(msg)

    # Massage token sequence into shape by removing padding and pruning out
    # numbers if enabled.
    tokens = tokens[:eos_pos]
    if _PRUNE_NUMBERS.value:
      tokens = tokens[end_of_numbers_pos:]
    tokens = np.array(tokens, dtype=np.int32)

    # Detokenize into sketch-3 format, denormalize (if normalization was
    # originally applied) and convert the resulting points to plottable
    # polylines.
    strokes = stroke_tokenizer.decode(tokens)
    strokes = stats_lib.denormalize_strokes_array(config, stroke_stats, strokes)
    nbest_strokes.append(strokes)
    strokes_list = stroke_lib.stroke3_deltas_to_polylines(strokes)
    logging.info(
        "%s %s %d tokens, %d stroke deltas",
        input_text, hyp_str, strokes.shape[0], len(strokes_list)
    )
    nbest_polylines.append(strokes_list)

  return nbest_strokes, nbest_polylines


def _plot_hypotheses(
    scorer_dict: dict[str, Any],
    input_text: str,
    title: str,
    nbest_polylines: list[list[stroke_lib.Array]],
    output_dir: str
) -> None:
  """Plots the n-best parsed strokes in polylines format."""
  if _PLOT_BEST_HYPOTHESIS_ONLY.value:
    nbest_polylines = [nbest_polylines[-1]]
  for idx, polylines in enumerate(nbest_polylines):
    hyp_str = f"[hyp-{idx}]: " if len(nbest_polylines) > 1 else ""
    plt.clf()
    if _SHOW_TITLE.value:
      plt.title(f"{hyp_str}{title}", fontsize=8, loc="center")
    for s in polylines:
      plt.plot(s[:, 0], -s[:, 1], color=_PLOT_COLOR.value)
      plt.axis("off")  # Don't show axes.
    plt.tight_layout()
    doc_id = scorer_dict["doc.id"]
    filename = input_text.replace(" ", "_")
    filename = (
        f"{filename}_{doc_id}_{idx}.png" if len(nbest_polylines) > 1
        else f"{filename}_{doc_id}.png"
    )
    file_path = os.path.join(output_dir, filename)
    logging.info("%s: %s Saving %s", input_text, hyp_str, file_path)
    plot_kwargs = {
        "dpi": _DPI.value,
        "transparent": _TRANSPARENT.value,
    }
    with open(file_path, "wb") as f:
      plt.savefig(f, **plot_kwargs)
    if not _SHOW_TITLE.value:  # Save title separately.
      file_path = file_path.replace(".png", "_title.txt")
      with open(file_path, "w") as f:
        f.write(title)
