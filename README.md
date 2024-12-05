# ProtoScribe: Modeling the Evolution of Writing

![CI status](https://github.com/google-research/protoscribe/actions/workflows/validate.yaml/badge.svg?branch=main)

<center>
<font size="+4">𓏞</font>
<p>
<font size="+2">𒁾𒊬</font><br>
</center>

This repository contains the supporting code for experimenting with machine
learning approaches to evolution of writing.

NOTE: This is work in progress.

## Installation and testing

Note, the instructions in this section apply to Debian Linux where this project
was developed. Depending on your operating system some of the installation steps
in `setup.sh` may need to be amended.

Ideally the installation should happen in a Python virtual environment. The
installation is taken care of by the `setup.sh` script. Simply run

```shell
./setup.sh
```

from the root directory of the project. If all the dependencies are installed
correctly, run the tests using `pytest`:

```shell
./test.sh
```

Note, use `--continue-on-collection-errors` flag to calls to `pytest` inside
`test.sh` to see *all* the failing tests even if some of them cannot be loaded
correctly.

## The codebase

The important directories are:

*   [corpus](protoscribe/corpus): Utilities for
    building and parsing the corpus.

*   [data](protoscribe/data): This directory
    contains most of the bits out of which we generate our simulated
    dataset. For example, the various concept inventories can be found in
    [`concepts`](protoscribe/data/concepts/),
    while their corresponding numeric embeddings in
    [`semantics`](protoscribe/data/semantics/).
    The other types of data include things like various sets of SVGs for our
    glyphs and so on.

*   [evolution](protoscribe/evolution): Pipeline
    utilities and supporting scripts for simulation of writing system evolution.

*   [glyphs](protoscribe/glyphs): Libraries for
    dealing with SVGs, but also with discrete glyph vocabularies, i.e.
    [`glyph_vocab.py`](protoscribe/glyphs/glyph_vocab.py).

*   [language](protoscribe/language): Directory
    housing linguistic modeling APIs:

    *   [embeddings](protoscribe/language/embeddings):
    Various interfaces for (semantic) embeddings. The most relevant one is the
    [`embedder.py`](protoscribe/language/embeddings/embedder.py).
    Our configuration defaults to BNC. In addition, in the past we played with
    representing concepts by glosses -- shorts snippets of Wikipedia text
    explaining what a thing is -- these can then be encoded using a pretrained
    language model.

    *   **morphology/phonology/syntax**: Definitions of the phonology, morphology
    and syntax of the generated language. The core functionality for determining
    morpheme shape and what it means for two words to sound similar resides in
    [`phonology`](protoscribe/language/phonology/)
    and includes libraries for computing phonetic embeddings.

*   [glyphs](protoscribe/glyphs): Libraries for
    dealing with SVGs, but also with discrete glyph vocabularies, i.e.
    [`glyph_vocab.py`](protoscribe/glyphs/glyph_vocab.py).

*   [semantics](protoscribe/semantics): Basic helper
    packages for representing knowledge about categories.

*   [scoring](protoscribe/scoring): Tools for
    evaluating and scoring the resulting models.

*   [sketches](protoscribe/sketches/utils):
    Libraries for manipulating and modeling glyphs as sketches. Includes core
    libraries for representing glyphs as sequences of (possibly quantized)
    strokes in our models.

*   [speech](protoscribe/speech): Acoustic
    front-end components.

*   [texts](protoscribe/texts):
    The libraries for constructing the actual "accounting documents".

*   [vision](protoscribe/vision): Components for
    building and representing image features corresponding to semantic concepts.

## Disclaimers

This is not an official Google product.
