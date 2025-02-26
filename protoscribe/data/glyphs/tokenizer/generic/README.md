# Pretrained sketch tokenizer for generic glyphs

This directory contains sketch token clusters computed with k-means
algorithm. The run is time- and memory- consuming for the given number of
centroids. The procedure relies on the dataset that has already been generated.

Following quantization codebooks are available:

* `vocab2048_normalized_sketchrnn.npy`: Codebook with 2048 centroids trained on
vanilla glyphs normalized using producedure described in Sketch-RNN paper.

* `vocab2048_multiglyph_normalized_minmax.npy`: Codebook with 2048 centroids
trained on multiple glyph collections normalized using `min-max`.

* `vocab2048_multiglyph_normalized_sketchrnn.npy`: Codebook with 2048 centroids
trained on multiple glyph collections normalized using `sketch-rnn` approach.
