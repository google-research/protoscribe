`ipa_to_sampa.tsv` is a map from Unicode IPA to Sampa.

`wiktionary_syllable_stats.tsv` is statistics on syllable types computed over
IPA wordlists for 225 languages from Wiktionary:

Column 1 is the raw count of the coarse CVC profile in column 3.

Column 2 is the raw count of the fine sonority profile in column 4.

Column 3 gives the coarse CVC profile for the syllable type.

Column 4 gives a finer-grained sonority profile where `4` is a vowel, `3` is a
glide, `2` is a sonorant consonant, `1` is a continuant consonant, and `0` is
any other consonant. All features are based on Phoible.
