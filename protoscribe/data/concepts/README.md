*  `gardiner_concepts.tsv` and `concept_categories.tsv`: The first is a list of
all hieroglyphs from Alan Gardiner's list
(https://en.wikipedia.org/wiki/Gardiner%27s_sign_list), using Egyptian
hieroglyphs as a proxy for depictable objects that are likely to show up as
signs in early writing systems. In this table, the first column is the
hieroglyph, the second column is the proposed related concept (if any). Note
that this is *not* necessarily the same as what the sign was used for in
Egyptian. The third column indicates whether or not the first column makes sense
as a symbol in an original accounting or administrative document. This is
derived from intuition about what kinds of things early preliterate
administrators may have wanted to count.  The second is a hierarchical set of
categories, where the most important columns are the second "subcategory" and
the third, a list of instances of that subcategory.

* `administrative_categories.txt`: A list of core administrative concepts with a
  possible association to their more general category. The first column
  corresponds to category and the second to concept.  When there is a single
  column, the entry corresponds to both concept and category.

* `non_administrative_categories.txt`: A list of non-administrative categories.

*  `additional_non_administrative_concepts.txt`: is a hand-curated set of
    additional concepts that do not fit naturally into an administrative
    context, but are depictable and could therefore be used as the inspiration
    for further glyphs.

*  `concept_descriptions.json`: Association between glyphs, categories and short
   description of the concepts the glyphs represent.

*  `all_concepts_with(out)_pos.txt`: Administrative and non-administrative
   (flattened) concepts with/without part-of-speech tags.
