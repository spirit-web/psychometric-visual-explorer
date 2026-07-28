from utils import learning_content as lc


def test_get_concept_explanation_exact_match_case_insensitive():
    explanation = lc.get_concept_explanation("cronbach's alpha vs mcdonald's omega".upper())
    assert explanation is None  # not a real concept term, sanity check for the negative case
    explanation = lc.get_concept_explanation("mcdonald's omega".upper())
    assert explanation is not None
    assert "faktormodell" in explanation.lower()


def test_get_concept_explanation_strips_numbering_prefix():
    # The Validitet module's concept terms are numbered ("1. Innehåll (Content)");
    # lookups should work with or without that prefix.
    with_prefix = lc.get_concept_explanation("1. Innehåll (Content)")
    without_prefix = lc.get_concept_explanation("Innehåll (Content)")
    assert with_prefix is not None
    assert with_prefix == without_prefix


def test_get_concept_explanation_unknown_term_returns_none():
    assert lc.get_concept_explanation("Not A Real Concept") is None


def test_find_deeper_explanation_does_not_false_positive_on_short_formula_symbols():
    # Regression test: "T", "E", "M", "k", "X", "SD" are single/short formula
    # symbols in the Formelsamling module. Before the length-guard fix, any
    # label containing that letter (e.g. "Alpha om item tas bort" contains a
    # "t") would incorrectly match the "T" concept and show an unrelated
    # deep-dive explanation.
    assert lc.find_deeper_explanation("Alpha om item tas bort") is None


def test_find_deeper_explanation_matches_exact_concept_term():
    explanation = lc.find_deeper_explanation("ROC-kurva & AUC")
    assert explanation is not None
    assert "auc" in explanation.lower() or "särskilj" in explanation.lower()


def test_find_deeper_explanation_prefers_longest_match():
    # "Item-total korrelation" is a real concept; a shorter, coincidentally
    #-contained term should not win over it.
    explanation = lc.find_deeper_explanation("Item-total korrelation")
    assert explanation == lc.get_concept_explanation("Item-total korrelation")


def test_all_concept_terms_are_unique_enough_for_exact_lookup():
    # get_concept_explanation relies on exact (cleaned) term matching - fail
    # loudly if two concepts ever collide on the same cleaned term.
    cleaned_terms = [lc._clean_term(c.term) for m in lc.LEARNING_MODULES for c in m.concepts]
    assert len(cleaned_terms) == len(set(cleaned_terms))
