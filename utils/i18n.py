"""Minimal i18n for the skimmable UI structure - nav labels and page
titles for now. Scoped MVP per design discussion: translating every page's
internal KPI labels, tab names, and explanatory text would be a much larger
effort, so for now only the structure a user skims (not reads in depth) is
bilingual. Deep explanatory text (tooltips, warnings, table contents) stays
Swedish-first everywhere - extend TRANSLATIONS as more of the UI is covered.

No Streamlit imports except reading the language choice from session_state.
"""

import streamlit as st

TRANSLATIONS: dict[str, dict[str, str]] = {
    "Home": {"sv": "Home", "en": "Home"},
    "Importera Tester": {"sv": "Importera Tester", "en": "Import Tests"},
    "Dataset Overview": {"sv": "Dataset Overview", "en": "Dataset Overview"},
    "Psychometric QC": {"sv": "Psychometric QC", "en": "Psychometric QC"},
    "Reliability Explorer": {"sv": "Reliability Explorer", "en": "Reliability Explorer"},
    "Factor Explorer": {"sv": "Factor Explorer", "en": "Factor Explorer"},
    "Validity Dashboard": {"sv": "Validity Dashboard", "en": "Validity Dashboard"},
    "Norm Explorer": {"sv": "Norm Explorer", "en": "Norm Explorer"},
    "Measurement Error": {"sv": "Measurement Error", "en": "Measurement Error"},
    "Decision Support": {"sv": "Decision Support", "en": "Decision Support"},
    "Fairness Explorer": {"sv": "Fairness Explorer", "en": "Fairness Explorer"},
    "Machine Learning": {"sv": "Machine Learning", "en": "Machine Learning"},
    "Test Builder": {"sv": "Test Builder", "en": "Test Builder"},
    "Export": {"sv": "Export", "en": "Export"},
}


def get_language() -> str:
    return st.session_state.get("pve_language", "sv")


def t(key: str) -> str:
    """Translate a nav/title string for the current language. Falls back
    to the key itself if untranslated - never crashes on a missing entry."""
    entry = TRANSLATIONS.get(key)
    if entry is None:
        return key
    return entry.get(get_language(), key)
