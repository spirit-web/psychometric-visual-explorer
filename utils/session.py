"""Session-state helpers shared by every analysis page."""

import streamlit as st

from core.data_model import Dataset


def get_dataset() -> Dataset | None:
    return st.session_state.get("pve_dataset")


def require_dataset() -> Dataset:
    """Return the active Dataset, or render a friendly redirect and halt the
    page if none has been imported yet."""
    dataset = get_dataset()
    if dataset is None:
        st.info("Inget dataset laddat än. Gå till Import Wizard för att komma igång.")
        if st.button("Till Import Wizard →", type="primary"):
            st.switch_page("pages/1_Import_Wizard.py")
        st.stop()
    return dataset


# --- Client aliases (session-only display names, e.g. "Johan" for Person 7) ---
#
# Purely a presentation convenience for live demos/presentations - never
# written to the dataset or exported, and never used in any calculation.
# Stored as {person_id: alias} in session state, independent of which
# dataset is active (a fresh dataset just means the ids won't match any
# alias, which is harmless).


def get_client_alias(person_id) -> str | None:
    aliases = st.session_state.get("pve_client_aliases", {})
    return aliases.get(person_id)


def set_client_alias(person_id, name: str) -> None:
    aliases = st.session_state.setdefault("pve_client_aliases", {})
    name = name.strip()
    if name:
        aliases[person_id] = name
    else:
        aliases.pop(person_id, None)


def client_label(person_id) -> str:
    """Display label for a person-selector option: the alias if one has
    been set (e.g. "Johan (Person 7)"), otherwise just "Person 7"."""
    alias = get_client_alias(person_id)
    return f"{alias} (Person {person_id})" if alias else f"Person {person_id}"
