"""Sidebar branding block shown above the navigation menu."""

import streamlit as st

_LOGO_HTML = """
<div class="pve-logo">
    <span class="pve-logo-icon">\U0001F9E0</span>
    <div>
        <div class="pve-logo-text-main">PVE</div>
        <div class="pve-logo-text-sub">Psychometric<br/>Visual Explorer</div>
    </div>
</div>
"""


def render_logo() -> None:
    st.sidebar.markdown(_LOGO_HTML, unsafe_allow_html=True)


def render_mode_toggles() -> None:
    """Läroläge (Learning Mode) toggle and language toggle - rendered once
    in app.py, visible on every page via the sidebar (as high as Streamlit's
    navigation API allows: st.navigation reserves the very top of the
    sidebar for the page-link list itself, so these render directly below
    it regardless of call order - there is no supported way to place custom
    sidebar widgets above it). Läroläge makes every existing ⓘ tooltip show
    a deeper explanation (see components/concept_tooltip.py); by default
    tooltips stay short, on the assumption the user already knows most
    concepts."""
    st.session_state.setdefault("pve_learning_mode", False)
    st.session_state["pve_learning_mode"] = st.sidebar.toggle(
        "🎓 Läroläge",
        value=st.session_state["pve_learning_mode"],
        help="Visar mer utförliga förklaringar i info-rutorna (ⓘ) genom hela appen.",
    )

    st.session_state.setdefault("pve_language", "sv")
    is_english = st.sidebar.toggle(
        "🇬🇧 English",
        value=st.session_state["pve_language"] == "en",
        help="Växlar navigeringsmenyn och startsidan mellan svenska (av) och engelska (på).",
    )
    st.session_state["pve_language"] = "en" if is_english else "sv"
