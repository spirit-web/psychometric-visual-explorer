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
