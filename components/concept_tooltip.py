"""Small inline info-icon with a plain-language explanation, shown next to
any KPI or statistic - this is what makes the tool teach, not just report
numbers (see CLAUDE.md)."""

import streamlit as st


def concept_tooltip(label: str, explanation: str) -> None:
    with st.popover("ⓘ", use_container_width=False):
        st.markdown(f"**{label}**")
        st.write(explanation)
