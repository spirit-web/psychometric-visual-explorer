"""Small inline info-icon with a plain-language explanation, shown next to
any KPI or statistic - this is what makes the tool teach, not just report
numbers (see CLAUDE.md).

When Läroläge (Learning Mode) is toggled on in the sidebar, also shows a
deeper explanation sourced from utils/learning_content.py, if a matching
concept exists - this lets every existing tooltip call site gain a more
thorough explanation without being touched individually.
"""

import streamlit as st

from utils.learning_content import find_deeper_explanation, get_concept_explanation


def concept_tooltip(label: str, explanation: str, learning_key: str | None = None) -> None:
    """`learning_key` names the exact utils.learning_content concept term to
    use for the Läroläge deep-dive, when the visible label doesn't textually
    match it well enough for the fuzzy fallback to find it. Omit it when the
    label already is the concept term (e.g. "Youden's J")."""
    with st.popover("ⓘ", width="content"):
        st.markdown(f"**{label}**")
        st.write(explanation)
        if st.session_state.get("pve_learning_mode"):
            deeper = get_concept_explanation(learning_key) if learning_key else find_deeper_explanation(label)
            if deeper:
                st.divider()
                st.caption("📖 Läroläge – fördjupning")
                st.write(deeper)
