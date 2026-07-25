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
