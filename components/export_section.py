"""Reusable 'Exportera' section rendered at the bottom of analysis pages.
Rendering and download-button wiring only - all file generation lives in
core/export_engine.py, so no page duplicates export logic."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import export_engine as ee


def render_export_section(
    dataset,
    page_key: str,
    table: pd.DataFrame | None = None,
    table_label: str = "Tabell",
    figures: dict[str, go.Figure] | None = None,
) -> None:
    """A small expander offering the current page's table as CSV and/or its
    figures as PNG (single figure) or a ZIP (multiple figures)."""
    if (table is None or table.empty) and not figures:
        return

    with st.expander("📤 Exportera denna sida"):
        cols = st.columns(2)
        slot = 0
        if table is not None and not table.empty:
            with cols[slot % 2]:
                st.download_button(
                    f"⬇️ {table_label} (CSV)",
                    data=ee.dataframe_to_csv_bytes(table),
                    file_name=f"{page_key}_{dataset.questionnaire.plugin_id}.csv",
                    mime="text/csv",
                    width="stretch",
                    key=f"export_csv_{page_key}",
                )
            slot += 1
        if figures:
            with cols[slot % 2]:
                if len(figures) == 1:
                    name, fig = next(iter(figures.items()))
                    st.download_button(
                        "⬇️ Figur (PNG)",
                        data=ee.figure_to_png_bytes(fig),
                        file_name=f"{page_key}_{name}.png",
                        mime="image/png",
                        width="stretch",
                        key=f"export_png_{page_key}",
                    )
                else:
                    st.download_button(
                        f"⬇️ Figurer ({len(figures)} st, ZIP)",
                        data=ee.figures_to_zip_bytes(figures),
                        file_name=f"{page_key}_figurer.zip",
                        mime="application/zip",
                        width="stretch",
                        key=f"export_zip_{page_key}",
                    )
