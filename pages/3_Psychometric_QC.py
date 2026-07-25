"""Psychometric QC page. Rendering only - all statistics come from core/."""

import pandas as pd
import streamlit as st

from components.concept_tooltip import concept_tooltip
from components.kpi_card import kpi_card, status_icon
from core import stats_engine as se
from core import viz_engine as ve
from utils.session import require_dataset

dataset = require_dataset()
q = dataset.questionnaire

st.title("Psychometric QC")
st.caption(f"Automatisk kvalitetskontroll av ditt {q.test_name}-dataset")

summary = se.quality_summary(dataset)
missing_series = summary.missing_by_item

# --- KPI row -------------------------------------------------
kpi_cols = st.columns(4)
with kpi_cols[0]:
    kpi_card(
        "✅",
        "#D1FAE5",
        f"{summary.completeness.complete_pct:.1f}%",
        "Kompletta svar",
        caption=f"{summary.completeness.complete:,} av {summary.completeness.n:,}".replace(",", " "),
    )
with kpi_cols[1]:
    if not missing_series.empty:
        interval = f"{missing_series.min():.1f}% – {missing_series.max():.1f}%"
    else:
        interval = "–"
    kpi_card("⏱️", "#FFEDD5", interval, "Bortfall per item", caption="Intervall")
with kpi_cols[2]:
    kpi_card("⚠️", "#FEF3C7", str(summary.pattern_flags.total), "Mönsterflaggor", caption="Möjliga mönster")
with kpi_cols[3]:
    kpi_card("🛑", "#FEE2E2", str(summary.n_flagged_checks), "Problemflaggor", caption="Kräver uppmärksamhet")

st.write("")

# --- Missing per item chart / Kontroller table -------------------------------------------------
col_missing, col_checks = st.columns(2)
with col_missing:
    st.markdown("**Bortfall per item (%)**")
    if not missing_series.empty:
        st.plotly_chart(ve.missing_by_item_chart(missing_series), width="stretch")
    else:
        st.info("Inga items att visa.")

with col_checks:
    header_cols = st.columns([5, 1])
    header_cols[0].markdown("**Kontroller**")
    with header_cols[1]:
        concept_tooltip(
            "Kvalitetskontroller",
            "Automatiska kontroller som flaggar potentiella datakvalitetsproblem: konstanta eller "
            "lågvarierande items, golv-/takeffekter, univariata outliers (|z| > 3.29) och dubbletter.",
        )
    rows = [
        {"Kontroll": c.name, "Status": status_icon(c.status), "Kommentar": c.comment}
        for c in summary.checks
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

st.write("")

if summary.status == "good":
    st.success(f"✅ {summary.message}")
elif summary.status == "warning":
    st.warning(f"⚠️ {summary.message}")
else:
    st.error(f"🛑 {summary.message}")

st.write("")
col_back, _, col_next = st.columns([1, 3, 1])
with col_back:
    if st.button("Tillbaka", width="stretch"):
        st.switch_page("pages/2_Dataset_Overview.py")
with col_next:
    if st.button("Fortsätt till Test Profile Explorer →", type="primary", width="stretch"):
        st.switch_page("pages/4_Test_Profile_Explorer.py")
