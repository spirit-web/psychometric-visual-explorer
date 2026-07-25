"""Dataset Overview page. Rendering only - all statistics come from core/."""

import pandas as pd
import streamlit as st

from components.concept_tooltip import concept_tooltip
from components.export_section import render_export_section
from components.kpi_card import kpi_card, status_icon
from core import stats_engine as se
from core import viz_engine as ve
from utils.session import require_dataset

dataset = require_dataset()
q = dataset.questionnaire

st.title("Dataset Overview")
st.caption(f"Översikt av ditt {q.test_name}-dataset ({q.full_name})")

# --- KPI row -------------------------------------------------
kpi_cols = st.columns(4)
with kpi_cols[0]:
    kpi_card("👥", "#DBEAFE", f"{dataset.n:,}".replace(",", " "), "Deltagare")
with kpi_cols[1]:
    kpi_card("📋", "#D1FAE5", str(dataset.n_items), "Items (frågor)")
with kpi_cols[2]:
    kpi_card("🧩", "#EDE9FE", str(dataset.n_subscales), "Delskalor")
with kpi_cols[3]:
    kpi_card("📉", "#FFEDD5", f"{dataset.missing_pct:.1f}%", "Total bortfall")

st.write("")

# --- Response distribution / missing per item -------------------------------------------------
col_dist, col_missing = st.columns(2)
with col_dist:
    header_cols = st.columns([5, 1])
    header_cols[0].markdown("**Svarsfördelning (alla items)**")
    with header_cols[1]:
        concept_tooltip(
            "Svarsfördelning",
            "Visar hur stor andel av alla besvarade items (över samtliga deltagare) som föll i "
            "respektive svarskategori. Ojämna fördelningar kan tyda på golv-/takeffekter.",
        )
    dist_df = se.response_distribution(dataset)
    dist_fig = ve.response_distribution_chart(dist_df)
    st.plotly_chart(dist_fig, width="stretch")

with col_missing:
    header_cols = st.columns([5, 1])
    header_cols[0].markdown("**Bortfall per item (%)**")
    with header_cols[1]:
        concept_tooltip(
            "Bortfall",
            "Andel deltagare som inte besvarat respektive item. Höga värden (>5%) kan indikera "
            "otydliga frågor eller ett känsligt ämne.",
        )
    missing_series = se.missing_by_item(dataset)
    missing_fig = ve.missing_by_item_chart(missing_series)
    st.plotly_chart(missing_fig, width="stretch")

st.write("")

# --- Demographics / descriptive stats -------------------------------------------------
col_demo, col_desc = st.columns(2)
with col_demo:
    st.markdown("**Demografi – Kön**")
    gender = se.demographic_breakdown(dataset, "gender")
    if gender is not None and not gender.empty:
        st.plotly_chart(ve.demographics_donut(gender), width="stretch")
    else:
        st.info("Ingen könsvariabel hittades i datasetet.")

with col_desc:
    header_cols = st.columns([5, 1])
    header_cols[0].markdown("**Beskrivande statistik (totalpoäng)**")
    with header_cols[1]:
        concept_tooltip(
            "Beskrivande statistik",
            "Sammanfattar fördelningen av totalpoäng per delskala: medelvärde, standardavvikelse "
            "(SD), min-max samt skevhet (snedhet) och kurtosis (toppighet) i fördelningen.",
        )
    desc_rows = []
    for d in se.descriptive_stats_by_subscale(dataset):
        desc_rows.append(
            {
                "Delskala": d.subscale_name,
                "N": d.n,
                "Mean": round(d.mean, 2) if d.mean is not None else "–",
                "SD": round(d.sd, 2) if d.sd is not None else "–",
                "Min–Max": f"{d.minimum:g}–{d.maximum:g}" if d.minimum is not None else "–",
                "Skewness": round(d.skewness, 2) if d.skewness is not None else "–",
            }
        )
    desc_df = pd.DataFrame(desc_rows)
    st.dataframe(desc_df, width="stretch", hide_index=True)

st.write("")

# --- Reliability snapshot / data quality -------------------------------------------------
col_rel, col_qc = st.columns(2)
with col_rel:
    header_cols = st.columns([5, 1])
    header_cols[0].markdown("**Reliabilitet (Cronbach's alpha)**")
    with header_cols[1]:
        concept_tooltip(
            "Cronbach's alpha",
            "Mäter intern konsistens - hur väl items i en delskala mäter samma sak. Tumregel: "
            "≥0.90 utmärkt, ≥0.80 bra, ≥0.70 acceptabelt, <0.70 bör granskas.",
        )
    rows = []
    for r in se.reliability_snapshot_all_subscales(dataset):
        rows.append(
            {
                "Delskala": r.subscale_name,
                "Items": r.n_items,
                "N": r.n,
                "Alpha": round(r.alpha, 2) if r.alpha is not None else "–",
                "Medel item-total r": round(r.mean_item_total_r, 2) if r.mean_item_total_r is not None else "–",
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("Fullständig reliabilitetsanalys finns i Reliability Explorer.")

with col_qc:
    st.markdown("**Datakvalitet**")
    summary = se.quality_summary(dataset)
    label = {"good": "Bra", "warning": "Bör granskas", "bad": "Problem"}[summary.status]
    st.markdown(
        f"""
        <div class="pve-card" style="display:flex; flex-direction:column; justify-content:center; height:100%;">
            <div style="font-size:1.6rem;">{status_icon(summary.status)} <b>{label}</b></div>
            <div style="color:#6B7280; margin-top:0.4rem;">{summary.message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Detaljerad kvalitetskontroll finns i Psychometric QC.")

st.write("")

if q.source_citation:
    st.info(f"ℹ️ {q.source_citation}")

st.write("")
render_export_section(
    dataset,
    "dataset_overview",
    table=desc_df,
    table_label="Beskrivande statistik",
    figures={"svarsfordelning": dist_fig, "bortfall_per_item": missing_fig},
)

st.write("")
col_back, _, col_next = st.columns([1, 3, 1])
with col_back:
    if st.button("Tillbaka", width="stretch"):
        st.switch_page("pages/1_Import_Wizard.py")
with col_next:
    if st.button("Fortsätt till Psychometric QC →", type="primary", width="stretch"):
        st.switch_page("pages/3_Psychometric_QC.py")
