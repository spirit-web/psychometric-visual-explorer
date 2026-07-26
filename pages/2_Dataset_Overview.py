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

col_title, col_new = st.columns([5, 1])
with col_title:
    st.title("Dataset Overview")
    st.caption(f"Översikt av ditt {q.test_name}-dataset ({q.full_name})")
with col_new:
    st.write("")
    if st.button("📤 Ladda nytt dataset", width="stretch"):
        st.session_state["pve_dataset"] = None
        st.session_state["iw_step"] = 1
        st.session_state["iw_raw_df"] = None
        st.session_state["iw_auto_matched"] = False
        st.switch_page("pages/1_Import_Wizard.py")

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

# --- Descriptive stats (moved up - useful to see early) -------------------------------------------------
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

# --- Demographics / reliability snapshot -------------------------------------------------
col_demo, col_rel = st.columns(2)
with col_demo:
    st.markdown("**Demografi – Kön**")
    gender = se.demographic_breakdown(dataset, "gender")
    if gender is not None and not gender.empty:
        st.plotly_chart(ve.demographics_donut(gender), width="stretch")
    else:
        st.info("Ingen könsvariabel hittades i datasetet.")

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

st.markdown("**Datakvalitet**")
summary = se.quality_summary(dataset)
label = {"good": "Bra", "warning": "Bör granskas", "bad": "Problem"}[summary.status]
st.markdown(
    f"""
    <div class="pve-card">
        <div style="font-size:1.6rem;">{status_icon(summary.status)} <b>{label}</b></div>
        <div style="color:#6B7280; margin-top:0.4rem;">{summary.message}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Detaljerad kvalitetskontroll finns i Psychometric QC.")

st.write("")

# --- Testprofil: struktur, korrelationer och svarsfördelning per item -------------------------------------------------
st.markdown("### Testprofil")
scale = q.response_scale
reverse_ids = q.reverse_scored_ids
profile_kpi_cols = st.columns(2)
with profile_kpi_cols[0]:
    kpi_card(
        "⚖️", "#FFEDD5", f"{scale.min}–{scale.max}", f"Svarsskala (Likert {scale.max - scale.min + 1})",
        tooltip=(
            "Antalet svarsalternativ per fråga, t.ex. 0-3 (\"Inte alls\" till \"Nästan varje dag\"). "
            "En Likert-skala med fler steg ger mer detaljerad information men kan vara svårare att "
            "svara konsekvent på."
        ),
    )
with profile_kpi_cols[1]:
    kpi_card(
        "👤", "#DBEAFE", str(len(reverse_ids)), "Omvänt item" if len(reverse_ids) == 1 else "Omvända items",
        tooltip=(
            "Antal frågor som är formulerade \"åt andra hållet\" (t.ex. en positivt formulerad fråga i "
            "ett test om ångest) och därför poängsätts baklänges innan de summeras ihop med övriga "
            "frågor. Görs för att motverka slentrianmässiga svar."
        ),
    )

st.write("")
col_corr, col_dist_item = st.columns(2)
with col_corr:
    header_cols = st.columns([5, 1])
    header_cols[0].markdown("**Korrelationer mellan items**")
    with header_cols[1]:
        concept_tooltip(
            "Item-korrelationer",
            "Pearson-korrelation mellan varje par av items. Höga positiva värden (mörkrött) "
            "tyder på att items mäter samma underliggande konstrukt.",
        )
    corr = se.item_correlation_matrix(dataset)
    corr_fig = None
    if not corr.empty:
        corr_fig = ve.correlation_heatmap(corr)
        st.plotly_chart(corr_fig, width="stretch", key="corr_heatmap_overview")
    else:
        st.info("För få items för en korrelationsmatris.")

with col_dist_item:
    header_cols = st.columns([5, 1])
    header_cols[0].markdown("**Svarsfördelning per item**")
    with header_cols[1]:
        concept_tooltip(
            "Svarsfördelning per item",
            "Andel svar i varje kategori för respektive item. Används för att upptäcka "
            "golv-/takeffekter eller items som beter sig annorlunda än övriga.",
        )
    item_dist = se.response_distribution_by_item(dataset)
    item_dist_fig = None
    if not item_dist.empty:
        label_map = {item.id: f"{item.id} – {item.text}" for item in q.items}
        item_dist_labeled = item_dist.rename(index=label_map)
        item_dist_fig = ve.stacked_response_distribution_chart(item_dist_labeled)
        st.plotly_chart(item_dist_fig, width="stretch")
    else:
        st.info("Inga items att visa.")

st.write("")

if q.source_citation:
    st.info(f"ℹ️ {q.source_citation}")

st.write("")
export_figures = {"svarsfordelning": dist_fig, "bortfall_per_item": missing_fig}
if corr_fig is not None:
    export_figures["item_korrelationer"] = corr_fig
if item_dist_fig is not None:
    export_figures["svarsfordelning_per_item"] = item_dist_fig
render_export_section(
    dataset,
    "dataset_overview",
    table=desc_df,
    table_label="Beskrivande statistik",
    figures=export_figures,
)

st.write("")
col_back, _, col_next = st.columns([1, 3, 1])
with col_back:
    if st.button("Tillbaka", width="stretch"):
        st.switch_page("pages/1_Import_Wizard.py")
with col_next:
    if st.button("Fortsätt till Psychometric QC →", type="primary", width="stretch"):
        st.switch_page("pages/3_Psychometric_QC.py")
