"""Test Profile Explorer page. Rendering only - all statistics come from core/."""

import pandas as pd
import streamlit as st

from components.concept_tooltip import concept_tooltip
from components.kpi_card import kpi_card
from core import stats_engine as se
from core import viz_engine as ve
from utils.session import require_dataset

dataset = require_dataset()
q = dataset.questionnaire

st.title("Test Profile Explorer")
st.caption(f"Utforska struktur och egenskaper hos {q.test_name}")

reverse_ids = set(q.reverse_scored_ids)
scale = q.response_scale

# --- KPI row -------------------------------------------------
kpi_cols = st.columns(4)
with kpi_cols[0]:
    kpi_card("📋", "#D1FAE5", str(dataset.n_items), "Items (frågor)")
with kpi_cols[1]:
    kpi_card("🧩", "#EDE9FE", str(dataset.n_subscales), "Delskalor")
with kpi_cols[2]:
    kpi_card("⚖️", "#FFEDD5", f"{scale.min}–{scale.max}", f"Svarsskala (Likert {scale.max - scale.min + 1})")
with kpi_cols[3]:
    kpi_card("👤", "#DBEAFE", str(len(reverse_ids)), "Omvänd item" if len(reverse_ids) == 1 else "Omvända items")

st.write("")

tab_overview, tab_items, tab_subscales, tab_dist, tab_corr = st.tabs(
    ["Översikt", "Items", "Delskalar", "Svarsfördelningar", "Korrelationer"]
)

with tab_overview:
    col_items, col_corr = st.columns(2)
    with col_items:
        st.markdown("**Items – Snabböversikt**")
        rows = [
            {
                "Item": item.id,
                "Fråga": item.text,
                "Delskala": q.subscale_for_item(item.id) or "–",
                "Omvänd": "Ja" if item.reverse_scored else "Nej",
            }
            for item in q.items
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=380)

    with col_corr:
        header_cols = st.columns([5, 1])
        header_cols[0].markdown("**Korrelationer mellan items**")
        with header_cols[1]:
            concept_tooltip(
                "Item-korrelationer",
                "Pearson-korrelation mellan varje par av items. Höga positiva värden (mörkblått) "
                "tyder på att items mäter samma underliggande konstrukt.",
            )
        corr = se.item_correlation_matrix(dataset)
        if not corr.empty:
            st.plotly_chart(ve.correlation_heatmap(corr), width="stretch", key="corr_heatmap_overview")
        else:
            st.info("För få items för en korrelationsmatris.")

    st.write("")
    st.markdown("**Delskalöversikt**")
    sub_cols = st.columns(min(4, max(1, dataset.n_subscales)))
    for i, subscale in enumerate(q.subscales):
        with sub_cols[i % len(sub_cols)]:
            st.markdown(
                f"""
                <div class="pve-card">
                    <div style="font-weight:700;">{subscale.name}</div>
                    <div style="color:#6B7280; font-size:0.85rem; margin-top:0.3rem;">
                        {len(subscale.item_ids)} items · Skala: {scale.min}–{scale.max} ·
                        Summa: {subscale.score_range[0]}–{subscale.score_range[1]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with tab_items:
    st.markdown("**Item-nivå: beskrivande statistik och reliabilitetsbidrag**")
    rows = []
    for item_stats in se.item_level_table(dataset):
        rows.append(
            {
                "Item": item_stats.item_id,
                "Fråga": item_stats.text,
                "Delskala": item_stats.subscale_name,
                "Omvänd": "Ja" if item_stats.reverse_scored else "Nej",
                "Mean": round(item_stats.mean, 2) if item_stats.mean is not None else "–",
                "SD": round(item_stats.sd, 2) if item_stats.sd is not None else "–",
                "Item-total r": round(item_stats.item_total_r, 2) if item_stats.item_total_r is not None else "–",
                "Alpha om borttaget": round(item_stats.alpha_if_deleted, 3) if item_stats.alpha_if_deleted is not None else "–",
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

with tab_subscales:
    for subscale in q.subscales:
        with st.expander(f"{subscale.name} ({len(subscale.item_ids)} items)", expanded=dataset.n_subscales == 1):
            snapshot = se.reliability_snapshot(dataset, subscale.id)
            info_cols = st.columns(3)
            info_cols[0].metric("Antal items", len(subscale.item_ids))
            info_cols[1].metric("Summa-intervall", f"{subscale.score_range[0]}–{subscale.score_range[1]}")
            info_cols[2].metric("Cronbach's alpha", f"{snapshot.alpha:.2f}" if snapshot.alpha is not None else "–")
            item_rows = [
                {"Item": iid, "Fråga": next(i.text for i in q.items if i.id == iid)}
                for iid in subscale.item_ids
            ]
            st.dataframe(pd.DataFrame(item_rows), width="stretch", hide_index=True)

with tab_dist:
    header_cols = st.columns([5, 1])
    header_cols[0].markdown("**Svarsfördelning per item**")
    with header_cols[1]:
        concept_tooltip(
            "Svarsfördelning per item",
            "Andel svar i varje kategori för respektive item. Används för att upptäcka "
            "golv-/takeffekter eller items som beter sig annorlunda än övriga.",
        )
    item_dist = se.response_distribution_by_item(dataset)
    if not item_dist.empty:
        st.plotly_chart(ve.stacked_response_distribution_chart(item_dist), width="stretch")
    else:
        st.info("Inga items att visa.")

with tab_corr:
    st.markdown("**Korrelationer mellan items (fullständig)**")
    corr = se.item_correlation_matrix(dataset)
    if not corr.empty:
        st.plotly_chart(ve.correlation_heatmap(corr), width="stretch", key="corr_heatmap_full")
    else:
        st.info("För få items för en korrelationsmatris.")

st.write("")
col_back, _, col_next = st.columns([1, 3, 1])
with col_back:
    if st.button("Tillbaka", width="stretch"):
        st.switch_page("pages/3_Psychometric_QC.py")
with col_next:
    if st.button("Fortsätt till Reliability Explorer →", type="primary", width="stretch"):
        st.switch_page("pages/5_Reliability_Explorer.py")
