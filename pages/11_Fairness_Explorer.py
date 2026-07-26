"""Fairness Explorer page. Rendering only - all statistics come from core/."""

import pandas as pd
import streamlit as st

from components.client_selector import select_or_enter_client
from components.concept_tooltip import concept_tooltip
from components.export_section import render_export_section
from components.kpi_card import kpi_card, status_icon
from core import stats_engine as se
from core import viz_engine as ve
from utils.session import require_dataset

dataset = require_dataset()
q = dataset.questionnaire

st.title("Fairness Explorer")
st.caption(f"Utvärdera rättvisa och jämlikhet över grupper för {q.test_name}")

subscale_options = {s.name: s.id for s in q.subscales} if len(q.subscales) > 1 else {}
if subscale_options:
    subscale_name = st.selectbox("Delskala", subscale_options.keys())
    subscale_id = subscale_options[subscale_name]
else:
    subscale_id = q.subscales[0].id if q.subscales else None

results = se.all_group_comparisons(dataset, subscale_id)
summary = se.fairness_summary(results)

BIAS_LABELS = {"low": "Låg", "moderate": "Måttlig", "high": "Hög", "none": "Okänd"}
BIAS_STATUS = {"low": "good", "moderate": "warning", "high": "bad", "none": "warning"}

if not results:
    st.info(
        "Inga demografiska grupperingsvariabler (kön, ålder, utbildning, grupp) hittades i "
        "datasetet. Fairness Explorer kräver minst en sådan kolumn."
    )
    st.stop()

# --- KPI row -------------------------------------------------
kpi_cols = st.columns(4)
with kpi_cols[0]:
    kpi_card("👥", "#DBEAFE", str(summary.n_dimensions), "Grupper analyserade", caption=", ".join(sorted({r.dimension for r in results})))
with kpi_cols[1]:
    kpi_card("⚖️", "#D1FAE5", f"{summary.mean_fairness_index:.2f}" if summary.mean_fairness_index is not None else "–", "Rättviseindex (genomsnitt)")
with kpi_cols[2]:
    kpi_card("↔️", "#EDE9FE", f"{summary.max_abs_d:.2f}" if summary.max_abs_d is not None else "–", "Största skillnad", caption=summary.max_d_label)
with kpi_cols[3]:
    kpi_card(status_icon(BIAS_STATUS[summary.bias_level]), "#FFEDD5", BIAS_LABELS[summary.bias_level], "Systematisk bias")

st.write("")

tab_overview, tab_client, tab_invariance = st.tabs(["Översikt", "Klient vs. grupper", "DIF & Measurement Invariance"])

with tab_overview:
    col_table, col_chart = st.columns(2)
    with col_table:
        header_cols = st.columns([5, 1])
        header_cols[0].markdown("**Rättviseindex per grupp**")
        with header_cols[1]:
            concept_tooltip(
                "Rättviseindex",
                "Rättviseindex nära 1.0 indikerar hög rättvisa mellan grupper (ingen meningsfull "
                "skillnad i medelpoäng). Värden under 0.75 (motsvarande |d| ≥ 0.5) kan indikera "
                "potentiell orättvisa värd att undersöka vidare.",
            )
        rows = [
            {
                "Grupp": f"{r.dimension} ({r.comparison_group} vs {r.reference_group})",
                "Rättviseindex": round(r.fairness_index, 2),
                "Tolkning": r.interpretation,
            }
            for r in results
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with col_chart:
        st.markdown("**Skillnad i medelvärde mellan grupper**")
        labels = [f"{r.dimension}<br>{r.comparison_group} vs {r.reference_group}" for r in results]
        d_values = [r.cohens_d for r in results]
        fairness_fig = ve.group_comparison_bar_chart(labels, d_values)
        st.plotly_chart(fairness_fig, width="stretch")
        st.caption("Positiva värden indikerar högre medelvärde för jämförelsegruppen.")

    st.write("")
    st.markdown("**Rekommendationer för rättvisa**")
    if summary.bias_level == "low":
        st.success("✅ Inga meningsfulla gruppskillnader identifierade. Fortsätt övervaka rättvisa över tid och vid nya grupper.")
    elif summary.bias_level == "moderate":
        st.warning(f"⚠️ Måttlig skillnad identifierad ({summary.max_d_label}, d = {summary.max_abs_d:.2f}). Undersök källan närmare innan testet används för viktiga beslut.")
    else:
        st.error(f"🛑 Stor skillnad identifierad ({summary.max_d_label}, d = {summary.max_abs_d:.2f}). Överväg normjustering eller granska testet för potentiell bias innan det används för beslut.")

    st.write("")
    st.markdown("**Detaljerad tabell**")
    fairness_detail_df = pd.DataFrame(
        [
            {
                "Dimension": r.dimension,
                "Referensgrupp": f"{r.reference_group} (n={r.n_reference})",
                "Jämförelsegrupp": f"{r.comparison_group} (n={r.n_comparison})",
                "Medel (ref)": round(r.mean_reference, 2),
                "Medel (jämf.)": round(r.mean_comparison, 2),
                "Cohen's d": round(r.cohens_d, 3),
            }
            for r in results
        ]
    )
    st.dataframe(fairness_detail_df, width="stretch", hide_index=True)

with tab_client:
    st.caption(
        "Placera en enskild klients resultat mot varje grupps medelvärde - oavsett vilken grupp "
        "klienten själv tillhör, ser du om resultatet ser typiskt ut jämfört med alla grupper."
    )
    client_score = select_or_enter_client(dataset, subscale_id, key_prefix="fairness")

    if client_score is None:
        st.info("Otillräcklig data för att beräkna klientens poäng.")
    else:
        labels = [f"{r.dimension}: {r.comparison_group} vs {r.reference_group}" for r in results]
        ref_means = [r.mean_reference for r in results]
        comp_means = [r.mean_comparison for r in results]
        client_fig = ve.group_means_with_client_chart(
            labels, ref_means, comp_means, "Referensgrupp", "Jämförelsegrupp", client_score
        )
        st.plotly_chart(client_fig, width="stretch")
        st.metric("Klientens poäng", f"{client_score:g}")
        st.caption(
            "Om klientens poäng ligger nära alla gruppers medelvärden oavsett grupptillhörighet "
            "är resultatet svårt att förklara med systematisk bias mellan grupperna."
        )

with tab_invariance:
    st.write(
        "**Differentiell itemfunktion (DIF)** undersöker om enskilda frågor fungerar olika för "
        "olika grupper trots samma underliggande nivå på konstruktet. **Measurement invariance** "
        "testar (t.ex. via flergrupps-CFA) om hela mätmodellen - itemladdningar, intercept och "
        "residualvarianser - är jämförbar mellan grupper. Utan invarians kan man inte med "
        "säkerhet jämföra latenta medelvärden mellan grupperna."
    )
    st.warning(
        "⚠️ Fullständig DIF- och measurement invariance-analys är inte implementerad i denna "
        "version (kräver t.ex. ordinal logistisk regression per fråga eller flergrupps-CFA). "
        "Gruppjämförelserna i Översikt-fliken bygger på observerade medelvärden (Cohen's d), "
        "inte latenta poäng, och bör tolkas med försiktighet tills invarians är bekräftad."
    )
    st.selectbox(
        "Status för DIF/invarians-analys",
        ["none", "limited", "moderate", "strong"],
        format_func=lambda k: {"none": "Ej genomförd", "limited": "Planerad", "moderate": "Pågående", "strong": "Genomförd"}[k],
        key="fairness_invariance_status",
    )
    st.text_area("Anteckningar (sparas endast i denna session)", key="fairness_invariance_notes")

st.write("")
render_export_section(
    dataset,
    "fairness_explorer",
    table=fairness_detail_df,
    table_label="Gruppjämförelser",
    figures={"gruppskillnader": fairness_fig},
)

st.write("")
col_back, _, col_next = st.columns([1, 3, 1])
with col_back:
    if st.button("Tillbaka", width="stretch"):
        st.switch_page("pages/10_Decision_Support.py")
with col_next:
    if st.button("Fortsätt till Machine Learning →", type="primary", width="stretch"):
        st.switch_page("pages/12_Machine_Learning.py")
