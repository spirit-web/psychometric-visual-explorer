"""Validity Dashboard page. Rendering only - all statistics come from core/."""

import pandas as pd
import streamlit as st

from components.concept_tooltip import concept_tooltip
from components.export_section import render_export_section
from components.kpi_card import kpi_card, status_icon
from core import plugin_engine as pe
from core import stats_engine as se
from core import viz_engine as ve
from utils.session import require_dataset

dataset = require_dataset()
q = dataset.questionnaire

st.title("Validity Dashboard")
st.caption(f"Sammanställning av evidens för {q.test_name} enligt fem evidenskällor (AERA/APA/NCME)")

STATUS_OPTIONS = {
    "none": "Ingen information",
    "limited": "Begränsad evidens",
    "moderate": "Måttlig evidens",
    "strong": "Stark evidens",
}
STATUS_KEYS = list(STATUS_OPTIONS.keys())

# For the bundled, well-established tests, pre-fill response-process and
# consequences evidence from their published validation literature instead
# of defaulting to "no information" - that default is honest for a custom
# test built in Test Builder, but misleading for GAD-7/PHQ-9/Big Five, which
# do have this evidence, just not entered by the current user.
default_evidence = pe.default_validity_evidence(q.plugin_id)
if "vd_response_process_status" not in st.session_state and "response_processes" in default_evidence:
    st.session_state["vd_response_process_status"] = default_evidence["response_processes"][0]
if "vd_consequences_status" not in st.session_state and "consequences" in default_evidence:
    st.session_state["vd_consequences_status"] = default_evidence["consequences"][0]

response_process_status = st.session_state.get("vd_response_process_status", "none")
consequences_status = st.session_state.get("vd_consequences_status", "none")

response_default_status, response_default_summary = default_evidence.get("response_processes", (None, None))
consequences_default_status, consequences_default_summary = default_evidence.get("consequences", (None, None))

sources = se.validity_overview(
    dataset,
    response_process_status,
    consequences_status,
    response_process_summary=response_default_summary if response_process_status == response_default_status else None,
    consequences_summary=consequences_default_summary if consequences_status == consequences_default_status else None,
)
sources_by_key = {s.key: s for s in sources}
counts = se.validity_status_counts(sources)

if counts["strong"] >= 4:
    agg_status, agg_label = "good", "Bra"
elif counts["strong"] >= 2:
    agg_status, agg_label = "warning", "Måttlig"
else:
    agg_status, agg_label = "bad", "Bör stärkas"

# --- KPI row -------------------------------------------------
kpi_cols = st.columns(4)
with kpi_cols[0]:
    kpi_card(status_icon(agg_status), "#DBEAFE", agg_label, "Validitetssammanfattning")
with kpi_cols[1]:
    kpi_card("✅", "#D1FAE5", f"{counts['strong']} av 5", "Kriterier uppfyllda", caption=f"{counts['strong'] / 5 * 100:.0f}% uppfyllelse")
with kpi_cols[2]:
    kpi_card("📋", "#EDE9FE", "5", "Evidenskällor", caption="AERA/APA/NCME")
with kpi_cols[3]:
    kpi_card("🛡️", "#FFEDD5", agg_label, "Evidensnivå")

st.write("")

tab_overview, tab_content, tab_response, tab_structure, tab_relations, tab_consequences = st.tabs(
    ["Översikt", "Innehåll", "Responsprocesser", "Intern struktur", "Relation till andra variabler", "Konsekvenser"]
)

with tab_overview:
    col_table, col_donut = st.columns([2, 1])
    with col_table:
        st.markdown("**Evidensöversikt**")
        validity_df = pd.DataFrame(
            [
                {"Evidenskälla": s.label, "Status": f"{status_icon(s.status)} {STATUS_OPTIONS[s.status]}", "Sammanfattning": s.summary}
                for s in sources
            ]
        )
        st.dataframe(validity_df, width="stretch", hide_index=True)
    with col_donut:
        st.markdown("**Evidens i korthet**")
        donut_data = pd.Series({STATUS_OPTIONS[k]: v for k, v in counts.items() if v > 0})
        if not donut_data.empty:
            st.plotly_chart(ve.demographics_donut(donut_data), width="stretch", key="validity_donut")

    st.write("")
    if agg_status == "good":
        st.success(f"✅ Sammanfattning: {q.test_name} visar god evidens för de flesta validitetstyper.")
    elif agg_status == "warning":
        st.warning(f"⚠️ Sammanfattning: {q.test_name} visar måttlig evidens - vissa källor bör stärkas.")
    else:
        st.error("🛑 Sammanfattning: Flera evidenskällor saknar stark dokumentation.")

with tab_content:
    source = sources_by_key["content"]
    st.markdown(f"**Status:** {status_icon(source.status)} {STATUS_OPTIONS[source.status]}")
    st.write(
        "Innehållsvaliditet bedömer i vilken grad testets frågor återspeglar det avsedda "
        "konstruktets innehåll och domän."
    )
    info_cols = st.columns(3)
    info_cols[0].metric("Antal frågor", dataset.n_items)
    info_cols[1].metric("Antal delskalor", dataset.n_subscales)
    info_cols[2].metric("Källa angiven", "Ja" if q.source_citation else "Nej")
    if q.source_citation:
        st.info(f"ℹ️ {q.source_citation}")
    st.text_area(
        "Anteckningar om innehållsvaliditet (sparas endast i denna session)",
        key="vd_content_notes",
        placeholder="T.ex. expertgranskning, översättningsprocess, målgruppsanpassning...",
    )

with tab_response:
    source = sources_by_key["response_processes"]
    st.markdown(f"**Status:** {status_icon(source.status)} {STATUS_OPTIONS[source.status]}")
    st.write(
        "Responsprocesser handlar om huruvida testpersoner tolkar och besvarar frågorna på det sätt "
        "konstruktet avsåg. Detta går sällan att beräkna från enbart svarsdata - dokumentera det "
        "manuellt här, t.ex. via kognitiva intervjuer eller tänka-högt-studier."
    )
    if response_process_status == response_default_status and response_default_summary:
        st.info(
            f"ℹ️ Förifyllt utifrån testets publicerade valideringslitteratur: {response_default_summary} "
            "Ändra nedan om du vill dokumentera egen evidens istället."
        )
    st.selectbox(
        "Evidensstatus",
        STATUS_KEYS,
        format_func=lambda k: STATUS_OPTIONS[k],
        key="vd_response_process_status",
    )
    st.text_area("Anteckningar (sparas endast i denna session)", key="vd_response_process_notes")
    st.text_input("Länk till dokumentation (valfritt)", key="vd_response_process_link")

with tab_structure:
    source = sources_by_key["internal_structure"]
    st.markdown(f"**Status:** {status_icon(source.status)} {STATUS_OPTIONS[source.status]}")
    st.write(
        "Intern struktur bedömer om frågor och delskalor samspelar på ett sätt som stödjer den "
        "avsedda faktorstrukturen - reliabilitet och faktoranalys är kärnan i denna evidenskälla."
    )
    overall = se.reliability_snapshot(dataset)
    efa = se.efa_fit(dataset, max(1, len(q.subscales))) if q.subscales else None
    info_cols = st.columns(3)
    info_cols[0].metric("Cronbach's alpha", f"{overall.alpha:.2f}" if overall.alpha is not None else "–")
    info_cols[1].metric("RMSEA", f"{efa.fit.rmsea:.3f}" if efa is not None and efa.fit.rmsea is not None else "–")
    info_cols[2].metric("CFI", f"{efa.fit.cfi:.2f}" if efa is not None and efa.fit.cfi is not None else "–")
    st.caption("Fullständig analys finns i Reliability Explorer och Factor Explorer.")

with tab_relations:
    source = sources_by_key["relations"]
    st.markdown(f"**Status:** {status_icon(source.status)} {STATUS_OPTIONS[source.status]}")
    st.write(
        "Relation till andra variabler undersöker om testet korrelerar som förväntat med "
        "närliggande (konvergent) och orelaterade (diskriminant) mått."
    )
    st.info(
        "ℹ️ De kriterievariabler som används här är **simulerade** för demonstrationssyfte "
        "(se `data/generate_sample_data.py`) - inte riktiga externa mått."
    )
    cv = se.criterion_validity(dataset)
    col_conv, col_disc = st.columns(2)
    with col_conv:
        st.markdown("**Konvergent validitet**")
        if cv.convergent_r is not None:
            st.metric("Korrelation (r)", f"{cv.convergent_r:.2f}")
            x, y = cv.convergent_pair
            trendline = se.linear_fit(x, y)
            st.plotly_chart(
                ve.scatter_with_regression(x, y, "Totalpoäng", "Simulerad kriterievariabel", trendline),
                width="stretch",
                key="convergent_scatter",
            )
        else:
            st.info("Ingen konvergent kriterievariabel tillgänglig.")
    with col_disc:
        st.markdown("**Diskriminant validitet**")
        if cv.discriminant_r is not None:
            st.metric("Korrelation (r)", f"{cv.discriminant_r:.2f}")
            x, y = cv.discriminant_pair
            trendline = se.linear_fit(x, y)
            st.plotly_chart(
                ve.scatter_with_regression(x, y, "Totalpoäng", "Simulerad kriterievariabel", trendline),
                width="stretch",
                key="discriminant_scatter",
            )
        else:
            st.info("Ingen diskriminant kriterievariabel tillgänglig.")

with tab_consequences:
    source = sources_by_key["consequences"]
    st.markdown(f"**Status:** {status_icon(source.status)} {STATUS_OPTIONS[source.status]}")
    st.write(
        "Konsekvenser av testanvändning gäller vilka effekter testet får för individer och grupper "
        "- t.ex. om beslut baserade på testet är rättvisa mellan grupper. Se Decision Support och "
        "Fairness Explorer för relaterad analys."
    )
    if consequences_status == consequences_default_status and consequences_default_summary:
        st.info(
            f"ℹ️ Förifyllt utifrån testets publicerade valideringslitteratur: {consequences_default_summary} "
            "Ändra nedan om du vill dokumentera egen evidens istället."
        )
    st.selectbox(
        "Evidensstatus",
        STATUS_KEYS,
        format_func=lambda k: STATUS_OPTIONS[k],
        key="vd_consequences_status",
    )
    st.text_area("Anteckningar (sparas endast i denna session)", key="vd_consequences_notes")

st.write("")
render_export_section(dataset, "validity_dashboard", table=validity_df, table_label="Evidensöversikt")

st.write("")
col_back, _, col_next = st.columns([1, 3, 1])
with col_back:
    if st.button("Tillbaka", width="stretch"):
        st.switch_page("pages/6_Factor_Explorer.py")
with col_next:
    if st.button("Fortsätt till Machine Learning →", type="primary", width="stretch"):
        st.switch_page("pages/12_Machine_Learning.py")
