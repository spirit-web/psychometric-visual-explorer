"""Measurement Error page. Rendering only - all statistics come from core/."""

import pandas as pd
import streamlit as st

from components.concept_tooltip import concept_tooltip
from components.export_section import render_export_section
from components.kpi_card import kpi_card
from core import stats_engine as se
from core import viz_engine as ve
from utils.session import require_dataset

dataset = require_dataset()
q = dataset.questionnaire

st.title("Measurement Error")
st.caption(f"Kvantifiera osäkerhet i testresultat för {q.test_name}")

subscale_options = {s.name: s.id for s in q.subscales} if len(q.subscales) > 1 else {}
if subscale_options:
    subscale_name = st.selectbox("Delskala", subscale_options.keys())
    subscale_id = subscale_options[subscale_name]
else:
    subscale_id = q.subscales[0].id if q.subscales else None

me = se.measurement_error(dataset, subscale_id)
mrc = se.minimum_reliable_change(me.sem) if me.sem is not None else None
precision = se.precision_label(me.sem, me.sd)

# --- KPI row -------------------------------------------------
kpi_cols = st.columns(4)
with kpi_cols[0]:
    kpi_card(
        "🎯", "#DBEAFE", f"{me.sem:.2f}" if me.sem is not None else "–", "Standard Error of Measurement", caption="SEM",
        tooltip=(
            "Hur mycket en persons poäng kan variera av ren slump om de tog samma test igen utan att "
            "ha förändrats. Ju lägre SEM, desto mer kan du lita på en enskild mätning."
        ),
    )
with kpi_cols[1]:
    kpi_card(
        "✅", "#D1FAE5", f"{mrc:.1f}" if mrc is not None else "–", "Minimalt pålitligt förändringsvärde", caption="Reliable Change Index",
        tooltip=(
            "Hur stor förändring i poäng som krävs mellan två mättillfällen för att räknas som en "
            "verklig förändring - inte bara mätbrus. Central när du följer en klient över flera "
            "sessioner: en mindre förändring än detta bör inte tolkas som förbättring eller försämring."
        ),
    )
with kpi_cols[2]:
    kpi_card(
        "📈", "#EDE9FE", f"{me.alpha:.2f}" if me.alpha is not None else "–", "Reliabilitet (alpha)",
        tooltip="Samma Cronbach's alpha som i Reliability Explorer - visas här eftersom SEM räknas direkt utifrån den.",
    )
with kpi_cols[3]:
    kpi_card(
        "📊", "#FFEDD5", precision, "Precision",
        tooltip="En sammanfattande bedömning av mätsäkerheten (baserad på SEM i relation till skalans spridning) - Hög/Måttlig/Låg.",
    )

st.write("")

tab_overview, tab_ci, tab_change = st.tabs(["Översikt", "Konfidensintervall", "Reliable Change"])

with tab_overview:
    col_formula, col_curve = st.columns(2)
    with col_formula:
        st.markdown("**Vad är mätfel?**")
        st.write(
            "Mätfel (ME) är skillnaden mellan en observerad poäng (X) och den sanna poängen (T). "
            "Standard Error of Measurement (SEM) uppskattar storleken på det förväntade mätfelet."
        )
        st.latex(r"SEM = SD \times \sqrt{1 - \alpha}")
        st.latex(r"95\%\ KI = X \pm 1.96 \times SEM")
        if me.sd is not None and me.alpha is not None:
            st.caption(f"Här: SD = {me.sd:.2f}, alpha = {me.alpha:.2f} → SEM = {me.sem:.2f}")

    with col_curve:
        header_cols = st.columns([5, 1])
        header_cols[0].markdown("**Exempel: konfidensintervall för en person**")
        with header_cols[1]:
            concept_tooltip(
                "Sant poäng-intervall",
                "Kurvan visar den uppskattade fördelningen av den sanna poängen T runt en "
                "observerad poäng X, med SEM som spridningsmått. Det skuggade området är 95% KI.",
            )
        ids = se.person_ids(dataset)
        person_id = st.selectbox("Välj person", ids, format_func=lambda pid: f"Person {pid}", key="me_person")
        raw_score = se.person_raw_score(dataset, person_id, subscale_id)
        if raw_score is not None and me.sem:
            x, y = se.normal_curve_points(raw_score, me.sem)
            ci = se.confidence_interval(raw_score, me.sem)
            if len(x):
                st.plotly_chart(
                    ve.normal_curve_chart(x, y, raw_score, f"Observerad poäng {raw_score:g}", "Poäng", ci=ci),
                    width="stretch",
                )
            st.caption(f"95% KI: {ci[0]:.1f} – {ci[1]:.1f}")
        else:
            st.info("Otillräcklig data för att beräkna konfidensintervall.")

with tab_ci:
    header_cols = st.columns([5, 1])
    header_cols[0].markdown("**Konfidensintervall för alla individuella resultat**")
    with header_cols[1]:
        concept_tooltip(
            "Konfidensintervall",
            "Felintervall hjälper dig att tolka om förändringar i resultat är verkliga eller kan "
            "bero på mätosäkerhet.",
        )
    ci_table = se.all_person_confidence_intervals(dataset, subscale_id)
    ci_fig = None
    if not ci_table.empty:
        preview = ci_table.head(30)
        ci_fig = ve.ci_error_bar_chart(
            preview["Person"], preview["Observerad poäng"], preview["95% KI nedre"], preview["95% KI övre"], "Poäng"
        )
        st.plotly_chart(ci_fig, width="stretch")
        st.caption("Visar de första 30 personerna. Fullständig tabell nedan.")
        st.dataframe(ci_table, width="stretch", height=350)
    else:
        st.info("Otillräcklig data för konfidensintervall.")

with tab_change:
    header_cols = st.columns([5, 1])
    header_cols[0].markdown("**Reliable Change Index (test-retest)**")
    with header_cols[1]:
        concept_tooltip(
            "Reliable Change Index",
            "RCI = (poäng vid tid 2 − poäng vid tid 1) / SE_diff, där SE_diff = SEM × √2. "
            "|RCI| ≥ 1.96 tolkas som en statistiskt pålitlig förändring (Jacobson & Truax, 1991), "
            "inte bara mätbrus.",
        )
    rci_results = se.reliable_change_index(dataset, subscale_id)
    if rci_results:
        n_reliable = sum(1 for r in rci_results if r.reliable)
        metric_cols = st.columns(3)
        metric_cols[0].metric("Personer med retest-data", len(rci_results))
        metric_cols[1].metric("Pålitlig förändring", n_reliable)
        metric_cols[2].metric("Andel", f"{n_reliable / len(rci_results) * 100:.0f}%")

        rows = [
            {
                "Person": r.person_id,
                "Tid 1": r.score_t1,
                "Tid 2": r.score_t2,
                "Förändring": round(r.diff, 1),
                "RCI": round(r.rci, 2),
                "Pålitlig förändring": "Ja" if r.reliable else "Nej",
            }
            for r in rci_results
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", height=350)
        st.caption(
            "Baserat på simulerad retest-data (~2 veckor senare, se data/generate_sample_data.py) "
            "- endast den delmängd som besvarade testet vid båda tillfällena visas."
        )
    else:
        st.info("Ingen retest-data tillgänglig för denna delskala.")

st.write("")
render_export_section(
    dataset,
    "measurement_error",
    table=ci_table,
    table_label="Konfidensintervall per person",
    figures={"konfidensintervall": ci_fig} if ci_fig is not None else None,
)

st.write("")
col_back, _, col_next = st.columns([1, 3, 1])
with col_back:
    if st.button("Tillbaka", width="stretch"):
        st.switch_page("pages/8_Norm_Explorer.py")
with col_next:
    if st.button("Fortsätt till Decision Support →", type="primary", width="stretch"):
        st.switch_page("pages/10_Decision_Support.py")
