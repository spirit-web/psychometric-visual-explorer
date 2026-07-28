"""Norm Explorer page. Rendering only - all statistics come from core/."""

import streamlit as st

from components.client_selector import select_or_enter_client
from components.concept_tooltip import concept_tooltip
from components.export_section import render_export_section
from components.kpi_card import kpi_card
from core import stats_engine as se
from core import viz_engine as ve
from utils.session import require_dataset

dataset = require_dataset()
q = dataset.questionnaire

st.title("Norm Explorer")
st.caption(f"Jämför en klients resultat med normgruppen för {q.test_name}")

subscale_options = {s.name: s.id for s in q.subscales} if len(q.subscales) > 1 else {}
if subscale_options:
    subscale_name = st.selectbox("Delskala", subscale_options.keys())
    subscale_id = subscale_options[subscale_name]
else:
    subscale_id = q.subscales[0].id if q.subscales else None

stats = se.norm_stats(dataset, subscale_id)
me = se.measurement_error(dataset, subscale_id)

# --- KPI row -------------------------------------------------
kpi_cols = st.columns(4)
with kpi_cols[0]:
    kpi_card("👥", "#DBEAFE", "Denna datamängd", "Normgrupp", caption="Sample-baserad")
with kpi_cols[1]:
    kpi_card("🧮", "#D1FAE5", str(stats.n), "Antal i normgrupp", caption="Personer")
with kpi_cols[2]:
    kpi_card("📐", "#EDE9FE", "Percentil", "Normtyp")
with kpi_cols[3]:
    kpi_card("🧩", "#FFEDD5", str(dataset.n_subscales), "Delskalor")

st.info(
    "ℹ️ Normerna här är beräknade från datasetets egna medelvärde och standardavvikelse - "
    "detta är **inte** en extern, standardiserad populationsnorm, utan en sample-baserad "
    "referens. Ersätt med riktiga normdata om sådana finns tillgängliga."
)

st.write("")

tab_overview, tab_table = st.tabs(["Klientöversikt", "Omvandlingstabell"])

with tab_overview:
    raw_score = select_or_enter_client(dataset, subscale_id, key_prefix="norm")

    if raw_score is None or stats.mean is None or stats.sd is None or stats.sd == 0:
        st.info("Otillräcklig data för att beräkna normjämförelse.")
    else:
        conversion = se.score_conversion(raw_score, stats.mean, stats.sd)
        category = se.cutoff_category(q, raw_score)
        ci = se.confidence_interval(raw_score, me.sem) if me.sem is not None else None

        col_chart, col_summary = st.columns([2, 1])
        with col_chart:
            header_cols = st.columns([5, 1])
            header_cols[0].markdown("**Resultat i relation till normgruppen**")
            with header_cols[1]:
                concept_tooltip(
                    "Normfördelning",
                    "Den blå kurvan visar hur totalpoängen fördelar sig i normgruppen (antas "
                    "normalfördelad). Den streckade linjen markerar den valda personens resultat.",
                )
            x, y = se.normal_curve_points(stats.mean, stats.sd)
            if len(x):
                st.plotly_chart(
                    ve.normal_curve_chart(x, y, raw_score, f"Råpoäng {raw_score:g}", "Råpoäng"),
                    width="stretch",
                )
            metric_cols = st.columns(3)
            metric_cols[0].metric("Percentil", f"{conversion.percentile:.0f}")
            metric_cols[1].metric("T-poäng", f"{conversion.t:.0f}")
            metric_cols[2].metric("Z-poäng", f"{conversion.z:.2f}")
            st.caption("Högre värden indikerar högre nivå på egenskapen (om inget annat anges).")

        with col_summary:
            st.markdown("**Resultatsammanfattning**")
            category_row = (
                f"""<div style="display:flex; justify-content:space-between; margin-bottom:0.7rem;">
                        <span style="color:#6B7280;">Nivå</span>
                        <b>{category}</b>
                    </div>"""
                if category
                else ""
            )
            ci_row = (
                f"""<div style="display:flex; justify-content:space-between; margin-bottom:0.7rem;">
                        <span style="color:#6B7280;">95% KI (mätfel)</span>
                        <b>{ci[0]:.1f} – {ci[1]:.1f}</b>
                    </div>"""
                if ci is not None
                else ""
            )
            st.markdown(
                f"""
                <div class="pve-card">
                    <div style="display:flex; justify-content:space-between; margin-bottom:0.7rem;">
                        <span style="color:#6B7280;">Råpoäng</span>
                        <b>{raw_score:g}</b>
                    </div>
                    {category_row}
                    <div style="display:flex; justify-content:space-between; margin-bottom:0.7rem;">
                        <span style="color:#6B7280;">Percentil</span>
                        <b>{conversion.percentile:.0f}</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:0.7rem;">
                        <span style="color:#6B7280;">Jämfört med medel</span>
                        <b>{conversion.z:+.2f} SD</b>
                    </div>
                    {ci_row}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.info(f"ℹ️ Resultatet ligger högre än {conversion.percentile:.0f}% av personerna i normgruppen.")
            if ci is not None:
                st.caption(
                    f"95% KI byggt på mätfelet (SEM = {me.sem:.2f}) - se Measurement Error för djupare analys, "
                    "inklusive om en förändring över tid är statistiskt pålitlig."
                )

with tab_table:
    header_cols = st.columns([5, 1])
    header_cols[0].markdown("**Poängomvandlingstabell**")
    with header_cols[1]:
        concept_tooltip(
            "Stanine",
            "En niogradig skala (1-9) med medel 5 och SD 2, ofta använd för enkel kommunikation "
            "av resultat. Bygger på samma z-poäng som T-poäng och percentil.",
        )
    conversion_df = se.conversion_table(dataset, subscale_id)
    if not conversion_df.empty:
        st.dataframe(conversion_df, width="stretch", height=420)
        st.caption("T-poäng = 50 + 10 × Z. Percentil och stanine beräknas från den normala fördelningen.")
    else:
        st.info("Otillräcklig data för omvandlingstabell.")

st.write("")
render_export_section(dataset, "norm_explorer", table=conversion_df, table_label="Poängomvandlingstabell")

st.write("")
col_back, _, col_next = st.columns([1, 3, 1])
with col_back:
    if st.button("Tillbaka", width="stretch"):
        st.switch_page("pages/3_Psychometric_QC.py")
with col_next:
    if st.button("Fortsätt till Decision Support →", type="primary", width="stretch"):
        st.switch_page("pages/10_Decision_Support.py")
