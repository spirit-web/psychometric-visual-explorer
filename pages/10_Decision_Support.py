"""Decision Support page. Rendering only - all statistics come from core/."""

import streamlit as st

from components.concept_tooltip import concept_tooltip
from components.export_section import render_export_section
from components.kpi_card import kpi_card
from core import stats_engine as se
from core import viz_engine as ve
from utils.session import require_dataset

dataset = require_dataset()
q = dataset.questionnaire

st.title("Decision Support")
st.caption(f"ROC-analys och beslutsstöd för {q.test_name}")

if not se.has_outcome(dataset):
    st.info(
        "Ingen utfallsvariabel (`outcome_positive`) hittades i datasetet. Decision Support kräver "
        "en binär kriterievariabel (t.ex. klinisk diagnos) att utvärdera testpoängen mot. "
        "Exempeldatan innehåller en simulerad sådan variabel."
    )
    st.stop()

st.info(
    "ℹ️ Utfallsvariabeln som används här är **simulerad** för demonstrationssyfte "
    "(se `data/generate_sample_data.py`) - inte ett riktigt kliniskt facit."
)

subscale_options = {s.name: s.id for s in q.subscales} if len(q.subscales) > 1 else {}
if subscale_options:
    subscale_name = st.selectbox("Delskala", subscale_options.keys())
    subscale_id = subscale_options[subscale_name]
else:
    subscale_id = q.subscales[0].id if q.subscales else None

roc = se.roc_analysis(dataset, subscale_id)
optimal = se.youdens_optimal_threshold(dataset, subscale_id)

if roc is None or optimal is None:
    st.info("Otillräcklig data för ROC-analys (för få observationer eller endast en utfallsklass).")
    st.stop()

# --- KPI row -------------------------------------------------
kpi_cols = st.columns(4)
with kpi_cols[0]:
    ci_text = f"{roc.auc_ci[0]:.2f} – {roc.auc_ci[1]:.2f}" if roc.auc_ci else None
    kpi_card("📈", "#DBEAFE", f"{roc.auc:.2f}", "AUC", caption=f"95% KI: {ci_text}" if ci_text else None)
with kpi_cols[1]:
    kpi_card("🎯", "#D1FAE5", f"{optimal.threshold:g}", "Rekommenderad tröskel", caption=f"Youden's J = {optimal.youdens_j:.2f}")
with kpi_cols[2]:
    kpi_card("✅", "#EDE9FE", f"{optimal.sensitivity:.2f}", "Sensitivitet vid tröskel")
with kpi_cols[3]:
    kpi_card("🛡️", "#FFEDD5", f"{optimal.specificity:.2f}", "Specificitet vid tröskel")

st.write("")

tab_overview, tab_cutoffs = st.tabs(["Översikt", "Cut score & tröskelvärden"])

with tab_overview:
    lo, hi = q.get_subscale(subscale_id).score_range if subscale_id else (0, 0)
    threshold = st.slider("Välj tröskel", int(lo), int(hi), int(round(optimal.threshold)))
    metrics = se.confusion_at_threshold(dataset, threshold, subscale_id)

    col_roc, col_cm = st.columns(2)
    with col_roc:
        header_cols = st.columns([5, 1])
        header_cols[0].markdown("**ROC-kurva**")
        with header_cols[1]:
            concept_tooltip(
                "ROC-kurva & AUC",
                "ROC-kurvan visar avvägningen mellan sensitivitet och (1 - specificitet) vid olika "
                "trösklar. AUC (arean under kurvan) sammanfattar hur väl testet skiljer mellan "
                "positiva och negativa fall - 0.5 är slumpnivå, 1.0 är perfekt särskiljning. Det röda "
                "krysset visar var den valda tröskeln (i reglaget till höger) hamnar på kurvan.",
            )
        operating_point = (1 - metrics.specificity, metrics.sensitivity) if metrics is not None else None
        roc_fig = ve.roc_curve_chart(roc.fpr, roc.tpr, roc.auc, operating_point=operating_point)
        st.plotly_chart(roc_fig, width="stretch")

    with col_cm:
        st.markdown("**Konfusionsmatris**")
        if metrics is not None:
            st.plotly_chart(ve.confusion_matrix_heatmap(metrics.tp, metrics.fp, metrics.tn, metrics.fn), width="stretch")

    if metrics is not None:
        st.write("")
        st.markdown("**Prestandamått vid vald tröskel**")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Sensitivitet (TPR)", f"{metrics.sensitivity:.2f}")
        metric_cols[1].metric("Specificitet (TNR)", f"{metrics.specificity:.2f}")
        metric_cols[2].metric("PPV (Precision)", f"{metrics.ppv:.2f}")
        metric_cols[3].metric("NPV", f"{metrics.npv:.2f}")

        if threshold == round(optimal.threshold):
            st.success(f"✅ Denna tröskel maximerar Youden's J ({optimal.youdens_j:.2f}) - bästa balans mellan sensitivitet och specificitet.")

with tab_cutoffs:
    header_cols = st.columns([5, 1])
    header_cols[0].markdown("**Cut score & tröskelvärden**")
    with header_cols[1]:
        concept_tooltip(
            "Youden's J",
            "J = Sensitivitet + Specificitet - 1. Tröskeln som maximerar J ger den bästa balansen "
            "mellan att korrekt identifiera positiva och negativa fall.",
        )
    cutoff_df = se.cutoff_table(dataset, subscale_id)
    if not cutoff_df.empty:
        st.dataframe(cutoff_df, width="stretch", height=420)
        st.caption(
            f"Rekommenderad tröskel (max Youden's J): {optimal.threshold:g} "
            f"(Sensitivitet {optimal.sensitivity:.2f}, Specificitet {optimal.specificity:.2f})."
        )
    else:
        st.info("Otillräcklig data för tröskeltabell.")

st.write("")
render_export_section(
    dataset,
    "decision_support",
    table=cutoff_df,
    table_label="Cut score-tabell",
    figures={"roc_kurva": roc_fig},
)

st.write("")
col_back, _, col_next = st.columns([1, 3, 1])
with col_back:
    if st.button("Tillbaka", width="stretch"):
        st.switch_page("pages/9_Measurement_Error.py")
with col_next:
    if st.button("Fortsätt till Fairness Explorer →", type="primary", width="stretch"):
        st.switch_page("pages/11_Fairness_Explorer.py")
