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

st.markdown("<style>.block-container{padding-top:1rem;}</style>", unsafe_allow_html=True)
st.title("Decision Support")

if not se.has_outcome(dataset):
    st.info(
        "Ingen bekräftad diagnos (`outcome_positive`) hittades i datasetet. Decision Support kräver "
        "en binär referens att utvärdera testpoängen mot - t.ex. resultatet av en klinisk "
        "referensintervju, som den självrapporterade poängen sedan jämförs mot. Exempeldatan "
        "innehåller en simulerad sådan variabel."
    )
    st.stop()

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
    kpi_card(
        "📈", "#DBEAFE", f"{roc.auc:.2f}", "AUC", caption=f"95% KI: {ci_text}" if ci_text else None,
        tooltip="Area Under the Curve - sammanfattar testets förmåga att skilja mellan de med och utan "
        "utfallet, över alla möjliga trösklar samtidigt. 0.5 = slumpnivå, 1.0 = perfekt särskiljning. "
        "≥0.80 anses bra, ≥0.70 acceptabelt. Den bekräftade diagnosen som poängen jämförs mot här "
        "(**referensintervjun**) är **simulerad** för demonstrationssyfte (se "
        "`data/generate_sample_data.py`) - inte ett riktigt kliniskt facit.",
        learning_key="ROC-kurva & AUC",
    )
with kpi_cols[1]:
    kpi_card(
        "🎯", "#D1FAE5", f"{optimal.threshold:g}", "Rekommenderad tröskel", caption=f"Youden's J = {optimal.youdens_j:.2f}",
        tooltip="Den cut-off-poäng som maximerar Youden's J (bästa balans mellan sensitivitet och "
        "specificitet) för just detta test och denna data - inte nödvändigtvis samma tröskel som "
        "testets officiella cut-off.",
        learning_key="Cut score",
    )
with kpi_cols[2]:
    kpi_card(
        "✅", "#EDE9FE", f"{optimal.sensitivity:.2f}", "Sensitivitet vid tröskel",
        tooltip="Andel med utfallet (t.ex. diagnos) som testet korrekt identifierar som positiva vid "
        "denna tröskel. Hög sensitivitet minskar risken att missa någon som behöver hjälp "
        "(färre false negatives).",
        learning_key="Sensitivitet (TPR)",
    )
with kpi_cols[3]:
    kpi_card(
        "🛡️", "#FFEDD5", f"{optimal.specificity:.2f}", "Specificitet vid tröskel",
        tooltip="Andel utan utfallet som testet korrekt identifierar som negativa vid denna tröskel. "
        "Hög specificitet minskar risken för onödig oro eller åtgärd hos friska (färre false positives).",
        learning_key="Specificitet (TNR)",
    )

st.write("")

tab_overview, tab_cutoffs, tab_capacity = st.tabs(["Översikt", "Cut score & tröskelvärden", "Prioritering vid kapacitet"])

with tab_overview:
    lo, hi = q.get_subscale(subscale_id).score_range if subscale_id else (0, 0)
    threshold = st.slider("Välj tröskel", int(lo), int(hi), int(round(optimal.threshold)))
    metrics = se.confusion_at_threshold(dataset, threshold, subscale_id)

    if threshold == round(optimal.threshold):
        st.success(f"✅ Denna tröskel maximerar Youden's J ({optimal.youdens_j:.2f}) - bästa balans mellan sensitivitet och specificitet.")

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
                learning_key="ROC-kurva & AUC",
            )
        operating_point = (1 - metrics.specificity, metrics.sensitivity) if metrics is not None else None
        roc_fig = ve.roc_curve_chart(roc.fpr, roc.tpr, roc.auc, operating_point=operating_point)
        st.plotly_chart(roc_fig, width="stretch")

    with col_cm:
        header_cols = st.columns([5, 1])
        header_cols[0].markdown("**Konfusionsmatris**")
        with header_cols[1]:
            concept_tooltip(
                "Konfusionsmatris",
                "En 2×2-tabell över hur testets prediktioner (positiv/negativ vid vald tröskel) stämmer "
                "mot det faktiska utfallet. Diagonalen (TP, TN) är korrekta klassificeringar; övriga "
                "rutor (FP, FN) är fel av olika slag.",
                learning_key="Konfusionsmatris",
            )
        if metrics is not None:
            st.plotly_chart(ve.confusion_matrix_heatmap(metrics.tp, metrics.fp, metrics.tn, metrics.fn), width="stretch")

    if metrics is not None:
        metric_cols = st.columns([1, 1, 1, 1, 0.3])
        metric_cols[0].metric("Sensitivitet (TPR)", f"{metrics.sensitivity:.2f}")
        metric_cols[1].metric("Specificitet (TNR)", f"{metrics.specificity:.2f}")
        metric_cols[2].metric("PPV (Precision)", f"{metrics.ppv:.2f}")
        metric_cols[3].metric("NPV", f"{metrics.npv:.2f}")
        with metric_cols[4]:
            concept_tooltip(
                "PPV & NPV",
                "PPV (Precision): av alla som testet flaggar som positiva, hur många har verkligen "
                "utfallet? NPV: av alla som flaggas som negativa, hur många är verkligen utan utfallet? "
                "Till skillnad från sensitivitet/specificitet beror PPV/NPV på hur vanligt utfallet är "
                "i populationen (prevalens) - samma test kan ge olika PPV/NPV i olika grupper.",
                learning_key="PPV / NPV",
            )

with tab_cutoffs:
    header_cols = st.columns([5, 1])
    header_cols[0].markdown("**Cut score & tröskelvärden**")
    with header_cols[1]:
        concept_tooltip(
            "Youden's J",
            "J = Sensitivitet + Specificitet - 1. Tröskeln som maximerar J ger den bästa balansen "
            "mellan att korrekt identifiera positiva och negativa fall.",
            learning_key="Youden's J",
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

with tab_capacity:
    header_cols = st.columns([5, 1])
    header_cols[0].markdown("**Om en fast tröskel inte räcker: vem prioriterar du?**")
    with header_cols[1]:
        concept_tooltip(
            "Kapacitetsbegränsad prioritering",
            "En fast tröskel antar att du kan kalla in alla som hamnar över den. I praktiken har du "
            "ofta ett tak - t.ex. ett visst antal uppföljningstider per månad. Rangordna då alla efter "
            "poäng (högst risk först) och gå nedåt listan tills kapaciteten är slut, istället för att "
            "bara fråga ja/nej mot en gräns.",
            learning_key="Kapacitetsbegränsad prioritering",
        )
    st.caption(
        "Rangordnar alla i datasetet efter totalpoäng (högst först) och visar hur stor andel av de "
        "verkliga fallen du fångar in om du bara har resurser att kontakta en viss andel av dem."
    )
    capacity_pct = st.slider("Kapacitet - andel av populationen du kan kontakta", 1, 100, 15, format="%d%%") / 100
    curve = se.capture_curve_for_dataset(dataset, subscale_id)
    if curve is None:
        st.info("Otillräcklig data för en kapacitetskurva.")
    else:
        capture_rate = se.capture_rate_at_capacity(curve, capacity_pct)
        n_contacted = round(capacity_pct * curve.n)
        n_captured = round(capture_rate * curve.n_positive)
        cap_cols = st.columns(3)
        cap_cols[0].metric("Kontaktar", f"{n_contacted} av {curve.n}")
        cap_cols[1].metric("Fångar", f"{n_captured} av {curve.n_positive} sanna fall")
        cap_cols[2].metric("Fångstgrad", f"{capture_rate:.0%}")
        st.plotly_chart(ve.capture_curve_chart(curve.pct_contacted, curve.pct_captured, capacity_pct), width="stretch")
        st.caption(
            "Den streckade linjen är vad ett slumpmässigt urval av samma storlek skulle fånga i "
            "genomsnitt - avståndet upp till den blå kurvan är vad rangordningen efter poäng "
            "faktiskt ger dig. Jämför med samma flik på Machine Learning-sidan, som rangordnar efter "
            "en tränad modells sannolikhet istället för bara totalpoängen."
        )

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
        st.switch_page("pages/8_Norm_Explorer.py")
with col_next:
    if st.button("Fortsätt till Measurement Error →", type="primary", width="stretch"):
        st.switch_page("pages/9_Measurement_Error.py")
