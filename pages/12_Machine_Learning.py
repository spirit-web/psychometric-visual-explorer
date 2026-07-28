"""Machine Learning page. Rendering only - all statistics come from core/.

Implements the full pipeline required by the course assignment: data prep,
EDA (linked to other pages), unsupervised learning (KMeans/PCA), supervised
learning with non-linear models plus a deep learning option, a full
evaluation suite, and SHAP-based feature importance.
"""

import pandas as pd
import streamlit as st

from components.concept_tooltip import concept_tooltip
from components.export_section import render_export_section
from components.kpi_card import kpi_card
from core import ml_engine as ml
from core import stats_engine as se
from core import viz_engine as ve
from utils.session import require_dataset

dataset = require_dataset()
q = dataset.questionnaire

st.title("Machine Learning")
st.caption(f"Prediktiva insikter och mönsterupptäckt för {q.test_name}")

if not se.has_outcome(dataset):
    st.info(
        "Ingen målvariabel (`outcome_positive`) hittades i datasetet. Machine Learning-modulen "
        "kräver en binär målvariabel att träna klassificerare mot. Exempeldatan innehåller en "
        "simulerad sådan variabel."
    )
    st.stop()

st.info(
    "ℹ️ Målvariabeln är **simulerad** för demonstrationssyfte (se `data/generate_sample_data.py`). "
    "Detta är ett pedagogiskt exempel, inte ett kliniskt beslutsstödsverktyg. **Värdet för en "
    "psykolog:** se vilka svar som bäst förutsäger ett utfall (Feature Importance) och testa "
    "själv hur en förändrad profil skulle predikteras (Snabbprediktion). **Värdet för en "
    "forskare/student:** en fullständig ML-pipeline - dataförberedelse, unsupervised (PCA/kluster), "
    "flera icke-linjära modeller inklusive ett neuralt nätverk, och en fullständig utvärdering."
)

ml_data = ml.prepare_ml_data(dataset)
if ml_data is None:
    st.info("Otillräcklig data för maskininlärning (för få rader eller endast en klass i målvariabeln).")
    st.stop()


@st.cache_data(show_spinner="Tränar och utvärderar modeller...")
def _cached_run_all_models(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    data = ml.MLDataset(X=X, y=y, feature_names=list(X.columns), n_samples=len(X), report=None)
    return ml.run_all_models(data, seed=seed)


results = _cached_run_all_models(ml_data.X, ml_data.y)
best_name = ml.best_model_name(results)
best = results[best_name]

_ITEM_LABELS = {item.id: f"{item.id} – {item.text}" for item in q.items}


def _with_question_text(series: pd.Series) -> pd.Series:
    """Relabels item-column entries (e.g. GAD7_2) with their question text;
    non-item columns (age, gender_..., etc.) are left unchanged."""
    return series.rename(index=lambda key: _ITEM_LABELS.get(key, key))

# --- KPI row -------------------------------------------------
kpi_cols = st.columns(4)
with kpi_cols[0]:
    kpi_card(
        "🧠", "#EDE9FE", best_name, "Bästa modell", caption="Högst AUC",
        tooltip="Vilken av de tränade modellerna (Logistic Regression, Random Forest, XGBoost, Neural "
        "Network) som presterade bäst på testdatan, mätt med AUC. Se fliken Modeller & Utvärdering "
        "för att jämföra alla modeller.",
    )
with kpi_cols[1]:
    kpi_card(
        "📈", "#D1FAE5", f"{best.roc_auc:.2f}", "Model Performance (AUC)",
        tooltip="Area Under the Curve för den bästa modellen - hur väl den skiljer mellan de med och "
        "utan utfallet. 0.5 = slumpnivå, 1.0 = perfekt särskiljning.",
        learning_key="ROC-kurva & AUC",
    )
with kpi_cols[2]:
    kpi_card(
        "✅", "#DBEAFE", f"{best.cv_auc_mean:.2f} ± {best.cv_auc_std:.2f}", "Cross-Validation (5-fold)", caption="Medel AUC ± SD",
        tooltip="Modellen tränas och utvärderas om 5 gånger på olika delar av datan, för att ge en mer "
        "pålitlig skattning av hur den presterar på ny, osedd data - inte bara ett enda test-set. "
        "Liten SD betyder att prestandan är stabil mellan delarna.",
    )
with kpi_cols[3]:
    kpi_card(
        "🎯", "#FFEDD5", f"N={ml_data.n_samples}", "Prediktionsmål", caption="Binär klassificering",
        tooltip="Antal observationer som användes för att träna och utvärdera modellerna, efter att "
        "rader med saknad målvariabel tagits bort (se fliken Dataförberedelse).",
    )

st.write("")

tab_overview, tab_prep, tab_unsup, tab_models, tab_importance, tab_predict = st.tabs(
    ["Översikt", "Dataförberedelse", "Unsupervised", "Modeller & Utvärdering", "Feature Importance", "Snabbprediktion"]
)

with tab_overview:
    col_roc, col_metrics = st.columns(2)
    with col_roc:
        st.markdown(f"**Model Performance ({best_name})**")
        ml_roc_fig = ve.roc_curve_chart(best.fpr, best.tpr, best.roc_auc)
        st.plotly_chart(ml_roc_fig, width="stretch", key="ml_overview_roc")
    with col_metrics:
        st.markdown("**Prestandamått**")
        tn, fp, fn, tp = best.confusion.ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        metric_rows = [
            {"Mått": "AUC (ROC)", "Värde": f"{best.roc_auc:.2f}"},
            {"Mått": "Accuracy", "Värde": f"{best.accuracy:.2f}"},
            {"Mått": "Precision", "Värde": f"{best.precision:.2f}"},
            {"Mått": "Recall (Sensitivitet)", "Värde": f"{best.recall:.2f}"},
            {"Mått": "Specificitet", "Värde": f"{specificity:.2f}"},
            {"Mått": "F1-score", "Värde": f"{best.f1:.2f}"},
        ]
        st.dataframe(pd.DataFrame(metric_rows), width="stretch", hide_index=True)

    st.write("")
    header_cols = st.columns([5, 1])
    header_cols[0].markdown("**Top 10 viktigaste prediktorer**")
    with header_cols[1]:
        concept_tooltip(
            "Feature importance",
            "Visar vilka variabler som bidrar mest till modellens prediktioner. Högre vikt "
            "indikerar starkare bidrag - se fliken Feature Importance för SHAP-baserad analys.",
        )
    if best.feature_importance is not None:
        top10 = _with_question_text(best.feature_importance.head(10))
        st.plotly_chart(ve.horizontal_bar_chart(top10, "Relativ betydelse"), width="stretch", key="ml_overview_importance")
    else:
        st.info(f"{best_name} har ingen inbyggd feature importance. Se fliken Feature Importance för SHAP-analys.")

    st.write("")
    if best.roc_auc >= 0.8:
        st.success(f"✅ {best_name} visar god förmåga att skilja mellan grupperna (AUC = {best.roc_auc:.2f}).")
    elif best.roc_auc >= 0.7:
        st.warning(f"⚠️ {best_name} visar acceptabel men förbättringsbar förmåga att skilja mellan grupperna (AUC = {best.roc_auc:.2f}).")
    else:
        st.error(f"🛑 {best_name} visar svag förmåga att skilja mellan grupperna (AUC = {best.roc_auc:.2f}).")
    st.caption("Modellen är ett beslutsstöd och ersätter inte klinisk bedömning. EDA och deskriptiv statistik finns i Dataset Overview och Psychometric QC.")

with tab_prep:
    header_cols = st.columns([5, 1])
    header_cols[0].markdown("**Dataförberedelse**")
    with header_cols[1]:
        concept_tooltip(
            "Dataförberedelse",
            "Innan en modell kan tränas måste rådatan städas: saknade svar fylls i (imputeras), "
            "textkategorier (t.ex. kön, utbildning) omvandlas till siffror (one-hot-kodning), och rader "
            "utan målvariabel tas bort eftersom modellen inte kan lära sig av dem.",
        )
    report = ml_data.report
    info_cols = st.columns(4)
    info_cols[0].metric("Rader totalt", report.n_rows_total)
    info_cols[1].metric("Rader använda", report.n_rows_used)
    info_cols[2].metric("Rader borttagna", report.n_rows_dropped, help="Saknade målvariabeln")
    info_cols[3].metric("Antal features", len(report.feature_columns))

    st.write("")
    st.markdown(
        f"- **Saknade itemvärden imputerade:** {report.item_missing_values_imputed} (medianimputation per item)\n"
        f"- **Saknade demografivärden imputerade:** {report.demographic_missing_values_imputed}\n"
        f"- **Kategoriska kolumner one-hot-kodade:** {', '.join(report.categorical_columns_encoded) or 'inga'}\n"
    )

    st.write("")
    st.markdown("**Förhandsgranskning av modellklar data**")
    preview = ml_data.X.copy()
    preview["target"] = ml_data.y
    st.dataframe(preview.head(10), width="stretch")
    st.caption("Fullständig utforskande dataanalys (EDA) finns i Dataset Overview och Psychometric QC.")

with tab_unsup:
    header_cols = st.columns([5, 1])
    header_cols[0].markdown("**Unsupervised learning: KMeans-klustring**")
    with header_cols[1]:
        concept_tooltip(
            "Silhouette score",
            "Mäter hur väl-separerade och sammanhållna klustren är, från -1 (dåligt) till +1 "
            "(utmärkt). Värden nära 0 tyder på överlappande kluster utan tydlig struktur - prova ett "
            "annat antal kluster med reglaget.",
        )
    n_clusters = st.slider("Antal kluster", 2, 6, 3)
    clusters = ml.run_kmeans(ml_data, n_clusters=n_clusters)

    metric_cols = st.columns(2)
    metric_cols[0].metric("Silhouette score", f"{clusters.silhouette:.2f}" if clusters.silhouette is not None else "–")
    metric_cols[1].metric("Klusterstorlekar", ", ".join(f"K{k}: {v}" for k, v in clusters.cluster_sizes.items()))

    st.write("")
    header_cols = st.columns([5, 1])
    header_cols[0].markdown("**PCA-projektion (2D)**")
    with header_cols[1]:
        concept_tooltip(
            "PCA & klustring",
            "PCA (Principal Component Analysis) reducerar datan till två dimensioner för "
            "visualisering. Kluster som bildar tydliga grupper i denna projektion tyder på "
            "verklig struktur i datan, inte bara i de ursprungliga variablerna.",
        )
    pca = ml.run_pca(ml_data, n_components=2)
    col_cluster, col_outcome = st.columns(2)
    with col_cluster:
        st.markdown("*Färgad efter kluster*")
        st.plotly_chart(
            ve.scatter_2d_chart(pca.components[:, 0], pca.components[:, 1], [f"Kluster {c}" for c in clusters.labels], "PC1", "PC2"),
            width="stretch",
            key="ml_pca_cluster",
        )
    with col_outcome:
        st.markdown("*Färgad efter faktiskt utfall*")
        outcome_labels = ml_data.y.map({0: "Negativ", 1: "Positiv"})
        st.plotly_chart(
            ve.scatter_2d_chart(pca.components[:, 0], pca.components[:, 1], outcome_labels, "PC1", "PC2"),
            width="stretch",
            key="ml_pca_outcome",
        )
    st.caption(
        f"Förklarad varians: PC1 = {pca.explained_variance_ratio[0] * 100:.1f}%, "
        f"PC2 = {pca.explained_variance_ratio[1] * 100:.1f}%."
    )

with tab_models:
    header_cols = st.columns([5, 1])
    header_cols[0].markdown("**Modelljämförelse**")
    with header_cols[1]:
        concept_tooltip(
            "Utvärderingsmått",
            "Accuracy: andel korrekta prediktioner totalt. Precision: av de förutspådda positiva, hur "
            "många var rätt? Recall (sensitivitet): av de faktiskt positiva, hur många hittade "
            "modellen? F1: balans mellan precision och recall. ROC-AUC: förmåga att skilja klasserna "
            "över alla trösklar. CV AUC: samma mått men medelvärde över 5 korsvaliderings-omgångar, "
            "för en mer pålitlig skattning.",
        )
    comparison_df = pd.DataFrame(
        [
            {
                "Modell": name,
                "Accuracy": round(r.accuracy, 3),
                "Precision": round(r.precision, 3),
                "Recall": round(r.recall, 3),
                "F1": round(r.f1, 3),
                "ROC-AUC": round(r.roc_auc, 3),
                "CV AUC (medel ± SD)": f"{r.cv_auc_mean:.3f} ± {r.cv_auc_std:.3f}",
            }
            for name, r in results.items()
        ]
    )
    st.dataframe(comparison_df, width="stretch", hide_index=True)
    st.caption(
        "Random Forest och XGBoost är icke-linjära modeller; Neural Network (MLP) är ett litet "
        "neuralt nätverk (PyTorch) - alla utvärderas med samma metodik. Logistic Regression "
        "ingår som en tolkningsbar baslinje, inte som den enda modellen."
    )

    st.write("")
    selected_name = st.selectbox("Granska modell i detalj", list(results.keys()), index=list(results.keys()).index(best_name))
    selected = results[selected_name]

    col_cm, col_curves = st.columns(2)
    with col_cm:
        header_cols = st.columns([5, 1])
        header_cols[0].markdown("**Confusion Matrix**")
        with header_cols[1]:
            concept_tooltip(
                "Confusion Matrix",
                "En 2×2-tabell över hur modellens prediktioner (positiv/negativ) stämmer mot det "
                "faktiska utfallet. Diagonalen är korrekta klassificeringar; övriga rutor är fel.",
                learning_key="Konfusionsmatris",
            )
        tn, fp, fn, tp = selected.confusion.ravel()
        st.plotly_chart(ve.confusion_matrix_heatmap(int(tp), int(fp), int(tn), int(fn)), width="stretch", key="ml_model_cm")
        st.markdown("**Classification Report**")
        st.caption("Samma mått (precision, recall, F1) uppdelat per klass (Negativ/Positiv), i scikit-learns standardformat.")
        st.code(selected.classification_report)

    with col_curves:
        st.markdown("**ROC-kurva**")
        st.plotly_chart(ve.roc_curve_chart(selected.fpr, selected.tpr, selected.roc_auc), width="stretch", key="ml_model_roc")
        st.markdown("**Precision-Recall-kurva**")
        st.plotly_chart(ve.precision_recall_chart(selected.precision_curve, selected.recall_curve), width="stretch", key="ml_model_pr")

with tab_importance:
    st.markdown("**Feature Importance (SHAP)**")
    tree_models = {name: r for name, r in results.items() if name in ("Random Forest", "XGBoost")}
    if not tree_models:
        st.info("SHAP-analys stöds för trädbaserade modeller (Random Forest, XGBoost).")
    else:
        shap_model_name = st.selectbox("Modell", list(tree_models.keys()))
        shap_result = tree_models[shap_model_name]
        with st.spinner("Beräknar SHAP-värden..."):
            shap_importance = ml.shap_feature_importance(shap_result.model, ml_data.X)
        if shap_importance is not None:
            shap_top15 = _with_question_text(shap_importance.head(15))
            st.plotly_chart(ve.horizontal_bar_chart(shap_top15, "Medel absolut SHAP-värde"), width="stretch", key="ml_shap_chart")
            st.caption(
                "SHAP-värden visar varje variabels genomsnittliga bidrag (absolutbelopp) till "
                "modellens prediktioner över samtliga observationer."
            )
        else:
            st.info("SHAP-värden kunde inte beräknas för denna modell.")

with tab_predict:
    st.markdown("**Snabbprediktion**")
    st.write(
        "Vilka svar driver en prediktion mest? Ändra en klients svar med reglagen nedan och se "
        "hur den predikterade sannolikheten ändras direkt - eller välj en befintlig person ur "
        "datasetet för att jämföra mot det faktiska (simulerade) utfallet."
    )
    predict_model_name = st.selectbox("Modell att använda", list(results.keys()), index=list(results.keys()).index(best_name), key="predict_model")
    predict_model = results[predict_model_name].model

    predict_mode = st.radio(
        "Läge",
        ["Interaktiv (ändra svar med reglage)", "Befintlig person ur datasetet"],
        horizontal=True,
        key="predict_mode",
    )

    if predict_mode == "Befintlig person ur datasetet":
        person_ids = [pid for pid in se.person_ids(dataset) if ml.feature_vector_for_person(ml_data, dataset, pid) is not None]
        if not person_ids:
            st.info("Ingen person i datasetet har fullständig data för prediktion.")
        else:
            person_id = st.selectbox("Välj person", person_ids, format_func=lambda pid: f"Person {pid}")
            feature_vector = ml.feature_vector_for_person(ml_data, dataset, person_id)
            actual = ml.actual_outcome_for_person(dataset, person_id)

            proba = ml.predict_proba_for_vector(predict_model, feature_vector)
            predicted_class = "Positiv" if proba >= 0.5 else "Negativ"
            actual_class = {1: "Positiv", 0: "Negativ", None: "Okänd"}[actual]

            col_input, col_result = st.columns([2, 1])
            with col_input:
                st.markdown("**Indata (första 10 features)**")
                st.dataframe(feature_vector.head(10).to_frame(name="Värde"), width="stretch")
            with col_result:
                st.metric("Predikterad sannolikhet", f"{proba:.2f}")
                st.metric("Predikterad klass", predicted_class)
                st.metric("Faktiskt utfall (simulerat)", actual_class)
                if actual is not None:
                    if predicted_class == actual_class:
                        st.success("✅ Modellen predikterade rätt klass för denna person.")
                    else:
                        st.warning("⚠️ Modellen predikterade fel klass för denna person.")
    else:
        baseline = ml.baseline_feature_vector(ml_data)
        feature_vector = baseline.copy()
        scale = q.response_scale
        item_ids = [item_id for item_id in q.item_ids if item_id in feature_vector.index]

        col_sliders, col_result = st.columns([2, 1])
        with col_sliders:
            st.caption("Övriga features (t.ex. demografi) hålls på ett typiskt värde (medianen i datasetet).")
            slider_cols = st.columns(2)
            for i, item_id in enumerate(item_ids):
                item = next((it for it in q.items if it.id == item_id), None)
                label = f"{item_id} – {item.text}" if item else item_id
                with slider_cols[i % 2]:
                    feature_vector[item_id] = st.slider(
                        label, int(scale.min), int(scale.max), int(round(baseline[item_id])), key=f"predict_slider_{item_id}"
                    )

        proba = ml.predict_proba_for_vector(predict_model, feature_vector)
        predicted_class = "Positiv" if proba >= 0.5 else "Negativ"
        with col_result:
            st.metric("Predikterad sannolikhet", f"{proba:.2f}")
            st.metric("Predikterad klass", predicted_class)
            st.caption("Baserat på ett typiskt profil förutom de items du justerat ovan - ett pedagogiskt exempel, inte en riktig klientbedömning.")

st.write("")
render_export_section(
    dataset,
    "machine_learning",
    table=comparison_df,
    table_label="Modelljämförelse",
    figures={"roc_kurva_basta_modell": ml_roc_fig},
)

st.write("")
col_back, _, col_next = st.columns([1, 3, 1])
with col_back:
    if st.button("Tillbaka", width="stretch"):
        st.switch_page("pages/7_Validity_Dashboard.py")
with col_next:
    if st.button("Fortsätt till Test Builder →", type="primary", width="stretch"):
        st.switch_page("pages/13_Test_Builder.py")
