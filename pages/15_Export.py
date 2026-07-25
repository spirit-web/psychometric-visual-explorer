"""Export page. Rendering only - all report/figure/data generation lives in
core/export_engine.py (SRS §27 Export System)."""

from datetime import datetime

import streamlit as st

from components.kpi_card import kpi_card
from core import export_engine as ee
from core import stats_engine as se
from utils.session import require_dataset

dataset = require_dataset()
q = dataset.questionnaire

st.title("Export")
st.caption("Exportera analyser, rapporter och data")

has_outcome = se.has_outcome(dataset)
available_reports = 2 + (1 if has_outcome else 0)

# --- KPI row -------------------------------------------------
kpi_cols = st.columns(4)
with kpi_cols[0]:
    kpi_card("📄", "#DBEAFE", str(available_reports), "Tillgängliga rapporter", caption="Färdiga att exportera")
with kpi_cols[1]:
    kpi_card("🗂️", "#D1FAE5", q.test_name, "Aktivt dataset", caption=f"N={dataset.n}")
with kpi_cols[2]:
    kpi_card("📋", "#EDE9FE", str(dataset.n_items), "Items poängsatta")
with kpi_cols[3]:
    last_export = st.session_state.get("export_last_at")
    kpi_card("🕓", "#FFEDD5", last_export.strftime("%H:%M") if last_export else "Ingen ännu", "Senast exporterad (denna session)")

st.write("")


def _mark_exported() -> None:
    st.session_state["export_last_at"] = datetime.now()


def _offer_download(session_key: str, label: str, file_name: str, mime: str, builder) -> None:
    """Two-phase pattern: a 'Förbered'/generate button computes the bytes
    once and stores them in session state; a second, native download_button
    then performs the actual browser download."""
    if st.button(label, key=f"prepare_{session_key}", width="stretch"):
        with st.spinner("Förbereder export..."):
            data = builder()
        if data is None:
            st.warning("⚠️ Denna export kräver data som saknas i det aktiva datasetet.")
        else:
            st.session_state[f"export_bytes_{session_key}"] = data
            _mark_exported()
            st.rerun()
    if st.session_state.get(f"export_bytes_{session_key}") is not None:
        st.download_button(
            f"⬇️ Ladda ner",
            data=st.session_state[f"export_bytes_{session_key}"],
            file_name=file_name,
            mime=mime,
            width="stretch",
            key=f"download_{session_key}",
        )


# --- Content selection / format -------------------------------------------------
col_content, col_settings = st.columns([3, 2])

with col_content:
    st.markdown("**Välj exportinnehåll**")
    include_full = st.checkbox("Komplett analysrapport (PDF) – dataset, QC, reliabilitet, faktorer, validitet, normer, ML", value=True)
    include_summary = st.checkbox("Psykometrisk sammanfattning (PDF) – reliabilitet, validitet, rättvisa", value=False)
    include_decision = st.checkbox(
        "Beslutsstödsrapport (PDF)",
        value=False,
        disabled=not has_outcome,
        help="Kräver en utfallsvariabel i datasetet." if not has_outcome else None,
    )
    include_figures = st.checkbox("Datavisualiseringar (PNG, ZIP) – alla grafer från analyserna", value=False)
    include_raw = st.checkbox("Rådata (anonymiserad, CSV) – poängsatt data utan respondent-ID", value=False)

    n_selected = sum([include_full, include_summary, include_decision, include_figures, include_raw])
    st.caption(f"{n_selected} av 5 valda")

with col_settings:
    st.markdown("**Om exporten**")
    st.info(
        "ℹ️ Väljer du fler än en post byggs en kombinerad ZIP-fil. Rådata är alltid anonymiserad "
        "(inget respondent-ID) innan export."
    )
    if not has_outcome:
        st.caption("Beslutsstödsrapport kräver en utfallsvariabel (`outcome_positive`) i datasetet.")

st.write("")

if st.button("📦 Förbered export", type="primary", width="stretch", disabled=n_selected == 0):
    with st.spinner("Genererar exportfil..."):
        if n_selected == 1 and include_full:
            data, file_name, mime = ee.full_analysis_report_pdf(dataset), f"analysrapport_{q.plugin_id}.pdf", "application/pdf"
        elif n_selected == 1 and include_summary:
            data, file_name, mime = ee.psychometric_summary_pdf(dataset), f"sammanfattning_{q.plugin_id}.pdf", "application/pdf"
        elif n_selected == 1 and include_decision:
            data, file_name, mime = ee.decision_support_report_pdf(dataset), f"beslutsstod_{q.plugin_id}.pdf", "application/pdf"
        elif n_selected == 1 and include_figures:
            data, file_name, mime = ee.figures_to_zip_bytes(ee.key_figures(dataset)), f"figurer_{q.plugin_id}.zip", "application/zip"
        elif n_selected == 1 and include_raw:
            data = ee.dataframe_to_csv_bytes(ee.anonymized_scored_dataset(dataset))
            file_name, mime = f"data_anonymiserad_{q.plugin_id}.csv", "text/csv"
        else:
            data = ee.build_export_zip(
                dataset,
                include_full_report=include_full,
                include_summary_report=include_summary,
                include_decision_report=include_decision,
                include_figures=include_figures,
                include_raw_data=include_raw,
            )
            file_name, mime = f"pve_export_{q.plugin_id}.zip", "application/zip"
    st.session_state["export_bundle_bytes"] = data
    st.session_state["export_bundle_name"] = file_name
    st.session_state["export_bundle_mime"] = mime
    _mark_exported()
    st.rerun()

if st.session_state.get("export_bundle_bytes") is not None:
    st.download_button(
        f"⬇️ Ladda ner {st.session_state['export_bundle_name']}",
        data=st.session_state["export_bundle_bytes"],
        file_name=st.session_state["export_bundle_name"],
        mime=st.session_state["export_bundle_mime"],
        type="primary",
        width="stretch",
    )

st.write("")
st.markdown("**Snabbexport**")
st.caption("Exportera standardrapporter med ett klick.")
quick_cols = st.columns(4)
with quick_cols[0]:
    _offer_download(
        "quick_decision",
        "Beslutsstödsrapport (PDF)",
        f"beslutsstod_{q.plugin_id}.pdf",
        "application/pdf",
        lambda: ee.decision_support_report_pdf(dataset),
    )
with quick_cols[1]:
    _offer_download(
        "quick_summary",
        "Psykometrisk sammanfattning (PDF)",
        f"sammanfattning_{q.plugin_id}.pdf",
        "application/pdf",
        lambda: ee.psychometric_summary_pdf(dataset),
    )
with quick_cols[2]:
    _offer_download(
        "quick_figures",
        "Alla visualiseringar (ZIP)",
        f"figurer_{q.plugin_id}.zip",
        "application/zip",
        lambda: ee.figures_to_zip_bytes(ee.key_figures(dataset)),
    )
with quick_cols[3]:
    _offer_download(
        "quick_raw",
        "Anonymiserad data (CSV)",
        f"data_anonymiserad_{q.plugin_id}.csv",
        "text/csv",
        lambda: ee.dataframe_to_csv_bytes(ee.anonymized_scored_dataset(dataset)),
    )

st.write("")
st.info("ℹ️ Alla exporterade filer är anonyma - inget respondent-ID ingår i data- eller rapportexporter.")

st.write("")
col_back, _ = st.columns([1, 4])
with col_back:
    if st.button("Tillbaka", width="stretch"):
        st.switch_page("pages/14_Learning_Mode.py")
