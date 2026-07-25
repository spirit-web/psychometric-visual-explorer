"""Builds PDF reports, PNG figure bundles, and CSV/anonymized data exports
from a Dataset. One shared engine used by every page's export button and by
the central Export page - no per-page export code, no Streamlit imports.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from loguru import logger
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from core import stats_engine as se

PVE_BLUE = colors.HexColor("#2563EB")
PVE_GRAY = colors.HexColor("#F3F4F6")
PVE_BORDER = colors.HexColor("#D1D5DB")

# --- CSV / data export -------------------------------------------------


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def anonymized_scored_dataset(dataset) -> pd.DataFrame:
    """Scored dataset with direct identifiers removed - respondent_id is
    replaced by a sequential anonymous row number."""
    df = dataset.scored.copy()
    df = df.drop(columns=[c for c in ("respondent_id",) if c in df.columns])
    df.insert(0, "anon_id", range(1, len(df) + 1))
    return df


# --- Figure export -------------------------------------------------


def figure_to_png_bytes(fig: go.Figure, width: int = 1000, height: int = 600, scale: int = 2) -> bytes:
    return fig.to_image(format="png", width=width, height=height, scale=scale)


def figures_to_zip_bytes(named_figures: dict[str, go.Figure]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, fig in named_figures.items():
            try:
                zf.writestr(f"{name}.png", figure_to_png_bytes(fig))
            except Exception as exc:  # pragma: no cover - defensive, kaleido/env issues
                logger.warning(f"Skipped figure '{name}' in export: {exc}")
    buffer.seek(0)
    return buffer.getvalue()


def key_figures(dataset) -> dict[str, go.Figure]:
    """A representative set of figures across the whole analysis, for the
    'Datavisualiseringar' export option and the ZIP bundle."""
    from core import viz_engine as ve

    figures: dict[str, go.Figure] = {}
    try:
        figures["svarsfordelning"] = ve.response_distribution_chart(se.response_distribution(dataset))
    except Exception as exc:
        logger.warning(f"key_figures response_distribution failed: {exc}")
    try:
        missing = se.missing_by_item(dataset)
        if not missing.empty:
            figures["bortfall_per_item"] = ve.missing_by_item_chart(missing)
    except Exception as exc:
        logger.warning(f"key_figures missing_by_item failed: {exc}")
    try:
        corr = se.item_correlation_matrix(dataset)
        if not corr.empty:
            figures["item_korrelationer"] = ve.correlation_heatmap(corr)
    except Exception as exc:
        logger.warning(f"key_figures correlation failed: {exc}")
    try:
        pa = se.parallel_analysis(dataset)
        if len(pa.eigenvalues):
            figures["scree_plot"] = ve.scree_plot_chart(pa.eigenvalues, pa.simulated_eigenvalues)
    except Exception as exc:
        logger.warning(f"key_figures scree failed: {exc}")
    if se.has_outcome(dataset):
        try:
            subscale_id = dataset.questionnaire.subscales[0].id if dataset.questionnaire.subscales else None
            roc = se.roc_analysis(dataset, subscale_id)
            if roc is not None:
                figures["roc_kurva"] = ve.roc_curve_chart(roc.fpr, roc.tpr, roc.auc)
        except Exception as exc:
            logger.warning(f"key_figures roc failed: {exc}")
    return figures


# --- PDF report building -------------------------------------------------


@dataclass
class ReportSection:
    title: str
    paragraphs: list[str] = field(default_factory=list)
    table: pd.DataFrame | None = None
    bullet_points: list[str] = field(default_factory=list)


def _dataframe_to_table(df: pd.DataFrame, max_rows: int = 25) -> Table:
    data = [list(df.columns)] + df.head(max_rows).astype(str).values.tolist()
    table = Table(data, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PVE_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("GRID", (0, 0), (-1, -1), 0.4, PVE_BORDER),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PVE_GRAY]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def build_pdf_report(title: str, subtitle: str, sections: list[ReportSection]) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm, leftMargin=2 * cm, rightMargin=2 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("PVETitle", parent=styles["Title"], textColor=PVE_BLUE)
    heading_style = ParagraphStyle("PVEHeading", parent=styles["Heading2"], textColor=PVE_BLUE, spaceBefore=14)
    body_style = styles["BodyText"]

    story = [
        Paragraph(title, title_style),
        Paragraph(subtitle, styles["Normal"]),
        Paragraph(f"Genererad: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]),
        Spacer(1, 0.6 * cm),
    ]

    for section in sections:
        story.append(Paragraph(section.title, heading_style))
        for para in section.paragraphs:
            story.append(Paragraph(para, body_style))
        for bullet in section.bullet_points:
            story.append(Paragraph(f"• {bullet}", body_style))
        if section.table is not None and not section.table.empty:
            story.append(Spacer(1, 0.2 * cm))
            story.append(_dataframe_to_table(section.table))
        story.append(Spacer(1, 0.3 * cm))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# --- Report content builders (generic across GAD-7 / PHQ-9 / Big Five) ---


def dataset_overview_section(dataset) -> ReportSection:
    q = dataset.questionnaire
    return ReportSection(
        title="Datasetöversikt",
        paragraphs=[
            f"Test: {q.full_name} ({q.test_name}).",
            f"Antal deltagare: {dataset.n}. Antal items: {dataset.n_items}. Antal delskalor: {dataset.n_subscales}.",
            f"Totalt bortfall: {dataset.missing_pct:.1f}%.",
        ],
    )


def qc_section(dataset) -> ReportSection:
    summary = se.quality_summary(dataset)
    rows = [{"Kontroll": c.name, "Status": c.status, "Kommentar": c.comment} for c in summary.checks]
    return ReportSection(title="Psykometrisk kvalitetskontroll", paragraphs=[summary.message], table=pd.DataFrame(rows))


def reliability_section(dataset) -> ReportSection:
    rows = []
    for r in se.reliability_snapshot_all_subscales(dataset):
        omega = se.mcdonald_omega(dataset, r.subscale_id)
        rows.append(
            {
                "Delskala": r.subscale_name,
                "Items": r.n_items,
                "N": r.n,
                "Alpha": round(r.alpha, 3) if r.alpha is not None else None,
                "Omega": round(omega, 3) if omega is not None else None,
            }
        )
    overall = se.reliability_snapshot(dataset)
    paragraphs = (
        [f"Cronbach's alpha (hela testet): {overall.alpha:.2f} ({se.alpha_interpretation(overall.alpha)})."]
        if overall.alpha is not None
        else ["Otillräcklig data för reliabilitetsberäkning."]
    )
    return ReportSection(title="Reliabilitet", paragraphs=paragraphs, table=pd.DataFrame(rows))


def factor_section(dataset) -> ReportSection:
    q = dataset.questionnaire
    n_factors = max(1, len(q.subscales))
    efa = se.efa_fit(dataset, n_factors)
    if efa is None:
        return ReportSection(title="Faktorstruktur (EFA)", paragraphs=["Otillräcklig data för faktoranalys."])
    paragraphs = [f"Antal extraherade faktorer: {efa.n_factors}."]
    if efa.fit.rmsea is not None:
        cfi_text = f"{efa.fit.cfi:.2f}" if efa.fit.cfi is not None else "–"
        paragraphs.append(f"Modellanpassning: RMSEA = {efa.fit.rmsea:.3f}, CFI = {cfi_text}.")
    table = efa.loadings.round(2).reset_index().rename(columns={"index": "Item"})
    return ReportSection(title="Faktorstruktur (EFA)", paragraphs=paragraphs, table=table)


def validity_section(dataset) -> ReportSection:
    sources = se.validity_overview(dataset, "none", "none")
    rows = [{"Evidenskälla": s.label, "Status": s.status, "Sammanfattning": s.summary} for s in sources]
    return ReportSection(title="Validitet (fem evidenskällor)", table=pd.DataFrame(rows))


def norms_section(dataset) -> ReportSection:
    q = dataset.questionnaire
    rows = []
    for sub in q.subscales:
        stats = se.norm_stats(dataset, sub.id)
        rows.append(
            {
                "Delskala": sub.name,
                "N": stats.n,
                "Mean": round(stats.mean, 2) if stats.mean is not None else None,
                "SD": round(stats.sd, 2) if stats.sd is not None else None,
            }
        )
    return ReportSection(
        title="Normer",
        paragraphs=["Sample-baserade normer beräknade från detta dataset - inte en extern, standardiserad populationsnorm."],
        table=pd.DataFrame(rows),
    )


def measurement_error_section(dataset) -> ReportSection:
    q = dataset.questionnaire
    rows = []
    for sub in q.subscales:
        me = se.measurement_error(dataset, sub.id)
        rows.append(
            {
                "Delskala": sub.name,
                "SEM": round(me.sem, 2) if me.sem is not None else None,
                "Alpha": round(me.alpha, 2) if me.alpha is not None else None,
            }
        )
    return ReportSection(title="Mätfel (SEM)", table=pd.DataFrame(rows))


def decision_support_section(dataset) -> ReportSection | None:
    if not se.has_outcome(dataset):
        return None
    q = dataset.questionnaire
    subscale_id = q.subscales[0].id if q.subscales else None
    roc = se.roc_analysis(dataset, subscale_id)
    optimal = se.youdens_optimal_threshold(dataset, subscale_id)
    if roc is None or optimal is None:
        return None
    return ReportSection(
        title="Beslutsstöd (ROC-analys)",
        paragraphs=[
            f"AUC = {roc.auc:.2f}.",
            f"Rekommenderad tröskel: {optimal.threshold:g} (sensitivitet {optimal.sensitivity:.2f}, "
            f"specificitet {optimal.specificity:.2f}, Youden's J = {optimal.youdens_j:.2f}).",
            "Utfallsvariabeln är simulerad för demonstrationssyfte, inte ett kliniskt facit.",
        ],
    )


def fairness_section(dataset) -> ReportSection | None:
    q = dataset.questionnaire
    subscale_id = q.subscales[0].id if q.subscales else None
    results = se.all_group_comparisons(dataset, subscale_id)
    if not results:
        return None
    summary = se.fairness_summary(results)
    rows = [
        {
            "Grupp": f"{r.dimension}: {r.comparison_group} vs {r.reference_group}",
            "Cohen's d": round(r.cohens_d, 3),
            "Tolkning": r.interpretation,
        }
        for r in results
    ]
    fairness_text = f"{summary.mean_fairness_index:.2f}" if summary.mean_fairness_index is not None else "–"
    return ReportSection(
        title="Rättvisa (Fairness)",
        paragraphs=[f"Systematisk bias: {summary.bias_level}. Genomsnittligt rättviseindex: {fairness_text}."],
        table=pd.DataFrame(rows),
    )


def ml_section(dataset) -> ReportSection | None:
    from core import ml_engine as ml

    if not se.has_outcome(dataset):
        return None
    ml_data = ml.prepare_ml_data(dataset)
    if ml_data is None:
        return None
    results = ml.run_all_models(ml_data)
    best_name = ml.best_model_name(results)
    if best_name is None:
        return None
    best = results[best_name]
    rows = [
        {"Modell": name, "AUC": round(r.roc_auc, 3), "Accuracy": round(r.accuracy, 3), "F1": round(r.f1, 3)}
        for name, r in results.items()
    ]
    return ReportSection(
        title="Maskininlärning",
        paragraphs=[
            f"Bästa modell: {best_name} (AUC = {best.roc_auc:.2f}).",
            "Målvariabeln är simulerad för demonstrationssyfte - resultaten är pedagogiska, inte kliniska.",
        ],
        table=pd.DataFrame(rows),
    )


_OPTIONAL_SECTION_BUILDERS = (factor_section, validity_section, norms_section, measurement_error_section)
_CONDITIONAL_SECTION_BUILDERS = (decision_support_section, fairness_section)


def full_report_sections(dataset, include_ml: bool = True) -> list[ReportSection]:
    """The 'Complete Analysis Report' (SRS §27): dataset overview through
    machine learning, generic across all three test types. Sections that
    fail (e.g. too few items for EFA) are skipped rather than crashing the
    whole report."""
    sections = [dataset_overview_section(dataset), qc_section(dataset), reliability_section(dataset)]
    for builder in _OPTIONAL_SECTION_BUILDERS:
        try:
            sections.append(builder(dataset))
        except Exception as exc:
            logger.warning(f"Report section failed ({builder.__name__}): {exc}")
    for builder in _CONDITIONAL_SECTION_BUILDERS:
        try:
            section = builder(dataset)
            if section is not None:
                sections.append(section)
        except Exception as exc:
            logger.warning(f"Report section failed ({builder.__name__}): {exc}")
    if include_ml:
        try:
            section = ml_section(dataset)
            if section is not None:
                sections.append(section)
        except Exception as exc:
            logger.warning(f"ML report section failed: {exc}")
    return sections


def psychometric_summary_sections(dataset) -> list[ReportSection]:
    """Reliability + validity + fairness + profile summary - the
    'Psykometrisk sammanfattning' quick export."""
    sections = [dataset_overview_section(dataset), reliability_section(dataset), validity_section(dataset)]
    try:
        fairness = fairness_section(dataset)
        if fairness is not None:
            sections.append(fairness)
    except Exception as exc:
        logger.warning(f"Fairness section failed: {exc}")
    return sections


def decision_support_report_sections(dataset) -> list[ReportSection] | None:
    section = decision_support_section(dataset)
    if section is None:
        return None
    return [dataset_overview_section(dataset), section]


def full_analysis_report_pdf(dataset, include_ml: bool = True) -> bytes:
    q = dataset.questionnaire
    return build_pdf_report(
        f"Komplett analysrapport – {q.test_name}",
        q.full_name,
        full_report_sections(dataset, include_ml=include_ml),
    )


def psychometric_summary_pdf(dataset) -> bytes:
    q = dataset.questionnaire
    return build_pdf_report(
        f"Psykometrisk sammanfattning – {q.test_name}",
        q.full_name,
        psychometric_summary_sections(dataset),
    )


def decision_support_report_pdf(dataset) -> bytes | None:
    sections = decision_support_report_sections(dataset)
    if sections is None:
        return None
    q = dataset.questionnaire
    return build_pdf_report(f"Beslutsstödsrapport – {q.test_name}", q.full_name, sections)


# --- Combined ZIP export -------------------------------------------------


def build_export_zip(
    dataset,
    include_full_report: bool = True,
    include_summary_report: bool = False,
    include_decision_report: bool = False,
    include_figures: bool = False,
    include_raw_data: bool = False,
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        if include_full_report:
            zf.writestr("psykometrisk_rapport.pdf", full_analysis_report_pdf(dataset))
        if include_summary_report:
            zf.writestr("psykometrisk_sammanfattning.pdf", psychometric_summary_pdf(dataset))
        if include_decision_report:
            pdf = decision_support_report_pdf(dataset)
            if pdf is not None:
                zf.writestr("beslutsstodsrapport.pdf", pdf)
        if include_figures:
            for name, fig in key_figures(dataset).items():
                try:
                    zf.writestr(f"figurer/{name}.png", figure_to_png_bytes(fig))
                except Exception as exc:
                    logger.warning(f"Skipped figure '{name}' in ZIP export: {exc}")
        if include_raw_data:
            zf.writestr("data_anonymiserad.csv", dataframe_to_csv_bytes(anonymized_scored_dataset(dataset)))
    buffer.seek(0)
    return buffer.getvalue()
