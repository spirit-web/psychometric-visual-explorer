"""Descriptive, reliability, and quality-control statistics.

Pure functions operating on a core.data_model.Dataset. No Streamlit imports,
no plotting - see core/viz_engine.py for turning these results into figures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pingouin as pg

# --- shared helpers -------------------------------------------------------


def item_columns(dataset) -> list[str]:
    return [c for c in dataset.questionnaire.item_ids if c in dataset.scored.columns]


def total_score_series(dataset, subscale_id: str | None = None) -> pd.Series:
    q = dataset.questionnaire
    if subscale_id is None:
        if not q.subscales:
            return pd.Series(dtype=float)
        subscale_id = q.subscales[0].id
    col = f"{subscale_id}_total"
    return dataset.scored[col] if col in dataset.scored.columns else pd.Series(dtype=float)


# --- response distribution / missingness -----------------------------------


def response_distribution(dataset) -> pd.DataFrame:
    """% of all (item x person) responses falling in each response category,
    aggregated across every item - stays small regardless of item count."""
    cols = item_columns(dataset)
    scale = dataset.questionnaire.response_scale
    values = dataset.scored[cols].melt(value_name="response")["response"].dropna()
    counts = values.value_counts(normalize=True).sort_index() * 100

    all_levels = list(range(scale.min, scale.max + 1))
    counts = counts.reindex(all_levels, fill_value=0.0)
    return pd.DataFrame(
        {
            "value": [int(v) for v in counts.index],
            "label": [scale.labels.get(str(int(v)), str(int(v))) for v in counts.index],
            "pct": counts.values,
        }
    )


def missing_by_item(dataset) -> pd.Series:
    cols = item_columns(dataset)
    if not cols:
        return pd.Series(dtype=float)
    return (dataset.scored[cols].isna().mean() * 100).sort_values(ascending=False)


def demographic_breakdown(dataset, column: str) -> pd.Series | None:
    if column not in dataset.raw.columns:
        return None
    return dataset.raw[column].value_counts(normalize=True) * 100


@dataclass
class Completeness:
    n: int
    complete: int
    complete_pct: float
    partial: int
    partial_pct: float
    fully_missing: int
    fully_missing_pct: float


def completeness_summary(dataset) -> Completeness:
    cols = item_columns(dataset)
    n = len(dataset.scored)
    if not cols or n == 0:
        return Completeness(n, 0, 0.0, 0, 0.0, 0, 0.0)
    per_row_missing = dataset.scored[cols].isna().sum(axis=1)
    complete = int((per_row_missing == 0).sum())
    fully_missing = int((per_row_missing == len(cols)).sum())
    partial = n - complete - fully_missing
    return Completeness(
        n=n,
        complete=complete,
        complete_pct=complete / n * 100,
        partial=partial,
        partial_pct=partial / n * 100,
        fully_missing=fully_missing,
        fully_missing_pct=fully_missing / n * 100,
    )


# --- descriptive stats -------------------------------------------------


@dataclass
class DescriptiveStats:
    subscale_id: str
    subscale_name: str
    n: int
    mean: float | None
    sd: float | None
    minimum: float | None
    maximum: float | None
    skewness: float | None
    kurtosis: float | None


def descriptive_stats_by_subscale(dataset) -> list[DescriptiveStats]:
    results = []
    for subscale in dataset.questionnaire.subscales:
        series = total_score_series(dataset, subscale.id).dropna()
        if len(series) == 0:
            results.append(DescriptiveStats(subscale.id, subscale.name, 0, None, None, None, None, None, None))
            continue
        results.append(
            DescriptiveStats(
                subscale_id=subscale.id,
                subscale_name=subscale.name,
                n=len(series),
                mean=float(series.mean()),
                sd=float(series.std()),
                minimum=float(series.min()),
                maximum=float(series.max()),
                skewness=float(series.skew()),
                kurtosis=float(series.kurt()),
            )
        )
    return results


# --- reliability -------------------------------------------------


@dataclass
class ReliabilitySnapshot:
    subscale_id: str
    subscale_name: str
    n_items: int
    n: int
    alpha: float | None
    mean_item_total_r: float | None


def reliability_snapshot(dataset, subscale_id: str | None = None) -> ReliabilitySnapshot:
    q = dataset.questionnaire
    subscale = q.get_subscale(subscale_id) if subscale_id else (q.subscales[0] if q.subscales else None)
    if subscale is None:
        return ReliabilitySnapshot("", "", 0, 0, None, None)

    cols = [c for c in subscale.item_ids if c in dataset.scored.columns]
    data = dataset.scored[cols].dropna()
    if len(cols) < 2 or len(data) < 3:
        return ReliabilitySnapshot(subscale.id, subscale.name, len(cols), len(data), None, None)

    alpha = float(pg.cronbach_alpha(data=data)[0])
    item_total_rs = []
    for col in cols:
        rest = data.drop(columns=[col]).sum(axis=1)
        r = data[col].corr(rest)
        if pd.notna(r):
            item_total_rs.append(r)
    mean_r = float(np.mean(item_total_rs)) if item_total_rs else None
    return ReliabilitySnapshot(subscale.id, subscale.name, len(cols), len(data), alpha, mean_r)


def reliability_snapshot_all_subscales(dataset) -> list[ReliabilitySnapshot]:
    return [reliability_snapshot(dataset, s.id) for s in dataset.questionnaire.subscales]


# --- quality control checks -------------------------------------------------


def _format_item_list(items: list[str], max_show: int = 5) -> str:
    if not items:
        return ""
    shown = ", ".join(items[:max_show])
    if len(items) > max_show:
        shown += f", ... (+{len(items) - max_show} till)"
    return shown


@dataclass
class QCCheck:
    name: str
    status: str  # "good" | "warning" | "bad"
    comment: str
    affected: list[str] = field(default_factory=list)


def qc_constant_items(dataset) -> QCCheck:
    cols = item_columns(dataset)
    constant = [c for c in cols if dataset.scored[c].nunique(dropna=True) <= 1]
    if constant:
        return QCCheck("Konstanta items", "bad", f"{len(constant)} item(s): {_format_item_list(constant)}", constant)
    return QCCheck("Konstanta items", "good", "Inga konstanta items")


def qc_low_variance_items(dataset, threshold: float = 0.1) -> QCCheck:
    cols = item_columns(dataset)
    low = [c for c in cols if dataset.scored[c].var(skipna=True) < threshold]
    if low:
        return QCCheck("Låg varians items", "warning", f"{len(low)} item(s): {_format_item_list(low)}", low)
    return QCCheck("Låg varians items", "good", "Inga items med låg varians")


def qc_floor_effect(dataset, threshold_pct: float = 15.0) -> QCCheck:
    cols = item_columns(dataset)
    scale_min = dataset.questionnaire.response_scale.min
    flagged = []
    for c in cols:
        s = dataset.scored[c].dropna()
        if len(s) == 0:
            continue
        if (s == scale_min).mean() * 100 >= threshold_pct:
            flagged.append(c)
    name = f"Golveffekt (≥{threshold_pct:.0f}% vid min)"
    if flagged:
        return QCCheck(name, "warning", f"{len(flagged)} item(s): {_format_item_list(flagged)}", flagged)
    return QCCheck(name, "good", "Inga golveffekter")


def qc_ceiling_effect(dataset, threshold_pct: float = 15.0) -> QCCheck:
    cols = item_columns(dataset)
    scale_max = dataset.questionnaire.response_scale.max
    flagged = []
    for c in cols:
        s = dataset.scored[c].dropna()
        if len(s) == 0:
            continue
        if (s == scale_max).mean() * 100 >= threshold_pct:
            flagged.append(c)
    name = f"Takeffekt (≥{threshold_pct:.0f}% vid max)"
    if flagged:
        return QCCheck(name, "warning", f"{len(flagged)} item(s): {_format_item_list(flagged)}", flagged)
    return QCCheck(name, "good", "Inga takeffekter")


def qc_outliers(dataset, z_threshold: float = 3.29) -> QCCheck:
    totals = total_score_series(dataset).dropna()
    if totals.empty or totals.std(ddof=0) == 0:
        return QCCheck("Outliers (univariata)", "good", "Ej tillämpligt")
    z = (totals - totals.mean()) / totals.std(ddof=0)
    n_outliers = int((z.abs() > z_threshold).sum())
    if n_outliers > 0:
        return QCCheck("Outliers (univariata)", "warning", f"{n_outliers} deltagare")
    return QCCheck("Outliers (univariata)", "good", "Inga outliers")


def qc_duplicates(dataset) -> QCCheck:
    if "respondent_id" in dataset.raw.columns:
        n_dup = int(dataset.raw["respondent_id"].duplicated().sum())
    else:
        cols = item_columns(dataset)
        n_dup = int(dataset.scored[cols].duplicated().sum()) if cols else 0
    if n_dup > 0:
        return QCCheck("Dubbletter (ID)", "bad", f"{n_dup} dubbletter")
    return QCCheck("Dubbletter (ID)", "good", "Inga dubbletter")


def run_qc_checks(dataset) -> list[QCCheck]:
    return [
        qc_constant_items(dataset),
        qc_low_variance_items(dataset),
        qc_floor_effect(dataset),
        qc_ceiling_effect(dataset),
        qc_outliers(dataset),
        qc_duplicates(dataset),
    ]


@dataclass
class PatternFlags:
    all_same_answer: int
    near_same_answer: int
    total: int


def straightlining_patterns(dataset, near_threshold: float = 0.9) -> PatternFlags:
    """Detect suspicious response patterns: identical answer on every item, or
    near-identical (mode share >= near_threshold)."""
    cols = item_columns(dataset)
    if not cols:
        return PatternFlags(0, 0, 0)
    data = dataset.scored[cols]

    def _mode_share(row: pd.Series) -> float:
        vals = row.dropna()
        if len(vals) == 0:
            return 0.0
        return vals.value_counts(normalize=True).iloc[0]

    share = data.apply(_mode_share, axis=1)
    all_same = int((share >= 0.999).sum())
    near_same = int(((share >= near_threshold) & (share < 0.999)).sum())
    return PatternFlags(all_same, near_same, all_same + near_same)


@dataclass
class QualitySummary:
    checks: list[QCCheck]
    n_flagged_checks: int
    pattern_flags: PatternFlags
    completeness: Completeness
    missing_by_item: pd.Series
    status: str  # "good" | "warning" | "bad"
    message: str


def quality_summary(dataset) -> QualitySummary:
    checks = run_qc_checks(dataset)
    n_flagged = sum(1 for c in checks if c.status != "good")
    n_bad = sum(1 for c in checks if c.status == "bad")
    patterns = straightlining_patterns(dataset)
    completeness = completeness_summary(dataset)
    miss = missing_by_item(dataset)

    if n_bad > 0:
        status, message = "bad", "Allvarliga problem hittades. Granska datasetet innan vidare analys."
    elif n_flagged > 0:
        status, message = "warning", f"{n_flagged} punkt(er) att granska. Inga allvarliga problem."
    else:
        status, message = "good", "Inga allvarliga problem funna."

    return QualitySummary(checks, n_flagged, patterns, completeness, miss, status, message)
