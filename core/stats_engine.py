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


def subscale_item_columns(dataset, subscale_id: str | None = None) -> list[str]:
    """Public accessor: None returns ALL item columns (whole-test); an id
    returns just that subscale's items."""
    return _subscale_item_columns(dataset, subscale_id)


def linear_fit(x: pd.Series, y: pd.Series) -> tuple[float, float] | None:
    """Least-squares (slope, intercept) for a simple trend line, e.g. for the
    test-retest scatter. Returns None if there isn't enough variance to fit."""
    if len(x) < 2 or x.std() == 0:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


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


def response_distribution_by_item(dataset, subscale_id: str | None = None) -> pd.DataFrame:
    """% of responses per item falling in each response category - an item x
    category matrix used for the stacked per-item distribution chart."""
    cols = _subscale_item_columns(dataset, subscale_id)
    scale = dataset.questionnaire.response_scale
    all_levels = list(range(scale.min, scale.max + 1))
    rows = []
    for col in cols:
        counts = dataset.scored[col].value_counts(normalize=True) * 100
        counts = counts.reindex(all_levels, fill_value=0.0)
        rows.append(counts)
    result = pd.DataFrame(rows, index=cols)
    result.columns = [scale.labels.get(str(int(v)), str(int(v))) for v in all_levels]
    return result


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
    """subscale_id=None pools ALL items (whole-test reliability); pass a
    specific subscale id for that subscale's reliability alone."""
    q = dataset.questionnaire
    if subscale_id is None:
        cols = item_columns(dataset)
        sid, name = "", "Alla items"
    else:
        subscale = q.get_subscale(subscale_id)
        if subscale is None:
            return ReliabilitySnapshot("", "", 0, 0, None, None)
        cols = [c for c in subscale.item_ids if c in dataset.scored.columns]
        sid, name = subscale.id, subscale.name

    data = dataset.scored[cols].dropna()
    if len(cols) < 2 or len(data) < 3:
        return ReliabilitySnapshot(sid, name, len(cols), len(data), None, None)

    alpha = float(pg.cronbach_alpha(data=data)[0])
    item_total_rs = []
    for col in cols:
        rest = data.drop(columns=[col]).sum(axis=1)
        r = data[col].corr(rest)
        if pd.notna(r):
            item_total_rs.append(r)
    mean_r = float(np.mean(item_total_rs)) if item_total_rs else None
    return ReliabilitySnapshot(sid, name, len(cols), len(data), alpha, mean_r)


def reliability_snapshot_all_subscales(dataset) -> list[ReliabilitySnapshot]:
    return [reliability_snapshot(dataset, s.id) for s in dataset.questionnaire.subscales]


def _subscale_item_columns(dataset, subscale_id: str | None) -> list[str]:
    """None returns ALL item columns (whole-test); an id returns just that
    subscale's items."""
    if subscale_id is None:
        return item_columns(dataset)
    subscale = dataset.questionnaire.get_subscale(subscale_id)
    if subscale is None:
        return []
    return [c for c in subscale.item_ids if c in dataset.scored.columns]


def alpha_confidence_interval(alpha: float, n: int, k: int, conf: float = 0.95) -> tuple[float | None, float | None]:
    """Exact CI for Cronbach's alpha (Feldt, 1965), via the F-distribution."""
    from scipy import stats as sps

    if alpha is None or n <= 1 or k <= 1:
        return None, None
    df1 = n - 1
    df2 = (n - 1) * (k - 1)
    tail = (1 - conf) / 2
    lower = 1 - (1 - alpha) * sps.f.ppf(1 - tail, df1, df2)
    upper = 1 - (1 - alpha) * sps.f.ppf(tail, df1, df2)
    return float(lower), float(upper)


def item_correlation_matrix(dataset, subscale_id: str | None = None) -> pd.DataFrame:
    cols = _subscale_item_columns(dataset, subscale_id)
    if len(cols) < 2:
        return pd.DataFrame()
    return dataset.scored[cols].corr()


def mean_inter_item_correlation(dataset, subscale_id: str | None = None) -> float | None:
    corr = item_correlation_matrix(dataset, subscale_id)
    if corr.empty or corr.shape[0] < 2:
        return None
    mask = ~np.eye(corr.shape[0], dtype=bool)
    return float(corr.values[mask].mean())


def item_total_interpretation(r: float | None) -> str:
    if r is None:
        return "–"
    if r >= 0.5:
        return "Bra"
    if r >= 0.3:
        return "Acceptabel"
    return "Låg"


def alpha_interpretation(alpha: float | None) -> str:
    if alpha is None:
        return "–"
    if alpha >= 0.90:
        return "Utmärkt"
    if alpha >= 0.80:
        return "Bra"
    if alpha >= 0.70:
        return "Acceptabelt"
    if alpha >= 0.60:
        return "Tveksamt"
    return "Otillräckligt"


@dataclass
class ItemStats:
    item_id: str
    text: str
    subscale_name: str
    reverse_scored: bool
    mean: float | None
    sd: float | None
    item_total_r: float | None
    alpha_if_deleted: float | None


def item_level_table(dataset, subscale_id: str | None = None) -> list[ItemStats]:
    """Per-item descriptive + reliability contribution table, used by the
    Reliability Explorer."""
    q = dataset.questionnaire
    subscale = q.get_subscale(subscale_id) if subscale_id else None
    items = [i for i in q.items if subscale is None or i.id in subscale.item_ids]
    cols = [i.id for i in items if i.id in dataset.scored.columns]
    data = dataset.scored[cols].dropna()

    results = []
    for item in items:
        if item.id not in cols:
            results.append(ItemStats(item.id, item.text, q.subscale_for_item(item.id) or "", item.reverse_scored, None, None, None, None))
            continue
        series = dataset.scored[item.id].dropna()
        mean = float(series.mean()) if len(series) else None
        sd = float(series.std()) if len(series) else None

        item_total_r = None
        alpha_deleted = None
        if len(cols) >= 3 and len(data) >= 3:
            rest = data.drop(columns=[item.id]).sum(axis=1)
            r = data[item.id].corr(rest)
            item_total_r = float(r) if pd.notna(r) else None
            remaining_cols = [c for c in cols if c != item.id]
            if len(remaining_cols) >= 2:
                alpha_deleted = float(pg.cronbach_alpha(data=data[remaining_cols])[0])

        subscale_name = next((s.name for s in q.subscales if item.id in s.item_ids), "")
        results.append(ItemStats(item.id, item.text, subscale_name, item.reverse_scored, mean, sd, item_total_r, alpha_deleted))
    return results


def mcdonald_omega(dataset, subscale_id: str | None = None) -> float | None:
    """Omega-total from a single-factor loadings solution (factor_analyzer)."""
    from core._compat import patch_factor_analyzer

    patch_factor_analyzer()
    from factor_analyzer import FactorAnalyzer

    cols = _subscale_item_columns(dataset, subscale_id)
    data = dataset.scored[cols].dropna() if cols else pd.DataFrame()
    if len(cols) < 3 or len(data) < 10:
        return None
    try:
        fa = FactorAnalyzer(n_factors=1, rotation=None, method="minres")
        fa.fit(data)
        loadings = fa.loadings_.flatten()
    except Exception:
        return None

    loadings = np.clip(loadings, -0.999, 0.999)
    sum_loadings_sq = float(np.sum(loadings) ** 2)
    sum_uniqueness = float(np.sum(1 - loadings**2))
    denom = sum_loadings_sq + sum_uniqueness
    if denom <= 0:
        return None
    return float(np.clip(sum_loadings_sq / denom, 0, 1))


@dataclass
class SplitHalfResult:
    r_raw: float | None
    spearman_brown: float | None
    guttman: float | None
    n: int


def split_half_reliability(dataset, subscale_id: str | None = None) -> SplitHalfResult:
    """Odd-even split-half reliability, with Spearman-Brown correction and
    Guttman's lambda (which does not assume equal half-variances)."""
    cols = _subscale_item_columns(dataset, subscale_id)
    data = dataset.scored[cols].dropna() if cols else pd.DataFrame()
    if len(cols) < 4 or len(data) < 3:
        return SplitHalfResult(None, None, None, len(data))

    odd_cols = cols[0::2]
    even_cols = cols[1::2]
    odd_total = data[odd_cols].sum(axis=1)
    even_total = data[even_cols].sum(axis=1)
    total = data[cols].sum(axis=1)

    r = odd_total.corr(even_total)
    r = float(r) if pd.notna(r) else None
    spearman_brown = (2 * r) / (1 + r) if r is not None and (1 + r) != 0 else None

    var_odd, var_even, var_total = odd_total.var(ddof=1), even_total.var(ddof=1), total.var(ddof=1)
    guttman = 2 * (1 - (var_odd + var_even) / var_total) if var_total else None

    return SplitHalfResult(r, spearman_brown, float(guttman) if guttman is not None else None, len(data))


@dataclass
class TestRetestResult:
    r: float | None
    n: int
    ci_low: float | None
    ci_high: float | None


def _retest_paired_totals(dataset, subscale_id: str | None = None) -> tuple[pd.Series, pd.Series] | None:
    """t1/t2 total-score pairs for respondents who completed the simulated
    retest wave (`<item_id>_t2` columns), or None if unavailable."""
    q = dataset.questionnaire
    cols = _subscale_item_columns(dataset, subscale_id)
    t2_cols = [f"{c}_t2" for c in cols]
    if not cols or not all(c in dataset.raw.columns for c in t2_cols):
        return None

    scale_min, scale_max = q.response_scale.range
    t2 = dataset.raw[t2_cols].apply(pd.to_numeric, errors="coerce").copy()
    t2.columns = cols
    for item in q.items:
        if item.id in cols and item.reverse_scored:
            t2[item.id] = (scale_min + scale_max) - t2[item.id]

    retest_mask = t2.notna().all(axis=1)
    if retest_mask.sum() < 3:
        return None

    t1_total = dataset.scored.loc[retest_mask, cols].sum(axis=1)
    t2_total = t2.loc[retest_mask].sum(axis=1)
    return t1_total, t2_total


def test_retest_paired_scores(dataset, subscale_id: str | None = None) -> tuple[pd.Series, pd.Series]:
    """Public accessor for the t1/t2 total-score pairs, e.g. for a scatter plot."""
    paired = _retest_paired_totals(dataset, subscale_id)
    if paired is None:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    return paired


def test_retest_reliability(dataset, subscale_id: str | None = None) -> TestRetestResult:
    """Correlates t1 vs t2 total score using the `<item_id>_t2` columns
    present in the raw import (a simulated retest sub-sample for demo data)."""
    paired = _retest_paired_totals(dataset, subscale_id)
    if paired is None:
        return TestRetestResult(None, 0, None, None)
    t1_total, t2_total = paired

    r = t1_total.corr(t2_total)
    if pd.isna(r):
        return TestRetestResult(None, len(t1_total), None, None)

    n = len(t1_total)
    # Fisher z-transform CI for a Pearson correlation
    z = np.arctanh(r)
    se_r = 1 / np.sqrt(n - 3) if n > 3 else None
    if se_r is not None:
        z_lo, z_hi = z - 1.96 * se_r, z + 1.96 * se_r
        ci_low, ci_high = float(np.tanh(z_lo)), float(np.tanh(z_hi))
    else:
        ci_low, ci_high = None, None
    return TestRetestResult(float(r), n, ci_low, ci_high)


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
        return QCCheck("Konstanta frågor", "bad", f"{len(constant)} fråga/frågor: {_format_item_list(constant)}", constant)
    return QCCheck("Konstanta frågor", "good", "Inga konstanta frågor")


def qc_low_variance_items(dataset, threshold: float = 0.1) -> QCCheck:
    cols = item_columns(dataset)
    low = [c for c in cols if dataset.scored[c].var(skipna=True) < threshold]
    if low:
        return QCCheck("Låg varians frågor", "warning", f"{len(low)} fråga/frågor: {_format_item_list(low)}", low)
    return QCCheck("Låg varians frågor", "good", "Inga frågor med låg varians")


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
        return QCCheck(name, "warning", f"{len(flagged)} fråga/frågor: {_format_item_list(flagged)}", flagged)
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
        return QCCheck(name, "warning", f"{len(flagged)} fråga/frågor: {_format_item_list(flagged)}", flagged)
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
        status, message = (
            "bad",
            f"{n_bad} punkt(er) bör granskas - t.ex. dubbletter eller konstanta frågor tyder ofta på "
            "ett fel i datainsamlingen eller exporten, snarare än ett normalt drag i urvalet.",
        )
    elif n_flagged > 0:
        status, message = (
            "warning",
            f"{n_flagged} punkt(er) värda att känna till (t.ex. golv-/takeffekter eller enstaka "
            "outliers). Detta är oftast egenskaper hos urvalet, inte fel i datan - inget du behöver "
            "åtgärda, men bra att ha i åtanke när du tolkar reliabilitet och faktorstruktur.",
        )
    else:
        status, message = "good", "Inga allvarliga problem funna."

    return QualitySummary(checks, n_flagged, patterns, completeness, miss, status, message)


# --- factor analysis -------------------------------------------------


def eigenvalues_for_scree(dataset, subscale_id: str | None = None) -> np.ndarray:
    cols = _subscale_item_columns(dataset, subscale_id)
    data = dataset.scored[cols].dropna()
    if len(cols) < 2 or len(data) < 3:
        return np.array([])
    corr = data.corr().values
    return np.linalg.eigvalsh(corr)[::-1]


@dataclass
class ParallelAnalysisResult:
    eigenvalues: np.ndarray
    simulated_eigenvalues: np.ndarray
    suggested_n_factors: int


def parallel_analysis(dataset, subscale_id: str | None = None, n_iter: int = 50, seed: int = 42) -> ParallelAnalysisResult:
    """Horn's parallel analysis: compares the actual eigenvalues against the
    mean eigenvalues from many random (uncorrelated) datasets of the same
    shape. Factors are retained where the actual eigenvalue exceeds the
    random-data eigenvalue at that position."""
    cols = _subscale_item_columns(dataset, subscale_id)
    data = dataset.scored[cols].dropna()
    n, p = data.shape
    if p < 2 or n < 3:
        return ParallelAnalysisResult(np.array([]), np.array([]), 0)

    actual = eigenvalues_for_scree(dataset, subscale_id)
    rng = np.random.default_rng(seed)
    simulated = np.empty((n_iter, p))
    for i in range(n_iter):
        random_data = rng.normal(size=(n, p))
        simulated[i] = np.linalg.eigvalsh(np.corrcoef(random_data, rowvar=False))[::-1]
    simulated_mean = simulated.mean(axis=0)
    suggested = max(int(np.sum(actual > simulated_mean)), 1)
    return ParallelAnalysisResult(actual, simulated_mean, suggested)


@dataclass
class EFAFitIndices:
    chi2: float | None
    df: int | None
    rmsea: float | None
    cfi: float | None
    tli: float | None
    srmr: float | None


def _ml_discrepancy(sigma_hat: np.ndarray, sample_corr: np.ndarray, p: int) -> float | None:
    """F_ML(Sigma, S) = log|Sigma| + tr(Sigma^-1 S) - log|S| - p."""
    sign_hat, logdet_hat = np.linalg.slogdet(sigma_hat)
    sign_s, logdet_s = np.linalg.slogdet(sample_corr)
    if sign_hat <= 0 or sign_s <= 0:
        return None
    try:
        sigma_hat_inv = np.linalg.inv(sigma_hat)
    except np.linalg.LinAlgError:
        return None
    return float(logdet_hat + np.trace(sigma_hat_inv @ sample_corr) - logdet_s - p)


def _efa_fit_indices(sample_corr: np.ndarray, loadings: np.ndarray, uniquenesses: np.ndarray, n: int, p: int, k: int) -> EFAFitIndices:
    """RMSEA/CFI/TLI/SRMR for a maximum-likelihood EFA solution, computed
    against a k=0 (independence) baseline model - the standard SEM approach
    (Hu & Bentler, 1999) applied to an EFA loadings solution."""
    df = ((p - k) ** 2 - (p + k)) // 2
    if df <= 0:
        return EFAFitIndices(None, None, None, None, None, None)

    sigma_hat = loadings @ loadings.T + np.diag(uniquenesses)
    f_ml = _ml_discrepancy(sigma_hat, sample_corr, p)
    if f_ml is None:
        return EFAFitIndices(None, df, None, None, None, None)
    chi2 = max(0.0, (n - 1) * f_ml)

    sign_s, logdet_s = np.linalg.slogdet(sample_corr)
    if sign_s <= 0:
        return EFAFitIndices(chi2, df, None, None, None, None)
    df0 = p * (p - 1) // 2
    chi2_0 = max(0.0, -(n - 1) * logdet_s)

    rmsea = float(np.sqrt(max(0.0, chi2 - df) / (df * (n - 1))))

    residual = sample_corr - sigma_hat
    off_diag = residual[~np.eye(p, dtype=bool)]
    srmr = float(np.sqrt(np.mean(off_diag**2)))

    ratio_0 = chi2_0 / df0 if df0 > 0 else None
    ratio_k = chi2 / df
    tli = float((ratio_0 - ratio_k) / (ratio_0 - 1)) if ratio_0 and ratio_0 != 1 else None

    denom = max(chi2_0 - df0, chi2 - df, 1e-9)
    cfi = float(1 - max(chi2 - df, 0) / denom)

    return EFAFitIndices(chi2, df, rmsea, cfi, tli, srmr)


@dataclass
class EFAResult:
    n_factors: int
    n: int
    loadings: pd.DataFrame
    communalities: pd.Series
    variance_explained: pd.DataFrame
    phi: pd.DataFrame | None
    fit: EFAFitIndices


@dataclass
class SamplingAdequacy:
    kmo: float | None
    bartlett_chi2: float | None
    bartlett_p: float | None


def sampling_adequacy(dataset, subscale_id: str | None = None) -> SamplingAdequacy:
    """Kaiser-Meyer-Olkin (sampling adequacy) and Bartlett's test of
    sphericity - standard checks that the item correlation matrix is
    suitable for factor analysis at all."""
    from core._compat import patch_factor_analyzer

    patch_factor_analyzer()
    from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

    cols = _subscale_item_columns(dataset, subscale_id)
    data = dataset.scored[cols].dropna()
    if len(cols) < 3 or len(data) < 10:
        return SamplingAdequacy(None, None, None)
    try:
        _, kmo_model = calculate_kmo(data)
        chi2, p = calculate_bartlett_sphericity(data)
    except Exception:
        return SamplingAdequacy(None, None, None)
    return SamplingAdequacy(float(kmo_model), float(chi2), float(p))


def efa_fit(dataset, n_factors: int, subscale_id: str | None = None) -> EFAResult | None:
    """Exploratory factor analysis (maximum likelihood extraction; promax
    oblique rotation for n_factors > 1, matching Big Five's known correlated
    dimensions). Returns None if there isn't enough data to fit."""
    from core._compat import patch_factor_analyzer

    patch_factor_analyzer()
    from factor_analyzer import FactorAnalyzer

    cols = _subscale_item_columns(dataset, subscale_id)
    data = dataset.scored[cols].dropna()
    n, p = data.shape
    if p < 3 or n < 10 or n_factors < 1 or n_factors >= p:
        return None

    rotation = None if n_factors == 1 else "promax"
    try:
        fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method="ml")
        fa.fit(data)
    except Exception:
        return None

    factor_names = [f"F{i + 1}" for i in range(n_factors)]
    loadings = pd.DataFrame(fa.loadings_, index=cols, columns=factor_names)
    communalities = pd.Series(fa.get_communalities(), index=cols, name="communality")
    variance = fa.get_factor_variance()
    variance_df = pd.DataFrame(
        variance, index=["SS loadings", "Proportion Var", "Cumulative Var"], columns=factor_names
    ).T

    phi = None
    if n_factors > 1 and fa.phi_ is not None:
        phi = pd.DataFrame(fa.phi_, index=factor_names, columns=factor_names)

    fit = _efa_fit_indices(data.corr().values, fa.loadings_, fa.get_uniquenesses(), n, p, n_factors)
    return EFAResult(n_factors, n, loadings, communalities, variance_df, phi, fit)


def factor_item_groups(efa: EFAResult, questionnaire) -> list[tuple[str, list[tuple[str, str, float]]]]:
    """Groups items by their primary (highest-magnitude) factor loading -
    a plain-language view of "which questions make up which factor", since
    the raw loadings table alone is hard to read without a stats background.
    Returns one (factor_name, items) tuple per factor, items sorted by
    |loading| descending; each item is (item_id, text, signed loading)."""
    text_by_id = {item.id: item.text for item in questionnaire.items}
    primary_factor = efa.loadings.abs().idxmax(axis=1)
    groups = []
    for factor_name in efa.loadings.columns:
        item_ids = [iid for iid in efa.loadings.index if primary_factor[iid] == factor_name]
        items = sorted(
            ((iid, text_by_id.get(iid, iid), float(efa.loadings.loc[iid, factor_name])) for iid in item_ids),
            key=lambda t: abs(t[2]),
            reverse=True,
        )
        groups.append((factor_name, items))
    return groups


# --- validity -------------------------------------------------

CRITERION_CONVERGENT_COL = "criterion_convergent"
CRITERION_DISCRIMINANT_COL = "criterion_discriminant"


@dataclass
class CriterionValidity:
    convergent_r: float | None
    discriminant_r: float | None
    convergent_pair: tuple[pd.Series, pd.Series] | None
    discriminant_pair: tuple[pd.Series, pd.Series] | None


def criterion_validity(dataset) -> CriterionValidity:
    """Correlates the primary total score against the simulated external
    criterion variables (see data/generate_sample_data.py). Real datasets
    without those columns simply get None back - this never crashes on an
    unmapped/missing criterion."""
    primary_total = total_score_series(dataset)

    def _pair(col: str) -> tuple[float, tuple[pd.Series, pd.Series]] | tuple[None, None]:
        if col not in dataset.raw.columns or primary_total.empty:
            return None, None
        criterion = pd.to_numeric(dataset.raw[col], errors="coerce")
        paired = pd.concat([primary_total, criterion], axis=1).dropna()
        if len(paired) < 3:
            return None, None
        r = paired.iloc[:, 0].corr(paired.iloc[:, 1])
        if pd.isna(r):
            return None, None
        return float(r), (paired.iloc[:, 0], paired.iloc[:, 1])

    conv_r, conv_pair = _pair(CRITERION_CONVERGENT_COL)
    disc_r, disc_pair = _pair(CRITERION_DISCRIMINANT_COL)
    return CriterionValidity(conv_r, disc_r, conv_pair, disc_pair)


@dataclass
class ValiditySource:
    key: str
    label: str
    status: str  # "strong" | "moderate" | "limited" | "none"
    summary: str


VALIDITY_STATUS_LABELS = {
    "strong": "Stark evidens",
    "moderate": "Måttlig evidens",
    "limited": "Begränsad evidens",
    "none": "Ingen information",
}


def validity_overview(
    dataset,
    response_process_status: str = "none",
    consequences_status: str = "none",
    response_process_summary: str | None = None,
    consequences_summary: str | None = None,
) -> list[ValiditySource]:
    """The five evidence sources from the Standards for Educational and
    Psychological Testing (AERA/APA/NCME). Content, internal structure, and
    relations-to-other-variables are derived from the data; response
    processes and consequences are not computable from a dataset and are
    passed in as manually-set statuses from the page's own UI state (with an
    optional summary override, e.g. a citation for a well-established test)."""
    q = dataset.questionnaire
    sources: list[ValiditySource] = []

    content_status = "strong" if q.source_citation else "limited"
    content_summary = (
        f"Källa: {q.source_citation}" if q.source_citation else "Ingen källhänvisning angiven för detta test."
    )
    sources.append(ValiditySource("content", "Testinnehåll (Content)", content_status, content_summary))

    sources.append(
        ValiditySource(
            "response_processes",
            "Responsprocesser (Response Processes)",
            response_process_status,
            response_process_summary or "Manuellt dokumenterad status - se fliken för detaljer.",
        )
    )

    overall = reliability_snapshot(dataset)
    efa = efa_fit(dataset, max(1, len(q.subscales))) if q.subscales else None
    alpha_ok = overall.alpha is not None and overall.alpha >= 0.80
    rmsea_ok = efa is not None and efa.fit.rmsea is not None and efa.fit.rmsea < 0.08
    if alpha_ok and (efa is None or efa.fit.rmsea is None or rmsea_ok):
        structure_status = "strong"
    elif overall.alpha is not None and overall.alpha >= 0.70:
        structure_status = "moderate"
    elif overall.alpha is not None:
        structure_status = "limited"
    else:
        structure_status = "none"
    structure_parts = []
    if overall.alpha is not None:
        structure_parts.append(f"Cronbach's alpha = {overall.alpha:.2f}")
    if efa is not None and efa.fit.rmsea is not None:
        structure_parts.append(f"RMSEA = {efa.fit.rmsea:.3f}")
    structure_summary = ", ".join(structure_parts) if structure_parts else "Otillräcklig data."
    sources.append(ValiditySource("internal_structure", "Intern struktur (Internal Structure)", structure_status, structure_summary))

    cv = criterion_validity(dataset)
    if cv.convergent_r is not None and cv.discriminant_r is not None:
        if abs(cv.convergent_r) >= 0.5 and abs(cv.discriminant_r) < 0.3:
            relations_status = "strong"
        elif abs(cv.convergent_r) >= 0.3:
            relations_status = "moderate"
        else:
            relations_status = "limited"
        relations_summary = f"Konvergent r = {cv.convergent_r:.2f}, Diskriminant r = {cv.discriminant_r:.2f}"
    else:
        relations_status = "none"
        relations_summary = "Ingen kriterievariabel tillgänglig i datasetet."
    sources.append(
        ValiditySource("relations", "Relation till andra variabler (Relations to Other Variables)", relations_status, relations_summary)
    )

    sources.append(
        ValiditySource(
            "consequences",
            "Konsekvenser av testanvändning (Consequences)",
            consequences_status,
            consequences_summary
            or "Manuellt dokumenterad status - se Decision Support och Fairness Explorer för relaterad analys.",
        )
    )

    return sources


def validity_status_counts(sources: list[ValiditySource]) -> dict[str, int]:
    counts = {"strong": 0, "moderate": 0, "limited": 0, "none": 0}
    for source in sources:
        counts[source.status] += 1
    return counts


def validity_aggregate_status(counts: dict[str, int]) -> tuple[str, str]:
    """Rolls the five evidence-source statuses up into one headline verdict.

    Sources with no information ("none") are kept distinct from sources with
    weak-but-present evidence ("limited"/"moderate") - lumping them together
    would let a mostly-undocumented test (e.g. a freshly uploaded real
    dataset with no criterion variables and no response-process notes) read
    as "Måttlig" (moderate evidence) when the honest picture is "we don't
    know yet", not "we checked and it's so-so"."""
    if counts["strong"] >= 4:
        return "good", "Bra"
    if counts["none"] >= 3:
        return "warning", "Data saknas"
    if counts["strong"] >= 2:
        return "warning", "Måttlig"
    return "bad", "Bör stärkas"


# --- norms -------------------------------------------------

# Standard normal stanine cut-points (9-band scale with the classic
# 4/7/12/17/20/17/12/7/4 % distribution).
STANINE_Z_CUTS = [-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75]


@dataclass
class NormStats:
    mean: float | None
    sd: float | None
    n: int


def norm_stats(dataset, subscale_id: str | None = None) -> NormStats:
    """Sample-based normative reference: this demo dataset doubles as its own
    normative sample (no external population norm data exists for synthetic
    data). Documented explicitly in the UI, not just here."""
    series = total_score_series(dataset, subscale_id).dropna()
    if len(series) == 0:
        return NormStats(None, None, 0)
    return NormStats(float(series.mean()), float(series.std()), len(series))


def score_manual_responses(dataset, responses: dict[str, float], subscale_id: str | None = None) -> float | None:
    """Score a single new respondent's raw item answers (e.g. entered by
    hand for a client who isn't a row in the loaded dataset) using the same
    reverse-scoring and aggregation rule as core/import_engine.build_dataset.
    Returns None if no items in the subscale have an answer."""
    q = dataset.questionnaire
    if subscale_id is None:
        subscale = q.subscales[0] if q.subscales else None
    else:
        subscale = q.get_subscale(subscale_id)
    if subscale is None:
        return None

    scale_min, scale_max = q.response_scale.range
    items_by_id = {item.id: item for item in q.items}
    values = []
    for item_id in subscale.item_ids:
        if item_id not in responses or responses[item_id] is None:
            continue
        value = float(responses[item_id])
        item = items_by_id.get(item_id)
        if item is not None and item.reverse_scored:
            value = (scale_min + scale_max) - value
        values.append(value)

    if not values:
        return None
    if subscale.scoring_method == "mean":
        return sum(values) / len(values)
    return sum(values)


def cutoff_category(questionnaire, raw_score: float) -> str | None:
    """Which of the test's defined severity/cutoff bands a raw score falls
    into (e.g. GAD-7's "Mild", "Måttlig", "Svår"), or None if the test has
    no cutoffs defined or the score falls outside all of them."""
    for cutoff in questionnaire.cutoffs:
        lo, hi = cutoff.range
        if lo <= raw_score <= hi:
            return cutoff.label
    return None


def raw_to_z(raw: float, mean: float, sd: float) -> float:
    return (raw - mean) / sd if sd else 0.0


def z_to_t(z: float) -> float:
    return 50 + 10 * z


def z_to_percentile(z: float) -> float:
    from scipy.stats import norm as _norm

    return float(_norm.cdf(z) * 100)


def z_to_stanine(z: float) -> int:
    return int(np.digitize(z, STANINE_Z_CUTS) + 1)


@dataclass
class ScoreConversion:
    raw: float
    z: float
    t: float
    percentile: float
    stanine: int


def score_conversion(raw: float, mean: float, sd: float) -> ScoreConversion:
    z = raw_to_z(raw, mean, sd)
    return ScoreConversion(raw, z, z_to_t(z), z_to_percentile(z), z_to_stanine(z))


def conversion_table(dataset, subscale_id: str | None = None) -> pd.DataFrame:
    """Raw -> z -> T -> percentile -> stanine for every possible raw score in
    the subscale's range (not just observed scores) - a reference table."""
    q = dataset.questionnaire
    subscale = q.get_subscale(subscale_id) if subscale_id else (q.subscales[0] if q.subscales else None)
    stats = norm_stats(dataset, subscale_id)
    if subscale is None or stats.mean is None or stats.sd is None or stats.sd == 0:
        return pd.DataFrame()

    lo, hi = subscale.score_range
    rows = []
    for raw in range(int(lo), int(hi) + 1):
        c = score_conversion(raw, stats.mean, stats.sd)
        rows.append(
            {
                "Råpoäng": c.raw,
                "Z-poäng": round(c.z, 2),
                "T-poäng": round(c.t, 1),
                "Percentil": round(c.percentile, 1),
                "Stanine": c.stanine,
            }
        )
    return pd.DataFrame(rows)


def person_ids(dataset) -> list:
    if "respondent_id" in dataset.raw.columns:
        return dataset.raw["respondent_id"].tolist()
    return list(dataset.raw.index)


def person_raw_score(dataset, person_id, subscale_id: str | None = None) -> float | None:
    q = dataset.questionnaire
    subscale = q.get_subscale(subscale_id) if subscale_id else (q.subscales[0] if q.subscales else None)
    if subscale is None:
        return None
    col = f"{subscale.id}_total"
    if col not in dataset.scored.columns:
        return None

    if "respondent_id" in dataset.raw.columns:
        mask = dataset.raw["respondent_id"] == person_id
        if not mask.any():
            return None
        idx = dataset.raw.index[mask][0]
    else:
        idx = person_id

    if idx not in dataset.scored.index:
        return None
    val = dataset.scored.loc[idx, col]
    return float(val) if pd.notna(val) else None


def normal_curve_points(mean: float, sd: float, n: int = 200, span: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    from scipy.stats import norm as _norm

    if sd is None or sd <= 0:
        return np.array([]), np.array([])
    x = np.linspace(mean - span * sd, mean + span * sd, n)
    y = _norm.pdf(x, mean, sd)
    return x, y


# --- measurement error -------------------------------------------------


@dataclass
class MeasurementError:
    sem: float | None
    alpha: float | None
    sd: float | None


def measurement_error(dataset, subscale_id: str | None = None) -> MeasurementError:
    """SEM = SD * sqrt(1 - alpha) (classical test theory)."""
    alpha = reliability_snapshot(dataset, subscale_id).alpha
    stats = norm_stats(dataset, subscale_id)
    if alpha is None or stats.sd is None:
        return MeasurementError(None, alpha, stats.sd)
    sem = stats.sd * np.sqrt(max(0.0, 1 - alpha))
    return MeasurementError(float(sem), alpha, stats.sd)


def confidence_interval(raw_score: float, sem: float, z: float = 1.96) -> tuple[float, float]:
    """95% CI = score ± 1.96*SEM by default (the conventional psychometrics
    approximation; scipy's exact z_0.975 is 1.9600 to 4 decimal places, so
    the literal constant is used directly rather than recomputed)."""
    margin = z * sem
    return raw_score - margin, raw_score + margin


def all_person_confidence_intervals(dataset, subscale_id: str | None = None) -> pd.DataFrame:
    """95% CI = score ± 1.96*SEM for every respondent - the per-person table
    used by Measurement Error's confidence-interval tab."""
    me = measurement_error(dataset, subscale_id)
    if me.sem is None:
        return pd.DataFrame()
    series = total_score_series(dataset, subscale_id).dropna()
    id_col = dataset.raw["respondent_id"] if "respondent_id" in dataset.raw.columns else None

    rows = []
    for idx, score in series.items():
        lo, hi = confidence_interval(float(score), me.sem)
        label = str(id_col.loc[idx]) if id_col is not None and idx in id_col.index else str(idx)
        rows.append(
            {
                "Person": label,
                "Observerad poäng": score,
                "SEM": round(me.sem, 2),
                "95% KI nedre": round(lo, 1),
                "95% KI övre": round(hi, 1),
                "Intervall (±)": round(1.96 * me.sem, 1),
            }
        )
    return pd.DataFrame(rows)


def minimum_reliable_change(sem: float) -> float:
    """The smallest score difference that counts as reliable change at the
    95% level (Jacobson & Truax, 1991): 1.96 * SE_diff, SE_diff = SEM*sqrt(2)."""
    return 1.96 * np.sqrt(2) * sem


def precision_label(sem: float | None, sd: float | None) -> str:
    if sem is None or sd is None or sd == 0:
        return "Okänd"
    ratio = sem / sd
    if ratio < 0.35:
        return "Hög"
    if ratio < 0.5:
        return "Måttlig"
    return "Låg"


@dataclass
class ReliableChangeResult:
    person_id: str
    score_t1: float
    score_t2: float
    diff: float
    se_diff: float
    rci: float
    reliable: bool


def reliable_change_index(dataset, subscale_id: str | None = None) -> list[ReliableChangeResult]:
    """RCI for every respondent who completed the simulated retest wave,
    using the same t1/t2 pairs as Reliability Explorer's test-retest tab."""
    me = measurement_error(dataset, subscale_id)
    if me.sem is None or me.sem == 0:
        return []
    se_diff = me.sem * np.sqrt(2)
    t1, t2 = test_retest_paired_scores(dataset, subscale_id)
    id_col = dataset.raw["respondent_id"] if "respondent_id" in dataset.raw.columns else None
    results = []
    for idx in t1.index:
        diff = float(t2[idx] - t1[idx])
        rci = diff / se_diff
        label = str(id_col.loc[idx]) if id_col is not None else str(idx)
        results.append(ReliableChangeResult(label, float(t1[idx]), float(t2[idx]), diff, se_diff, rci, abs(rci) >= 1.96))
    return results


# --- decision support -------------------------------------------------

OUTCOME_COL = "outcome_positive"


def has_outcome(dataset) -> bool:
    return OUTCOME_COL in dataset.raw.columns


def _paired_score_outcome(dataset, subscale_id: str | None = None) -> pd.DataFrame:
    """Aligned (score, outcome) pairs with missing values dropped."""
    if not has_outcome(dataset):
        return pd.DataFrame()
    score = total_score_series(dataset, subscale_id)
    outcome = pd.to_numeric(dataset.raw[OUTCOME_COL], errors="coerce")
    paired = pd.concat([score, outcome], axis=1).dropna()
    paired.columns = ["score", "outcome"]
    return paired


@dataclass
class ROCResult:
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc: float
    auc_ci: tuple[float, float] | None
    n: int


def roc_analysis(dataset, subscale_id: str | None = None, n_bootstrap: int = 200, seed: int = 7) -> ROCResult | None:
    from sklearn.metrics import roc_auc_score, roc_curve

    paired = _paired_score_outcome(dataset, subscale_id)
    if len(paired) < 10 or paired["outcome"].nunique() < 2:
        return None

    fpr, tpr, thresholds = roc_curve(paired["outcome"], paired["score"])
    auc = float(roc_auc_score(paired["outcome"], paired["score"]))

    rng = np.random.default_rng(seed)
    boot_aucs = []
    n = len(paired)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        sample = paired.iloc[idx]
        if sample["outcome"].nunique() < 2:
            continue
        boot_aucs.append(roc_auc_score(sample["outcome"], sample["score"]))
    ci = (float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5))) if len(boot_aucs) >= 20 else None

    return ROCResult(fpr, tpr, thresholds, auc, ci, n)


@dataclass
class CutoffMetrics:
    threshold: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    youdens_j: float
    tp: int
    fp: int
    tn: int
    fn: int


def confusion_at_threshold(dataset, threshold: float, subscale_id: str | None = None) -> CutoffMetrics | None:
    """Classifies score >= threshold as predicted-positive."""
    paired = _paired_score_outcome(dataset, subscale_id)
    if paired.empty:
        return None

    predicted_positive = paired["score"] >= threshold
    actual_positive = paired["outcome"] == 1

    tp = int((predicted_positive & actual_positive).sum())
    fp = int((predicted_positive & ~actual_positive).sum())
    fn = int((~predicted_positive & actual_positive).sum())
    tn = int((~predicted_positive & ~actual_positive).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    youdens_j = sensitivity + specificity - 1

    return CutoffMetrics(threshold, sensitivity, specificity, ppv, npv, youdens_j, tp, fp, tn, fn)


def cutoff_table(dataset, subscale_id: str | None = None, max_rows: int = 15) -> pd.DataFrame:
    """Sensitivity/specificity/PPV/NPV/Youden's J across candidate cut
    scores spanning the subscale's possible range."""
    q = dataset.questionnaire
    subscale = q.get_subscale(subscale_id) if subscale_id else (q.subscales[0] if q.subscales else None)
    if subscale is None or not has_outcome(dataset):
        return pd.DataFrame()

    lo, hi = subscale.score_range
    span = int(hi) - int(lo)
    step = max(1, span // max_rows)
    rows = []
    for t in range(int(lo), int(hi) + 1, step):
        m = confusion_at_threshold(dataset, t, subscale_id)
        if m is None:
            continue
        rows.append(
            {
                "Tröskel": t,
                "Sensitivitet": round(m.sensitivity, 2),
                "Specificitet": round(m.specificity, 2),
                "PPV": round(m.ppv, 2),
                "NPV": round(m.npv, 2),
                "Youden's J": round(m.youdens_j, 2),
            }
        )
    return pd.DataFrame(rows)


def youdens_optimal_threshold(dataset, subscale_id: str | None = None) -> CutoffMetrics | None:
    """The cut score maximizing Youden's J (sensitivity + specificity - 1)."""
    q = dataset.questionnaire
    subscale = q.get_subscale(subscale_id) if subscale_id else (q.subscales[0] if q.subscales else None)
    if subscale is None or not has_outcome(dataset):
        return None

    lo, hi = subscale.score_range
    best = None
    for t in range(int(lo), int(hi) + 1):
        m = confusion_at_threshold(dataset, t, subscale_id)
        if m is None:
            continue
        if best is None or m.youdens_j > best.youdens_j:
            best = m
    return best


# --- fairness -------------------------------------------------

FAIRNESS_DIMENSIONS = {
    "gender": "Kön",
    "age_group": "Ålder",
    "education": "Utbildning",
    "group": "Grupp",
}


def available_fairness_dimensions(dataset) -> dict[str, str]:
    """Which of the standard fairness grouping variables are present in
    this dataset's raw import."""
    available = {}
    for key, label in FAIRNESS_DIMENSIONS.items():
        source_col = "age" if key == "age_group" else key
        if source_col in dataset.raw.columns:
            available[key] = label
    return available


def age_group_series(dataset) -> pd.Series | None:
    if "age" not in dataset.raw.columns:
        return None
    age = pd.to_numeric(dataset.raw["age"], errors="coerce")
    return pd.cut(age, bins=[0, 25, 200], labels=["18-25", "26-65"]).astype(str)


def fairness_dimension_series(dataset, dimension_key: str) -> pd.Series | None:
    if dimension_key == "age_group":
        return age_group_series(dataset)
    if dimension_key in dataset.raw.columns:
        return dataset.raw[dimension_key]
    return None


def cohens_d_interpretation(d: float) -> str:
    magnitude = abs(d)
    if magnitude < 0.2:
        return "Ingen/försumbar skillnad"
    if magnitude < 0.5:
        return "Liten skillnad"
    if magnitude < 0.8:
        return "Måttlig skillnad"
    return "Stor skillnad"


def fairness_index(d: float) -> float:
    """A simple 0-1 index for display: 1.0 = no group difference at all,
    decreasing linearly to 0 at |d| >= 2 (a very large effect)."""
    return float(max(0.0, 1 - min(abs(d) / 2, 1)))


@dataclass
class GroupComparisonResult:
    dimension: str
    reference_group: str
    comparison_group: str
    mean_reference: float
    mean_comparison: float
    n_reference: int
    n_comparison: int
    cohens_d: float
    interpretation: str
    fairness_index: float


def group_comparison(dataset, group_series: pd.Series, dimension_label: str, subscale_id: str | None = None) -> list[GroupComparisonResult]:
    """Cohen's d for every group vs. the largest (reference) group on the
    subscale's total score."""
    score = total_score_series(dataset, subscale_id)
    paired = pd.concat([score, group_series.rename("group")], axis=1).dropna()
    paired.columns = ["score", "group"]

    counts = paired["group"].value_counts()
    if len(counts) < 2:
        return []

    reference = counts.index[0]
    ref_scores = paired.loc[paired["group"] == reference, "score"]
    results = []
    for candidate in counts.index[1:]:
        comp_scores = paired.loc[paired["group"] == candidate, "score"]
        n1, n2 = len(ref_scores), len(comp_scores)
        pooled_var = ((n1 - 1) * ref_scores.var(ddof=1) + (n2 - 1) * comp_scores.var(ddof=1)) / (n1 + n2 - 2) if (n1 + n2) > 2 else 0
        pooled_sd = np.sqrt(pooled_var) if pooled_var > 0 else 0
        d = (comp_scores.mean() - ref_scores.mean()) / pooled_sd if pooled_sd else 0.0
        results.append(
            GroupComparisonResult(
                dimension=dimension_label,
                reference_group=str(reference),
                comparison_group=str(candidate),
                mean_reference=float(ref_scores.mean()),
                mean_comparison=float(comp_scores.mean()),
                n_reference=n1,
                n_comparison=n2,
                cohens_d=float(d),
                interpretation=cohens_d_interpretation(d),
                fairness_index=fairness_index(d),
            )
        )
    return results


def all_group_comparisons(dataset, subscale_id: str | None = None) -> list[GroupComparisonResult]:
    """Runs group_comparison for every available fairness dimension."""
    results = []
    for key, label in available_fairness_dimensions(dataset).items():
        series = fairness_dimension_series(dataset, key)
        if series is None:
            continue
        results.extend(group_comparison(dataset, series, label, subscale_id))
    return results


@dataclass
class FairnessSummary:
    n_dimensions: int
    n_comparisons: int
    mean_fairness_index: float | None
    max_abs_d: float | None
    max_d_label: str | None
    bias_level: str  # "low" | "moderate" | "high" | "none"


def fairness_summary(results: list[GroupComparisonResult]) -> FairnessSummary:
    if not results:
        return FairnessSummary(0, 0, None, None, None, "none")

    dimensions = {r.dimension for r in results}
    mean_fi = float(np.mean([r.fairness_index for r in results]))
    worst = max(results, key=lambda r: abs(r.cohens_d))
    max_d = abs(worst.cohens_d)

    if max_d < 0.2:
        level = "low"
    elif max_d < 0.5:
        level = "moderate"
    else:
        level = "high"

    label = f"{worst.dimension}: {worst.reference_group} vs {worst.comparison_group}"
    return FairnessSummary(len(dimensions), len(results), mean_fi, max_d, label, level)
