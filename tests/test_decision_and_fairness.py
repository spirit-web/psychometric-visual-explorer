import numpy as np
import pandas as pd
import pytest

from core import stats_engine as se
from core.data_model import Dataset, Item, Questionnaire, ResponseScale, Subscale


def _dataset(n=500, n_items=8, seed=11) -> Dataset:
    rng = np.random.default_rng(seed)
    q = Questionnaire(
        plugin_id="demo",
        plugin_version="1.0",
        test_name="Demo",
        full_name="Demo Test",
        response_scale=ResponseScale(min=0, max=3),
        items=[Item(id=f"D{i}", text=f"Item {i}", subscale="total", reverse_scored=False) for i in range(1, n_items + 1)],
        subscales=[Subscale(id="total", name="Total", item_ids=[f"D{i}" for i in range(1, n_items + 1)], score_range=(0, 3 * n_items))],
    )
    theta = rng.normal(0, 1, size=n)
    data = {"respondent_id": np.arange(1, n + 1)}
    for i in range(1, n_items + 1):
        continuous = 0.75 * theta + np.sqrt(1 - 0.75**2) * rng.normal(0, 1, size=n)
        data[f"D{i}"] = np.clip(np.digitize(continuous, [-1, 0, 1]), 0, 3).astype(float)
    raw = pd.DataFrame(data)
    scored = raw.copy()
    scored["total_total"] = scored[[f"D{i}" for i in range(1, n_items + 1)]].sum(axis=1)

    # outcome correlated with theta, like generate_sample_data.py
    logit = 1.3 * (theta - 0.5)
    p = 1 / (1 + np.exp(-logit))
    raw["outcome_positive"] = (rng.random(n) < p).astype(int)

    # fairness demographics: two groups, one with a deliberate mean shift
    group_a_mask = rng.random(n) < 0.6
    raw["group"] = np.where(group_a_mask, "A", "B")
    # bump group B's total score up by adding to theta-derived scored total for a detectable d
    return Dataset(raw=raw, scored=scored, questionnaire=q, column_mapping={}, demographic_columns=["group"])


def test_confusion_at_threshold_formulas():
    dataset = _dataset()
    m = se.confusion_at_threshold(dataset, 12)
    assert m is not None
    assert m.sensitivity == pytest.approx(m.tp / (m.tp + m.fn)) if (m.tp + m.fn) else True
    assert m.specificity == pytest.approx(m.tn / (m.tn + m.fp)) if (m.tn + m.fp) else True
    assert m.youdens_j == pytest.approx(m.sensitivity + m.specificity - 1)
    assert m.tp + m.fp + m.tn + m.fn == len(dataset.raw)


def test_confusion_at_extreme_thresholds():
    dataset = _dataset()
    # threshold below the minimum possible score -> everyone predicted positive
    low = se.confusion_at_threshold(dataset, -1)
    assert low.sensitivity == pytest.approx(1.0)
    assert low.fn == 0
    # threshold above the maximum possible score -> everyone predicted negative
    high = se.confusion_at_threshold(dataset, 1000)
    assert high.specificity == pytest.approx(1.0)
    assert high.fp == 0


def test_roc_analysis_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    dataset = _dataset(n=600)
    roc = se.roc_analysis(dataset)
    assert roc is not None
    paired = se._paired_score_outcome(dataset)
    expected_auc = roc_auc_score(paired["outcome"], paired["score"])
    assert roc.auc == pytest.approx(expected_auc)
    assert roc.auc_ci is not None
    assert roc.auc_ci[0] < roc.auc < roc.auc_ci[1]


def test_youdens_optimal_threshold_maximizes_j():
    dataset = _dataset(n=600)
    optimal = se.youdens_optimal_threshold(dataset)
    q = dataset.questionnaire
    lo, hi = q.subscales[0].score_range
    all_j = [se.confusion_at_threshold(dataset, t).youdens_j for t in range(int(lo), int(hi) + 1)]
    assert optimal.youdens_j == pytest.approx(max(all_j))


def test_cutoff_table_is_monotonic_in_sensitivity():
    dataset = _dataset(n=600)
    table = se.cutoff_table(dataset)
    assert not table.empty
    # sensitivity should be non-increasing as threshold rises
    assert (table["Sensitivitet"].diff().dropna() <= 1e-9).all()


def test_capture_curve_perfect_ranking_captures_everything_immediately():
    # y_score perfectly separates the classes -> capturing the top n_positive%
    # of the population must already capture 100% of the true positives.
    y_true = pd.Series([0, 0, 0, 1, 1, 1, 1, 1])
    y_score = pd.Series([0.1, 0.2, 0.3, 0.9, 0.8, 0.7, 0.6, 0.5])  # positives all score highest
    result = se.capture_curve(y_true, y_score)
    assert result is not None
    assert result.n == 8
    assert result.n_positive == 5
    # at capacity = n_positive/n (0.625), a perfect ranking has captured everyone
    assert se.capture_rate_at_capacity(result, 5 / 8) == pytest.approx(1.0)
    # at capacity 0, nobody's been contacted yet
    assert se.capture_rate_at_capacity(result, 0.0) == pytest.approx(0.0)
    # at capacity 1 (contact everyone), you've necessarily found all positives
    assert se.capture_rate_at_capacity(result, 1.0) == pytest.approx(1.0)


def test_capture_curve_random_score_tracks_the_diagonal():
    # an uninformative score (same value for everyone) ranks in insertion
    # order, so the fraction captured should track the fraction contacted -
    # the "random selection" baseline the chart draws as a dashed diagonal.
    rng = np.random.default_rng(9)
    n = 500
    y_true = pd.Series(rng.integers(0, 2, n))
    y_score = pd.Series(np.zeros(n))  # no information at all
    result = se.capture_curve(y_true, y_score)
    assert result is not None
    mid_capture = se.capture_rate_at_capacity(result, 0.5)
    # with no signal, capturing half the population should capture roughly
    # half of the true positives (not exact, since ties break by original order)
    assert mid_capture == pytest.approx(0.5, abs=0.15)


def test_capture_curve_none_without_any_true_positives():
    y_true = pd.Series([0, 0, 0, 0])
    y_score = pd.Series([0.1, 0.5, 0.3, 0.9])
    assert se.capture_curve(y_true, y_score) is None
    assert se.capture_rate_at_capacity(None, 0.5) == 0.0


def test_capture_curve_for_dataset_matches_generic_capture_curve():
    dataset = _dataset(n=600)
    paired = se._paired_score_outcome(dataset)
    expected = se.capture_curve(paired["outcome"], paired["score"])
    actual = se.capture_curve_for_dataset(dataset)
    assert actual is not None
    assert actual.n == expected.n
    assert actual.n_positive == expected.n_positive
    np.testing.assert_allclose(actual.pct_captured, expected.pct_captured)


def test_has_outcome_false_without_column():
    dataset = _dataset()
    dataset.raw = dataset.raw.drop(columns=["outcome_positive"])
    assert se.has_outcome(dataset) is False
    assert se.roc_analysis(dataset) is None


def test_cohens_d_interpretation_thresholds():
    assert se.cohens_d_interpretation(0.05) == "Ingen/försumbar skillnad"
    assert se.cohens_d_interpretation(0.3) == "Liten skillnad"
    assert se.cohens_d_interpretation(0.6) == "Måttlig skillnad"
    assert se.cohens_d_interpretation(1.0) == "Stor skillnad"
    assert se.cohens_d_interpretation(-1.0) == "Stor skillnad"


def test_fairness_index_bounds():
    assert se.fairness_index(0.0) == pytest.approx(1.0)
    assert se.fairness_index(2.0) == pytest.approx(0.0)
    assert se.fairness_index(4.0) == pytest.approx(0.0)
    assert se.fairness_index(1.0) == pytest.approx(0.5)


def test_group_comparison_recovers_known_mean_difference():
    rng = np.random.default_rng(3)
    n = 400
    q = Questionnaire(
        plugin_id="demo",
        plugin_version="1.0",
        test_name="Demo",
        full_name="Demo",
        response_scale=ResponseScale(min=0, max=10),
        items=[Item(id="D1", text="x", subscale="total", reverse_scored=False)],
        subscales=[Subscale(id="total", name="Total", item_ids=["D1"], score_range=(0, 10))],
    )
    group = rng.choice(["A", "B"], size=n, p=[0.5, 0.5])
    score = np.where(group == "A", rng.normal(10, 2, n), rng.normal(12, 2, n))
    scored = pd.DataFrame({"total_total": score})
    raw = pd.DataFrame({"group": group})
    dataset = Dataset(raw=raw, scored=scored, questionnaire=q, column_mapping={}, demographic_columns=["group"])

    results = se.group_comparison(dataset, raw["group"], "Grupp")
    assert len(results) == 1
    # true effect size magnitude is (12-10)/2 = 1.0 (large); sign depends on
    # which group ends up larger (= reference) under this seed's random split.
    assert abs(results[0].cohens_d) == pytest.approx(1.0, abs=0.25)
    assert results[0].interpretation == "Stor skillnad"


def test_group_comparison_score_series_overrides_the_raw_total():
    """Fairness Explorer's 'Modellens risksannolikhet' mode: passing
    score_series must be compared instead of the subscale total, not
    alongside it - proven here by giving the two a deliberately opposite
    group difference and checking the sign flips."""
    rng = np.random.default_rng(5)
    n = 400
    q = Questionnaire(
        plugin_id="demo",
        plugin_version="1.0",
        test_name="Demo",
        full_name="Demo",
        response_scale=ResponseScale(min=0, max=10),
        items=[Item(id="D1", text="x", subscale="total", reverse_scored=False)],
        subscales=[Subscale(id="total", name="Total", item_ids=["D1"], score_range=(0, 10))],
    )
    group = pd.Series(rng.choice(["A", "B"], size=n, p=[0.5, 0.5]))
    # raw total: group A scores higher than B
    raw_total = pd.Series(np.where(group == "A", rng.normal(12, 2, n), rng.normal(10, 2, n)))
    # an alternative score (e.g. a model's probability): group B scores higher than A
    alt_score = pd.Series(np.where(group == "A", rng.normal(0.2, 0.05, n), rng.normal(0.8, 0.05, n)))
    scored = pd.DataFrame({"total_total": raw_total})
    raw = pd.DataFrame({"group": group})
    dataset = Dataset(raw=raw, scored=scored, questionnaire=q, column_mapping={}, demographic_columns=["group"])

    default_results = se.group_comparison(dataset, group, "Grupp")
    overridden_results = se.group_comparison(dataset, group, "Grupp", score_series=alt_score)

    # reference group is picked by group size alone, so it's identical either way -
    # only the score being compared changes.
    assert overridden_results[0].reference_group == default_results[0].reference_group
    # deliberately constructed with opposite group orderings, so the sign must flip.
    assert default_results[0].cohens_d * overridden_results[0].cohens_d < 0
    assert overridden_results[0].mean_reference == pytest.approx(
        alt_score[group == default_results[0].reference_group].mean(), abs=0.15
    )


def test_all_group_comparisons_passes_score_series_through():
    dataset = _dataset(n=300)
    default = se.all_group_comparisons(dataset)
    alt_score = pd.Series(np.zeros(len(dataset.raw)))  # no variation at all -> d must be exactly 0 everywhere
    overridden = se.all_group_comparisons(dataset, score_series=alt_score)
    assert len(default) == len(overridden)
    assert all(r.cohens_d == 0.0 for r in overridden)


def test_available_fairness_dimensions_and_age_binning():
    rng = np.random.default_rng(5)
    n = 100
    q = Questionnaire(
        plugin_id="demo",
        plugin_version="1.0",
        test_name="Demo",
        full_name="Demo",
        response_scale=ResponseScale(min=0, max=10),
        items=[Item(id="D1", text="x", subscale="total", reverse_scored=False)],
        subscales=[Subscale(id="total", name="Total", item_ids=["D1"], score_range=(0, 10))],
    )
    raw = pd.DataFrame({
        "age": rng.integers(18, 70, n),
        "gender": rng.choice(["Kvinna", "Man"], n),
    })
    scored = pd.DataFrame({"total_total": rng.normal(5, 2, n)})
    dataset = Dataset(raw=raw, scored=scored, questionnaire=q, column_mapping={}, demographic_columns=["age", "gender"])

    dims = se.available_fairness_dimensions(dataset)
    assert set(dims.keys()) == {"gender", "age_group"}
    age_groups = se.age_group_series(dataset)
    assert set(age_groups.unique()) <= {"18-25", "26-65"}


def test_fairness_summary_low_bias_when_no_differences():
    rng = np.random.default_rng(9)
    n = 300
    q = Questionnaire(
        plugin_id="demo",
        plugin_version="1.0",
        test_name="Demo",
        full_name="Demo",
        response_scale=ResponseScale(min=0, max=10),
        items=[Item(id="D1", text="x", subscale="total", reverse_scored=False)],
        subscales=[Subscale(id="total", name="Total", item_ids=["D1"], score_range=(0, 10))],
    )
    raw = pd.DataFrame({"group": rng.choice(["A", "B"], n)})
    scored = pd.DataFrame({"total_total": rng.normal(5, 2, n)})  # independent of group
    dataset = Dataset(raw=raw, scored=scored, questionnaire=q, column_mapping={}, demographic_columns=["group"])

    results = se.all_group_comparisons(dataset)
    summary = se.fairness_summary(results)
    assert summary.bias_level in {"low", "moderate"}  # sampling noise only, should not be "high"
    assert summary.n_comparisons == 1
