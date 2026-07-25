import numpy as np
import pandas as pd
import pytest

from core import stats_engine as se
from core.data_model import Dataset, Item, Questionnaire, ResponseScale, Subscale


def _make_dataset(n=300, seed=0, n_items=6, with_retest=True) -> Dataset:
    rng = np.random.default_rng(seed)
    q = Questionnaire(
        plugin_id="demo",
        plugin_version="1.0",
        test_name="Demo",
        full_name="Demo Test",
        response_scale=ResponseScale(min=0, max=3),
        items=[
            Item(id=f"D{i}", text=f"Item {i}", subscale="total", reverse_scored=False)
            for i in range(1, n_items + 1)
        ],
        subscales=[
            Subscale(id="total", name="Total", item_ids=[f"D{i}" for i in range(1, n_items + 1)], score_range=(0, 3 * n_items)),
        ],
    )
    theta = rng.normal(0, 1, size=n)
    data = {"respondent_id": np.arange(n)}
    loadings = {}
    for i in range(1, n_items + 1):
        # Make the last item a deliberately weak/noise item (low item-total r)
        loading = 0.05 if i == n_items else 0.75
        loadings[f"D{i}"] = loading
        continuous = loading * theta + np.sqrt(1 - loading**2) * rng.normal(0, 1, size=n)
        data[f"D{i}"] = np.clip(np.digitize(continuous, [-1, 0, 1]), 0, 3).astype(float)
    raw = pd.DataFrame(data)
    scored = raw.copy()

    if with_retest:
        stability = 0.85
        theta_t2 = stability * theta + np.sqrt(1 - stability**2) * rng.normal(0, 1, size=n)
        in_retest = rng.random(n) < 0.4
        for i in range(1, n_items + 1):
            loading = loadings[f"D{i}"]
            continuous = loading * theta_t2 + np.sqrt(1 - loading**2) * rng.normal(0, 1, size=n)
            vals = np.clip(np.digitize(continuous, [-1, 0, 1]), 0, 3).astype(float)
            vals[~in_retest] = np.nan
            raw[f"D{i}_t2"] = vals

    scored["total_total"] = scored[[f"D{i}" for i in range(1, n_items + 1)]].sum(axis=1)
    return Dataset(raw=raw, scored=scored, questionnaire=q, column_mapping={}, demographic_columns=[])


def test_alpha_confidence_interval_brackets_the_point_estimate():
    dataset = _make_dataset(n=500)
    snapshot = se.reliability_snapshot(dataset)
    lower, upper = se.alpha_confidence_interval(snapshot.alpha, snapshot.n, snapshot.n_items)
    assert lower < snapshot.alpha < upper


def test_item_correlation_matrix_is_symmetric_with_unit_diagonal():
    dataset = _make_dataset()
    corr = se.item_correlation_matrix(dataset)
    assert corr.shape == (6, 6)
    assert np.allclose(np.diag(corr.values), 1.0)
    assert np.allclose(corr.values, corr.values.T)


def test_weak_item_has_lower_item_total_r_than_strong_items():
    dataset = _make_dataset()
    items = {i.item_id: i for i in se.item_level_table(dataset)}
    weak_r = items["D6"].item_total_r
    strong_rs = [items[f"D{i}"].item_total_r for i in range(1, 6)]
    assert weak_r < min(strong_rs)


def test_removing_weak_item_would_raise_alpha_removing_strong_item_would_lower_it():
    """Core internal-consistency check from the Sprint 3 done-when criterion:
    alpha-if-deleted must move in the direction the item's quality implies."""
    dataset = _make_dataset()
    overall = se.reliability_snapshot(dataset)
    items = {i.item_id: i for i in se.item_level_table(dataset)}

    # The deliberately weak item (near-zero loading) should IMPROVE alpha if dropped.
    assert items["D6"].alpha_if_deleted > overall.alpha
    # A normal, well-correlated item should WORSEN alpha if dropped.
    assert items["D1"].alpha_if_deleted < overall.alpha


def test_mcdonald_omega_within_unit_bounds_and_close_to_alpha():
    dataset = _make_dataset(n=500)
    alpha = se.reliability_snapshot(dataset).alpha
    omega = se.mcdonald_omega(dataset)
    assert omega is not None
    assert 0 <= omega <= 1
    assert abs(omega - alpha) < 0.15


def test_split_half_spearman_brown_at_least_raw_r():
    dataset = _make_dataset(n=500)
    result = se.split_half_reliability(dataset)
    assert result.r_raw is not None
    # Spearman-Brown correction should raise reliability relative to the raw half-test r
    assert result.spearman_brown >= result.r_raw


def test_test_retest_reliability_recovers_simulated_stability():
    dataset = _make_dataset(n=500, with_retest=True)
    result = se.test_retest_reliability(dataset)
    assert result.r is not None
    assert result.n > 0
    # simulated at 0.85 trait stability - total-score r should be positive and substantial
    assert result.r > 0.4


def test_test_retest_reliability_returns_none_without_retest_columns():
    dataset = _make_dataset(with_retest=False)
    result = se.test_retest_reliability(dataset)
    assert result.r is None
    assert result.n == 0


def test_alpha_interpretation_labels():
    assert se.alpha_interpretation(0.95) == "Utmärkt"
    assert se.alpha_interpretation(0.85) == "Bra"
    assert se.alpha_interpretation(0.75) == "Acceptabelt"
    assert se.alpha_interpretation(0.65) == "Tveksamt"
    assert se.alpha_interpretation(0.40) == "Otillräckligt"
    assert se.alpha_interpretation(None) == "–"
