import math

import numpy as np
import pandas as pd
import pytest

from core import stats_engine as se
from core.data_model import Dataset, Item, Questionnaire, ResponseScale, Subscale


def _dataset(n=500, n_items=8, seed=7, with_retest=True) -> Dataset:
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
    loadings = {}
    for i in range(1, n_items + 1):
        loading = 0.75
        loadings[f"D{i}"] = loading
        continuous = loading * theta + np.sqrt(1 - loading**2) * rng.normal(0, 1, size=n)
        data[f"D{i}"] = np.clip(np.digitize(continuous, [-1, 0, 1]), 0, 3).astype(float)
    raw = pd.DataFrame(data)
    scored = raw.copy()
    scored["total_total"] = scored[[f"D{i}" for i in range(1, n_items + 1)]].sum(axis=1)

    if with_retest:
        stability = 0.85
        theta_t2 = stability * theta + np.sqrt(1 - stability**2) * rng.normal(0, 1, size=n)
        in_retest = rng.random(n) < 0.3
        for i in range(1, n_items + 1):
            loading = loadings[f"D{i}"]
            continuous = loading * theta_t2 + np.sqrt(1 - loading**2) * rng.normal(0, 1, size=n)
            vals = np.clip(np.digitize(continuous, [-1, 0, 1]), 0, 3).astype(float)
            vals[~in_retest] = np.nan
            raw[f"D{i}_t2"] = vals

    return Dataset(raw=raw, scored=scored, questionnaire=q, column_mapping={}, demographic_columns=[])


def test_t_score_formula():
    for z in [-2.0, -0.5, 0.0, 0.92, 1.75, 3.1]:
        assert se.z_to_t(z) == pytest.approx(50 + 10 * z)


def test_z_score_recovers_from_raw():
    z = se.raw_to_z(raw=15, mean=10, sd=5)
    assert z == pytest.approx(1.0)
    assert se.z_to_t(z) == pytest.approx(60.0)


def test_percentile_of_mean_is_50():
    assert se.z_to_percentile(0.0) == pytest.approx(50.0, abs=0.1)


def test_stanine_boundaries():
    assert se.z_to_stanine(-3.0) == 1
    assert se.z_to_stanine(0.0) == 5
    assert se.z_to_stanine(3.0) == 9
    # exactly on a cut boundary should fall into the band above
    assert se.z_to_stanine(0.25) == 6


def test_conversion_table_spans_full_score_range():
    dataset = _dataset(n_items=7)
    table = se.conversion_table(dataset)
    assert table["Råpoäng"].min() == 0
    assert table["Råpoäng"].max() == 21  # 3 * 7 items
    assert len(table) == 22
    # T-score formula must hold for every row
    for _, row in table.iterrows():
        assert row["T-poäng"] == pytest.approx(50 + 10 * row["Z-poäng"], abs=0.05)


def test_measurement_error_sem_formula():
    dataset = _dataset(n=600)
    result = se.measurement_error(dataset)
    assert result.sem == pytest.approx(result.sd * math.sqrt(1 - result.alpha))


def test_confidence_interval_uses_1_96():
    lo, hi = se.confidence_interval(raw_score=20, sem=2.0)
    assert lo == pytest.approx(20 - 1.96 * 2.0)
    assert hi == pytest.approx(20 + 1.96 * 2.0)


def test_minimum_reliable_change_formula():
    mrc = se.minimum_reliable_change(sem=2.0)
    assert mrc == pytest.approx(1.96 * math.sqrt(2) * 2.0)


def test_precision_label_thresholds():
    assert se.precision_label(sem=1.0, sd=5.0) == "Hög"  # ratio 0.2
    assert se.precision_label(sem=2.2, sd=5.0) == "Måttlig"  # ratio 0.44
    assert se.precision_label(sem=3.0, sd=5.0) == "Låg"  # ratio 0.6


def test_person_raw_score_matches_reliable_change_t1():
    dataset = _dataset(n=500, with_retest=True)
    rci_results = se.reliable_change_index(dataset)
    assert len(rci_results) > 0
    first = rci_results[0]
    looked_up = se.person_raw_score(dataset, int(first.person_id))
    assert looked_up == pytest.approx(first.score_t1)


def test_reliable_change_index_flags_large_diffs_as_reliable():
    dataset = _dataset(n=500, with_retest=True)
    results = se.reliable_change_index(dataset)
    for r in results:
        expected_reliable = abs(r.rci) >= 1.96
        assert r.reliable == expected_reliable
        assert r.rci == pytest.approx(r.diff / r.se_diff)


def test_all_person_confidence_intervals_match_formula():
    dataset = _dataset(n=200, with_retest=False)
    table = se.all_person_confidence_intervals(dataset)
    me = se.measurement_error(dataset)
    row = table.iloc[0]
    assert row["95% KI nedre"] == pytest.approx(row["Observerad poäng"] - 1.96 * me.sem, abs=0.05)
    assert row["95% KI övre"] == pytest.approx(row["Observerad poäng"] + 1.96 * me.sem, abs=0.05)


def _dataset_with_reverse_item(n=50, seed=1) -> Dataset:
    rng = np.random.default_rng(seed)
    q = Questionnaire(
        plugin_id="demo_rev",
        plugin_version="1.0",
        test_name="Demo",
        full_name="Demo Test",
        response_scale=ResponseScale(min=0, max=3),
        items=[
            Item(id="D1", text="Item 1", subscale="total", reverse_scored=False),
            Item(id="D2", text="Item 2", subscale="total", reverse_scored=True),
            Item(id="D3", text="Item 3", subscale="total", reverse_scored=False),
        ],
        subscales=[Subscale(id="total", name="Total", item_ids=["D1", "D2", "D3"], score_range=(0, 9))],
    )
    raw = pd.DataFrame({"respondent_id": np.arange(1, n + 1), "D1": rng.integers(0, 4, n), "D2": rng.integers(0, 4, n), "D3": rng.integers(0, 4, n)})
    scored = raw.copy()
    return Dataset(raw=raw, scored=scored, questionnaire=q, column_mapping={})


def test_score_manual_responses_applies_reverse_scoring():
    dataset = _dataset_with_reverse_item()
    # D2 is reverse-scored (0-3 scale): raw 1 -> reverse-scored 2
    score = se.score_manual_responses(dataset, {"D1": 3, "D2": 1, "D3": 0})
    assert score == pytest.approx(3 + 2 + 0)


def test_score_manual_responses_none_when_no_answers():
    dataset = _dataset_with_reverse_item()
    assert se.score_manual_responses(dataset, {}) is None


def test_score_manual_responses_ignores_missing_items():
    dataset = _dataset_with_reverse_item()
    score = se.score_manual_responses(dataset, {"D1": 2})
    assert score == pytest.approx(2)


def test_cutoff_category_finds_matching_band():
    from core.data_model import Cutoff

    q = Questionnaire(
        plugin_id="demo_cut",
        plugin_version="1.0",
        test_name="Demo",
        full_name="Demo Test",
        response_scale=ResponseScale(min=0, max=3),
        items=[Item(id="D1", text="Item 1", subscale="total", reverse_scored=False)],
        subscales=[Subscale(id="total", name="Total", item_ids=["D1"], score_range=(0, 21))],
        cutoffs=[
            Cutoff(label="Minimal", range=(0, 4)),
            Cutoff(label="Mild", range=(5, 9)),
            Cutoff(label="Måttlig", range=(10, 14)),
            Cutoff(label="Svår", range=(15, 21)),
        ],
    )
    assert se.cutoff_category(q, 3) == "Minimal"
    assert se.cutoff_category(q, 12) == "Måttlig"
    assert se.cutoff_category(q, 21) == "Svår"


def test_cutoff_category_none_when_no_cutoffs_defined():
    q = Questionnaire(
        plugin_id="demo_nocut",
        plugin_version="1.0",
        test_name="Demo",
        full_name="Demo Test",
        response_scale=ResponseScale(min=0, max=3),
        items=[Item(id="D1", text="Item 1", subscale="total", reverse_scored=False)],
        subscales=[Subscale(id="total", name="Total", item_ids=["D1"], score_range=(0, 3))],
    )
    assert se.cutoff_category(q, 2) is None
