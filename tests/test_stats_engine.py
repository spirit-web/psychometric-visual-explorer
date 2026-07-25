import numpy as np
import pandas as pd
import pytest

from core import stats_engine as se
from core.data_model import Dataset, Item, Questionnaire, ResponseScale, Subscale


def _make_dataset(n=200, seed=0) -> Dataset:
    rng = np.random.default_rng(seed)
    q = Questionnaire(
        plugin_id="demo",
        plugin_version="1.0",
        test_name="Demo",
        full_name="Demo Test",
        response_scale=ResponseScale(min=0, max=3),
        items=[
            Item(id=f"D{i}", text=f"Item {i}", subscale="total", reverse_scored=False)
            for i in range(1, 6)
        ],
        subscales=[
            Subscale(id="total", name="Total", item_ids=[f"D{i}" for i in range(1, 6)], score_range=(0, 15)),
        ],
    )
    theta = rng.normal(0, 1, size=n)
    data = {"respondent_id": np.arange(n)}
    for i in range(1, 6):
        continuous = 0.7 * theta + rng.normal(0, 1, size=n)
        data[f"D{i}"] = np.clip(np.digitize(continuous, [-1, 0, 1]), 0, 3)
    data["gender"] = rng.choice(["Kvinna", "Man"], size=n)
    raw = pd.DataFrame(data)
    scored = raw.copy()
    for i in range(1, 6):
        scored[f"D{i}"] = scored[f"D{i}"].astype(float)
    scored["total_total"] = scored[[f"D{i}" for i in range(1, 6)]].sum(axis=1)
    return Dataset(raw=raw, scored=scored, questionnaire=q, column_mapping={}, demographic_columns=["gender"])


def test_response_distribution_sums_to_100():
    dataset = _make_dataset()
    dist = se.response_distribution(dataset)
    assert set(dist["value"]) == {0, 1, 2, 3}
    assert pytest.approx(dist["pct"].sum(), abs=0.01) == 100.0


def test_missing_by_item_reports_zero_when_no_missing():
    dataset = _make_dataset()
    missing = se.missing_by_item(dataset)
    assert (missing == 0).all()


def test_reliability_snapshot_reasonable_alpha():
    dataset = _make_dataset(n=500)
    snapshot = se.reliability_snapshot(dataset)
    assert snapshot.n_items == 5
    assert snapshot.alpha is not None
    assert 0 <= snapshot.alpha <= 1


def test_qc_constant_items_flags_zero_variance_item():
    dataset = _make_dataset()
    dataset.scored["D1"] = 2  # make item constant
    check = se.qc_constant_items(dataset)
    assert check.status == "bad"
    assert "D1" in check.affected


def test_qc_duplicates_detects_duplicate_respondent_ids():
    dataset = _make_dataset()
    dataset.raw.loc[1, "respondent_id"] = dataset.raw.loc[0, "respondent_id"]
    check = se.qc_duplicates(dataset)
    assert check.status == "bad"


def test_straightlining_flags_all_same_answer_respondent():
    dataset = _make_dataset()
    item_cols = [f"D{i}" for i in range(1, 6)]
    dataset.scored.loc[0, item_cols] = 1
    patterns = se.straightlining_patterns(dataset)
    assert patterns.all_same_answer >= 1


def test_quality_summary_status_reflects_worst_check():
    dataset = _make_dataset()
    dataset.scored["D1"] = 2  # constant -> "bad"
    summary = se.quality_summary(dataset)
    assert summary.status == "bad"
    assert summary.n_flagged_checks >= 1
