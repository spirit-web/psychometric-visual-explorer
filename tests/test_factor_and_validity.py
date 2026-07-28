import numpy as np
import pandas as pd
import pytest

from core import stats_engine as se
from core.data_model import Dataset, Item, Questionnaire, ResponseScale, Subscale


def _single_factor_dataset(n=400, n_items=8, seed=1) -> Dataset:
    rng = np.random.default_rng(seed)
    q = Questionnaire(
        plugin_id="demo1",
        plugin_version="1.0",
        test_name="Demo1",
        full_name="Demo Single Factor",
        response_scale=ResponseScale(min=0, max=3),
        items=[Item(id=f"D{i}", text=f"Item {i}", subscale="total", reverse_scored=False) for i in range(1, n_items + 1)],
        subscales=[Subscale(id="total", name="Total", item_ids=[f"D{i}" for i in range(1, n_items + 1)], score_range=(0, 3 * n_items))],
        source_citation="Demo et al. (2020)",
    )
    theta = rng.normal(0, 1, size=n)
    data = {}
    for i in range(1, n_items + 1):
        continuous = 0.75 * theta + np.sqrt(1 - 0.75**2) * rng.normal(0, 1, size=n)
        data[f"D{i}"] = np.clip(np.digitize(continuous, [-1, 0, 1]), 0, 3).astype(float)
    raw = pd.DataFrame(data)
    convergent = 0.6 * theta + np.sqrt(1 - 0.6**2) * rng.normal(0, 1, size=n)
    raw["criterion_convergent"] = 50 + 15 * convergent
    raw["criterion_discriminant"] = 50 + 15 * rng.normal(0, 1, size=n)
    scored = raw.copy()
    scored["total_total"] = scored[[f"D{i}" for i in range(1, n_items + 1)]].sum(axis=1)
    return Dataset(raw=raw, scored=scored, questionnaire=q, column_mapping={}, demographic_columns=[])


def _two_factor_dataset(n=400, seed=2) -> Dataset:
    rng = np.random.default_rng(seed)
    items = []
    for i in range(1, 6):
        items.append(Item(id=f"A{i}", text=f"A item {i}", subscale="A", reverse_scored=False))
    for i in range(1, 6):
        items.append(Item(id=f"B{i}", text=f"B item {i}", subscale="B", reverse_scored=False))
    q = Questionnaire(
        plugin_id="demo2",
        plugin_version="1.0",
        test_name="Demo2",
        full_name="Demo Two Factor",
        response_scale=ResponseScale(min=1, max=5),
        items=items,
        subscales=[
            Subscale(id="A", name="Factor A", item_ids=[f"A{i}" for i in range(1, 6)], score_range=(5, 25)),
            Subscale(id="B", name="Factor B", item_ids=[f"B{i}" for i in range(1, 6)], score_range=(5, 25)),
        ],
    )
    factor_corr = np.array([[1.0, 0.2], [0.2, 1.0]])
    scores = rng.multivariate_normal([0, 0], factor_corr, size=n)
    theta_a, theta_b = scores[:, 0], scores[:, 1]
    data = {}
    for i in range(1, 6):
        continuous = 0.7 * theta_a + np.sqrt(1 - 0.7**2) * rng.normal(0, 1, size=n)
        data[f"A{i}"] = np.clip(np.digitize(continuous, [-1.28, -0.52, 0.52, 1.28]) + 1, 1, 5).astype(float)
    for i in range(1, 6):
        continuous = 0.7 * theta_b + np.sqrt(1 - 0.7**2) * rng.normal(0, 1, size=n)
        data[f"B{i}"] = np.clip(np.digitize(continuous, [-1.28, -0.52, 0.52, 1.28]) + 1, 1, 5).astype(float)
    raw = pd.DataFrame(data)
    scored = raw.copy()
    scored["A_total"] = scored[[f"A{i}" for i in range(1, 6)]].sum(axis=1)
    scored["B_total"] = scored[[f"B{i}" for i in range(1, 6)]].sum(axis=1)
    return Dataset(raw=raw, scored=scored, questionnaire=q, column_mapping={}, demographic_columns=[])


def test_parallel_analysis_suggests_one_factor_for_unidimensional_data():
    dataset = _single_factor_dataset()
    result = se.parallel_analysis(dataset)
    assert result.suggested_n_factors == 1
    assert result.eigenvalues[0] > result.simulated_eigenvalues[0]


def test_parallel_analysis_suggests_two_factors_for_two_factor_data():
    dataset = _two_factor_dataset()
    result = se.parallel_analysis(dataset)
    assert result.suggested_n_factors == 2


def test_efa_fit_single_factor_has_no_phi_and_reasonable_loadings():
    dataset = _single_factor_dataset()
    efa = se.efa_fit(dataset, 1)
    assert efa is not None
    assert efa.phi is None
    assert efa.loadings.shape == (8, 1)
    assert (efa.loadings["F1"] > 0.4).all()
    assert (efa.communalities > 0).all()


def test_efa_fit_two_factor_recovers_simple_structure_and_phi():
    dataset = _two_factor_dataset()
    efa = se.efa_fit(dataset, 2)
    assert efa is not None
    assert efa.phi is not None
    assert efa.phi.shape == (2, 2)
    # each item should load much more strongly on its own factor
    a_items = [f"A{i}" for i in range(1, 6)]
    b_items = [f"B{i}" for i in range(1, 6)]
    max_col_for_a = efa.loadings.loc[a_items].abs().mean().idxmax()
    max_col_for_b = efa.loadings.loc[b_items].abs().mean().idxmax()
    assert max_col_for_a != max_col_for_b


def test_factor_item_groups_assigns_each_item_to_its_own_factor():
    dataset = _two_factor_dataset()
    efa = se.efa_fit(dataset, 2)
    groups = se.factor_item_groups(efa, dataset.questionnaire)
    assert len(groups) == 2
    by_factor = {name: [iid for iid, _text, _loading in items] for name, items in groups}
    a_items = {f"A{i}" for i in range(1, 6)}
    b_items = {f"B{i}" for i in range(1, 6)}
    groups_as_sets = list(by_factor.values())
    assert {*groups_as_sets[0]} in (a_items, b_items)
    assert {*groups_as_sets[1]} in (a_items, b_items)
    assert groups_as_sets[0] != groups_as_sets[1]


def test_efa_fit_indices_are_within_plausible_ranges():
    dataset = _single_factor_dataset(n=500)
    efa = se.efa_fit(dataset, 1)
    assert efa.fit.rmsea is not None
    assert 0 <= efa.fit.rmsea <= 1
    assert efa.fit.srmr is not None
    assert 0 <= efa.fit.srmr <= 1
    if efa.fit.cfi is not None:
        assert efa.fit.cfi <= 1.05  # allow tiny numerical slack above 1


def test_efa_fit_returns_none_with_too_few_items():
    dataset = _single_factor_dataset(n_items=2)
    assert se.efa_fit(dataset, 1) is None


def test_sampling_adequacy_reasonable_for_correlated_items():
    dataset = _single_factor_dataset(n=500)
    adequacy = se.sampling_adequacy(dataset)
    assert adequacy.kmo is not None
    assert 0 <= adequacy.kmo <= 1
    assert adequacy.bartlett_p is not None
    assert adequacy.bartlett_p < 0.05


def test_criterion_validity_recovers_simulated_correlation():
    dataset = _single_factor_dataset(n=500)
    cv = se.criterion_validity(dataset)
    assert cv.convergent_r is not None
    assert cv.convergent_r > 0.4
    assert cv.discriminant_r is not None
    assert abs(cv.discriminant_r) < 0.3


def test_criterion_validity_none_when_columns_missing():
    dataset = _two_factor_dataset()
    cv = se.criterion_validity(dataset)
    assert cv.convergent_r is None
    assert cv.discriminant_r is None


def test_validity_overview_has_five_sources_with_expected_keys():
    dataset = _single_factor_dataset(n=500)
    sources = se.validity_overview(dataset, response_process_status="moderate", consequences_status="limited")
    keys = {s.key for s in sources}
    assert keys == {"content", "response_processes", "internal_structure", "relations", "consequences"}
    by_key = {s.key: s for s in sources}
    assert by_key["response_processes"].status == "moderate"
    assert by_key["consequences"].status == "limited"
    assert by_key["content"].status == "strong"  # source_citation is set


def test_validity_status_counts_sum_to_source_count():
    dataset = _single_factor_dataset(n=500)
    sources = se.validity_overview(dataset)
    counts = se.validity_status_counts(sources)
    assert sum(counts.values()) == len(sources)


def test_validity_aggregate_status_four_strong_is_good():
    counts = {"strong": 4, "moderate": 0, "limited": 0, "none": 1}
    assert se.validity_aggregate_status(counts) == ("good", "Bra")


def test_validity_aggregate_status_mostly_undocumented_is_data_saknas_not_mattlig():
    # 2 strong + 3 none used to fall into the "strong >= 2" bucket and read as
    # "Måttlig" (moderate evidence) even though most sources are simply
    # undocumented, not weak - this is the exact bug Fas 6 fixes.
    counts = {"strong": 2, "moderate": 0, "limited": 0, "none": 3}
    assert se.validity_aggregate_status(counts) == ("warning", "Data saknas")


def test_validity_aggregate_status_two_strong_with_documented_weak_is_mattlig():
    counts = {"strong": 2, "moderate": 2, "limited": 1, "none": 0}
    assert se.validity_aggregate_status(counts) == ("warning", "Måttlig")


def test_validity_aggregate_status_weak_evidence_is_bor_starkas():
    counts = {"strong": 0, "moderate": 1, "limited": 2, "none": 2}
    assert se.validity_aggregate_status(counts) == ("bad", "Bör stärkas")
