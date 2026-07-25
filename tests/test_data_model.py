import pandas as pd

from core.data_model import Dataset, Item, Questionnaire, ResponseScale, Subscale


def _make_questionnaire() -> Questionnaire:
    return Questionnaire(
        plugin_id="demo",
        plugin_version="1.0",
        test_name="Demo",
        full_name="Demo Test",
        response_scale=ResponseScale(min=0, max=3),
        items=[
            Item(id="D1", text="Item 1", subscale="total", reverse_scored=False),
            Item(id="D2", text="Item 2", subscale="total", reverse_scored=True),
        ],
        subscales=[
            Subscale(id="total", name="Total", item_ids=["D1", "D2"], score_range=(0, 6)),
        ],
    )


def test_questionnaire_derived_properties():
    q = _make_questionnaire()
    assert q.item_ids == ["D1", "D2"]
    assert q.reverse_scored_ids == ["D2"]
    assert q.subscale_for_item("D1") == "total"
    assert q.subscale_for_item("missing") is None


def test_dataset_summary_properties():
    q = _make_questionnaire()
    scored = pd.DataFrame({"D1": [1, 2, None, 3], "D2": [2, 1, 1, None]})
    dataset = Dataset(
        raw=scored.copy(),
        scored=scored,
        questionnaire=q,
        column_mapping={"D1": "D1", "D2": "D2"},
    )
    assert dataset.n == 4
    assert dataset.n_items == 2
    assert dataset.n_subscales == 1
    assert dataset.missing_pct == 25.0

    stats = dataset.to_statistics()
    assert stats.n == 4
    assert stats.item_count == 2
    assert stats.likert_range == (0, 3)
    assert stats.reverse_scored_items == ["D2"]
