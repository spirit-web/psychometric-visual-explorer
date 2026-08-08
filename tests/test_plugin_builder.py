import pandas as pd
import pytest

from core import plugin_engine as pe
from core.data_model import Item, Questionnaire, ResponseScale, Subscale
from core.import_engine import build_dataset


def _sample_questionnaire() -> Questionnaire:
    return Questionnaire(
        plugin_id="demo_test",
        plugin_version="1.0",
        test_name="Demo",
        full_name="Demo Full",
        response_scale=ResponseScale(min=0, max=3, labels={"0": "Aldrig", "3": "Alltid"}),
        items=[
            Item(id="D1", text="Fråga 1", subscale="total", reverse_scored=False),
            Item(id="D2", text="Fråga 2", subscale="total", reverse_scored=True),
        ],
        subscales=[Subscale(id="total", name="Total", item_ids=["D1", "D2"], score_range=(0, 6))],
    )


def test_validate_plugin_id_rejects_bad_ids():
    assert pe.validate_plugin_id("") is not None
    assert pe.validate_plugin_id("Bad-ID!") is not None
    assert pe.validate_plugin_id("1starts_with_digit") is not None
    assert pe.validate_plugin_id("ab") is not None  # too short
    assert pe.validate_plugin_id("good_id_1") is None


def test_save_and_reload_round_trips(tmp_path):
    q = _sample_questionnaire()
    ok, msg = pe.save_plugin(q, plugins_dir=tmp_path)
    assert ok, msg

    reloaded = pe.load_plugin(tmp_path / "demo_test.json")
    assert reloaded is not None
    assert reloaded.model_dump() == q.model_dump()


def test_save_plugin_refuses_overwrite_by_default(tmp_path):
    q = _sample_questionnaire()
    ok1, _ = pe.save_plugin(q, plugins_dir=tmp_path)
    assert ok1
    ok2, msg2 = pe.save_plugin(q, plugins_dir=tmp_path)
    assert not ok2
    assert "finns redan" in msg2

    ok3, _ = pe.save_plugin(q, plugins_dir=tmp_path, overwrite=True)
    assert ok3


def test_save_plugin_rejects_invalid_id(tmp_path):
    q = _sample_questionnaire()
    q2 = pe.duplicate_questionnaire(q, "Not Valid!")
    ok, msg = pe.save_plugin(q2, plugins_dir=tmp_path)
    assert not ok
    assert "gemener" in msg or "Plugin-id" in msg


def test_save_plugin_rejects_empty_items_or_subscales(tmp_path):
    q = _sample_questionnaire()
    data = q.model_dump()
    data["items"] = []
    q_no_items = Questionnaire.model_validate(data)
    ok, msg = pe.save_plugin(q_no_items, plugins_dir=tmp_path)
    assert not ok
    assert "item" in msg.lower()


def test_duplicate_questionnaire_changes_id_and_name_only():
    q = _sample_questionnaire()
    dup = pe.duplicate_questionnaire(q, "demo_test_copy", "Demo (kopia)")
    assert dup.plugin_id == "demo_test_copy"
    assert dup.test_name == "Demo (kopia)"
    assert dup.items == q.items
    assert dup.subscales == q.subscales
    assert dup.response_scale == q.response_scale


def test_blank_questionnaire_has_minimal_valid_structure():
    blank = pe.blank_questionnaire("new_id", "New Test", 1, 5)
    assert blank.plugin_id == "new_id"
    assert len(blank.items) == 1
    assert len(blank.subscales) == 1
    assert blank.response_scale.range == (1, 5)


def test_questionnaire_from_tables_builds_valid_questionnaire():
    items_df = pd.DataFrame(
        [
            {"id": "Q1", "text": "Fråga 1", "subscale": "total", "reverse_scored": False},
            {"id": "Q2", "text": "Fråga 2", "subscale": "total", "reverse_scored": True},
        ]
    )
    subscales_df = pd.DataFrame(
        [{"id": "total", "name": "Total", "item_ids": "Q1, Q2", "score_min": 0, "score_max": 10, "scoring_method": "sum"}]
    )
    cutoffs_df = pd.DataFrame([{"label": "Low", "range_min": 0, "range_max": 5}])

    q, error = pe.questionnaire_from_tables(
        plugin_id="custom_test",
        test_name="Custom Test",
        full_name="Custom Test Full",
        language="sv",
        source_citation=None,
        scale_min=0,
        scale_max=5,
        scale_prompt=None,
        scale_labels={},
        items_df=items_df,
        subscales_df=subscales_df,
        cutoffs_df=cutoffs_df,
    )
    assert error is None
    assert q.plugin_id == "custom_test"
    assert [i.id for i in q.items] == ["Q1", "Q2"]
    assert q.items[1].reverse_scored is True
    assert q.subscales[0].item_ids == ["Q1", "Q2"]
    assert q.cutoffs[0].label == "Low"


def test_questionnaire_from_tables_errors_on_no_items():
    empty_items = pd.DataFrame(columns=["id", "text", "subscale", "reverse_scored"])
    subscales_df = pd.DataFrame([{"id": "total", "name": "Total", "item_ids": "", "score_min": 0, "score_max": 10, "scoring_method": "sum"}])
    q, error = pe.questionnaire_from_tables(
        plugin_id="empty_test",
        test_name="Empty",
        full_name="Empty",
        language="sv",
        source_citation=None,
        scale_min=0,
        scale_max=5,
        scale_prompt=None,
        scale_labels={},
        items_df=empty_items,
        subscales_df=subscales_df,
        cutoffs_df=pd.DataFrame(columns=["label", "range_min", "range_max"]),
    )
    assert q is None
    assert error is not None


def test_apply_draft_to_dataset_rescopes_using_the_same_raw_data():
    q = _sample_questionnaire()  # items D1, D2 (D2 reverse-scored), subscale "total"
    raw = pd.DataFrame(
        {"resp_id": [1, 2], "col_d1": [1, 2], "col_d2": [0, 1], "age": [30, 40]}
    )
    mapping = {"D1": "col_d1", "D2": "col_d2"}
    dataset = build_dataset(raw=raw, questionnaire=q, column_mapping=mapping, demographic_columns=["age"], name="Demo")

    # Draft: Test Builder's "Korta ner testet" dropped D2 - only D1 remains.
    draft_items_df = pd.DataFrame([{"id": "D1", "text": "Fråga 1", "subscale": "total", "reverse_scored": False}])
    draft_subscales_df = pd.DataFrame(
        [{"id": "total", "name": "Total", "item_ids": "D1", "score_min": 0, "score_max": 3, "scoring_method": "sum"}]
    )
    draft, error = pe.questionnaire_from_tables(
        plugin_id="demo_test_utkast",
        test_name="Demo",
        full_name="Demo Full",
        language="sv",
        source_citation=None,
        scale_min=0,
        scale_max=3,
        scale_prompt=None,
        scale_labels={},
        items_df=draft_items_df,
        subscales_df=draft_subscales_df,
        cutoffs_df=pd.DataFrame(columns=["label", "range_min", "range_max"]),
    )
    assert error is None

    new_dataset = pe.apply_draft_to_dataset(dataset, draft)

    # Only D1 stayed in the column mapping - D2 was dropped, not carried over.
    assert list(new_dataset.column_mapping.keys()) == ["D1"]
    # Re-scored against the same raw data, using only the retained item.
    assert new_dataset.scored["total_total"].tolist() == [1, 2]
    # Untouched fields pass through unchanged.
    assert new_dataset.raw is dataset.raw
    assert new_dataset.demographic_columns == ["age"]
    assert new_dataset.questionnaire.plugin_id == "demo_test_utkast"


def test_apply_draft_to_dataset_tolerates_a_newly_added_unmapped_item():
    q = _sample_questionnaire()
    raw = pd.DataFrame({"resp_id": [1, 2], "col_d1": [1, 2], "col_d2": [0, 1]})
    mapping = {"D1": "col_d1", "D2": "col_d2"}
    dataset = build_dataset(raw=raw, questionnaire=q, column_mapping=mapping, name="Demo")

    # Draft adds a brand-new item ("D3") with no corresponding raw column yet.
    draft_items_df = pd.DataFrame(
        [
            {"id": "D1", "text": "Fråga 1", "subscale": "total", "reverse_scored": False},
            {"id": "D2", "text": "Fråga 2", "subscale": "total", "reverse_scored": True},
            {"id": "D3", "text": "Ny egen fråga", "subscale": "total", "reverse_scored": False},
        ]
    )
    draft_subscales_df = pd.DataFrame(
        [{"id": "total", "name": "Total", "item_ids": "D1, D2, D3", "score_min": 0, "score_max": 9, "scoring_method": "sum"}]
    )
    draft, error = pe.questionnaire_from_tables(
        plugin_id="demo_test_utkast",
        test_name="Demo",
        full_name="Demo Full",
        language="sv",
        source_citation=None,
        scale_min=0,
        scale_max=3,
        scale_prompt=None,
        scale_labels={},
        items_df=draft_items_df,
        subscales_df=draft_subscales_df,
        cutoffs_df=pd.DataFrame(columns=["label", "range_min", "range_max"]),
    )
    assert error is None

    # Never raises, even though D3 has no raw column to map.
    new_dataset = pe.apply_draft_to_dataset(dataset, draft)
    assert "D3" not in new_dataset.column_mapping
    assert "D3" not in new_dataset.scored.columns


def test_questionnaire_from_tables_errors_on_no_subscales():
    items_df = pd.DataFrame([{"id": "Q1", "text": "x", "subscale": "total", "reverse_scored": False}])
    empty_subscales = pd.DataFrame(columns=["id", "name", "item_ids", "score_min", "score_max", "scoring_method"])
    q, error = pe.questionnaire_from_tables(
        plugin_id="no_subscale_test",
        test_name="Test",
        full_name="Test",
        language="sv",
        source_citation=None,
        scale_min=0,
        scale_max=5,
        scale_prompt=None,
        scale_labels={},
        items_df=items_df,
        subscales_df=empty_subscales,
        cutoffs_df=pd.DataFrame(columns=["label", "range_min", "range_max"]),
    )
    assert q is None
    assert error is not None
