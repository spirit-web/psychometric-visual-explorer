import pandas as pd

from core.data_model import Item, Questionnaire, ResponseScale, Subscale
from core.import_engine import build_dataset, identify_column_types, read_file


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


def test_read_file_parses_csv_bytes():
    csv_bytes = b"col_a,col_b\n1,2\n3,4\n"
    df, error = read_file(csv_bytes, "data.csv")
    assert error is None
    assert list(df.columns) == ["col_a", "col_b"]
    assert len(df) == 2


def test_read_file_rejects_unsupported_extension():
    df, error = read_file(b"whatever", "data.sav")
    assert df is None
    assert "stöds inte" in error


def test_read_file_reports_parse_errors_without_raising():
    df, error = read_file(b"", "empty.csv")
    assert df is None
    assert error is not None


def test_identify_column_types_splits_numeric_from_other():
    df = pd.DataFrame({
        "item_1": [0, 1, 2, 3],
        "item_2": [1, 1, 0, 2],
        "gender": ["Kvinna", "Man", "Kvinna", "Annat"],
    })
    result = identify_column_types(df)
    assert set(result["likely_items"]) == {"item_1", "item_2"}
    assert result["likely_other"] == ["gender"]


def test_build_dataset_reverse_scores_and_sums_subscale():
    q = _make_questionnaire()
    raw = pd.DataFrame({"col_d1": [0, 1, 2], "col_d2": [0, 1, 2], "gender": ["A", "B", "A"]})
    mapping = {"D1": "col_d1", "D2": "col_d2"}

    dataset = build_dataset(raw, q, mapping, demographic_columns=["gender"])

    # D2 is reverse-scored on a 0-3 scale: 3 - value
    assert dataset.scored["D2"].tolist() == [3, 2, 1]
    assert dataset.scored["D1"].tolist() == [0, 1, 2]
    assert dataset.scored["total_total"].tolist() == [3, 3, 3]
    assert dataset.demographic_columns == ["gender"]
    assert dataset.n == 3
