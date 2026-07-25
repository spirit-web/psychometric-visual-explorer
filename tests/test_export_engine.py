import zipfile
from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from core import export_engine as ee
from core.data_model import Dataset, Item, Questionnaire, ResponseScale, Subscale


def _dataset(n=300, n_items=8, seed=3, with_outcome=True, with_demographics=True) -> Dataset:
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

    demographic_columns = []
    if with_outcome:
        logit = 1.3 * (theta - 0.5)
        p = 1 / (1 + np.exp(-logit))
        raw["outcome_positive"] = (rng.random(n) < p).astype(int)
    if with_demographics:
        raw["group"] = np.where(rng.random(n) < 0.6, "A", "B")
        demographic_columns = ["group"]

    return Dataset(raw=raw, scored=scored, questionnaire=q, column_mapping={}, demographic_columns=demographic_columns)


def test_dataframe_to_csv_bytes_roundtrips():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    csv_bytes = ee.dataframe_to_csv_bytes(df)
    back = pd.read_csv(BytesIO(csv_bytes))
    assert list(back["a"]) == [1, 2]
    assert list(back["b"]) == ["x", "y"]


def test_anonymized_scored_dataset_drops_respondent_id_and_adds_anon_id():
    dataset = _dataset()
    anon = ee.anonymized_scored_dataset(dataset)
    assert "respondent_id" not in anon.columns
    assert "anon_id" in anon.columns
    assert list(anon["anon_id"]) == list(range(1, len(anon) + 1))


def test_figure_to_png_bytes_produces_valid_png():
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Bar(x=[1, 2], y=[3, 4])])
    png_bytes = ee.figure_to_png_bytes(fig, width=200, height=150)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_figures_to_zip_bytes_contains_all_names():
    import plotly.graph_objects as go

    figures = {"a": go.Figure(data=[go.Bar(x=[1], y=[2])]), "b": go.Figure(data=[go.Bar(x=[1], y=[2])])}
    zip_bytes = ee.figures_to_zip_bytes(figures)
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        names = set(zf.namelist())
    assert names == {"a.png", "b.png"}


def test_build_pdf_report_produces_valid_pdf():
    sections = [ee.ReportSection(title="Test", paragraphs=["Hello"], table=pd.DataFrame({"x": [1]}))]
    pdf_bytes = ee.build_pdf_report("Titel", "Undertitel", sections)
    assert pdf_bytes[:5] == b"%PDF-"


def test_full_report_sections_includes_core_sections():
    dataset = _dataset()
    sections = ee.full_report_sections(dataset, include_ml=False)
    titles = [s.title for s in sections]
    assert "Datasetöversikt" in titles
    assert "Psykometrisk kvalitetskontroll" in titles
    assert "Reliabilitet" in titles


def test_full_analysis_report_pdf_runs_end_to_end():
    dataset = _dataset()
    pdf_bytes = ee.full_analysis_report_pdf(dataset, include_ml=False)
    assert pdf_bytes[:5] == b"%PDF-"
    assert len(pdf_bytes) > 1000


def test_psychometric_summary_pdf_runs():
    dataset = _dataset()
    pdf_bytes = ee.psychometric_summary_pdf(dataset)
    assert pdf_bytes[:5] == b"%PDF-"


def test_decision_support_report_present_when_outcome_available():
    dataset = _dataset(with_outcome=True)
    pdf_bytes = ee.decision_support_report_pdf(dataset)
    assert pdf_bytes is not None
    assert pdf_bytes[:5] == b"%PDF-"


def test_decision_support_report_none_without_outcome():
    dataset = _dataset(with_outcome=False)
    assert ee.decision_support_report_pdf(dataset) is None


def test_fairness_section_none_without_demographics():
    dataset = _dataset(with_demographics=False)
    assert ee.fairness_section(dataset) is None


def test_build_export_zip_contains_requested_parts():
    dataset = _dataset()
    zip_bytes = ee.build_export_zip(
        dataset,
        include_full_report=True,
        include_summary_report=False,
        include_decision_report=True,
        include_figures=False,
        include_raw_data=True,
    )
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        names = set(zf.namelist())
    assert "psykometrisk_rapport.pdf" in names
    assert "beslutsstodsrapport.pdf" in names
    assert "data_anonymiserad.csv" in names
    assert "psykometrisk_sammanfattning.pdf" not in names


def test_key_figures_returns_nonempty_dict_for_typical_dataset():
    dataset = _dataset()
    figures = ee.key_figures(dataset)
    assert "svarsfordelning" in figures
    assert len(figures) >= 1
