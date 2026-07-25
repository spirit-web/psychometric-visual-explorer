"""File parsing and Dataset construction for the Import Wizard.

Pure logic only - no Streamlit imports. Never raises on bad input; callers
get either a result or a human-readable error string.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from loguru import logger

from core.data_model import Dataset, Questionnaire

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def read_file(file_bytes: bytes, filename: str) -> tuple[pd.DataFrame | None, str | None]:
    """Parse an uploaded file into a DataFrame. Returns (df, error_message)."""
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".csv":
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(io.BytesIO(file_bytes))
        elif suffix == ".sav":
            return None, (
                "SPSS-filer (.sav) stöds inte ännu. Exportera till CSV eller "
                "Excel och försök igen."
            )
        else:
            return None, f"Filtypen '{suffix}' stöds inte. Använd CSV eller Excel."
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        logger.warning(f"Failed to parse uploaded file {filename}: {exc}")
        return None, f"Kunde inte läsa filen: {exc}"

    if df.empty:
        return None, "Filen innehåller ingen data."
    return df, None


def identify_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """Split columns into a rough 'likely item' vs 'likely demographic/other'
    heuristic for the review step: numeric columns with a small value range
    are treated as candidate item columns."""
    likely_items: list[str] = []
    likely_other: list[str] = []
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        non_null = series.dropna()
        if len(non_null) == 0:
            likely_other.append(col)
            continue
        is_mostly_numeric = non_null.shape[0] >= 0.9 * df[col].dropna().shape[0]
        small_range = (non_null.max() - non_null.min()) <= 10
        looks_integer_like = (non_null % 1 == 0).mean() > 0.95
        if is_mostly_numeric and small_range and looks_integer_like:
            likely_items.append(col)
        else:
            likely_other.append(col)
    return {"likely_items": likely_items, "likely_other": likely_other}


def build_dataset(
    raw: pd.DataFrame,
    questionnaire: Questionnaire,
    column_mapping: dict[str, str],
    demographic_columns: list[str] | None = None,
    name: str | None = None,
) -> Dataset:
    """Build a Dataset: renames mapped columns to canonical item ids, coerces
    to numeric, reverse-scores flagged items, and adds a total-score column
    per subscale."""
    demographic_columns = demographic_columns or []
    scored = raw.copy()

    rename_map = {source_col: item_id for item_id, source_col in column_mapping.items()}
    scored = scored.rename(columns=rename_map)

    scale_min, scale_max = questionnaire.response_scale.range
    for item in questionnaire.items:
        if item.id not in scored.columns:
            continue
        scored[item.id] = pd.to_numeric(scored[item.id], errors="coerce")
        if item.reverse_scored:
            scored[item.id] = (scale_min + scale_max) - scored[item.id]

    for subscale in questionnaire.subscales:
        present_cols = [c for c in subscale.item_ids if c in scored.columns]
        if not present_cols:
            continue
        if subscale.scoring_method == "mean":
            scored[f"{subscale.id}_total"] = scored[present_cols].mean(axis=1)
        else:
            scored[f"{subscale.id}_total"] = scored[present_cols].sum(axis=1, min_count=1)

    return Dataset(
        raw=raw,
        scored=scored,
        questionnaire=questionnaire,
        column_mapping=column_mapping,
        demographic_columns=[c for c in demographic_columns if c in raw.columns],
        name=name or questionnaire.test_name,
    )
