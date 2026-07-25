"""Loads and validates JSON test-plugin definitions, and matches an uploaded
dataset's columns against them. Contains no Streamlit imports.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger
from pydantic import ValidationError

from core.data_model import Questionnaire

PLUGINS_DIR = Path(__file__).resolve().parent.parent / "plugins"

# Fraction of a plugin's items that must be found among the dataset's columns
# for it to be considered a match.
MATCH_THRESHOLD = 0.6


def load_plugin(path: Path) -> Questionnaire | None:
    """Load and validate a single plugin JSON file. Returns None on failure,
    logging the reason - a malformed plugin must never crash the app."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return Questionnaire.model_validate(raw)
    except (json.JSONDecodeError, ValidationError, OSError) as exc:
        logger.warning(f"Skipping invalid plugin {path.name}: {exc}")
        return None


def load_all_plugins(plugins_dir: Path | str = PLUGINS_DIR) -> dict[str, Questionnaire]:
    """Load every *.json plugin in the given directory (files ending in
    .example.json are reference templates and are skipped)."""
    plugins_dir = Path(plugins_dir)
    plugins: dict[str, Questionnaire] = {}
    if not plugins_dir.exists():
        logger.warning(f"Plugins directory not found: {plugins_dir}")
        return plugins

    for path in sorted(plugins_dir.glob("*.json")):
        if path.name.endswith(".example.json"):
            continue
        questionnaire = load_plugin(path)
        if questionnaire is not None:
            plugins[questionnaire.plugin_id] = questionnaire
    return plugins


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def match_plugin(
    columns: list[str], plugins: dict[str, Questionnaire]
) -> tuple[Questionnaire, dict[str, str]] | None:
    """Find the best-matching plugin for a set of dataset column names.

    Matching is done on normalized item ids (case/underscore/dash-insensitive
    substring match). Returns (questionnaire, column_mapping) for the best
    match at or above MATCH_THRESHOLD, or None if nothing matches well enough.
    """
    normalized_columns = {_normalize(c): c for c in columns}

    best: tuple[Questionnaire, dict[str, str], float] | None = None
    for questionnaire in plugins.values():
        mapping: dict[str, str] = {}
        for item in questionnaire.items:
            norm_id = _normalize(item.id)
            match = normalized_columns.get(norm_id)
            if match is None:
                # fall back to substring match either direction
                for norm_col, original_col in normalized_columns.items():
                    if norm_id and (norm_id in norm_col or norm_col in norm_id):
                        match = original_col
                        break
            if match is not None:
                mapping[item.id] = match

        score = len(mapping) / len(questionnaire.items) if questionnaire.items else 0
        if score >= MATCH_THRESHOLD and (best is None or score > best[2]):
            best = (questionnaire, mapping, score)

    if best is None:
        return None
    questionnaire, mapping, score = best
    logger.info(f"Matched plugin '{questionnaire.plugin_id}' ({score:.0%} of items)")
    return questionnaire, mapping


def match_plugin_to_dataframe(
    df: pd.DataFrame, plugins: dict[str, Questionnaire]
) -> tuple[Questionnaire, dict[str, str]] | None:
    return match_plugin(list(df.columns), plugins)
