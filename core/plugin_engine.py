"""Loads and validates JSON test-plugin definitions, matches an uploaded
dataset's columns against them, and builds/saves new plugin definitions for
Test Builder. Contains no Streamlit imports.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from loguru import logger
from pydantic import ValidationError

from core.data_model import Cutoff, Item, Questionnaire, ResponseScale, Subscale

PLUGINS_DIR = Path(__file__).resolve().parent.parent / "plugins"
PLUGIN_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,49}$")

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


# --- Test Builder: constructing and saving new plugins -------------------------------------------------


def validate_plugin_id(plugin_id: str) -> str | None:
    """Returns an error message if invalid, else None."""
    if not plugin_id:
        return "Plugin-id kan inte vara tomt."
    if not PLUGIN_ID_PATTERN.match(plugin_id):
        return "Plugin-id måste börja med en bokstav och innehålla enbart gemener, siffror och understreck (3-50 tecken)."
    return None


def plugin_path_for_id(plugin_id: str, plugins_dir: Path | str = PLUGINS_DIR) -> Path:
    return Path(plugins_dir) / f"{plugin_id}.json"


def save_plugin(
    questionnaire: Questionnaire, plugins_dir: Path | str = PLUGINS_DIR, overwrite: bool = False
) -> tuple[bool, str]:
    """Serializes and writes a Questionnaire to plugins/<plugin_id>.json.
    Never overwrites an existing file unless overwrite=True. Returns
    (success, message) - never raises."""
    error = validate_plugin_id(questionnaire.plugin_id)
    if error:
        return False, error
    if not questionnaire.items:
        return False, "Testet måste ha minst ett item."
    if not questionnaire.subscales:
        return False, "Testet måste ha minst en delskala."

    path = plugin_path_for_id(questionnaire.plugin_id, plugins_dir)
    if path.exists() and not overwrite:
        return False, f"En plugin med id '{questionnaire.plugin_id}' finns redan. Välj ett annat id."

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(questionnaire.model_dump_json(indent=2), encoding="utf-8")
    except OSError as exc:
        return False, f"Kunde inte spara filen: {exc}"

    logger.info(f"Saved plugin '{questionnaire.plugin_id}' to {path}")
    return True, f"Sparade som {path.name}."


def duplicate_questionnaire(source: Questionnaire, new_plugin_id: str, new_test_name: str | None = None) -> Questionnaire:
    """A copy of `source` with a new plugin_id (and optionally display name)
    - Test Builder's "duplicate test" flow. Does not save to disk."""
    data = source.model_dump()
    data["plugin_id"] = new_plugin_id
    if new_test_name:
        data["test_name"] = new_test_name
    return Questionnaire.model_validate(data)


def blank_questionnaire(plugin_id: str, test_name: str, scale_min: int = 0, scale_max: int = 3) -> Questionnaire:
    """A minimal starter template for Test Builder's "create new from
    scratch" flow: one placeholder item and subscale to seed the editors."""
    return Questionnaire(
        plugin_id=plugin_id,
        plugin_version="1.0",
        test_name=test_name,
        full_name=test_name,
        language="sv",
        response_scale=ResponseScale(min=scale_min, max=scale_max, labels={}, prompt=None),
        items=[Item(id="ITEM_1", text="Ny fråga", subscale="total", reverse_scored=False)],
        subscales=[Subscale(id="total", name="Total", item_ids=["ITEM_1"], score_range=(scale_min, scale_max), scoring_method="sum")],
        cutoffs=[],
    )


def questionnaire_from_tables(
    plugin_id: str,
    test_name: str,
    full_name: str,
    language: str,
    source_citation: str | None,
    scale_min: int,
    scale_max: int,
    scale_prompt: str | None,
    scale_labels: dict[str, str],
    items_df: pd.DataFrame,
    subscales_df: pd.DataFrame,
    cutoffs_df: pd.DataFrame,
) -> tuple[Questionnaire | None, str | None]:
    """Reconstructs a Questionnaire from Test Builder's editable tables
    (as produced by st.data_editor). Returns (questionnaire, error_message);
    never raises on malformed input."""
    try:
        items = []
        for _, row in items_df.iterrows():
            item_id = str(row.get("id", "")).strip()
            if not item_id:
                continue
            items.append(
                Item(
                    id=item_id,
                    text=str(row.get("text", "")).strip(),
                    subscale=str(row.get("subscale", "")).strip(),
                    reverse_scored=bool(row.get("reverse_scored", False)),
                )
            )
        if not items:
            return None, "Lägg till minst ett item."

        subscales = []
        for _, row in subscales_df.iterrows():
            sub_id = str(row.get("id", "")).strip()
            if not sub_id:
                continue
            item_ids = [s.strip() for s in str(row.get("item_ids", "")).split(",") if s.strip()]
            subscales.append(
                Subscale(
                    id=sub_id,
                    name=str(row.get("name", sub_id)).strip() or sub_id,
                    item_ids=item_ids,
                    score_range=(int(row.get("score_min", scale_min)), int(row.get("score_max", scale_max))),
                    scoring_method=str(row.get("scoring_method", "sum")).strip() or "sum",
                )
            )
        if not subscales:
            return None, "Lägg till minst en delskala."

        cutoffs = []
        for _, row in cutoffs_df.iterrows():
            label = str(row.get("label", "")).strip()
            if not label:
                continue
            cutoffs.append(Cutoff(label=label, range=(int(row.get("range_min", 0)), int(row.get("range_max", 0)))))

        questionnaire = Questionnaire(
            plugin_id=plugin_id,
            plugin_version="1.0",
            test_name=test_name,
            full_name=full_name or test_name,
            language=language or "sv",
            response_scale=ResponseScale(min=scale_min, max=scale_max, labels=scale_labels, prompt=scale_prompt),
            items=items,
            subscales=subscales,
            cutoffs=cutoffs,
            source_citation=source_citation or None,
        )
        return questionnaire, None
    except (ValidationError, ValueError, TypeError) as exc:
        return None, f"Ogiltig testdefinition: {exc}"
