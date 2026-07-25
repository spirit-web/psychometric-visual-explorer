"""Core domain objects: Questionnaire (from a plugin), Dataset, Statistics.

Pure data shapes only - no statistics are calculated here. See
core/plugin_engine.py for loading/matching and core/stats_engine.py for
calculations that consume these objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field


class ResponseScale(BaseModel):
    type: str = "likert"
    min: int
    max: int
    labels: dict[str, str] = Field(default_factory=dict)
    prompt: str | None = None

    @property
    def range(self) -> tuple[int, int]:
        return (self.min, self.max)


class Item(BaseModel):
    id: str
    text: str
    subscale: str
    reverse_scored: bool = False


class Subscale(BaseModel):
    id: str
    name: str
    item_ids: list[str]
    score_range: tuple[int, int]
    scoring_method: str = "sum"


class Cutoff(BaseModel):
    label: str
    range: tuple[int, int]


class NormReference(BaseModel):
    population: str | None = None
    sample_size: int | None = None
    standardization_year: int | None = None
    note: str | None = None


class Questionnaire(BaseModel):
    """A parsed, validated test-plugin definition."""

    plugin_id: str
    plugin_version: str
    test_name: str
    full_name: str
    language: str = "sv"
    response_scale: ResponseScale
    items: list[Item]
    subscales: list[Subscale]
    cutoffs: list[Cutoff] = Field(default_factory=list)
    norm_reference: NormReference | None = None
    source_citation: str | None = None

    @property
    def item_ids(self) -> list[str]:
        return [item.id for item in self.items]

    @property
    def reverse_scored_ids(self) -> list[str]:
        return [item.id for item in self.items if item.reverse_scored]

    def subscale_for_item(self, item_id: str) -> str | None:
        for subscale in self.subscales:
            if item_id in subscale.item_ids:
                return subscale.id
        return None

    def get_subscale(self, subscale_id: str) -> Subscale | None:
        return next((s for s in self.subscales if s.id == subscale_id), None)


@dataclass
class Statistics:
    """Lightweight summary bundle shown on KPI cards (Dataset Overview etc.).

    Populated by core/stats_engine.py; kept here as the shared shape so pages
    never need to know how the numbers were derived.
    """

    n: int = 0
    item_count: int = 0
    subscale_count: int = 0
    missing_pct: float = 0.0
    likert_range: tuple[int, int] = (0, 0)
    reverse_scored_items: list[str] = field(default_factory=list)
    collected_at: datetime | None = None


@dataclass
class Dataset:
    """A respondent dataset mapped to a Questionnaire, ready for analysis."""

    raw: pd.DataFrame
    scored: pd.DataFrame
    questionnaire: Questionnaire
    column_mapping: dict[str, str]
    demographic_columns: list[str] = field(default_factory=list)
    name: str = "Dataset"
    imported_at: datetime = field(default_factory=datetime.now)

    @property
    def n(self) -> int:
        return len(self.scored)

    @property
    def n_items(self) -> int:
        return len(self.questionnaire.items)

    @property
    def n_subscales(self) -> int:
        return len(self.questionnaire.subscales)

    @property
    def missing_pct(self) -> float:
        cols = [c for c in self.questionnaire.item_ids if c in self.scored.columns]
        if not cols:
            return 0.0
        return float(self.scored[cols].isna().mean().mean() * 100)

    def to_statistics(self) -> Statistics:
        return Statistics(
            n=self.n,
            item_count=self.n_items,
            subscale_count=self.n_subscales,
            missing_pct=self.missing_pct,
            likert_range=self.questionnaire.response_scale.range,
            reverse_scored_items=self.questionnaire.reverse_scored_ids,
            collected_at=self.imported_at,
        )
