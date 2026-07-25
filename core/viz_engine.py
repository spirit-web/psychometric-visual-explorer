"""Builds Plotly figures from core/stats_engine outputs.

Returns figures only - never calculates statistics, never imports Streamlit.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

COLOR_PRIMARY = "#2F5FE0"
COLOR_GOOD = "#16A34A"
COLOR_WARNING = "#F59E0B"
COLOR_BAD = "#DC2626"
CATEGORICAL_PALETTE = ["#2F5FE0", "#16A34A", "#7C3AED", "#EA580C", "#0EA5E9", "#DB2777"]

_LAYOUT_DEFAULTS = dict(
    margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#1A1F36"),
)


def response_distribution_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=df["label"],
            y=df["pct"],
            marker_color=COLOR_PRIMARY,
            text=[f"{v:.1f}%" for v in df["pct"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        yaxis_title="Andel av alla svar (%)",
        xaxis_title="Svarsalternativ",
        height=320,
        **_LAYOUT_DEFAULTS,
    )
    return fig


def missing_by_item_chart(series: pd.Series) -> go.Figure:
    ordered = series.sort_values(ascending=True)
    fig = go.Figure(
        go.Bar(
            x=ordered.values,
            y=ordered.index,
            orientation="h",
            marker_color=COLOR_PRIMARY,
            text=[f"{v:.1f}%" for v in ordered.values],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis_title="Bortfall (%)",
        height=max(240, 26 * len(ordered)),
        **_LAYOUT_DEFAULTS,
    )
    return fig


def demographics_donut(series: pd.Series) -> go.Figure:
    fig = go.Figure(
        go.Pie(
            labels=series.index.astype(str),
            values=series.values,
            hole=0.55,
            marker=dict(colors=CATEGORICAL_PALETTE),
            textinfo="percent",
        )
    )
    fig.update_layout(height=280, legend=dict(orientation="v"), **_LAYOUT_DEFAULTS)
    return fig


def score_histogram(series: pd.Series, cutoffs: list | None = None) -> go.Figure:
    fig = go.Figure(go.Histogram(x=series.dropna(), marker_color=COLOR_PRIMARY, nbinsx=20))
    if cutoffs:
        colors = CATEGORICAL_PALETTE
        for i, cutoff in enumerate(cutoffs):
            fig.add_vline(
                x=cutoff.range[0],
                line_dash="dot",
                line_color=colors[i % len(colors)],
                annotation_text=cutoff.label,
                annotation_position="top",
            )
    fig.update_layout(xaxis_title="Totalpoäng", yaxis_title="Antal", height=320, **_LAYOUT_DEFAULTS)
    return fig
