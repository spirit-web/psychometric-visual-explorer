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


def horizontal_bar_chart(series: pd.Series, x_title: str, ascending: bool = True, x_range: tuple | None = None) -> go.Figure:
    ordered = series.sort_values(ascending=ascending)
    fig = go.Figure(
        go.Bar(
            x=ordered.values,
            y=ordered.index,
            orientation="h",
            marker_color=COLOR_PRIMARY,
            text=[f"{v:.2f}" for v in ordered.values],
            textposition="outside",
        )
    )
    fig.update_layout(xaxis_title=x_title, height=max(240, 26 * len(ordered)), **_LAYOUT_DEFAULTS)
    if x_range:
        fig.update_xaxes(range=x_range)
    return fig


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            colorbar=dict(title=""),
        )
    )
    fig.update_layout(height=max(320, 24 * len(corr)), **_LAYOUT_DEFAULTS)
    fig.update_yaxes(autorange="reversed")
    return fig


def alpha_if_deleted_chart(items: pd.DataFrame, baseline_alpha: float | None) -> go.Figure:
    """items: DataFrame indexed by item id with column 'alpha_if_deleted'."""
    ordered = items.sort_values("alpha_if_deleted", ascending=True)
    colors = [COLOR_BAD if v > (baseline_alpha or 0) else COLOR_PRIMARY for v in ordered["alpha_if_deleted"]]
    fig = go.Figure(
        go.Bar(
            x=ordered["alpha_if_deleted"],
            y=ordered.index,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in ordered["alpha_if_deleted"]],
            textposition="outside",
        )
    )
    if baseline_alpha is not None:
        fig.add_vline(x=baseline_alpha, line_dash="dot", line_color=COLOR_WARNING, annotation_text="Nuvarande alpha")
    fig.update_layout(xaxis_title="Alpha om borttaget", height=max(240, 26 * len(ordered)), **_LAYOUT_DEFAULTS)
    return fig


def scatter_with_regression(
    x: pd.Series, y: pd.Series, x_title: str, y_title: str, trendline: tuple[float, float] | None = None
) -> go.Figure:
    """trendline: precomputed (slope, intercept) from stats_engine.linear_fit - this
    function only draws, it does not fit anything."""
    fig = go.Figure(go.Scatter(x=x, y=y, mode="markers", marker=dict(color=COLOR_PRIMARY, opacity=0.6)))
    if trendline is not None:
        slope, intercept = trendline
        x_line = [float(x.min()), float(x.max())]
        y_line = [slope * xv + intercept for xv in x_line]
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", line=dict(color=COLOR_WARNING, dash="dash")))
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title, height=340, showlegend=False, **_LAYOUT_DEFAULTS)
    return fig


def stacked_response_distribution_chart(item_dist: pd.DataFrame) -> go.Figure:
    """item_dist: item x category %-matrix from stats_engine.response_distribution_by_item."""
    fig = go.Figure()
    for i, category in enumerate(item_dist.columns):
        fig.add_trace(
            go.Bar(
                name=str(category),
                x=item_dist[category],
                y=item_dist.index,
                orientation="h",
                marker_color=CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)],
            )
        )
    fig.update_layout(
        barmode="stack",
        xaxis_title="Andel av svar (%)",
        height=max(280, 24 * len(item_dist)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **_LAYOUT_DEFAULTS,
    )
    return fig


def scree_plot_chart(actual_eigenvalues, simulated_eigenvalues) -> go.Figure:
    x = list(range(1, len(actual_eigenvalues) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=actual_eigenvalues, mode="lines+markers", name="Egenvärden (data)", line=dict(color=COLOR_PRIMARY))
    )
    if len(simulated_eigenvalues):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=simulated_eigenvalues,
                mode="lines+markers",
                name="Parallel analysis (slumpdata)",
                line=dict(color=COLOR_WARNING, dash="dash"),
            )
        )
    fig.add_hline(y=1.0, line_dash="dot", line_color="#9AA4C7")
    fig.update_layout(
        xaxis_title="Antal faktorer",
        yaxis_title="Egenvärde",
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **_LAYOUT_DEFAULTS,
    )
    return fig


def normal_curve_chart(x, y, marker_x: float | None, marker_label: str, x_title: str, ci: tuple[float, float] | None = None) -> go.Figure:
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", fill="tozeroy", line=dict(color=COLOR_PRIMARY), fillcolor="rgba(47,95,224,0.15)"))
    if ci is not None:
        lo, hi = ci
        fig.add_vrect(x0=lo, x1=hi, fillcolor=COLOR_WARNING, opacity=0.12, line_width=0)
    if marker_x is not None:
        fig.add_vline(x=marker_x, line_color=COLOR_WARNING, line_dash="dash", annotation_text=marker_label)
    fig.update_layout(xaxis_title=x_title, yaxis_title="Täthet", height=340, **_LAYOUT_DEFAULTS)
    return fig


def ci_error_bar_chart(labels: list, centers: list[float], lowers: list[float], uppers: list[float], y_title: str) -> go.Figure:
    errors_plus = [u - c for u, c in zip(uppers, centers)]
    errors_minus = [c - l for c, l in zip(centers, lowers)]
    fig = go.Figure(
        go.Scatter(
            x=list(labels),
            y=centers,
            mode="markers",
            marker=dict(color=COLOR_PRIMARY, size=9),
            error_y=dict(type="data", symmetric=False, array=errors_plus, arrayminus=errors_minus, color=COLOR_PRIMARY),
        )
    )
    fig.update_layout(yaxis_title=y_title, height=380, **_LAYOUT_DEFAULTS)
    return fig


def roc_curve_chart(fpr, tpr, auc: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {auc:.2f})", line=dict(color=COLOR_PRIMARY, width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Slumpnivå", line=dict(color="#9AA4C7", dash="dash")))
    fig.update_layout(
        xaxis_title="1 - Specificitet (FPR)",
        yaxis_title="Sensitivitet (TPR)",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **_LAYOUT_DEFAULTS,
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig


def confusion_matrix_heatmap(tp: int, fp: int, tn: int, fn: int) -> go.Figure:
    z = [[tp, fn], [fp, tn]]
    text = [[f"TP<br>{tp}", f"FN<br>{fn}"], [f"FP<br>{fp}", f"TN<br>{tn}"]]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=["Predikterat positiv", "Predikterat negativ"],
            y=["Observerat positiv", "Observerat negativ"],
            text=text,
            texttemplate="%{text}",
            colorscale=[[0, "#F5F6FA"], [1, COLOR_PRIMARY]],
            showscale=False,
        )
    )
    fig.update_layout(height=320, **_LAYOUT_DEFAULTS)
    fig.update_yaxes(autorange="reversed")
    return fig


def group_comparison_bar_chart(labels: list[str], d_values: list[float]) -> go.Figure:
    colors = [COLOR_BAD if abs(d) >= 0.5 else (COLOR_WARNING if abs(d) >= 0.2 else COLOR_GOOD) for d in d_values]
    fig = go.Figure(go.Bar(x=labels, y=d_values, marker_color=colors, text=[f"{d:.2f}" for d in d_values], textposition="outside"))
    for y, text in [(0.5, "Måttlig skillnad"), (-0.5, "Måttlig skillnad"), (0.8, "Stor skillnad"), (-0.8, "Stor skillnad")]:
        fig.add_hline(y=y, line_dash="dot", line_color="#D1D5DB", annotation_text=text, annotation_font_size=10)
    fig.update_layout(yaxis_title="Skillnad (Cohen's d)", height=380, **_LAYOUT_DEFAULTS)
    return fig
