"""Pedagogical wheel diagram showing the five validity evidence sources
(AERA/APA/NCME) radiating from the test itself, color-coded by their
current evidence status for the active test. Pure rendering - the status
values it draws come from core.stats_engine.validity_overview.

Every element is built as a single line with no leading whitespace: st.markdown
runs the string through a CommonMark parser first, and 4+ leading spaces on a
line is indented-code-block syntax - it would render the SVG as literal text
instead of markup.
"""

import math

import streamlit as st

_ORDER = ["content", "response_processes", "internal_structure", "relations", "consequences"]

_SHORT_LABELS = {
    "content": "Innehåll",
    "response_processes": "Responsprocesser",
    "internal_structure": "Intern struktur",
    "relations": "Relation till andra variabler",
    "consequences": "Konsekvenser",
}

_DESCRIPTIONS = {
    "content": "Täcker frågorna rätt innehåll?",
    "response_processes": "Tolkar testtagare frågorna rätt?",
    "internal_structure": "Samspelar frågorna som väntat?",
    "relations": "Stämmer korrelationer med andra mått?",
    "consequences": "Är effekterna av testet rättvisa?",
}

_STATUS_COLORS = {"strong": "#16A34A", "moderate": "#F59E0B", "limited": "#EA580C", "none": "#9CA3AF"}
_STATUS_TEXT = {
    "strong": "Stark evidens",
    "moderate": "Måttlig evidens",
    "limited": "Begränsad evidens",
    "none": "Ingen information",
}


def render_validity_wheel(sources, test_name: str) -> None:
    by_key = {s.key: s for s in sources}
    cx, cy, radius = 400, 300, 205
    box_w, box_h = 190, 100

    parts = []
    for i, key in enumerate(_ORDER):
        angle = math.radians(-90 + i * 72)
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        color = _STATUS_COLORS[by_key[key].status]
        parts.append(
            f'<line x1="{cx}" y1="{cy}" x2="{x:.0f}" y2="{y:.0f}" stroke="{color}" stroke-width="3" stroke-dasharray="6,4" />'
        )

    parts.append(f'<circle cx="{cx}" cy="{cy}" r="90" fill="#0E1638" />')
    parts.append(
        f'<foreignObject x="{cx - 85}" y="{cy - 60}" width="170" height="120">'
        '<div xmlns="http://www.w3.org/1999/xhtml" style="color:#FFFFFF; text-align:center; display:flex; flex-direction:column; justify-content:center; height:100%; font-family:inherit;">'
        '<div style="font-size:12px; opacity:0.75;">TESTET</div>'
        f'<div style="font-weight:700; font-size:15px; line-height:1.25;">{test_name}</div>'
        '<div style="font-size:10px; opacity:0.6; margin-top:4px;">Mäter det avsedda konstruktet?</div>'
        "</div></foreignObject>"
    )

    for i, key in enumerate(_ORDER):
        angle = math.radians(-90 + i * 72)
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        color = _STATUS_COLORS[by_key[key].status]
        parts.append(
            f'<foreignObject x="{x - box_w / 2:.0f}" y="{y - box_h / 2:.0f}" width="{box_w}" height="{box_h}">'
            f'<div xmlns="http://www.w3.org/1999/xhtml" style="background:#FFFFFF; border:2px solid {color}; border-radius:12px; padding:8px 10px; height:100%; box-sizing:border-box; display:flex; flex-direction:column; justify-content:center; font-family:inherit;">'
            f'<div style="font-weight:700; font-size:13px; color:#1A1F36; line-height:1.2;">{_SHORT_LABELS[key]}</div>'
            f'<div style="font-size:11px; color:{color}; font-weight:600; margin:2px 0;">{_STATUS_TEXT[by_key[key].status]}</div>'
            f'<div style="font-size:11px; color:#6B7280; line-height:1.25;">{_DESCRIPTIONS[key]}</div>'
            "</div></foreignObject>"
        )

    svg = (
        '<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:auto;">'
        + "".join(parts)
        + "</svg>"
    )
    st.markdown(svg, unsafe_allow_html=True)
