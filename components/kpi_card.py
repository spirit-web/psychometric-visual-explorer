"""Reusable KPI card widget used across the top of most analysis pages."""

import streamlit as st


def kpi_card(icon: str, icon_bg: str, value: str, label: str, caption: str | None = None) -> None:
    caption_html = f'<div style="color:#9AA4C7; font-size:0.78rem; margin-top:0.2rem;">{caption}</div>' if caption else ""
    st.markdown(
        f"""
        <div class="pve-kpi-card">
          <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.5rem;">
            <div style="background:{icon_bg}; width:2.2rem; height:2.2rem; border-radius:8px;
                 display:flex; align-items:center; justify-content:center; font-size:1.1rem; flex-shrink:0;">{icon}</div>
            <div style="color:#6B7280; font-size:0.85rem;">{label}</div>
          </div>
          <div style="font-size:1.6rem; font-weight:700;">{value}</div>
          {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(status: str, text: str) -> str:
    """Returns an inline HTML badge for a QC/status value: good | warning | bad."""
    css_class = {"good": "pve-badge-good", "warning": "pve-badge-warning", "bad": "pve-badge-bad"}.get(status, "pve-badge-good")
    return f'<span class="pve-badge {css_class}">{text}</span>'


def status_icon(status: str) -> str:
    return {"good": "✅", "warning": "⚠️", "bad": "🛑"}.get(status, "•")
