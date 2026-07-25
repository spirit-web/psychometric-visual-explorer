"""Shared design tokens and CSS injection for the PVE Streamlit UI."""

import streamlit as st

COLORS = {
    "navy": "#0E1638",
    "navy_light": "#16204A",
    "accent_blue": "#2F5FE0",
    "accent_green": "#16A34A",
    "accent_purple": "#7C3AED",
    "accent_orange": "#EA580C",
    "bg": "#F5F6FA",
    "card_bg": "#FFFFFF",
    "text": "#1A1F36",
    "text_muted": "#6B7280",
    "status_good": "#16A34A",
    "status_warning": "#F59E0B",
    "status_bad": "#DC2626",
}

_CSS = """
<style>
[data-testid="stSidebar"] {
    background-color: %(navy)s;
}
[data-testid="stSidebar"] * {
    color: #E2E5F1;
}
[data-testid="stSidebarNav"] a,
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] {
    border-radius: 8px;
}
[data-testid="stSidebar"] [aria-current="page"] {
    background-color: %(accent_blue)s;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] [aria-current="page"] * {
    color: #FFFFFF !important;
}
.pve-logo {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.75rem 0.25rem 1.25rem 0.25rem;
}
.pve-logo-icon {
    font-size: 1.8rem;
}
.pve-logo-text-main {
    font-weight: 700;
    font-size: 1.15rem;
    color: #FFFFFF;
    line-height: 1.1;
}
.pve-logo-text-sub {
    font-size: 0.72rem;
    color: #9AA4C7;
    line-height: 1.1;
}
.pve-card {
    background-color: %(card_bg)s;
    border-radius: 14px;
    padding: 1.5rem;
    box-shadow: 0 1px 4px rgba(20, 24, 60, 0.08);
    height: 100%%;
}
.pve-kpi-card {
    background-color: %(card_bg)s;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 1px 4px rgba(20, 24, 60, 0.08);
}
.pve-status-good { color: %(status_good)s; font-weight: 600; }
.pve-status-warning { color: %(status_warning)s; font-weight: 600; }
.pve-status-bad { color: %(status_bad)s; font-weight: 600; }
.pve-badge {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
}
.pve-badge-good { background-color: #DCFCE7; color: %(status_good)s; }
.pve-badge-warning { background-color: #FEF3C7; color: #B45309; }
.pve-badge-bad { background-color: #FEE2E2; color: %(status_bad)s; }
</style>
""" % COLORS


def apply_theme() -> None:
    """Inject the PVE design system CSS. Call once at the top of every page."""
    st.markdown(_CSS, unsafe_allow_html=True)
