"""PVE - Psychometric Visual Explorer. Streamlit entry point and page navigation."""

import streamlit as st

from components.sidebar import render_logo
from utils.theme import apply_theme

st.set_page_config(
    page_title="Psychometric Visual Explorer",
    page_icon="\U0001F9E0",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()
render_logo()


def home() -> None:
    st.markdown(
        "<div style='color:#2F5FE0; font-weight:700; letter-spacing:0.08em;'>"
        "VÄLKOMMEN TILL</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h1 style='margin-top:0;'>Psychometric Visual Explorer</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:1.1rem; color:#6B7280; max-width:40rem;'>"
        "Din plattform för att analysera, visualisera och förstå psykologiska "
        "test – från rådata till insikter.</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    cards = [
        {
            "icon": "\U0001F4C1",
            "icon_bg": "#DBEAFE",
            "title": "Öppna dataset",
            "desc": "Ladda upp ditt eget dataset eller öppna ett tidigare projekt.",
            "button": "Öppna",
            "target": "pages/1_Import_Wizard.py",
        },
        {
            "icon": "\U0001F5C4️",
            "icon_bg": "#D1FAE5",
            "title": "Exempeldata",
            "desc": "Utforska exempel på vanliga psykologiska test.",
            "button": "Utforska",
            "target": "pages/1_Import_Wizard.py",
        },
        {
            "icon": "\U0001F393",
            "icon_bg": "#EDE9FE",
            "title": "Learning Mode",
            "desc": "Lär dig psykometri med interaktiva förklaringar och exempel.",
            "button": "Starta",
            "target": "pages/14_Learning_Mode.py",
        },
        {
            "icon": "\U0001F4D6",
            "icon_bg": "#FFEDD5",
            "title": "Dokumentation",
            "desc": "Guider, begrepp och metoder för alla analyser.",
            "button": "Läs mer",
            "target": None,
        },
    ]

    cols = st.columns(4)
    for col, card in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="pve-card">
                    <div style="background-color:{card['icon_bg']}; width:3rem;
                        height:3rem; border-radius:12px; display:flex;
                        align-items:center; justify-content:center;
                        font-size:1.4rem; margin-bottom:0.9rem;">
                        {card['icon']}
                    </div>
                    <div style="font-weight:700; font-size:1.05rem;
                        margin-bottom:0.4rem;">{card['title']}</div>
                    <div style="color:#6B7280; font-size:0.88rem;
                        min-height:2.6rem;">{card['desc']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            if card["target"] is not None:
                if st.button(card["button"], key=f"home_{card['title']}", use_container_width=True):
                    if card["title"] == "Exempeldata":
                        st.session_state["pve_use_example"] = True
                    st.switch_page(card["target"])
            else:
                with st.popover(card["button"], use_container_width=True):
                    st.write(
                        "Dokumentationen byggs ut i takt med att modulerna "
                        "färdigställs. Se README.md i projektets rot under tiden."
                    )


PAGES = [
    st.Page(home, title="Home", icon=":material/home:", default=True, url_path="home"),
    st.Page("pages/1_Import_Wizard.py", title="Import Wizard", icon=":material/upload_file:"),
    st.Page("pages/2_Dataset_Overview.py", title="Dataset Overview", icon=":material/dataset:"),
    st.Page("pages/3_Psychometric_QC.py", title="Psychometric QC", icon=":material/verified:"),
    st.Page("pages/4_Test_Profile_Explorer.py", title="Test Profile Explorer", icon=":material/list_alt:"),
    st.Page("pages/5_Reliability_Explorer.py", title="Reliability Explorer", icon=":material/speed:"),
    st.Page("pages/6_Factor_Explorer.py", title="Factor Explorer", icon=":material/scatter_plot:"),
    st.Page("pages/7_Validity_Dashboard.py", title="Validity Dashboard", icon=":material/track_changes:"),
    st.Page("pages/8_Norm_Explorer.py", title="Norm Explorer", icon=":material/bar_chart:"),
    st.Page("pages/9_Measurement_Error.py", title="Measurement Error", icon=":material/straighten:"),
    st.Page("pages/10_Decision_Support.py", title="Decision Support", icon=":material/rule:"),
    st.Page("pages/11_Fairness_Explorer.py", title="Fairness Explorer", icon=":material/balance:"),
    st.Page("pages/12_Machine_Learning.py", title="Machine Learning", icon=":material/model_training:"),
    st.Page("pages/13_Test_Builder.py", title="Test Builder", icon=":material/build:"),
    st.Page("pages/14_Learning_Mode.py", title="Learning Mode", icon=":material/school:"),
    st.Page("pages/15_Export.py", title="Export", icon=":material/ios_share:"),
]

nav = st.navigation(PAGES, position="sidebar")
nav.run()
