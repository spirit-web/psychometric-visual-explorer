"""PVE - Psychometric Visual Explorer. Streamlit entry point and page navigation."""

import streamlit as st

from components.sidebar import render_logo, render_mode_toggles
from utils.i18n import get_language, t
from utils.theme import apply_theme

st.set_page_config(
    page_title="Psychometric Visual Explorer",
    page_icon="\U0001F9E0",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()
render_logo()
render_mode_toggles()


def home() -> None:
    lang = get_language()
    intro_label = "VÄLKOMMEN TILL" if lang == "sv" else "WELCOME TO"
    tagline = (
        "Din plattform för att analysera, visualisera och förstå psykologiska "
        "test – från rådata till insikter."
        if lang == "sv"
        else "Your platform for analyzing, visualizing, and understanding "
        "psychological tests – from raw data to insight."
    )
    st.markdown(
        f"<div style='color:#2F5FE0; font-weight:700; letter-spacing:0.08em;'>"
        f"{intro_label}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h1 style='margin-top:0;'>Psychometric Visual Explorer</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='font-size:1.1rem; color:#6B7280; max-width:40rem;'>{tagline}</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    if lang == "sv":
        value_props = [
            "Vill du veta var en specifik klient hamnar jämfört med normgruppen?",
            "Vill du se om testet fungerar lika rättvist oavsett kön, ålder eller utbildning?",
            "Vill du se vilka frågor som är viktigast — eller bygga ett eget test?",
        ]
    else:
        value_props = [
            "Want to know where a specific client falls compared to the norm group?",
            "Want to see if the test performs equally well regardless of gender, age, or education?",
            "Want to see which questions matter most — or build your own test?",
        ]
    st.markdown(
        "<ul style='color:#374151; font-size:1.02rem; line-height:1.9; "
        "max-width:40rem; padding-left:1.2rem;'>"
        + "".join(f"<li>{prop}</li>" for prop in value_props)
        + "</ul>",
        unsafe_allow_html=True,
    )
    st.write("")

    if lang == "sv":
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
                "id": "sample_data",
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
    else:
        cards = [
            {
                "icon": "\U0001F4C1",
                "icon_bg": "#DBEAFE",
                "title": "Open dataset",
                "desc": "Upload your own dataset or open a previous project.",
                "button": "Open",
                "target": "pages/1_Import_Wizard.py",
            },
            {
                "id": "sample_data",
                "icon": "\U0001F5C4️",
                "icon_bg": "#D1FAE5",
                "title": "Sample data",
                "desc": "Explore examples of common psychological tests.",
                "button": "Explore",
                "target": "pages/1_Import_Wizard.py",
            },
            {
                "icon": "\U0001F393",
                "icon_bg": "#EDE9FE",
                "title": "Learning Mode",
                "desc": "Learn psychometrics with interactive explanations and examples.",
                "button": "Start",
                "target": "pages/14_Learning_Mode.py",
            },
            {
                "icon": "\U0001F4D6",
                "icon_bg": "#FFEDD5",
                "title": "Documentation",
                "desc": "Guides, concepts, and methods for every analysis.",
                "button": "Read more",
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
                if st.button(card["button"], key=f"home_{card.get('id', card['title'])}", width="stretch"):
                    if card.get("id") == "sample_data":
                        st.session_state["pve_use_example"] = True
                    st.switch_page(card["target"])
            else:
                with st.popover(card["button"], width="stretch"):
                    st.write(
                        "Dokumentationen byggs ut i takt med att modulerna "
                        "färdigställs. Se README.md i projektets rot under tiden."
                    )


PAGES = [
    st.Page(home, title=t("Home"), icon=":material/home:", default=True, url_path="home"),
    st.Page("pages/1_Import_Wizard.py", title=t("Importera Tester"), icon=":material/upload_file:"),
    st.Page("pages/2_Dataset_Overview.py", title=t("Dataset Overview"), icon=":material/dataset:"),
    st.Page("pages/3_Psychometric_QC.py", title=t("Psychometric QC"), icon=":material/verified:"),
    st.Page("pages/5_Reliability_Explorer.py", title=t("Reliability Explorer"), icon=":material/speed:"),
    st.Page("pages/6_Factor_Explorer.py", title=t("Factor Explorer"), icon=":material/scatter_plot:"),
    st.Page("pages/7_Validity_Dashboard.py", title=t("Validity Dashboard"), icon=":material/track_changes:"),
    st.Page("pages/8_Norm_Explorer.py", title=t("Norm Explorer"), icon=":material/bar_chart:"),
    st.Page("pages/9_Measurement_Error.py", title=t("Measurement Error"), icon=":material/straighten:"),
    st.Page("pages/10_Decision_Support.py", title=t("Decision Support"), icon=":material/rule:"),
    st.Page("pages/11_Fairness_Explorer.py", title=t("Fairness Explorer"), icon=":material/balance:"),
    st.Page("pages/12_Machine_Learning.py", title=t("Machine Learning"), icon=":material/model_training:"),
    st.Page("pages/13_Test_Builder.py", title=t("Test Builder"), icon=":material/build:"),
    st.Page("pages/14_Learning_Mode.py", title=t("Learning Mode"), icon=":material/school:"),
    st.Page("pages/15_Export.py", title=t("Export"), icon=":material/ios_share:"),
]

nav = st.navigation(PAGES, position="sidebar")
nav.run()
