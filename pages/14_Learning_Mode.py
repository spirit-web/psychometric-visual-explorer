"""Learning Mode page. Rendering only - lesson content lives in
utils/learning_content.py."""

import streamlit as st

from components.kpi_card import kpi_card
from utils.learning_content import LEARNING_MODULES

PAGE_PATHS = {
    "Dataset Overview": "pages/2_Dataset_Overview.py",
    "Psychometric QC": "pages/3_Psychometric_QC.py",
    "Test Profile Explorer": "pages/4_Test_Profile_Explorer.py",
    "Reliability Explorer": "pages/5_Reliability_Explorer.py",
    "Factor Explorer": "pages/6_Factor_Explorer.py",
    "Validity Dashboard": "pages/7_Validity_Dashboard.py",
    "Norm Explorer": "pages/8_Norm_Explorer.py",
    "Measurement Error": "pages/9_Measurement_Error.py",
    "Decision Support": "pages/10_Decision_Support.py",
    "Fairness Explorer": "pages/11_Fairness_Explorer.py",
    "Machine Learning": "pages/12_Machine_Learning.py",
}

st.title("Learning Mode")
st.caption("Lär dig psykometri steg för steg - varje modul länkar till hur begreppet används i appen.")

if "learning_progress" not in st.session_state:
    st.session_state["learning_progress"] = {m.key: False for m in LEARNING_MODULES}

progress = st.session_state["learning_progress"]
n_done = sum(1 for v in progress.values() if v)
n_total = len(LEARNING_MODULES)

# --- KPI row -------------------------------------------------
kpi_cols = st.columns(4)
with kpi_cols[0]:
    kpi_card("📚", "#DBEAFE", str(n_total), "Moduler totalt")
with kpi_cols[1]:
    kpi_card("✅", "#D1FAE5", str(n_done), "Moduler lästa")
with kpi_cols[2]:
    kpi_card("📊", "#EDE9FE", f"{n_done / n_total * 100:.0f}%", "Framsteg")
with kpi_cols[3]:
    next_module = next((m for m in LEARNING_MODULES if not progress.get(m.key)), None)
    kpi_card("➡️", "#FFEDD5", next_module.title if next_module else "Klart!", "Nästa modul")

st.progress(n_done / n_total)
st.write("")

tabs = st.tabs([m.title for m in LEARNING_MODULES])

for tab, module in zip(tabs, LEARNING_MODULES):
    with tab:
        st.markdown(f"### {module.title}")
        st.caption(module.subtitle)
        st.write(module.intro)

        st.write("")
        st.markdown("**Begrepp**")
        for concept in module.concepts:
            st.markdown(f"- **{concept.term}** - {concept.explanation}")

        if module.formulas:
            st.write("")
            st.markdown("**Formler**")
            for formula in module.formulas:
                st.latex(formula.latex)
                if formula.description:
                    st.caption(f"{formula.name}: {formula.description}")

        if module.example:
            st.write("")
            st.info(f"💡 {module.example}")

        if module.related_pages:
            st.write("")
            st.markdown("**Se i praktiken**")
            link_cols = st.columns(len(module.related_pages))
            for col, page_name in zip(link_cols, module.related_pages):
                with col:
                    if page_name in PAGE_PATHS and st.button(f"→ {page_name}", key=f"link_{module.key}_{page_name}", width="stretch"):
                        st.switch_page(PAGE_PATHS[page_name])

        st.write("")
        is_done = st.checkbox("Markera som läst", value=progress.get(module.key, False), key=f"done_{module.key}")
        progress[module.key] = is_done

st.write("")
col_back, _, col_next = st.columns([1, 3, 1])
with col_back:
    if st.button("Tillbaka", width="stretch"):
        st.switch_page("pages/13_Test_Builder.py")
with col_next:
    if st.button("Fortsätt till Export →", type="primary", width="stretch"):
        st.switch_page("pages/15_Export.py")
