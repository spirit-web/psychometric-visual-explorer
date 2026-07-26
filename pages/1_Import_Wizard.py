"""Import Wizard page: 2-step flow (Ladda upp -> Granska) from raw file to
a mapped, scored Dataset.

Rendering only - all parsing/matching/scoring logic lives in
core/import_engine.py and core/plugin_engine.py.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from core.import_engine import build_dataset, read_file
from core.plugin_engine import load_all_plugins, match_plugin_to_dataframe

SAMPLE_DATASETS = {
    "gad7": {
        "label": "GAD-7 (Ångest)",
        "path": Path("data/sample_datasets/gad7_sample.csv"),
    },
    "phq9": {
        "label": "PHQ-9 (Depression)",
        "path": Path("data/sample_datasets/phq9_sample.csv"),
    },
    "ipip_bigfive": {
        "label": "IPIP Big Five (Personlighet)",
        "path": Path("data/sample_datasets/bigfive_sample.csv"),
    },
}

STEP_LABELS = ["Ladda upp", "Granska"]

DEFAULTS = {
    "iw_step": 1,
    "iw_raw_df": None,
    "iw_source_name": None,
    "iw_plugins": None,
    "iw_questionnaire": None,
    "iw_column_mapping": {},
    "iw_auto_matched": False,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

if st.session_state["iw_plugins"] is None:
    st.session_state["iw_plugins"] = load_all_plugins()


def go_to(step: int) -> None:
    st.session_state["iw_step"] = step
    st.rerun()


def reset_wizard() -> None:
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    st.rerun()


def set_raw_data(df: pd.DataFrame, source_name: str) -> None:
    st.session_state["iw_raw_df"] = df
    st.session_state["iw_source_name"] = source_name
    st.session_state["iw_auto_matched"] = False


def render_stepper(current: int) -> None:
    cols = st.columns(len(STEP_LABELS))
    for i, (col, label) in enumerate(zip(cols, STEP_LABELS), start=1):
        with col:
            if i < current:
                st.markdown(f"<div style='text-align:center; color:#2F5FE0;'>✓<br/>{label}</div>", unsafe_allow_html=True)
            elif i == current:
                st.markdown(f"<div style='text-align:center; color:#2F5FE0; font-weight:700;'>{i}<br/>{label}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align:center; color:#9AA4C7;'>{i}<br/>{label}</div>", unsafe_allow_html=True)


st.title("Importera Tester")
st.caption(f"Steg {st.session_state['iw_step']} av 2 – {STEP_LABELS[st.session_state['iw_step'] - 1]}")
render_stepper(st.session_state["iw_step"])
st.write("")

step = st.session_state["iw_step"]

# --- Step 1: Ladda upp -------------------------------------------------
if step == 1:
    use_example_hint = st.session_state.pop("pve_use_example", False)

    tab_upload, tab_example = st.tabs(["Ladda upp fil", "Exempeldata"])

    with tab_upload:
        st.caption("Ladda upp data från ett psykologiskt test (t.ex. GAD-7, PHQ-9) - en rad per deltagare.")
        uploaded = st.file_uploader(
            "Dra och släpp din fil här, eller välj fil",
            type=["csv", "xlsx", "xls"],
        )
        if uploaded is not None:
            df, error = read_file(uploaded.getvalue(), uploaded.name)
            if error:
                st.error(error)
            else:
                set_raw_data(df, uploaded.name)

    with tab_example:
        st.write("Välj ett exempeldataset att utforska:")
        example_cols = st.columns(3)
        for col, (key, meta) in zip(example_cols, SAMPLE_DATASETS.items()):
            with col:
                st.markdown(f"**{meta['label']}**")
                if st.button("Använd", key=f"use_example_{key}", width="stretch"):
                    df = pd.read_csv(meta["path"])
                    set_raw_data(df, meta["path"].name)
                    go_to(2)

    if use_example_hint:
        st.info("Välj ett exempeldataset i fliken \"Exempeldata\" ovan.")

    st.write("")
    col_cancel, _, col_next = st.columns([1, 3, 1])
    with col_cancel:
        if st.button("Avbryt", width="stretch"):
            reset_wizard()
    with col_next:
        if st.button("Nästa →", type="primary", width="stretch", disabled=st.session_state["iw_raw_df"] is None):
            go_to(2)

# --- Step 2: Granska -------------------------------------------------
elif step == 2:
    df = st.session_state["iw_raw_df"]
    plugins = st.session_state["iw_plugins"]
    columns = list(df.columns)

    if not st.session_state["iw_auto_matched"]:
        match = match_plugin_to_dataframe(df, plugins)
        if match is not None:
            questionnaire, mapping = match
        else:
            questionnaire, mapping = None, {}
        st.session_state["iw_questionnaire"] = questionnaire
        st.session_state["iw_column_mapping"] = mapping
        st.session_state["iw_auto_matched"] = True

    questionnaire = st.session_state["iw_questionnaire"]
    mapping = dict(st.session_state["iw_column_mapping"])

    if questionnaire is not None:
        st.success(f"Testet identifierades automatiskt som **{questionnaire.test_name}** ({questionnaire.full_name}).")
    else:
        st.warning("Inget test kunde identifieras automatiskt. Välj ett test manuellt nedan.")
        options = {"— Välj test —": None}
        options.update({q.test_name: pid for pid, q in plugins.items()})
        choice = st.selectbox("Test", options.keys())
        selected_id = options[choice]
        if selected_id is not None:
            questionnaire = plugins[selected_id]
            st.session_state["iw_questionnaire"] = questionnaire
            mapping = {}

    n_matched = len(mapping)
    n_total_items = len(questionnaire.items) if questionnaire is not None else 0
    fully_matched = questionnaire is not None and n_matched == n_total_items

    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Deltagare", len(df))
    kpi_cols[1].metric("Kolumner totalt", len(df.columns))
    kpi_cols[2].metric("Identifierade frågor", f"{n_matched} / {n_total_items}" if questionnaire is not None else "–")

    st.write("**Förhandsgranskning**")
    st.dataframe(df.head(10), width="stretch")

    if questionnaire is not None:
        with st.expander(
            "Justera kartläggning av frågor" if fully_matched else "Kartlägg frågor till kolumner",
            expanded=not fully_matched,
        ):
            options_col = ["— Ingen —"] + columns
            for item in questionnaire.items:
                current = mapping.get(item.id, "— Ingen —")
                choice = st.selectbox(
                    f"{item.id} – {item.text}",
                    options_col,
                    index=options_col.index(current) if current in options_col else 0,
                    key=f"map_{item.id}",
                )
                if choice != "— Ingen —":
                    mapping[item.id] = choice
                else:
                    mapping.pop(item.id, None)
            st.session_state["iw_column_mapping"] = mapping

    st.write("")
    col_back, _, col_next = st.columns([1, 3, 1])
    with col_back:
        if st.button("Tillbaka", width="stretch"):
            go_to(1)
    with col_next:
        ready = questionnaire is not None and len(mapping) > 0
        if st.button("Skapa dataset →", type="primary", width="stretch", disabled=not ready):
            demographic_columns = [c for c in columns if c not in mapping.values()]
            dataset = build_dataset(
                raw=df,
                questionnaire=questionnaire,
                column_mapping=mapping,
                demographic_columns=demographic_columns,
                name=st.session_state["iw_source_name"] or questionnaire.test_name,
            )
            st.session_state["pve_dataset"] = dataset
            st.switch_page("pages/2_Dataset_Overview.py")
