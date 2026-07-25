"""Import Wizard page: 5-step flow from raw file to a mapped Dataset.

Rendering only - all parsing/matching/scoring logic lives in
core/import_engine.py and core/plugin_engine.py.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from core.import_engine import build_dataset, identify_column_types, read_file
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

STEP_LABELS = ["Ladda upp", "Identifiera", "Granska", "Karta", "Slutför"]

DEFAULTS = {
    "iw_step": 1,
    "iw_raw_df": None,
    "iw_source_name": None,
    "iw_plugins": None,
    "iw_questionnaire": None,
    "iw_column_mapping": {},
    "iw_demographic_columns": [],
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


st.title("Import Wizard")
st.caption(f"Steg {st.session_state['iw_step']} av 5 – {STEP_LABELS[st.session_state['iw_step'] - 1]}")
render_stepper(st.session_state["iw_step"])
st.write("")

step = st.session_state["iw_step"]

# --- Step 1: Ladda upp -------------------------------------------------
if step == 1:
    use_example_hint = st.session_state.pop("pve_use_example", False)

    tab_upload, tab_example = st.tabs(["Ladda upp fil", "Exempeldata"])

    with tab_upload:
        uploaded = st.file_uploader(
            "Dra och släpp din fil här, eller välj fil",
            type=["csv", "xlsx", "xls"],
        )
        if uploaded is not None:
            df, error = read_file(uploaded.getvalue(), uploaded.name)
            if error:
                st.error(error)
            else:
                st.session_state["iw_raw_df"] = df
                st.session_state["iw_source_name"] = uploaded.name

    with tab_example:
        st.write("Välj ett exempeldataset att utforska:")
        example_cols = st.columns(3)
        for col, (key, meta) in zip(example_cols, SAMPLE_DATASETS.items()):
            with col:
                st.markdown(f"**{meta['label']}**")
                if st.button("Använd", key=f"use_example_{key}", use_container_width=True):
                    df = pd.read_csv(meta["path"])
                    st.session_state["iw_raw_df"] = df
                    st.session_state["iw_source_name"] = meta["path"].name
                    go_to(2)

    if use_example_hint:
        st.info("Välj ett exempeldataset i fliken \"Exempeldata\" ovan.")

    st.write("")
    col_cancel, _, col_next = st.columns([1, 3, 1])
    with col_cancel:
        if st.button("Avbryt", use_container_width=True):
            reset_wizard()
    with col_next:
        if st.button("Nästa →", type="primary", use_container_width=True, disabled=st.session_state["iw_raw_df"] is None):
            go_to(2)

# --- Step 2: Identifiera -------------------------------------------------
elif step == 2:
    df = st.session_state["iw_raw_df"]
    plugins = st.session_state["iw_plugins"]

    st.write(f"Filen innehåller **{len(df)} rader** och **{len(df.columns)} kolumner**.")

    match = match_plugin_to_dataframe(df, plugins)
    if match is not None:
        questionnaire, mapping = match
        st.success(
            f"Testet identifierades automatiskt som **{questionnaire.test_name}** "
            f"({questionnaire.full_name}) – {len(mapping)} av {len(questionnaire.items)} items matchade."
        )
        st.session_state["iw_questionnaire"] = questionnaire
        st.session_state["iw_column_mapping"] = mapping
    else:
        st.warning(
            "Inget test kunde identifieras automatiskt. Välj ett test manuellt "
            "nedan, eller gå vidare och kartlägg items för hand i nästa steg."
        )
        options = {"— Ingen matchning (manuell kartläggning) —": None}
        options.update({q.test_name: pid for pid, q in plugins.items()})
        choice = st.selectbox("Välj test manuellt", options.keys())
        selected_id = options[choice]
        st.session_state["iw_questionnaire"] = plugins.get(selected_id) if selected_id else None
        st.session_state["iw_column_mapping"] = {}

    st.write("")
    col_back, _, col_next = st.columns([1, 3, 1])
    with col_back:
        if st.button("Tillbaka", use_container_width=True):
            go_to(1)
    with col_next:
        if st.button("Nästa →", type="primary", use_container_width=True):
            go_to(3)

# --- Step 3: Granska -------------------------------------------------
elif step == 3:
    df = st.session_state["iw_raw_df"]
    types = identify_column_types(df)

    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Deltagare", len(df))
    kpi_cols[1].metric("Kolumner totalt", len(df.columns))
    kpi_cols[2].metric("Sannolika items", len(types["likely_items"]))

    st.write("**Förhandsgranskning**")
    st.dataframe(df.head(10), use_container_width=True)

    with st.expander("Kolumnöversikt"):
        st.write("Sannolika items:", ", ".join(types["likely_items"]) or "–")
        st.write("Övriga kolumner:", ", ".join(types["likely_other"]) or "–")

    st.write("")
    col_back, _, col_next = st.columns([1, 3, 1])
    with col_back:
        if st.button("Tillbaka", use_container_width=True):
            go_to(2)
    with col_next:
        if st.button("Nästa →", type="primary", use_container_width=True):
            go_to(4)

# --- Step 4: Karta -------------------------------------------------
elif step == 4:
    df = st.session_state["iw_raw_df"]
    questionnaire = st.session_state["iw_questionnaire"]
    columns = list(df.columns)

    if questionnaire is None:
        st.error("Inget test valt. Gå tillbaka till steg 2 och välj ett test.")
    else:
        st.write(f"Kartlägg varje item i **{questionnaire.test_name}** till en kolumn i din fil.")
        mapping = dict(st.session_state["iw_column_mapping"])
        options = ["— Ingen —"] + columns
        for item in questionnaire.items:
            current = mapping.get(item.id, "— Ingen —")
            choice = st.selectbox(
                f"{item.id} – {item.text}",
                options,
                index=options.index(current) if current in options else 0,
                key=f"map_{item.id}",
            )
            if choice != "— Ingen —":
                mapping[item.id] = choice
            else:
                mapping.pop(item.id, None)
        st.session_state["iw_column_mapping"] = mapping

        mapped_columns = set(mapping.values())
        remaining_columns = [c for c in columns if c not in mapped_columns]
        st.session_state["iw_demographic_columns"] = st.multiselect(
            "Demografiska/övriga kolumner att spara med datasetet",
            remaining_columns,
            default=[c for c in remaining_columns if c in st.session_state["iw_demographic_columns"]],
        )

    st.write("")
    col_back, _, col_next = st.columns([1, 3, 1])
    with col_back:
        if st.button("Tillbaka", use_container_width=True):
            go_to(3)
    with col_next:
        n_mapped = len(st.session_state["iw_column_mapping"])
        ready = questionnaire is not None and n_mapped > 0
        if st.button("Nästa →", type="primary", use_container_width=True, disabled=not ready):
            go_to(5)

# --- Step 5: Slutför -------------------------------------------------
elif step == 5:
    df = st.session_state["iw_raw_df"]
    questionnaire = st.session_state["iw_questionnaire"]
    mapping = st.session_state["iw_column_mapping"]
    demographic_columns = st.session_state["iw_demographic_columns"]

    dataset = build_dataset(
        raw=df,
        questionnaire=questionnaire,
        column_mapping=mapping,
        demographic_columns=demographic_columns,
        name=st.session_state["iw_source_name"] or questionnaire.test_name,
    )
    st.session_state["pve_dataset"] = dataset

    st.success(f"Dataset skapat: **{questionnaire.test_name}** med {dataset.n} deltagare.")

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Deltagare", dataset.n)
    kpi_cols[1].metric("Items", dataset.n_items)
    kpi_cols[2].metric("Delskalor", dataset.n_subscales)
    kpi_cols[3].metric("Total bortfall", f"{dataset.missing_pct:.1f}%")

    st.write("**Poängsatt data (förhandsvisning)**")
    st.dataframe(dataset.scored.head(10), use_container_width=True)

    st.write("")
    col_back, _, col_next = st.columns([1, 3, 1])
    with col_back:
        if st.button("Tillbaka", use_container_width=True):
            go_to(4)
    with col_next:
        if st.button("Till Dataset Overview →", type="primary", use_container_width=True):
            st.switch_page("pages/2_Dataset_Overview.py")
