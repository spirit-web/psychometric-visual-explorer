"""Shared 'whose score is this?' widget: pick an existing row from the
loaded dataset, or enter one new client's item answers by hand. Used by
Norm Explorer and Fairness Explorer so a client doesn't need to already be
a row in the dataset to be placed against it. Rendering only - scoring
logic lives in core/stats_engine.score_manual_responses."""

import streamlit as st

from core import stats_engine as se


def select_or_enter_client(dataset, subscale_id: str | None, key_prefix: str) -> float | None:
    """Renders the person-selector/manual-entry UI and returns the raw
    subscale score for whichever client the user selected or described."""
    q = dataset.questionnaire
    source = st.radio(
        "Vems resultat?",
        ["Person ur datasetet", "En ny klients svar (manuellt)"],
        horizontal=True,
        key=f"{key_prefix}_source",
    )

    if source == "Person ur datasetet":
        ids = se.person_ids(dataset)
        person_id = st.selectbox("Välj person", ids, format_func=lambda pid: f"Person {pid}", key=f"{key_prefix}_person")
        return se.person_raw_score(dataset, person_id, subscale_id)

    st.caption("Ange klientens svar på varje fråga - klienten behöver inte redan finnas i det inlästa datasetet.")
    subscale = q.get_subscale(subscale_id) if subscale_id else (q.subscales[0] if q.subscales else None)
    item_ids = subscale.item_ids if subscale else []
    items_by_id = {item.id: item for item in q.items}
    scale = q.response_scale
    value_options = list(range(scale.min, scale.max + 1))
    responses = {}
    cols = st.columns(2)
    for i, item_id in enumerate(item_ids):
        item = items_by_id.get(item_id)
        if item is None:
            continue
        with cols[i % 2]:
            responses[item_id] = st.selectbox(
                f"{item.id} – {item.text}",
                value_options,
                format_func=lambda v: scale.labels.get(str(v), str(v)),
                key=f"{key_prefix}_manual_{item_id}",
            )
    return se.score_manual_responses(dataset, responses, subscale_id)
