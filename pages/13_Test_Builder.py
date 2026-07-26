"""Test Builder page. Rendering only - construction/save logic lives in
core/plugin_engine.py."""

import pandas as pd
import streamlit as st

from components.kpi_card import kpi_card
from core import plugin_engine as pe
from core import stats_engine as se
from utils.session import require_dataset

dataset = require_dataset()
q = dataset.questionnaire

st.title("Test Builder")
st.caption(f"Skapa och anpassa testdefinitionen för {q.test_name}")


def _questionnaire_to_tables(questionnaire):
    items_df = pd.DataFrame(
        [{"id": i.id, "text": i.text, "subscale": i.subscale, "reverse_scored": i.reverse_scored} for i in questionnaire.items]
    )
    subscales_df = pd.DataFrame(
        [
            {
                "id": s.id,
                "name": s.name,
                "item_ids": ", ".join(s.item_ids),
                "score_min": s.score_range[0],
                "score_max": s.score_range[1],
                "scoring_method": s.scoring_method,
            }
            for s in questionnaire.subscales
        ]
    )
    cutoffs_df = (
        pd.DataFrame([{"label": c.label, "range_min": c.range[0], "range_max": c.range[1]} for c in questionnaire.cutoffs])
        if questionnaire.cutoffs
        else pd.DataFrame(columns=["label", "range_min", "range_max"])
    )
    labels_df = (
        pd.DataFrame([{"value": k, "label": v} for k, v in questionnaire.response_scale.labels.items()])
        if questionnaire.response_scale.labels
        else pd.DataFrame(columns=["value", "label"])
    )
    return items_df, subscales_df, cutoffs_df, labels_df


def _load_working_copy(questionnaire, source_id: str) -> None:
    items_df, subscales_df, cutoffs_df, labels_df = _questionnaire_to_tables(questionnaire)
    st.session_state["tb_source_plugin_id"] = source_id
    st.session_state["tb_test_name"] = questionnaire.test_name
    st.session_state["tb_full_name"] = questionnaire.full_name
    st.session_state["tb_language"] = questionnaire.language
    st.session_state["tb_source_citation"] = questionnaire.source_citation or ""
    st.session_state["tb_scale_min"] = questionnaire.response_scale.min
    st.session_state["tb_scale_max"] = questionnaire.response_scale.max
    st.session_state["tb_scale_prompt"] = questionnaire.response_scale.prompt or ""
    st.session_state["tb_scale_labels_df"] = labels_df
    st.session_state["tb_items_df"] = items_df
    st.session_state["tb_subscales_df"] = subscales_df
    st.session_state["tb_cutoffs_df"] = cutoffs_df


# (Re)initialize the working copy whenever the active test changes.
if st.session_state.get("tb_source_plugin_id") != q.plugin_id:
    _load_working_copy(q, q.plugin_id)

col_reset1, col_reset2 = st.columns(2)
with col_reset1:
    if st.button(f"↩️ Återställ till {q.test_name}", width="stretch"):
        _load_working_copy(q, q.plugin_id)
        st.rerun()
with col_reset2:
    if st.button("🆕 Börja om från ett tomt test", width="stretch"):
        blank = pe.blank_questionnaire("nytt_test", "Nytt test")
        _load_working_copy(blank, "__blank__")
        st.rerun()

st.write("")

# --- KPI row -------------------------------------------------
kpi_cols = st.columns(4)
with kpi_cols[0]:
    kpi_card("📚", "#DBEAFE", st.session_state["tb_test_name"], "Testnamn (utkast)")
with kpi_cols[1]:
    kpi_card("📋", "#D1FAE5", str(len(st.session_state["tb_items_df"])), "Items i utkast")
with kpi_cols[2]:
    kpi_card("🧩", "#EDE9FE", str(len(st.session_state["tb_subscales_df"])), "Delskalor i utkast")
with kpi_cols[3]:
    kpi_card("🏷️", "#FFEDD5", q.plugin_id, "Källa (aktivt test)")

st.write("")

tab_shorten, tab_info, tab_items, tab_subscales, tab_cutoffs, tab_save = st.tabs(
    ["Korta ner testet", "Grundinfo & svarsskala", "Items", "Delskalor", "Cut-offs", "Spara"]
)

with tab_shorten:
    st.markdown("**Vilka frågor är säkrast att ta bort?**")
    st.caption(
        f"Baserat på verklig data för det aktiva testet ({q.test_name}) - visar vilka items som "
        "bidrar minst till reliabiliteten och därför är säkrast att ta bort om du vill korta ner testet."
    )
    item_stats = se.item_level_table(dataset)
    shorten_rows = [
        {
            "Item": i.item_id,
            "Fråga": i.text,
            "Item-total r": round(i.item_total_r, 2) if i.item_total_r is not None else None,
            "Alpha om borttaget": round(i.alpha_if_deleted, 3) if i.alpha_if_deleted is not None else None,
            "Tolkning": se.item_total_interpretation(i.item_total_r),
        }
        for i in item_stats
    ]
    shorten_df = pd.DataFrame(shorten_rows).sort_values("Item-total r", na_position="last")
    st.dataframe(shorten_df, width="stretch", hide_index=True)

    weak_items = [i.item_id for i in item_stats if i.item_total_r is not None and i.item_total_r < 0.30]
    if weak_items:
        st.warning(
            f"⚠️ {len(weak_items)} item(er) har låg item-total-korrelation (<0.30) och bidrar minst "
            f"till skalan: {', '.join(weak_items)}. Kandidater att ta bort om testet ska kortas ner."
        )
    else:
        st.success("✅ Alla items har rimlig item-total-korrelation (≥0.30) - inget uppenbart att ta bort.")

    to_remove = st.multiselect(
        "Välj items att ta bort från utkastet",
        [i.item_id for i in item_stats],
        default=weak_items,
        key="tb_shorten_selection",
    )
    if st.button("➖ Ta bort valda items från utkastet", disabled=not to_remove):
        items_df = st.session_state["tb_items_df"]
        st.session_state["tb_items_df"] = items_df[~items_df["id"].isin(to_remove)].reset_index(drop=True)

        def _strip_removed_ids(item_ids_str: str) -> str:
            remaining = [s.strip() for s in str(item_ids_str).split(",") if s.strip() and s.strip() not in to_remove]
            return ", ".join(remaining)

        subscales_df = st.session_state["tb_subscales_df"].copy()
        subscales_df["item_ids"] = subscales_df["item_ids"].apply(_strip_removed_ids)
        st.session_state["tb_subscales_df"] = subscales_df

        st.success(f"Tog bort {len(to_remove)} item(er) från utkastet. Gå till fliken Spara för att spara som en ny, kortare testdefinition.")
        st.rerun()

with tab_info:
    col1, col2 = st.columns(2)
    with col1:
        st.session_state["tb_test_name"] = st.text_input("Testnamn", st.session_state["tb_test_name"])
        st.session_state["tb_full_name"] = st.text_input("Fullständigt namn", st.session_state["tb_full_name"])
        lang_options = ["sv", "en"]
        current_lang = st.session_state["tb_language"] if st.session_state["tb_language"] in lang_options else "sv"
        st.session_state["tb_language"] = st.selectbox("Språk", lang_options, index=lang_options.index(current_lang))
        st.session_state["tb_source_citation"] = st.text_area("Källhänvisning (valfritt)", st.session_state["tb_source_citation"])
    with col2:
        st.session_state["tb_scale_min"] = st.number_input("Svarsskala - min", value=int(st.session_state["tb_scale_min"]), step=1)
        st.session_state["tb_scale_max"] = st.number_input("Svarsskala - max", value=int(st.session_state["tb_scale_max"]), step=1)
        st.session_state["tb_scale_prompt"] = st.text_input("Instruktion till testtagare (valfritt)", st.session_state["tb_scale_prompt"])
        st.markdown("**Svarsalternativ (etiketter per värde)**")
        st.session_state["tb_scale_labels_df"] = st.data_editor(
            st.session_state["tb_scale_labels_df"],
            num_rows="dynamic",
            width="stretch",
            key="tb_labels_editor",
            column_config={"value": "Värde", "label": "Etikett"},
        )

with tab_items:
    st.markdown("**Items**")
    st.caption("`subscale` måste matcha ett delskale-ID i fliken Delskalor.")
    st.session_state["tb_items_df"] = st.data_editor(
        st.session_state["tb_items_df"],
        num_rows="dynamic",
        width="stretch",
        key="tb_items_editor",
        column_config={
            "id": "Item-ID",
            "text": "Frågetext",
            "subscale": "Delskala-ID",
            "reverse_scored": st.column_config.CheckboxColumn("Omvänd"),
        },
    )

with tab_subscales:
    st.markdown("**Delskalor**")
    st.caption("`item_ids` är en kommaseparerad lista med item-ID:n som ingår i delskalan.")
    st.session_state["tb_subscales_df"] = st.data_editor(
        st.session_state["tb_subscales_df"],
        num_rows="dynamic",
        width="stretch",
        key="tb_subscales_editor",
        column_config={
            "id": "Delskala-ID",
            "name": "Namn",
            "item_ids": "Item-ID:n (kommaseparerat)",
            "score_min": "Min poäng",
            "score_max": "Max poäng",
            "scoring_method": "Poängmetod (sum/mean)",
        },
    )

with tab_cutoffs:
    st.markdown("**Cut-offs / diagnostiska nivåer**")
    st.caption("Valfritt - lämna tomt om testet inte använder fasta gränsvärden.")
    st.session_state["tb_cutoffs_df"] = st.data_editor(
        st.session_state["tb_cutoffs_df"],
        num_rows="dynamic",
        width="stretch",
        key="tb_cutoffs_editor",
        column_config={"label": "Nivå", "range_min": "Från", "range_max": "Till"},
    )

with tab_save:
    st.markdown("**Spara testdefinition**")
    st.info(
        "ℹ️ Ändringar sparas alltid som ett **nytt** plugin-id - det aktiva testets originalfil "
        "skrivs aldrig över automatiskt. Detta skyddar de medföljande testdefinitionerna "
        "(GAD-7, PHQ-9, IPIP Big Five)."
    )
    default_new_id = "nytt_test" if st.session_state["tb_source_plugin_id"] == "__blank__" else f"{q.plugin_id}_custom"
    new_plugin_id = st.text_input(
        "Nytt plugin-id",
        value=default_new_id,
        help="Gemener, siffror och understreck, minst 3 tecken, måste börja med en bokstav.",
    )
    overwrite = st.checkbox("Skriv över om detta plugin-id redan finns", value=False)

    if st.button("💾 Spara som ny testdefinition", type="primary"):
        scale_labels = {
            str(row["value"]).strip(): str(row["label"]).strip()
            for _, row in st.session_state["tb_scale_labels_df"].iterrows()
            if str(row.get("value", "")).strip()
        }
        questionnaire, error = pe.questionnaire_from_tables(
            plugin_id=new_plugin_id.strip(),
            test_name=st.session_state["tb_test_name"],
            full_name=st.session_state["tb_full_name"],
            language=st.session_state["tb_language"],
            source_citation=st.session_state["tb_source_citation"],
            scale_min=int(st.session_state["tb_scale_min"]),
            scale_max=int(st.session_state["tb_scale_max"]),
            scale_prompt=st.session_state["tb_scale_prompt"] or None,
            scale_labels=scale_labels,
            items_df=st.session_state["tb_items_df"],
            subscales_df=st.session_state["tb_subscales_df"],
            cutoffs_df=st.session_state["tb_cutoffs_df"],
        )
        if error:
            st.error(f"🛑 {error}")
        else:
            success, message = pe.save_plugin(questionnaire, overwrite=overwrite)
            if success:
                st.success(f"✅ {message} Testet är nu tillgängligt via Import Wizard nästa gång du laddar upp data.")
            else:
                st.error(f"🛑 {message}")

st.write("")
col_back, _, col_next = st.columns([1, 3, 1])
with col_back:
    if st.button("Tillbaka", width="stretch"):
        st.switch_page("pages/12_Machine_Learning.py")
with col_next:
    if st.button("Fortsätt till Learning Mode →", type="primary", width="stretch"):
        st.switch_page("pages/14_Learning_Mode.py")
