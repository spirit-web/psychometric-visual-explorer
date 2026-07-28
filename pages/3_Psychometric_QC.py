"""Psychometric QC page. Rendering only - all statistics come from core/."""

import pandas as pd
import streamlit as st

from components.concept_tooltip import concept_tooltip
from components.export_section import render_export_section
from components.kpi_card import kpi_card, status_icon
from core import stats_engine as se
from core import viz_engine as ve
from utils.session import require_dataset

dataset = require_dataset()
q = dataset.questionnaire

st.title("Psychometric QC")
st.caption(f"Automatisk kvalitetskontroll av ditt {q.test_name}-dataset")

summary = se.quality_summary(dataset)
missing_series = summary.missing_by_item

# --- KPI row -------------------------------------------------
kpi_cols = st.columns(4)
with kpi_cols[0]:
    kpi_card(
        "✅",
        "#D1FAE5",
        f"{summary.completeness.complete_pct:.1f}%",
        "Kompletta svar",
        caption=f"{summary.completeness.complete:,} av {summary.completeness.n:,}".replace(",", " "),
        tooltip="Andel deltagare som besvarat samtliga frågor i testet, utan några saknade värden. Ett "
        "lågt värde kan bero på ett långt formulär, ett känsligt ämne eller ett tekniskt problem vid "
        "insamlingen.",
    )
with kpi_cols[1]:
    if not missing_series.empty:
        interval = f"{missing_series.min():.1f}% – {missing_series.max():.1f}%"
    else:
        interval = "–"
    kpi_card(
        "⏱️", "#FFEDD5", interval, "Bortfall per fråga", caption="Intervall",
        tooltip="Intervallet mellan den fråga med lägst och den med högst andel obesvarade svar. Höga "
        "värden (>5%) på enstaka frågor kan indikera en otydlig eller känslig fråga - se grafen till "
        "vänster för vilken fråga det gäller.",
    )
with kpi_cols[2]:
    kpi_card(
        "⚠️", "#FEF3C7", str(summary.pattern_flags.total), "Mönsterflaggor", caption="Möjliga mönster",
        tooltip="Antal deltagare som svarat likadant (eller nästan likadant) på alla frågor, t.ex. "
        "alltid \"2\". Kan tyda på slarviga svar, men kan också vara ett korrekt mönster för en person "
        "med en verkligt jämn nivå. Går inte att åtgärda i efterhand - se det som något att vara "
        "medveten om, inte ett fel att rätta till.",
    )
with kpi_cols[3]:
    kpi_card(
        "🛑", "#FEE2E2", str(summary.n_flagged_checks), "Problemflaggor", caption="Se Kontroller nedan",
        tooltip="Antal av de 6 kontrollerna nedan som fick status varning (⚠️) eller problem (🛑). Se "
        "tabellen \"Kontroller\" för vilka det gäller. Endast 🛑 bör aktivt granskas (t.ex. dubbletter) "
        "- ⚠️ är oftare en egenskap hos urvalet än ett fel i datan.",
    )

st.write("")

# --- Missing per item chart / Kontroller table -------------------------------------------------
col_missing, col_checks = st.columns(2)
with col_missing:
    st.markdown("**Bortfall per fråga (%)**")
    if not missing_series.empty:
        missing_fig = ve.missing_by_item_chart(missing_series)
        st.plotly_chart(missing_fig, width="stretch")
    else:
        missing_fig = None
        st.info("Inga frågor att visa.")

with col_checks:
    header_cols = st.columns([5, 1])
    header_cols[0].markdown("**Kontroller**")
    with header_cols[1]:
        concept_tooltip(
            "Kvalitetskontroller",
            "En varningstriangel (⚠️) betyder att något är värt att känna till - oftast en egenskap "
            "hos urvalet, inte ett fel. En stoppskylt (🛑) betyder att det är värt att aktivt granska, "
            "eftersom det ofta beror på ett fel i datainsamlingen eller exporten.\n\n"
            "- **Konstanta frågor** 🛑 - alla svarade exakt likadant (ingen variation alls). Tyder ofta "
            "på ett tekniskt fel vid export eller kodning.\n"
            "- **Låg varians frågor** ⚠️ - nästan alla svarade likadant. Gör frågan mindre användbar "
            "statistiskt, men är inte ett fel i sig.\n"
            "- **Golveffekt** ⚠️ - många svarade med lägsta möjliga poäng. Vanligt i friska/normal-"
            "populationer på kliniska test.\n"
            "- **Takeffekt** ⚠️ - många svarade med högsta möjliga poäng. Kan begränsa hur väl testet "
            "särskiljer höga nivåer.\n"
            "- **Outliers (univariata)** ⚠️ - enstaka deltagare med en totalpoäng som avviker kraftigt "
            "(|z| > 3,29) från övriga. Värt att dubbelkolla att det inte är en inmatningsmiss.\n"
            "- **Dubbletter (ID)** 🛑 - samma deltagare eller identiska svarsrader förekommer flera "
            "gånger. Bör tas bort innan vidare analys, annars räknas de dubbelt.",
        )
    checks_df = pd.DataFrame(
        [{"Kontroll": c.name, "Status": status_icon(c.status), "Kommentar": c.comment} for c in summary.checks]
    )
    st.dataframe(checks_df, width="stretch", hide_index=True)

st.write("")

if summary.status == "good":
    st.success(f"✅ {summary.message}")
elif summary.status == "warning":
    st.warning(f"⚠️ {summary.message}")
else:
    st.error(f"🛑 {summary.message}")

st.write("")
render_export_section(
    dataset,
    "psychometric_qc",
    table=checks_df,
    table_label="Kvalitetskontroller",
    figures={"bortfall_per_item": missing_fig} if missing_fig is not None else None,
)

st.write("")
col_back, _, col_next = st.columns([1, 3, 1])
with col_back:
    if st.button("Tillbaka", width="stretch"):
        st.switch_page("pages/2_Dataset_Overview.py")
with col_next:
    if st.button("Fortsätt till Norm Explorer →", type="primary", width="stretch"):
        st.switch_page("pages/8_Norm_Explorer.py")
