"""Reliability Explorer page. Rendering only - all statistics come from core/."""

import pandas as pd
import streamlit as st

from components.concept_tooltip import concept_tooltip
from components.export_section import render_export_section
from components.kpi_card import kpi_card
from core import stats_engine as se
from core import viz_engine as ve
from utils.session import require_dataset

dataset = require_dataset()
q = dataset.questionnaire

st.title("Reliability Explorer")
st.caption(f"Analysera intern konsistens och reliabilitet för {q.test_name}")

overall = se.reliability_snapshot(dataset)
ci_low, ci_high = se.alpha_confidence_interval(overall.alpha, overall.n, overall.n_items) if overall.alpha else (None, None)
mean_r = se.mean_inter_item_correlation(dataset)

# --- KPI row -------------------------------------------------
kpi_cols = st.columns(4)
with kpi_cols[0]:
    kpi_card(
        "📈", "#DBEAFE",
        f"{overall.alpha:.2f}" if overall.alpha is not None else "–",
        "Cronbach's alpha",
        caption=se.alpha_interpretation(overall.alpha),
        tooltip=(
            "Mäter intern konsistens - hur väl items i skalan samvarierar. Tumregel: ≥0.90 "
            "utmärkt, ≥0.80 bra, ≥0.70 acceptabelt, <0.70 bör granskas."
        ),
    )
with kpi_cols[1]:
    kpi_card(
        "📋", "#D1FAE5", str(overall.n_items), "Items (frågor)", caption=f"{len(q.reverse_scored_ids)} omvänt",
        tooltip="Antal frågor (items) som ingår i skalan. 'Omvänt' anger hur många av dem som poängsätts baklänges (t.ex. en positivt formulerad fråga i ett ångesttest).",
    )
with kpi_cols[2]:
    kpi_card(
        "⚖️", "#EDE9FE", f"{mean_r:.2f}" if mean_r is not None else "–", "Medel inter-item r",
        tooltip="Genomsnittlig korrelation mellan alla par av items. Bör ligga runt 0.15-0.50 - för lågt tyder på att items mäter olika saker, för högt tyder på onödigt upprepade frågor.",
    )
with kpi_cols[3]:
    ci_text = f"{ci_low:.2f} – {ci_high:.2f}" if ci_low is not None else "–"
    kpi_card(
        "📊", "#FFEDD5", ci_text, "95% KI (alpha)", caption="Konfidensintervall",
        tooltip="Intervallet där det sanna alpha-värdet troligen ligger, baserat på antal items och urvalsstorlek. Ett smalt intervall betyder en mer precis skattning.",
    )

st.write("")

tab_overview, tab_item, tab_scale, tab_split, tab_retest = st.tabs(
    ["Översikt", "Item-reliabilitet", "Skala-reliabilitet", "Split-half", "Tidsstabilitet"]
)

item_stats = se.item_level_table(dataset)
item_df = pd.DataFrame(
    {
        "item": [i.item_id for i in item_stats],
        "item_total_r": [i.item_total_r for i in item_stats],
        "alpha_if_deleted": [i.alpha_if_deleted for i in item_stats],
        "reverse": [i.reverse_scored for i in item_stats],
    }
).set_index("item")

with tab_overview:
    col_r, col_alpha = st.columns(2)
    with col_r:
        header_cols = st.columns([5, 1])
        header_cols[0].markdown("**Item-total korrelationer**")
        with header_cols[1]:
            concept_tooltip(
                "Item-total korrelation",
                "Korrelationen mellan ett items poäng och summan av övriga items. Låga värden "
                "(<0.30) tyder på att itemet mäter något annat än resten av skalan.",
            )
        r_series = item_df["item_total_r"].dropna()
        item_total_fig = None
        if not r_series.empty:
            item_total_fig = ve.horizontal_bar_chart(r_series, "Item-total korrelation", x_range=(0, 1))
            st.plotly_chart(item_total_fig, width="stretch")
        else:
            st.info("För få data för item-total korrelationer.")

    with col_alpha:
        header_cols = st.columns([5, 1])
        header_cols[0].markdown("**Cronbach's alpha om item tas bort**")
        with header_cols[1]:
            concept_tooltip(
                "Alpha om item tas bort",
                "Vad Cronbach's alpha skulle bli om detta item togs bort ur skalan. Ett värde "
                "högre än nuvarande alpha betyder att itemet försämrar reliabiliteten.",
            )
        rows = [{"Aktuellt": "Alla items", "Alpha": round(overall.alpha, 3) if overall.alpha else "–"}]
        for item_id, row in item_df.iterrows():
            rows.append({"Aktuellt": item_id, "Alpha": round(row["alpha_if_deleted"], 3) if pd.notna(row["alpha_if_deleted"]) else "–"})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.write("")
    worsening = [i for i in item_stats if i.alpha_if_deleted is not None and overall.alpha is not None and i.alpha_if_deleted > overall.alpha]
    if worsening:
        names = ", ".join(i.item_id for i in worsening)
        st.warning(f"⚠️ {len(worsening)} item(er) skulle höja alpha om de togs bort: {names}.")
    else:
        st.success("✅ Reliabiliteten är god. Inget item försämrar skalan om det tas bort.")

with tab_item:
    st.markdown("**Item-total korrelationer och tolkning**")
    item_total_rows = []
    for i in item_stats:
        item_total_rows.append(
            {
                "Item": i.item_id,
                "Fråga": i.text,
                "r (item-total)": round(i.item_total_r, 2) if i.item_total_r is not None else "–",
                "Tolkning": se.item_total_interpretation(i.item_total_r),
                "Alpha om borttaget": round(i.alpha_if_deleted, 3) if i.alpha_if_deleted is not None else "–",
                "Förändring": (
                    round(i.alpha_if_deleted - overall.alpha, 3)
                    if i.alpha_if_deleted is not None and overall.alpha is not None
                    else "–"
                ),
            }
        )
    item_total_df = pd.DataFrame(item_total_rows)
    st.dataframe(item_total_df, width="stretch", hide_index=True)

with tab_scale:
    header_cols = st.columns([5, 1])
    header_cols[0].markdown("**Reliabilitet per delskala**")
    with header_cols[1]:
        concept_tooltip(
            "Cronbach's alpha vs McDonald's omega",
            "Alpha antar att alla items väger lika mycket; omega bygger på en faktormodell och "
            "är ofta ett mer robust mått, särskilt när items varierar i hur starkt de laddar på konstruktet.",
        )
    rows = []
    for snapshot in se.reliability_snapshot_all_subscales(dataset):
        omega = se.mcdonald_omega(dataset, snapshot.subscale_id)
        rows.append(
            {
                "Delskala": snapshot.subscale_name,
                "Items": snapshot.n_items,
                "N": snapshot.n,
                "Alpha": round(snapshot.alpha, 3) if snapshot.alpha is not None else "–",
                "Omega": round(omega, 3) if omega is not None else "–",
            }
        )
    overall_omega = se.mcdonald_omega(dataset)
    rows.append(
        {
            "Delskala": "Total testskala",
            "Items": overall.n_items,
            "N": overall.n,
            "Alpha": round(overall.alpha, 3) if overall.alpha is not None else "–",
            "Omega": round(overall_omega, 3) if overall_omega is not None else "–",
        }
    )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

with tab_split:
    st.markdown("**Split-half-reliabilitet (odd-even)**")
    scope_options = {"Alla items": None}
    if dataset.n_subscales > 1:
        scope_options.update({s.name: s.id for s in q.subscales})
    scope_label = st.selectbox("Omfattning", scope_options.keys(), key="split_half_scope")
    result = se.split_half_reliability(dataset, scope_options[scope_label])

    kpi_row = st.columns(3)
    kpi_row[0].metric("Korrelationskoefficient (r)", f"{result.r_raw:.2f}" if result.r_raw is not None else "–")
    kpi_row[1].metric("Spearman-Brown (korrigerad)", f"{result.spearman_brown:.2f}" if result.spearman_brown is not None else "–")
    kpi_row[2].metric("Guttman split-half", f"{result.guttman:.2f}" if result.guttman is not None else "–")
    st.caption(
        "Metod: items delas i udda/jämna grupper. Spearman-Brown korrigerar för att varje "
        "halva har färre items än hela skalan; Guttmans lambda antar inte lika varians i halvorna."
    )

with tab_retest:
    st.markdown("**Test-retest-reliabilitet**")
    scope_options = {"Alla items": None}
    if dataset.n_subscales > 1:
        scope_options.update({s.name: s.id for s in q.subscales})
    scope_label = st.selectbox("Omfattning", scope_options.keys(), key="retest_scope")
    subscale_id = scope_options[scope_label]
    result = se.test_retest_reliability(dataset, subscale_id)

    if result.r is None:
        st.info("Ingen retest-data tillgänglig för detta urval.")
    else:
        kpi_row = st.columns(3)
        kpi_row[0].metric("Korrelationskoefficient (r)", f"{result.r:.2f}")
        kpi_row[1].metric("Antal personer", result.n)
        ci_text = f"{result.ci_low:.2f} – {result.ci_high:.2f}" if result.ci_low is not None else "–"
        kpi_row[2].metric("95% KI", ci_text)

        t1_total, t2_total = se.test_retest_paired_scores(dataset, subscale_id)
        trendline = se.linear_fit(t1_total, t2_total)
        st.plotly_chart(
            ve.scatter_with_regression(t1_total, t2_total, "Tid 1 (summa poäng)", "Tid 2 (summa poäng)", trendline),
            width="stretch",
        )
        st.caption(
            "Simulerad retest-data: en delmängd av deltagarna besvarade testet igen ~2 veckor "
            "senare. Genereras av data/generate_sample_data.py för demonstrationssyfte."
        )

st.write("")
render_export_section(
    dataset,
    "reliability",
    table=item_total_df,
    table_label="Item-total korrelationer",
    figures={"item_total_korrelation": item_total_fig} if item_total_fig is not None else None,
)

st.write("")
col_back, _, col_next = st.columns([1, 3, 1])
with col_back:
    if st.button("Tillbaka", width="stretch"):
        st.switch_page("pages/4_Test_Profile_Explorer.py")
with col_next:
    if st.button("Fortsätt till Factor Explorer →", type="primary", width="stretch"):
        st.switch_page("pages/6_Factor_Explorer.py")
