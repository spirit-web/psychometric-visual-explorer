"""Factor Explorer page. Rendering only - all statistics come from core/."""

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

st.title("Factor Explorer")
st.caption(f"Utforska underliggande faktorer i {q.test_name}")

pa = se.parallel_analysis(dataset)
adequacy = se.sampling_adequacy(dataset)

default_k = max(1, len(q.subscales)) if q.subscales else max(1, pa.suggested_n_factors)
max_k = max(1, min(dataset.n_items - 2, 10))
n_factors = st.number_input(
    "Antal faktorer att extrahera",
    min_value=1,
    max_value=max_k,
    value=min(default_k, max_k),
    help="Förvalt värde är antalet delskalor i testdefinitionen. Parallel analysis nedan ger ett "
    "empiriskt förslag baserat på dina data.",
)
st.caption(
    "Varför skiljer sig antalet faktorer mellan test? Dels beror det på hur testet är konstruerat "
    "- t.ex. är IPIP Big Five designat för att mäta 5 oberoende dimensioner, medan GAD-7 är "
    "designat som en enda dimension (ångest). Dels beror det på vad dina data faktiskt visar, via "
    "Parallel analysis nedan. Om det föreslagna antalet skiljer sig från testets originaldesign kan "
    "det bero på urvalsstorlek eller sammansättning - testa gärna olika värden ovan."
)
efa = se.efa_fit(dataset, int(n_factors))

st.write("")

# --- KPI row -------------------------------------------------
kpi_cols = st.columns(4)
with kpi_cols[0]:
    kpi_card(
        "🧩", "#DBEAFE", str(pa.suggested_n_factors), "Föreslagna faktorer", caption="Parallel analysis",
        tooltip="Det statistiskt motiverade antalet faktorer att behålla, baserat på jämförelse med slumpdata av samma storlek. Använd detta som riktmärke för värdet ovan.",
        learning_key="Parallel analysis",
    )
with kpi_cols[1]:
    explained = f"{efa.variance_explained['Cumulative Var'].iloc[-1] * 100:.0f}%" if efa is not None else "–"
    kpi_card(
        "📊", "#D1FAE5", explained, "Förklarad varians", caption="av total varians",
        tooltip="Hur stor andel av all variation i svaren som de extraherade faktorerna tillsammans fångar upp. Högre är bättre, men sällan över 60-70% ens för bra test.",
    )
with kpi_cols[2]:
    kmo_text = f"{adequacy.kmo:.2f}" if adequacy.kmo is not None else "–"
    kpi_card(
        "✅", "#EDE9FE", kmo_text, "KMO", caption="Sampling adequacy",
        tooltip="Kaiser-Meyer-Olkin-måttet: om datan lämpar sig för faktoranalys överhuvudtaget. ≥0.80 bra, ≥0.60 acceptabelt, under det bör faktoranalys tolkas med försiktighet.",
        learning_key="KMO & Bartlett's test",
    )
with kpi_cols[3]:
    bartlett_text = "< .001" if adequacy.bartlett_p is not None and adequacy.bartlett_p < 0.001 else (
        f"{adequacy.bartlett_p:.3f}" if adequacy.bartlett_p is not None else "–"
    )
    kpi_card(
        "🔬", "#FFEDD5", bartlett_text, "Bartlett's test (p)", caption="Signifikant om < .05",
        tooltip="Testar om frågorna korrelerar tillräckligt med varandra för att faktoranalys ska vara meningsfull. Ett lågt p-värde (< .05) är det önskade utfallet här.",
        learning_key="KMO & Bartlett's test",
    )

st.write("")

tab_overview, tab_loadings, tab_corr, tab_fit = st.tabs(["Översikt", "Faktorladdningar", "Korrelationer", "Model Fit"])

with tab_overview:
    col_summary, col_loadings = st.columns(2)
    with col_summary:
        header_cols = st.columns([5, 1])
        header_cols[0].markdown("**Sammanfattning av faktorstruktur**")
        with header_cols[1]:
            concept_tooltip(
                "Parallel analysis",
                "Jämför egenvärden från dina data med egenvärden från slumpmässiga data av samma "
                "storlek. Behåll faktorer vars egenvärde överstiger slumpdatans - ett mer robust "
                "kriterium än Kaisers regel (egenvärde > 1).",
                learning_key="Parallel analysis",
            )
        if efa is not None:
            rotation_text = "Ingen (1 faktor)" if efa.n_factors == 1 else "Promax (oblik)"
            st.markdown(
                f"""
                <div class="pve-card" style="padding:1rem 1.3rem;">
                  <div style="display:flex; justify-content:space-between; padding:0.4rem 0;
                      border-bottom:1px solid #F0F1F5;">
                    <span style="color:#6B7280;">Antal extraherade faktorer</span><b>{efa.n_factors}</b>
                  </div>
                  <div style="display:flex; justify-content:space-between; padding:0.4rem 0;
                      border-bottom:1px solid #F0F1F5;">
                    <span style="color:#6B7280;">Extraktionsmetod</span><b>Maximum Likelihood</b>
                  </div>
                  <div style="display:flex; justify-content:space-between; padding:0.4rem 0;">
                    <span style="color:#6B7280;">Rotation</span><b>{rotation_text}</b>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Otillräcklig data för faktoranalys.")

    with col_loadings:
        st.markdown("**Faktorladdningar (kompakt)**")
        if efa is not None:
            preview = efa.loadings.copy()
            preview.insert(0, "Fråga", [next(i.text for i in q.items if i.id == iid) for iid in preview.index])
            st.dataframe(preview.round(2), width="stretch", height=320)
        else:
            st.info("Ingen EFA-lösning tillgänglig.")

    st.write("")
    header_cols = st.columns([5, 1])
    header_cols[0].markdown("**Vilka frågor tillhör varje faktor?**")
    with header_cols[1]:
        concept_tooltip(
            "Faktortillhörighet",
            "Varje fråga listas under den faktor den laddar starkast på (primär faktor). "
            "Laddningen inom parentes visar hur starkt - högre absolutbelopp betyder att frågan är "
            "mer representativ för just den faktorn. En svag laddning (<0.30) betyder att frågan "
            "inte passar särskilt väl in i någon faktor.",
        )
    if efa is not None:
        factor_groups = se.factor_item_groups(efa, q)
        group_cols = st.columns(min(len(factor_groups), 3))
        for i, (factor_name, items) in enumerate(factor_groups):
            with group_cols[i % len(group_cols)]:
                st.markdown(f"**Faktor {i + 1} ({factor_name})**")
                for item_id, text, loading in items:
                    weak = " ⚠️" if abs(loading) < 0.30 else ""
                    st.caption(f"{text} ({loading:+.2f}){weak}")
    else:
        st.info("Ingen EFA-lösning tillgänglig.")

    st.write("")
    col_scree, col_fit = st.columns(2)
    with col_scree:
        header_cols = st.columns([5, 1])
        header_cols[0].markdown("**Scree plot & Parallel analysis**")
        with header_cols[1]:
            concept_tooltip(
                "Scree plot",
                "Visar egenvärdena i fallande ordning. Antalet faktorer att behålla är där kurvan "
                "planar ut (\"armbågen\"), eller där den ligger över parallel analysis-linjen.",
                learning_key="Scree plot",
            )
        scree_fig = None
        if len(pa.eigenvalues):
            scree_fig = ve.scree_plot_chart(pa.eigenvalues, pa.simulated_eigenvalues)
            st.plotly_chart(scree_fig, width="stretch")
        else:
            st.info("Otillräcklig data för scree plot.")

    with col_fit:
        header_cols = st.columns([5, 1])
        header_cols[0].markdown("**Modellanpassning**")
        with header_cols[1]:
            concept_tooltip(
                "Modellanpassning",
                "Hur väl den valda faktorlösningen stämmer med de observerade korrelationerna. "
                "God passning kräver RMSEA < 0.08 OCH (om beräknat) CFI ≥ 0.90 - annars visas "
                "\"Måttlig\", vilket ofta betyder att ett annat antal faktorer bör provas.",
            )
        if efa is not None and efa.fit.rmsea is not None:
            good_fit = efa.fit.rmsea < 0.08 and (efa.fit.cfi is None or efa.fit.cfi >= 0.90)
            if good_fit:
                st.success("✅ God modellpassning. Modellen visar bra passform till data.")
            else:
                st.warning("⚠️ Måttlig modellpassning. Överväg ett annat antal faktorer.")
            fit_cols = st.columns(4)
            fit_cols[0].metric("RMSEA", f"{efa.fit.rmsea:.3f}")
            fit_cols[1].metric("CFI", f"{efa.fit.cfi:.2f}" if efa.fit.cfi is not None else "–")
            fit_cols[2].metric("TLI", f"{efa.fit.tli:.2f}" if efa.fit.tli is not None else "–")
            fit_cols[3].metric("SRMR", f"{efa.fit.srmr:.3f}" if efa.fit.srmr is not None else "–")
        else:
            st.info("Modellanpassning kunde inte beräknas (för få frihetsgrader eller data).")

with tab_loadings:
    header_cols = st.columns([5, 1])
    header_cols[0].markdown("**Faktorladdningar och kommunaliteter**")
    with header_cols[1]:
        concept_tooltip(
            "Kommunalitet",
            "Andelen varians i en fråga som förklaras av de extraherade faktorerna tillsammans. "
            "Låg kommunalitet (<0.30) betyder att frågan till stor del mäter något faktorerna inte fångar.",
            learning_key="Kommunalitet",
        )
    if efa is not None:
        table = efa.loadings.copy()
        table["Kommunalitet"] = efa.communalities.round(2)
        table.insert(0, "Fråga", [next(i.text for i in q.items if i.id == iid) for iid in table.index])
        loadings_export_df = table.round(2).reset_index().rename(columns={"index": "Item"})
        st.dataframe(table.round(2), width="stretch")

        st.write("")
        st.markdown("**Faktorladdningar (heatmap)**")
        st.plotly_chart(ve.correlation_heatmap(efa.loadings), width="stretch", key="loadings_heatmap")

        st.write("")
        st.markdown("**Förklarad varians per faktor**")
        st.dataframe((efa.variance_explained * 100).round(1).rename(columns=lambda c: f"{c} (%)"), width="stretch")
    else:
        st.info("Ingen EFA-lösning tillgänglig.")

with tab_corr:
    header_cols = st.columns([5, 1])
    header_cols[0].markdown("**Korrelationer mellan faktorer**")
    with header_cols[1]:
        concept_tooltip(
            "Faktorkorrelationer",
            "Vid oblik rotation (flera faktorer) tillåts faktorerna korrelera med varandra, precis "
            "som verkliga personlighets- eller symtomdimensioner ofta gör.",
        )
    if efa is not None and efa.phi is not None:
        st.plotly_chart(ve.correlation_heatmap(efa.phi), width="stretch", key="phi_heatmap")
        st.dataframe(efa.phi.round(2), width="stretch")
    else:
        st.info("Endast en faktor extraherad - inga faktorkorrelationer att visa.")

with tab_fit:
    st.markdown("**Modellanpassning i detalj**")
    if efa is not None and efa.fit.rmsea is not None:
        rows = [
            {"Mått": "Chi-square (χ²)", "Värde": f"{efa.fit.chi2:.1f}", "Tolkning": "Lägre är bättre (känsligt för N)"},
            {"Mått": "df", "Värde": str(efa.fit.df), "Tolkning": "Frihetsgrader"},
            {"Mått": "RMSEA", "Värde": f"{efa.fit.rmsea:.3f}", "Tolkning": "< 0.06 utmärkt, < 0.08 acceptabelt"},
            {"Mått": "CFI", "Värde": f"{efa.fit.cfi:.3f}" if efa.fit.cfi is not None else "–", "Tolkning": "> 0.95 utmärkt, > 0.90 acceptabelt"},
            {"Mått": "TLI", "Värde": f"{efa.fit.tli:.3f}" if efa.fit.tli is not None else "–", "Tolkning": "> 0.95 utmärkt, > 0.90 acceptabelt"},
            {"Mått": "SRMR", "Värde": f"{efa.fit.srmr:.3f}" if efa.fit.srmr is not None else "–", "Tolkning": "< 0.08 acceptabelt"},
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(
            "Beräknat via maximum-likelihood-diskrepans mellan modellimplicerad och observerad "
            "korrelationsmatris, jämfört med en oberoende-modell (0 faktorer) som baslinje - samma "
            "princip som CFA-modellanpassning (Hu & Bentler, 1999)."
        )
    else:
        st.info("Modellanpassning kräver fler frihetsgrader (fler frågor relativt antal faktorer) eller mer data.")

st.write("")
render_export_section(
    dataset,
    "factor_explorer",
    table=loadings_export_df if efa is not None else None,
    table_label="Faktorladdningar",
    figures={"scree_plot": scree_fig} if scree_fig is not None else None,
)

st.write("")
col_back, _, col_next = st.columns([1, 3, 1])
with col_back:
    if st.button("Tillbaka", width="stretch"):
        st.switch_page("pages/5_Reliability_Explorer.py")
with col_next:
    if st.button("Fortsätt till Validity Dashboard →", type="primary", width="stretch"):
        st.switch_page("pages/7_Validity_Dashboard.py")
