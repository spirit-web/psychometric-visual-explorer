"""Structured lesson content for Learning Mode - the exam-prep payload.

Sourced from docs/reference/psykometri_begrepp_concept_map.png ("Psykometri
- hela systemet på en sida"): the six pillars (Verkligheten -> Testutveckling
-> Validitet -> Reliabilitet -> Tolka poäng -> Beslut) plus Fairness, and the
formula quick-reference at the bottom of the concept map.

Pure content data - no Streamlit imports, no calculations.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConceptItem:
    term: str
    explanation: str


@dataclass
class FormulaItem:
    name: str
    latex: str
    description: str


@dataclass
class LearningModule:
    key: str
    title: str
    subtitle: str
    intro: str
    concepts: list[ConceptItem]
    formulas: list[FormulaItem] = field(default_factory=list)
    example: str = ""
    related_pages: list[str] = field(default_factory=list)


LEARNING_MODULES: list[LearningModule] = [
    LearningModule(
        key="verkligheten",
        title="1. Verkligheten",
        subtitle="Osynligt konstrukt",
        intro=(
            "Vi kan inte mäta osynliga psykologiska konstrukt (t.ex. depression, ångest, "
            "intelligens) direkt. Istället observerar vi ledtrådar - svar på testitems - och "
            "skapar test som tolkar dessa ledtrådar med hjälp av statistik."
        ),
        concepts=[
            ConceptItem(
                "Konstrukt",
                "Ett abstrakt psykologiskt begrepp som inte kan observeras direkt, t.ex. ångest, "
                "allmän begåvning eller extraversion.",
            ),
            ConceptItem(
                "Operationalisering",
                "Processen att definiera ett osynligt konstrukt genom mätbara indikatorer, t.ex. "
                "svar på specifika frågor i ett frågeformulär.",
            ),
            ConceptItem(
                "Latent variabel",
                "Den underliggande, icke-observerade variabeln (konstruktet) som ett tests items "
                "antas mäta gemensamt - motsvarar theta i faktoranalys.",
            ),
        ],
        related_pages=["Dataset Overview"],
    ),
    LearningModule(
        key="testutveckling",
        title="2. Testutveckling",
        subtitle="Från idé till test",
        intro=(
            "Ett psykometriskt test byggs stegvis: från item-generering, via pilottestning och "
            "faktoranalys, till ett färdigt och validerat instrument."
        ),
        concepts=[
            ConceptItem("Item development", "Att formulera kandidat-items som täcker konstruktets olika aspekter."),
            ConceptItem("Pilottest", "Ett första test av items på en mindre grupp för att identifiera problem innan full datainsamling."),
            ConceptItem(
                "EFA (Exploratory Factor Analysis)",
                "Utforskande faktoranalys - används tidigt för att upptäcka hur många underliggande "
                "faktorer items faktiskt grupperar sig i, utan att man i förväg bestämmer strukturen.",
            ),
            ConceptItem(
                "CFA (Confirmatory Factor Analysis)",
                "Bekräftande faktoranalys - testar om en i förväg specificerad faktorstruktur "
                "stämmer med observerad data. Används för att bekräfta EFA-resultat på nya data.",
            ),
            ConceptItem(
                "Parallel analysis",
                "Jämför egenvärden från de riktiga data med egenvärden från slumpmässiga data av "
                "samma storlek. Behåll faktorer vars egenvärde överstiger slumpdatans - ett mer "
                "robust kriterium för antal faktorer än Kaisers regel (egenvärde > 1).",
            ),
            ConceptItem(
                "Scree plot",
                "Visar egenvärdena i fallande ordning som en linjediagram. Antalet faktorer att "
                "behålla är ofta där kurvan planar ut (\"armbågen\"), eller där den korsar en "
                "parallel analysis-linje.",
            ),
            ConceptItem(
                "Kommunalitet",
                "Andelen varians i en fråga som förklaras av de extraherade faktorerna tillsammans. "
                "Låg kommunalitet (<0.30) betyder att frågan till stor del mäter något faktorerna "
                "inte fångar upp.",
            ),
            ConceptItem(
                "KMO & Bartlett's test",
                "Två kontroller av om data överhuvudtaget lämpar sig för faktoranalys, innan man "
                "tolkar resultatet. KMO (Kaiser-Meyer-Olkin) mäter sampling adequacy (≥0.80 bra, "
                "≥0.60 acceptabelt); Bartlett's test kontrollerar att frågorna korrelerar tillräckligt "
                "med varandra (signifikant, p < .05, är det önskade utfallet).",
            ),
            ConceptItem("Revidering", "Att ta bort eller justera items med dåliga psykometriska egenskaper och bygga ett färdigt test."),
        ],
        related_pages=["Factor Explorer"],
    ),
    LearningModule(
        key="validitet",
        title="3. Validitet",
        subtitle="Mäter testet rätt sak?",
        intro=(
            "Validitet handlar om evidens för tolkningen av testpoängen - inte en egenskap hos "
            "testet i sig, utan hos hur poängen används. Standards for Educational and "
            "Psychological Testing beskriver fem evidenskällor."
        ),
        concepts=[
            ConceptItem(
                "1. Innehåll (Content)",
                "Täcker testet det som konstruktet ska innehålla? Bygger vanligen på expertbedömningar "
                "av item-innehållet mot konstruktets definition.",
            ),
            ConceptItem(
                "2. Responsprocesser (Response Processes)",
                "Förstår och tolkar testpersoner frågorna på det sätt som avsetts? Undersöks med "
                "t.ex. kognitiva intervjuer eller tänka-högt-studier.",
            ),
            ConceptItem(
                "3. Intern struktur (Internal Structure)",
                "Har items den struktur som teorin förutsäger? Undersöks med faktoranalys (EFA/CFA) "
                "och reliabilitetsanalys.",
            ),
            ConceptItem(
                "4. Relation till andra variabler (Relations to Other Variables)",
                "Uppför sig testet som förväntat i relation till andra mått? Inkluderar samtidig "
                "(konkurrent) validitet, prediktiv validitet (förutsäger framtida utfall), "
                "konvergent validitet (korrelerar med liknande mått) och diskriminant validitet "
                "(låg korrelation med orelaterade konstrukt).",
            ),
            ConceptItem(
                "5. Konsekvenser (Consequences)",
                "Ger testanvändningen positiva konsekvenser och minimerar negativa - på individnivå "
                "(nytta/skada), gruppnivå (rättvisa) och samhällsnivå (resursfördelning, beslutskvalitet).",
            ),
        ],
        example=(
            "Exempel: Ett nytt ångesttest korrelerar r = 0.65 med ett etablerat ångestmått (konvergent "
            "validitet) och nära r = 0 med ett mått på matematikkunskap (diskriminant validitet) - "
            "detta stödjer att testet mäter ångest specifikt, inte generell testförmåga."
        ),
        related_pages=["Validity Dashboard"],
    ),
    LearningModule(
        key="reliabilitet",
        title="4. Reliabilitet",
        subtitle="Mäter testet stabilt?",
        intro=(
            "All mätning innehåller fel. Reliabilitet handlar om hur lite slumpmässigt mätfel som "
            "finns i testpoängen - inte om testet mäter \"rätt sak\" (det är validitet), utan om "
            "det mäter konsekvent."
        ),
        concepts=[
            ConceptItem(
                "Klassisk testteori (KTT)",
                "Grundmodellen: Observerad poäng (X) = Sann poäng (T) + Mätfel (E). Den sanna "
                "poängen kan aldrig observeras direkt, bara skattas.",
            ),
            ConceptItem(
                "Intern konsistens (Cronbach's alpha)",
                "Mäter hur väl items i en delskala samvarierar - hög alpha betyder att items mäter "
                "samma underliggande konstrukt.",
            ),
            ConceptItem(
                "Item-total korrelation",
                "Korrelationen mellan en enskild frågas poäng och summan av alla övriga frågor i "
                "skalan. Låga värden (<0.30) betyder att frågan mäter något annat än resten av "
                "skalan - en av de vanligaste anledningarna att ta bort eller revidera en fråga.",
            ),
            ConceptItem(
                "McDonald's omega",
                "Ett reliabilitetsmått baserat på en faktormodell, ofta mer robust än alpha "
                "eftersom det inte antar att alla items väger lika mycket.",
            ),
            ConceptItem("Test-retest-reliabilitet", "Korrelationen mellan poäng vid två tillfällen - mäter stabilitet över tid."),
            ConceptItem("Split-half-reliabilitet", "Testet delas i två halvor (t.ex. udda/jämna items) och halvornas poäng korreleras."),
            ConceptItem(
                "SEM (Standard Error of Measurement)",
                "Anger hur mycket den observerade poängen förväntas variera på grund av mätfel. "
                "Ju högre reliabilitet, desto lägre SEM.",
            ),
            ConceptItem(
                "Konfidensintervall (KI) för en poäng",
                "Intervallet där personens sanna poäng troligen ligger, baserat på SEM.",
            ),
            ConceptItem(
                "Reliable Change Index (RCI)",
                "RCI = (poäng vid tid 2 − poäng vid tid 1) / SE_diff, där SE_diff = SEM × √2. |RCI| ≥ "
                "1.96 tolkas som en statistiskt pålitlig förändring (Jacobson & Truax, 1991) - inte "
                "bara mätbrus. Används för att avgöra om en förändring hos en klient över tid är "
                "verklig eller inom det förväntade mätfelet.",
            ),
        ],
        formulas=[
            FormulaItem("Klassisk testteori", r"X = T + E", "Observerad poäng = Sann poäng + Mätfel"),
            FormulaItem("Reliabilitet", r"Rel = \frac{Var(T)}{Var(X)} = 1 - \frac{Var(E)}{Var(X)}", "Andel av observerad varians som är sann varians"),
            FormulaItem("Cronbach's alpha", r"\alpha = \frac{k}{k-1}\left(1 - \frac{\sum \sigma_i^2}{\sigma_x^2}\right)", "k = antal items, σᵢ² = itemvarians, σₓ² = testvarians"),
            FormulaItem("SEM", r"SEM = SD \times \sqrt{1 - \alpha}", "SD = standardavvikelse, α = reliabilitet"),
            FormulaItem("95% konfidensintervall", r"KI = X \pm 1{,}96 \times SEM", "X = observerad poäng"),
        ],
        example=(
            "Exempel: SD = 4, alpha = 0.84 → SEM = 4×√(1-0.84) = 1.6. En observerad poäng på 12 ger "
            "95% KI = 12 ± 1,96×1,6 = 8,9-15,1."
        ),
        related_pages=["Reliability Explorer", "Measurement Error"],
    ),
    LearningModule(
        key="tolka_poang",
        title="5. Tolka poäng",
        subtitle="Från råpoäng till meningsfull tolkning",
        intro=(
            "Samma information kan uttryckas på flera skalor. Standardisering gör det möjligt att "
            "jämföra en persons resultat med en normgrupp (referensgrupp)."
        ),
        concepts=[
            ConceptItem("Råpoäng (X)", "Den faktiska poängen på testet, t.ex. summan av alla items."),
            ConceptItem("Z-poäng", "Hur många standardavvikelser en persons poäng ligger från normgruppens medelvärde."),
            ConceptItem("T-poäng", "En standardiserad skala med medelvärde 50 och SD 10 - undviker negativa tal och decimaler."),
            ConceptItem("Percentil", "Andelen (%) i normgruppen som presterar lägre än personen."),
            ConceptItem("Stanine", "En niogradig skala (1-9) med medel 5 och SD 2, ofta använd för enkel kommunikation."),
            ConceptItem("Normgrupp (referensgrupp)", "Den grupp vars resultat används som jämförelsegrund - normernas kvalitet avgör tolkningens giltighet."),
        ],
        formulas=[
            FormulaItem("Z-poäng", r"z = \frac{X - M}{SD}", "M = normgruppens medelvärde, SD = normgruppens standardavvikelse"),
            FormulaItem("T-poäng", r"T = 50 + 10z", "Medel = 50, SD = 10"),
            FormulaItem("Z från T", r"z = \frac{T - 50}{10}", "Omvänd omvandling"),
            FormulaItem("Råpoäng från T", r"X = M + z \times SD", "Går tillbaka till ursprunglig skala"),
        ],
        example=(
            "Exempel: X = 42, M = 35, SD = 7 → z = (42-35)/7 = 1,0 → T = 50+10(1,0) = 60, vilket "
            "motsvarar ungefär 84:e percentilen (eftersom ca 84% av en normalfördelning ligger under z = 1)."
        ),
        related_pages=["Norm Explorer"],
    ),
    LearningModule(
        key="beslut",
        title="6. Beslut",
        subtitle="Vad betyder resultatet för individen?",
        intro=(
            "Diagnostiska eller praktiska beslut bygger ofta på gränsvärden (cut scores) som delar "
            "upp testpoängen i \"positiv\"/\"negativ\" eller riskkategorier."
        ),
        concepts=[
            ConceptItem("Cut score", "Gränsen mellan ett positivt och negativt testresultat. Valet påverkar avvägningen mellan sensitivitet och specificitet."),
            ConceptItem(
                "Konfusionsmatris",
                "En 2×2-tabell över hur testets prediktioner (positiv/negativ) stämmer mot det "
                "faktiska utfallet: sant positiva (TP), sant negativa (TN), falskt positiva (FP) "
                "och falskt negativa (FN). Grunden för sensitivitet, specificitet, PPV och NPV.",
            ),
            ConceptItem("Sensitivitet (TPR)", "Sannolikheten att testet korrekt identifierar de som verkligen har tillståndet."),
            ConceptItem("Specificitet (TNR)", "Sannolikheten att testet korrekt identifierar de som inte har tillståndet."),
            ConceptItem("False Negative (FN)", "Testet visar negativt trots att personen har tillståndet - risk: utebliven hjälp."),
            ConceptItem("False Positive (FP)", "Testet visar positivt trots att personen inte har tillståndet - risk: onödig oro eller åtgärd."),
            ConceptItem("PPV / NPV", "Positivt/negativt prediktivt värde - sannolikheten att ett positivt (PPV) eller negativt (NPV) testresultat är korrekt."),
            ConceptItem("ROC-kurva & AUC", "Visar avvägningen mellan sensitivitet och (1-specificitet) vid alla möjliga trösklar; AUC sammanfattar testets samlade förmåga att särskilja grupper."),
            ConceptItem("Youden's J", "J = Sensitivitet + Specificitet - 1. Tröskeln som maximerar J ger bästa balans mellan de två."),
        ],
        formulas=[
            FormulaItem("Sensitivitet", r"Sens = \frac{TP}{TP + FN}", "Andel verkligt sjuka som testet upptäcker"),
            FormulaItem("Specificitet", r"Spec = \frac{TN}{TN + FP}", "Andel verkligt friska som testet korrekt friskförklarar"),
        ],
        example="Se en fullständig 2×2-tabell, ROC-kurva och cut score-tabell i Decision Support.",
        related_pages=["Decision Support"],
    ),
    LearningModule(
        key="fairness",
        title="7. Fairness",
        subtitle="Fungerar allt lika bra för alla?",
        intro=(
            "Ett test är rättvist om det mäter samma sak, på samma sätt, för olika grupper "
            "(t.ex. kön, ålder, kultur) - annars riskerar testresultat att systematiskt missgynna vissa grupper."
        ),
        concepts=[
            ConceptItem(
                "DIF (Differential Item Functioning)",
                "Undersöker om enskilda items fungerar olika för olika grupper trots att personerna "
                "har samma nivå på det underliggande konstruktet - ett tecken på bias i enskilda items.",
            ),
            ConceptItem(
                "Measurement invariance (mätinvarians)",
                "Testar om hela testets struktur - inte bara enskilda items - är jämförbar mellan grupper.",
            ),
            ConceptItem("Konfigural invarians", "Samma faktorstruktur (vilka items laddar på vilken faktor) i alla grupper."),
            ConceptItem("Metric/svag invarians", "Samma faktorladdningar i alla grupper - nödvändigt för att jämföra samband mellan grupper."),
            ConceptItem("Scalar/stark invarians", "Samma item-intercept i alla grupper - nödvändigt för att jämföra medelvärden mellan grupper."),
            ConceptItem("Strikt invarians", "Samma residualvarianser i alla grupper - den strängaste nivån."),
            ConceptItem("Cohen's d", "Ett standardiserat mått på skillnaden mellan två gruppers medelvärden, uttryckt i standardavvikelser."),
            ConceptItem(
                "Rättviseindex",
                "Ett praktiskt, sammanfattande mått: 1 - |Cohen's d| / 2 (aldrig under 0) - alltså "
                "1.0 vid ingen skillnad, avtagande mot 0 vid en mycket stor skillnad (|d| ≥ 2). Ett "
                "enklare komplement till formell DIF- och invariansanalys, inte en ersättning för den.",
            ),
        ],
        example=(
            "Tumregel för Cohen's d: |d| < 0.2 försumbar, 0.2-0.5 liten, 0.5-0.8 måttlig, ≥ 0.8 stor skillnad."
        ),
        related_pages=["Fairness Explorer"],
    ),
    LearningModule(
        key="formelsamling",
        title="8. Formelsamling",
        subtitle="Snabböversikt över viktiga formler",
        intro="En samlad referens över de formler som används genomgående i appen - bra för repetition inför tentan.",
        concepts=[
            ConceptItem("X", "Observerad poäng"),
            ConceptItem("T", "Sann poäng (klassisk testteori) - notera att detta T skiljer sig från T-poäng nedan"),
            ConceptItem("E", "Mätfel"),
            ConceptItem("M", "Medelvärde"),
            ConceptItem("SD", "Standardavvikelse"),
            ConceptItem("k", "Antal items"),
            ConceptItem("TP / FN / FP / TN", "True/False Positive/Negative i en 2×2-klassificeringstabell"),
        ],
        formulas=[
            FormulaItem("Z-poäng", r"z = \frac{X - M}{SD}", "Standardiserad poäng"),
            FormulaItem("T-poäng", r"T = 50 + 10z", "Medel 50, SD 10"),
            FormulaItem("Z från T", r"z = \frac{T - 50}{10}", ""),
            FormulaItem("Råpoäng från T", r"X = M + z \times SD", ""),
            FormulaItem("SEM", r"SEM = SD \times \sqrt{1 - r_{xx}}", "r_xx = reliabilitet"),
            FormulaItem("95% konfidensintervall", r"KI = X \pm 1{,}96 \times SEM", ""),
            FormulaItem("Sensitivitet", r"\frac{TP}{TP+FN}", ""),
            FormulaItem("Specificitet", r"\frac{TN}{TN+FP}", ""),
            FormulaItem("Reliabilitet (KTT)", r"Rel = \frac{Var(T)}{Var(X)} = 1 - \frac{Var(E)}{Var(X)}", ""),
            FormulaItem("Cronbach's alpha", r"\alpha = \frac{k}{k-1}\left(1 - \frac{\sum \sigma_i^2}{\sigma_x^2}\right)", ""),
            FormulaItem("Test-retest", r"r = r(test_1, test_2)", "Korrelation mellan två mättillfällen"),
            FormulaItem("Youden's J", r"J = Sens + Spec - 1", ""),
        ],
        related_pages=[],
    ),
]


def get_module(key: str) -> LearningModule | None:
    return next((m for m in LEARNING_MODULES if m.key == key), None)


_ALL_CONCEPTS: list[ConceptItem] = [c for m in LEARNING_MODULES for c in m.concepts]


def _clean_term(term: str) -> str:
    """Strips a leading "1. " / "3. " numbering prefix used by the Validitet
    module's concept list, so matching works against the plain term."""
    import re

    return re.sub(r"^\d+\.\s*", "", term.lower())


def get_concept_explanation(term: str) -> str | None:
    """Exact (case-insensitive) lookup by concept term - use this from a
    call site via `learning_key=` when you know precisely which concept a
    tooltip corresponds to. Guaranteed correct, unlike the fuzzy fallback
    below."""
    target = _clean_term(term)
    for concept in _ALL_CONCEPTS:
        if _clean_term(concept.term) == target:
            return concept.explanation
    return None


def find_deeper_explanation(label: str) -> str | None:
    """Best-effort fallback match of a components.concept_tooltip label
    (e.g. "Cronbach's alpha") against concept terms, so Läroläge can show a
    deeper explanation without every call site passing an explicit
    `learning_key`. Concept terms shorter than 3 characters (formula symbols
    like "T", "E", "k") are skipped entirely - matching them by substring
    would trigger on almost any label that happens to contain that letter.
    When multiple terms match, the longest (most specific) one wins."""
    label_lower = label.lower()
    best_explanation: str | None = None
    best_len = 0
    for concept in _ALL_CONCEPTS:
        term_clean = _clean_term(concept.term)
        if len(term_clean) < 3:
            continue
        if (term_clean in label_lower or label_lower in term_clean) and len(term_clean) > best_len:
            best_explanation = concept.explanation
            best_len = len(term_clean)
    return best_explanation
