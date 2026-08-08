"""Structured lesson content for Learning Mode - the exam-prep payload.

Sourced from docs/reference/psykometri_begrepp_concept_map.png ("Psykometri
- hela systemet på en sida"): the six pillars (Verkligheten -> Testutveckling
-> Validitet -> Reliabilitet -> Tolka poäng -> Beslut) plus Fairness, and the
formula quick-reference at the bottom of the concept map.

Each concept explanation ends with three optional standard blocks, used by
the concepts central to the "Sara/Johan/Henrik, 1000 patienter"
presentation story:
- **I storyn:** how this concept shows up in the presentation's narrative.
- **AI-kursen:** Kärna / Försvarbar bonus / Ingen direkt koppling - an
  honest note on whether the Applied ML course rubric actually asks for
  this, for calibrating how much presentation time it deserves.
- **Tenta:** the exact exam question (Standards page + M1 question number)
  this concept answers, sourced from the user's own exam-question
  compilation - a repetition aid for the psychology-programme psychometrics
  exam, not something to show the AI class.

These three blocks are exam-prep scaffolding for the author, not something
meant to survive into the version shown to the class/grader - see
components/kpi_card.py and components/concept_tooltip.py: they only render
inside the "Läroläge - fördjupning" section (Läroläge toggle ON), never in
the always-visible base tooltip.

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
                "faktorer items faktiskt grupperar sig i, utan att man i förväg bestämmer strukturen.\n\n"
                "**I storyn:** Det här är steget Sara tar FÖRST, innan hon rör en enda fråga i den "
                "kortade enkäten - hon måste veta om GAD-7 mäter en sak (ångest) eller flera (t.ex. "
                "oro och fysiska symtom separat) innan hon vågar ta bort något. Factor Explorer kör "
                "EFA, inte CFA.\n\n"
                "**AI-kursen:** Försvarbar bonus - nära släkt med PCA, som nämns explicit i "
                "kursuppgiften som exempel på datareduktion. Båda letar efter underliggande struktur "
                "i data utan en förhandsbestämd modell; Factor Explorers scree plot/egenvärden ÄR i "
                "praktiken samma matematik som PCA använder.\n\n"
                "**Tenta:** M1 F23 (Standards s. 15-16, 26-28, 88) - exakt fråga: \"Beskriv skillnaden "
                "mellan Exploratory Factor Analysis (EFA) och Confirmatory Factor Analysis (CFA) samt "
                "ge exempel på när respektive metod kan användas vid testutveckling eller validering.\"",
            ),
            ConceptItem(
                "CFA (Confirmatory Factor Analysis)",
                "Bekräftande faktoranalys - testar om en i förväg specificerad faktorstruktur "
                "stämmer med observerad data. Används för att bekräfta EFA-resultat på nya data.\n\n"
                "**I storyn:** Factor Explorer kör inte CFA - men modellanpassningsmåtten som redan "
                "visas (RMSEA/CFI/TLI/SRMR) är just de mått en riktig CFA skulle rapportera för att "
                "testa om en i förväg bestämd struktur (\"GAD-7 är en enda faktor\") håller. Skillnaden: "
                "EFA låter strukturen växa fram ur datan, CFA testar om en struktur du redan bestämt "
                "dig för stämmer.\n\n"
                "**AI-kursen:** Ingen direkt koppling som ML-teknik, men samma logik ('testa en "
                "förutbestämd hypotes mot data, snarare än att utforska fritt') återkommer när ni "
                "jämför en enkel baslinjemodell mot en mer komplex modell i Machine Learning-delen.\n\n"
                "**Tenta:** Samma fråga som EFA ovan, M1 F23.",
            ),
            ConceptItem(
                "Parallel analysis",
                "Jämför egenvärden från de riktiga data med egenvärden från slumpmässiga data av "
                "samma storlek. Behåll faktorer vars egenvärde överstiger slumpdatans - ett mer "
                "robust kriterium för antal faktorer än Kaisers regel (egenvärde > 1).\n\n"
                "**I storyn:** Det här är exakt vad KPI:n \"Föreslagna faktorer: 1\" i Factor Explorer "
                "bygger på - bekräftar att Saras enkät är rätt att räkna ihop till en enda totalpoäng.\n\n"
                "**AI-kursen:** Försvarbar bonus - att jämföra en modell mot en slumpmässig baslinje "
                "är samma grundidé som permutationstester i ML, men metoden i sig är specifik för "
                "faktoranalys.\n\n"
                "**Tenta:** Underlag för M1 F23 (EFA-resonemang); nämns som ett mer robust alternativ "
                "till Kaisers regel i föreläsningsmaterialet.",
            ),
            ConceptItem(
                "Scree plot",
                "Visar egenvärdena i fallande ordning som en linjediagram. Antalet faktorer att "
                "behålla är ofta där kurvan planar ut (\"armbågen\"), eller där den korsar en "
                "parallel analysis-linje.\n\n"
                "**I storyn:** Din tydligaste PCA-koppling i hela appen - scree plot bygger på "
                "egenvärden från exakt samma matematik (eigenvalue decomposition) som PCA använder. "
                "Om du inte hinner visa PCA-fliken i Machine Learning, är det här ditt sätt att ändå "
                "visa att du förstår tekniken, fast i psykometrins eget språk.\n\n"
                "**AI-kursen:** Försvarbar bonus, en av de starkare kopplingarna i hela appen.\n\n"
                "**Tenta:** Bakgrund till M1 F23.",
            ),
            ConceptItem(
                "Kommunalitet",
                "Andelen varians i en fråga som förklaras av de extraherade faktorerna tillsammans. "
                "Låg kommunalitet (<0.30) betyder att frågan till stor del mäter något faktorerna "
                "inte fångar upp.\n\n"
                "**AI-kursen:** Försvarbar bonus - motsvarar hur mycket av en variabels varians som "
                "\"förklaras\" av de behållna dimensionerna, samma grundfråga som förklarad varians i PCA.\n\n"
                "**Tenta:** Stöddefinition för M1 F23.",
            ),
            ConceptItem(
                "KMO & Bartlett's test",
                "Två kontroller av om data överhuvudtaget lämpar sig för faktoranalys, innan man "
                "tolkar resultatet. KMO (Kaiser-Meyer-Olkin) mäter sampling adequacy (≥0.80 bra, "
                "≥0.60 acceptabelt); Bartlett's test kontrollerar att frågorna korrelerar tillräckligt "
                "med varandra (signifikant, p < .05, är det önskade utfallet).\n\n"
                "**I storyn:** Sara kollar de här INNAN hon tolkar faktorlösningen i steg 1 - en "
                "förutsättningskontroll, inte en slutsats i sig.\n\n"
                "**AI-kursen:** Försvarbar bonus - samma princip som att kolla datakvalitet innan man "
                "tränar en ML-modell.\n\n"
                "**Tenta:** Stöddefinition för M1 F23, nämns i föreläsningsmaterialet om faktoranalys.",
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
                "av item-innehållet mot konstruktets definition.\n\n"
                "**I storyn:** Tänk simkunnighetsprov - det borde innehålla simning, inte en skriftlig "
                "frågesport om simning. När Sara tar bort fråga 4 och 7 rubbar hon (lite) innehållet; "
                "om hon någon gång lägger till en egen fråga rubbar hon det desto mer.\n\n"
                "**AI-kursen:** Nästan ingen koppling - ren psykometriteori, ingen kvantitativ ML-vinkel.\n\n"
                "**Tenta:** M1 F22 (Standards s. 14).",
            ),
            ConceptItem(
                "2. Responsprocesser (Response Processes)",
                "Förstår och tolkar testpersoner frågorna på det sätt som avsetts? Undersöks med "
                "t.ex. kognitiva intervjuer eller tänka-högt-studier.\n\n"
                "**I storyn:** Tänker en 70-åring och en 20-åring på samma sätt när de läser \"blir "
                "lätt irriterad eller lättretlig\"? Det här är evidenskällan appen ärligt lämnar som "
                "\"dokumentation väntar\" i Validity Dashboard - inte en lucka du missat, utan rätt svar.\n\n"
                "**AI-kursen:** Ingen koppling.\n\n"
                "**Tenta:** M1 F22 (Standards s. 15).",
            ),
            ConceptItem(
                "3. Intern struktur (Internal Structure)",
                "Har items den struktur som teorin förutsäger? Undersöks med faktoranalys (EFA/CFA) "
                "och reliabilitetsanalys.\n\n"
                "**I storyn:** Det här är precis vad steg 1-2 (Factor Explorer + Reliability Explorer) "
                "levererar - den enda av de fem evidenskällorna Sara redan fullt ut har täckt innan hon "
                "ens når Validity Dashboard.\n\n"
                "**AI-kursen:** Försvarbar bonus, via Factor/Reliability Explorer.\n\n"
                "**Tenta:** M1 F22/F23 (Standards s. 15-16).",
            ),
            ConceptItem(
                "4. Relation till andra variabler (Relations to Other Variables)",
                "Uppför sig testet som förväntat i relation till andra mått? Inkluderar samtidig "
                "(konkurrent) validitet, prediktiv validitet (förutsäger framtida utfall), "
                "konvergent validitet (korrelerar med liknande mått) och diskriminant validitet "
                "(låg korrelation med orelaterade konstrukt).\n\n"
                "**I storyn:** Det här ÄR i praktiken vad Decision Support och Machine Learning mäter "
                "i akt 2 - hur väl testpoängen (eller modellens sannolikhet) stämmer med den bekräftade "
                "diagnosen vid referensintervjun. Sensitivitet/specificitet/AUC är kriterievaliditet i "
                "siffror, du bara inte kallat det så förut.\n\n"
                "**AI-kursen:** Kärna-adjacent - detta är samma fråga Decision Support/Machine Learning "
                "redan svarar på, bara med psykometrins ordval.\n\n"
                "**Tenta:** M1 omtenta F22 (Standards s. 16-18, 218-219) - definiera kriterievaliditet, "
                "samtidig/concurrent, prediktiv och diskriminant validitet.",
            ),
            ConceptItem(
                "5. Konsekvenser (Consequences)",
                "Ger testanvändningen positiva konsekvenser och minimerar negativa - på individnivå "
                "(nytta/skada), gruppnivå (rättvisa) och samhällsnivå (resursfördelning, beslutskvalitet).\n\n"
                "**I storyn:** Det här är hela akt 2 i ett ord - vad HÄNDER när Sara faktiskt använder "
                "poängen för att prioritera vem av 1000 som kallas in, och gör det rättvist (Fairness "
                "Explorer, steg 7)? Validity Dashboard flaggar det här som en öppen fråga i akt 1; akt "
                "2 svarar på den.\n\n"
                "**AI-kursen:** Försvarbar bonus - motsvarar \"vad kostar en felaktig prediktion\", "
                "kärnan i hur man läser en confusion matrix i praktiken.\n\n"
                "**Tenta:** M1 F22 (Standards s. 19-21).",
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
                "poängen kan aldrig observeras direkt, bara skattas.\n\n"
                "**I storyn:** Grundvalen för allt Reliability Explorer och Measurement Error visar - "
                "nämns sällan explicit i appens UI, men det är teorin bakom alpha och SEM.\n\n"
                "**AI-kursen:** Ingen koppling.\n\n"
                "**Tenta:** M1 F18 (Standards s. 34-35, 37-39, 45, 218, 224) - exakt fråga: \"Förklara "
                "de två grundantagandena... och innebörden av de efterföljande implikationerna, det "
                "vill säga ekvationerna 3-6. En mening per ekvation.\" Facit: X = T + E, mätfel har "
                "medelvärde 0 och är slumpmässiga/okorrelerade, Var(X) = Var(T) + Var(E).",
            ),
            ConceptItem(
                "Intern konsistens (Cronbach's alpha)",
                "Mäter hur väl items i en delskala samvarierar - hög alpha betyder att items mäter "
                "samma underliggande konstrukt.\n\n"
                "**I storyn:** Steg 2 i akt 1 - alpha 0.88 för alla sju GAD-7-frågor, oförändrat efter "
                "att fråga 4 och 7 tagits bort i Test Builder.\n\n"
                "**AI-kursen:** Försvarbar bonus - kan ramas in som \"granska feature-kvalitet innan "
                "modellering\".\n\n"
                "**Tenta:** M1 F19/F20 (Standards s. 35-37, 44-45, 220, 223-224) - frågar specifikt "
                "efter tre saker: Data = alla items, ETT tillfälle. Beräkning: α = k/(k-1)·(1-"
                "Σitemvarianser/testvarians). Antaganden: samma latenta konstrukt, unidimensionalitet, "
                "positiv samvariation mellan items.",
            ),
            ConceptItem(
                "Item-total korrelation",
                "Korrelationen mellan en enskild frågas poäng och summan av alla övriga frågor i "
                "skalan. Låga värden (<0.30) betyder att frågan mäter något annat än resten av "
                "skalan - en av de vanligaste anledningarna att ta bort eller revidera en fråga.\n\n"
                "**I storyn:** Grunden för Test Builders \"Vilka frågor är säkrast att ta bort?\" - "
                "fråga 4 (0.57) och fråga 7 (0.61) är de svagaste av GAD-7:s sju frågor.\n\n"
                "**AI-kursen:** Försvarbar bonus - motsvarar att granska feature-kvalitet/redundans "
                "innan man tränar en modell.\n\n"
                "**Tenta:** Stöd för M1 F19-F20 (samma resonemang som alpha, på itemnivå).",
            ),
            ConceptItem(
                "McDonald's omega",
                "Ett reliabilitetsmått baserat på en faktormodell, ofta mer robust än alpha "
                "eftersom det inte antar att alla items väger lika mycket.\n\n"
                "**I storyn:** Bra att nämna kort om läraren frågar \"varför inte bara alpha\" - visar "
                "att du känner till begränsningen utan att behöva bygga in det i huvudberättelsen.\n\n"
                "**AI-kursen:** Ingen koppling.\n\n"
                "**Tenta:** Nämns inte som egen fråga i din sammanställning, men hör till samma "
                "frågekomplex som alpha (M1 F19-F20) om det dyker upp som följdfråga.",
            ),
            ConceptItem(
                "Test-retest-reliabilitet",
                "Korrelationen mellan poäng vid två tillfällen - mäter stabilitet över tid.\n\n"
                "**I storyn:** Reliability Explorers Tidsstabilitet-flik finns redan byggd och kostar "
                "dig noll extra utveckling att visa - en helt separat tentafråga från alpha.\n\n"
                "**AI-kursen:** Ingen koppling.\n\n"
                "**Tenta:** M1 omtenta F19 (Standards s. 36-38, 40, 223-224) - egen delfråga, skild "
                "från alpha-frågan: data = samma personer, samma test, två tillfällen; beräkning = "
                "korrelationen mellan tillfälle 1 och 2; antaganden = konstruktet är stabilt över tid, "
                "inga stora övnings-/minneseffekter, mätfelen är slumpmässiga.",
            ),
            ConceptItem("Split-half-reliabilitet", "Testet delas i två halvor (t.ex. udda/jämna items) och halvornas poäng korreleras."),
            ConceptItem(
                "SEM (Standard Error of Measurement)",
                "Anger hur mycket den observerade poängen förväntas variera på grund av mätfel. "
                "Ju högre reliabilitet, desto lägre SEM.\n\n"
                "**AI-kursen:** Ingen koppling.\n\n"
                "**Tenta:** M1 F17 (Standards s. 39, 45, 223 + 95/102) - exakt fråga bygger vidare på "
                "en redan beräknad T-poäng: \"...T-poäng 63 och SEM för detta estimat är 3,65. Ange "
                "nedre gränsen för ett 95% konfidensintervall.\"",
            ),
            ConceptItem(
                "Konfidensintervall (KI) för en poäng",
                "Intervallet där personens sanna poäng troligen ligger, baserat på SEM.\n\n"
                "**I storyn:** Visas direkt i Norm Explorer för Johan/Henrik - \"95% KI (mätfel)\".\n\n"
                "**AI-kursen:** Ingen koppling.\n\n"
                "**Tenta:** Samma fråga som SEM, M1 F17. Formel: KI = X ± 1,96 × SEM.",
            ),
            ConceptItem(
                "Reliable Change Index (RCI)",
                "RCI = (poäng vid tid 2 − poäng vid tid 1) / SE_diff, där SE_diff = SEM × √2. |RCI| ≥ "
                "1.96 tolkas som en statistiskt pålitlig förändring (Jacobson & Truax, 1991) - inte "
                "bara mätbrus. Används för att avgöra om en förändring hos en klient över tid är "
                "verklig eller inom det förväntade mätfelet.\n\n"
                "**I storyn:** Relevant om Johan eller Henrik gör om testet vid ett återbesök - har "
                "de faktiskt förbättrats, eller ligger skillnaden inom mätfelet?\n\n"
                "**AI-kursen:** Ingen koppling.\n\n"
                "**Tenta:** Inte en egen fråga i din sammanställning, men bygger direkt på SEM (M1 F17) "
                "- bra följdfråga att kunna svara på.",
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
            ConceptItem(
                "Z-poäng",
                "Hur många standardavvikelser en persons poäng ligger från normgruppens medelvärde.\n\n"
                "**I storyn:** Räknas fram bakom kulisserna varje gång Norm Explorer visar Johans eller "
                "Henriks resultat mot normgruppen.\n\n"
                "**AI-kursen:** Försvarbar bonus - z-scoring är exakt vad StandardScaler gör som "
                "preprocessing-steg innan man tränar en ML-modell.\n\n"
                "**Tenta:** M1 F1-F10 - den absolut största frågebanken i hela tentan, alla bygger på "
                "z = (X-M)/SD och T = 50+10z.",
            ),
            ConceptItem(
                "T-poäng",
                "En standardiserad skala med medelvärde 50 och SD 10 - undviker negativa tal och decimaler.\n\n"
                "**I storyn:** Norm Explorer visar Johans/Henriks T-poäng direkt - säg formeln högt "
                "(\"T = 50 + 10 gånger z\") när du visar den, det är gratis repetition.\n\n"
                "**AI-kursen:** Försvarbar bonus, se Z-poäng.\n\n"
                "**Tenta:** M1 F1-F10 (Standards s. 95, 97, 102, 104) - t.ex. \"Om ett T-poäng på ett "
                "prestationstest bestäms till 30, hur stor andel presterade bättre i normeringsgruppen?\"",
            ),
            ConceptItem(
                "Percentil",
                "Andelen (%) i normgruppen som presterar lägre än personen.\n\n"
                "**I storyn:** \"Resultatet ligger högre än 93% av personerna i normgruppen\" - direkt "
                "citat från Norm Explorer.\n\n"
                "**AI-kursen:** Ingen direkt koppling.\n\n"
                "**Tenta:** M1 F1-F10, samma frågebank som T-poäng.",
            ),
            ConceptItem("Stanine", "En niogradig skala (1-9) med medel 5 och SD 2, ofta använd för enkel kommunikation."),
            ConceptItem(
                "Normgrupp (referensgrupp)",
                "Den grupp vars resultat används som jämförelsegrund - normernas kvalitet avgör "
                "tolkningens giltighet.\n\n"
                "**I storyn:** Norm Explorer är tydlig med att detta är en sample-baserad, inte en "
                "officiell, norm - en viktig ärlighet att peka ut för klassen, och en poäng du kan "
                "koppla till M1 F22-F23:s resonemang om urvalsstorlek/restricted range.\n\n"
                "**AI-kursen:** Ingen direkt koppling.\n\n"
                "**Tenta:** Bakgrund till M1 F1-F10 (normtabeller/normreferens).",
            ),
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
            ConceptItem(
                "Cut score",
                "Gränsen mellan ett positivt och negativt testresultat. Valet påverkar avvägningen "
                "mellan sensitivitet och specificitet.\n\n"
                "**I storyn:** Steg 5 i akt 2 - Decision Support föreslår tröskel 10 för GAD-7 (Youden's "
                "J maximerad), innan Sara inser att en fast gräns inte räcker när hon bara har 150 "
                "platser för 1000 svar och behöver rangordna istället.\n\n"
                "**AI-kursen:** Kärna - \"Confusion Matrix, AUC-ROC, precision-recall\" nämns ordagrant "
                "i kursuppgiften.\n\n"
                "**Tenta:** M1 F11-F14 (Standards s. 100-101, 107) - HADS-exemplet med cut-off 11 "
                "poäng är nästan identiskt med hur Decision Support fungerar.",
            ),
            ConceptItem(
                "Konfusionsmatris",
                "En 2×2-tabell över hur testets prediktioner (positiv/negativ) stämmer mot det "
                "faktiska utfallet: sant positiva (TP), sant negativa (TN), falskt positiva (FP) "
                "och falskt negativa (FN). Grunden för sensitivitet, specificitet, PPV och NPV.\n\n"
                "**I storyn:** Visas rakt av i både Decision Support och Machine Learning - samma "
                "tabellform som i tentans HADS-fråga (FN=37, TP=123, TN=612, FP=41).\n\n"
                "**AI-kursen:** Kärna, ordagrant efterfrågad.\n\n"
                "**Tenta:** M1 F11-F14 - räkneexemplen ger dig FN/TP/TN/FP direkt och ber dig räkna "
                "ut sensitivitet/specificitet, exakt vad konfusionsmatrisen visar grafiskt.",
            ),
            ConceptItem(
                "Sensitivitet (TPR)",
                "Sannolikheten att testet korrekt identifierar de som verkligen har tillståndet.\n\n"
                "**I storyn:** \"Vid tröskel 10 fångar testet 68% av de som verkligen har hög ångest\" - "
                "direkt från Decision Support, och den siffra kapacitetskurvan i steg 6 försöker höja.\n\n"
                "**AI-kursen:** Kärna, ordagrant efterfrågad (recall i scikit-learns terminologi).\n\n"
                "**Tenta:** M1 F11/F13 - exakt formel Sens = TP/(TP+FN), samma räkneuppgift oavsett "
                "om instrumentet heter GAD-7 eller HADS.",
            ),
            ConceptItem(
                "Specificitet (TNR)",
                "Sannolikheten att testet korrekt identifierar de som inte har tillståndet.\n\n"
                "**I storyn:** \"76% av de utan hög ångest identifieras korrekt som negativa.\"\n\n"
                "**AI-kursen:** Kärna, ordagrant efterfrågad.\n\n"
                "**Tenta:** M1 F12/F14 - exakt formel Spec = TN/(TN+FP).",
            ),
            ConceptItem("False Negative (FN)", "Testet visar negativt trots att personen har tillståndet - risk: utebliven hjälp."),
            ConceptItem("False Positive (FP)", "Testet visar positivt trots att personen inte har tillståndet - risk: onödig oro eller åtgärd."),
            ConceptItem(
                "PPV / NPV",
                "Positivt/negativt prediktivt värde - sannolikheten att ett positivt (PPV) eller "
                "negativt (NPV) testresultat är korrekt.\n\n"
                "**I storyn:** Till skillnad från sensitivitet/specificitet beror PPV/NPV på hur "
                "vanlig ångest är i just Saras 1000 patienter (prevalens) - samma test kan alltså ge "
                "olika PPV i en högriskgrupp och en bred befolkningsscreening.\n\n"
                "**AI-kursen:** Kärna - motsvarar precision i scikit-learns terminologi (precision-recall "
                "nämns ordagrant i kursuppgiften).\n\n"
                "**Tenta:** Inte en egen räkneuppgift i din sammanställning, men samma 2×2-tabell som "
                "M1 F11-F14 - bra att kunna definiera om det dyker upp som följdfråga.",
            ),
            ConceptItem(
                "ROC-kurva & AUC",
                "Visar avvägningen mellan sensitivitet och (1-specificitet) vid alla möjliga trösklar; "
                "AUC sammanfattar testets samlade förmåga att särskilja grupper.\n\n"
                "**I storyn:** AUC 0,80 för den enkla totalpoängen (steg 5) mot AUC 0,81 för den bästa "
                "modellen (steg 6) - en liten men konsekvent förbättring.\n\n"
                "**AI-kursen:** Kärna, ordagrant efterfrågad.\n\n"
                "**Tenta:** Inte en direkt räkneuppgift, men samma tabell/resonemang som M1 F11-F14 "
                "bygger på över alla trösklar samtidigt.",
            ),
            ConceptItem(
                "Youden's J",
                "J = Sensitivitet + Specificitet - 1. Tröskeln som maximerar J ger bästa balans mellan "
                "de två.\n\n"
                "**I storyn:** Det här är hur appen räknar fram \"Rekommenderad tröskel: 10\" automatiskt.\n\n"
                "**AI-kursen:** Försvarbar bonus - ett konkret, förklarbart sätt att välja "
                "beslutsgräns, i linje med kursens fokus på tolkningsbara resultat.\n\n"
                "**Tenta:** Inte namngiven i din sammanställning, men själva avvägningen (\"en högre "
                "tröskel fångar färre falska positiva men missar fler sanna fall\") är precis vad M1 "
                "F11-F14 vill att du resonerar kring.",
            ),
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
                "har samma nivå på det underliggande konstruktet - ett tecken på bias i enskilda items.\n\n"
                "**I storyn:** Nämns explicit i Fairness Explorers ärliga platshållar-flik - fullt "
                "implementerad DIF-analys kräver t.ex. ordinal logistisk regression per fråga, vilket "
                "appen inte gör.\n\n"
                "**AI-kursen:** Ingen direkt koppling.\n\n"
                "**Tenta:** Bakgrund till M1 omtenta F24 (Standards s. 54-58, 63-70).",
            ),
            ConceptItem(
                "Measurement invariance (mätinvarians)",
                "Testar om hela testets struktur - inte bara enskilda items - är jämförbar mellan grupper.\n\n"
                "**I storyn:** KRITISK DISTINKTION för presentationen: Fairness Explorers Cohen's "
                "d-jämförelse är en förenklad, beskrivande gruppskillnad - INTE samma sak som formell "
                "mätinvarians. Säg det rakt ut på scen, det visar att du förstår gränsen för vad "
                "appen faktiskt gör.\n\n"
                "**AI-kursen:** Försvarbar bonus, starkare om Fairness Explorer körs på modellens "
                "beslut (steg 6-utökningen) - då blir det en riktig algoritmisk bias-granskning.\n\n"
                "**Tenta:** M1 omtenta F24, exakt fråga: \"A. Vilken frågeställning vill man besvara "
                "när man genomför en mätinvariansanalys? B. Vilken statistisk metod används typiskt "
                "för att testa mätinvarians? C. Resonera kring vad resultaten i figuren visar.\" Facit: "
                "A. Om testet mäter samma latenta konstrukt jämförbart i olika grupper. B. Multigrupps-"
                "konfirmatorisk faktoranalys (multigroup CFA).",
            ),
            ConceptItem(
                "Konfigural invarians",
                "Samma faktorstruktur (vilka items laddar på vilken faktor) i alla grupper.\n\n"
                "**Tenta:** Del av facit till M1 omtenta F24 (frågeled C) - den svagaste av de fyra "
                "invariansnivåerna.",
            ),
            ConceptItem(
                "Metric/svag invarians",
                "Samma faktorladdningar i alla grupper - nödvändigt för att jämföra samband mellan grupper.\n\n"
                "**Tenta:** Del av facit till M1 omtenta F24, nivå 2 av 4.",
            ),
            ConceptItem(
                "Scalar/stark invarians",
                "Samma item-intercept i alla grupper - nödvändigt för att jämföra medelvärden mellan grupper.\n\n"
                "**Tenta:** Del av facit till M1 omtenta F24 - oftast den nivå en tentafigur visar "
                "(faktorladdningar och intercept lika, men residualer skiljer sig).",
            ),
            ConceptItem(
                "Strikt invarians",
                "Samma residualvarianser i alla grupper - den strängaste nivån.\n\n"
                "**Tenta:** Del av facit till M1 omtenta F24, den starkaste av de fyra nivåerna.",
            ),
            ConceptItem(
                "Cohen's d",
                "Ett standardiserat mått på skillnaden mellan två gruppers medelvärden, uttryckt i standardavvikelser.\n\n"
                "**I storyn:** d = 0,28 mellan Kön: Kvinna vs Annat i steg 7 - en måttlig men inte "
                "alarmerande skillnad, vilket appen själv flaggar.\n\n"
                "**AI-kursen:** Försvarbar bonus, starkare om kopplad till modellens beslut istället "
                "för bara råpoängen (se steg 6-utökningen).\n\n"
                "**Tenta:** Tumregel att kunna utantill: |d| < 0.2 försumbar, 0.2-0.5 liten, 0.5-0.8 "
                "måttlig, ≥ 0.8 stor skillnad. Relevant bakgrund till M1 omtenta F24.",
            ),
            ConceptItem(
                "Rättviseindex",
                "Ett praktiskt, sammanfattande mått: 1 - |Cohen's d| / 2 (aldrig under 0) - alltså "
                "1.0 vid ingen skillnad, avtagande mot 0 vid en mycket stor skillnad (|d| ≥ 2). Ett "
                "enklare komplement till formell DIF- och invariansanalys, inte en ersättning för den.\n\n"
                "**I storyn:** 0,95 i genomsnitt över alla gruppjämförelser i steg 7 - appens eget, "
                "förenklade sammanfattningsmått, inte ett etablerat begrepp från litteraturen.\n\n"
                "**AI-kursen:** Försvarbar bonus.\n\n"
                "**Tenta:** Inte ett Standards-begrepp - nämn det som \"appens egna, förenklade mått\", "
                "inte som facit-terminologi om det dyker upp på tentan.",
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
