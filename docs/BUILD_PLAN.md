# Build Plan — Psychometric Visual Explorer

Work through these sprints in order. After each sprint: run the app, fix
anything broken, run `pytest`, commit, merge to `main`, push (see CLAUDE.md
git workflow). Report a short summary before moving to the next sprint so
progress is checkpointed — don't silently blast through all ten in one shot
without ever running the app.

Reference screenshots live in `docs/mockups/`, named `<test>_<module-order>_
<module-name>.png` (e.g. `gad7_05_reliability_explorer.png`,
`phq9_05_reliability_explorer.png`, `bigfive_05_reliability_explorer.png`).
GAD-7 has the most complete set (all 17 modules incl. Fairness Explorer,
Learning Mode, Export, Settings); PHQ-9 and Big Five only have a subset —
use those to confirm the same page works with a different item/subscale
count, and fall back to the GAD-7 version of a page for any module PHQ-9/Big
Five don't have a screenshot for. Use the GAD-7 ones as the primary layout
density target (see CLAUDE.md design system section). The original assignment
brief and SRS are in `docs/reference/` and the exam concept map is
`docs/reference/psykometri_begrepp_concept_map.png` — use the latter as the
content outline for Learning Mode.

## Sprint 0 — Cleanup & shell

- Consolidate the folder structure to match CLAUDE.md (delete the now-unused
  `config/, ml/, models/, psychometrics/, services/, statistics/,
  visualization/` folders and their placeholder `__init__.py` files).
- Set up `app.py` with Streamlit multi-page navigation, dark navy sidebar,
  PVE logo/wordmark, and the Home page (open dataset / example data / learning
  mode / documentation cards, per the Home mockup).
- Update `requirements.txt`: add `xgboost`, `shap`, `umap-learn`, `tensorflow`
  (or `torch`, your call — pick one and be consistent), `reportlab`,
  `XlsxWriter`, `joblib`, `pytest`. Install into the existing `.venv`.
- Empty `pages/` stubs for all 15 pages so navigation works end to end (each
  just renders a "coming soon" placeholder for now).
- **Done when**: `streamlit run app.py` launches, sidebar nav works, no errors.

## Sprint 1 — Data model, plugin engine, sample data, Import Wizard

- `core/data_model.py`: `Dataset`, `Questionnaire`, `Statistics` objects per
  the fields implied by the mockups (N, item count, subscales, missing %,
  Likert range, reverse-scored items, etc.).
- `core/plugin_engine.py`: loads/validates JSON plugins from `plugins/`,
  matches an uploaded dataset's columns against installed plugins, falls back
  to a manual-mapping flow if nothing matches. Never crash on a bad plugin —
  skip it and continue.
- `plugins/gad7.json`, `plugins/phq9.json`, `plugins/ipip_bigfive.json` —
  use `plugins/gad7.example.json` (provided) as the schema template; write
  the other two the same way using the item text visible in the mockups.
- `data/generate_sample_data.py` + committed CSVs for all three tests (see
  CLAUDE.md "Sample data").
- `core/import_engine.py` + `pages/1_Import_Wizard.py`: 5-step wizard (upload
  → identify columns → review → map items/scales → finish), auto-detects a
  known plugin or hands off to manual mapping.
- **Done when**: uploading any of the 3 sample CSVs (or picking "Exempeldata")
  correctly identifies the test and produces a `Dataset` + `Questionnaire`.

## Sprint 2 — Dataset Overview + Psychometric QC

- `pages/2_Dataset_Overview.py`: N, item count, response distribution,
  missing %, demographics, descriptive stats, reliability snapshot.
- `pages/3_Psychometric_QC.py`: missing data, floor/ceiling effects, low
  variance items, duplicates, outliers, suspicious response patterns
  (straight-lining), overall quality score + recommendations.
- Both driven entirely by `core/stats_engine.py` functions, generic across
  all three tests.
- **Done when**: both pages render correctly for GAD-7, PHQ-9, and Big Five
  sample data with no test-specific code in the page files.

## Sprint 3 — Test Profile Explorer + Reliability Explorer

- Item overview, subscale structure, response distributions, item
  correlation heatmap.
- Cronbach's alpha, McDonald's omega, item-total correlations, "alpha if item
  deleted", split-half, test-retest (simulate a second timepoint for the
  demo data), reliability by subscale.
- **Done when**: matches the mockup layout and numbers are internally
  consistent (e.g. removing a low-correlation item should move alpha the
  direction the table claims).

## Sprint 4 — Factor Explorer + Validity Dashboard

- EFA via `factor_analyzer`: scree plot + parallel analysis, factor loadings,
  communalities, model fit (RMSEA/CFI/TLI/SRMR), factor correlations for
  multi-factor tests (Big Five).
- Validity Dashboard: five evidence sources (see CLAUDE.md), convergent/
  discriminant scatter, criterion correlations. Simulate plausible external
  criterion variables for the demo data (e.g. a correlated BDI-II proxy) —
  document clearly in code comments that these are simulated for
  demonstration.
- **Done when**: single-factor GAD-7/PHQ-9 and five-factor Big Five both
  render correctly through the same code path.

## Sprint 5 — Norm Explorer + Measurement Error

- Raw → z → T-score → percentile → stanine conversion table, normal curve
  with the person's score marked, per-dimension norms for Big Five.
- SEM, 95% CI per person, reliable change index, precision summary.
- **Done when**: math is internally consistent (T = 50 + 10z, SEM = SD√(1−α),
  CI = score ± 1.96×SEM).

## Sprint 6 — Decision Support + Fairness Explorer

- Decision Support: ROC curve, AUC, confusion matrix at a selectable
  threshold, sensitivity/specificity/PPV/NPV table across multiple cut
  scores, Youden's J recommendation.
- Fairness Explorer: group comparisons (age/gender/education/simulated
  ethnicity groups) with Cohen's d, a fairness index per group, plain-
  language interpretation, DIF/measurement-invariance explanation even if the
  invariance testing itself is a stretch goal.
- **Done when**: both pages work off simulated demographic + outcome columns
  in the sample data.

## Sprint 7 — Machine Learning Explorer (graded module — see CLAUDE.md)

- Implement the full pipeline described in CLAUDE.md's ML section: data prep,
  unsupervised (clustering/PCA), supervised with a non-linear model, a deep
  learning option, cross-validation, full evaluation suite, feature
  importance/SHAP.
- UI: model comparison, ROC curve, feature importance chart, a "quick
  prediction" form for a single respondent (per the PHQ-9 mockup).
- **Done when**: you could screenshot this page's code and metrics directly
  into the AI course submission and it would satisfy every bullet in "Vad ni
  skall göra?" from the assignment PDF.

## Sprint 8 — Test Builder + Learning Mode

- Test Builder: view/edit items, scales, response options, cut-offs for the
  active test; "duplicate test" / "create new from scratch" flow that
  produces a new plugin JSON.
- Learning Mode: module list mirroring the app's own pages, progress
  tracking (in-memory/session state is fine, no auth needed), short
  explanations sourced from the psychometrics concept map + the psychometri
  begrepp list — this is the exam-prep payload, make it good.
- **Done when**: a user could learn every concept in the psykometri_begrepp
  list without leaving the app.

## Sprint 9 — Export + polish + README + tests

- `core/export_engine.py`: PDF report (reportlab), PNG figures, CSV/anonymized
  data export, ZIP bundling — one shared engine used by every page's export
  button, not per-page export code.
- Fill in `README.md` per SRS section 28 (vision, features, tech stack,
  install steps, folder structure, screenshots).
- `tests/`: at least one test per `core/` engine (reliability calculation,
  plugin loading/validation, import column detection, ML pipeline runs
  end-to-end on sample data without error).
- Final pass: every page has a working "Nästa"/"Tillbaka" flow matching the
  workflow diagram, no dead links, no leftover placeholder text.
- **Done when**: `pytest` passes, `streamlit run app.py` has zero console
  errors clicking through all 15 pages for all 3 sample datasets, README is
  something you'd be comfortable linking in a job application.

## After Sprint 9

Stop and wait for direction — don't invent Sprint 10 on your own. Good
next-step candidates to *suggest* (not build unprompted): PANAS/PSS/SSP
plugins, CFA in addition to EFA, real DIF/measurement-invariance testing,
Streamlit Cloud deployment.
