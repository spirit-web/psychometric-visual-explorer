# Psychometric Visual Explorer (PVE) — Project Memory

This file is read automatically by Claude Code at the start of every session in
this repo. It is the single source of truth for architecture, standards, and
workflow. Read it fully before doing any work. `docs/BUILD_PLAN.md` has the
sprint-by-sprint task list — work through it in order unless told otherwise.

## What this project is

A Streamlit application that helps psychologists/researchers/students import a
psychological questionnaire dataset (GAD-7, PHQ-9, IPIP Big Five to start,
generic tests later) and walks them through the full psychometric workflow:
import → QC → describe the test → reliability → factor structure → validity →
norms → measurement error → decision support → fairness → machine learning →
build your own test → learn the theory → export.

It serves four purposes at once — keep all four in mind for every module:
1. A real submission for an Applied AI course (full ML pipeline: data prep,
   EDA, unsupervised + supervised + a deep learning option, proper evaluation
   — see the ML Explorer requirements below, this is graded).
2. A study aid for a psychometrics/differential-psychology exam (every
   statistical concept should be explained in plain language somewhere in the
   UI, not just computed).
3. A GitHub portfolio piece aimed at research-assistant type roles.
4. A genuinely useful tool for someone analyzing a psych test.

It is **not** a replacement for SPSS/R/jamovi — it's a visual, pedagogical
layer on top of standard psychometrics.

## Non-negotiable architecture rules

- **UI pages never calculate statistics.** They call functions in `core/` and
  render the returned objects. If you catch yourself writing `np.mean(...)`
  inside a file under `pages/`, stop and move it to `core/`.
- **Engines never render UI** and never import `streamlit`.
- **One page per module, not one page per test.** `pages/6_Factor_Explorer.py`
  must work for GAD-7, PHQ-9, Big Five, and any future test by reading the
  active `Questionnaire` object. Do not create `factor_explorer_gad7.py`,
  `factor_explorer_phq9.py`, etc. The three sets of mockup screenshots you'll
  see in `docs/mockups/` are the *same page* rendering three different
  datasets — treat them as one implementation target, not three.
- **Plugins contain no executable code.** A plugin is a JSON file describing a
  questionnaire (items, subscales, Likert range, reverse-scored items, scoring
  rule, cut-offs, norm reference). All logic that *uses* a plugin lives in
  `core/plugin_engine.py` and `core/stats_engine.py`. See
  `plugins/gad7.example.json` for the shape to follow.
- **Communication between engines happens via pandas DataFrames and small
  dataclasses/pydantic models**, not shared globals.

## Folder structure (target — reconcile the existing empty folders to this)

The repo already has: `assets, components, config, core, docs, ml, models,
pages, psychometrics, services, statistics, tests, utils, visualization` plus
`app.py, requirements.txt, README.md, .gitignore, pyproject.toml` and a
working `.venv` with streamlit/pandas/numpy/scipy/plotly/matplotlib/
scikit-learn/pingouin/factor_analyzer/openpyxl/pydantic/loguru/Pillow already
installed. **Reuse the venv — do not recreate it.**

Consolidate to this structure in Sprint 0 (delete `config/, ml/, models/,
psychometrics/, services/, statistics/, visualization/` — their contents fold
into `core/`):

```
psychometric-visual-explorer/
├── app.py                     # Streamlit entry point, page nav, PVE branding
├── CLAUDE.md
├── requirements.txt
├── README.md
├── .gitignore
├── pyproject.toml
├── docs/
│   ├── BUILD_PLAN.md
│   └── mockups/                # reference screenshots (see below)
├── data/
│   └── sample_datasets/        # generated synthetic CSVs, committed
├── plugins/                    # JSON test definitions, no code
│   ├── gad7.json
│   ├── phq9.json
│   └── ipip_bigfive.json
├── core/                       # pure logic, no streamlit imports, unit-testable
│   ├── data_model.py            # Dataset, Questionnaire, Statistics objects
│   ├── import_engine.py
│   ├── plugin_engine.py
│   ├── stats_engine.py          # reliability, factor analysis, validity, norms, SEM, decision support, fairness
│   ├── ml_engine.py
│   ├── viz_engine.py            # returns plotly figures, does not calculate
│   └── export_engine.py
├── pages/
│   ├── 1_Import_Wizard.py
│   ├── 2_Dataset_Overview.py
│   ├── 3_Psychometric_QC.py
│   ├── 4_Test_Profile_Explorer.py
│   ├── 5_Reliability_Explorer.py
│   ├── 6_Factor_Explorer.py
│   ├── 7_Validity_Dashboard.py
│   ├── 8_Norm_Explorer.py
│   ├── 9_Measurement_Error.py
│   ├── 10_Decision_Support.py
│   ├── 11_Fairness_Explorer.py
│   ├── 12_Machine_Learning.py
│   ├── 13_Test_Builder.py
│   ├── 14_Learning_Mode.py
│   └── 15_Export.py
├── components/                  # shared widgets: kpi_card, concept_tooltip, sidebar
├── utils/                       # config, i18n strings, small helpers
└── tests/                       # pytest, one file per core/ engine
```

## Design system baseline

Match the **GAD-7 mockups' information density** as the target for every page
(dark navy sidebar with the PVE brain-network logo, light content area, white
KPI cards with soft shadow in a 4-column row at the top, tabbed sub-sections,
a plotly chart + a table/detail card side by side, a blue "Nästa →" button
bottom-right, "Tillbaka" bottom-left). The Depression and Personality mockups
are visually busier — don't copy their density, just confirm the same
components work with more items/subscales (9 PHQ-9 items, 5 Big Five
dimensions x 10 items). Status colors: green = good, orange = warning/review,
red = problem. All **UI text is in Swedish** (labels, buttons, explanations);
code, comments, and docstrings are in English.

## The ML Explorer must satisfy the course rubric, not just look nice

This module is graded against a specific assignment. It must include, all
running against the same dataset:
- Data prep (handle missing values, encode categoricals, build a clean
  model-ready dataframe) — reuse output from `import_engine`/`stats_engine`.
- EDA/statistics already covered by other pages — link to them, don't repeat.
- **Unsupervised learning**: clustering (e.g. KMeans) and/or PCA to discover
  structure or generate a target where none exists.
- **Supervised learning with at least one non-linear model** (e.g. Random
  Forest or XGBoost) — plain linear/multiple linear regression alone does
  **not** satisfy the assignment.
- **A deep learning option** (a small Keras/TensorFlow or PyTorch MLP) the
  user can choose to run — doesn't need to outperform XGBoost, needs to exist
  and be evaluated the same way.
- **Full evaluation suite**: train/test split, cross-validation, confusion
  matrix, classification report, ROC-AUC, precision-recall curve.
- Feature importance / SHAP explanation of the winning model.

## Validity Dashboard — five evidence sources

Per *Standards for Educational and Psychological Testing*: content evidence,
**response processes** (often "documentation pending" rather than computed —
give it a place with a status + free-text/link field, don't skip it),
internal structure, relations to other variables, consequences of testing.

## Decision Support & Fairness Explorer

Two pages not in the original SRS numbering but required: Decision Support
(cut scores, sensitivity/specificity, PPV/NPV, ROC/AUC, confusion matrix at a
chosen threshold) and Fairness Explorer (DIF, measurement invariance framing,
group comparisons, Cohen's d). Both are shown in the mockups — build to those.

## Concepts that must be teachable somewhere (Learning Mode + tooltips)

Reuse the psychometrics course concept map (`docs/mockups/psykometri_begrepp.png`)
as the content outline for Learning Mode: the six pillars (Verkligheten →
Testutveckling → Validitet → Reliabilitet → Tolka poäng → Beslut), plus
Fairness. Every KPI card that shows a statistic (α, SEM, factor loading, T-
score, AUC, Cohen's d, etc.) should have a small info icon with a 1–3 sentence
plain-language explanation — this is what makes the tool "teach," not just
report numbers.

## Sample data

No real datasets — generate synthetic but psychometrically plausible data
(correlated items via a latent-trait/multivariate-normal simulation, matching
realistic reliability ~0.85-0.92) for GAD-7 (N≈2450), PHQ-9 (N≈250), IPIP Big
Five (N≈250). Keep the generation script in `data/generate_sample_data.py`,
seeded, and commit the resulting CSVs so the app works out of the box.

## Git workflow (solo project — keep it simple)

The repo already has an initial commit on `main` and a GitHub remote
(`origin`) configured, pushed manually before this session started. This is a
solo project with no reviewers, so **skip the pull-request flow**:
- Work normally in your session.
- At the end of each sprint, once `streamlit run app.py` runs cleanly and the
  sprint's Definition of Done is met, commit with a clear message, then merge
  your session branch into `main` and `git push origin main` directly
  (`git checkout main && git merge <session-branch> && git push`). Do not
  leave finished work stranded only on an isolated session branch.
- Small, frequent commits per logical unit of work (e.g. "Add plugin engine
  and GAD-7/PHQ-9/Big Five plugin definitions") rather than one giant commit
  per sprint.

## Definition of Done (per module)

A module/page is complete only when it has: working functionality across all
three test types, at least one unit test in `tests/` for its `core/` logic,
graceful error handling (missing data, wrong file type, no plugin match),
an interpretation/explanation panel (not just raw numbers), and export support
via `export_engine`.

## When something in this file conflicts with docs/BUILD_PLAN.md

This file wins on architecture/standards; BUILD_PLAN.md wins on sequencing
and scope of what to build next.
