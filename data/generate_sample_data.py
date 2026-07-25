"""Generates synthetic but psychometrically plausible sample datasets for the
three bundled plugins (GAD-7, PHQ-9, IPIP Big Five).

Item responses are simulated from a latent-trait (factor) model: each item is
a noisy, discretized function of one or more underlying trait scores, which
produces realistic inter-item correlations and Cronbach's alpha in the
0.85-0.92 range without hand-tuning every cell. Seeded for reproducibility;
run this script to regenerate the committed CSVs in data/sample_datasets/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data_model import Questionnaire
from core.plugin_engine import load_plugin

PLUGINS_DIR = Path(__file__).resolve().parent.parent / "plugins"
OUTPUT_DIR = Path(__file__).resolve().parent / "sample_datasets"

GENDER_CATEGORIES = ["Kvinna", "Man", "Annat"]
GENDER_PROBS = [0.62, 0.37, 0.01]


def _discretize(continuous: np.ndarray, thresholds: list[float]) -> np.ndarray:
    """Bin a continuous z-score array into 0..len(thresholds) ordinal levels."""
    return np.digitize(continuous, thresholds)


def _add_demographics(rng: np.random.Generator, n: int) -> pd.DataFrame:
    age = rng.normal(38, 13, size=n).clip(18, 75).round().astype(int)
    gender = rng.choice(GENDER_CATEGORIES, size=n, p=GENDER_PROBS)
    return pd.DataFrame({"age": age, "gender": gender})


def generate_single_factor_dataset(
    questionnaire: Questionnaire,
    n: int,
    theta_mean: float,
    theta_sd: float,
    thresholds: list[float],
    loading_range: tuple[float, float],
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    theta = rng.normal(theta_mean, theta_sd, size=n)

    columns: dict[str, np.ndarray] = {"respondent_id": np.arange(1, n + 1)}
    for item in questionnaire.items:
        loading = rng.uniform(*loading_range)
        noise = rng.normal(0, 1, size=n)
        continuous = loading * theta + np.sqrt(max(1 - loading**2, 0.05)) * noise
        responses = _discretize(continuous, thresholds).astype(float)

        # sprinkle a small amount of missingness, as in a real dataset
        missing_mask = rng.random(n) < 0.01
        responses[missing_mask] = np.nan
        columns[item.id] = responses

    df = pd.DataFrame(columns)
    demographics = _add_demographics(rng, n)
    return pd.concat([df, demographics], axis=1)


def generate_multi_factor_dataset(
    questionnaire: Questionnaire,
    n: int,
    factor_corr: pd.DataFrame,
    thresholds: list[float],
    loading_range: tuple[float, float],
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factor_ids = list(factor_corr.columns)
    factor_scores = rng.multivariate_normal(
        mean=np.zeros(len(factor_ids)), cov=factor_corr.values, size=n
    )
    theta_by_factor = {fid: factor_scores[:, i] for i, fid in enumerate(factor_ids)}

    columns: dict[str, np.ndarray] = {"respondent_id": np.arange(1, n + 1)}
    for item in questionnaire.items:
        theta = theta_by_factor[item.subscale]
        loading = rng.uniform(*loading_range)
        sign = -1.0 if item.reverse_scored else 1.0
        noise = rng.normal(0, 1, size=n)
        continuous = sign * loading * theta + np.sqrt(max(1 - loading**2, 0.05)) * noise
        responses = _discretize(continuous, thresholds).astype(float)

        missing_mask = rng.random(n) < 0.01
        responses[missing_mask] = np.nan
        columns[item.id] = responses

    df = pd.DataFrame(columns)
    demographics = _add_demographics(rng, n)
    return pd.concat([df, demographics], axis=1)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gad7 = load_plugin(PLUGINS_DIR / "gad7.json")
    phq9 = load_plugin(PLUGINS_DIR / "phq9.json")
    bigfive = load_plugin(PLUGINS_DIR / "ipip_bigfive.json")
    assert gad7 and phq9 and bigfive, "Plugins must be valid before generating sample data"

    # GAD-7: general-population screening sample, floor-effect skew typical
    # of anxiety screening (most respondents endorse "not at all"/"several days").
    gad7_df = generate_single_factor_dataset(
        gad7,
        n=2450,
        theta_mean=-0.2,
        theta_sd=1.0,
        thresholds=[-1.0, 0.25, 1.25],
        loading_range=(0.65, 0.82),
        seed=101,
    )
    gad7_df.to_csv(OUTPUT_DIR / "gad7_sample.csv", index=False)

    # PHQ-9: primary-care sample, slightly right-shifted so all severity
    # bands are represented (matches the reference mockup's cut-off spread).
    phq9_df = generate_single_factor_dataset(
        phq9,
        n=250,
        theta_mean=0.3,
        theta_sd=1.1,
        thresholds=[-1.0, 0.25, 1.25],
        loading_range=(0.6, 0.8),
        seed=202,
    )
    phq9_df.to_csv(OUTPUT_DIR / "phq9_sample.csv", index=False)

    # IPIP Big Five: five mildly-correlated latent factors (E, A, C, N, O),
    # reflecting typical small empirical inter-correlations between traits.
    factor_ids = ["E", "A", "C", "N", "O"]
    corr = pd.DataFrame(np.eye(5), index=factor_ids, columns=factor_ids)
    corr.loc["E", "N"] = corr.loc["N", "E"] = -0.25
    corr.loc["A", "C"] = corr.loc["C", "A"] = 0.20
    corr.loc["A", "N"] = corr.loc["N", "A"] = -0.15
    corr.loc["O", "E"] = corr.loc["E", "O"] = 0.15
    bigfive_df = generate_multi_factor_dataset(
        bigfive,
        n=250,
        factor_corr=corr,
        thresholds=[-1.28, -0.52, 0.52, 1.28],
        loading_range=(0.6, 0.8),
        seed=303,
    )
    bigfive_df.to_csv(OUTPUT_DIR / "bigfive_sample.csv", index=False)

    print(f"Wrote {len(gad7_df)} GAD-7 rows, {len(phq9_df)} PHQ-9 rows, "
          f"{len(bigfive_df)} Big Five rows to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
