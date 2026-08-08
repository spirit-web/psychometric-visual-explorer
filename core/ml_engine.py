"""Machine learning pipeline for the Machine Learning Explorer: data prep,
unsupervised learning (KMeans/PCA), supervised learning (Logistic
Regression, Random Forest, XGBoost, a small PyTorch MLP), full evaluation,
and SHAP-based feature importance.

Pure logic only - no Streamlit imports, no plotting (see core/viz_engine.py).
Built against core.data_model.Dataset the same way as core/stats_engine.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

TARGET_COL = "outcome_positive"
DEMOGRAPHIC_CATEGORICALS = ["gender", "education", "group"]


# --- data preparation -------------------------------------------------


@dataclass
class DataPrepReport:
    n_rows_total: int
    n_rows_used: int
    n_rows_dropped: int
    item_missing_values_imputed: int
    demographic_missing_values_imputed: int
    categorical_columns_encoded: list[str]
    feature_columns: list[str]


@dataclass
class MLDataset:
    X: pd.DataFrame
    y: pd.Series
    feature_names: list[str]
    n_samples: int
    report: DataPrepReport


def prepare_ml_data(dataset, target_col: str = TARGET_COL) -> MLDataset | None:
    """Builds a clean, model-ready dataframe from item responses and
    demographics: missing item values are median-imputed, categorical
    demographics are one-hot encoded, and rows missing the target are
    dropped (they can't be used for supervised training)."""
    if target_col not in dataset.raw.columns:
        return None

    q = dataset.questionnaire
    item_cols = [c for c in q.item_ids if c in dataset.scored.columns]
    if not item_cols:
        return None

    items = dataset.scored[item_cols].copy()
    items_missing = int(items.isna().sum().sum())
    items = items.fillna(items.median(numeric_only=True))

    demo_frames = []
    demo_missing = 0
    if "age" in dataset.raw.columns:
        age = pd.to_numeric(dataset.raw["age"], errors="coerce")
        demo_missing += int(age.isna().sum())
        age = age.fillna(age.median())
        demo_frames.append(age.rename("age"))

    encoded_cols: list[str] = []
    for cat_col in DEMOGRAPHIC_CATEGORICALS:
        if cat_col in dataset.raw.columns:
            series = dataset.raw[cat_col]
            demo_missing += int(series.isna().sum())
            dummies = pd.get_dummies(series, prefix=cat_col, dtype=float)
            demo_frames.append(dummies)
            encoded_cols.append(cat_col)

    X_full = pd.concat([items] + demo_frames, axis=1) if demo_frames else items
    y_full = pd.to_numeric(dataset.raw[target_col], errors="coerce")

    combined = pd.concat([X_full, y_full.rename("__target__")], axis=1).dropna(subset=["__target__"])
    n_total = len(dataset.raw)
    n_used = len(combined)
    if n_used < 20 or combined["__target__"].nunique() < 2:
        return None

    y = combined.pop("__target__").astype(int)
    X = combined

    report = DataPrepReport(
        n_rows_total=n_total,
        n_rows_used=n_used,
        n_rows_dropped=n_total - n_used,
        item_missing_values_imputed=items_missing,
        demographic_missing_values_imputed=demo_missing,
        categorical_columns_encoded=encoded_cols,
        feature_columns=list(X.columns),
    )
    return MLDataset(X=X, y=y, feature_names=list(X.columns), n_samples=n_used, report=report)


# --- unsupervised learning -------------------------------------------------


@dataclass
class ClusterResult:
    labels: np.ndarray
    n_clusters: int
    silhouette: float | None
    cluster_sizes: dict[int, int]
    cluster_profile: pd.DataFrame


def run_kmeans(ml_data: MLDataset, n_clusters: int = 3, seed: int = 42) -> ClusterResult:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    X_scaled = StandardScaler().fit_transform(ml_data.X)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(X_scaled)

    silhouette = None
    if n_clusters > 1 and len(set(labels)) > 1:
        silhouette = float(silhouette_score(X_scaled, labels))

    sizes = {int(k): int(v) for k, v in pd.Series(labels).value_counts().sort_index().items()}
    profile = ml_data.X.copy()
    profile["cluster"] = labels
    cluster_profile = profile.groupby("cluster").mean()

    return ClusterResult(labels, n_clusters, silhouette, sizes, cluster_profile)


@dataclass
class PCAResult:
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    loadings: pd.DataFrame


def run_pca(ml_data: MLDataset, n_components: int = 2, seed: int = 42) -> PCAResult:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X_scaled = StandardScaler().fit_transform(ml_data.X)
    n_components = min(n_components, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    components = pca.fit_transform(X_scaled)
    loadings = pd.DataFrame(
        pca.components_.T, index=ml_data.X.columns, columns=[f"PC{i + 1}" for i in range(n_components)]
    )
    return PCAResult(components, pca.explained_variance_ratio_, loadings)


# --- supervised learning -------------------------------------------------


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """A small PyTorch MLP wrapped in a scikit-learn-compatible interface
    (fit/predict/predict_proba, inherited get_params/set_params/tags via
    BaseEstimator/ClassifierMixin) so it can go through exactly the same
    train/evaluate/cross-validate pipeline as the other models - the deep
    learning option doesn't need its own special-cased code path."""

    def __init__(self, hidden_sizes: tuple[int, ...] = (32, 16), epochs: int = 60, lr: float = 1e-3, seed: int = 42):
        self.hidden_sizes = hidden_sizes
        self.epochs = epochs
        self.lr = lr
        self.seed = seed

    def fit(self, X, y):
        import torch
        from sklearn.preprocessing import StandardScaler
        from torch import nn

        torch.manual_seed(self.seed)
        X_arr = X.values if hasattr(X, "values") else np.asarray(X)
        y_arr = y.values if hasattr(y, "values") else np.asarray(y)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_arr)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(y_arr, dtype=torch.float32).view(-1, 1)

        layers = []
        in_size = X_scaled.shape[1]
        for hidden in self.hidden_sizes:
            layers += [nn.Linear(in_size, hidden), nn.ReLU()]
            in_size = hidden
        layers += [nn.Linear(in_size, 1)]
        self._model = nn.Sequential(*layers)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()
        self._model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = criterion(self._model(X_t), y_t)
            loss.backward()
            optimizer.step()

        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        import torch

        X_arr = X.values if hasattr(X, "values") else np.asarray(X)
        X_scaled = self._scaler.transform(X_arr)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        self._model.eval()
        with torch.no_grad():
            proba_pos = torch.sigmoid(self._model(X_t)).numpy().flatten()
        return np.column_stack([1 - proba_pos, proba_pos])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_test_split_data(ml_data: MLDataset, test_size: float = 0.25, seed: int = 42):
    from sklearn.model_selection import train_test_split

    return train_test_split(ml_data.X, ml_data.y, test_size=test_size, random_state=seed, stratify=ml_data.y)


def make_model(name: str, seed: int = 42):
    """Factory for the four models compared on the Machine Learning page.
    Logistic Regression is included as an interpretable baseline alongside
    the required non-linear models, not as the sole model."""
    if name == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=1000, random_state=seed)
    if name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=200, max_depth=6, random_state=seed)
    if name == "XGBoost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="logloss", random_state=seed
        )
    if name == "Neural Network (MLP)":
        return TorchMLPClassifier(seed=seed)
    raise ValueError(f"Unknown model: {name}")


MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "Neural Network (MLP)"]


@dataclass
class ModelEvalResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    cv_auc_mean: float
    cv_auc_std: float
    confusion: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    precision_curve: np.ndarray
    recall_curve: np.ndarray
    classification_report: str
    feature_importance: pd.Series | None
    model: object = field(repr=False)


def evaluate_model(
    model_name: str, X_train, X_test, y_train, y_test, cv_folds: int = 5, seed: int = 42
) -> ModelEvalResult:
    from sklearn.metrics import (
        accuracy_score,
        classification_report as sk_classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    model = make_model(model_name, seed=seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(make_model(model_name, seed=seed), X_train, y_train, cv=cv, scoring="roc_auc")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)

    importance = None
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    return ModelEvalResult(
        model_name=model_name,
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_test, y_proba)),
        cv_auc_mean=float(cv_scores.mean()),
        cv_auc_std=float(cv_scores.std()),
        confusion=confusion_matrix(y_test, y_pred),
        fpr=fpr,
        tpr=tpr,
        precision_curve=prec_curve,
        recall_curve=rec_curve,
        classification_report=sk_classification_report(y_test, y_pred, zero_division=0, target_names=["Negativ", "Positiv"]),
        feature_importance=importance,
        model=model,
    )


def run_all_models(ml_data: MLDataset, model_names: list[str] | None = None, test_size: float = 0.25, seed: int = 42) -> dict[str, ModelEvalResult]:
    X_train, X_test, y_train, y_test = train_test_split_data(ml_data, test_size=test_size, seed=seed)
    names = model_names or MODEL_NAMES
    return {name: evaluate_model(name, X_train, X_test, y_train, y_test, seed=seed) for name in names}


def best_model_name(results: dict[str, ModelEvalResult]) -> str | None:
    if not results:
        return None
    return max(results, key=lambda name: results[name].roc_auc)


# --- feature importance / SHAP -------------------------------------------------


def shap_feature_importance(model, X_sample: pd.DataFrame, max_samples: int = 100) -> pd.Series | None:
    """Mean absolute SHAP value per feature for a tree-based model
    (TreeExplainer). Returns None for model types SHAP's fast tree explainer
    doesn't support (e.g. the PyTorch MLP) - callers should fall back to
    the model's built-in feature_importances_ in that case."""
    import shap

    sample = X_sample.sample(min(max_samples, len(X_sample)), random_state=42) if len(X_sample) > max_samples else X_sample
    try:
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(sample)
    except Exception:
        return None

    # shap's return shape varies across library/sklearn versions - normalize
    # to a single (n_samples, n_features) array for the positive class.
    if isinstance(raw, list):
        values = raw[1] if len(raw) > 1 else raw[0]
    else:
        values = raw
        if values.ndim == 3:
            values = values[:, :, 1]

    mean_abs = np.abs(values).mean(axis=0)
    return pd.Series(mean_abs, index=sample.columns).sort_values(ascending=False)


# --- quick prediction (single respondent) -------------------------------------------------


def _row_index_for_person(dataset, person_id):
    if "respondent_id" in dataset.raw.columns:
        mask = dataset.raw["respondent_id"] == person_id
        if not mask.any():
            return None
        return dataset.raw.index[mask][0]
    return person_id


def feature_vector_for_person(ml_data: MLDataset, dataset, person_id) -> pd.Series | None:
    idx = _row_index_for_person(dataset, person_id)
    if idx is None or idx not in ml_data.X.index:
        return None
    return ml_data.X.loc[idx]


def actual_outcome_for_person(dataset, person_id) -> int | None:
    if TARGET_COL not in dataset.raw.columns:
        return None
    idx = _row_index_for_person(dataset, person_id)
    if idx is None or idx not in dataset.raw.index:
        return None
    val = dataset.raw.loc[idx, TARGET_COL]
    return int(val) if pd.notna(val) else None


def baseline_feature_vector(ml_data: MLDataset) -> pd.Series:
    """A 'typical' feature row (median of every column) - seeds the
    interactive prediction sliders and fills in any feature the UI doesn't
    expose a control for (e.g. one-hot demographic columns)."""
    return ml_data.X.median(numeric_only=True)


def predict_proba_for_vector(model, feature_vector: pd.Series) -> float:
    """Predicted probability of the positive class for a single feature
    row - used for the live 'what if this item answer changed' slider UI."""
    return float(model.predict_proba(feature_vector.to_frame().T)[:, 1][0])


def predict_proba_for_all(model, X: pd.DataFrame) -> pd.Series:
    """Predicted probability of the positive class for every row in X,
    indexed the same as X (a subset of the original dataset's index - see
    prepare_ml_data). Used to audit whether a trained model's risk score is
    distributed fairly across demographic groups (Fairness Explorer's
    'Modellens risksannolikhet' comparison mode), rather than only ever
    checking fairness of the raw questionnaire total."""
    proba = model.predict_proba(X)[:, 1]
    return pd.Series(proba, index=X.index, name="predicted_probability")