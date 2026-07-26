import numpy as np
import pandas as pd
import pytest

from core import ml_engine as ml
from core.data_model import Dataset, Item, Questionnaire, ResponseScale, Subscale


def _dataset(n=300, n_items=8, seed=21, with_missing=True) -> Dataset:
    rng = np.random.default_rng(seed)
    q = Questionnaire(
        plugin_id="demo",
        plugin_version="1.0",
        test_name="Demo",
        full_name="Demo Test",
        response_scale=ResponseScale(min=0, max=3),
        items=[Item(id=f"D{i}", text=f"Item {i}", subscale="total", reverse_scored=False) for i in range(1, n_items + 1)],
        subscales=[Subscale(id="total", name="Total", item_ids=[f"D{i}" for i in range(1, n_items + 1)], score_range=(0, 3 * n_items))],
    )
    theta = rng.normal(0, 1, size=n)
    data = {"respondent_id": np.arange(1, n + 1)}
    for i in range(1, n_items + 1):
        continuous = 0.75 * theta + np.sqrt(1 - 0.75**2) * rng.normal(0, 1, size=n)
        vals = np.clip(np.digitize(continuous, [-1, 0, 1]), 0, 3).astype(float)
        if with_missing:
            missing_mask = rng.random(n) < 0.03
            vals[missing_mask] = np.nan
        data[f"D{i}"] = vals
    raw = pd.DataFrame(data)
    scored = raw.copy()
    scored["total_total"] = scored[[f"D{i}" for i in range(1, n_items + 1)]].sum(axis=1)

    raw["age"] = rng.integers(18, 70, n).astype(float)
    raw["gender"] = rng.choice(["Kvinna", "Man"], n)
    raw["education"] = rng.choice(["Gymnasium", "Högskola/Universitet"], n)
    raw["group"] = rng.choice(["Grupp A", "Grupp B"], n)

    logit = 1.3 * (theta - 0.5)
    p = 1 / (1 + np.exp(-logit))
    raw["outcome_positive"] = (rng.random(n) < p).astype(int)

    return Dataset(raw=raw, scored=scored, questionnaire=q, column_mapping={}, demographic_columns=["age", "gender", "education", "group"])


def test_prepare_ml_data_imputes_missing_and_encodes_categoricals():
    dataset = _dataset()
    ml_data = ml.prepare_ml_data(dataset)
    assert ml_data is not None
    assert ml_data.X.isna().sum().sum() == 0
    assert ml_data.report.item_missing_values_imputed > 0
    assert set(ml_data.report.categorical_columns_encoded) == {"gender", "education", "group"}
    assert "gender_Kvinna" in ml_data.X.columns
    assert "age" in ml_data.X.columns
    assert ml_data.n_samples == len(dataset.raw)


def test_prepare_ml_data_drops_rows_missing_target():
    dataset = _dataset()
    dataset.raw.loc[dataset.raw.index[:10], "outcome_positive"] = np.nan
    ml_data = ml.prepare_ml_data(dataset)
    assert ml_data.report.n_rows_dropped == 10
    assert ml_data.n_samples == len(dataset.raw) - 10


def test_prepare_ml_data_none_without_target_column():
    dataset = _dataset()
    dataset.raw = dataset.raw.drop(columns=["outcome_positive"])
    assert ml.prepare_ml_data(dataset) is None


def test_run_kmeans_cluster_sizes_sum_to_n_samples():
    dataset = _dataset(n=200)
    ml_data = ml.prepare_ml_data(dataset)
    clusters = ml.run_kmeans(ml_data, n_clusters=3)
    assert sum(clusters.cluster_sizes.values()) == ml_data.n_samples
    assert set(clusters.labels) <= {0, 1, 2}
    assert clusters.silhouette is not None


def test_run_pca_shapes_and_variance():
    dataset = _dataset(n=200)
    ml_data = ml.prepare_ml_data(dataset)
    pca = ml.run_pca(ml_data, n_components=2)
    assert pca.components.shape == (ml_data.n_samples, 2)
    assert len(pca.explained_variance_ratio) == 2
    assert pca.explained_variance_ratio.sum() <= 1.0 + 1e-9
    assert pca.loadings.shape[1] == 2


@pytest.mark.parametrize("model_name", ml.MODEL_NAMES)
def test_evaluate_model_metrics_within_bounds(model_name):
    dataset = _dataset(n=300)
    ml_data = ml.prepare_ml_data(dataset)
    X_train, X_test, y_train, y_test = ml.train_test_split_data(ml_data)
    result = ml.evaluate_model(model_name, X_train, X_test, y_train, y_test)

    for value in (result.accuracy, result.precision, result.recall, result.f1, result.roc_auc, result.cv_auc_mean):
        assert 0.0 <= value <= 1.0
    assert result.confusion.sum() == len(y_test)
    assert result.model_name == model_name


def test_torch_mlp_is_sklearn_compatible_for_cross_validation():
    from sklearn.model_selection import cross_val_score

    dataset = _dataset(n=200)
    ml_data = ml.prepare_ml_data(dataset)
    model = ml.TorchMLPClassifier(epochs=10)
    scores = cross_val_score(model, ml_data.X, ml_data.y, cv=3, scoring="roc_auc")
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_best_model_name_picks_highest_auc():
    dataset = _dataset(n=300)
    ml_data = ml.prepare_ml_data(dataset)
    X_train, X_test, y_train, y_test = ml.train_test_split_data(ml_data)
    results = {name: ml.evaluate_model(name, X_train, X_test, y_train, y_test) for name in ["Logistic Regression", "Random Forest"]}
    best = ml.best_model_name(results)
    assert best in results
    assert results[best].roc_auc == max(r.roc_auc for r in results.values())


def test_shap_feature_importance_matches_features_and_nonnegative():
    dataset = _dataset(n=300)
    ml_data = ml.prepare_ml_data(dataset)
    X_train, X_test, y_train, y_test = ml.train_test_split_data(ml_data)
    result = ml.evaluate_model("Random Forest", X_train, X_test, y_train, y_test)
    shap_importance = ml.shap_feature_importance(result.model, X_test)
    assert shap_importance is not None
    assert set(shap_importance.index) == set(X_test.columns)
    assert (shap_importance >= 0).all()


def test_feature_vector_and_actual_outcome_for_person():
    dataset = _dataset(n=200)
    ml_data = ml.prepare_ml_data(dataset)
    person_id = int(dataset.raw["respondent_id"].iloc[0])
    fv = ml.feature_vector_for_person(ml_data, dataset, person_id)
    assert fv is not None
    assert set(fv.index) == set(ml_data.X.columns)

    actual = ml.actual_outcome_for_person(dataset, person_id)
    assert actual in (0, 1)
    assert actual == int(dataset.raw.loc[dataset.raw["respondent_id"] == person_id, "outcome_positive"].iloc[0])


def test_feature_vector_for_person_returns_none_for_unknown_id():
    dataset = _dataset(n=100)
    ml_data = ml.prepare_ml_data(dataset)
    assert ml.feature_vector_for_person(ml_data, dataset, 999999) is None


def test_baseline_feature_vector_covers_all_columns_and_is_within_range():
    dataset = _dataset(n=200)
    ml_data = ml.prepare_ml_data(dataset)
    baseline = ml.baseline_feature_vector(ml_data)
    assert set(baseline.index) == set(ml_data.X.columns)
    for col in ml_data.X.columns:
        assert ml_data.X[col].min() <= baseline[col] <= ml_data.X[col].max()


def test_predict_proba_for_vector_matches_manual_call():
    dataset = _dataset(n=200)
    ml_data = ml.prepare_ml_data(dataset)
    result = ml.run_all_models(ml_data, model_names=["Logistic Regression"])
    model = result["Logistic Regression"].model
    baseline = ml.baseline_feature_vector(ml_data)
    proba = ml.predict_proba_for_vector(model, baseline)
    expected = float(model.predict_proba(baseline.to_frame().T)[:, 1][0])
    assert proba == pytest.approx(expected)
    assert 0.0 <= proba <= 1.0
