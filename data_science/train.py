from __future__ import annotations

import json
import os
import warnings

if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)
warnings.filterwarnings(
    "ignore",
    message=".*Could not find the number of physical cores.*",
    category=UserWarning,
)

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from data_science.config import CONFIG, ensure_project_dirs


CATEGORICAL_FEATURES = [
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "perceived_colour_value_name",
    "perceived_colour_master_name",
    "department_name",
    "index_name",
    "section_name",
    "garment_group_name",
    "prediction_season",
]

NUMERIC_FEATURES = [
    "article_count",
    "hist_units",
    "hist_customers",
    "hist_avg_price",
    "hist_max_price",
    "hist_active_days",
    "recent_units",
    "recent_customers",
    "recent_avg_price",
    "recent_max_price",
    "recent_active_days",
    "prior_units",
    "prior_customers",
    "prior_avg_price",
    "prior_max_price",
    "prior_active_days",
    "last14_units",
    "last14_customers",
    "last14_active_days",
    "prev14_units",
    "prev14_customers",
    "prev14_active_days",
    "hist_units_per_active_day",
    "recent_vs_prior_units",
    "last14_vs_prev14_units",
    "unit_demand_acceleration",
    "customer_demand_acceleration",
    "days_since_last_sale",
    "observed_style_age_days",
    "prediction_month",
    "prediction_month_sin",
    "prediction_month_cos",
    "recent_units_vs_category_median",
    "momentum_vs_category_median",
    "acceleration_vs_category_median",
    "recent_category_share",
]

EXPLANATION_GROUPS = {
    "Recent demand": [
        "recent_units",
        "recent_customers",
        "recent_active_days",
        "last14_units",
        "last14_customers",
        "last14_active_days",
    ],
    "Historical demand": [
        "article_count",
        "hist_units",
        "hist_customers",
        "hist_active_days",
        "hist_units_per_active_day",
    ],
    "Momentum and acceleration": [
        "prior_units",
        "prior_customers",
        "prev14_units",
        "prev14_customers",
        "recent_vs_prior_units",
        "last14_vs_prev14_units",
        "unit_demand_acceleration",
        "customer_demand_acceleration",
        "days_since_last_sale",
    ],
    "Category-relative demand": [
        "recent_units_vs_category_median",
        "momentum_vs_category_median",
        "acceleration_vs_category_median",
        "recent_category_share",
    ],
    "Lifecycle and season": [
        "observed_style_age_days",
        "prediction_month",
        "prediction_month_sin",
        "prediction_month_cos",
        "prediction_season",
    ],
    "Price behavior": [
        "hist_avg_price",
        "hist_max_price",
        "recent_avg_price",
        "recent_max_price",
        "prior_avg_price",
        "prior_max_price",
    ],
    "Product attributes": [
        feature for feature in CATEGORICAL_FEATURES if feature != "prediction_season"
    ],
}

MODEL_LABELS = {
    "rolling_regression": "Two-period rolling Random Forest log-sales ranker",
    "expanding_regression": "Expanding-window Random Forest log-sales ranker",
    "top_decile_classifier": "Random Forest top-decile classifier",
    "blended_ranker": "55% rolling-regression rank + 45% top-decile probability rank",
}

BLEND_REGRESSION_WEIGHT = 0.55
BLEND_CLASSIFIER_WEIGHT = 0.45


def _load_dataset() -> pd.DataFrame:
    if not CONFIG.processed_dataset_path.exists():
        raise FileNotFoundError(
            f"Missing {CONFIG.processed_dataset_path}. Run data_science.features.build_features first."
        )
    return pd.read_csv(CONFIG.processed_dataset_path)


def _available_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return numeric, categorical


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def build_regression_model(numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", _preprocessor(numeric, categorical)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=120,
                    min_samples_leaf=5,
                    max_features=0.75,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def build_classifier_model(numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", _preprocessor(numeric, categorical)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=160,
                    min_samples_leaf=20,
                    max_features=0.75,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def _precision_at_k(y_true: pd.Series, y_score: pd.Series, k: int) -> float:
    k = min(k, len(y_true))
    if k == 0:
        return 0.0
    actual_top = set(y_true.sort_values(ascending=False).head(k).index)
    predicted_top = set(y_score.sort_values(ascending=False).head(k).index)
    return len(actual_top & predicted_top) / k


def _rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _top_decile_labels(frame: pd.DataFrame) -> pd.Series:
    thresholds = frame.groupby("snapshot_id")["target_units"].transform(
        lambda values: values.quantile(CONFIG.top_decile_quantile)
    )
    return (frame["target_units"] >= thresholds).astype(int)


def _classifier_probability(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(frame)
    classes = list(model.named_steps["model"].classes_)
    if 1 not in classes:
        return np.zeros(len(frame), dtype=float)
    return probabilities[:, classes.index(1)]


def _predicted_units(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    return np.expm1(model.predict(frame)).clip(min=0)


def _empirical_percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=float))
    values = np.asarray(values, dtype=float)
    if len(reference) == 0:
        return np.zeros(len(values), dtype=float)
    return np.searchsorted(reference, values, side="right") / len(reference)


def _candidate_scores(
    regression_model: Pipeline,
    classifier_model: Pipeline,
    training_features: pd.DataFrame,
    scoring_features: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    train_units = _predicted_units(regression_model, training_features)
    train_probability = _classifier_probability(classifier_model, training_features)
    predicted_units = _predicted_units(regression_model, scoring_features)
    classifier_probability = _classifier_probability(classifier_model, scoring_features)
    regression_rank = pd.Series(predicted_units).rank(pct=True, method="average").to_numpy()
    classifier_rank = (
        pd.Series(classifier_probability).rank(pct=True, method="average").to_numpy()
    )
    scores = {
        "regression": predicted_units,
        "top_decile_classifier": classifier_probability,
        "blended_ranker": (
            BLEND_REGRESSION_WEIGHT * regression_rank
            + BLEND_CLASSIFIER_WEIGHT * classifier_rank
        ),
    }
    references = {
        "regression_units": train_units,
        "classifier_probability": train_probability,
    }
    return scores, predicted_units, classifier_probability, references


def _fold_metrics(
    validation: pd.DataFrame,
    scores: dict[str, np.ndarray],
    unit_predictions: dict[str, np.ndarray],
) -> dict:
    candidate_metrics = {}
    for name, score in scores.items():
        score_series = pd.Series(score, index=validation.index)
        candidate_metrics[name] = {
            "precision_at_3": float(
                _precision_at_k(validation["target_units"], score_series, 3)
            ),
            "precision_at_10": float(
                _precision_at_k(validation["target_units"], score_series, 10)
            ),
        }
    for name, values in unit_predictions.items():
        prediction_series = pd.Series(values, index=validation.index)
        candidate_metrics[name]["mae"] = float(
            mean_absolute_error(validation["target_units"], prediction_series)
        )
        candidate_metrics[name]["rmse"] = _rmse(
            validation["target_units"], prediction_series
        )
    baseline = validation["recent_units"].clip(lower=0)
    baseline_metrics = {
        "mae": float(mean_absolute_error(validation["target_units"], baseline)),
        "rmse": _rmse(validation["target_units"], baseline),
        "precision_at_3": float(
            _precision_at_k(validation["target_units"], baseline, 3)
        ),
        "precision_at_10": float(
            _precision_at_k(validation["target_units"], baseline, 10)
        ),
    }
    return {"candidates": candidate_metrics, "baseline_recent_sales": baseline_metrics}


def _aggregate_backtests(folds: list[dict]) -> tuple[dict, dict]:
    aggregate = {}
    for name in MODEL_LABELS:
        p3 = np.array([fold["candidates"][name]["precision_at_3"] for fold in folds])
        p10 = np.array([fold["candidates"][name]["precision_at_10"] for fold in folds])
        stability = max(0.0, 1.0 - float((p3.std() + p10.std()) / 2))
        selection_score = (
            float(p3.mean())
            + 0.5 * float(p10.mean())
            - 0.25 * float(p3.std())
            - 0.10 * float(p10.std())
        )
        aggregate[name] = {
            "model_name": MODEL_LABELS[name],
            "mean_precision_at_3": float(p3.mean()),
            "std_precision_at_3": float(p3.std()),
            "mean_precision_at_10": float(p10.mean()),
            "std_precision_at_10": float(p10.std()),
            "ranking_stability": stability,
            "selection_score": selection_score,
        }
        if "mae" in folds[0]["candidates"][name]:
            aggregate[name]["mean_mae"] = float(
                np.mean([fold["candidates"][name]["mae"] for fold in folds])
            )
            aggregate[name]["mean_rmse"] = float(
                np.mean([fold["candidates"][name]["rmse"] for fold in folds])
            )

    baseline_p3 = np.array(
        [fold["baseline_recent_sales"]["precision_at_3"] for fold in folds]
    )
    baseline_p10 = np.array(
        [fold["baseline_recent_sales"]["precision_at_10"] for fold in folds]
    )
    baseline = {
        "mean_mae": float(
            np.mean([fold["baseline_recent_sales"]["mae"] for fold in folds])
        ),
        "mean_rmse": float(
            np.mean([fold["baseline_recent_sales"]["rmse"] for fold in folds])
        ),
        "mean_precision_at_3": float(baseline_p3.mean()),
        "std_precision_at_3": float(baseline_p3.std()),
        "mean_precision_at_10": float(baseline_p10.mean()),
        "std_precision_at_10": float(baseline_p10.std()),
    }
    return aggregate, baseline


def _reference_values(
    training: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> dict[str, object]:
    reference: dict[str, object] = {}
    for feature in numeric:
        value = training[feature].median()
        reference[feature] = float(value) if pd.notna(value) else 0.0
    for feature in categorical:
        mode = training[feature].dropna().mode()
        reference[feature] = mode.iloc[0] if not mode.empty else ""
    return reference


def _single_ranking_score(
    selected_model: str,
    regression_model: Pipeline,
    classifier_model: Pipeline,
    frame: pd.DataFrame,
    ranking_references: dict[str, np.ndarray],
) -> float:
    units = _predicted_units(regression_model, frame)
    probability = _classifier_probability(classifier_model, frame)
    if selected_model in {"rolling_regression", "expanding_regression"}:
        return float(units[0])
    if selected_model == "top_decile_classifier":
        return float(probability[0])
    regression_rank = _empirical_percentile(
        units, ranking_references["regression_units"]
    )[0]
    classifier_rank = _empirical_percentile(
        probability, ranking_references["classifier_probability"]
    )[0]
    return float(
        BLEND_REGRESSION_WEIGHT * regression_rank
        + BLEND_CLASSIFIER_WEIGHT * classifier_rank
    )


def _group_observation(group: str, row: pd.Series) -> str:
    if group == "Recent demand":
        return (
            f"{int(row.get('recent_units', 0)):,} recent units and "
            f"{int(row.get('last14_units', 0)):,} units in the latest 14 days"
        )
    if group == "Historical demand":
        return (
            f"{int(row.get('hist_units', 0)):,} units from "
            f"{int(row.get('hist_customers', 0)):,} customers across "
            f"{int(row.get('hist_active_days', 0))} selling days"
        )
    if group == "Momentum and acceleration":
        return (
            f"recent/prior ratio {float(row.get('recent_vs_prior_units', 0)):.2f}, "
            f"14-day acceleration {float(row.get('unit_demand_acceleration', 0)):+.0f}, and "
            f"{int(row.get('days_since_last_sale', 0))} days since last sale"
        )
    if group == "Category-relative demand":
        return (
            f"recent demand is {float(row.get('recent_units_vs_category_median', 0)):.2f}x "
            "its garment-group median"
        )
    if group == "Lifecycle and season":
        return (
            f"observed age {int(row.get('observed_style_age_days', 0))} days, "
            f"scored for {row.get('prediction_season', 'unknown season')}"
        )
    if group == "Price behavior":
        return f"recent average price {float(row.get('recent_avg_price', 0)):.4f}"
    return (
        f"{row.get('product_type_name', 'unknown category')}, "
        f"{row.get('garment_group_name', 'unknown group')}, "
        f"{row.get('colour_group_name', 'unknown colour')}"
    )


def _model_explanations(
    selected_model: str,
    regression_model: Pipeline,
    classifier_model: Pipeline,
    row: pd.Series,
    features: list[str],
    reference: dict[str, object],
    ranking_references: dict[str, np.ndarray],
) -> tuple[list[str], list[dict]]:
    observed = pd.DataFrame([{feature: row.get(feature) for feature in features}])
    base_score = _single_ranking_score(
        selected_model,
        regression_model,
        classifier_model,
        observed,
        ranking_references,
    )
    details = []
    for group, group_features in EXPLANATION_GROUPS.items():
        available = [feature for feature in group_features if feature in features]
        if not available:
            continue
        counterfactual = observed.copy()
        for feature in available:
            counterfactual.loc[0, feature] = reference[feature]
        counterfactual_score = _single_ranking_score(
            selected_model,
            regression_model,
            classifier_model,
            counterfactual,
            ranking_references,
        )
        delta = base_score - counterfactual_score
        details.append(
            {
                "group": group,
                "direction": "supports" if delta >= 0 else "tempers",
                "delta_ranking_score": round(float(delta), 5),
                "counterfactual_ranking_score": round(counterfactual_score, 5),
                "features": available,
                "observed_summary": _group_observation(group, row),
            }
        )

    details.sort(key=lambda item: abs(item["delta_ranking_score"]), reverse=True)
    selected = details[:3]
    explanations = []
    for item in selected:
        raw_delta = abs(item["delta_ranking_score"])
        verb = "are" if item["group"] == "Product attributes" else "is"
        effect = "lowers" if item["direction"] == "supports" else "raises"
        signal_label = "supporting" if item["direction"] == "supports" else "tempering"
        if selected_model in {"rolling_regression", "expanding_regression"}:
            delta_text = f"the predicted demand by about {raw_delta:,.0f} units"
        else:
            delta_text = f"the ranking score by about {raw_delta * 100:.1f} points"
        explanations.append(
            f"{item['group']} {verb} a {signal_label} local ranking signal: "
            f"{item['observed_summary']}. Replacing this group with typical development-set "
            f"values {effect} {delta_text}."
        )
    return explanations, selected


def _importance_rows(model: Pipeline) -> dict[str, float]:
    transformed = model.named_steps["preprocess"].get_feature_names_out()
    values = model.named_steps["model"].feature_importances_
    return {
        str(feature).split("__", 1)[-1]: float(importance)
        for feature, importance in zip(transformed, values)
    }


def _global_feature_importance(
    selected_model: str,
    regression_model: Pipeline,
    classifier_model: Pipeline,
) -> list[dict]:
    regression = _importance_rows(regression_model)
    classifier = _importance_rows(classifier_model)
    rows = []
    for feature in regression:
        if selected_model in {"rolling_regression", "expanding_regression"}:
            importance = regression[feature]
        elif selected_model == "top_decile_classifier":
            importance = classifier[feature]
        else:
            importance = (
                BLEND_REGRESSION_WEIGHT * regression[feature]
                + BLEND_CLASSIFIER_WEIGHT * classifier[feature]
            )
        rows.append({"feature": feature, "importance": round(float(importance), 8)})
    return sorted(rows, key=lambda item: item["importance"], reverse=True)


def _write_explanation_report(
    styles: list[dict], feature_importance: list[dict], method: str
) -> None:
    lines = [
        "# Model Explanation Report",
        "",
        "## Method",
        "",
        method,
        "",
        "Global importance uses impurity-based feature importance from the fitted Random Forest component(s). It is descriptive, can favor high-cardinality or correlated features, and is not causal.",
        "",
        "## Global Feature Importance",
        "",
        "| Rank | Feature | Importance |",
        "| ---: | --- | ---: |",
    ]
    for rank, item in enumerate(feature_importance[:15], start=1):
        lines.append(f"| {rank} | `{item['feature']}` | {item['importance']:.4f} |")
    lines.extend(["", "## Top-Style Local Sensitivity", ""])
    for style in styles:
        lines.extend(
            [
                f"### #{style['rank']} {style['name']} (`{style['style_id']}`)",
                "",
                f"Ranking score: {style['prediction_score']:.4f}; regression estimate: {style['predicted_units']:,.2f} units",
                "",
            ]
        )
        lines.extend(f"- {reason}" for reason in style["model_explanation"])
        lines.append("")
    CONFIG.explanation_path.write_text("\n".join(lines), encoding="utf-8")


def article_image_url(article_id: int | float | str) -> str | None:
    try:
        article = f"{int(article_id):010d}"
    except (TypeError, ValueError):
        return None
    folder = article[:3]
    candidate = CONFIG.images_dir / folder / f"{article}.jpg"
    if candidate.exists():
        return f"/images/{folder}/{article}.jpg"
    return None


def train_and_predict() -> dict:
    ensure_project_dirs()
    df = _load_dataset()
    development = df[df["snapshot_role"] == "backtest"].copy()
    final = df[df["snapshot_role"] == "final_holdout"].copy()
    if development.empty or final.empty:
        raise ValueError("Expected backtest snapshots and a final_holdout snapshot.")

    numeric, categorical = _available_features(df)
    features = numeric + categorical
    fold_results = []
    validation_orders = sorted(development["snapshot_order"].unique())[
        CONFIG.min_backtest_train_periods :
    ]
    for order in validation_orders:
        expanding_training = development[development["snapshot_order"] < order].copy()
        rolling_training = expanding_training[
            expanding_training["snapshot_order"]
            >= order - CONFIG.rolling_train_periods
        ].copy()
        validation = development[development["snapshot_order"] == order].copy()
        rolling_regression_model = build_regression_model(numeric, categorical)
        expanding_regression_model = build_regression_model(numeric, categorical)
        classifier_model = build_classifier_model(numeric, categorical)
        rolling_regression_model.fit(
            rolling_training[features], np.log1p(rolling_training["target_units"])
        )
        expanding_regression_model.fit(
            expanding_training[features], np.log1p(expanding_training["target_units"])
        )
        classifier_model.fit(
            rolling_training[features], _top_decile_labels(rolling_training)
        )
        rolling_scores, rolling_units, _, _ = _candidate_scores(
            rolling_regression_model,
            classifier_model,
            rolling_training[features],
            validation[features],
        )
        expanding_train_units = _predicted_units(
            expanding_regression_model, expanding_training[features]
        )
        expanding_units = _predicted_units(
            expanding_regression_model, validation[features]
        )
        scores = {
            "rolling_regression": rolling_scores["regression"],
            "expanding_regression": expanding_units,
            "top_decile_classifier": rolling_scores["top_decile_classifier"],
            "blended_ranker": rolling_scores["blended_ranker"],
        }
        result = _fold_metrics(
            validation,
            scores,
            {
                "rolling_regression": rolling_units,
                "expanding_regression": expanding_units,
            },
        )
        result.update(
            {
                "fold": str(validation["snapshot_id"].iloc[0]),
                "rolling_training_snapshots": int(
                    rolling_training["snapshot_id"].nunique()
                ),
                "expanding_training_snapshots": int(
                    expanding_training["snapshot_id"].nunique()
                ),
                "target_start": str(validation["target_start"].iloc[0]),
                "target_end": str(validation["target_end"].iloc[0]),
                "styles_scored": int(len(validation)),
            }
        )
        fold_results.append(result)

    if not fold_results:
        raise ValueError("No rolling validation folds were created.")
    aggregate, baseline_aggregate = _aggregate_backtests(fold_results)
    selected_model = max(aggregate, key=lambda name: aggregate[name]["selection_score"])

    final_training_order = int(development["snapshot_order"].max())
    rolling_development = development[
        development["snapshot_order"]
        > final_training_order - CONFIG.rolling_train_periods
    ].copy()
    final_regression_model = build_regression_model(numeric, categorical)
    expanding_regression_model = build_regression_model(numeric, categorical)
    final_classifier_model = build_classifier_model(numeric, categorical)
    final_regression_model.fit(
        rolling_development[features], np.log1p(rolling_development["target_units"])
    )
    expanding_regression_model.fit(
        development[features], np.log1p(development["target_units"])
    )
    final_classifier_model.fit(
        rolling_development[features], _top_decile_labels(rolling_development)
    )
    rolling_final_scores, final_predicted_units, final_probabilities, ranking_references = (
        _candidate_scores(
            final_regression_model,
            final_classifier_model,
            rolling_development[features],
            final[features],
        )
    )
    expanding_development_units = _predicted_units(
        expanding_regression_model, development[features]
    )
    expanding_final_units = _predicted_units(expanding_regression_model, final[features])
    final_scores = {
        "rolling_regression": rolling_final_scores["regression"],
        "expanding_regression": expanding_final_units,
        "top_decile_classifier": rolling_final_scores["top_decile_classifier"],
        "blended_ranker": rolling_final_scores["blended_ranker"],
    }
    explanation_regression_model = (
        expanding_regression_model
        if selected_model == "expanding_regression"
        else final_regression_model
    )
    selected_predicted_units = (
        expanding_final_units
        if selected_model == "expanding_regression"
        else final_predicted_units
    )
    if selected_model == "expanding_regression":
        ranking_references = {
            **ranking_references,
            "regression_units": expanding_development_units,
        }
    final["predicted_units"] = selected_predicted_units
    final["top_decile_probability"] = final_probabilities
    final["prediction_score"] = final_scores[selected_model]
    score_max = final["prediction_score"].max()
    final["prediction_score_normalized"] = (
        final["prediction_score"] / score_max if score_max > 0 else 0.0
    )
    final["baseline_score"] = final["recent_units"].clip(lower=0)
    final_holdout = _fold_metrics(
        final,
        final_scores,
        {
            "rolling_regression": final_predicted_units,
            "expanding_regression": expanding_final_units,
        },
    )
    selected_holdout = final_holdout["candidates"][selected_model]
    if "mae" not in selected_holdout:
        selected_holdout["mae"] = final_holdout["candidates"][
            "rolling_regression"
        ]["mae"]
        selected_holdout["rmse"] = final_holdout["candidates"][
            "rolling_regression"
        ]["rmse"]
    final["rank"] = final["prediction_score"].rank(
        method="first", ascending=False
    ).astype(int)
    final = final.sort_values("rank")
    final["image_url"] = final["representative_article_id"].apply(article_image_url)

    explanation_method = (
        f"Grouped counterfactual sensitivity for the selected {MODEL_LABELS[selected_model]}. "
        "One feature group at a time is replaced with development-set medians/modes and the "
        "locked ranking score is recalculated. Deltas are diagnostic, non-additive, and not causal."
    )
    metrics = {
        "target": "future style unit sales over the next 28-day window",
        "style_definition": "product_code grouped as stable style_id",
        "candidate_model_name": MODEL_LABELS[selected_model],
        "selected_model_key": selected_model,
        "selection_protocol": (
            "Model selected only from expanding-window rolling backtests before the final holdout. "
            "Selection score = mean P@3 + 0.5*mean P@10 - 0.25*SD(P@3) - 0.10*SD(P@10)."
        ),
        "rolling_backtest": {
            "fold_count": len(fold_results),
            "folds": fold_results,
            "candidate_summary": aggregate,
            "baseline_recent_sales": baseline_aggregate,
        },
        "candidate_model": {
            "mean_precision_at_3": aggregate[selected_model]["mean_precision_at_3"],
            "std_precision_at_3": aggregate[selected_model]["std_precision_at_3"],
            "mean_precision_at_10": aggregate[selected_model]["mean_precision_at_10"],
            "std_precision_at_10": aggregate[selected_model]["std_precision_at_10"],
            "ranking_stability": aggregate[selected_model]["ranking_stability"],
            "regression_backbone_mean_mae": aggregate[
                "rolling_regression"
            ]["mean_mae"],
            "regression_backbone_mean_rmse": aggregate[
                "rolling_regression"
            ]["mean_rmse"],
        },
        "baseline_recent_sales": baseline_aggregate,
        "final_holdout": {
            "period": {
                "feature_start": str(final["feature_start"].iloc[0]),
                "feature_end": str(final["feature_end"].iloc[0]),
                "target_start": str(final["target_start"].iloc[0]),
                "target_end": str(final["target_end"].iloc[0]),
            },
            "candidate_model": selected_holdout,
            "baseline_recent_sales": final_holdout["baseline_recent_sales"],
            "note": (
                "The final holdout was evaluated once after model selection and was not used to "
                "choose features, objective, blend weights, or hyperparameters. MAE/RMSE refer to "
                "the regression unit forecast even when the selected ranking objective differs."
            ),
        },
        "limitations": [
            "No reliable stock-on-hand or availability history is present, so sales may mix customer demand with stock constraints.",
            "Price exists at transaction level but explicit markdown, campaign, and promotion metadata is unavailable.",
            "Observed launch age starts at the first sale in this dataset, not necessarily the true product launch date.",
            "Only one untouched 28-day final holdout exists; rolling backtests reduce but do not eliminate temporal uncertainty.",
            "Concept image generation depends on catalogue-image availability for selected styles.",
        ],
        "explanation_method": explanation_method,
    }

    keep_cols = [
        "rank",
        "style_id",
        "representative_article_id",
        "prod_name",
        "product_type_name",
        "product_group_name",
        "graphical_appearance_name",
        "colour_group_name",
        "department_name",
        "section_name",
        "garment_group_name",
        "prediction_score",
        "prediction_score_normalized",
        "predicted_units",
        "top_decile_probability",
        "target_units",
        "hist_units",
        "recent_units",
        "prior_units",
        "last14_units",
        "prev14_units",
        "unit_demand_acceleration",
        "recent_units_vs_category_median",
        "momentum_vs_category_median",
        "observed_style_age_days",
        "prediction_month",
        "prediction_season",
        "hist_customers",
        "hist_active_days",
        "recent_vs_prior_units",
        "days_since_last_sale",
        "feature_start",
        "feature_end",
        "target_start",
        "target_end",
        "image_url",
        "detail_desc",
    ]
    final[[c for c in keep_cols if c in final.columns]].to_csv(
        CONFIG.predictions_path, index=False
    )

    top = final.head(3).copy()
    reference = _reference_values(development, numeric, categorical)
    feature_importance = _global_feature_importance(
        selected_model, explanation_regression_model, final_classifier_model
    )
    CONFIG.feature_importance_path.write_text(
        json.dumps(
            {
                "selected_model": MODEL_LABELS[selected_model],
                "method": "Impurity-based importance for the selected Random Forest component(s)",
                "caveat": (
                    "Importances are descriptive, can be shared across correlated features, "
                    "can favor high-cardinality features, and are not causal."
                ),
                "features": feature_importance,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    styles = []
    for _, row in top.iterrows():
        model_explanation, explanation_details = _model_explanations(
            selected_model,
            explanation_regression_model,
            final_classifier_model,
            row,
            features,
            reference,
            ranking_references,
        )
        styles.append(
            {
                "rank": int(row["rank"]),
                "style_id": str(row["style_id"]),
                "representative_article_id": int(row["representative_article_id"]),
                "name": str(row.get("prod_name", "")),
                "prediction_score": round(float(row["prediction_score_normalized"]), 4),
                "predicted_units": round(float(row["predicted_units"]), 2),
                "top_decile_probability": round(
                    float(row["top_decile_probability"]), 4
                ),
                "ranking_model": MODEL_LABELS[selected_model],
                "category": str(row.get("product_type_name", "")),
                "colour": str(row.get("colour_group_name", "")),
                "garment_group": str(row.get("garment_group_name", "")),
                "sales_history": int(row.get("hist_units", 0)),
                "recent_sales": int(row.get("recent_units", 0)),
                "prior_sales": int(row.get("prior_units", 0)),
                "actual_holdout_sales": int(row.get("target_units", 0)),
                "image_url": row.get("image_url")
                if pd.notna(row.get("image_url"))
                else None,
                "model_explanation": model_explanation,
                "model_explanation_details": explanation_details,
                "explanation_method": explanation_method,
                "attributes": {
                    "product_group": str(row.get("product_group_name", "")),
                    "graphical_appearance": str(
                        row.get("graphical_appearance_name", "")
                    ),
                    "department": str(row.get("department_name", "")),
                    "section": str(row.get("section_name", "")),
                },
                "historical_performance": {
                    "feature_window": [
                        str(row.get("feature_start")),
                        str(row.get("feature_end")),
                    ],
                    "prediction_window": [
                        str(row.get("target_start")),
                        str(row.get("target_end")),
                    ],
                    "hist_units": int(row.get("hist_units", 0)),
                    "hist_customers": int(row.get("hist_customers", 0)),
                    "active_days": int(row.get("hist_active_days", 0)),
                    "recent_vs_prior_units": float(
                        row.get("recent_vs_prior_units", 0)
                    ),
                    "unit_demand_acceleration": float(
                        row.get("unit_demand_acceleration", 0)
                    ),
                    "category_relative_recent_demand": float(
                        row.get("recent_units_vs_category_median", 0)
                    ),
                    "observed_style_age_days": int(
                        row.get("observed_style_age_days", 0)
                    ),
                    "prediction_season": str(row.get("prediction_season", "")),
                },
                "description": str(row.get("detail_desc", "")),
            }
        )

    top_payload = {
        "styles": styles,
        "metrics": metrics,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "data_note": (
            "Rankings use historical transactions only. The final holdout is reported honestly "
            "but was not used for model selection; stock constraints remain unknown."
        ),
    }
    CONFIG.top_styles_path.write_text(json.dumps(top_payload, indent=2), encoding="utf-8")
    CONFIG.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_explanation_report(styles, feature_importance, explanation_method)
    model_artifact = {
        "selected_model": selected_model,
        "features": features,
        "metrics": metrics,
    }
    if selected_model in {"rolling_regression", "expanding_regression"}:
        model_artifact["model"] = explanation_regression_model
    elif selected_model == "top_decile_classifier":
        model_artifact["model"] = final_classifier_model
    else:
        model_artifact["regression_model"] = final_regression_model
        model_artifact["classifier_model"] = final_classifier_model
        model_artifact["ranking_references"] = ranking_references
    joblib.dump(model_artifact, CONFIG.model_path, compress=3)
    return top_payload


def main() -> None:
    payload = train_and_predict()
    print(f"Wrote {CONFIG.top_styles_path} with {len(payload['styles'])} top styles.")
    print(f"Wrote metrics to {CONFIG.metrics_path}.")


if __name__ == "__main__":
    main()
