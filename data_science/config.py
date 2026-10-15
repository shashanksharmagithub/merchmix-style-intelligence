from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
DEPLOYMENT_IMAGES_DIR = PROJECT_ROOT / "assets" / "images"


@dataclass(frozen=True)
class PipelineConfig:
    """Central knobs for a practical, time-aware assignment run."""

    transactions_path: Path = RAW_DIR / "transactions_train.csv"
    articles_path: Path = RAW_DIR / "articles.csv"
    images_dir: Path = RAW_DIR / "images"
    deployment_images_dir: Path = DEPLOYMENT_IMAGES_DIR
    processed_dataset_path: Path = PROCESSED_DIR / "style_training_dataset.csv"
    model_path: Path = MODELS_DIR / "style_winner_model.joblib"
    metrics_path: Path = OUTPUTS_DIR / "model_metrics.json"
    feature_importance_path: Path = OUTPUTS_DIR / "model_feature_importance.json"
    predictions_path: Path = OUTPUTS_DIR / "style_predictions.csv"
    top_styles_path: Path = OUTPUTS_DIR / "top_styles.json"
    explanation_path: Path = OUTPUTS_DIR / "style_explanations.md"
    concept_brief_path: Path = OUTPUTS_DIR / "concept_briefs.json"
    concept_image_path: Path = OUTPUTS_DIR / "generated_concepts.png"
    fallback_concept_image_path: Path = OUTPUTS_DIR / "generated_concepts_fallback.png"
    seasonal_styles_path: Path = OUTPUTS_DIR / "seasonal_styles.json"
    prediction_period_days: int = 28
    validation_period_days: int = 28
    lookback_days: int = 84
    backtest_periods: int = 8
    min_backtest_train_periods: int = 2
    rolling_train_periods: int = 2
    top_decile_quantile: float = 0.90
    min_history_units: int = 5
    max_transactions: int | None = None


CONFIG = PipelineConfig()


def ensure_project_dirs() -> None:
    for directory in [RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
