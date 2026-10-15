from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from data_science.config import CONFIG, PROJECT_ROOT


class PredictionDataUnavailable(RuntimeError):
    pass


class SeasonalDataUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_payload() -> dict:
    if not CONFIG.top_styles_path.exists():
        raise PredictionDataUnavailable(
            "Prediction output is missing. Run the data science pipeline first."
        )
    return json.loads(CONFIG.top_styles_path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_seasonal_payload() -> dict:
    if not CONFIG.seasonal_styles_path.exists():
        raise SeasonalDataUnavailable("Seasonal output is missing. Run the data science pipeline first.")
    return json.loads(CONFIG.seasonal_styles_path.read_text(encoding="utf-8"))


def refresh_cache() -> None:
    load_payload.cache_clear()
    load_seasonal_payload.cache_clear()


def get_top_styles() -> dict:
    payload = load_payload()
    return {
        "styles": payload.get("styles", []),
        "metrics": payload.get("metrics", {}),
        "data_note": payload.get("data_note", ""),
        "generated_at": payload.get("generated_at"),
    }


def get_style(style_id: str) -> dict | None:
    payload = load_payload()
    for style in payload.get("styles", []):
        if str(style.get("style_id")) == str(style_id):
            return style
    return None


def get_seasons() -> dict:
    return load_seasonal_payload()


def get_seasonal_styles(season: str) -> dict | None:
    payload = load_seasonal_payload()
    season = season.lower()
    for item in payload.get("seasons", []):
        if item.get("season") == season:
            return item
    return None


def get_project_status() -> dict:
    return {
        "has_predictions": CONFIG.top_styles_path.exists(),
        "has_model": CONFIG.model_path.exists(),
        "has_concepts": CONFIG.concept_image_path.exists(),
        "has_seasonal_view": CONFIG.seasonal_styles_path.exists(),
        "expected_data_files": [
            str(CONFIG.transactions_path.relative_to(PROJECT_ROOT)),
            str(CONFIG.articles_path.relative_to(PROJECT_ROOT)),
            str(CONFIG.images_dir.relative_to(PROJECT_ROOT)),
        ],
    }
