from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_science.config import CONFIG, PROCESSED_DIR, ensure_project_dirs


ARTICLE_ATTRIBUTES = [
    "prod_name",
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
    "detail_desc",
]

SEASON_BY_MONTH = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Autumn",
    10: "Autumn",
    11: "Autumn",
}


def _csv_data_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def _read_transactions(max_transactions: int | None = None) -> pd.DataFrame:
    read_kwargs = {
        "parse_dates": ["t_dat"],
        "dtype": {"article_id": "int64", "customer_id": "string"},
    }
    if max_transactions is not None:
        total_rows = _csv_data_rows(CONFIG.transactions_path)
        skip_until = max(1, total_rows - max_transactions + 1)
        read_kwargs["skiprows"] = range(1, skip_until)
    return pd.read_csv(CONFIG.transactions_path, **read_kwargs)


def _read_inputs(max_transactions: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not CONFIG.transactions_path.exists() or not CONFIG.articles_path.exists():
        raise FileNotFoundError(
            "Missing H&M data. Expected transactions_train.csv and articles.csv in data/raw/."
        )
    tx = _read_transactions(max_transactions)
    articles = pd.read_csv(CONFIG.articles_path, dtype={"article_id": "int64"})
    return tx, articles


def add_style_id(articles: pd.DataFrame) -> pd.DataFrame:
    """Use product_code as a stable style family and expose it as style_id."""
    articles = articles.copy()
    if "product_code" not in articles:
        articles["product_code"] = articles["article_id"].astype(str).str[:7].astype(int)
    articles["style_id"] = articles["product_code"].astype(str)
    return articles


def _article_has_image(article_id: int | float | str) -> bool:
    try:
        article = f"{int(article_id):010d}"
    except (TypeError, ValueError):
        return False
    return (CONFIG.images_dir / article[:3] / f"{article}.jpg").exists()


def _window_aggregate(tx: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, prefix: str) -> pd.DataFrame:
    columns = [
        "style_id",
        f"{prefix}_units",
        f"{prefix}_customers",
        f"{prefix}_avg_price",
        f"{prefix}_max_price",
        f"{prefix}_active_days",
        f"{prefix}_last_sale",
    ]
    window = tx[(tx["t_dat"] >= start) & (tx["t_dat"] < end)]
    if window.empty:
        return pd.DataFrame(columns=columns)
    agg = (
        window.groupby("style_id")
        .agg(
            **{
                f"{prefix}_units": ("article_id", "size"),
                f"{prefix}_customers": ("customer_id", "nunique"),
                f"{prefix}_avg_price": ("price", "mean"),
                f"{prefix}_max_price": ("price", "max"),
                f"{prefix}_active_days": ("t_dat", "nunique"),
                f"{prefix}_last_sale": ("t_dat", "max"),
            }
        )
        .reset_index()
    )
    return agg


def _style_article_attributes(articles: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in ARTICLE_ATTRIBUTES if c in articles.columns]
    rows = []
    for style_id, group in articles.groupby("style_id", sort=False):
        row = {"style_id": style_id, "article_count": group["article_id"].nunique()}
        for column in available:
            mode = group[column].dropna().mode()
            row[column] = mode.iloc[0] if not mode.empty else ""

        representative = group.iloc[0]
        image_group = group[group["article_id"].apply(_article_has_image)]
        if not image_group.empty:
            colour = row.get("colour_group_name")
            colour_match = image_group[image_group["colour_group_name"] == colour] if "colour_group_name" in image_group else image_group
            representative = (colour_match if not colour_match.empty else image_group).iloc[0]
        row["representative_article_id"] = int(representative["article_id"])
        rows.append(row)
    return pd.DataFrame(rows)


def build_training_dataset(max_transactions: int | None = None) -> pd.DataFrame:
    ensure_project_dirs()
    tx, articles = _read_inputs(max_transactions)
    articles = add_style_id(articles)
    tx = tx.merge(articles[["article_id", "style_id"]], on="article_id", how="inner")

    max_date = tx["t_dat"].max().normalize()
    min_date = tx["t_dat"].min().normalize()
    required_days = CONFIG.lookback_days + (
        CONFIG.validation_period_days * (CONFIG.backtest_periods + 1)
    )
    observed_days = (max_date - min_date).days + 1
    if observed_days < required_days:
        raise ValueError(
            "Not enough date coverage for leakage-free train/validation/final windows. "
            f"Observed {observed_days} days from {min_date.date()} to {max_date.date()}, "
            f"but at least {required_days} days are needed. "
            "Use the full data or increase --max-transactions; for this dataset, "
            "--max-transactions 8000000 is a practical smoke-test minimum."
        )

    prediction_start = (
        max_date - pd.Timedelta(days=CONFIG.prediction_period_days) + pd.Timedelta(days=1)
    )
    snapshots = []
    for order, steps_back in enumerate(range(CONFIG.backtest_periods, 0, -1)):
        target_start = prediction_start - pd.Timedelta(
            days=steps_back * CONFIG.validation_period_days
        )
        snapshots.append(
            {
                "snapshot_id": f"backtest_{order + 1:02d}",
                "snapshot_order": order,
                "snapshot_role": "backtest",
                "feature_start": target_start - pd.Timedelta(days=CONFIG.lookback_days),
                "feature_end": target_start,
                "target_start": target_start,
                "target_end": target_start
                + pd.Timedelta(days=CONFIG.validation_period_days),
            }
        )
    snapshots.append(
        {
            "snapshot_id": "final_holdout",
            "snapshot_order": CONFIG.backtest_periods,
            "snapshot_role": "final_holdout",
            "feature_start": prediction_start - pd.Timedelta(days=CONFIG.lookback_days),
            "feature_end": prediction_start,
            "target_start": prediction_start,
            "target_end": max_date + pd.Timedelta(days=1),
        }
    )

    attrs = _style_article_attributes(articles)
    first_sale = (
        tx.groupby("style_id", as_index=False)["t_dat"]
        .min()
        .rename(columns={"t_dat": "observed_first_sale"})
    )
    attrs = attrs.merge(first_sale, on="style_id", how="left")
    all_rows = []
    for snapshot in snapshots:
        feature_start = snapshot["feature_start"]
        feature_end = snapshot["feature_end"]
        target_start = snapshot["target_start"]
        target_end = snapshot["target_end"]
        hist = _window_aggregate(tx, feature_start, feature_end, "hist")
        recent = _window_aggregate(tx, feature_end - pd.Timedelta(days=28), feature_end, "recent")
        prior = _window_aggregate(tx, feature_end - pd.Timedelta(days=56), feature_end - pd.Timedelta(days=28), "prior")
        last14 = _window_aggregate(tx, feature_end - pd.Timedelta(days=14), feature_end, "last14")
        prev14 = _window_aggregate(
            tx,
            feature_end - pd.Timedelta(days=28),
            feature_end - pd.Timedelta(days=14),
            "prev14",
        )
        target = _window_aggregate(tx, target_start, target_end, "target")

        df = attrs.merge(hist, on="style_id", how="left")
        df = df.merge(recent, on="style_id", how="left")
        df = df.merge(prior, on="style_id", how="left")
        df = df.merge(last14, on="style_id", how="left")
        df = df.merge(prev14, on="style_id", how="left")
        df = df.merge(target[["style_id", "target_units"]], on="style_id", how="left")
        df["snapshot_id"] = snapshot["snapshot_id"]
        df["snapshot_order"] = snapshot["snapshot_order"]
        df["snapshot_role"] = snapshot["snapshot_role"]
        df["split"] = snapshot["snapshot_role"]
        df["feature_start"] = feature_start.date().isoformat()
        df["feature_end"] = feature_end.date().isoformat()
        df["target_start"] = target_start.date().isoformat()
        df["target_end"] = (target_end - pd.Timedelta(days=1)).date().isoformat()
        all_rows.append(df)

    dataset = pd.concat(all_rows, ignore_index=True)
    count_cols = [c for c in dataset.columns if c.endswith(("units", "customers", "active_days"))]
    dataset[count_cols] = dataset[count_cols].fillna(0)
    dataset["target_units"] = dataset["target_units"].fillna(0)
    dataset["hist_units_per_active_day"] = np.where(
        dataset["hist_active_days"] > 0,
        dataset["hist_units"] / dataset["hist_active_days"],
        0,
    )
    dataset["recent_vs_prior_units"] = np.where(
        dataset["prior_units"] > 0,
        dataset["recent_units"] / dataset["prior_units"],
        dataset["recent_units"],
    )
    dataset["days_since_last_sale"] = (
        pd.to_datetime(dataset["feature_end"]) - pd.to_datetime(dataset["hist_last_sale"])
    ).dt.days.fillna(CONFIG.lookback_days + CONFIG.validation_period_days)
    dataset["observed_style_age_days"] = (
        pd.to_datetime(dataset["feature_end"])
        - pd.to_datetime(dataset["observed_first_sale"])
    ).dt.days.clip(lower=0)
    dataset["last14_vs_prev14_units"] = np.where(
        dataset["prev14_units"] > 0,
        dataset["last14_units"] / dataset["prev14_units"],
        dataset["last14_units"],
    )
    dataset["unit_demand_acceleration"] = dataset["last14_units"] - dataset["prev14_units"]
    dataset["customer_demand_acceleration"] = (
        dataset["last14_customers"] - dataset["prev14_customers"]
    )

    prediction_month = pd.to_datetime(dataset["target_start"]).dt.month
    dataset["prediction_month"] = prediction_month
    dataset["prediction_month_sin"] = np.sin(2 * np.pi * prediction_month / 12)
    dataset["prediction_month_cos"] = np.cos(2 * np.pi * prediction_month / 12)
    dataset["prediction_season"] = prediction_month.map(SEASON_BY_MONTH)

    dataset = dataset[dataset["hist_units"] >= CONFIG.min_history_units].copy()
    group_keys = ["snapshot_id", "garment_group_name"]
    category_recent_median = dataset.groupby(group_keys)["recent_units"].transform("median")
    category_momentum_median = dataset.groupby(group_keys)["recent_vs_prior_units"].transform("median")
    category_acceleration_median = dataset.groupby(group_keys)[
        "unit_demand_acceleration"
    ].transform("median")
    category_recent_total = dataset.groupby(group_keys)["recent_units"].transform("sum")
    dataset["recent_units_vs_category_median"] = dataset["recent_units"] / (
        category_recent_median + 1.0
    )
    dataset["momentum_vs_category_median"] = (
        dataset["recent_vs_prior_units"] - category_momentum_median
    )
    dataset["acceleration_vs_category_median"] = (
        dataset["unit_demand_acceleration"] - category_acceleration_median
    )
    dataset["recent_category_share"] = np.where(
        category_recent_total > 0,
        dataset["recent_units"] / category_recent_total,
        0,
    )
    dataset.to_csv(CONFIG.processed_dataset_path, index=False)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-transactions", type=int, default=CONFIG.max_transactions)
    args = parser.parse_args()
    dataset = build_training_dataset(args.max_transactions)
    print(f"Wrote {CONFIG.processed_dataset_path} with {len(dataset):,} rows.")


if __name__ == "__main__":
    main()
