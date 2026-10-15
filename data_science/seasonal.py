from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_science.config import CONFIG, ensure_project_dirs
from data_science.features.build_features import _style_article_attributes, add_style_id
from data_science.train import article_image_url


SEASON_BY_MONTH = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}

SEASON_LABELS = {
    "spring": "Spring",
    "summer": "Summer",
    "autumn": "Autumn",
    "winter": "Winter",
}


def _concept_direction(row: pd.Series, season_label: str) -> str:
    category = row.get("product_type_name") or "garment"
    colour = row.get("colour_group_name") or "core colour"
    group = row.get("garment_group_name") or row.get("product_group_name") or "fashion"
    return (
        f"For {season_label.lower()}, retain the proven {colour.lower()} {category.lower()} cue "
        f"and {group.lower()} context, then update the styling, finishings, and proportion for a fresh seasonal range."
    )


def build_seasonal_styles(top_n: int = 10) -> dict:
    ensure_project_dirs()
    if not CONFIG.transactions_path.exists() or not CONFIG.articles_path.exists():
        raise FileNotFoundError("Missing raw H&M CSV files in data/raw/.")

    tx = pd.read_csv(
        CONFIG.transactions_path,
        usecols=["t_dat", "article_id", "customer_id", "price"],
        parse_dates=["t_dat"],
        dtype={"article_id": "int64", "customer_id": "string"},
    )
    articles = add_style_id(pd.read_csv(CONFIG.articles_path, dtype={"article_id": "int64"}))
    attrs = _style_article_attributes(articles)

    tx = tx.merge(articles[["article_id", "style_id"]], on="article_id", how="inner")
    tx["season"] = tx["t_dat"].dt.month.map(SEASON_BY_MONTH)

    seasonal = (
        tx.groupby(["season", "style_id"], observed=True)
        .agg(
            season_units=("article_id", "size"),
            season_customers=("customer_id", "nunique"),
            active_days=("t_dat", "nunique"),
            avg_price=("price", "mean"),
            first_sale=("t_dat", "min"),
            last_sale=("t_dat", "max"),
        )
        .reset_index()
        .merge(attrs, on="style_id", how="left")
    )

    payload = {
        "season_definitions": {
            "spring": "March-May",
            "summer": "June-August",
            "autumn": "September-November",
            "winter": "December-February",
        },
        "seasons": [],
    }

    for season in ["spring", "summer", "autumn", "winter"]:
        season_df = seasonal[seasonal["season"] == season].copy()
        season_df = season_df.sort_values(["season_units", "season_customers"], ascending=False).head(top_n)
        total_units = int(seasonal.loc[seasonal["season"] == season, "season_units"].sum())
        styles = []
        for rank, (_, row) in enumerate(season_df.iterrows(), start=1):
            image_url = article_image_url(row.get("representative_article_id"))
            season_label = SEASON_LABELS[season]
            styles.append(
                {
                    "rank": rank,
                    "season": season,
                    "season_label": season_label,
                    "style_id": str(row["style_id"]),
                    "representative_article_id": int(row["representative_article_id"]),
                    "name": str(row.get("prod_name", "")),
                    "category": str(row.get("product_type_name", "")),
                    "colour": str(row.get("colour_group_name", "")),
                    "garment_group": str(row.get("garment_group_name", "")),
                    "season_units": int(row.get("season_units", 0)),
                    "season_customers": int(row.get("season_customers", 0)),
                    "active_days": int(row.get("active_days", 0)),
                    "avg_price": round(float(row.get("avg_price", 0)), 5),
                    "season_sales_share": round(float(row.get("season_units", 0)) / total_units, 5)
                    if total_units
                    else 0,
                    "first_sale": str(pd.Timestamp(row["first_sale"]).date()),
                    "last_sale": str(pd.Timestamp(row["last_sale"]).date()),
                    "image_url": image_url,
                    "attributes": {
                        "product_group": str(row.get("product_group_name", "")),
                        "graphical_appearance": str(row.get("graphical_appearance_name", "")),
                        "department": str(row.get("department_name", "")),
                        "section": str(row.get("section_name", "")),
                    },
                    "description": str(row.get("detail_desc", "")),
                    "seasonal_concept_direction": _concept_direction(row, season_label),
                }
            )
        payload["seasons"].append(
            {
                "season": season,
                "label": SEASON_LABELS[season],
                "definition": payload["season_definitions"][season],
                "total_units": total_units,
                "styles": styles,
            }
        )

    CONFIG.seasonal_styles_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    payload = build_seasonal_styles()
    print(f"Wrote {CONFIG.seasonal_styles_path} with {len(payload['seasons'])} seasons.")


if __name__ == "__main__":
    main()
