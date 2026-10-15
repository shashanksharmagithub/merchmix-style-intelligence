from __future__ import annotations

import json
import os
from pathlib import Path

from data_science.config import PROJECT_ROOT

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_science.config import CONFIG, OUTPUTS_DIR, ensure_project_dirs
from data_science.features.build_features import _read_transactions, add_style_id


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
SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]
MONTH_ORDER = list(range(1, 13))
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _save_plots(
    daily: pd.DataFrame,
    category: pd.DataFrame,
    customer_purchases: pd.Series,
    calendar_month: pd.DataFrame,
    seasonal_mix: pd.DataFrame,
) -> list[Path]:
    plots_dir = OUTPUTS_DIR / "eda_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 5))
    sns.lineplot(data=daily, x="t_dat", y="transactions", color="#2f6f73", linewidth=1.2)
    plt.title("Daily Transaction Trend")
    plt.xlabel("Date")
    plt.ylabel("Transactions")
    plt.tight_layout()
    trend_path = plots_dir / "daily_transactions.png"
    plt.savefig(trend_path, dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=category, y="garment_group_name", x="transactions", color="#40798c")
    plt.title("Top Garment Groups by Transactions")
    plt.xlabel("Transactions")
    plt.ylabel("Garment group")
    plt.tight_layout()
    category_path = plots_dir / "garment_group_transactions.png"
    plt.savefig(category_path, dpi=160)
    plt.close()

    customer_cap = max(1, int(customer_purchases.quantile(0.99)))
    plt.figure(figsize=(10, 5))
    sns.histplot(customer_purchases.clip(upper=customer_cap), bins=40, color="#d0744c")
    plt.title("Customer Purchase Frequency (capped at 99th percentile)")
    plt.xlabel("Transaction rows per customer")
    plt.ylabel("Customers")
    plt.tight_layout()
    customer_path = plots_dir / "customer_purchase_frequency.png"
    plt.savefig(customer_path, dpi=160)
    plt.close()

    calendar_month = calendar_month.copy()
    calendar_month["month_label"] = calendar_month["calendar_month"].map(dict(zip(MONTH_ORDER, MONTH_LABELS)))
    plt.figure(figsize=(11, 5))
    sns.barplot(
        data=calendar_month,
        x="month_label",
        y="avg_daily_transactions",
        hue="month_label",
        palette="crest",
        legend=False,
    )
    plt.title("Average Daily Transactions by Calendar Month")
    plt.xlabel("Calendar month")
    plt.ylabel("Average daily transactions")
    plt.tight_layout()
    month_path = plots_dir / "calendar_month_seasonality.png"
    plt.savefig(month_path, dpi=160)
    plt.close()

    plt.figure(figsize=(12, 5))
    sns.heatmap(
        seasonal_mix,
        annot=True,
        fmt=".1%",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Share within season"},
    )
    plt.title("Top Garment Group Mix by Season")
    plt.xlabel("Garment group")
    plt.ylabel("Season")
    plt.tight_layout()
    seasonal_path = plots_dir / "seasonal_garment_mix.png"
    plt.savefig(seasonal_path, dpi=160)
    plt.close()

    return [trend_path, category_path, customer_path, month_path, seasonal_path]


def run_eda(max_transactions: int | None = None) -> Path:
    ensure_project_dirs()
    if not CONFIG.transactions_path.exists() or not CONFIG.articles_path.exists():
        raise FileNotFoundError("Missing raw H&M CSV files in data/raw/.")

    (PROJECT_ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
    tx = _read_transactions(max_transactions)
    articles = add_style_id(pd.read_csv(CONFIG.articles_path, dtype={"article_id": "int64"}))
    article_columns = [
        "article_id",
        "style_id",
        "prod_name",
        "product_type_name",
        "garment_group_name",
        "colour_group_name",
    ]
    merged = tx.merge(articles[article_columns], on="article_id", how="left")

    daily = merged.groupby("t_dat").size().rename("transactions").reset_index()
    daily["calendar_month"] = daily["t_dat"].dt.month
    daily["season"] = daily["calendar_month"].map(SEASON_BY_MONTH)
    calendar_month = (
        daily.groupby("calendar_month", observed=True)["transactions"]
        .mean()
        .reindex(MONTH_ORDER)
        .rename("avg_daily_transactions")
        .reset_index()
    )
    seasonal_daily = (
        daily.groupby("season", observed=True)["transactions"]
        .mean()
        .reindex(SEASON_ORDER)
        .rename("avg_daily_transactions")
    )

    category_all = merged.groupby("garment_group_name", observed=True).size().sort_values(ascending=False)
    category = category_all.head(12).rename("transactions").reset_index()
    style_sales = merged.groupby("style_id", observed=True).size().rename("units")
    customer_purchases = tx.groupby("customer_id", observed=True).size().rename("transactions")

    merged["season"] = merged["t_dat"].dt.month.map(SEASON_BY_MONTH)
    top_mix_groups = category_all.head(8).index
    seasonal_counts = (
        merged.loc[merged["garment_group_name"].isin(top_mix_groups)]
        .groupby(["season", "garment_group_name"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=SEASON_ORDER, columns=top_mix_groups, fill_value=0)
    )
    seasonal_totals = merged.groupby("season", observed=True).size().reindex(SEASON_ORDER)
    seasonal_mix = seasonal_counts.div(seasonal_totals, axis=0)

    transaction_missing = {column: int(tx[column].isna().sum()) for column in tx.columns}
    article_quality_columns = [
        "product_code",
        "prod_name",
        "product_type_name",
        "garment_group_name",
        "colour_group_name",
        "detail_desc",
    ]
    article_missing = {
        column: int(articles[column].isna().sum())
        for column in article_quality_columns
        if column in articles.columns
    }
    missing_article_matches = int(merged["style_id"].isna().sum())
    nonpositive_prices = int((tx["price"] <= 0).sum())
    duplicate_article_ids = int(articles["article_id"].duplicated().sum())

    top_day = daily.loc[daily["transactions"].idxmax()]
    busiest_month = calendar_month.loc[calendar_month["avg_daily_transactions"].idxmax()]
    busiest_season = seasonal_daily.idxmax()
    top_category = category.iloc[0]
    top_style_id = str(style_sales.idxmax())
    style_names = articles.drop_duplicates("style_id").set_index("style_id")["prod_name"]
    top_style_name = str(style_names.get(top_style_id, "Unknown"))
    channel_share = tx["sales_channel_id"].value_counts(normalize=True).sort_index()

    plots = _save_plots(daily, category, customer_purchases, calendar_month, seasonal_mix)

    summary = {
        "run_scope": "full dataset" if max_transactions is None else f"latest {len(tx):,} transaction rows",
        "transactions": int(len(tx)),
        "customers": int(tx["customer_id"].nunique()),
        "articles": int(articles["article_id"].nunique()),
        "styles": int(articles["style_id"].nunique()),
        "date_min": str(tx["t_dat"].min().date()),
        "date_max": str(tx["t_dat"].max().date()),
        "sales_trends": {
            "average_daily_transactions": round(float(daily["transactions"].mean()), 2),
            "peak_day": str(pd.Timestamp(top_day["t_dat"]).date()),
            "peak_day_transactions": int(top_day["transactions"]),
            "busiest_calendar_month": MONTH_LABELS[int(busiest_month["calendar_month"]) - 1],
            "busiest_calendar_month_avg_daily_transactions": round(
                float(busiest_month["avg_daily_transactions"]), 2
            ),
        },
        "customer_behavior": {
            "median_transactions_per_customer": float(customer_purchases.median()),
            "p90_transactions_per_customer": float(customer_purchases.quantile(0.90)),
            "p99_transactions_per_customer": float(customer_purchases.quantile(0.99)),
            "repeat_customer_share": float((customer_purchases > 1).mean()),
            "single_transaction_customer_share": float((customer_purchases == 1).mean()),
            "channel_share": {str(int(key)): float(value) for key, value in channel_share.items()},
        },
        "product_behavior": {
            "median_units_per_style": float(style_sales.median()),
            "p90_units_per_style": float(style_sales.quantile(0.90)),
            "top_1pct_style_sales_share": float(
                style_sales.sort_values(ascending=False).head(max(1, len(style_sales) // 100)).sum()
                / style_sales.sum()
            ),
            "top_style_id": top_style_id,
            "top_style_name": top_style_name,
            "top_style_units": int(style_sales.max()),
            "top_garment_group": str(top_category["garment_group_name"]),
            "top_garment_group_share": float(top_category["transactions"] / len(merged)),
        },
        "seasonality": {
            "definition": {
                "Spring": "March-May",
                "Summer": "June-August",
                "Autumn": "September-November",
                "Winter": "December-February",
            },
            "average_daily_transactions": {
                season: round(float(value), 2) for season, value in seasonal_daily.items()
            },
            "busiest_season_by_average_day": str(busiest_season),
        },
        "data_quality": {
            "transaction_missing_values": transaction_missing,
            "article_missing_values": article_missing,
            "transaction_rows_without_article_metadata": missing_article_matches,
            "nonpositive_prices": nonpositive_prices,
            "duplicate_article_ids": duplicate_article_ids,
            "duplicate_transaction_note": (
                "Exact repeated transaction rows are retained because the dataset has no transaction_id or "
                "quantity field; identical rows can represent multiple purchased units."
            ),
        },
        "plots": [str(path) for path in plots],
    }

    report_lines = [
        "# Exploratory Data Analysis Summary",
        "",
        "## Scope",
        "",
        f"- Run scope: {summary['run_scope']}",
        f"- Transactions analyzed: {summary['transactions']:,}",
        f"- Unique customers: {summary['customers']:,}",
        f"- Unique articles: {summary['articles']:,}",
        f"- Unique style_ids using product_code: {summary['styles']:,}",
        f"- Date range: {summary['date_min']} to {summary['date_max']}",
        "",
        "## Sales Trends",
        "",
        f"- Average daily volume is {summary['sales_trends']['average_daily_transactions']:,.0f} transactions.",
        f"- The highest-volume day is {summary['sales_trends']['peak_day']} with {summary['sales_trends']['peak_day_transactions']:,} transactions.",
        f"- {summary['sales_trends']['busiest_calendar_month']} has the highest average daily volume across observed years at {summary['sales_trends']['busiest_calendar_month_avg_daily_transactions']:,.0f} transactions.",
        "- Sharp one-day peaks suggest campaigns, releases, or calendar effects that the provided data cannot identify directly.",
        "",
        "## Customer Behavior",
        "",
        f"- Median transactions per customer: {summary['customer_behavior']['median_transactions_per_customer']:.0f}.",
        f"- 90th percentile transactions per customer: {summary['customer_behavior']['p90_transactions_per_customer']:.0f}; 99th percentile: {summary['customer_behavior']['p99_transactions_per_customer']:.0f}.",
        f"- Repeat-customer share: {summary['customer_behavior']['repeat_customer_share']:.1%}; single-transaction share: {summary['customer_behavior']['single_transaction_customer_share']:.1%}.",
        f"- Sales-channel mix: channel 1 {summary['customer_behavior']['channel_share'].get('1', 0):.1%}, channel 2 {summary['customer_behavior']['channel_share'].get('2', 0):.1%}.",
        "",
        "## Product And Category Performance",
        "",
        f"- Median units per style: {summary['product_behavior']['median_units_per_style']:.0f}; 90th percentile: {summary['product_behavior']['p90_units_per_style']:.0f}.",
        f"- The top 1% of styles account for {summary['product_behavior']['top_1pct_style_sales_share']:.1%} of transaction rows, showing a concentrated winner/long-tail pattern.",
        f"- Highest-volume style: {summary['product_behavior']['top_style_id']} ({summary['product_behavior']['top_style_name']}) with {summary['product_behavior']['top_style_units']:,} units.",
        f"- Largest garment group: {summary['product_behavior']['top_garment_group']} at {summary['product_behavior']['top_garment_group_share']:.1%} of transaction rows.",
        "",
        "## Seasonality",
        "",
        f"- Seasons use meteorological month groups: Spring Mar-May, Summer Jun-Aug, Autumn Sep-Nov, Winter Dec-Feb.",
        f"- {summary['seasonality']['busiest_season_by_average_day']} has the highest average daily volume after controlling for unequal season coverage.",
    ]
    report_lines.extend(
        f"- {season}: {value:,.0f} average daily transactions."
        for season, value in summary["seasonality"]["average_daily_transactions"].items()
    )
    report_lines.extend(
        [
            "- Garment mix changes by season; the heatmap shows within-season shares for the eight largest garment groups.",
            "",
            "## Data Quality",
            "",
            f"- Transaction rows without article metadata: {summary['data_quality']['transaction_rows_without_article_metadata']:,}.",
            f"- Non-positive prices: {summary['data_quality']['nonpositive_prices']:,}.",
            f"- Duplicate article IDs: {summary['data_quality']['duplicate_article_ids']:,}.",
            f"- Missing article descriptions: {summary['data_quality']['article_missing_values'].get('detail_desc', 0):,}.",
            "- No missing values were found in the transaction columns used by the model." if not any(transaction_missing.values()) else "- Transaction-column missing values are listed in `eda_summary.json`.",
            f"- {summary['data_quality']['duplicate_transaction_note']}",
            "",
            "## Style Definition",
            "",
            "A style is H&M `product_code`, exposed as string `style_id`. This groups article/colour variants into a stable commercial product family without merging unrelated products.",
            "",
            "## Success Definition And Prediction Period",
            "",
            "A successful style has high transaction-row unit sales in the next 28-day holdout window. The 28-day horizon gives a practical buying signal while preserving three leakage-free snapshots inside the available history. It is a next-period proxy for the assignment's next-season decision, not a full three-month seasonal demand forecast.",
            "",
            "## Modeling Implications",
            "",
            "Customer breadth, repeated selling days, recent momentum, category, and product attributes are useful demand signals. The forecasting model itself does not include an explicit calendar-season feature; seasonality is analyzed separately and exposed through the seasonal view.",
            "",
            "## Stock Limitation",
            "",
            "The data does not include reliable stock-on-hand or availability history. Low sales can mean weak demand, stockout, limited distribution, or late launch. The model therefore predicts observed sales performance, not unconstrained customer demand.",
            "",
            "## Generated Plots",
            "",
        ]
    )
    report_lines.extend(f"- `{Path(path).relative_to(PROJECT_ROOT)}`" for path in plots)

    report = OUTPUTS_DIR / "eda_summary.md"
    report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (OUTPUTS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    path = run_eda()
    print(f"Wrote {path}")
