# Submission Notes

## How To Run

Use Python 3 from the repository root. The full H&M files must be available at:

```text
data/raw/articles.csv
data/raw/transactions_train.csv
data/raw/images/<prefix>/<article_id>.jpg
```

```bash
python3 -m pip install -r requirements.txt
python3 run_pipeline.py
python3 -m backend.api
```

Open `http://127.0.0.1:8000/app/`. A faster development run is available with `python3 run_pipeline.py --max-transactions 8000000`; the submitted outputs were produced from the full dataset.

## Final Top 3

| Rank | Style | Product | Score | Predicted units |
| ---: | --- | --- | ---: | ---: |
| 1 | `610776` | Tilly (1) | 100.00% | 5,981.97 |
| 2 | `717490` | Cat Tee. | 97.44% | 5,828.85 |
| 3 | `599580` | Timeless Midrise Brief | 96.04% | 5,744.93 |

The score is normalized for ranking and is not a probability. The selected style ID is H&M `product_code`, which groups article/color variants into a stable style.

## Model Metrics

The selected model is a two-period rolling `RandomForestRegressor`. It won a six-fold chronological comparison against an expanding regression, a top-decile classifier, and a blended ranker. The selection score uses average Precision@3/10 and penalizes fold variance.

Rolling-development averages:

| Metric | Selected model | Baseline |
| --- | ---: | ---: |
| MAE | **46.83** | 49.15 |
| RMSE | 179.32 | **174.19** |
| Precision@3 | 0.667 ± 0.000 | 0.667 ± 0.000 |
| Precision@10 | **0.683 ± 0.069** | 0.650 ± 0.096 |

Untouched final holdout (`2020-08-26` to `2020-09-22`):

| Metric | Selected model | Baseline |
| --- | ---: | ---: |
| MAE | **43.36** | 64.32 |
| RMSE | **199.40** | 229.01 |
| Precision@3 | 0.00 | 0.00 |
| Precision@10 | 0.20 | 0.20 |

The model is an improvement in rolling Precision@10 and forecast error, not a claim of solved top-three ranking. The final holdout was evaluated once after selection; it was not used to choose features, objective, horizon, blend, or hyperparameters.

### Final Holdout Diagnosis

The predicted winners were not the actual holdout top three:

| Predicted rank | Style | Actual holdout rank | What happened |
| ---: | --- | ---: | --- |
| 1 | `610776` Tilly (1) | 12 | Remained strong, but did not reach the extreme top three. |
| 2 | `717490` Cat Tee. | 22 | Recent demand was high, but its latest 14-day demand was already slowing. |
| 3 | `599580` Timeless Midrise Brief | 136 | Very high recent sales were followed by a sharp decline in the holdout. |

The actual holdout winners were `751471` Pluto RW slacks (predicted rank 8), `706016` Jade HW Skinny Denim TRS (predicted rank 4), and `762846` Lucy blouse (predicted rank 18). All three were accelerating before the holdout, but their lower absolute recent volumes kept them below established high-volume styles in the model ranking.

This pattern suggests temporal demand and product-lifecycle shift, but the dataset cannot establish the cause. It lacks stock availability, true launch dates, planned distribution, promotions, markdown campaigns, and browsing intent. Consequently, a fall in sales may reflect demand decay or reduced availability, while a surge may reflect genuine demand, a launch, wider distribution, or promotion.

Production corrections would be to add in-stock days and distribution coverage, true launch age, promotion/markdown signals, page views and add-to-cart activity; retrain more frequently; monitor ranking drift; and evaluate lifecycle-aware or category-specific learning-to-rank models across additional untouched seasonal holdouts.

### Business Definition and Better Production Ranking

The submitted ranking defines a winner as a style with high predicted unit sales over the next 28 days because transactions and product attributes are the reliable supplied inputs. It does not claim that unit volume alone is the retailer's optimal commercial objective.

The available data supports demand history, product attributes, price, customer breadth, seasonality, acceleration and category-relative momentum. It does not contain reliable stock exposure, true launch dates, cost/margin, returns, promotion history, browsing intent, planned distribution or supply constraints.

In production, the business should first choose the decision objective: availability-adjusted units, revenue, gross margin, sell-through, emerging-trend growth, or an assortment-balanced score. The recommended process is to keep the demand forecast as one input, then apply stock/exposure adjustment, profitability and lifecycle scoring, uncertainty, and category/season/supply constraints before selecting the final three products. These rules require stakeholder agreement and should be validated on future periods rather than tuned to this final holdout.

## Limitations

- Stock-on-hand and availability history are unavailable, so the target is observed future sales, not unconstrained customer demand.
- Transaction rows are treated as unit events; explicit basket quantity, markdown, campaign, and promotion fields are unavailable.
- Six rolling validation folds reduce dependence on one month, but the single final holdout still shows meaningful temporal regime risk and weak extreme-top ranking.
- The top-decile classifier and classifier/regression blend were tested but rejected because they were not competitive across folds.
- Observed launch age begins at the first sale in this dataset and may not equal the true product launch date.
- Forecast scores rank styles across the final scoring window. The Spring/Summer/Autumn/Winter views are descriptive historical views, not four separately trained seasonal forecast models.
- The final AI concept board is a visual design hypothesis, not a manufacturability or commercial-demand guarantee. The deterministic fallback remains available for reproducibility.

## Walkthrough Order

1. Open `outputs/eda_summary.md` and establish the full-data scope, customer behavior, winner concentration, seasonality, and data-quality findings.
2. Open the Forecast tab and explain the top-three ranking, predicted units, and winner-level model signals.
3. Open each winning style to show source imagery, product attributes, history, and recency signals.
4. Show `outputs/model_metrics.json`, then `outputs/style_explanations.md`: six-fold selection, final-holdout honesty, global importance, and grouped local sensitivity.
5. Switch to Summer (then another season) to demonstrate the descriptive historical seasonal view.
6. Scroll to Generated Concepts and connect each named design to its corresponding ranked winner from left to right.
7. Show Swagger at `/docs`, then close with `outputs/ai_concept_generation.md`, limitations, and the flow from `run_pipeline.py` to API to UI.

## Key Artifacts

```text
outputs/top_styles.json
outputs/model_metrics.json
outputs/model_feature_importance.json
outputs/style_explanations.md
outputs/seasonal_styles.json
outputs/generated_concepts.png
outputs/ai_concept_generation.md
models/style_winner_model.joblib
docs/screenshots/forecast-dashboard.png
docs/screenshots/summer-seasonal-view.png
docs/screenshots/mobile-forecast.png
```
