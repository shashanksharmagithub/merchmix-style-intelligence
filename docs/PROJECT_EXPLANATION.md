# Project Explanation

## 1. Problem Definition

The business problem is to help a fashion retailer decide which styles to develop for the next selling period. The assignment requires an end-to-end flow:

```text
Historical retail data -> predictive model -> top winning styles -> new product concepts -> API -> user interface
```

The primary modeling objective is to rank styles by expected future sales performance. The product objective is to turn those model outputs into something a business user can inspect and discuss.

## 2. Data Used

The intended data source is the H&M Personalized Fashion Recommendations dataset from Kaggle:

- `transactions_train.csv`
- `articles.csv`
- matching catalogue images linked by `article_id`

The submitted artifacts use the complete transaction file: 31,788,324 rows from 2018-09-20 through 2020-09-22. The project also supports a recent-row development sample through `--max-transactions`. A sampled run can underrepresent rare styles, long-tail categories, and customer breadth, so it is not used for the submitted results.

## 3. Style Definition

I define a style as H&M `product_code`, exposed as string `style_id`.

Why this decision:

- An individual `article_id` can be too granular because colour or variant-level sparsity makes future sales harder to estimate.
- A broad attribute combination such as product type plus colour can merge unrelated products.
- `product_code` is a stable commercial grouping that usually represents the same product family and gives enough sales volume for modeling.

The representative article image is taken from one article belonging to that `style_id`.

## 4. Success Definition

A successful style is a style with high future unit sales in the selected prediction window.

The target is:

```text
target_units = transaction count for style_id during the future prediction window
```

This is practical because the assignment data reliably contains transactions over time. The target avoids invented assumptions about inventory, margin, stock, or markdowns.

### Business Definition Used and How It Could Evolve

For this assignment, a winner means **high observed unit sales during the next 28 days**. This is the most defensible target supported directly by the supplied data, but it is not automatically equivalent to the best merchandising investment.

What the supplied data supports:

- transaction timing, price, customer, article and product-code demand history
- product category, colour, department, appearance and description
- catalogue images
- derived recency, acceleration, season, category-relative momentum and observed-age signals

What the supplied data does not support:

- stock-on-hand, in-stock days, lost sales or store/channel distribution exposure
- true launch and end-of-life dates
- unit cost, gross margin or buying quantity
- explicit markdown, promotion and campaign history
- returns, page views, search activity, wish lists and add-to-cart intent
- future assortment, supply capacity and category-space constraints

With those fields, the business could define a winner differently:

| Business objective | Better target or ranking measure |
| --- | --- |
| Demand winner | Availability-adjusted future units |
| Commercial winner | Future revenue or gross margin |
| Inventory winner | Sell-through rate and weeks of supply |
| Emerging trend | Growth/acceleration adjusted for launch age |
| Low-risk winner | Expected value penalized by forecast uncertainty and returns |
| Assortment winner | Model score after category, season, price-point and supply constraints |

The recommended production design separates prediction from the decision policy:

```text
Demand forecast -> availability/exposure adjustment -> margin and lifecycle score -> assortment constraints -> final ranked styles
```

The weights and constraints should be agreed with merchandising, buying, finance and supply teams before retraining. They should not be chosen afterward merely to make one historical holdout look better.

## 5. Prediction Period and Validation

The pipeline uses time-aware windows:

- historical feature window: previous 84 days
- target period per snapshot: next 28 days
- development snapshots: eight consecutive periods from January through August 2020
- model-selection folds: six rolling validations after a two-snapshot minimum training history
- untouched final holdout: 2020-08-26 through 2020-09-22

The model never uses target-period transactions when constructing a snapshot's features. Each fold trains only on earlier snapshots. Model objective and training horizon are selected from the six development folds; the final holdout is evaluated once afterward and is not used for tuning.

## 6. Feature Engineering

The feature table is built at `style_id` level.

Main feature groups:

- historical demand: total units, customer count, active selling days
- recent momentum: last 28 days of units and customers
- prior comparison: previous 28-day window
- trend proxy: recent units divided by prior units
- short-term acceleration: latest 14 days versus the preceding 14 days, for units and customers
- category-normalized momentum: demand level, momentum, acceleration, and share relative to garment group
- recency: days since last sale
- lifecycle: days since the style's first observed sale in this dataset
- calendar: prediction month, cyclical month encoding, and explicit Spring/Summer/Autumn/Winter feature
- price behavior: average and max transaction price
- product attributes: product type, garment group, department, colour, appearance, section

These features are chosen because they balance predictive signal and business interpretability. A buyer or merchandiser can understand why recent momentum, category, breadth of demand, and product attributes matter.

## 7. Model Choice

The selected model is a two-period rolling `RandomForestRegressor` from scikit-learn, trained on `log1p(target_units)`.

Why this model:

- handles non-linear demand patterns better than a plain linear model
- works well on mixed numeric and encoded categorical retail features
- the short rolling training horizon adapts better to fashion-demand drift than equal weighting of all earlier snapshots
- trains quickly enough for an assignment setting on the style-level feature table
- does not require external gradient-boosting packages
- supports a clean reproducible pipeline with numeric and categorical preprocessing

The target is transformed with `log1p(target_units)` during training and transformed back for prediction. This reduces the effect of extreme blockbuster styles while preserving ranking behavior.

The rolling comparison also evaluates:

- an expanding-window Random Forest regression
- a Random Forest top-decile classification objective
- a 55/45 blend of regression rank and top-decile probability rank

The top-decile classifier and blend are retained as evaluated alternatives, not presented as successful models. They underperform on extreme-top ranking in these folds.

## 8. Baseline

The baseline ranks styles by recent sales.

Why this baseline:

- simple and business-plausible
- hard enough to beat in retail forecasting
- exposes whether the model adds value beyond momentum

The evaluation writes every fold, candidate summary, baseline summary, stability measure, selection formula, and untouched final-holdout result to `outputs/model_metrics.json`.

Metrics:

- MAE
- RMSE
- Precision@3
- Precision@10

The precision-at-K metrics are especially relevant because the assignment asks for the top three winning styles.

Across six development folds, the selected model ties baseline average Precision@3 (`0.667`) and improves Precision@10 (`0.683` versus `0.650`) and MAE (`46.83` versus `49.15`). Baseline average RMSE is slightly better (`174.19` versus `179.32`), so no broader claim is made.

On the untouched final holdout, the selected model improves MAE/RMSE (`43.36/199.40` versus `64.32/229.01`) but ties the baseline at weak ranking precision (`P@3 0.0`, `P@10 0.2`). This indicates temporal regime risk: the model predicts sales magnitude better, but does not reliably identify the exact extreme winners in that month.

### Final Holdout Error Analysis

The model's predicted top three had actual holdout ranks 12 (`610776` Tilly), 22 (`717490` Cat Tee), and 136 (`599580` Timeless Midrise Brief). The actual top three were `751471` Pluto RW slacks, `706016` Jade HW Skinny Denim TRS, and `762846` Lucy blouse; the model ranked them 8, 4, and 18 respectively.

The miss is interpretable from the available signals. The predicted styles had very high absolute recent demand, which the fitted model correctly learned as the dominant historical signal. However, Cat Tee and Timeless Midrise Brief had negative latest-14-day acceleration, while the actual winners were accelerating from lower bases. Tilly remained commercially strong at actual rank 12, so that prediction was directionally useful but not precise enough for the top-three decision.

This is evidence of temporal and lifecycle drift, not proof of a single causal explanation. Transaction data alone cannot reveal whether the change came from consumer preference, stock-outs, product launch timing, distribution expansion, markdowns, or campaigns. Those variables are absent and were not invented.

The implemented mitigation is already stronger than a single split: six rolling validation folds, a short rolling training horizon, explicit season/month signals, launch-age proxy, category-relative momentum, and demand acceleration. Further correction requires in-stock exposure, true launch dates, promotion/markdown and distribution data, browsing-intent signals, more frequent retraining and drift monitoring, and additional untouched seasonal holdouts for lifecycle-aware or category-specific ranking models.

## 9. Top Style Selection

After training, the pipeline writes:

```text
outputs/style_predictions.csv
outputs/top_styles.json
```

Each top style includes:

- rank
- `style_id`
- representative article
- prediction score
- predicted units
- category
- colour
- garment group
- sales history
- recent sales
- model explanation
- image URL where available

The project provides two fitted-model explanation layers:

- global Random Forest impurity-based feature importance in `outputs/model_feature_importance.json`
- grouped local counterfactual sensitivity for each winner in `outputs/style_explanations.md`

For local sensitivity, one feature group at a time is replaced with training medians/modes and the model prediction is recalculated. The resulting unit delta is diagnostic, non-additive, and non-causal, but it directly reflects the fitted model rather than a post-hoc business rule.

## 10. Concept Generation Logic

The concept stage uses the top three predicted styles and creates one brief per style.

Each concept keeps:

- the strongest source category cue
- the colour family
- the garment/product context
- the commercial demand signals from the model explanation

The final concepts follow the locked rank order and introduce specific changes:

- rank 1, `610776`: cropped camp-collar botanical top with engineered print placement and gathered side
- rank 2, `717490`: architectural white jersey top with curved seams, tonal eyelet inserts, and mock neck
- rank 3, `599580`: asymmetric high-waist wrap swim brief with diagonal waistband and mixed rib/matte panels

This avoids simply copying the source product. The final concept image should be saved at:

```text
outputs/generated_concepts.png
```

The final concept image in `outputs/generated_concepts.png` is generated with AI using the selected catalogue images as visual references. The deterministic board is retained separately as `outputs/generated_concepts_fallback.png`, and the source references plus prompt are documented in `outputs/ai_concept_generation.md`.

## 11. Backend Design

The backend separates serving logic from model artifacts:

- `backend/model_service.py` loads prediction JSON and exposes lookup functions.
- `backend/api.py` defines routes and HTTP behavior.

Required endpoints:

```text
GET /styles/top
GET /styles/{style_id}
```

The API returns:

- `503` if predictions have not been generated yet
- `404` if a requested `style_id` is unknown

It also mounts product images under `/images` when the catalogue image directory exists.

Endpoint contracts, response semantics, errors, cache refresh behavior, and interactive-documentation URLs are recorded in `docs/API.md`. With FastAPI installed, Swagger UI is available at `/docs` and OpenAPI JSON at `/openapi.json`.

## 12. Frontend Design

The frontend is intentionally simple and business-focused.

Users can:

- view the ranked top predicted styles
- inspect product image and attributes
- see prediction score
- see historical and recent sales
- read the model reasoning

The frontend has no build step, which makes it easy to run during a technical walkthrough.

## 12a. Seasonal View

For bonus coverage, the project also builds `outputs/seasonal_styles.json`. This file aggregates historical performance by Spring, Summer, Autumn, and Winter, then exposes the top seasonal styles through the API and frontend. The UI lets a business user switch from the forecasted winners to seasonal historical winners and inspect how product direction changes by season.

## 13. Known Limitation: No Stock Data

The biggest modeling limitation is missing stock-on-hand and availability data.

Impact:

- low sales may mean low demand, but may also mean the item was out of stock
- high sales may reflect strong availability or broad distribution
- model predictions are for observed sales, not pure customer demand

Additional data that would improve the model:

- daily stock-on-hand by article/store/channel
- product launch date
- markdowns and promotions
- store/channel availability
- product page views and add-to-cart events
- returns
- margin
- planned season/calendar metadata

## 14. Future Improvements

Best next improvements:

- add image embeddings from catalogue images
- add stock-aware and category-specific ranking objectives
- extend evaluation to more years and multiple untouched seasonal holdouts
- add SHAP explanations
- add a concept-generation agent that uses selected style images and briefs
- add stock, promotion, returns, margin, and availability signals

## 15. How to Explain This in the Walkthrough

Recommended walkthrough order:

1. Start with the business objective.
2. Explain why `product_code` is the style definition.
3. Show the eight time-aware development snapshots, six rolling folds, and untouched final holdout.
4. Compare average and variance versus baseline, then state the final-holdout result without overclaiming.
5. Open `top_styles.json` and explain the three selected styles.
6. Show the fitted-model explanation report, concept briefs, and final generated concept image.
7. Start the API and show `/styles/top`.
8. Open the frontend and click into a style detail page.
9. Close with limitations and the data that would improve production quality.
