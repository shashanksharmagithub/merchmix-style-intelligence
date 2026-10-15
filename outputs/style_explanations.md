# Model Explanation Report

## Method

Grouped counterfactual sensitivity for the selected Two-period rolling Random Forest log-sales ranker. One feature group at a time is replaced with development-set medians/modes and the locked ranking score is recalculated. Deltas are diagnostic, non-additive, and not causal.

Global importance uses impurity-based feature importance from the fitted Random Forest component(s). It is descriptive, can favor high-cardinality or correlated features, and is not causal.

## Global Feature Importance

| Rank | Feature | Importance |
| ---: | --- | ---: |
| 1 | `last14_units` | 0.6039 |
| 2 | `last14_customers` | 0.1291 |
| 3 | `recent_units` | 0.1037 |
| 4 | `recent_customers` | 0.0342 |
| 5 | `observed_style_age_days` | 0.0170 |
| 6 | `recent_units_vs_category_median` | 0.0088 |
| 7 | `recent_category_share` | 0.0063 |
| 8 | `recent_avg_price` | 0.0057 |
| 9 | `momentum_vs_category_median` | 0.0056 |
| 10 | `hist_units_per_active_day` | 0.0056 |
| 11 | `article_count` | 0.0052 |
| 12 | `hist_avg_price` | 0.0045 |
| 13 | `days_since_last_sale` | 0.0042 |
| 14 | `prior_avg_price` | 0.0040 |
| 15 | `department_name` | 0.0040 |

## Top-Style Local Sensitivity

### #1 Tilly (1) (`610776`)

Ranking score: 1.0000; regression estimate: 5,981.97 units

- Recent demand is a supporting local ranking signal: 6,715 recent units and 3,481 units in the latest 14 days. Replacing this group with typical development-set values lowers the predicted demand by about 5,976 units.
- Historical demand is a supporting local ranking signal: 22,407 units from 14,184 customers across 84 selling days. Replacing this group with typical development-set values lowers the predicted demand by about 3,178 units.
- Category-relative demand is a supporting local ranking signal: recent demand is 419.69x its garment-group median. Replacing this group with typical development-set values lowers the predicted demand by about 1,538 units.

### #2 Cat Tee. (`717490`)

Ranking score: 0.9744; regression estimate: 5,828.85 units

- Recent demand is a supporting local ranking signal: 6,419 recent units and 2,684 units in the latest 14 days. Replacing this group with typical development-set values lowers the predicted demand by about 5,824 units.
- Historical demand is a supporting local ranking signal: 18,226 units from 11,898 customers across 84 selling days. Replacing this group with typical development-set values lowers the predicted demand by about 3,048 units.
- Category-relative demand is a supporting local ranking signal: recent demand is 401.19x its garment-group median. Replacing this group with typical development-set values lowers the predicted demand by about 1,413 units.

### #3 Timeless Midrise Brief (`599580`)

Ranking score: 0.9604; regression estimate: 5,744.93 units

- Recent demand is a supporting local ranking signal: 8,049 recent units and 2,434 units in the latest 14 days. Replacing this group with typical development-set values lowers the predicted demand by about 5,739 units.
- Historical demand is a supporting local ranking signal: 31,118 units from 20,433 customers across 84 selling days. Replacing this group with typical development-set values lowers the predicted demand by about 3,018 units.
- Category-relative demand is a supporting local ranking signal: recent demand is 365.86x its garment-group median. Replacing this group with typical development-set values lowers the predicted demand by about 1,444 units.
