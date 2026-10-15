# Exploratory Data Analysis Summary

## Scope

- Run scope: full dataset
- Transactions analyzed: 31,788,324
- Unique customers: 1,362,281
- Unique articles: 105,542
- Unique style_ids using product_code: 47,224
- Date range: 2018-09-20 to 2020-09-22

## Sales Trends

- Average daily volume is 43,308 transactions.
- The highest-volume day is 2019-09-28 with 198,622 transactions.
- Jun has the highest average daily volume across observed years at 61,178 transactions.
- Sharp one-day peaks suggest campaigns, releases, or calendar effects that the provided data cannot identify directly.

## Customer Behavior

- Median transactions per customer: 9.
- 90th percentile transactions per customer: 60; 99th percentile: 187.
- Repeat-customer share: 90.3%; single-transaction share: 9.7%.
- Sales-channel mix: channel 1 29.6%, channel 2 70.4%.

## Product And Category Performance

- Median units per style: 95; 90th percentile: 1398.
- The top 1% of styles account for 30.2% of transaction rows, showing a concentrated winner/long-tail pattern.
- Highest-volume style: 706016 (Jade HW Skinny Denim TRS) with 183,921 units.
- Largest garment group: Jersey Fancy at 16.3% of transaction rows.

## Seasonality

- Seasons use meteorological month groups: Spring Mar-May, Summer Jun-Aug, Autumn Sep-Nov, Winter Dec-Feb.
- Summer has the highest average daily volume after controlling for unequal season coverage.
- Spring: 43,880 average daily transactions.
- Summer: 50,654 average daily transactions.
- Autumn: 41,258 average daily transactions.
- Winter: 37,355 average daily transactions.
- Garment mix changes by season; the heatmap shows within-season shares for the eight largest garment groups.

## Data Quality

- Transaction rows without article metadata: 0.
- Non-positive prices: 0.
- Duplicate article IDs: 0.
- Missing article descriptions: 416.
- No missing values were found in the transaction columns used by the model.
- Exact repeated transaction rows are retained because the dataset has no transaction_id or quantity field; identical rows can represent multiple purchased units.

## Style Definition

A style is H&M `product_code`, exposed as string `style_id`. This groups article/colour variants into a stable commercial product family without merging unrelated products.

## Success Definition And Prediction Period

A successful style has high transaction-row unit sales in the next 28-day holdout window. The 28-day horizon gives a practical buying signal while preserving three leakage-free snapshots inside the available history. It is a next-period proxy for the assignment's next-season decision, not a full three-month seasonal demand forecast.

## Modeling Implications

Customer breadth, repeated selling days, recent momentum, category, and product attributes are useful demand signals. The forecasting model itself does not include an explicit calendar-season feature; seasonality is analyzed separately and exposed through the seasonal view.

## Stock Limitation

The data does not include reliable stock-on-hand or availability history. Low sales can mean weak demand, stockout, limited distribution, or late launch. The model therefore predicts observed sales performance, not unconstrained customer demand.

## Generated Plots

- `outputs/eda_plots/daily_transactions.png`
- `outputs/eda_plots/garment_group_transactions.png`
- `outputs/eda_plots/customer_purchase_frequency.png`
- `outputs/eda_plots/calendar_month_seasonality.png`
- `outputs/eda_plots/seasonal_garment_mix.png`
