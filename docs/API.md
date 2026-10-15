# API Documentation

Base URL for local development:

```text
http://127.0.0.1:8000
```

After installing `requirements.txt`, FastAPI provides interactive documentation at:

- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

The dependency-free fallback server supports the same business endpoints but does not provide Swagger UI.

## Endpoint Summary

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Confirm service and artifact readiness |
| `GET` | `/styles/top` | Return the three forecast winners |
| `GET` | `/styles/{style_id}` | Return one winner's detailed record |
| `GET` | `/seasons` | Return every historical seasonal view |
| `GET` | `/styles/season/{season}` | Return historical winners for one season |
| `POST` | `/admin/refresh` | Clear JSON caches after regenerating artifacts |

## GET /styles/top

Returns the ranked forecast winners plus evaluation metadata.

```json
{
  "styles": [
    {
      "rank": 1,
      "style_id": "610776",
      "name": "Tilly (1)",
      "prediction_score": 1.0,
      "predicted_units": 5981.97,
      "ranking_model": "Two-period rolling Random Forest log-sales ranker",
      "category": "T-shirt",
      "sales_history": 22407,
      "model_explanation": [
        "Recent demand is a supporting local ranking signal: 6,715 recent units and 3,481 units in the latest 14 days..."
      ],
      "explanation_method": "Grouped counterfactual sensitivity for the selected two-period rolling model..."
    }
  ],
  "metrics": {},
  "data_note": "The final holdout is reported but was not used for model selection; stock constraints remain unknown.",
  "generated_at": "ISO-8601 timestamp"
}
```

`prediction_score` is normalized relative to the highest-scoring style and is not a probability. `predicted_units` is the selected rolling Random Forest estimate for the 28-day holdout period. `top_decile_probability` is emitted for diagnostic comparison but does not drive the selected ranking.

## GET /styles/{style_id}

Returns one current forecast winner. In addition to the summary fields, the response contains:

- `attributes`: product group, appearance, department, and section
- `historical_performance`: feature and prediction windows, units, customers, active days, acceleration, category-relative demand, observed age, and prediction season
- `model_explanation_details`: grouped counterfactual deltas in predicted units
- `explanation_method`: interpretation and caveat for those deltas

The explanation deltas are diagnostic, non-additive, and non-causal. For the selected regression they are changes in predicted units when a feature group is replaced by typical development-set values.

## Seasonal Endpoints

Valid season values are `spring`, `summer`, `autumn`, and `winter`. Seasonal endpoints describe historical performance and are separate from the 28-day forecast model.

## Errors

Unknown style or season:

```json
{
  "detail": "Unknown style_id: 999999"
}
```

HTTP status: `404 Not Found`

Missing generated artifacts:

```json
{
  "detail": "Prediction output is missing. Run the data science pipeline first."
}
```

HTTP status: `503 Service Unavailable`

## Artifact Refresh

The API reads pipeline output from `outputs/top_styles.json` and `outputs/seasonal_styles.json`. After regenerating those files while the server is running, call:

```bash
curl -X POST http://127.0.0.1:8000/admin/refresh
```

This clears the in-process cache; it does not retrain the model.
