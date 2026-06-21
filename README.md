# Risk Signals Credit Lab

Credit scoring, model validation and portfolio stress testing in a self-contained case study.

This project takes synthetic customer-level credit data and walks through the full analytics chain: SQL exploration, default prediction with multiple models, probability calibration, SHAP explainability, Monte Carlo loss simulation and counterfactual analysis. It's not a production model. The goal is to show how these pieces connect in a realistic workflow.

## What's in here

- SQL queries for segment-level portfolio checks (the kind of thing you'd run first in any credit review)
- Logistic regression baseline plus decision tree, random forest and gradient boosting comparisons
- Model validation: AUC, Brier score, calibration curves, risk band checks, threshold sensitivity
- SHAP explainability for global and individual-level feature attribution
- Monte Carlo portfolio loss simulation under baseline, mild and severe stress
- DiCE counterfactual explanations for high-risk customers
- Scenario-based time series view of PD under stress multipliers

## Why I built it

I wanted a project that goes past the usual "train a classifier and report accuracy" pattern. In credit risk, the interesting questions start *after* you have a model. Is it calibrated? Do the risk bands make sense? What happens to portfolio losses under stress? Which feature changes would actually move a customer out of the high-risk band?

I structured it as a small model review case study where the predictions get checked, interpreted and translated into portfolio-level risk measures.

## Project structure

```text
data/raw/                       # Synthetic customer data
data/processed/                 # Model outputs, scores, stress results
sql/                            # Portfolio and segment queries
notebooks/                      # Executed analysis notebooks
reports/                        # Written summaries
figures/                        # SVG charts
src/risk_signals_credit_lab.py  # Main reproducible pipeline
tests/                          # Small smoke checks
LICENSE
requirements.txt
```

## How to run

```bash
pip install -r requirements.txt
python src/risk_signals_credit_lab.py
```

This generates the data, runs the models, produces figures and writes the reports. The notebooks can then be run interactively for the sklearn/SHAP/DiCE extensions.

The main pipeline uses only numpy, pandas and the standard library. Notebooks pull in sklearn, shap and dice-ml for the richer analysis.

## Tests

The tests are small smoke checks. They cover synthetic data generation, the feature split with a lightweight logistic model, and the validation plus SQL summary outputs.

```bash
pytest
```

## Sample outputs

![Default rate by segment](figures/default_rate_by_segment.svg?v=4)

![ROC curve](figures/roc_curve.svg?v=4)

![Calibration curve](figures/calibration_curve.svg?v=4)

![Risk band observed default](figures/risk_band_observed_default.svg?v=4)

![Feature importance](figures/explainability_feature_importance.svg?v=4)

![Monte Carlo loss distribution](figures/monte_carlo_loss_distribution.svg?v=4)

## Stress testing results

| Scenario | Expected loss | VaR 95 | CVaR 95 |
|----------|--------------|--------|---------|
| Baseline | 6,668,905 | 7,185,520 | 7,325,264 |
| Mild stress | 8,153,222 | 8,747,561 | 8,890,582 |
| Severe stress | 10,147,509 | 10,807,880 | 10,991,792 |

Full simulation output in `data/processed/stress_test_summary.csv`.

## Limitations

- The data is synthetic, so the results shouldn't be read as real credit performance
- Stress multipliers are flat rather than macro-linked
- Risk bands are for interpretation only, not regulatory scorecards
- The Monte Carlo setup shows the workflow but isn't calibrated for capital estimation
- No reject inference, population stability or fairness analysis yet

## License

MIT License.
