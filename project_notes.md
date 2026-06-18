# Project Notes

## What this is for

A credit risk case study that covers more than just building a classifier. The pipeline goes from data checks through to portfolio stress testing, with model validation, explainability and counterfactual analysis along the way.

It is not meant to be a bank-grade model. It is a learning project with clear assumptions.

## Key design decisions

- Used synthetic data to avoid licensing/privacy issues while keeping the full modelling workflow intact
- SQL exploration comes first because that's how credit risk review usually starts in practice
- Logistic regression is the interpretable baseline; tree models are there for comparison
- Probability quality gets its own notebook (calibration, Brier score, risk bands) because in credit risk, good ranking matters more than raw accuracy
- SHAP is in a separate notebook since it needs the `shap` package
- Monte Carlo connects PD → EAD → LGD → portfolio loss distribution
- DiCE counterfactuals give an "actionable" angle on the model output

## Data notes

The synthetic dataset follows the structure of common credit card default datasets, with credit limits, demographics, repayment history, bill/payment amounts and a binary default target. The generation process uses realistic correlations (utilization drives default, education affects income, etc.) but the numbers aren't from real customers.

## Questions this tries to answer

1. Which segments show higher default rates?
2. Does the model rank risk in the right order?
3. Are the predicted probabilities calibrated?
4. What's driving the predictions?
5. How does portfolio loss change under stress?
6. What would need to change for a high-risk customer to be predicted as lower risk?
7. Where are the gaps before this could be used for real decisions?
