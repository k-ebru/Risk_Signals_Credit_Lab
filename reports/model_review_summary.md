# Model Review Summary

## Objective

This report reviews a credit default modelling workflow for a generated customer portfolio. The focus is not only prediction quality, but also model review, calibration, explainability and business interpretation.

## Dataset overview

The modelling table contains 3,200 customers. The observed default rate in the full portfolio is 0.132.

The train and test split is stored in `data/processed/train_test_split_summary.csv`.

## Model candidates

1. Logistic regression
2. Decision tree
3. Random forest
4. Gradient boosting

## Selected model

The selected model is `logistic_regression` based on test AUC with Brier score used as a probability quality check.

## Performance summary

1. logistic_regression: AUC 0.812, Brier 0.093, precision 0.441, recall 0.488
2. random_forest: AUC 0.802, Brier 0.096, precision 0.495, recall 0.411
3. gradient_boosting: AUC 0.796, Brier 0.098, precision 0.527, recall 0.372
4. decision_tree: AUC 0.778, Brier 0.101, precision 0.389, recall 0.434

## Calibration review

The calibration review compares predicted probability of default against observed default rate across ten score buckets. The detailed output is stored in `data/processed/calibration_table.csv`.

The selected model has test AUC 0.812 and Brier score 0.093.

## Risk band review

1. Band A: customers 640, average predicted PD 0.015, observed default rate 0.014
2. Band B: customers 640, average predicted PD 0.035, observed default rate 0.033
3. Band C: customers 640, average predicted PD 0.068, observed default rate 0.073
4. Band D: customers 640, average predicted PD 0.143, observed default rate 0.141
5. Band E: customers 640, average predicted PD 0.408, observed default rate 0.397

The bands are intended for interpretation only. They are not a regulatory scorecard.

## Explainability summary

The lightweight pipeline creates a coefficient based explanation summary for the logistic baseline. The SHAP notebook extends this with package based model explanations when the optional dependency is installed.

1. utilization_ratio: importance 0.566, direction increases_risk
2. months_with_delay: importance 0.409, direction increases_risk
3. bill_amt_6: importance 0.188, direction reduces_risk
4. pay_status_1: importance 0.154, direction increases_risk
5. bill_amt_3: importance 0.148, direction increases_risk
6. pay_amt_1: importance 0.133, direction reduces_risk
7. age_group_40_to_49: importance 0.130, direction increases_risk
8. bill_amt_2: importance 0.125, direction increases_risk

## DiCE and counterfactual explanation review

The project includes `notebooks/09_counterfactual_explanations.ipynb` and `data/processed/counterfactual_examples.csv`.

The notebook includes a DiCE example using `dice-ml` and a separate set of generated counterfactual examples from the main pipeline. These examples focus on selected high risk customers and compare baseline predicted default probability with practical feature changes. The combined scenario lowers utilisation and improves recent repayment behaviour.

1. combined: starting PD 0.912, counterfactual PD 0.034, average reduction 0.878
2. improve_repayment: starting PD 0.912, counterfactual PD 0.075, average reduction 0.837
3. lower_utilization: starting PD 0.912, counterfactual PD 0.673, average reduction 0.239

These counterfactuals are used for model interpretation. They are not presented as causal claims.

## Main limitations

1. The dataset is generated and should not be interpreted as real bank data.
2. The feature set is limited to a small credit card style portfolio view.
3. The model does not include reject inference or full population stability testing.
4. The SHAP section is a notebook extension and depends on optional packages.

## Recommendations

1. Treat the model output as a risk ranking and review tool rather than an automated decision engine.
2. Use calibration and risk band checks before interpreting probability values.
3. Add fairness, stability and macroeconomic validation before using this workflow for higher stakes decisions.
