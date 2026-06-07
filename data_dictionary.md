# Data Dictionary

## Customer identifiers

`customer_id`

Unique customer identifier used inside the generated study dataset.

`statement_month`

Synthetic monthly observation used for trend and scenario review.

## Credit and income fields

`limit_balance`

Approved credit limit for the customer.

`annual_income`

Generated annual income estimate used for affordability style signals.

`utilization_ratio`

Average balance divided by credit limit across recent statements.

`payment_ratio`

Total recent payments divided by total recent bill amount.

## Customer attributes

`sex`

Generated customer category.

`education`

Generated education category.

`marital_status`

Generated marital status category.

`age`

Customer age in years.

`age_group`

Age band used for SQL exploration.

`credit_limit_band`

Credit limit band used for portfolio checks.

## Repayment behaviour

`pay_status_1` to `pay_status_6`

Recent repayment status indicators. Higher values represent more delay.

`months_with_delay`

Number of recent months with positive repayment delay.

`bill_amt_1` to `bill_amt_6`

Recent bill amounts.

`pay_amt_1` to `pay_amt_6`

Recent payment amounts.

## Target and model outputs

`default_next_month`

Binary target. A value of one means the customer defaults in the next month.

`predicted_pd`

Estimated probability of default from the selected model.

`risk_band`

Interpretation band from A to E. A is the lowest risk group and E is the highest risk group.

## Stress testing fields

`ead`

Exposure at default assumption. This project uses recent balance and credit limit information to approximate exposure.

`lgd`

Loss given default assumption. Scenario values are documented in the stress testing report.

`loss`

Simulated portfolio credit loss from a Monte Carlo run.

