# Portfolio Stress Testing Report

## Portfolio assumptions

The portfolio uses customer level predicted probability of default from the selected model. Exposure at default is approximated from credit limit and recent balance information. Loss given default is a scenario assumption.

## PD EAD and LGD setup

1. PD comes from the selected model output.
2. EAD is approximated as a share of credit limit.
3. LGD is set at 0.45 in baseline, 0.50 in mild stress and 0.55 in severe stress.

## Baseline loss estimate

Baseline expected loss is 6,668,905. Baseline VaR 95 is 7,185,520.

## Mild and severe stress scenarios

1. baseline: expected loss 6,668,905, VaR 95 7,185,520, CVaR 95 7,325,264
2. mild_stress: expected loss 8,153,222, VaR 95 8,747,561, CVaR 95 8,890,582
3. severe_stress: expected loss 10,147,509, VaR 95 10,807,880, CVaR 95 10,991,792

The severe stress scenario increases expected loss by 52.2 percent compared with baseline.

## Time series risk signal

The highest monthly average predicted PD appears in July 2025 with an average predicted PD of 0.151.

The time series view is scenario based because the generated study data is not a full macroeconomic panel.

## Monte Carlo loss distribution

The simulation runs 10000 portfolio loss draws for each scenario. The full distribution is stored in `data/processed/monte_carlo_loss_distribution.csv`.

## VaR and CVaR summary

1. baseline: VaR 95 7,185,520, CVaR 95 7,325,264, VaR 99 7,416,164
2. mild stress: VaR 95 8,747,561, CVaR 95 8,890,582, VaR 99 8,986,730
3. severe stress: VaR 95 10,807,880, CVaR 95 10,991,792, VaR 99 11,100,145

## Interpretation

The simulation translates individual customer probability estimates into a portfolio view. This makes it easier to discuss expected loss and tail risk under documented assumptions.

## Limitations

1. The PD stress multipliers are simplified scenario assumptions.
2. The LGD values are fixed by scenario rather than estimated from recovery data.
3. The simulation does not estimate regulatory capital.
4. Correlation between defaults is not explicitly modelled.
