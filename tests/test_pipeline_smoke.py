from __future__ import annotations

import numpy as np
import pandas as pd

import risk_signals_credit_lab as lab


def test_customer_data_generation_has_expected_fields() -> None:
    raw = lab.make_customer_data(rows=150, seed=7)
    model_table = lab.add_business_fields(raw)

    expected_columns = {
        "customer_id",
        "statement_month",
        "limit_balance",
        "utilization_ratio",
        "payment_ratio",
        "age_group",
        "credit_limit_band",
        "default_next_month",
    }

    assert len(model_table) == 150
    assert expected_columns.issubset(model_table.columns)
    assert model_table["customer_id"].is_unique
    assert model_table["default_next_month"].isin([0, 1]).all()
    assert model_table.isna().sum().sum() == 0


def test_feature_split_and_logistic_predictions_are_valid() -> None:
    raw = lab.make_customer_data(rows=260, seed=11)
    model_table = lab.add_business_fields(raw)
    features = lab.build_feature_data(model_table, seed=11)

    weights = lab.fit_logistic_regression(features.x_train, features.y_train)
    probabilities = lab.predict_logistic(weights, features.x_test)

    assert len(features.train_index) + len(features.test_index) == len(model_table)
    assert features.x_train.shape[1] == features.x_test.shape[1]
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert probabilities.std() > 0


def test_auc_is_stable_when_scores_have_ties() -> None:
    y_true = np.array([1, 0, 1, 0, 1, 0], dtype=float)
    y_score = np.array([0.8, 0.8, 0.4, 0.4, 0.4, 0.1], dtype=float)
    order = np.array([1, 0, 3, 4, 2, 5])

    assert lab.auc_score(y_true, y_score) == lab.auc_score(y_true[order], y_score[order])


def test_validation_and_sql_outputs_are_written(tmp_path, monkeypatch) -> None:
    raw = lab.make_customer_data(rows=120, seed=3)
    model_table = lab.add_business_fields(raw)

    monkeypatch.setattr(lab, "DATA_PROCESSED", tmp_path)
    lab.write_validation_checks(model_table)
    lab.run_sql_outputs(model_table)

    expected_files = [
        "validation_checks.csv",
        "sql_education_segments.csv",
        "sql_age_segments.csv",
        "sql_limit_segments.csv",
        "sql_portfolio_summary.csv",
    ]

    for file_name in expected_files:
        assert (tmp_path / file_name).exists()

    validation = pd.read_csv(tmp_path / "validation_checks.csv")
    portfolio = pd.read_csv(tmp_path / "sql_portfolio_summary.csv")

    assert validation.loc[validation["check_name"] == "rows", "value"].iloc[0] == 120
    assert int(portfolio["customers"].iloc[0]) == 120
