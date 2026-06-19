from __future__ import annotations

import html
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"
SQL_DIR = ROOT / "sql"


def ensure_directories() -> None:
    for path in [DATA_RAW, DATA_PROCESSED, FIGURES, REPORTS]:
        path.mkdir(parents=True, exist_ok=True)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -35, 35)
    return 1.0 / (1.0 + np.exp(-values))


def make_customer_data(rows: int = 3200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    customer_id = np.arange(1, rows + 1)
    statement_months = pd.date_range("2025-01-01", periods=12, freq="MS").strftime("%Y-%m")
    statement_month = rng.choice(statement_months, size=rows, replace=True)

    sex = rng.choice(["female", "male"], size=rows, p=[0.54, 0.46])
    education = rng.choice(
        ["graduate", "university", "high_school", "other"],
        size=rows,
        p=[0.28, 0.44, 0.20, 0.08],
    )
    marital_status = rng.choice(
        ["single", "married", "other"],
        size=rows,
        p=[0.47, 0.45, 0.08],
    )

    age = np.clip(rng.normal(38, 10, size=rows).round(), 21, 72).astype(int)
    income_base = rng.lognormal(mean=10.65, sigma=0.42, size=rows)
    education_income = np.select(
        [education == "graduate", education == "university", education == "other"],
        [1.25, 1.08, 0.82],
        default=0.92,
    )
    annual_income = np.clip(income_base * education_income, 18000, 180000).round(0)

    limit_noise = rng.normal(1.0, 0.22, size=rows)
    limit_balance = np.clip((annual_income * rng.uniform(1.1, 2.4, size=rows) * limit_noise), 5000, 260000)
    limit_balance = (np.round(limit_balance / 1000) * 1000).astype(float)

    base_utilization = rng.beta(2.4, 3.0, size=rows)
    stress_trait = rng.normal(0, 1, size=rows)
    utilization_target = np.clip(base_utilization + 0.08 * stress_trait, 0.03, 0.98)

    delay_lambda = np.clip(0.35 + 1.2 * utilization_target + 0.24 * stress_trait, 0.05, 2.6)
    pay_status = []
    for _ in range(6):
        raw_delay = rng.poisson(delay_lambda)
        late_flag = rng.binomial(1, np.clip(0.18 + utilization_target * 0.45, 0.05, 0.82))
        pay_status.append(np.clip(raw_delay * late_flag, 0, 5))
    pay_status = np.array(pay_status).T

    bill_amounts = []
    pay_amounts = []
    for month in range(6):
        seasonal_noise = rng.normal(1.0, 0.12, size=rows)
        bill = np.clip(limit_balance * utilization_target * seasonal_noise, 0, limit_balance * 1.05)
        delay_effect = np.clip(1.0 - 0.08 * pay_status[:, month], 0.35, 1.0)
        payment = np.clip(bill * rng.uniform(0.12, 0.50, size=rows) * delay_effect, 0, bill)
        bill_amounts.append(bill.round(0))
        pay_amounts.append(payment.round(0))
    bill_amounts = np.array(bill_amounts).T
    pay_amounts = np.array(pay_amounts).T

    months_with_delay = (pay_status > 0).sum(axis=1)
    average_bill = bill_amounts.mean(axis=1)
    utilization_ratio = np.clip(average_bill / np.maximum(limit_balance, 1), 0, 1.2)
    payment_ratio = pay_amounts.sum(axis=1) / np.maximum(bill_amounts.sum(axis=1), 1)

    education_risk = np.select(
        [education == "graduate", education == "university", education == "other"],
        [-0.22, 0.02, 0.18],
        default=0.12,
    )
    age_risk = np.clip((age - 42) / 35, -0.45, 0.55)
    logit = (
        -3.60
        + 2.85 * utilization_ratio
        + 0.52 * months_with_delay
        + 0.24 * pay_status[:, 0]
        - 0.000006 * limit_balance
        - 0.80 * payment_ratio
        + education_risk
        + 0.24 * age_risk
        + rng.normal(0, 0.18, size=rows)
    )
    default_probability = sigmoid(logit)
    default_next_month = rng.binomial(1, default_probability)

    data = {
        "customer_id": customer_id,
        "statement_month": statement_month,
        "limit_balance": limit_balance,
        "annual_income": annual_income,
        "sex": sex,
        "education": education,
        "marital_status": marital_status,
        "age": age,
        "utilization_ratio": utilization_ratio.round(4),
        "payment_ratio": payment_ratio.round(4),
        "months_with_delay": months_with_delay.astype(int),
        "default_next_month": default_next_month.astype(int),
    }
    for i in range(6):
        data[f"pay_status_{i + 1}"] = pay_status[:, i].astype(int)
    for i in range(6):
        data[f"bill_amt_{i + 1}"] = bill_amounts[:, i]
    for i in range(6):
        data[f"pay_amt_{i + 1}"] = pay_amounts[:, i]

    df = pd.DataFrame(data)
    return df


def add_business_fields(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["age_group"] = pd.cut(
        output["age"],
        bins=[20, 29, 39, 49, 59, 80],
        labels=["21_to_29", "30_to_39", "40_to_49", "50_to_59", "60_plus"],
        include_lowest=True,
    ).astype(str)
    output["credit_limit_band"] = pd.cut(
        output["limit_balance"],
        bins=[0, 30000, 70000, 120000, 180000, np.inf],
        labels=["very_low", "low", "medium", "high", "very_high"],
        include_lowest=True,
    ).astype(str)
    output["average_bill_amount"] = output[[f"bill_amt_{i}" for i in range(1, 7)]].mean(axis=1).round(2)
    output["average_payment_amount"] = output[[f"pay_amt_{i}" for i in range(1, 7)]].mean(axis=1).round(2)
    output["recent_delay_flag"] = (output["pay_status_1"] > 0).astype(int)
    output["high_utilization_flag"] = (output["utilization_ratio"] >= 0.80).astype(int)
    output["income_to_limit"] = (output["annual_income"] / np.maximum(output["limit_balance"], 1)).round(4)
    ordered_columns = [
        "customer_id",
        "statement_month",
        "limit_balance",
        "annual_income",
        "sex",
        "education",
        "marital_status",
        "age",
        "age_group",
        "credit_limit_band",
        "utilization_ratio",
        "payment_ratio",
        "months_with_delay",
        "recent_delay_flag",
        "high_utilization_flag",
        "income_to_limit",
        "average_bill_amount",
        "average_payment_amount",
    ]
    ordered_columns += [f"pay_status_{i}" for i in range(1, 7)]
    ordered_columns += [f"bill_amt_{i}" for i in range(1, 7)]
    ordered_columns += [f"pay_amt_{i}" for i in range(1, 7)]
    ordered_columns += ["default_next_month"]
    return output[ordered_columns]


def write_validation_checks(df: pd.DataFrame) -> None:
    checks = pd.DataFrame(
        [
            {"check_name": "rows", "value": len(df)},
            {"check_name": "duplicate_customer_ids", "value": int(df["customer_id"].duplicated().sum())},
            {"check_name": "missing_values", "value": int(df.isna().sum().sum())},
            {"check_name": "observed_default_rate", "value": round(float(df["default_next_month"].mean()), 4)},
            {"check_name": "average_limit_balance", "value": round(float(df["limit_balance"].mean()), 2)},
            {"check_name": "average_utilization", "value": round(float(df["utilization_ratio"].mean()), 4)},
        ]
    )
    checks.to_csv(DATA_PROCESSED / "validation_checks.csv", index=False)


def run_sql_outputs(df: pd.DataFrame) -> None:
    connection = sqlite3.connect(":memory:")
    df.to_sql("customer_risk_table", connection, index=False, if_exists="replace")

    segment_query = """
        SELECT
            education,
            COUNT(*) AS customers,
            ROUND(AVG(default_next_month), 3) AS observed_default_rate,
            ROUND(AVG(limit_balance), 0) AS average_limit,
            ROUND(AVG(utilization_ratio), 3) AS average_utilization
        FROM customer_risk_table
        GROUP BY education
        ORDER BY observed_default_rate DESC
    """
    age_query = """
        SELECT
            age_group,
            COUNT(*) AS customers,
            ROUND(AVG(default_next_month), 3) AS observed_default_rate,
            ROUND(AVG(limit_balance), 0) AS average_limit,
            ROUND(AVG(months_with_delay), 2) AS average_months_with_delay
        FROM customer_risk_table
        GROUP BY age_group
        ORDER BY observed_default_rate DESC
    """
    limit_query = """
        SELECT
            credit_limit_band,
            COUNT(*) AS customers,
            ROUND(AVG(default_next_month), 3) AS observed_default_rate,
            ROUND(AVG(limit_balance), 0) AS average_limit,
            ROUND(AVG(utilization_ratio), 3) AS average_utilization
        FROM customer_risk_table
        GROUP BY credit_limit_band
        ORDER BY observed_default_rate DESC
    """
    portfolio_query = """
        SELECT
            COUNT(*) AS customers,
            ROUND(SUM(limit_balance), 0) AS total_limit_balance,
            ROUND(AVG(limit_balance), 0) AS average_limit_balance,
            ROUND(AVG(utilization_ratio), 3) AS average_utilization,
            ROUND(AVG(default_next_month), 3) AS observed_default_rate
        FROM customer_risk_table
    """

    pd.read_sql_query(segment_query, connection).to_csv(DATA_PROCESSED / "sql_education_segments.csv", index=False)
    pd.read_sql_query(age_query, connection).to_csv(DATA_PROCESSED / "sql_age_segments.csv", index=False)
    pd.read_sql_query(limit_query, connection).to_csv(DATA_PROCESSED / "sql_limit_segments.csv", index=False)
    pd.read_sql_query(portfolio_query, connection).to_csv(DATA_PROCESSED / "sql_portfolio_summary.csv", index=False)
    connection.close()


@dataclass
class FeatureData:
    feature_names: list[str]
    train_index: np.ndarray
    test_index: np.ndarray
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    x_all: np.ndarray
    means: np.ndarray
    stds: np.ndarray


def build_feature_data(df: pd.DataFrame, seed: int = 42) -> FeatureData:
    feature_columns = [
        "limit_balance",
        "annual_income",
        "age",
        "utilization_ratio",
        "payment_ratio",
        "months_with_delay",
        "recent_delay_flag",
        "high_utilization_flag",
        "income_to_limit",
        "average_bill_amount",
        "average_payment_amount",
        "sex",
        "education",
        "marital_status",
        "age_group",
        "credit_limit_band",
    ]
    feature_columns += [f"pay_status_{i}" for i in range(1, 7)]
    feature_columns += [f"bill_amt_{i}" for i in range(1, 7)]
    feature_columns += [f"pay_amt_{i}" for i in range(1, 7)]

    features = pd.get_dummies(df[feature_columns], drop_first=True, dtype=float)
    feature_names = list(features.columns)
    x_all_raw = features.to_numpy(dtype=float)
    y_all = df["default_next_month"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(df))
    test_size = int(round(len(df) * 0.30))
    test_index = np.sort(shuffled[:test_size])
    train_index = np.sort(shuffled[test_size:])

    x_train_raw = x_all_raw[train_index]
    x_test_raw = x_all_raw[test_index]
    means = x_train_raw.mean(axis=0)
    stds = x_train_raw.std(axis=0)
    stds[stds == 0] = 1.0

    x_train = (x_train_raw - means) / stds
    x_test = (x_test_raw - means) / stds
    x_all = (x_all_raw - means) / stds
    return FeatureData(
        feature_names=feature_names,
        train_index=train_index,
        test_index=test_index,
        x_train=x_train,
        x_test=x_test,
        y_train=y_all[train_index],
        y_test=y_all[test_index],
        x_all=x_all,
        means=means,
        stds=stds,
    )


def transform_feature_frame(df: pd.DataFrame, features: FeatureData) -> np.ndarray:
    feature_columns = [
        "limit_balance",
        "annual_income",
        "age",
        "utilization_ratio",
        "payment_ratio",
        "months_with_delay",
        "recent_delay_flag",
        "high_utilization_flag",
        "income_to_limit",
        "average_bill_amount",
        "average_payment_amount",
        "sex",
        "education",
        "marital_status",
        "age_group",
        "credit_limit_band",
    ]
    feature_columns += [f"pay_status_{i}" for i in range(1, 7)]
    feature_columns += [f"bill_amt_{i}" for i in range(1, 7)]
    feature_columns += [f"pay_amt_{i}" for i in range(1, 7)]

    encoded = pd.get_dummies(df[feature_columns], drop_first=True, dtype=float)
    encoded = encoded.reindex(columns=features.feature_names, fill_value=0.0)
    return (encoded.to_numpy(dtype=float) - features.means) / features.stds


def fit_logistic_regression(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    x_design = np.column_stack([np.ones(len(x_train)), x_train])
    weights = np.zeros(x_design.shape[1])
    learning_rate = 0.04
    regularization = 0.02
    for _ in range(2200):
        probability = sigmoid(x_design @ weights)
        gradient = x_design.T @ (probability - y_train) / len(y_train)
        gradient[1:] += regularization * weights[1:] / len(y_train)
        weights -= learning_rate * gradient
    return weights


def predict_logistic(weights: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    x_design = np.column_stack([np.ones(len(x_values)), x_values])
    return sigmoid(x_design @ weights)


def gini_impurity(y_values: np.ndarray) -> float:
    if len(y_values) == 0:
        return 0.0
    probability = float(y_values.mean())
    return 1.0 - probability * probability - (1.0 - probability) * (1.0 - probability)


def best_tree_split(
    x_values: np.ndarray,
    y_values: np.ndarray,
    feature_indices: np.ndarray,
    min_leaf: int,
) -> tuple[int | None, float | None, float]:
    best_feature = None
    best_threshold = None
    best_score = math.inf
    parent_count = len(y_values)
    for feature in feature_indices:
        column = x_values[:, feature]
        thresholds = np.unique(np.quantile(column, np.linspace(0.12, 0.88, 9)))
        for threshold in thresholds:
            left_mask = column <= threshold
            left_count = int(left_mask.sum())
            right_count = parent_count - left_count
            if left_count < min_leaf or right_count < min_leaf:
                continue
            left_y = y_values[left_mask]
            right_y = y_values[~left_mask]
            score = (left_count * gini_impurity(left_y) + right_count * gini_impurity(right_y)) / parent_count
            if score < best_score:
                best_score = score
                best_feature = int(feature)
                best_threshold = float(threshold)
    return best_feature, best_threshold, best_score


def build_tree(
    x_values: np.ndarray,
    y_values: np.ndarray,
    depth: int,
    max_depth: int,
    min_leaf: int,
    rng: np.random.Generator,
    max_features: int | None = None,
) -> dict:
    probability = float(y_values.mean()) if len(y_values) else 0.0
    if depth >= max_depth or len(y_values) < min_leaf * 2 or probability in {0.0, 1.0}:
        return {"type": "leaf", "probability": probability}

    feature_count = x_values.shape[1]
    if max_features is None or max_features >= feature_count:
        feature_indices = np.arange(feature_count)
    else:
        feature_indices = np.sort(rng.choice(feature_count, size=max_features, replace=False))

    feature, threshold, _ = best_tree_split(x_values, y_values, feature_indices, min_leaf)
    if feature is None or threshold is None:
        return {"type": "leaf", "probability": probability}

    left_mask = x_values[:, feature] <= threshold
    if left_mask.sum() < min_leaf or (~left_mask).sum() < min_leaf:
        return {"type": "leaf", "probability": probability}

    return {
        "type": "node",
        "probability": probability,
        "feature": feature,
        "threshold": threshold,
        "left": build_tree(x_values[left_mask], y_values[left_mask], depth + 1, max_depth, min_leaf, rng, max_features),
        "right": build_tree(x_values[~left_mask], y_values[~left_mask], depth + 1, max_depth, min_leaf, rng, max_features),
    }


def predict_tree(tree: dict, x_values: np.ndarray) -> np.ndarray:
    predictions = np.zeros(len(x_values))
    for row_number, row in enumerate(x_values):
        node = tree
        while node["type"] != "leaf":
            if row[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        predictions[row_number] = node["probability"]
    return predictions


def fit_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    tree_count: int = 35,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    forest = []
    max_features = max(3, int(math.sqrt(x_train.shape[1])))
    for _ in range(tree_count):
        sample_index = rng.integers(0, len(y_train), size=len(y_train))
        tree = build_tree(
            x_train[sample_index],
            y_train[sample_index],
            depth=0,
            max_depth=5,
            min_leaf=45,
            rng=rng,
            max_features=max_features,
        )
        forest.append(tree)
    return forest


def predict_random_forest(forest: list[dict], x_values: np.ndarray) -> np.ndarray:
    predictions = np.column_stack([predict_tree(tree, x_values) for tree in forest])
    return predictions.mean(axis=1)


def fit_regression_stump(x_values: np.ndarray, residuals: np.ndarray, min_leaf: int = 50) -> dict:
    best_feature = 0
    best_threshold = float(np.median(x_values[:, 0]))
    best_left = float(residuals.mean())
    best_right = float(residuals.mean())
    best_score = math.inf
    for feature in range(x_values.shape[1]):
        column = x_values[:, feature]
        thresholds = np.unique(np.quantile(column, np.linspace(0.15, 0.85, 7)))
        for threshold in thresholds:
            left_mask = column <= threshold
            left_count = int(left_mask.sum())
            right_count = len(residuals) - left_count
            if left_count < min_leaf or right_count < min_leaf:
                continue
            left_value = float(residuals[left_mask].mean())
            right_value = float(residuals[~left_mask].mean())
            prediction = np.where(left_mask, left_value, right_value)
            score = float(np.mean((residuals - prediction) ** 2))
            if score < best_score:
                best_score = score
                best_feature = int(feature)
                best_threshold = float(threshold)
                best_left = left_value
                best_right = right_value
    return {
        "feature": best_feature,
        "threshold": best_threshold,
        "left_value": best_left,
        "right_value": best_right,
    }


def fit_gradient_boosting(
    x_train: np.ndarray,
    y_train: np.ndarray,
    estimator_count: int = 90,
    learning_rate: float = 0.16,
) -> dict:
    base_rate = np.clip(y_train.mean(), 0.001, 0.999)
    base_logit = float(np.log(base_rate / (1.0 - base_rate)))
    current_score = np.full(len(y_train), base_logit)
    stumps = []
    for _ in range(estimator_count):
        probability = sigmoid(current_score)
        residuals = y_train - probability
        stump = fit_regression_stump(x_train, residuals)
        update = np.where(
            x_train[:, stump["feature"]] <= stump["threshold"],
            stump["left_value"],
            stump["right_value"],
        )
        current_score += learning_rate * update
        stumps.append(stump)
    return {"base_logit": base_logit, "learning_rate": learning_rate, "stumps": stumps}


def predict_gradient_boosting(model: dict, x_values: np.ndarray) -> np.ndarray:
    score = np.full(len(x_values), model["base_logit"])
    for stump in model["stumps"]:
        update = np.where(
            x_values[:, stump["feature"]] <= stump["threshold"],
            stump["left_value"],
            stump["right_value"],
        )
        score += model["learning_rate"] * update
    return sigmoid(score)


def predict_named_model(model_name: str, model_objects: dict, x_values: np.ndarray) -> np.ndarray:
    if model_name == "logistic_regression":
        return predict_logistic(model_objects["logistic_regression"], x_values)
    if model_name == "decision_tree":
        return predict_tree(model_objects["decision_tree"], x_values)
    if model_name == "random_forest":
        return predict_random_forest(model_objects["random_forest"], x_values)
    if model_name == "gradient_boosting":
        return predict_gradient_boosting(model_objects["gradient_boosting"], x_values)
    raise ValueError(f"Unknown model name: {model_name}")


def roc_curve_points(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"observed": y_true, "score": y_score})
    grouped = (
        frame.groupby("score", observed=True)
        .agg(
            positives=("observed", "sum"),
            customers=("observed", "size"),
        )
        .reset_index()
        .sort_values("score", ascending=False)
    )
    positives = float(frame["observed"].sum())
    negatives = float(len(frame) - positives)
    true_positive = grouped["positives"].cumsum().to_numpy(dtype=float)
    false_positive = (grouped["customers"] - grouped["positives"]).cumsum().to_numpy(dtype=float)
    tpr = np.concatenate([[0.0], true_positive / max(positives, 1.0)])
    fpr = np.concatenate([[0.0], false_positive / max(negatives, 1.0)])
    return pd.DataFrame({"fpr": fpr, "tpr": tpr})


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    points = roc_curve_points(y_true, y_score)
    return float(np.trapezoid(points["tpr"], points["fpr"]))


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    prediction = (y_score >= threshold).astype(int)
    tp = int(((prediction == 1) & (y_true == 1)).sum())
    fp = int(((prediction == 1) & (y_true == 0)).sum())
    tn = int(((prediction == 0) & (y_true == 0)).sum())
    fn = int(((prediction == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "roc_auc": auc_score(y_true, y_score),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "brier_score": float(np.mean((y_score - y_true) ** 2)),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "threshold": threshold,
    }


def calibration_table(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"observed": y_true, "predicted": y_score})
    frame["bin"] = pd.qcut(frame["predicted"], q=bins, duplicates="drop")
    table = (
        frame.groupby("bin", observed=True)
        .agg(
            customers=("observed", "size"),
            average_predicted_pd=("predicted", "mean"),
            observed_default_rate=("observed", "mean"),
        )
        .reset_index(drop=True)
    )
    table.insert(0, "calibration_bin", np.arange(1, len(table) + 1))
    return table


def risk_band_summary(df: pd.DataFrame, predicted_pd: np.ndarray) -> pd.DataFrame:
    frame = df[["customer_id", "limit_balance", "default_next_month"]].copy()
    frame["predicted_pd"] = predicted_pd
    frame["risk_band"] = pd.qcut(
        frame["predicted_pd"],
        q=5,
        labels=["A", "B", "C", "D", "E"],
        duplicates="drop",
    ).astype(str)
    summary = (
        frame.groupby("risk_band", observed=True)
        .agg(
            customers=("customer_id", "count"),
            average_predicted_pd=("predicted_pd", "mean"),
            observed_default_rate=("default_next_month", "mean"),
            average_limit_balance=("limit_balance", "mean"),
        )
        .reset_index()
        .sort_values("risk_band")
    )
    return summary


def threshold_sensitivity(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.arange(0.10, 0.56, 0.05):
        metrics = classification_metrics(y_true, y_score, float(threshold))
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "flagged_share": float((y_score >= threshold).mean()),
            }
        )
    return pd.DataFrame(rows)


def run_models(df: pd.DataFrame, features: FeatureData) -> dict:
    logistic_weights = fit_logistic_regression(features.x_train, features.y_train)
    logistic_train = predict_logistic(logistic_weights, features.x_train)
    logistic_test = predict_logistic(logistic_weights, features.x_test)
    logistic_all = predict_logistic(logistic_weights, features.x_all)

    decision_tree = build_tree(
        features.x_train,
        features.y_train,
        depth=0,
        max_depth=5,
        min_leaf=50,
        rng=np.random.default_rng(101),
    )
    tree_train = predict_tree(decision_tree, features.x_train)
    tree_test = predict_tree(decision_tree, features.x_test)
    tree_all = predict_tree(decision_tree, features.x_all)

    forest = fit_random_forest(features.x_train, features.y_train)
    forest_train = predict_random_forest(forest, features.x_train)
    forest_test = predict_random_forest(forest, features.x_test)
    forest_all = predict_random_forest(forest, features.x_all)

    boosting = fit_gradient_boosting(features.x_train, features.y_train)
    boosting_train = predict_gradient_boosting(boosting, features.x_train)
    boosting_test = predict_gradient_boosting(boosting, features.x_test)
    boosting_all = predict_gradient_boosting(boosting, features.x_all)

    model_objects = {
        "logistic_regression": logistic_weights,
        "decision_tree": decision_tree,
        "random_forest": forest,
        "gradient_boosting": boosting,
    }

    model_predictions = {
        "logistic_regression": {"train": logistic_train, "test": logistic_test, "all": logistic_all},
        "decision_tree": {"train": tree_train, "test": tree_test, "all": tree_all},
        "random_forest": {"train": forest_train, "test": forest_test, "all": forest_all},
        "gradient_boosting": {"train": boosting_train, "test": boosting_test, "all": boosting_all},
    }

    metrics_rows = []
    threshold = 0.30
    for model_name, prediction_group in model_predictions.items():
        metrics = classification_metrics(features.y_test, prediction_group["test"], threshold)
        metrics["model"] = model_name
        metrics_rows.append(metrics)
    metrics_table = pd.DataFrame(metrics_rows)[
        [
            "model",
            "roc_auc",
            "precision",
            "recall",
            "f1_score",
            "brier_score",
            "true_positive",
            "false_positive",
            "true_negative",
            "false_negative",
            "threshold",
        ]
    ].sort_values(["roc_auc", "brier_score"], ascending=[False, True])
    metrics_table.to_csv(DATA_PROCESSED / "model_metrics.csv", index=False)
    selected_model = str(metrics_table.iloc[0]["model"])

    score_frame = df[["customer_id", "statement_month", "limit_balance", "default_next_month"]].copy()
    score_frame["split"] = "train"
    score_frame.loc[features.test_index, "split"] = "test"
    for model_name, prediction_group in model_predictions.items():
        score_frame[f"pd_{model_name}"] = prediction_group["all"]
    score_frame["selected_model"] = selected_model
    score_frame["predicted_pd"] = model_predictions[selected_model]["all"]
    score_frame["risk_band"] = pd.qcut(
        score_frame["predicted_pd"],
        q=5,
        labels=["A", "B", "C", "D", "E"],
        duplicates="drop",
    ).astype(str)
    score_frame.to_csv(DATA_PROCESSED / "model_scores.csv", index=False)

    calibration = calibration_table(features.y_test, model_predictions[selected_model]["test"])
    calibration.to_csv(DATA_PROCESSED / "calibration_table.csv", index=False)

    bands = risk_band_summary(df, model_predictions[selected_model]["all"])
    bands.to_csv(DATA_PROCESSED / "risk_band_summary.csv", index=False)

    sensitivity = threshold_sensitivity(features.y_test, model_predictions[selected_model]["test"])
    sensitivity.to_csv(DATA_PROCESSED / "threshold_sensitivity.csv", index=False)

    roc_points = roc_curve_points(features.y_test, model_predictions[selected_model]["test"])
    roc_points.to_csv(DATA_PROCESSED / "roc_curve_points.csv", index=False)

    train_test_summary = pd.DataFrame(
        [
            {
                "split": "train",
                "customers": len(features.y_train),
                "observed_default_rate": float(features.y_train.mean()),
            },
            {
                "split": "test",
                "customers": len(features.y_test),
                "observed_default_rate": float(features.y_test.mean()),
            },
        ]
    )
    train_test_summary.to_csv(DATA_PROCESSED / "train_test_split_summary.csv", index=False)

    explainability = build_explainability_summary(logistic_weights, features)
    explainability.to_csv(DATA_PROCESSED / "explainability_summary.csv", index=False)

    return {
        "selected_model": selected_model,
        "metrics_table": metrics_table,
        "score_frame": score_frame,
        "calibration": calibration,
        "risk_bands": bands,
        "sensitivity": sensitivity,
        "roc_points": roc_points,
        "explainability": explainability,
        "confusion_metrics": classification_metrics(features.y_test, model_predictions[selected_model]["test"], threshold),
        "model_objects": model_objects,
    }


def apply_counterfactual_actions(row: pd.Series, action: str) -> pd.Series:
    changed = row.copy()
    if action in {"lower_utilization", "combined"}:
        target_utilization = min(float(changed["utilization_ratio"]), 0.55)
        if changed["utilization_ratio"] > 0:
            scale = target_utilization / float(changed["utilization_ratio"])
        else:
            scale = 1.0
        changed["utilization_ratio"] = target_utilization
        changed["high_utilization_flag"] = int(target_utilization >= 0.80)
        for i in range(1, 7):
            changed[f"bill_amt_{i}"] = round(float(changed[f"bill_amt_{i}"]) * scale, 0)
        changed["average_bill_amount"] = round(
            float(np.mean([changed[f"bill_amt_{i}"] for i in range(1, 7)])),
            2,
        )

    if action in {"improve_repayment", "combined"}:
        for i in range(1, 7):
            changed[f"pay_status_{i}"] = 0
        changed["months_with_delay"] = int(sum(int(changed[f"pay_status_{i}"]) > 0 for i in range(1, 7)))
        changed["recent_delay_flag"] = int(changed["pay_status_1"] > 0)
        changed["payment_ratio"] = max(float(changed["payment_ratio"]), 0.45)
        for i in range(1, 7):
            minimum_payment = float(changed[f"bill_amt_{i}"]) * 0.45
            changed[f"pay_amt_{i}"] = round(max(float(changed[f"pay_amt_{i}"]), minimum_payment), 0)
        changed["average_payment_amount"] = round(
            float(np.mean([changed[f"pay_amt_{i}"] for i in range(1, 7)])),
            2,
        )

    return changed


def create_counterfactual_outputs(df: pd.DataFrame, features: FeatureData, outputs: dict) -> pd.DataFrame:
    selected_model = outputs["selected_model"]
    model_objects = outputs["model_objects"]
    score_frame = outputs["score_frame"]
    candidate_ids = (
        score_frame.loc[score_frame["risk_band"] == "E"]
        .sort_values("predicted_pd", ascending=False)
        .head(8)["customer_id"]
        .tolist()
    )

    action_labels = {
        "lower_utilization": "Lower utilisation to 55 percent where possible",
        "improve_repayment": "Improve recent repayment pattern and payment ratio",
        "combined": "Combine lower utilisation with improved repayment behaviour",
    }
    rows = []
    for customer_id in candidate_ids:
        original = df.loc[df["customer_id"] == customer_id].iloc[0]
        baseline_pd = float(score_frame.loc[score_frame["customer_id"] == customer_id, "predicted_pd"].iloc[0])
        for action, label in action_labels.items():
            changed = apply_counterfactual_actions(original, action)
            transformed = transform_feature_frame(pd.DataFrame([changed]), features)
            counterfactual_pd = float(predict_named_model(selected_model, model_objects, transformed)[0])
            rows.append(
                {
                    "customer_id": int(customer_id),
                    "selected_model": selected_model,
                    "baseline_pd": baseline_pd,
                    "counterfactual_action": action,
                    "counterfactual_description": label,
                    "counterfactual_pd": counterfactual_pd,
                    "pd_reduction": baseline_pd - counterfactual_pd,
                    "starting_utilization_ratio": float(original["utilization_ratio"]),
                    "counterfactual_utilization_ratio": float(changed["utilization_ratio"]),
                    "starting_months_with_delay": int(original["months_with_delay"]),
                    "counterfactual_months_with_delay": int(changed["months_with_delay"]),
                    "starting_payment_ratio": float(original["payment_ratio"]),
                    "counterfactual_payment_ratio": float(changed["payment_ratio"]),
                }
            )

    counterfactuals = pd.DataFrame(rows).sort_values(["customer_id", "counterfactual_action"])
    counterfactuals.to_csv(DATA_PROCESSED / "counterfactual_examples.csv", index=False)
    return counterfactuals


def build_explainability_summary(weights: np.ndarray, features: FeatureData) -> pd.DataFrame:
    coefficients = weights[1:]
    contribution = np.abs(features.x_all * coefficients)
    importance = contribution.mean(axis=0)
    rows = []
    for feature_name, coefficient, score in zip(features.feature_names, coefficients, importance):
        direction = "increases_risk" if coefficient > 0 else "reduces_risk"
        rows.append(
            {
                "feature": feature_name,
                "importance": float(score),
                "coefficient": float(coefficient),
                "direction": direction,
            }
        )
    return pd.DataFrame(rows).sort_values("importance", ascending=False).head(15)


def create_time_series_outputs(score_frame: pd.DataFrame) -> pd.DataFrame:
    trend = (
        score_frame.groupby("statement_month")
        .agg(
            customers=("customer_id", "count"),
            average_predicted_pd=("predicted_pd", "mean"),
            observed_default_rate=("default_next_month", "mean"),
            total_limit_balance=("limit_balance", "sum"),
        )
        .reset_index()
        .sort_values("statement_month")
    )
    trend["rolling_predicted_pd"] = trend["average_predicted_pd"].rolling(3, min_periods=1).mean()
    trend["baseline_pd"] = trend["average_predicted_pd"]
    trend["mild_stress_pd"] = np.clip(trend["average_predicted_pd"] * 1.10, 0, 1)
    trend["severe_stress_pd"] = np.clip(trend["average_predicted_pd"] * 1.25, 0, 1)
    trend.to_csv(DATA_PROCESSED / "time_series_risk_signals.csv", index=False)
    return trend


def run_monte_carlo(score_frame: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    base_pd = score_frame["predicted_pd"].to_numpy()
    ead = np.maximum(score_frame["limit_balance"].to_numpy() * 0.45, 1000.0)
    scenarios = [
        {"scenario": "baseline", "pd_multiplier": 1.00, "lgd": 0.45},
        {"scenario": "mild_stress", "pd_multiplier": 1.10, "lgd": 0.50},
        {"scenario": "severe_stress", "pd_multiplier": 1.25, "lgd": 0.55},
    ]
    summary_rows = []
    distribution_rows = []
    simulations = 10000
    chunk_size = 500
    for scenario in scenarios:
        scenario_pd = np.clip(base_pd * scenario["pd_multiplier"], 0, 0.95)
        loss_weight = ead * scenario["lgd"]
        losses = []
        remaining = simulations
        while remaining > 0:
            current = min(chunk_size, remaining)
            default_events = rng.random((current, len(scenario_pd))) < scenario_pd
            chunk_losses = default_events @ loss_weight
            losses.append(chunk_losses)
            remaining -= current
        losses = np.concatenate(losses)
        var_95 = float(np.quantile(losses, 0.95))
        cvar_95 = float(losses[losses >= var_95].mean())
        expected_loss = float(losses.mean())
        summary_rows.append(
            {
                "scenario": scenario["scenario"],
                "pd_multiplier": scenario["pd_multiplier"],
                "lgd": scenario["lgd"],
                "expected_loss": expected_loss,
                "value_at_risk_95": var_95,
                "conditional_value_at_risk_95": cvar_95,
                "value_at_risk_99": float(np.quantile(losses, 0.99)),
                "expected_loss_rate": expected_loss / float(ead.sum()),
            }
        )
        distribution_rows.extend(
            {
                "scenario": scenario["scenario"],
                "simulation": int(i + 1),
                "loss": float(loss),
            }
            for i, loss in enumerate(losses)
        )

    summary = pd.DataFrame(summary_rows)
    distribution = pd.DataFrame(distribution_rows)
    summary.to_csv(DATA_PROCESSED / "stress_test_summary.csv", index=False)
    distribution.to_csv(DATA_PROCESSED / "monte_carlo_loss_distribution.csv", index=False)
    return summary, distribution


def svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        "<style>"
        "text{font-family:Arial,Helvetica,sans-serif;fill:#27313f}"
        ".title{font-size:24px;font-weight:700}"
        ".label{font-size:13px}"
        ".axis-title{font-size:16px;font-weight:700;fill:#27313f}"
        ".small{font-size:11px;fill:#566474}"
        ".axis{stroke:#8d99a8;stroke-width:1}"
        ".grid{stroke:#d9dee7;stroke-width:1}"
        "</style>"
    )


def svg_text(x: float, y: float, text: str, class_name: str = "label", anchor: str = "start") -> str:
    escaped = html.escape(str(text))
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{class_name}" text-anchor="{anchor}">{escaped}</text>'


def svg_rotated_text(x: float, y: float, text: str, class_name: str = "axis-title", anchor: str = "middle") -> str:
    escaped = html.escape(str(text))
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" class="{class_name}" text-anchor="{anchor}" '
        f'transform="rotate(-90 {x:.1f} {y:.1f})">{escaped}</text>'
    )


def write_bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    x_axis_label: str,
    y_axis_label: str,
    color: str = "#2474a6",
    tick_format: str = "{:.2f}",
    value_format: str = "{:.2f}",
) -> None:
    width, height = 920, 520
    margin_left, margin_right, margin_top, margin_bottom = 96, 42, 70, 118
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    max_value = max(values) * 1.20 if values else 1
    bar_gap = 18
    bar_width = (chart_width - bar_gap * (len(values) - 1)) / max(len(values), 1)
    parts = [svg_header(width, height), svg_text(32, 38, title, "title")]
    for tick in np.linspace(0, max_value, 5):
        y = margin_top + chart_height - chart_height * tick / max_value
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" class="grid"/>')
        parts.append(svg_text(margin_left - 10, y + 4, tick_format.format(tick), "small", "end"))
    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_height}" class="axis"/>')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{width - margin_right}" y2="{margin_top + chart_height}" class="axis"/>')
    parts.append(svg_rotated_text(34, margin_top + chart_height / 2, y_axis_label))
    parts.append(svg_text(margin_left + chart_width / 2, height - 28, x_axis_label, "axis-title", "middle"))
    for i, (label, value) in enumerate(zip(labels, values)):
        x = margin_left + i * (bar_width + bar_gap)
        bar_height = chart_height * value / max_value
        y = margin_top + chart_height - bar_height
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="4" fill="{color}"/>')
        parts.append(svg_text(x + bar_width / 2, y - 8, value_format.format(value), "small", "middle"))
        parts.append(svg_text(x + bar_width / 2, margin_top + chart_height + 26, label, "small", "middle"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_grouped_bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    first_values: list[float],
    second_values: list[float],
    first_label: str,
    second_label: str,
    x_axis_label: str,
    y_axis_label: str,
    tick_format: str = "{:.2f}",
) -> None:
    width, height = 940, 540
    margin_left, margin_right, margin_top, margin_bottom = 96, 46, 78, 116
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    max_value = max(first_values + second_values) * 1.22
    group_gap = 28
    group_width = (chart_width - group_gap * (len(labels) - 1)) / max(len(labels), 1)
    bar_width = group_width * 0.40
    parts = [svg_header(width, height), svg_text(32, 38, title, "title")]
    parts.append(f'<rect x="660" y="24" width="14" height="14" fill="#2474a6"/>')
    parts.append(svg_text(682, 36, first_label, "small"))
    parts.append(f'<rect x="790" y="24" width="14" height="14" fill="#d95f36"/>')
    parts.append(svg_text(812, 36, second_label, "small"))
    for tick in np.linspace(0, max_value, 5):
        y = margin_top + chart_height - chart_height * tick / max_value
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" class="grid"/>')
        parts.append(svg_text(margin_left - 10, y + 4, tick_format.format(tick), "small", "end"))
    parts.append(f'<line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{width - margin_right}" y2="{margin_top + chart_height}" class="axis"/>')
    parts.append(svg_rotated_text(34, margin_top + chart_height / 2, y_axis_label))
    parts.append(svg_text(margin_left + chart_width / 2, height - 28, x_axis_label, "axis-title", "middle"))
    for i, label in enumerate(labels):
        x_group = margin_left + i * (group_width + group_gap)
        for j, (value, color) in enumerate([(first_values[i], "#2474a6"), (second_values[i], "#d95f36")]):
            x = x_group + j * (bar_width + 5)
            bar_height = chart_height * value / max_value
            y = margin_top + chart_height - bar_height
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="4" fill="{color}"/>')
        parts.append(svg_text(x_group + group_width / 2, margin_top + chart_height + 26, label, "small", "middle"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_line_chart(
    path: Path,
    title: str,
    x_values: list[float],
    y_values: list[float],
    x_axis_label: str,
    y_axis_label: str,
    diagonal: bool = False,
) -> None:
    width, height = 900, 520
    margin_left, margin_right, margin_top, margin_bottom = 90, 44, 70, 96
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    parts = [svg_header(width, height), svg_text(32, 38, title, "title")]
    for tick in np.linspace(0, 1, 6):
        x = margin_left + chart_width * tick
        y = margin_top + chart_height - chart_height * tick
        parts.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + chart_height}" class="grid"/>')
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" class="grid"/>')
        parts.append(svg_text(x, margin_top + chart_height + 24, f"{tick:.1f}", "small", "middle"))
        parts.append(svg_text(margin_left - 12, y + 4, f"{tick:.1f}", "small", "end"))
    parts.append(f'<line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{width - margin_right}" y2="{margin_top + chart_height}" class="axis"/>')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_height}" class="axis"/>')
    parts.append(svg_rotated_text(34, margin_top + chart_height / 2, y_axis_label))
    parts.append(svg_text(margin_left + chart_width / 2, height - 28, x_axis_label, "axis-title", "middle"))
    if diagonal:
        parts.append(
            f'<line x1="{margin_left}" y1="{margin_top + chart_height}" '
            f'x2="{margin_left + chart_width}" y2="{margin_top}" stroke="#8996a3" stroke-width="2" stroke-dasharray="6 5"/>'
        )
    points = []
    for x_value, y_value in zip(x_values, y_values):
        x = margin_left + chart_width * float(x_value)
        y = margin_top + chart_height - chart_height * float(y_value)
        points.append(f"{x:.1f},{y:.1f}")
    parts.append(f'<polyline fill="none" stroke="#2474a6" stroke-width="3" points="{" ".join(points)}"/>')
    for point in points[:: max(1, len(points) // 12)]:
        x, y = point.split(",")
        parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="#2474a6"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_horizontal_bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    x_axis_label: str,
    y_axis_label: str,
) -> None:
    width, height = 960, 560
    margin_left, margin_right, margin_top, margin_bottom = 250, 44, 70, 72
    chart_width = width - margin_left - margin_right
    row_height = (height - margin_top - margin_bottom) / max(len(labels), 1)
    max_value = max(values) * 1.18 if values else 1
    parts = [svg_header(width, height), svg_text(32, 38, title, "title")]
    parts.append(f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" class="axis"/>')
    parts.append(svg_rotated_text(34, margin_top + (height - margin_top - margin_bottom) / 2, y_axis_label))
    parts.append(svg_text(margin_left + chart_width / 2, height - 24, x_axis_label, "axis-title", "middle"))
    for i, (label, value) in enumerate(zip(labels, values)):
        y = margin_top + i * row_height + 8
        bar_width = chart_width * value / max_value
        parts.append(svg_text(margin_left - 12, y + row_height * 0.55, label[:30], "small", "end"))
        parts.append(f'<rect x="{margin_left}" y="{y:.1f}" width="{bar_width:.1f}" height="{row_height * 0.58:.1f}" rx="4" fill="#2474a6"/>')
        parts.append(svg_text(margin_left + bar_width + 8, y + row_height * 0.55, f"{value:.3f}", "small"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_histogram(path: Path, title: str, values: np.ndarray, x_axis_label: str, y_axis_label: str) -> None:
    counts, bin_edges = np.histogram(values, bins=28)
    labels = [f"{edge / 1_000_000:.1f}" for edge in bin_edges[:-1]]
    scaled_values = counts.astype(float).tolist()
    write_bar_chart(
        path,
        title,
        labels,
        scaled_values,
        x_axis_label,
        y_axis_label,
        color="#5a7f3b",
        tick_format="{:.0f}",
        value_format="{:.0f}",
    )


def write_confusion_matrix(path: Path, metrics: dict) -> None:
    width, height = 620, 560
    parts = [svg_header(width, height), svg_text(32, 38, "Confusion matrix at threshold 0.30", "title")]
    values = np.array(
        [
            [metrics["true_negative"], metrics["false_positive"]],
            [metrics["false_negative"], metrics["true_positive"]],
        ],
        dtype=float,
    )
    labels = [["True negative", "False positive"], ["False negative", "True positive"]]
    max_value = values.max()
    cell = 170
    start_x, start_y = 140, 110
    for row in range(2):
        for col in range(2):
            value = values[row, col]
            intensity = int(235 - 125 * value / max_value)
            color = f"rgb({intensity},{intensity + 10},235)"
            x = start_x + col * cell
            y = start_y + row * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="#ffffff" stroke-width="4"/>')
            parts.append(svg_text(x + cell / 2, y + 72, labels[row][col], "label", "middle"))
            parts.append(svg_text(x + cell / 2, y + 112, f"{int(value):,}", "title", "middle"))
    parts.append(svg_text(start_x + cell, start_y + cell * 2 + 38, "Predicted class", "axis-title", "middle"))
    parts.append(svg_rotated_text(54, start_y + cell, "Actual class"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def make_figures(outputs: dict, stress_summary: pd.DataFrame, distribution: pd.DataFrame) -> None:
    education = pd.read_csv(DATA_PROCESSED / "sql_education_segments.csv")
    write_bar_chart(
        FIGURES / "default_rate_by_segment.svg",
        "Observed default rate by education segment",
        education["education"].tolist(),
        education["observed_default_rate"].astype(float).tolist(),
        "Education segment",
        "Observed default rate",
    )

    roc = outputs["roc_points"]
    write_line_chart(
        FIGURES / "roc_curve.svg",
        "ROC curve for selected model",
        roc["fpr"].astype(float).tolist(),
        roc["tpr"].astype(float).tolist(),
        "False positive rate",
        "True positive rate",
        diagonal=True,
    )

    calibration = outputs["calibration"]
    write_line_chart(
        FIGURES / "calibration_curve.svg",
        "Calibration review",
        calibration["average_predicted_pd"].astype(float).tolist(),
        calibration["observed_default_rate"].astype(float).tolist(),
        "Average predicted PD",
        "Observed default rate",
        diagonal=True,
    )

    bands = outputs["risk_bands"]
    write_grouped_bar_chart(
        FIGURES / "risk_band_observed_default.svg",
        "Predicted and observed default by risk band",
        bands["risk_band"].astype(str).tolist(),
        bands["average_predicted_pd"].astype(float).tolist(),
        bands["observed_default_rate"].astype(float).tolist(),
        "Predicted PD",
        "Observed rate",
        "Risk band",
        "Default rate",
    )

    explainability = outputs["explainability"].head(10)
    write_horizontal_bar_chart(
        FIGURES / "explainability_feature_importance.svg",
        "Top model explanation signals",
        explainability["feature"].astype(str).tolist(),
        explainability["importance"].astype(float).tolist(),
        "Relative importance",
        "Feature",
    )

    baseline_losses = distribution.loc[distribution["scenario"] == "baseline", "loss"].to_numpy()
    write_histogram(
        FIGURES / "monte_carlo_loss_distribution.svg",
        "Monte Carlo baseline loss distribution",
        baseline_losses,
        "Portfolio loss (GBP millions)",
        "Simulation count",
    )

    write_grouped_bar_chart(
        FIGURES / "stress_scenario_losses.svg",
        "Expected loss and VaR by stress scenario",
        stress_summary["scenario"].astype(str).tolist(),
        (stress_summary["expected_loss"] / 1_000_000).astype(float).tolist(),
        (stress_summary["value_at_risk_95"] / 1_000_000).astype(float).tolist(),
        "Expected loss",
        "VaR 95",
        "Stress scenario",
        "Loss (GBP millions)",
    )

    write_confusion_matrix(FIGURES / "confusion_matrix.svg", outputs["confusion_metrics"])


def write_reports(
    outputs: dict,
    stress_summary: pd.DataFrame,
    trend: pd.DataFrame,
    counterfactuals: pd.DataFrame,
) -> None:
    metrics = outputs["metrics_table"].copy()
    best = metrics.iloc[0]
    bands = outputs["risk_bands"].copy()
    explainability = outputs["explainability"].copy()
    baseline_stress = stress_summary.loc[stress_summary["scenario"] == "baseline"].iloc[0]
    severe_stress = stress_summary.loc[stress_summary["scenario"] == "severe_stress"].iloc[0]

    model_lines = []
    for _, row in metrics.iterrows():
        model_lines.append(
            f"{row['model']}: AUC {row['roc_auc']:.3f}, Brier {row['brier_score']:.3f}, "
            f"precision {row['precision']:.3f}, recall {row['recall']:.3f}"
        )

    band_lines = []
    for _, row in bands.iterrows():
        band_lines.append(
            f"Band {row['risk_band']}: customers {int(row['customers'])}, "
            f"average predicted PD {row['average_predicted_pd']:.3f}, "
            f"observed default rate {row['observed_default_rate']:.3f}"
        )

    feature_lines = []
    for _, row in explainability.head(8).iterrows():
        feature_lines.append(
            f"{row['feature']}: importance {row['importance']:.3f}, direction {row['direction']}"
        )

    counterfactual_summary = (
        counterfactuals.groupby("counterfactual_action")
        .agg(
            average_starting_pd=("baseline_pd", "mean"),
            average_counterfactual_pd=("counterfactual_pd", "mean"),
            average_pd_reduction=("pd_reduction", "mean"),
        )
        .reset_index()
        .sort_values("average_pd_reduction", ascending=False)
    )
    counterfactual_lines = []
    for _, row in counterfactual_summary.iterrows():
        counterfactual_lines.append(
            f"{row['counterfactual_action']}: starting PD {row['average_starting_pd']:.3f}, "
            f"counterfactual PD {row['average_counterfactual_pd']:.3f}, "
            f"average reduction {row['average_pd_reduction']:.3f}"
        )

    model_report = f"""# Model Review Summary

## Objective

This report reviews a credit default modelling workflow for a generated customer portfolio. The focus is not only prediction quality, but also model review, calibration, explainability and business interpretation.

## Dataset overview

The modelling table contains {len(outputs['score_frame']):,} customers. The observed default rate in the full portfolio is {outputs['score_frame']['default_next_month'].mean():.3f}.

The train and test split is stored in `data/processed/train_test_split_summary.csv`.

## Model candidates

1. Logistic regression
2. Decision tree
3. Random forest
4. Gradient boosting

## Selected model

The selected model is `{outputs['selected_model']}` based on test AUC with Brier score used as a probability quality check.

## Performance summary

{chr(10).join(f'{i + 1}. {line}' for i, line in enumerate(model_lines))}

## Calibration review

The calibration review compares predicted probability of default against observed default rate across ten score buckets. The detailed output is stored in `data/processed/calibration_table.csv`.

The selected model has test AUC {best['roc_auc']:.3f} and Brier score {best['brier_score']:.3f}.

## Risk band review

{chr(10).join(f'{i + 1}. {line}' for i, line in enumerate(band_lines))}

The bands are intended for interpretation only. They are not a regulatory scorecard.

## Explainability summary

The lightweight pipeline creates a coefficient based explanation summary for the logistic baseline. The SHAP notebook extends this with package based model explanations when the optional dependency is installed.

{chr(10).join(f'{i + 1}. {line}' for i, line in enumerate(feature_lines))}

## DiCE and counterfactual explanation review

The project includes `notebooks/09_counterfactual_explanations.ipynb` and `data/processed/counterfactual_examples.csv`.

The notebook includes a DiCE example using `dice-ml` and a separate set of generated counterfactual examples from the main pipeline. These examples focus on selected high risk customers and compare baseline predicted default probability with practical feature changes. The combined scenario lowers utilisation and improves recent repayment behaviour.

{chr(10).join(f'{i + 1}. {line}' for i, line in enumerate(counterfactual_lines))}

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
"""
    (REPORTS / "model_review_summary.md").write_text(model_report, encoding="utf-8")

    stress_lines = []
    for _, row in stress_summary.iterrows():
        stress_lines.append(
            f"{row['scenario']}: expected loss {row['expected_loss']:,.0f}, "
            f"VaR 95 {row['value_at_risk_95']:,.0f}, "
            f"CVaR 95 {row['conditional_value_at_risk_95']:,.0f}"
        )
    var_lines = []
    for _, row in stress_summary.iterrows():
        scenario_name = str(row["scenario"]).replace("_", " ")
        var_lines.append(
            f"{scenario_name}: VaR 95 {row['value_at_risk_95']:,.0f}, "
            f"CVaR 95 {row['conditional_value_at_risk_95']:,.0f}, "
            f"VaR 99 {row['value_at_risk_99']:,.0f}"
        )

    trend_peak = trend.loc[trend["average_predicted_pd"].idxmax()]
    trend_peak_month = pd.Period(trend_peak["statement_month"], freq="M").strftime("%B %Y")

    stress_report = f"""# Portfolio Stress Testing Report

## Portfolio assumptions

The portfolio uses customer level predicted probability of default from the selected model. Exposure at default is approximated from credit limit and recent balance information. Loss given default is a scenario assumption.

## PD EAD and LGD setup

1. PD comes from the selected model output.
2. EAD is approximated as a share of credit limit.
3. LGD is set at 0.45 in baseline, 0.50 in mild stress and 0.55 in severe stress.

## Baseline loss estimate

Baseline expected loss is {baseline_stress['expected_loss']:,.0f}. Baseline VaR 95 is {baseline_stress['value_at_risk_95']:,.0f}.

## Mild and severe stress scenarios

{chr(10).join(f'{i + 1}. {line}' for i, line in enumerate(stress_lines))}

The severe stress scenario increases expected loss by {(severe_stress['expected_loss'] / baseline_stress['expected_loss'] - 1) * 100:.1f} percent compared with baseline.

## Time series risk signal

The highest monthly average predicted PD appears in {trend_peak_month} with an average predicted PD of {trend_peak['average_predicted_pd']:.3f}.

The time series view is scenario based because the generated study data is not a full macroeconomic panel.

## Monte Carlo loss distribution

The simulation runs 10000 portfolio loss draws for each scenario. The full distribution is stored in `data/processed/monte_carlo_loss_distribution.csv`.

## VaR and CVaR summary

{chr(10).join(f'{i + 1}. {line}' for i, line in enumerate(var_lines))}

## Interpretation

The simulation translates individual customer probability estimates into a portfolio view. This makes it easier to discuss expected loss and tail risk under documented assumptions.

## Limitations

1. The PD stress multipliers are simplified scenario assumptions.
2. The LGD values are fixed by scenario rather than estimated from recovery data.
3. The simulation does not estimate regulatory capital.
4. Correlation between defaults is not explicitly modelled.
"""
    (REPORTS / "portfolio_stress_testing_report.md").write_text(stress_report, encoding="utf-8")


def main() -> None:
    ensure_directories()
    raw = make_customer_data()
    raw.to_csv(DATA_RAW / "synthetic_credit_customers.csv", index=False)

    model_table = add_business_fields(raw)
    model_table.to_csv(DATA_PROCESSED / "credit_risk_model_table.csv", index=False)
    write_validation_checks(model_table)
    run_sql_outputs(model_table)

    features = build_feature_data(model_table)
    outputs = run_models(model_table, features)
    counterfactuals = create_counterfactual_outputs(model_table, features, outputs)
    trend = create_time_series_outputs(outputs["score_frame"])
    stress_summary, distribution = run_monte_carlo(outputs["score_frame"])
    make_figures(outputs, stress_summary, distribution)
    write_reports(outputs, stress_summary, trend, counterfactuals)

    print("Risk Signals Credit Lab outputs generated")
    print(f"Selected model: {outputs['selected_model']}")
    print(f"Figures: {FIGURES}")
    print(f"Reports: {REPORTS}")


if __name__ == "__main__":
    main()
