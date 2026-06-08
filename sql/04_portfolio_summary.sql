SELECT
    COUNT(*) AS customers,
    ROUND(SUM(limit_balance), 0) AS total_limit_balance,
    ROUND(AVG(limit_balance), 0) AS average_limit_balance,
    ROUND(AVG(utilization_ratio), 3) AS average_utilization,
    ROUND(AVG(default_next_month), 3) AS observed_default_rate
FROM customer_risk_table;

SELECT
    credit_limit_band,
    COUNT(*) AS customers,
    ROUND(SUM(limit_balance), 0) AS total_limit_balance,
    ROUND(AVG(default_next_month), 3) AS observed_default_rate
FROM customer_risk_table
GROUP BY credit_limit_band
ORDER BY total_limit_balance DESC;

