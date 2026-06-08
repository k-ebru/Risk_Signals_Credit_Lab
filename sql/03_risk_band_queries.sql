SELECT
    credit_limit_band,
    COUNT(*) AS customers,
    ROUND(AVG(default_next_month), 3) AS observed_default_rate,
    ROUND(AVG(limit_balance), 0) AS average_limit,
    ROUND(AVG(utilization_ratio), 3) AS average_utilization
FROM customer_risk_table
GROUP BY credit_limit_band
ORDER BY observed_default_rate DESC;

SELECT
    months_with_delay,
    COUNT(*) AS customers,
    ROUND(AVG(default_next_month), 3) AS observed_default_rate,
    ROUND(AVG(limit_balance), 0) AS average_limit
FROM customer_risk_table
GROUP BY months_with_delay
ORDER BY months_with_delay;

