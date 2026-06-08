SELECT
    education,
    COUNT(*) AS customers,
    ROUND(AVG(default_next_month), 3) AS observed_default_rate,
    ROUND(AVG(limit_balance), 0) AS average_limit,
    ROUND(AVG(utilization_ratio), 3) AS average_utilization
FROM customer_risk_table
GROUP BY education
ORDER BY observed_default_rate DESC;

SELECT
    age_group,
    COUNT(*) AS customers,
    ROUND(AVG(default_next_month), 3) AS observed_default_rate,
    ROUND(AVG(limit_balance), 0) AS average_limit,
    ROUND(AVG(months_with_delay), 2) AS average_months_with_delay
FROM customer_risk_table
GROUP BY age_group
ORDER BY observed_default_rate DESC;

