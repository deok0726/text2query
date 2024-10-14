WITH monthly_discounts AS (
    SELECT strftime(
            '%Y-%m',
            substr(SL_DT, 1, 4) || '-' || substr(SL_DT, 5, 2) || '-' || substr(SL_DT, 7, 2)
        ) AS month,
        SUM(BLL_SV_AM) AS total_discount
    FROM WBM_T_BLL_SPEC_IZ
    WHERE ACCTNO = '70018819695'
        AND SL_DT BETWEEN strftime('%Y%m%d', date('now', 'start of month')) AND strftime('%Y%m%d', date('now', 'start of month', '+1 month', '-1 day'))
        AND BLL_MC_NM LIKE '%스타벅스%'
        AND BLL_SV_DC IN (
            SELECT SV_C
            FROM WPD_T_SV_SNGL_PRP_INF
        )
    GROUP BY strftime(
            '%Y-%m',
            substr(SL_DT, 1, 4) || '-' || substr(SL_DT, 5, 2) || '-' || substr(SL_DT, 7, 2)
        )
),
total_spent AS (
    SELECT SUM(COALESCE(BIL_PRN, SL_AM)) AS total_spent
    FROM WBM_T_BLL_SPEC_IZ
    WHERE ACCTNO = '70018819695'
        AND SL_DT BETWEEN '20240701' AND '20240930'
        AND BLL_MC_NM LIKE '%스타벅스%'
),
total_discounted AS (
    SELECT SUM(BLL_SV_AM) AS total_discounted
    FROM WBM_T_BLL_SPEC_IZ
    WHERE ACCTNO = '70018819695'
        AND SL_DT BETWEEN '20240701' AND '20240930'
        AND BLL_MC_NM LIKE '%스타벅스%'
        AND BLL_SV_DC IN (
            SELECT SV_C
            FROM WPD_T_SV_SNGL_PRP_INF
        )
)
SELECT (
        SELECT total_spent
        FROM total_spent
    ) AS total_spent,
    (
        SELECT total_discounted
        FROM total_discounted
    ) AS total_discounted,
    (
        SELECT MLIM_AM
        FROM WPD_T_SV_SNGL_PRP_INF
        WHERE SV_C = 'SP03608'
    ) - COALESCE(
        (
            SELECT total_discount
            FROM monthly_discounts
            WHERE month = strftime('%Y-%m', 'now')
        ),
        0
    ) AS remaining_discount_limit;