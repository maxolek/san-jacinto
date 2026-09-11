WITH base_metrics AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY search_id ORDER BY depth) -
        ROW_NUMBER() OVER (PARTITION BY search_id, move ORDER BY depth) AS move_grp
    FROM search_iteration_metrics
)
SELECT
    * EXCLUDE (move_grp),
    SUM(time_ms) OVER (PARTITION BY search_id ORDER BY depth ASC) AS running_time_ms,
    -- Across-iteration metrics are evaluated only when this view is requested.
    time_ms / NULLIF(LAG(time_ms) OVER (PARTITION BY search_id ORDER BY depth ASC), 0) as time_increase_ratio,
    total_nodes / NULLIF(LAG(total_nodes) OVER (PARTITION BY search_id ORDER BY depth ASC), 0) as ebf,
    qnodes / NULLIF(LAG(qnodes) OVER (PARTITION BY search_id ORDER BY depth ASC), 0) as qebf,

    -- stability metrics (use numeric eval)
    eval - LAG(eval) OVER (PARTITION BY search_id ORDER BY depth ASC) as prior_eval_delta,
    eval - FIRST_VALUE(eval) OVER (PARTITION BY search_id ORDER BY depth ASC) as first_eval_delta,
    CASE
        WHEN (eval > 0 AND LAG(eval) OVER (PARTITION BY search_id ORDER BY depth ASC) < 0) OR
             (eval < 0 AND LAG(eval) OVER (PARTITION BY search_id ORDER BY depth ASC) > 0) THEN 1
        ELSE 0
    END as eval_sign_flips,
    STDDEV(eval) OVER (PARTITION BY search_id ORDER BY depth ASC) AS stddev_eval,
    STDDEV(eval) OVER (PARTITION BY search_id ORDER BY depth ASC ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS stddev_last5_eval,

    ROW_NUMBER() OVER (PARTITION BY search_id, move_grp ORDER BY depth) - 1 AS move_stability

FROM base_metrics
