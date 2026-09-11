SELECT
    search_id,
    -- We define the total search time once here
        MAX(CASE WHEN function = 'ROOT' THEN total_time_ms END) AS total_search_time,

    -- MAKEMOVE
    MAX(CASE WHEN function = 'MAKEMOVE' THEN total_time_ms END) AS make_move_total_ms,
    MAX(CASE WHEN function = 'MAKEMOVE' THEN total_time_ms / NULLIF(num_calls, 0) END) AS make_move_avg_ms,

    -- UnMakeMove
    MAX(CASE WHEN function = 'UNMAKE_MOVE' THEN total_time_ms END) AS unmake_move_total_ms,
    MAX(CASE WHEN function = 'UNMAKE_MOVE' THEN total_time_ms / NULLIF(num_calls, 0) END) AS unmake_move_avg_ms,

    -- Movegen
    MAX(CASE WHEN function = 'MOVEGEN' THEN total_time_ms END) AS movegen_total_ms,
    MAX(CASE WHEN function = 'MOVEGEN' THEN total_time_ms / NULLIF(num_calls, 0) END) AS movegen_avg_ms,

    -- Score_Order (Move Order)
    MAX(CASE WHEN function = 'SCORE_ORDER' THEN total_time_ms END) AS move_order_total_ms,
    MAX(CASE WHEN function = 'SCORE_ORDER' THEN total_time_ms / NULLIF(num_calls, 0) END) AS move_order_avg_ms,

    -- NNUE
    MAX(CASE WHEN function = 'NNUE' THEN total_time_ms END) AS nnue_total_ms,
    MAX(CASE WHEN function = 'NNUE' THEN total_time_ms / NULLIF(num_calls, 0) END) AS nnue_avg_ms,

    -- Eval (Static)
    MAX(CASE WHEN function = 'EVAL' THEN total_time_ms END) AS static_eval_total_ms,
    MAX(CASE WHEN function = 'EVAL' THEN total_time_ms / NULLIF(num_calls, 0) END) AS static_eval_avg_ms,

    -- SEE
    MAX(CASE WHEN function = 'SEE' THEN total_time_ms END) AS see_total_ms,
    MAX(CASE WHEN function = 'SEE' THEN total_time_ms / NULLIF(num_calls, 0) END) AS see_avg_ms,

    -- TT_PROBE
    MAX(CASE WHEN function = 'TT_PROBE' THEN total_time_ms END) AS tt_probe_total_ms,
    MAX(CASE WHEN function = 'TT_PROBE' THEN total_time_ms / NULLIF(num_calls, 0) END) AS tt_probe_avg_ms,

    -- TT_STORE
    MAX(CASE WHEN function = 'TT_STORE' THEN total_time_ms END) AS tt_store_total_ms,
    MAX(CASE WHEN function = 'TT_STORE' THEN total_time_ms / NULLIF(num_calls, 0) END) AS tt_store_avg_ms

FROM search_timings

GROUP BY search_id
