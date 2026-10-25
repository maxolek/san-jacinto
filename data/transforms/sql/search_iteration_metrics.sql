SELECT
    search_id, depth, qdepth,
    time_ms,
    eval AS eval, move,
    -- evaluation without mate scores (+/-(100000 - plies)), and the mate side
    CASE WHEN abs(eval) < 99000 THEN eval END AS eval_cp,
    CASE WHEN eval >= 99000 THEN 1 WHEN eval <= -99000 THEN -1 WHEN eval IS NOT NULL THEN 0 END AS mate,
    nodes AS nodes, qnodes AS qnodes,
    tt_stores AS tt_stores, tt_hits AS tt_hits,
    fail_highs AS fail_highs, fail_lows AS fail_lows,
    fail_high_researches, fail_low_researches,
    fh_index_0, fh_index_1, fh_index_2, fh_index_3, fh_index_4to7, fh_index_8plus,
    see_prunes AS see_prunes, delta_prunes AS delta_prunes,
    pvs_researches AS pvs_researches,
    nmp AS nmp, nmp_failhigh AS nmp_failhigh,

    -- within iteration derived metrics
    total_nodes,
    total_nodes / (NULLIF(time_ms, 0) / 1000) AS nps,
    qnodes / NULLIF(total_nodes, 0) AS qratio,
    tt_hits / NULLIF(total_nodes, 0) AS tt_hit_ratio,
    tt_stores / NULLIF(total_nodes, 0) AS tt_store_ratio,
    fail_highs / NULLIF(total_nodes, 0) AS fail_high_ratio,
    fail_lows / NULLIF(total_nodes, 0) AS fail_low_ratio,
    fh_index_0 / NULLIF(fail_highs, 0) AS fh_index_0_ratio,
    fh_index_1 / NULLIF(fail_highs, 0) AS fh_index_1_ratio,
    fh_index_2 / NULLIF(fail_highs, 0) AS fh_index_2_ratio,
    fh_index_3 / NULLIF(fail_highs, 0) AS fh_index_3_ratio,
    fh_index_4to7 / NULLIF(fail_highs, 0) AS fh_index_4to7_ratio,
    fh_index_8plus / NULLIF(fail_highs, 0) AS fh_index_8plus_ratio,
    see_prunes / NULLIF(qnodes, 0) AS see_prune_ratio,
    delta_prunes / NULLIF(qnodes, 0) AS delta_prune_ratio,
    (see_prunes + delta_prunes) / NULLIF(qnodes, 0) AS prune_ratio,
    nmp / NULLIF(total_nodes, 0) AS nmp_ratio,
    nmp_failhigh / NULLIF(nmp, 0) AS nmp_failhigh_ratio,
    pvs_researches / NULLIF(total_nodes, 0) AS pvs_research_ratio
FROM (
    -- pvs_researches: the count older builds log, else the sum of the three kinds newer builds log
    -- when any was logged, else NULL (as in search_metrics.sql)
    SELECT * REPLACE (coalesce(pvs_researches,
               CASE WHEN pvs_researches_depth IS NOT NULL OR pvs_researches_full IS NOT NULL
                         OR pvs_researches_root IS NOT NULL
                    THEN coalesce(pvs_researches_depth, 0) + coalesce(pvs_researches_full, 0)
                         + coalesce(pvs_researches_root, 0) END) AS pvs_researches),
           nodes + qnodes AS total_nodes
    FROM (SELECT * FROM iterative_deepening_stats UNION ALL BY NAME
          SELECT NULL::BIGINT AS pvs_researches_depth, NULL::BIGINT AS pvs_researches_full,
                 NULL::BIGINT AS pvs_researches_root WHERE false)
)
