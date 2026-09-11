SELECT
    search_id, depth,
    tt_stores AS tt_stores, tt_hits AS tt_hits,
    nodes AS nodes, qnodes AS qnodes,
    fail_highs AS fail_highs, fail_lows AS fail_lows,
    fh_index_0, fh_index_1, fh_index_2, fh_index_3, fh_index_4to7, fh_index_8plus,
    see_prunes AS see_prunes, delta_prunes AS delta_prunes,
    pvs_researches AS pvs_researches,
    nmp AS nmp, nmp_failhigh AS nmp_failhigh,

    nodes + qnodes AS total_nodes,
    qnodes / NULLIF(nodes + qnodes, 0) as qratio,
    tt_hits / NULLIF(nodes + qnodes, 0) AS tt_hit_ratio,
    tt_stores / NULLIF(nodes + qnodes, 0) AS tt_store_ratio,
    fail_highs / NULLIF(nodes + qnodes, 0) AS fail_high_ratio,
    fail_lows / NULLIF(nodes + qnodes, 0) AS fail_low_ratio,
    fh_index_0 / NULLIF(fail_highs, 0) AS fh_index_0_ratio,
    fh_index_1 / NULLIF(fail_highs, 0) AS fh_index_1_ratio,
    fh_index_2 / NULLIF(fail_highs, 0) AS fh_index_2_ratio,
    fh_index_3 / NULLIF(fail_highs, 0) AS fh_index_3_ratio,
    fh_index_4to7 / NULLIF(fail_highs, 0) AS fh_index_4to7_ratio,
    fh_index_8plus / NULLIF(fail_highs, 0) AS fh_index_8plus_ratio,
    see_prunes / NULLIF(qnodes, 0) AS see_prune_ratio,
    delta_prunes / NULLIF(qnodes, 0) AS delta_prune_ratio,
    (see_prunes + delta_prunes) / NULLIF(qnodes, 0) AS prune_ratio,
    nmp / NULLIF(nodes + qnodes, 0) AS nmp_ratio,
    nmp_failhigh / NULLIF(nmp, 0) AS nmp_failhigh_ratio,
    pvs_researches / NULLIF(nodes + qnodes, 0) AS pvs_research_ratio,

    (nodes + qnodes) / NULLIF(LAG(nodes + qnodes) OVER (PARTITION BY search_id ORDER BY depth asc), 0) as ebf,
    qnodes / NULLIF(LAG(qnodes) OVER (PARTITION BY search_id ORDER BY depth asc), 0) as qebf

FROM search_tree_stats
