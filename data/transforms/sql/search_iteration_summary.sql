SELECT

    search_id,

    AVG(itdeep.nps) AS avg_nps,
    STDDEV(itdeep.nps) AS stddev_nps,
    MAX(itdeep.nps) AS peak_nps,
    MIN(itdeep.nps) AS worst_nps,
    MAX_BY(itdeep.nps, itdeep.depth) AS final_nps,

    AVG(itdeep.qratio) AS avg_qratio,
    MAX(itdeep.qratio) AS max_qratio,
    STDDEV(itdeep.qratio) AS stddev_qratio,

    AVG(itdeep.ebf) AS avg_ebf,
    MAX(itdeep.ebf) AS max_ebf,
    exp(avg(ln(CASE WHEN itdeep.ebf > 0 THEN itdeep.ebf END))) AS geo_mean_ebf,

    AVG(itdeep.qebf) AS avg_qebf,
    MAX(itdeep.qebf) AS max_qebf,
    exp(avg(ln(CASE WHEN itdeep.qebf > 0 THEN itdeep.qebf END))) AS geo_mean_qebf,

    AVG(itdeep.tt_hit_ratio) AS avg_tt_hit_ratio,
    MAX(itdeep.tt_hit_ratio) AS max_tt_hit_ratio,
    STDDEV(itdeep.tt_hit_ratio) AS stddev_tt_hit_ratio,

    AVG(itdeep.tt_store_ratio) AS avg_tt_store_ratio,
    MAX(itdeep.tt_store_ratio) AS max_tt_store_ratio,
    STDDEV(itdeep.tt_store_ratio) AS stddev_tt_store_ratio,

    AVG(itdeep.fail_high_ratio) AS avg_fail_high_ratio,
    MAX(itdeep.fail_high_ratio) AS max_fail_high_ratio,
    AVG(itdeep.fail_low_ratio) AS avg_fail_low_ratio,
    MAX(itdeep.fail_low_ratio) AS max_fail_low_ratio,

    AVG(itdeep.fh_index_0_ratio) AS avg_fh_index_0_ratio,
    AVG(itdeep.fh_index_1_ratio) AS avg_fh_index_1_ratio,
    AVG(itdeep.fh_index_2_ratio) AS avg_fh_index_2_ratio,
    AVG(itdeep.fh_index_3_ratio) AS avg_fh_index_3_ratio,
    AVG(itdeep.fh_index_4to7_ratio) AS avg_fh_index_4to7_ratio,
    AVG(itdeep.fh_index_8plus_ratio) AS avg_fh_index_8plus_ratio,

    AVG(itdeep.see_prune_ratio) AS avg_see_prune_ratio,
    AVG(itdeep.delta_prune_ratio) AS avg_delta_prune_ratio,
    AVG(itdeep.prune_ratio) AS avg_prune_ratio,

    MAX(itdeep.fail_high_researches) AS max_fail_high_researches,
    MAX(itdeep.fail_low_researches) AS max_fail_low_researches,

    AVG(itdeep.nmp_ratio) AS avg_nmp_ratio,
    MAX(itdeep.nmp_ratio) AS max_nmp_ratio,
    AVG(itdeep.nmp_failhigh_ratio) AS avg_nmp_failhigh_ratio,
    AVG(itdeep.pvs_research_ratio) AS avg_pvs_research_ratio,
    MAX(itdeep.pvs_research_ratio) AS max_pvs_research_ratio,

    MAX(itdeep.move_stability) AS max_move_stability,
    MAX_BY(itdeep.move_stability, itdeep.depth) AS final_move_stability,

    SUM(itdeep.eval_sign_flips) AS eval_sign_flips,

    MAX(itdeep.eval) AS max_eval,
    AVG(itdeep.eval) AS avg_eval,
    STDDEV(itdeep.eval) AS stddev_eval
FROM search_iteration_features itdeep

GROUP BY search_id
