SELECT
    s.id AS id, s.id AS search_id, s.fen, s.ply, s.engine_id, s.game_id, s.sts_id,
    s.nodes, s.qnodes, s.tt_hits, s.tt_stores, s.tt_fill, s.fail_highs, s.fail_lows, s.nmp, s.nmp_failhigh,
    s.time_ms AS total_time_ms,
    s.eval AS final_eval,
    s.move AS best_move,
    s.move AS move,
    s.principal_variation,
    --   search stats
    s.completed_depth  AS completed_depth,
    s.depth AS max_depth,
    s.qdepth AS max_qdepth,
    s.nodes AS total_internal_nodes,
    s.qnodes AS total_qnodes,
    s.nodes + s.qnodes AS total_nodes,
    s.tt_stores AS total_tt_stores,
    s.tt_hits AS total_tt_hits,
    s.fail_highs AS total_fail_highs,
    s.fail_lows AS total_fail_low,
    s.fail_lows AS total_fail_lows,
    s.fail_high_researches AS total_fail_high_researches,
    s.fail_low_researches AS total_fail_low_researches,
    s.fh_index_0 AS total_fh_index_0,
    s.fh_index_1 AS total_fh_index_1,
    s.fh_index_2 AS total_fh_index_2,
    s.fh_index_3 AS total_fh_index_3,
    s.fh_index_4to7 AS total_fh_index_4to7,
    s.fh_index_8plus AS total_fh_index_8plus,
    s.see_prunes AS total_see_prunes,
    s.delta_prunes AS total_delta_prunes,
    s.pvs_researches AS total_pvs_researches,
    s.nmp AS total_nmp,
    s.nmp_failhigh AS total_nmp_fail,
    s.nmp_failhigh AS total_nmp_failhigh,
    s.tt_overwritten AS total_tt_overwritten,

    -- evaluation from the side to move; mate scores are +/-(100000 - plies to mate)
    s.eval AS eval,
    CASE WHEN abs(s.eval) < 99000 THEN s.eval END AS eval_cp,
    CASE WHEN s.eval >= 99000 THEN 1 WHEN s.eval <= -99000 THEN -1 WHEN s.eval IS NOT NULL THEN 0 END AS mate,
    CASE WHEN abs(s.eval) >= 99000 THEN sign(s.eval) * (100000 - abs(s.eval)) END AS mate_plies,
    s.depth AS depth,

    -- stockfish ground-truth (added by transform_positions), also from the side to move
    s.sf_eval AS sf_eval,
    CASE WHEN abs(s.sf_eval) < 99000 THEN s.sf_eval END AS sf_eval_cp,
    s.sf_best_move AS sf_best_move,
    s.sf_pv AS sf_pv,
    CASE WHEN abs(s.eval) < 99000 AND abs(s.sf_eval) < 99000 THEN s.eval - s.sf_eval END AS eval_diff,
    -- Stockfish's top lines: the engine's move among them (1..sf_multipv, 0 = outside the lines,
    -- NULL = unknown) and its loss against Stockfish's first line, both evals clamped to +/-1000 cp.
    s.sf_depth AS sf_depth,
    s.sf_multipv AS sf_multipv,
    s.sf_move_rank AS sf_move_rank,
    s.sf_move_eval AS sf_move_eval,
    CASE WHEN s.sf_eval IS NOT NULL AND s.sf_move_eval IS NOT NULL
         THEN CAST(greatest(0, least(1000, greatest(-1000, s.sf_eval))
                               - least(1000, greatest(-1000, s.sf_move_eval))) AS DOUBLE) END AS sf_cp_loss,
    -- Older analyses without Stockfish's lines only know whether the move was the first one.
    coalesce(s.sf_move_rank, CASE WHEN s.move = s.sf_best_move THEN 1 END) AS engine_move_rank,
    CASE WHEN s.move IS NULL OR s.sf_best_move IS NULL THEN NULL
         WHEN s.move = s.sf_best_move THEN 1 ELSE 0 END AS best_move_match,

    -- total ratios
    (s.nodes + s.qnodes) / NULLIF(s.time_ms / 1000,0) AS nps,
    s.qnodes / NULLIF(s.nodes + s.qnodes,0) AS qratio,
    s.tt_hits / NULLIF(s.nodes + s.qnodes,0) AS tt_hit_ratio,
    s.tt_stores / NULLIF(s.nodes + s.qnodes,0) AS tt_store_ratio,
    s.fail_highs / NULLIF(s.nodes + s.qnodes,0) AS fail_high_ratio,
    s.fail_lows / NULLIF(s.nodes + s.qnodes,0) AS fail_low_ratio,
    s.fh_index_0 / NULLIF(s.fail_highs,0) AS fh_index_0_ratio,
    s.fh_index_1 / NULLIF(s.fail_highs,0) AS fh_index_1_ratio,
    s.fh_index_2 / NULLIF(s.fail_highs,0) AS fh_index_2_ratio,
    s.fh_index_3 / NULLIF(s.fail_highs,0) AS fh_index_3_ratio,
    s.fh_index_4to7 / NULLIF(s.fail_highs,0) AS fh_index_4to7_ratio,
    s.fh_index_8plus / NULLIF(s.fail_highs,0) AS fh_index_8plus_ratio,
    s.fail_high_researches / GREATEST(1, 1 + s.depth - 6) AS fail_high_researches_per_depth,
    s.fail_low_researches / GREATEST(1, 1 + s.depth - 6) AS fail_low_researches_per_depth,
    s.see_prunes / NULLIF(s.qnodes, 0) AS see_prune_ratio,
    s.delta_prunes / NULLIF(s.qnodes, 0) AS delta_prune_ratio,
    s.nmp / NULLIF(s.nodes + s.qnodes, 0) AS nmp_ratio,
    s.nmp_failhigh / NULLIF(s.nmp, 0) AS nmp_failhigh_ratio,

    -- environment of the game the search was made in (match-runner games only)
    g.env_core_type,
    g.env_core_mixed,
    g.env_mhz_mean,
    g.env_host,
    g.env_cpu_model
FROM search_stats s
LEFT JOIN game_stats g ON s.game_id = g.id
