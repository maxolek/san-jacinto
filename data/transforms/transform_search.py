"""
function(s) that perform ETL on raw database 
to turn raw data into analytics ready data

single wide fact table for dashboarding and analysis: 
`search_features` which denormalizes search_stats with position features, iterative deepening features, and timing breakdown features.
"""

ASPIRATION_START_DEPTH = 6


def _table_exists(cnxn, table):
    try:
        cnxn.execute(f"SELECT 1 FROM {table} LIMIT 1")
        return True
    except Exception:
        return False


def _max_search_id(cnxn, table, col='search_id'):
    """Get the max search_id already processed in a feature table."""
    try:
        val = cnxn.execute(f"SELECT COALESCE(MAX({col}), 0) FROM {table}").fetchone()[0]
        return val or 0
    except Exception:
        return 0

def build_search_iterations_features(cnxn, full=False):
    print("  [BUILD] Creating search iteration features...")
    rolling_window = 5
    window_spec = "(PARTITION BY search_id ORDER BY depth ASC)"
    rolling_window_spec = f"(PARTITION BY search_id ORDER BY depth ASC ROWS BETWEEN {rolling_window} PRECEDING AND CURRENT ROW)"

    incremental = not full and _table_exists(cnxn, 'search_iteration_features')
    if incremental:
        max_id = _max_search_id(cnxn, 'search_iteration_features')
        where_clause = f"WHERE search_id > {max_id}"
        print(f"    (incremental: processing search_id > {max_id})")
    else:
        where_clause = ""

    source_query = f"SELECT * FROM iterative_deepening_stats {where_clause}"

    select_sql = f"""
        WITH base_metrics AS (
            SELECT *,
                -- total nodes and move grouping come from original numeric columns
                (nodes + qnodes) AS total_nodes,
                ROW_NUMBER() OVER (PARTITION BY search_id ORDER BY depth) -
                ROW_NUMBER() OVER (PARTITION BY search_id, move ORDER BY depth) as move_grp
            FROM ({source_query})
        )
        SELECT
            search_id, depth, qdepth,
            time_ms AS time_ms, SUM(time_ms) OVER {window_spec} as running_time_ms,
            eval AS eval, move,
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
            pvs_researches / NULLIF(total_nodes, 0) AS pvs_research_ratio,

            -- across iterations derived metrics
            time_ms / LAG(time_ms) OVER {window_spec} as time_increase_ratio,
            total_nodes / LAG(total_nodes) OVER {window_spec} as ebf,
            qnodes / LAG(qnodes) OVER {window_spec} as qebf,

            -- stability metrics (use numeric eval)
            eval - LAG(eval) OVER {window_spec} as prior_eval_delta,
            eval - FIRST_VALUE(eval) OVER {window_spec} as first_eval_delta,
            CASE
                WHEN (eval > 0 AND LAG(eval) OVER {window_spec} < 0) OR 
                     (eval < 0 AND LAG(eval) OVER {window_spec} > 0) THEN 1
                ELSE 0
            END as eval_sign_flips,
            STDDEV(eval) OVER {window_spec} AS stddev_eval,
            STDDEV(eval) OVER {rolling_window_spec} AS stddev_last5_eval,

            ROW_NUMBER() OVER (PARTITION BY search_id, move_grp ORDER BY depth) - 1 AS move_stability

        FROM base_metrics
    """

    if incremental:
        cnxn.execute(f"INSERT INTO search_iteration_features {select_sql}")
        new_count = cnxn.execute(f"SELECT COUNT(*) FROM search_iteration_features WHERE search_id > {max_id}").fetchone()[0]
        print(f"    +{new_count:,} rows inserted")
    else:
        cnxn.execute(f"CREATE OR REPLACE TABLE search_iteration_features AS {select_sql}")

def build_search_tree_features(cnxn, full=False):
    print("  [BUILD] Creating search tree features...")
    window_spec = "(PARTITION BY search_id ORDER BY depth asc)"

    incremental = not full and _table_exists(cnxn, 'search_tree_features')
    if incremental:
        max_id = _max_search_id(cnxn, 'search_tree_features')
        where_clause = f"WHERE search_id > {max_id}"
        print(f"    (incremental: processing search_id > {max_id})")
    else:
        where_clause = ""

    select_sql = f"""
        SELECT
            search_id, depth,
            tt_stores AS tt_stores, tt_hits AS tt_hits,
            nodes AS nodes, qnodes AS qnodes,
            fail_highs AS fail_highs, fail_lows AS fail_lows,
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

            (nodes + qnodes) / NULLIF(LAG(nodes + qnodes) OVER {window_spec}, 0) as ebf,
            qnodes / NULLIF(LAG(qnodes) OVER {window_spec}, 0) as qebf

        FROM search_tree_stats {where_clause}
    """

    if incremental:
        cnxn.execute(f"INSERT INTO search_tree_features {select_sql}")
        new_count = cnxn.execute(f"SELECT COUNT(*) FROM search_tree_features WHERE search_id > {max_id}").fetchone()[0]
        print(f"    +{new_count:,} rows inserted")
    else:
        cnxn.execute(f"CREATE OR REPLACE TABLE search_tree_features AS {select_sql}")
    
def build_search_features(cnxn, full=False):
    print("  [BUILD] Creating search features...")

    incremental = not full and _table_exists(cnxn, 'search_features')
    if incremental:
        max_id = _max_search_id(cnxn, 'search_features', col='search_id')
        where_clause = f"WHERE s.id > {max_id}"
        print(f"    (incremental: processing search_id > {max_id})")
    else:
        where_clause = ""

    # Ensure numeric search-level columns for arithmetic
    # Use search tables' original numeric columns directly
    select_sql = f"""
        WITH times AS (
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
                MAX(CASE WHEN function = 'TT_STORE' THEN total_time_ms / NULLIF(num_calls, 0) END) AS tt_store_avg_ms,
                    
            FROM search_timings
            {"WHERE search_id > " + str(max_id) if incremental else ""}
            GROUP BY search_id
        ),
                 

        iterative_depth AS (
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
            exp(avg(log(CASE WHEN itdeep.ebf > 0 THEN itdeep.ebf ELSE 1 END))) AS geo_mean_ebf,

            AVG(itdeep.qebf) AS avg_qebf,
            MAX(itdeep.qebf) AS max_qebf,
            exp(avg(log(CASE WHEN itdeep.qebf > 0 THEN itdeep.qebf ELSE 1 END))) AS geo_mean_qebf,

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

            MAX_BY(itdeep.eval_sign_flips, itdeep.depth) AS eval_sign_flips,
            
            MAX(itdeep.eval) AS max_eval,
            AVG(itdeep.eval) AS avg_eval,
            STDDEV(itdeep.eval) AS stddev_eval,
            
            FROM search_iteration_features itdeep
            {"WHERE itdeep.search_id > " + str(max_id) if incremental else ""}
            GROUP BY search_id
        )


        SELECT
            s.id as search_id,
            s.fen,
            s.ply,
            --  engine info
            s.engine_id,
            e.name AS engine_name,
            e.version AS engine_version,
            /* BRING IN SEARCH PARAM INFO FOR EASIER ANALYSIS */
            --  game info
            s.game_id,
            CASE
                 WHEN (s.engine_id = g.white_engine_id) AND (g.result = 'white') THEN 1
                 WHEN (s.engine_id = g.black_engine_id) AND (g.result = 'black') THEN 1
                 WHEN g.result = 'draw' THEN 0
                 ELSE -1
            END AS game_score,
            g.opening as game_opening,
            g.opening_eco as game_eco,
            --   sts info
            s.sts_id,
            sts.suite AS sts_suite,
            sts.position_name AS sts_position_name,
            sts.move_is_correct as sts_move_is_correct,
            --   search results
            s.time_ms AS total_time_ms,
            s.eval AS final_eval,
            s.move AS best_move,
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
            s.nmp AS total_nmp,
            s.nmp_failhigh AS total_nmp_fail,
            s.tt_overwritten AS total_tt_overwritten,

            -- stockfish ground-truth (added by transform_positions)
            s.sf_eval AS sf_eval,
            s.sf_best_move AS sf_best_move,
            s.sf_pv AS sf_pv,
            s.eval AS eval,
            s.depth AS depth,
            CASE WHEN s.eval IS NOT NULL AND s.sf_eval IS NOT NULL THEN s.eval - s.sf_eval ELSE NULL END AS eval_diff,
            -- engine_move_rank: 1..5 if engine's move matches Stockfish PV position, 0 if present but not top-5, NULL if no data
            CASE
                WHEN s.move IS NOT NULL AND s.sf_pv IS NOT NULL AND json_extract(s.sf_pv, '$[0]') = '"' || s.move || '"' THEN 1
                WHEN s.move IS NOT NULL AND s.sf_pv IS NOT NULL AND json_extract(s.sf_pv, '$[1]') = '"' || s.move || '"' THEN 2
                WHEN s.move IS NOT NULL AND s.sf_pv IS NOT NULL AND json_extract(s.sf_pv, '$[2]') = '"' || s.move || '"' THEN 3
                WHEN s.move IS NOT NULL AND s.sf_pv IS NOT NULL AND json_extract(s.sf_pv, '$[3]') = '"' || s.move || '"' THEN 4
                WHEN s.move IS NOT NULL AND s.sf_pv IS NOT NULL AND json_extract(s.sf_pv, '$[4]') = '"' || s.move || '"' THEN 5
                WHEN s.move IS NOT NULL AND s.sf_pv IS NOT NULL AND (json_extract(s.sf_pv, '$[0]') IS NOT NULL OR json_extract(s.sf_pv, '$[1]') IS NOT NULL OR json_extract(s.sf_pv, '$[2]') IS NOT NULL OR json_extract(s.sf_pv, '$[3]') IS NOT NULL OR json_extract(s.sf_pv, '$[4]') IS NOT NULL) THEN 0
                ELSE NULL
            END AS engine_move_rank,
            CASE WHEN s.move IS NOT NULL AND s.sf_pv IS NOT NULL AND json_extract(s.sf_pv, '$[0]') = '"' || s.move || '"' THEN 1 WHEN s.move IS NULL OR s.sf_pv IS NULL THEN NULL ELSE 0 END AS best_move_match,

            -- total ratios
            (s.nodes + s.qnodes) / NULLIF(s.time_ms / 1000,0) AS nps,
            s.qnodes / NULLIF(s.nodes + s.qnodes,0) AS qratio,
            s.tt_hits / NULLIF(s.nodes + s.qnodes,0) AS tt_hit_ratio,
            s.tt_stores / NULLIF(s.nodes + s.qnodes,0) AS tt_store_ratio,
            s.fail_highs / NULLIF(s.nodes + s.qnodes,0) AS fail_high_ratio,
            s.fail_lows / NULLIF(s.nodes + s.qnodes,0) AS fail_low_ratio,
            s.fh_index_0 / NULLIF(s.fail_highs,1) AS fh_index_0_ratio,
            s.fh_index_1 / NULLIF(s.fail_highs,1) AS fh_index_1_ratio,
            s.fh_index_2 / NULLIF(s.fail_highs,1) AS fh_index_2_ratio,
            s.fh_index_3 / NULLIF(s.fail_highs,1) AS fh_index_3_ratio,
            s.fh_index_4to7 / NULLIF(s.fail_highs,1) AS fh_index_4to7_ratio,
            s.fh_index_8plus / NULLIF(s.fail_highs,1) AS fh_index_8plus_ratio,
            s.fail_high_researches / GREATEST(1, 1 + s.depth - {ASPIRATION_START_DEPTH}) AS fail_high_researches_per_depth,
            s.fail_low_researches / GREATEST(1, 1 + s.depth - {ASPIRATION_START_DEPTH}) AS fail_low_researches_per_depth,
            s.nmp / NULLIF(s.nodes + s.qnodes, 0) AS nmp_ratio,
            s.nmp_failhigh / NULLIF(s.nmp, 0) AS nmp_failhigh_ratio,

            -- iterative deepening aggregated stats
            itdeep.* EXCLUDE (search_id),
            itdeep.eval_sign_flips / NULLIF(s.depth,1) AS eval_sign_flips_per_depth,

            -- timing stats
            t.make_move_avg_ms AS make_move_avg_ms, 
            100 * t.make_move_total_ms / s.time_ms AS make_move_perc_total_time,
            t.unmake_move_avg_ms AS unmake_move_avg_ms, 
            100 * t.unmake_move_total_ms / s.time_ms AS unmake_move_perc_total_time,
            t.movegen_avg_ms AS movegen_avg_ms, 
            100 * t.movegen_total_ms / s.time_ms AS movegen_perc_total_time,
            t.move_order_avg_ms AS move_order_avg_ms, 
            100 * t.move_order_total_ms / s.time_ms AS move_order_perc_total_time,
            t.tt_probe_avg_ms AS tt_probe_avg_ms, 
            100 * t.tt_probe_total_ms / s.time_ms AS tt_probe_perc_total_time,
            t.tt_store_avg_ms AS tt_store_avg_ms, 
            100 * t.tt_store_total_ms / s.time_ms AS tt_store_perc_total_time,
            t.see_avg_ms AS see_avg_ms, 
            100 * t.see_total_ms / s.time_ms AS see_perc_total_time,
            t.nnue_avg_ms AS nnue_avg_ms, 
            100 * t.nnue_total_ms / s.time_ms AS nnue_perc_total_time,
            t.static_eval_avg_ms AS static_eval_avg_ms, 
            100 * t.static_eval_total_ms / s.time_ms AS static_eval_perc_total_time,
            
            -- position features
            pf.position_type AS pos_label,
            pf.game_phase AS game_phase,
            pf.position_type AS position_type,
            pf.pos_tactical AS position_tactical_score,
            pf.pos_positional AS position_positional_score,
            pf.pos_endgame AS position_endgame_score,
            pf.balance AS position_balance,
            pf.white_backwards AS position_white_backwards,
            pf.white_doubled AS position_white_doubled,
            pf.white_passed AS position_white_passed,
            pf.black_backwards AS position_black_backwards,
            pf.black_doubled AS position_black_doubled,
            pf.black_passed AS position_black_passed,
            pf.white_shield_pawns AS position_white_shield_pawns,
            pf.white_open_files AS position_white_open_files,
            pf.white_tropism AS position_white_tropism,
            pf.black_shield_pawns AS position_black_shield_pawns,
            pf.black_open_files AS position_black_open_files,
            pf.black_tropism AS position_black_tropism,
            pf.white_num_moves AS position_white_num_moves,
            pf.white_capture_ratio AS position_white_capture_ratio,
            pf.white_check_ratio AS position_white_check_ratio,
            pf.white_legal_enemy AS position_white_legal_enemy,
            pf.white_controlled_enemy AS position_white_controlled_enemy,
            pf.black_num_moves AS position_black_num_moves,
            pf.black_capture_ratio AS position_black_capture_ratio,
            pf.black_check_ratio AS position_black_check_ratio,
            pf.black_legal_enemy AS position_black_legal_enemy,
            pf.black_controlled_enemy AS position_black_controlled_enemy


        FROM search_stats s
        -- additional search stats
        LEFT JOIN iterative_depth itdeep
            ON s.id = itdeep.search_id
        LEFT JOIN times t 
            ON s.id = t.search_id
        LEFT JOIN position_features pf
            ON s.id = pf.search_id
        -- search environment info
        LEFT JOIN engines e
            on s.engine_id = e.id
        LEFT JOIN game_stats g
            ON s.game_id = g.id
        LEFT JOIN sts_runs sts
            ON s.sts_id = sts.id
        {where_clause}
    """

    if incremental:
        cnxn.execute(f"INSERT INTO search_features {select_sql}")
        new_count = cnxn.execute(f"SELECT COUNT(*) FROM search_features WHERE search_id > {max_id}").fetchone()[0]
        print(f"    +{new_count:,} rows inserted")
    else:
        cnxn.execute(f"CREATE OR REPLACE TABLE search_features AS {select_sql}")

import duckdb
import os
import argparse
from ..etl.paths import ANALYTICS_DB

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build search feature tables")
    parser.add_argument('--full', action='store_true', help='Full rebuild (drop + recreate all feature tables)')
    args = parser.parse_args()

    DB = os.environ.get('CHESS_ANALYTICS_DB') or str(ANALYTICS_DB)
    cnxn = duckdb.connect(DB)

    full = args.full
    if full:
        print("  [BUILD] Full rebuild requested")

    build_search_tree_features(cnxn, full=full)
    build_search_iterations_features(cnxn, full=full)
    build_search_features(cnxn, full=full)

    # Persist eval_diff into `search_stats` so downstream tools (dashboard, queries)
    # can read it directly from the table instead of recomputing client-side.
    print("  [BUILD] Executing...")
    try:
        cur = cnxn.execute("PRAGMA table_info('search_stats')").fetchall()
        cols = {r[1] for r in cur}
        if 'eval_diff' not in cols:
            cnxn.execute("ALTER TABLE search_stats ADD COLUMN eval_diff DOUBLE")
        # populate eval_diff from existing eval and sf_eval values
        cnxn.execute("UPDATE search_stats SET eval_diff = CASE WHEN eval IS NOT NULL AND sf_eval IS NOT NULL THEN eval - sf_eval ELSE NULL END WHERE eval_diff IS NULL")
        print("  [BUILD] Completed building (wide) feature tables.")
    except Exception:
        print("  [BUILD] Failed to build tables !!!!!!")
        # non-fatal: leave as-is if ALTER/UPDATE unsupported for some DB backends
        pass

    cnxn.close()