SELECT s.*,

    e.name AS engine_name,
    e.version AS engine_version,

    --  game info
    CASE
         WHEN (s.engine_id = g.white_engine_id) AND (g.result = 'white') THEN 1
         WHEN (s.engine_id = g.black_engine_id) AND (g.result = 'black') THEN 1
         WHEN g.result = 'draw' THEN 0
         WHEN g.result IN ('white', 'black') AND s.engine_id IN (g.white_engine_id, g.black_engine_id) THEN -1
         ELSE NULL
    END AS game_score,
    g.opening as game_opening,
    g.opening_eco as game_eco,
    --   sts info
    sts.suite AS sts_suite,
    sts.position_name AS sts_position_name,
    sts.move_is_correct as sts_move_is_correct
FROM search_metrics s
LEFT JOIN engines e ON s.engine_id = e.id
LEFT JOIN game_stats g ON s.game_id = g.id
LEFT JOIN sts_runs sts ON s.sts_id = sts.id
