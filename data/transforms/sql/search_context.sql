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
    g.opening_variation as game_opening_variation,
    --   sts info
    sts.suite AS sts_suite,
    sts.position_name AS sts_position_name,
    sts.move_is_correct as sts_move_is_correct
FROM search_metrics s
LEFT JOIN engines e ON s.engine_id = e.id
LEFT JOIN game_stats g ON s.game_id = g.id
-- searches.sts_id holds the STS *experiment* id; the position is matched by its
-- EPD part (placement, side, castling, en passant), one row per experiment + position.
LEFT JOIN (
    SELECT experiment_id,
           split_part(fen, ' ', 1) || ' ' || split_part(fen, ' ', 2) || ' ' ||
           split_part(fen, ' ', 3) || ' ' || split_part(fen, ' ', 4) AS position_key,
           arg_min(suite, id) AS suite,
           arg_min(position_name, id) AS position_name,
           arg_min(move_is_correct, id) AS move_is_correct
    FROM sts_runs
    GROUP BY experiment_id, position_key
) sts ON sts.experiment_id = s.sts_id
     AND sts.position_key = split_part(s.fen, ' ', 1) || ' ' || split_part(s.fen, ' ', 2) || ' ' ||
                            split_part(s.fen, ' ', 3) || ' ' || split_part(s.fen, ' ', 4)
