import sqlite3
from pathlib import Path
from ..etl.paths import RAW_DB
from .run_analytics_pipeline import run_module

# long form function that create the database
# and some associated tables

def init_raw_db(db_dir=None) -> None:

    """
        input: directory of database
        output: None

        executed: creates the engine database and 6 tables
            engines         -- information about engines and versions
            engine_ratings  -- ratings for ^^ calculated from round-robin tournament
            experiments     -- information about engine processes (e.g. sprt/sts)
            search_summary  -- high level search info (e.g. nodes, time)
            search_depth    -- per-depth search info (e.g. eval per depth)
            timing          -- computation time (e.g. how long does movegen take)
            games           -- summary game info (players, result, moves, etc)
    """

    # dir + path
    db_dir = Path(db_dir) if db_dir else RAW_DB.parent
    db_dir.mkdir(parents=True, exist_ok=True)

    db_path = db_dir / "chess.db"
    print(f"[DB] Initializing raw database at: {db_path}")

    # cnxn + cursor
    cnxn = sqlite3.connect(db_path)
    cur = cnxn.cursor()


    # CREATE DB STRINGS

    # engine binaries info
    # immutable
    engines_str = """
        CREATE TABLE IF NOT EXISTS engines (
            id                          INTEGER PRIMARY KEY AUTOINCREMENT,
            name                        TEXT NOT NULL,
            version                     TEXT NOT NULL,
            description                 TEXT,
            compile_flags               TEXT,
            
            -- UCI engine options
            move_overhead_ms            INTEGER,
            max_threads                 INTEGER,
            hash_size_mb                INTEGER,
            pondering                   BOOLEAN,

            -- search params
            delta_prune_threshold       INTEGER,
            see_prune_threshold         INTEGER,
            aspiration_window           INTEGER,
            aspiration_start_depth      INTEGER,
            aspiration_depth_scale      INTEGER,
            aspiration_research_scale   REAL,
            draw_eval                   INTEGER,
            contempt                    INTEGER,
            r_nmp                       INTEGER,
            r_lmr_const                 REAL,
            r_lmr_denom                 REAL,
            lmr_move_order_threshold    INTEGER,
            lmr_depth_threshold         INTEGER,

            --

            ingestion_timestamp_utc   DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """

    # key wrapper for engine processes
    # tracks what engine ran what 
    #  and provides FKs for table joins
    experiments_str = """
        CREATE TABLE IF NOT EXISTS experiments (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            engine_id               INTEGER NOT NULL,
            comparison_engine_id    INTEGER NULL, -- e.g. old engine, stockfish, in sprt
            type                    TEXT NOT NULL, -- sts, perft, sprt
            start_time_utc          DATETIME DEFAULT CURRENT_TIMESTAMP,
            end_time_utc            DATETIME,
            metadata                TEXT, 

            FOREIGN KEY (engine_id) REFERENCES engines(id)
        );
    """

    engine_ratings_str = """
        CREATE TABLE IF NOT EXISTS engine_ratings (
            id                          INTEGER PRIMARY KEY AUTOINCREMENT,
            engine_id                   INTEGER NOT NULL,

            -- elo by time control
            elo_ultra_fast              REAL, -- < 15 sec/game
            elo_bullet                  REAL, -- < 1 min/game
            elo_blitz                   REAL, -- < 10 min/game
            elo_rapid                   REAL, -- < 30 min/game
            elo_classical               REAL, -- >= 30 min/game

            -- supporting stats
            games_ultra_fast            INTEGER DEFAULT 0,
            games_bullet                INTEGER DEFAULT 0,
            games_blitz                 INTEGER DEFAULT 0,
            games_rapid                 INTEGER DEFAULT 0,
            games_classical             INTEGER DEFAULT 0,

            ingestion_timestamp_utc     DATETIME DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (engine_id) REFERENCES engines(id)
        );
    """

    # high level search info
    search_summary_stats_str = """
        CREATE TABLE IF NOT EXISTS searches (
        -- metadata
            id                          INTEGER PRIMARY KEY AUTOINCREMENT,
            engine_id                   INTEGER NULL,
            game_id                     INTEGER NULL, -- link to game/sts
            sts_id                      INTEGER NULL, 
        -- position
            fen                         TEXT NOT NULL,
            ply                         INTEGER,
            time_ms                     REAL,
            eval                        INTEGER,
            completed_depth             INTEGER, -- fully completed (all root moves searched) depth (max_depth -= 1/0)
            depth                       INTEGER,
            qdepth                      INTEGER, -- depth reached through quiescence (leaf extension of search depth ^^)
            move                        TEXT,
            principal_variation         TEXT,
        -- stats
            nodes                       INTEGER,
            qnodes                      INTEGER,
            tt_stores                   INTEGER,
            tt_hits                     INTEGER,
            tt_fill                     REAL,
            fail_highs                  INTEGER,
            fail_lows                   INTEGER,
        -- specifics
            fail_high_researches        INTEGER,
            fail_low_researches         INTEGER,
            -- fail-high move index histogram [bucket0..bucket5]
            fh_index_0                  INTEGER, -- move index 0 (TT/hash move)
            fh_index_1                  INTEGER, -- move index 1
            fh_index_2                  INTEGER, -- move index 2
            fh_index_3                  INTEGER, -- move index 3
            fh_index_4to7               INTEGER, -- move index 4-7
            fh_index_8plus              INTEGER, -- move index 8+
            -- pruning
            see_prunes                  INTEGER,
            delta_prunes                INTEGER,
            pvs_researches              INTEGER,
            nmp                         INTEGER,
            nmp_failhigh                    INTEGER,
            tt_overwritten              INTEGER,

            ingestion_timestamp_utc     DATETIME DEFAULT CURRENT_TIMESTAMP,
            
            FOREIGN KEY (engine_id) REFERENCES engines(id),
            FOREIGN KEY (game_id) REFERENCES games(id),
            FOREIGN KEY (sts_id) REFERENCES sts(id)
        );
    """

    # per-iteration-depth search info
    search_depth_stats_str = """
        CREATE TABLE IF NOT EXISTS searches_by_iteration (
        -- metadata
            search_id                   INTEGER NOT NULL,
            depth                       INTEGER NOT NULL,
        -- results
            time_ms                     REAL,
            eval                        INTEGER,
            move                        TEXT,
        -- stats
            nodes                       INTEGER,
            qnodes                      INTEGER,
            qdepth                      INTEGER,
            tt_stores                   INTEGER,
            tt_hits                     INTEGER,
            tt_fill                     REAL,
            fail_highs                  INTEGER,
            fail_lows                   INTEGER,
        -- specifics
            fail_high_researches        INTEGER,
            fail_low_researches         INTEGER,
            -- fail-high move index histogram per iteration
            fh_index_0                  INTEGER,
            fh_index_1                  INTEGER,
            fh_index_2                  INTEGER,
            fh_index_3                  INTEGER,
            fh_index_4to7               INTEGER,
            fh_index_8plus              INTEGER,
            see_prunes                  INTEGER,
            delta_prunes                INTEGER,
            pvs_researches              INTEGER,
            nmp                         INTEGER,
            nmp_failhigh                    INTEGER,

            ingestion_timestamp_utc     DATETIME DEFAULT CURRENT_TIMESTAMP,
            
            PRIMARY KEY (search_id, depth),
            FOREIGN KEY (search_id) REFERENCES searches(id)
        );
    """

    # per-search_tree-ply search info
    search_ply_stats_str = """
        CREATE TABLE IF NOT EXISTS searches_by_tree_depth (
        --metadata
            search_id                   INTEGER NOT NULL,
            depth                       INTEGER NOT NULL,
            --time_ms                     REAL,
        -- stats
            nodes                       INTEGER,
            qnodes                      INTEGER,
            tt_stores                   INTEGER,
            tt_hits                     INTEGER,
            --tt_fill                     REAL,
            fail_highs                  INTEGER,
            fail_lows                   INTEGER,
            -- fail-high move index histogram per tree depth
            fh_index_0                  INTEGER,
            fh_index_1                  INTEGER,
            fh_index_2                  INTEGER,
            fh_index_3                  INTEGER,
            fh_index_4to7               INTEGER,
            fh_index_8plus              INTEGER,
            see_prunes                  INTEGER,
            delta_prunes                INTEGER,
            pvs_researches              INTEGER,
            nmp                         INTEGER,
            nmp_failhigh                    INTEGER,

            ingestion_timestamp_utc     DATETIME DEFAULT CURRENT_TIMESTAMP,

            PRIMARY KEY (search_id, depth),
            FOREIGN KEY (search_id) REFERENCES searches(id)
        );
    """

    # computation time info
    timing_stats_str = """
        CREATE TABLE IF NOT EXISTS timing (
            search_id                   INTEGER NOT NULL,
            function                    TEXT NOT NULL,
            total_time_ms               REAL,
            num_calls                   INTEGER,

            ingestion_timestamp_utc   DATETIME DEFAULT CURRENT_TIMESTAMP,

            PRIMARY KEY (search_id, function),
            FOREIGN KEY (search_id) REFERENCES searches(id)
        );
    """

    # per-root-move timing and node counts
    root_moves_str = """
        CREATE TABLE IF NOT EXISTS root_moves (
            id                          INTEGER PRIMARY KEY AUTOINCREMENT,
            search_id                   INTEGER NOT NULL,
            depth                       INTEGER NOT NULL,
            move_index                  INTEGER NOT NULL,
            move                        TEXT NOT NULL,
            eval                        INTEGER,
            time_ms                     INTEGER,
            nodes                       INTEGER,

            ingestion_timestamp_utc     DATETIME DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (search_id) REFERENCES searches(id)
        );
    """

    # game info
    game_stats_str = """
        CREATE TABLE IF NOT EXISTS games (
        -- metadata
            id                          INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id               INTEGER NULL, -- link to sprt
            white_engine_id             INTEGER NULL, -- nullable since may not have 2 local engines playing (e.g. lichess games)
            black_engine_id             INTEGER NULL,
        -- info
            wtime                       INTEGER,
            winc                        INTEGER,
            btime                       INTEGER,
            binc                        INTEGER,
            movestogo                   INTEGER,
            depth                       INTEGER,
            nodes                       INTEGER,
            movetime                    INTEGER,
            --white_player                TEXT,
            --black_player                TEXT,
            --white_elo                   INTEGER,
            --black_elo                   INTEGER,
        -- results
            result                      TEXT, -- 1-0 , 0-1 , 1/2-1/2
            termination                 TEXT, -- mate, repetition, etc.
            opening                     TEXT,
            start_fen                   TEXT, -- typically START_POS but sometimes games are loaded from position
            moves                       TEXT, -- array ["e2e4", "e7e5", etc.]
            run_time_s                  REAL,

            ingestion_timestamp_utc     DATETIME DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (experiment_id) REFERENCES experiments(id),
            FOREIGN KEY (white_engine_id) REFERENCES engines(id),
            FOREIGN KEY (black_engine_id) REFERENCES engines(id)
        );
    """

    # SPRT tests
    sprt_test_str = """
        CREATE TABLE IF NOT EXISTS sprt (
            id                              INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id                   INTEGER NOT NULL, 
        -- settings
            baseline_engine_id              INTEGER NOT NULL,
            candidate_engine_id             INTEGER NOT NULL,
            opening_book                    TEXT,
            book_depth                      INTEGER,
            time_control                    TEXT,
            time_per_move                   REAL,
            depth_per_move                  INTEGER,
            elo0                            INTEGER,
            elo1                            INTEGER,
            alpha                           REAL,
            beta                            REAL,
        -- results
            result                          TEXT,    -- pass/fail/inconclusive
            elo_diff                        INTEGER,
            llr                             REAL,
            los                             REAL, -- liklihood of superiority
            candidate_wins                  INTEGER, -- all of these wdl stats are for candidate engine
            candidate_losses                INTEGER,
            candidate_draws                 INTEGER,
            candidate_white_wins            INTEGER,
            candidate_white_losses          INTEGER,
            candidate_white_draws           INTEGER,
            candidate_black_wins            INTEGER,
            candidate_black_losses          INTEGER,
            candidate_black_draws           INTEGER,
        -- summaries
            games_played                    INTEGER,
            run_time_s                      REAL,

            ingestion_timestamp_utc         DATETIME DEFAULT CURRENT_TIMESTAMP,
            
            FOREIGN KEY (baseline_engine_id) REFERENCES engines(id),
            FOREIGN KEY (candidate_engine_id) REFERENCES engines(id),
            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        );
    """

    # STS tests
    sts_test_str = """
        CREATE TABLE IF NOT EXISTS sts (
        -- metadata
            id                          INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id               INTEGER NOT NULL,
            suite                       TEXT,
            position_name               TEXT,
            fen                         TEXT NOT NULL,
            search_time_ms              REAL,
            search_depth                INTEGER,
        -- results
            engine_move                 TEXT NOT NULL,
            engine_score                INTEGER,
            expected_move               TEXT NULL,
            expected_score              INTEGER,
            alt_expected_move           TEXT NULL, -- sometimes a 2nd move is acceptable (in cases of >2 we just ignore those)
            avoid_move                  TEXT NULL, -- am (instead of bm) 
            move_is_correct             BOOLEAN,
            
            ingestion_timestamp_utc     DATETIME DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        );
    """

    # PERFT tests
    perft_test_str = """
        CREATE TABLE IF NOT EXISTS perft (
            id                          INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id               INTEGER NOT NULL,
            fen                         TEXT NOT NULL,
            depth                       INTEGER,
            nodes                       INTEGER,
            expected_nodes              INTEGER,
            correct                     BOOLEAN,
            time_ms                     REAL,

            ingestion_timestamp_utc     DATETIME DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        );
    """

    # prelim execute
    cur.execute("""
        PRAGMA foreign_keys = ON; 
    """)

    # execute + commit
    #   SQLite requires that reference tables exist at insert time, not create
    #   so order isnt too important but better safe than sorry
    #   order: top -> down ... engines -> tests -> games -> searches -> timing/details
    for script in [
        engines_str,
        engine_ratings_str,         # FK(engines.id)
        experiments_str,            # FK(engines.id)
        sprt_test_str,              # FK(engines.id)
        sts_test_str,               # FK(engines.id)
        perft_test_str,             # FK(engines.id)
        game_stats_str,             # FK(engines.id) + FK(sprt.id)
        search_summary_stats_str,   # FK(engines.id) + FK(games.id) + FK(sprt.id) + FK(sts.id)
        search_depth_stats_str,     # FK(searches.id)
        search_ply_stats_str,       # FK(searches.id)
        timing_stats_str,           # FK(searches.id)
        root_moves_str              # FK(searches.id)
    ]:
        cur.executescript(script)
    cnxn.commit()
    print(f"[DB] Raw database initialized with schema at {db_path}")

if __name__ == "__main__":
    init_raw_db()

    # migrate schema after init db
    # (will throw warning for analytics if any migrations there yet)
    run_module("data.transforms.migrate_schema")