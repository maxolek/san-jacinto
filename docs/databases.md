# Database Schemas

## Raw Layer — SQLite (`chess.db`)

```mermaid
graph TD
    subgraph Raw["chess.db — SQLite"]

        subgraph Core["Core"]
            ENG["<b>engines</b><br/><i>Engine versions & UCI params</i>"]
            RAT["<b>engine_ratings</b><br/><i>Elo per time control</i>"]
            EXP["<b>experiments</b><br/><i>Test runs (sprt/sts/perft)</i>"]
            GAM["<b>games</b><br/><i>Game results & metadata</i>"]
        end

        subgraph SearchLayer["Searches"]
            SEA["<b>searches</b><br/><i>Final search stats per position</i>"]
            SBI["<b>searches_by_iteration</b><br/><i>Per-depth iteration stats</i>"]
            SBT["<b>searches_by_tree_depth</b><br/><i>Per-ply tree depth stats</i>"]
            TIM["<b>timing</b><br/><i>Function-level profiling</i>"]
            RM["<b>root_moves</b><br/><i>Per-depth root move scores</i>"]
        end

        subgraph Tests["Test Results"]
            SPRT["<b>sprt</b><br/><i>SPRT test results</i>"]
            STS["<b>sts</b><br/><i>Strategic Test Suite results</i>"]
            PERFT["<b>perft</b><br/><i>Move gen correctness tests</i>"]
        end

        MIG["<b>schema_migrations</b><br/><i>Applied migrations</i>"]

        ENG --> RAT
        ENG --> EXP
        ENG --> GAM
        ENG --> SEA
        GAM --> SEA
        GAM --> SPRT
        STS --> SEA

        EXP --> SPRT
        EXP --> STS
        EXP --> PERFT

        SEA -..- SBI
        SEA -..- SBT
        SEA -..- TIM
        SEA -..- RM
    end

    style ENG fill:#1a2e1a,stroke:#7fff6b,stroke-width:2px,color:#fff
    style SEA fill:#1a1f2e,stroke:#00d2ff,stroke-width:2px,color:#fff
    style GAM fill:#1a1f2e,stroke:#00d2ff,stroke-width:2px,color:#fff
    style EXP fill:#2d1b00,stroke:#f7b731,stroke-width:2px,color:#fff
    style SPRT fill:#2d1b00,stroke:#f7b731,stroke-width:2px,color:#fff
    style STS fill:#2d1b00,stroke:#f7b731,stroke-width:2px,color:#fff
    style PERFT fill:#2d1b00,stroke:#f7b731,stroke-width:2px,color:#fff
    style RAT fill:#1a2e1a,stroke:#7fff6b,stroke-width:2px,color:#fff
    style SBI fill:#1a1f2e,stroke:#6366f1,stroke-width:2px,color:#fff
    style SBT fill:#1a1f2e,stroke:#6366f1,stroke-width:2px,color:#fff
    style TIM fill:#1a1f2e,stroke:#6366f1,stroke-width:2px,color:#fff
    style RM fill:#1a1f2e,stroke:#6366f1,stroke-width:2px,color:#fff
    style MIG fill:#1a1f2e,stroke:#8892a4,stroke-width:1px,color:#aaa
```

| Group | Tables | Row Scale |
|-------|--------|-----------|
| Core entities | `engines`, `experiments`, `engine_ratings` | 10s |
| Game data | `games` | 1000s |
| Search data | `searches`, `searches_by_iteration`, `searches_by_tree_depth`, `timing`, `root_moves` | 100k+ |
| Test results | `sprt`, `sts`, `perft` | 100s |

Arrows indicate foreign-key direction (parent → child).

---

## Analytics Layer — DuckDB (`chess_analytics.duckdb`)

```mermaid
graph TD
    subgraph Analytics["chess_analytics.duckdb — DuckDB"]

        subgraph Dims["Dimension Tables"]
            ENG2["<b>engines</b><br/><i>Copy from raw</i>"]
            EXP2["<b>experiments</b><br/><i>Copy from raw</i>"]
            RAT2["<b>engine_ratings</b><br/><i>+ engine_name, engine_version</i>"]
            GS["<b>game_stats</b><br/><i>Games with text result/termination, ECO</i>"]
            DIM["<b>dim_positions</b><br/><i>Unique (search_id, fen)</i>"]
            SPRT2["<b>sprt_runs</b><br/><i>SPRT results</i>"]
            STS3["<b>sts_runs</b><br/><i>STS results</i>"]
        end

        subgraph Facts["Fact Tables"]
            SS["<b>search_stats</b><br/><i>Searches + Stockfish eval columns</i>"]
            IDS["<b>iterative_deepening_stats</b><br/><i>Per-iteration raw stats</i>"]
            STS2["<b>search_tree_stats</b><br/><i>Per-tree-ply raw stats</i>"]
            ST["<b>search_timings</b><br/><i>Function-level timing</i>"]
            RM2["<b>root_moves</b><br/><i>Per-move root scores</i>"]
        end

        subgraph Features["Feature Tables (transforms)"]
            PF["<b>position_features</b><br/><i>Position analysis & Stockfish eval comparison</i>"]
            SIF["<b>search_iteration_features</b><br/><i>Derived per-iteration metrics<br/>(nps, ebf, qratio, stability)</i>"]
            STF["<b>search_tree_features</b><br/><i>Derived per-ply metrics<br/>(ratios, ebf per tree depth)</i>"]
        end

        subgraph Wide["Wide Fact Table"]
            SF["<b>search_features</b><br/><i>Main denormalized wide table<br/>for dashboarding & analysis</i>"]
        end

        MIG2["<b>schema_migrations</b><br/><i>Applied migrations</i>"]

        SS --> SIF
        IDS --> SIF

        STS2 --> STF

        SS --> PF
        DIM --> PF

        SS --> SF
        SIF --> SF
        ST --> SF
        PF --> SF
        ENG2 --> SF
        GS --> SF
    end

    style SF fill:#1a0a2e,stroke:#ff6b35,stroke-width:3px,color:#fff
    style SS fill:#1a1f2e,stroke:#00d2ff,stroke-width:2px,color:#fff
    style IDS fill:#1a1f2e,stroke:#00d2ff,stroke-width:2px,color:#fff
    style STS2 fill:#1a1f2e,stroke:#00d2ff,stroke-width:2px,color:#fff
    style ST fill:#1a1f2e,stroke:#00d2ff,stroke-width:2px,color:#fff
    style RM2 fill:#1a1f2e,stroke:#00d2ff,stroke-width:2px,color:#fff
    style PF fill:#1a2e1a,stroke:#7fff6b,stroke-width:2px,color:#fff
    style SIF fill:#1a2e1a,stroke:#7fff6b,stroke-width:2px,color:#fff
    style STF fill:#1a2e1a,stroke:#7fff6b,stroke-width:2px,color:#fff
    style ENG2 fill:#2d1b00,stroke:#f7b731,stroke-width:2px,color:#fff
    style GS fill:#2d1b00,stroke:#f7b731,stroke-width:2px,color:#fff
    style DIM fill:#2d1b00,stroke:#f7b731,stroke-width:2px,color:#fff
    style MIG2 fill:#1a1f2e,stroke:#8892a4,stroke-width:1px,color:#aaa
```

| Layer | Tables | Purpose |
|-------|--------|---------|
| Dimensions (7) | `engines`, `experiments`, `engine_ratings`, `game_stats`, `dim_positions`, `sprt_runs`, `sts_runs` | Context/lookup tables |
| Facts (5) | `search_stats`, `iterative_deepening_stats`, `search_tree_stats`, `search_timings`, `root_moves` | Raw search measurement data |
| Features (3) | `position_features`, `search_iteration_features`, `search_tree_features` | Derived metrics via Python + SQL transforms |
| Wide Fact (1) | `search_features` | Denormalized join of all above — primary dashboard source |

Arrows indicate data flow into derived tables.
