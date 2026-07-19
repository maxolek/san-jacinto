<div align="center">



# San Jacinto

[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)]()
[![JavaScript](https://img.shields.io/badge/JavaScript-blue.svg)]()

[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)]()

Development, testing, tuning, and analytics framework for the **Tomahawk chess engine**

Tomahawk itself is maintained separately

See `docs/` for more info.

</div>

---

# Overview

San Jacinto provides the infrastructure surrounding engine development:
- SPRT strength testing
- tournament automation
- SPSA parameter tuning
- NNUE tooling
- game/search telemetry processing
- database pipelines
- analytics dashboards

---

## Repository Relationship

```
projects/
│
├── tomahawk/
│   └── Chess engine (C++ / UCI)
│
└── san-jacinto/
    └── Development framework (this repo)
```

San Jacinto communicates with Tomahawk through the UCI protocol and manages the experimentation lifecycle around engine development.

---

# Features

## Testing

Located in:

```
tests/
├── bench.py
├── compare.py
├── perft.py
├── sprt.py
├── sts.py
├── tournament.py
└── release.py
```

---

## Tuning

Located in:

```
tuning/
```

Contains tools for automated parameter optimization.

Current tools:

- SPSA tuning
- CMA-ES tuning
- UCI parameter generation

Used for optimizing search parameters such as:
- LMR reductions
- pruning margins
- evaluation constants
- move ordering parameters

---

## Data Pipeline

Located in:

```
data/
```

The data pipeline handles engine telemetry, game records, and analytics.

Structure:

```
data/
├── databases/
├── etl/
├── transforms/
├── dashboard/
└── openings/
```

---

# NNUE Tools

Located in:

```
nnue/
```

Contains utilities for:

- NNUE data preparation
- training workflows
- network management

The trained networks are consumed by Tomahawk.

---

# Requirements

## Python

Recommended:

```
Python >= 3.11
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## External Dependencies

Required:

- Tomahawk engine repository
- cutechess
- Stockfish (for comparison/testing)
- Python data stack

---

# Configuration

San Jacinto expects Tomahawk to exist alongside this repository:

```
projects/
├── tomahawk/
└── san-jacinto/
```

Engine paths should be configured through the project configuration system.

---

# Development Workflow

Typical engine development cycle:

```
Modify Tomahawk
        |
        v
Build engine
        |
        v
Run San Jacinto tests
        |
        v
SPRT / Tournament validation
        |
        v
Analyze results
        |
        v
Tune parameters
        |
        v
      Repeat
```
