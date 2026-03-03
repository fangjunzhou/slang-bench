# Slang Bench

A Slang benchmark report on the performance of linear layers in Slang.

## Getting Started

### Requirements

- Python 3.12+
- `uv` (recommended) or `pip`
- A working Slang/SlangPy-compatible runtime
- Optional (for report build): LaTeX toolchain with `pdflatex` and `biber`

### Installation

#### Option A: Install with uv (recommended)

```bash
uv sync
```

This creates/syncs the project environment from `pyproject.toml`, including the workspace dependency `external/slangpy-nn`.

#### Option B: Install with pip (no uv)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e external/slangpy-nn
pip install -e .
```

### Usage

Activate your environment first:

- If using `uv`: `uv run ...`
- If using `pip` + venv: `source .venv/bin/activate`

Run all benchmarks:

```bash
pytest tests
```

Or with `uv`:

```bash
uv run pytest tests
```

Common `pytest-benchmark` options:

```bash
# Save full benchmark output to JSON
pytest tests --benchmark-json report/benchmark.json

# Group and sort benchmark table output
pytest tests --benchmark-group-by=fullname --benchmark-sort=mean

# Filter which tests to run
pytest tests -k "linear_layer and not atomic"

# Change rounds/warmup behavior
pytest tests --benchmark-min-rounds=50 --benchmark-warmup=on
```

## Report

- Main source: `report/main.tex`
- Build from `report/`:

```bash
make
```

- Output PDF: [`report/main.pdf`](report/main.pdf)
