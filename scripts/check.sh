#!/usr/bin/env bash
# Fast feedback: lint, then every test (about two minutes on a laptop CPU).
set -euo pipefail
cd "$(dirname "$0")/.."
ruff check .
ruff format --check tcs world evals tests scripts
python3 -m pytest -q
