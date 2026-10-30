#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r backend/requirements-core.txt
python backend/scripts/check_env.py
python -m pytest backend/tests -q
cat <<'MSG'

Core API dependencies and tests are ready.
Optional pretrained-vision stack:
  source .venv/bin/activate
  python -m pip install -r backend/requirements-vision.txt
  python backend/scripts/check_env.py
MSG
