#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi

if [ -d ".venv/bin" ]; then
  source .venv/bin/activate
fi

exec python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload