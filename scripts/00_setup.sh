#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e .
if [ ! -d vendor/polyAfilter ]; then
  mkdir -p vendor
  git clone https://github.com/MarekSvob/polyAfilter.git vendor/polyAfilter
fi
git -C vendor/polyAfilter checkout --quiet 5ce0098
git -C vendor/polyAfilter rev-parse --short=7 HEAD
