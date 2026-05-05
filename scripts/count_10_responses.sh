#!/usr/bin/env bash
set -euo pipefail

python src/count_responses.py \
  --data-dir data \
  --students-dir data/students \
  --n-llm-responses 10 \
  --output-dir outputs/counts_10LLM
