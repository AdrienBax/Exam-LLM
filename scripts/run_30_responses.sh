#!/usr/bin/env bash
set -euo pipefail

python src/llm_exam_proximity.py   --data-dir data   --students-dir data/students   --n-llm-responses 30   --output-dir outputs/output_30LLM
