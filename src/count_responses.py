#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Count R1..R4 responses in LLM and student input files."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from llm_exam_proximity import (
    DEFAULT_MODELS,
    build_ia_files,
    build_anonymization_map,
    extract_R_blocks,
    read_txt_robust,
    split_variants,
)


def count_llm_files(data_dir: Path, n_llm_responses: int) -> pd.DataFrame:
    rows = []
    ia_files = build_ia_files(data_dir, n_llm_responses, DEFAULT_MODELS)

    for model, path in ia_files.items():
        if not path.exists():
            rows.append({"type": "IA", "group": model, "source_file": path.name, "R": None, "n_responses": 0, "status": "missing"})
            continue

        raw = read_txt_robust(path)
        variants = split_variants(raw)
        for variant_id, text in variants:
            blocks = extract_R_blocks(text)
            for rnum in [1, 2, 3, 4]:
                rows.append({
                    "type": "IA",
                    "group": model,
                    "variant": variant_id,
                    "source_file": path.name,
                    "R": rnum,
                    "n_responses": len(blocks.get(rnum, [])),
                    "status": "ok",
                })
    return pd.DataFrame(rows)


def count_student_files(students_dir: Path, output_dir: Path) -> pd.DataFrame:
    anon_map, _ = build_anonymization_map(students_dir, output_dir, save_map=False)
    rows = []

    for path in sorted(students_dir.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file() or path.suffix.lower() != ".txt":
            continue
        raw = read_txt_robust(path)
        blocks = extract_R_blocks(raw)
        sid = anon_map.get(path.name, path.stem)
        for rnum in [1, 2, 3, 4]:
            rows.append({
                "type": "STUDENT",
                "student_id": sid,
                "source_file": path.name,
                "R": rnum,
                "n_responses": len(blocks.get(rnum, [])),
                "status": "ok",
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Count LLM and student answers per question.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--students-dir", type=Path, default=Path("data/students"))
    parser.add_argument("--n-llm-responses", type=int, choices=[10, 20, 30], default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/counts"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    llm_counts = count_llm_files(args.data_dir, args.n_llm_responses)
    student_counts = count_student_files(args.students_dir, args.output_dir)
    all_counts = pd.concat([llm_counts, student_counts], ignore_index=True)

    all_counts.to_csv(args.output_dir / f"response_counts_{args.n_llm_responses}LLM.csv", index=False, encoding="utf-8")
    print(all_counts.to_string(index=False))
    print(f"\nWritten: {args.output_dir / f'response_counts_{args.n_llm_responses}LLM.csv'}")


if __name__ == "__main__":
    main()
