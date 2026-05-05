#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Count R1..R4 responses in LLM and anonymized student input files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from llm_exam_proximity import (
    build_llm_file_map,
    extract_R_blocks,
    list_student_files,
    read_txt_robust,
    split_variants,
)


def count_llm_files(data_dir: Path, n_llm_responses: int) -> pd.DataFrame:
    rows = []
    llm_files = build_llm_file_map(data_dir, n_llm_responses)

    for model, path in llm_files.items():
        if not path.exists():
            rows.append({
                "type": "IA",
                "group": model,
                "public_id": None,
                "source_file": path.name,
                "R": None,
                "n_responses": 0,
                "status": "missing",
            })
            continue

        raw = read_txt_robust(path)
        variants = split_variants(raw)

        for variant_id, text in variants:
            blocks = extract_R_blocks(text)
            for rnum in [1, 2, 3, 4]:
                rows.append({
                    "type": "IA",
                    "group": model,
                    "public_id": variant_id,
                    "source_file": path.name,
                    "R": rnum,
                    "n_responses": len(blocks.get(rnum, [])),
                    "status": "ok",
                })

    return pd.DataFrame(rows)


def count_student_files(students_dir: Path) -> pd.DataFrame:
    rows = []

    for path in list_student_files(students_dir):
        raw = read_txt_robust(path)
        blocks = extract_R_blocks(raw)
        public_id = path.stem

        for rnum in [1, 2, 3, 4]:
            rows.append({
                "type": "STUDENT",
                "group": "Students",
                "public_id": public_id,
                "source_file": path.name,
                "R": rnum,
                "n_responses": len(blocks.get(rnum, [])),
                "status": "ok",
            })

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count R1..R4 blocks in input files.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--students-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/counts"))
    parser.add_argument("--n-llm-responses", type=int, choices=[10, 20, 30], default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = args.data_dir
    students_dir = args.students_dir or (data_dir / "students")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = pd.concat(
        [
            count_llm_files(data_dir, args.n_llm_responses),
            count_student_files(students_dir),
        ],
        ignore_index=True,
    )

    output_file = output_dir / f"input_response_counts_{args.n_llm_responses}LLM.csv"
    counts.to_csv(output_file, index=False, encoding="utf-8")

    print(counts.to_string(index=False))
    print(f"\nCount table written: {output_file}")


if __name__ == "__main__":
    main()
