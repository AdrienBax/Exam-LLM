#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM exam proximity analysis.

This script compares anonymized student exam answers with LLM-generated answers.

Input structure expected:
    data/
    ├── chatgpt10.txt, chatgpt20.txt, chatgpt30.txt
    ├── claude10.txt,  claude20.txt,  claude30.txt
    ├── deepseek10.txt, ...
    ├── gemini10.txt, ...
    ├── mistral10.txt, ...
    ├── grok10.txt, ...
    └── students/
        ├── E01.txt
        ├── E02.txt
        ├── ...
        └── META.txt

Important:
    Student filenames are assumed to be already anonymized.
    No anonymization map is read or written by this script.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_MODELS = ["ChatGPT", "Claude", "DeepSeek", "Gemini", "Mistral", "Grok"]

MODEL_FILE_PREFIX = {
    "ChatGPT": "chatgpt",
    "Claude": "claude",
    "DeepSeek": "deepseek",
    "Gemini": "gemini",
    "Mistral": "mistral",
    "Grok": "grok",
}

LLM_COLORS = {
    "ChatGPT": "red",
    "Claude": "blue",
    "DeepSeek": "green",
    "Gemini": "orange",
    "Mistral": "purple",
    "Grok": "brown",
}

PANEL_TITLES = {
    1: "A) Question 1",
    2: "B) Question 2",
    3: "C) Question 3",
    4: "D) Question 4",
}

TSNE_RANDOM_STATE = 42
TOP_K_NEIGHBORS = 5
TOP_STUDENTS_IN_BOX = 21

STUDENT_SIZE_MIN = 70
STUDENT_SIZE_MAX = 260

FIGSIZE = (20, 14)
TITLE_FONTSIZE = 19
AXIS_LABEL_FONTSIZE = 15
TICK_LABELSIZE = 13
ANNOTATION_FONTSIZE = 12
RANKING_FONTSIZE = 14
LEGEND_FONTSIZE = 14

LLM_MARKER_SIZE = 100
LLM_MARKER_LINEWIDTH = 2.0
AXIS_LINEWIDTH = 1.5
ZERO_LINEWIDTH = 1.4


# ============================================================
# TEXT CLEANING AND READING
# ============================================================

def clean_text(text) -> str:
    """Normalize spacing, line breaks, and invisible characters."""
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r", "").replace("\ufeff", "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_txt_robust(path: Path) -> str:
    """Read a text file using several fallback encodings."""
    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return clean_text(path.read_text(encoding=encoding))
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not read {path}") from last_error


# ============================================================
# PARSING ANSWERS
# ============================================================

def split_variants(text: str) -> List[Tuple[str, str]]:
    """
    Split LLM files into variants if they contain COPIE A/B/C/D markers.
    Otherwise return a single variant named ALL.
    """
    text = clean_text(text)

    if re.search(r"\bCOPIE\s+A\b", text, flags=re.IGNORECASE):
        variants = []
        for label in ["A", "B", "C", "D"]:
            match = re.search(
                rf"\bCOPIE\s+{label}\b(.*?)(?=\bCOPIE\s+[A-D]\b|$)",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if match:
                variants.append((f"COPIE_{label}", clean_text(match.group(1))))
        return variants if variants else [("ALL", text)]

    return [("ALL", text)]


_R_MARK = re.compile(r"(?:^|\n)\s*R\s*([1-4])\s*[:\-\u2014]?\s*", flags=re.IGNORECASE)


def extract_R_blocks(text: str) -> Dict[int, List[str]]:
    """
    Extract R1..R4 answer blocks.

    Returns:
        {1: [...], 2: [...], 3: [...], 4: [...]}
    """
    text = clean_text(text)
    matches = list(_R_MARK.finditer(text))

    out = {1: [], 2: [], 3: [], 4: []}
    if not matches:
        return out

    for i, match in enumerate(matches):
        rnum = int(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        chunk = clean_text(text[start:end])
        if chunk:
            out[rnum].append(chunk)

    return out


# ============================================================
# INPUT DISCOVERY
# ============================================================

def build_llm_file_map(data_dir: Path, n_llm_responses: int) -> Dict[str, Path]:
    """
    Build the LLM file dictionary for 10, 20, or 30 generated answers per model.
    """
    ia_files = {}
    for model_name in DEFAULT_MODELS:
        prefix = MODEL_FILE_PREFIX[model_name]
        path = data_dir / f"{prefix}{n_llm_responses}.txt"
        ia_files[model_name] = path
    return ia_files


def list_student_files(students_dir: Path) -> List[Path]:
    """
    Return all anonymized student files.

    The filename stem is used directly as the public student id:
        E01.txt  -> E01
        META.txt -> META
    """
    return sorted(
        [p for p in students_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"],
        key=lambda p: p.stem.lower(),
    )


# ============================================================
# CORPUS CONSTRUCTION
# ============================================================

def build_R_corpus(
    ia_files: Dict[str, Path],
    students_dir: Path,
    rnum: int,
) -> pd.DataFrame:
    """Build the corpus for one question."""
    rows = []

    # LLM answers
    for model_name, path in ia_files.items():
        if not path.exists():
            print(f"Warning: missing LLM file: {path}")
            continue

        raw_text = read_txt_robust(path)
        variants = split_variants(raw_text)

        for variant_id, variant_text in variants:
            responses = extract_R_blocks(variant_text)
            for j, text in enumerate(responses.get(rnum, []), start=1):
                rows.append({
                    "type": "IA",
                    "group": model_name,
                    "id": f"{model_name}::{variant_id}::R{rnum}::{j}",
                    "text": text,
                    "source_file": path.name,
                })

    # Anonymized student answers
    for path in list_student_files(students_dir):
        raw_text = read_txt_robust(path)
        responses = extract_R_blocks(raw_text)
        texts = responses.get(rnum, [])

        public_id = path.stem

        for j, text in enumerate(texts, start=1):
            rows.append({
                "type": "STUDENT",
                "group": "Students",
                "id": f"{public_id}#{j}" if len(texts) > 1 else public_id,
                "text": text,
                "source_file": path.name,
            })

    df = pd.DataFrame(rows)

    if len(df) > 0:
        df["text"] = df["text"].apply(clean_text)
        df = df[df["text"].str.len() > 0].copy().reset_index(drop=True)

    return df


# ============================================================
# PROXIMITY METRICS
# ============================================================

def compute_student_ai_scores(df: pd.DataFrame, X, rnum: int, top_k: int = 5) -> pd.DataFrame:
    """Compute proximity metrics in the original TF-IDF space."""
    sim_all = cosine_similarity(X)
    type_upper = df["type"].str.upper()

    ia_indices = np.where((type_upper == "IA").values)[0]
    st_indices = np.where((type_upper == "STUDENT").values)[0]

    if len(ia_indices) == 0 or len(st_indices) == 0:
        raise ValueError(f"R{rnum}: missing LLM or student texts.")

    rows = []

    for i in st_indices:
        sims = sim_all[i].copy()
        sims[i] = -np.inf

        sims_ai = sims[ia_indices]
        best_ai_idx = ia_indices[int(np.argmax(sims_ai))]
        best_ai_similarity = float(sims[best_ai_idx])

        ranked_idx = np.argsort(sims)[::-1]
        topk_idx = ranked_idx[:top_k]
        topk_types = df.iloc[topk_idx]["type"].tolist()
        ai_in_topk = sum(t == "IA" for t in topk_types)

        other_students = st_indices[st_indices != i]
        mean_similarity_to_ai = float(np.mean(sims[ia_indices]))
        mean_similarity_to_students = (
            float(np.mean(sims[other_students])) if len(other_students) > 0 else np.nan
        )

        ratio_ai_students = (
            mean_similarity_to_ai / mean_similarity_to_students
            if pd.notna(mean_similarity_to_students) and mean_similarity_to_students != 0
            else np.nan
        )

        rows.append({
            "R": rnum,
            "student_id": df.iloc[i]["id"],
            "best_similarity_to_ai": best_ai_similarity,
            "closest_ai_id": df.iloc[best_ai_idx]["id"],
            "closest_ai_model": df.iloc[best_ai_idx]["group"],
            "ai_in_topk": ai_in_topk,
            "top_k": top_k,
            "mean_similarity_to_ai": mean_similarity_to_ai,
            "mean_similarity_to_students": mean_similarity_to_students,
            "ai_to_students_similarity_ratio": ratio_ai_students,
        })

    return pd.DataFrame(rows).sort_values("best_similarity_to_ai", ascending=False)


# ============================================================
# FIGURE HELPERS
# ============================================================

def build_ranking_box_text(scores: pd.DataFrame, top_n: int = 21) -> str:
    """Build the ranking text shown in each panel."""
    if scores.empty:
        return "Closest to AI\nNo student scores"

    lines = ["Closest to AI", "Student Score  Model", "----------------------------"]

    for _, row in scores.head(top_n).iterrows():
        sid = str(row["student_id"])
        sim = float(row["best_similarity_to_ai"])
        model = str(row["closest_ai_model"])
        lines.append(f"{sid:>8}  {sim:>4.2f}  {model:<8}")

    return "\n".join(lines)


def compute_student_marker_sizes(
    df_plot: pd.DataFrame,
    scores: pd.DataFrame,
    size_min: float = 70,
    size_max: float = 260,
) -> np.ndarray:
    """Assign student marker sizes from their best LLM similarity."""
    size_map = {
        row["student_id"]: row["best_similarity_to_ai"]
        for _, row in scores.iterrows()
    }

    sims = np.array([
        float(size_map.get(row["id"], 0.0)) if str(row["type"]).upper() == "STUDENT" else np.nan
        for _, row in df_plot.iterrows()
    ], dtype=float)

    valid = sims[~np.isnan(sims)]
    sizes = np.full(len(df_plot), size_min, dtype=float)

    if len(valid) == 0:
        return sizes

    smin = float(np.min(valid))
    smax = float(np.max(valid))

    if smax > smin:
        norm = (valid - smin) / (smax - smin)
        scaled = size_min + norm * (size_max - size_min)
    else:
        scaled = np.full(len(valid), 0.5 * (size_min + size_max))

    sizes[~np.isnan(sims)] = scaled
    return sizes


def style_axis(ax):
    """Apply consistent axis styling."""
    ax.tick_params(axis="both", labelsize=TICK_LABELSIZE, width=AXIS_LINEWIDTH)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINEWIDTH)


# ============================================================
# ANALYSIS
# ============================================================

def analyze_question(df: pd.DataFrame, rnum: int, out_dir: Path) -> Dict:
    """Compute TF-IDF, cosine similarity, t-SNE coordinates, and scores."""
    if len(df) < 5:
        raise ValueError(f"Not enough texts for R{rnum}.")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=1,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(df["text"])
    sim_all = cosine_similarity(X)

    scores = compute_student_ai_scores(df, X, rnum, top_k=TOP_K_NEIGHBORS)

    X_dense = X.toarray()
    n = len(df)
    perplexity = min(10, max(3, (n - 1) // 3))
    perplexity = min(perplexity, n - 1)

    Z = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=TSNE_RANDOM_STATE,
        method="exact",
        max_iter=1000,
    ).fit_transform(X_dense)

    df_plot = df.copy()
    df_plot["tsne_1"] = Z[:, 0]
    df_plot["tsne_2"] = Z[:, 1]

    df_plot.to_csv(out_dir / f"R{rnum}_tsne_coordinates.csv", index=False, encoding="utf-8")

    sim_df = pd.DataFrame(sim_all, index=df["id"], columns=df["id"])
    sim_df.to_csv(out_dir / f"R{rnum}_full_similarity_matrix.csv", encoding="utf-8")

    scores = scores.merge(
        df_plot[["id", "tsne_1", "tsne_2"]],
        left_on="student_id",
        right_on="id",
        how="left",
    ).drop(columns=["id"])

    scores.to_csv(out_dir / f"scores_R{rnum}.csv", index=False, encoding="utf-8")

    return {
        "rnum": rnum,
        "df_plot": df_plot,
        "scores": scores,
        "ranking_text": build_ranking_box_text(scores, top_n=TOP_STUDENTS_IN_BOX),
        "student_sizes": compute_student_marker_sizes(df_plot, scores),
        "sim_all": sim_all,
    }


def plot_four_panels(results_by_question: Dict[int, Dict], out_dir: Path):
    """Create one 2x2 figure for questions 1..4."""
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    axes = axes.flatten()

    for ax, rnum in zip(axes, [1, 2, 3, 4]):
        if rnum not in results_by_question:
            ax.axis("off")
            ax.text(0.5, 0.5, f"No data for Question {rnum}", ha="center", va="center", fontsize=16)
            continue

        result = results_by_question[rnum]
        df_plot = result["df_plot"]
        ranking_text = result["ranking_text"]
        student_sizes = result["student_sizes"]

        Zx = df_plot["tsne_1"].to_numpy()
        Zy = df_plot["tsne_2"].to_numpy()

        xmin_data, xmax_data = np.min(Zx), np.max(Zx)
        ymin_data, ymax_data = np.min(Zy), np.max(Zy)

        xrange_data = max(xmax_data - xmin_data, 1e-6)
        yrange_data = max(ymax_data - ymin_data, 1e-6)

        xmin_plot = xmin_data - 0.08 * xrange_data
        xmax_plot = xmax_data + 1.10 * xrange_data
        ymin_plot = ymin_data - 0.10 * yrange_data
        ymax_plot = ymax_data + 0.10 * yrange_data

        for group_name in df_plot[df_plot["type"] == "IA"]["group"].unique():
            idx = ((df_plot["type"] == "IA") & (df_plot["group"] == group_name)).to_numpy()
            ax.scatter(
                Zx[idx], Zy[idx],
                marker="x",
                s=LLM_MARKER_SIZE,
                linewidths=LLM_MARKER_LINEWIDTH,
                color=LLM_COLORS.get(group_name, None),
                label=f"LLM: {group_name}",
                alpha=0.95,
            )

        idx_students = (df_plot["type"] == "STUDENT").to_numpy()
        ax.scatter(
            Zx[idx_students], Zy[idx_students],
            marker="o",
            s=student_sizes[idx_students],
            color="grey",
            alpha=0.80,
            edgecolors="none",
            label="Students",
        )

        for i in np.where(idx_students)[0]:
            ax.annotate(
                df_plot.loc[i, "id"],
                (Zx[i], Zy[i]),
                fontsize=ANNOTATION_FONTSIZE,
                alpha=0.97,
            )

        ax.set_xlim(xmin_plot, xmax_plot)
        ax.set_ylim(ymin_plot, ymax_plot)

        if xmin_plot < 0 < xmax_plot:
            ax.axvline(0, linestyle="--", linewidth=ZERO_LINEWIDTH, color="gray")
        if ymin_plot < 0 < ymax_plot:
            ax.axhline(0, linestyle="--", linewidth=ZERO_LINEWIDTH, color="gray")

        ax.text(
            0.98, 0.50,
            ranking_text,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=RANKING_FONTSIZE,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.40", facecolor="white", edgecolor="gray", alpha=0.95),
        )

        ax.set_title(PANEL_TITLES[rnum], fontsize=TITLE_FONTSIZE, fontweight="bold")
        ax.set_xlabel("t-SNE dimension 1", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("t-SNE dimension 2", fontsize=AXIS_LABEL_FONTSIZE)

        style_axis(ax)

    handles, labels, seen = [], [], set()
    for result in results_by_question.values():
        df_plot = result["df_plot"]
        for group_name in df_plot[df_plot["type"] == "IA"]["group"].unique():
            label = f"LLM: {group_name}"
            if label not in seen:
                seen.add(label)
                handles.append(
                    plt.Line2D(
                        [0], [0],
                        marker="x",
                        color=LLM_COLORS.get(group_name, "black"),
                        linestyle="None",
                        markersize=10,
                        markeredgewidth=2.0,
                    )
                )
                labels.append(label)

    handles.append(plt.Line2D([0], [0], marker="o", color="grey", linestyle="None", markersize=9))
    labels.append("Students")

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
        bbox_to_anchor=(0.5, 0.02),
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_dir / "Figure_all_questions_4panels.png", dpi=320, bbox_inches="tight")
    fig.savefig(out_dir / "Figure_all_questions_4panels.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# COMMAND LINE
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare anonymized student answers with LLM-generated answers."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--students-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-llm-responses", type=int, choices=[10, 20, 30], default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = args.data_dir
    students_dir = args.students_dir or (data_dir / "students")
    output_dir = args.output_dir or Path("outputs") / f"output_{args.n_llm_responses}LLM"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")
    if not students_dir.exists():
        raise FileNotFoundError(f"Missing student directory: {students_dir}")

    student_files = list_student_files(students_dir)
    if not student_files:
        raise FileNotFoundError(f"No .txt files found in {students_dir}")

    print("Student files used as public IDs:")
    for path in student_files:
        print(f"  - {path.name} -> {path.stem}")

    ia_files = build_llm_file_map(data_dir, args.n_llm_responses)

    all_scores = []
    summary_rows = []
    results_by_question = {}

    for rnum in [1, 2, 3, 4]:
        print(f"\n=== Processing R{rnum} ===")
        df_r = build_R_corpus(ia_files, students_dir, rnum)
        print(f"Total texts (LLM + students): {len(df_r)}")

        if len(df_r) < 5:
            print(f"Skipped R{rnum}: not enough texts.")
            continue

        n_ai = int((df_r["type"] == "IA").sum())
        n_students = int((df_r["type"] == "STUDENT").sum())

        try:
            result_r = analyze_question(df_r, rnum, output_dir)
            results_by_question[rnum] = result_r

            scores_r = result_r["scores"]
            all_scores.append(scores_r)

            summary_rows.append({
                "R": rnum,
                "n_texts_total": len(df_r),
                "n_ai": n_ai,
                "n_students": n_students,
                "mean_best_similarity_to_ai": scores_r["best_similarity_to_ai"].mean(),
                "max_best_similarity_to_ai": scores_r["best_similarity_to_ai"].max(),
            })

            print(f"Top students closest to an LLM answer for R{rnum}:")
            print(scores_r.head(10).to_string(index=False))

        except Exception as exc:
            print(f"Error while processing R{rnum}: {exc}")

    if results_by_question:
        plot_four_panels(results_by_question, output_dir)
        print(f"\nCombined figure written to: {output_dir}")

    if all_scores:
        scores_all = pd.concat(all_scores, ignore_index=True)
        scores_all.to_csv(output_dir / "scores_all_questions.csv", index=False, encoding="utf-8")
        print(f"Combined score file written: {output_dir / 'scores_all_questions.csv'}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / "summary_all_questions.csv", index=False, encoding="utf-8")
        print(f"Summary written: {output_dir / 'summary_all_questions.csv'}")
        print(summary_df.to_string(index=False))
    else:
        print("\nNo scores generated.")


if __name__ == "__main__":
    main()
