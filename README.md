# Exam-LLM

Code and anonymized input data associated with the manuscript:

**Ranking proximity between student and Large Language Model answers in academic exams**  
Adrien Deroubaix and Inga Labuhn

Repository: <https://github.com/AdrienBax/Exam-LLM>

## Purpose

This repository provides a transparent and reproducible workflow to rank the textual proximity between anonymized student exam answers and LLM-generated reference answers.

The method is based on classical text-similarity tools used in plagiarism-detection contexts:

- TF-IDF vectorization;
- unigrams and bigrams;
- cosine similarity;
- t-SNE visualization.

The goal is **not** to provide a black-box AI detector or proof of misconduct. The workflow is intended as an exploratory and transparent support tool for instructors, allowing student answers to be positioned relative to a corpus of LLM-generated answers.

## Repository structure

```text
Exam-LLM/
├── data/
│   ├── students/
│   │   ├── E01.txt
│   │   ├── E02.txt
│   │   ├── ...
│   │   ├── E17.txt
│   │   └── META.txt
│   ├── chatgpt10.txt
│   ├── chatgpt20.txt
│   ├── chatgpt30.txt
│   ├── claude10.txt
│   ├── claude20.txt
│   ├── claude30.txt
│   ├── deepseek10.txt
│   ├── deepseek20.txt
│   ├── deepseek30.txt
│   ├── gemini10.txt
│   ├── gemini20.txt
│   ├── gemini30.txt
│   ├── grok10.txt
│   ├── grok20.txt
│   ├── grok30.txt
│   ├── mistral10.txt
│   ├── mistral20.txt
│   └── mistral30.txt
├── src/
│   ├── llm_exam_proximity.py
│   └── count_responses.py
├── scripts/
│   ├── run_10_responses.sh
│   ├── run_20_responses.sh
│   ├── run_30_responses.sh
│   └── count_10_responses.sh
├── docs/
├── paper/
├── outputs/
├── requirements.txt
├── environment.yml
├── CITATION.cff
├── LICENSE
└── README.md
```

## Data organization

The LLM-generated answers are stored directly in `data/`. The suffix `10`, `20`, or `30` indicates the number of generated answers per LLM configuration.

Student answers are stored in `data/students/` and are already anonymized at the filename level:

```text
data/students/E01.txt
data/students/E02.txt
...
data/students/E17.txt
```

The file `META.txt` is intentionally kept with its explicit label because it corresponds to a seeded control answer, not to a real student identity.

The private anonymization map linking original filenames or student names to `E01`, `E02`, etc. is **not included** in this repository and must not be committed.

## Ethical use of student data

The anonymized student answer files are provided only for research transparency, reproducibility, and teaching-oriented methodological discussion.

They must not be used to identify, profile, rank, or sanction individual students. The similarity scores produced by this workflow are exploratory indicators and should not be interpreted as proof of academic misconduct.

## Method summary

For each question `R1` to `R4`, the workflow:

1. reads LLM answers and anonymized student/control answers from `.txt` files;
2. extracts answer blocks labelled `R1`, `R2`, `R3`, and `R4`;
3. builds a TF-IDF representation using unigrams and bigrams;
4. computes cosine similarities in TF-IDF space;
5. ranks student/control proximity to the LLM answer set;
6. projects the corpus using t-SNE for visualization;
7. exports figures and CSV tables.

The TF-IDF vectorizer uses:

```python
TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    max_df=0.95,
    min_df=1,
    sublinear_tf=True,
)
```

This means that the text representation uses **unigrams and bigrams**, not trigrams or longer n-grams.

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate exam-llm
```

## Run the analysis

For the 30-answer-per-LLM configuration used for the main figure:

```bash
python src/llm_exam_proximity.py \
  --data-dir data \
  --students-dir data/students \
  --n-llm-responses 30 \
  --output-dir outputs/output_30LLM
```

Or simply:

```bash
bash scripts/run_30_responses.sh
```

For the supplementary sensitivity tests using 20 or 10 answers per LLM:

```bash
bash scripts/run_20_responses.sh
bash scripts/run_10_responses.sh
```

## Count input responses

To check the number of detected `R1` to `R4` blocks:

```bash
bash scripts/count_10_responses.sh
```

The output is written to:

```text
outputs/counts/input_response_counts_10LLM.csv
```

## Outputs

The main script writes:

```text
outputs/output_30LLM/
├── Figure_all_questions_4panels.png
├── Figure_all_questions_4panels.pdf
├── R1_tsne_coordinates.csv
├── R1_full_similarity_matrix.csv
├── scores_R1.csv
├── ...
├── scores_all_questions.csv
└── summary_all_questions.csv
```

The same structure is produced for `output_20LLM` and `output_10LLM`.

## Important privacy note

The repository intentionally does **not** contain:

```text
student_anonymization_map.csv
```

This file must remain private. The `.gitignore` is configured to prevent accidental upload of anonymization maps, generated outputs, local caches, archives, and Windows system files.

## License

The code is released under the MIT License.

The input text data are provided for research transparency and reproducibility under the restrictions described in the ethical-use section above.

## Citation

If you use this repository, please cite the associated manuscript:

```text
Deroubaix, A. and Labuhn, I. Ranking proximity between student and Large Language Model answers in academic exams. Manuscript submitted to Assessment & Evaluation in Higher Education.
```

A DOI and formal citation will be added after publication.
