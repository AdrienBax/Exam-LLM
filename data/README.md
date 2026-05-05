# Input data layout

This directory is expected to contain the textual input files used by the analysis.

## Expected structure

```text
data/
├── chatgpt10.txt
├── chatgpt20.txt
├── chatgpt30.txt
├── claude10.txt
├── claude20.txt
├── claude30.txt
├── deepseek10.txt
├── deepseek20.txt
├── deepseek30.txt
├── gemini10.txt
├── gemini20.txt
├── gemini30.txt
├── mistral10.txt
├── mistral20.txt
├── mistral30.txt
├── grok10.txt
├── grok20.txt
├── grok30.txt
└── students/
    ├── E01.txt
    ├── E02.txt
    ├── E03.txt
    └── ...
```

The script expects each file to contain four blocks labelled `R1`, `R2`, `R3`, and `R4`.

## LLM answer files

The LLM files can contain one or several answer variants. If variants are labelled as `COPIE A`, `COPIE B`, etc., the script automatically splits them into separate reference answers.

## Student answer files

Student files should be anonymized before public release. Recommended file names are:

```text
E01.txt
E02.txt
E03.txt
...
```

Seeded or test names such as `META`, `ALIBABA`, `IBM`, `KIMI`, or `CHEAT` can be preserved when they correspond to artificial controls rather than real students.

## Files that should not be committed

Do not publish the anonymization correspondence table:

```text
student_anonymization_map.csv
```

This file is generated in the output directory and is ignored by Git.

Compressed archives such as `.rar` or `.zip` are also ignored by default. Extract the anonymized `.txt` files instead.
