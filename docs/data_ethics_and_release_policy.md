# Data ethics and release policy

This repository can include two different kinds of text data:

1. LLM-generated answers used as reference corpora.
2. Student answers that have been anonymized before public release.

The anonymization correspondence table linking original student identities to public identifiers must never be committed to this repository.

## Public student identifiers

For real student answers, use non-identifying names such as `E01.txt`, `E02.txt`, etc.

For seeded/test answers that are not real students, explicit labels may be kept. The analysis code preserves the following labels:

```text
META
ALIBABA
IBM
KIMI
CHEAT
```

These labels are treated as special test names and are not remapped to `E01`, `E02`, etc.

## Intended use

The dataset and code are provided for transparency, reproducibility, and teaching/research purposes. The method is a similarity-based exploratory visualization and must not be used as stand-alone evidence of AI use or academic misconduct.
