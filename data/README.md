# Data directory

This directory contains the input text files used by the analysis.

## LLM files

The LLM files are stored directly in `data/` and follow this naming convention:

```text
chatgpt10.txt, chatgpt20.txt, chatgpt30.txt
claude10.txt,  claude20.txt,  claude30.txt
deepseek10.txt, deepseek20.txt, deepseek30.txt
gemini10.txt, gemini20.txt, gemini30.txt
mistral10.txt, mistral20.txt, mistral30.txt
grok10.txt, grok20.txt, grok30.txt
```

The suffix `10`, `20`, or `30` indicates the number of generated answers per LLM configuration.

## Student files

The student answer files are stored in:

```text
data/students/
```

Their filenames are already anonymized:

```text
E01.txt
E02.txt
...
E17.txt
```

The file `META.txt` is kept with its explicit label because it is a seeded test answer, not a real student identity.

## Private files not included

The private anonymization map linking original student names or filenames to public identifiers is not included and must not be committed:

```text
student_anonymization_map.csv
```
