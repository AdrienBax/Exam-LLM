# Input data

This folder contains the input corpus used by the analysis scripts.

## Expected structure after extraction

Extract `data/raw/data.rar` into the `data/` directory so that the repository contains:

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
├── grok10.txt
├── grok20.txt
├── grok30.txt
├── mistral10.txt
├── mistral20.txt
├── mistral30.txt
└── students/
    ├── UE1_partiel-*.txt
    └── ...
```

The script expects student files to be located in `data/students/` and LLM reference answers to be named as `<model><n>.txt`, for example `chatgpt10.txt` or `claude30.txt`.

## Privacy note

Student names are anonymized by the analysis script into `E01`, `E02`, etc. Special seeded or test entries such as `META`, `ALIBABA`, `IBM`, `KIMI`, and `CHEAT` are intentionally preserved in the figures and output tables.
