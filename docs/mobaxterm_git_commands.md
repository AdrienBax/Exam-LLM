# Rebuild and push the repository from MobaXterm

These commands assume that the repository folder is located at:

```bash
/home/mobaxterm/MyDocuments/Exam-LLM
```

## 1. Check the repository

```bash
cd /home/mobaxterm/MyDocuments/Exam-LLM
pwd
ls
```

## 2. Clean Windows files if needed

```bash
find . -name desktop.ini -delete
```

## 3. Add files

```bash
git add README.md LICENSE .gitignore requirements.txt environment.yml src scripts docs data/README.md data/students/.gitkeep data/raw/.gitkeep outputs/.gitkeep
```

If anonymized `.txt` input data are ready to publish:

```bash
git add data/*.txt data/students/*.txt
```

If Git says the files are ignored but you are sure they are anonymized:

```bash
git add -f data/*.txt data/students/*.txt
```

## 4. Commit

```bash
git commit -m "Update repository structure and input data policy"
```

## 5. Push

```bash
git push -u origin main
```

If the remote repository already contains files and you want to replace it entirely:

```bash
git push -u origin main --force
```

Use force push only if you are sure there is nothing important on GitHub that is not present locally.
