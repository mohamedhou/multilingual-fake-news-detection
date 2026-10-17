# Multilingual Fake News Detection

Research project inspired by *Enhancing Fake News Detection with a Hybrid NLP-Machine Learning Framework*.

## Research goals
- Build reproducible fake-news classification baselines.
- Compare TF-IDF + Naive Bayes, Linear SVM, and Logistic Regression.
- Extend the experiments to English, French, and optionally Arabic.
- Evaluate accuracy, precision, recall, F1-score, confusion matrix, and cross-language generalization.
- Provide a Streamlit demo. Predictions are model classifications, not factual verification.

## Project status
**Phase 0 — repository scaffold.** No dataset or trained model is included yet. Do not report results until experiments have actually been run.

## Requirements
- Python 3.10+
- Git
- A GitHub account (to publish the repository)

## Setup (Windows PowerShell)
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Dataset format
Create `data/raw/news.csv` with these columns:
- `text`: article text
- `label`: `fake` / `real` (or `1` / `0`, where 1 = fake and 0 = real)
- `language`: `en`, `fr`, or `ar` (optional for the first baseline)

Only use datasets whose licenses and terms allow the intended use. Keep original dataset files out of Git unless their license explicitly permits redistribution. The `.gitignore` excludes data files by default.

## Train the first baseline
After adding `data/raw/news.csv`:
```powershell
python -m src.train_baseline --data data/raw/news.csv --text-column text --label-column label
```
The script writes a model to `models/tfidf_logreg.joblib` and evaluation files to `reports/`.

## Run the demo
```powershell
streamlit run app.py
```
The app becomes usable after a model has been trained and saved.

## Git and GitHub
```powershell
git init
git add .
git commit -m "chore: initialize multilingual fake news project"
git branch -M main
```
Then create an **empty** GitHub repository named `multilingual-fake-news-detection` (do not add a README/license there if you already have them locally), and connect it:
```powershell
git remote add origin https://github.com/YOUR-USERNAME/multilingual-fake-news-detection.git
git push -u origin main
```
Replace `YOUR-USERNAME` with your GitHub username.

## Research protocol
1. Verify the dataset source, label definition, license, language distribution, duplicates, and missing values.
2. Create fixed train/validation/test splits with stratification.
3. Establish classical baselines before testing multilingual transformers.
4. Report per-language and macro-averaged metrics; include confusion matrices and error analysis.
5. Keep dataset versions, random seeds, and model configurations documented.
6. Avoid data leakage: near-duplicate articles and syndicated copies must not cross splits.

## Suggested structure
```text
data/       # datasets (not committed)
notebooks/  # exploration and experiments
src/        # reusable code
models/     # trained artifacts (not committed by default)
reports/    # metrics and figures
tests/      # tests
```
