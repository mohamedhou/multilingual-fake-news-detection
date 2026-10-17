"""Train a reproducible TF-IDF + Logistic Regression baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def normalize_label(value):
    """Normalize common fake/real label encodings to stable strings."""
    raw = str(value).strip().lower()
    mapping = {
        "1": "fake", "fake": "fake", "false": "fake",
        "0": "real", "real": "real", "true": "real",
    }
    return mapping.get(raw, raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/news.csv")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--language-column", default="language")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. See README.md for the required CSV format."
        )

    df = pd.read_csv(data_path)
    for column in (args.text_column, args.label_column):
        if column not in df.columns:
            raise ValueError(f"Required column '{column}' not found. Columns: {list(df.columns)}")

    df = df[[args.text_column, args.label_column] +
            ([args.language_column] if args.language_column in df.columns else [])].copy()
    df = df.dropna(subset=[args.text_column, args.label_column])
    df[args.text_column] = df[args.text_column].astype(str).str.strip()
    df = df[df[args.text_column].str.len() > 0]
    df["__label__"] = df[args.label_column].map(normalize_label)

    if df["__label__"].nunique() < 2:
        raise ValueError("At least two distinct classes are required.")
    if df["__label__"].value_counts().min() < 2:
        raise ValueError("Each class needs at least two examples for a stratified split.")

    X_train, X_test, y_train, y_test = train_test_split(
        df[args.text_column],
        df["__label__"],
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["__label__"],
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            max_features=200_000,
            min_df=2,
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight=None,
            random_state=args.seed,
        )),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0
    )
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    labels = sorted(df["__label__"].unique())
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    metrics = {
        "dataset": str(data_path),
        "n_rows_after_cleaning": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "seed": args.seed,
        "test_size": args.test_size,
        "labels_order": labels,
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "note": "Baseline only; random article-level split may overestimate generalization if duplicates or source leakage exist."
    }

    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    joblib.dump(model, "models/tfidf_logreg.joblib")
    Path("reports/baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "confusion_matrix": metrics["confusion_matrix"],
        "labels_order": labels,
    }, indent=2))
    print("\nSaved model: models/tfidf_logreg.joblib")
    print("Saved metrics: reports/baseline_metrics.json")


if __name__ == "__main__":
    main()
