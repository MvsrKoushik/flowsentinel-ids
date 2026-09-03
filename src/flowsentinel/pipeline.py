from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from .drift import population_stability_index


def clean_flows(frame: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.Series]:
    if label not in frame:
        raise ValueError(f"label column '{label}' is missing")
    y = frame[label].astype(str).str.strip()
    X = frame.drop(columns=[label]).select_dtypes(include=np.number).replace([np.inf, -np.inf], np.nan)
    if X.empty:
        raise ValueError("no numeric feature columns found")
    return X, y


def train_baseline(frame: pd.DataFrame, label: str, output: Path) -> dict[str, object]:
    X, y = clean_flows(frame, label)
    split = int(len(frame) * 0.8)
    if split < 2 or split >= len(frame):
        raise ValueError("at least three ordered rows are required")
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42, n_jobs=-1)),
    ])
    model.fit(X.iloc[:split], y.iloc[:split])
    prediction = model.predict(X.iloc[split:])
    metrics: dict[str, object] = {
        "balanced_accuracy": float(balanced_accuracy_score(y.iloc[split:], prediction)),
        "f1_macro": float(f1_score(y.iloc[split:], prediction, average="macro")),
        "classification_report": classification_report(y.iloc[split:], prediction, output_dict=True, zero_division=0),
        "psi": {column: population_stability_index(X[column].iloc[:split].dropna(), X[column].iloc[split:].dropna()) for column in X if X[column].notna().any()},
    }
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
