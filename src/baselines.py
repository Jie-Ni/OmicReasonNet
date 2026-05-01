"""Classical-ML baselines (RF, NB, optional XGBoost) and a MOGONET-lite re-impl."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize


def _safe_auc(y_true: np.ndarray, prob: np.ndarray, n_classes: int) -> float:
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, prob[:, 1]))
        yb = label_binarize(y_true, classes=list(range(n_classes)))
        return float(roc_auc_score(yb, prob, multi_class="ovr", average="weighted"))
    except Exception:
        return float("nan")


def fit_eval_classical(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray,
    n_classes: int, model: str = "rf", weights: np.ndarray | None = None, seed: int = 42,
) -> dict:
    """Fit RF/NB on Xtr (optionally feature-weighted) and report metrics on Xte."""
    if weights is not None:
        Xtr = Xtr * weights[None, :]
        Xte = Xte * weights[None, :]
    if model == "rf":
        clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    elif model == "nb":
        clf = GaussianNB()
    else:
        raise ValueError(model)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    prob = clf.predict_proba(Xte)
    return {
        "model": model,
        "acc": float(accuracy_score(yte, pred)),
        "f1_w": float(f1_score(yte, pred, average="weighted")),
        "f1_m": float(f1_score(yte, pred, average="macro")),
        "auc": _safe_auc(yte, prob, n_classes),
    }
