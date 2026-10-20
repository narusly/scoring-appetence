"""Indicateurs de performance et de stabilité d'un score."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score


def ks_statistic(y: np.ndarray, score: np.ndarray) -> float:
    """Écart maximal entre les fonctions de répartition du score chez les souscripteurs et les autres."""
    y = np.asarray(y)
    return float(stats.ks_2samp(score[y == 1], score[y == 0]).statistic)


def capture_rate(y: np.ndarray, score: np.ndarray, share: float) -> float:
    """Part des souscripteurs touchés en ne contactant que la fraction `share` des clients les mieux scorés."""
    y = np.asarray(y)
    order = np.argsort(-np.asarray(score), kind="stable")
    top = order[: int(round(share * len(y)))]
    return float(y[top].sum() / y.sum())


def gains_curve(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    order = np.argsort(-np.asarray(score), kind="stable")
    captured = np.r_[0, np.cumsum(y[order])] / y.sum()
    contacted = np.arange(len(y) + 1) / len(y)
    return contacted, captured


def performance(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    auc = roc_auc_score(y, proba)
    return {"AUC": auc, "Gini": 2 * auc - 1, "KS": ks_statistic(y, proba), "Brier": brier_score_loss(y, proba),
            "capture 10 %": capture_rate(y, proba, 0.10), "capture 20 %": capture_rate(y, proba, 0.20)}


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index entre une population de référence et une population récente.

    Repères usuels : < 0,10 stable, 0,10 à 0,25 dérive modérée, > 0,25 dérive forte.
    Numérique : déciles de la référence ; catégoriel : modalités.
    """
    if pd.api.types.is_numeric_dtype(expected) and expected.nunique() > bins:
        # Classes ]a ; b] : une valeur très fréquente (ex. −1 = « jamais contacté ») forme sa propre classe.
        inner = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)[1:-1]))
        e = pd.Series(np.digitize(expected, inner, right=True)).value_counts(normalize=True)
        a = pd.Series(np.digitize(actual, inner, right=True)).value_counts(normalize=True)
    else:
        e = expected.astype(str).value_counts(normalize=True)
        a = actual.astype(str).value_counts(normalize=True)
    idx = e.index.union(a.index)
    e = e.reindex(idx, fill_value=0) + 1e-4
    a = a.reindex(idx, fill_value=0) + 1e-4
    return float(((a - e) * np.log(a / e)).sum())


