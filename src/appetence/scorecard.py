"""Grille de score : discrétisation, Weight of Evidence (WoE), Information Value (IV), régression logistique
et conversion en points.

Convention : l'événement est la souscription (y = 1). WoE = ln(%événements / %non-événements) dans la
classe ; une WoE positive signale une appétence supérieure à la moyenne.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

SMOOTHING = 0.5          # évite ln(0) pour une classe sans événement
MIN_SHARE = 0.03         # une classe doit regrouper au moins 3 % de l'échantillon d'apprentissage


def woe_table(bins: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Effectifs, WoE et contribution à l'IV de chaque classe."""
    t = pd.crosstab(bins, y).reindex(columns=[0, 1], fill_value=0)
    t.columns = ["non_evenements", "evenements"]
    dist_e = (t["evenements"] + SMOOTHING) / (t["evenements"].sum() + SMOOTHING * len(t))
    dist_n = (t["non_evenements"] + SMOOTHING) / (t["non_evenements"].sum() + SMOOTHING * len(t))
    t["taux"] = t["evenements"] / (t["evenements"] + t["non_evenements"])
    t["woe"] = np.log(dist_e / dist_n)
    t["iv"] = (dist_e - dist_n) * t["woe"]
    return t


@dataclass
class Binning:
    """Discrétisation d'une variable apprise sur l'échantillon d'apprentissage."""
    name: str
    kind: str                              # "numeric" ou "categorical"
    edges: np.ndarray | None = None        # bornes (numérique)
    groups: dict = field(default_factory=dict)   # modalité → classe (catégoriel)

    def transform(self, x: pd.Series) -> pd.Series:
        if self.kind == "numeric":
            # Classes ]a ; b] : une valeur très fréquente (ex. −1 = « jamais contacté ») garde sa propre classe.
            return pd.Series(np.digitize(x.to_numpy(float), self.edges[1:-1], right=True), index=x.index)
        return x.map(self.groups).fillna("autres")


def fit_binning(x: pd.Series, kind: str, max_bins: int = 6) -> Binning:
    if kind == "numeric":
        # Bornes intérieures seulement : si 93 % des valeurs valent −1, la borne −1 est conservée.
        inner = np.unique(np.quantile(x, np.linspace(0, 1, max_bins + 1)[1:-1]))
        edges = np.r_[-np.inf, inner, np.inf]
        binning = Binning(x.name, kind, edges=edges)
        # Fusionne les classes trop petites avec leur voisine de droite (ou de gauche pour la dernière).
        while True:
            share = binning.transform(x).value_counts(normalize=True).sort_index()
            small = share[share < MIN_SHARE]
            if small.empty or len(binning.edges) <= 3:
                return binning
            i = int(small.index[0])
            drop = i + 1 if i + 1 < len(binning.edges) - 1 else i
            binning.edges = np.delete(binning.edges, drop)
    share = x.value_counts(normalize=True)
    groups = {m: (m if s >= MIN_SHARE else "autres") for m, s in share.items()}
    return Binning(x.name, kind, groups=groups)


class Scorecard:
    """WoE + régression logistique, puis points : +PDO points quand l'appétence (en cote) double."""

    def __init__(self, numeric: list[str], categorical: list[str], min_iv: float = 0.02,
                 pdo: float = 20.0, base_score: float = 600.0, base_odds: float = 1 / 19,
                 exclude: tuple[str, ...] = ()):
        self.numeric = [c for c in numeric if c not in exclude]
        self.categorical = [c for c in categorical if c not in exclude]
        self.min_iv = min_iv
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "Scorecard":
        self.binnings, self.tables, self.iv = {}, {}, {}
        for col in self.numeric + self.categorical:
            b = fit_binning(df[col], "numeric" if col in self.numeric else "categorical")
            t = woe_table(b.transform(df[col]), y)
            self.binnings[col], self.tables[col], self.iv[col] = b, t, float(t["iv"].sum())
        self.selected = [c for c, v in sorted(self.iv.items(), key=lambda kv: -kv[1]) if v >= self.min_iv]
        # Sur des WoE, chaque coefficient doit être positif. Un signe négatif trahit une colinéarité :
        # la grille contredirait alors la WoE (classe favorable pénalisée). On retire la variable et on réestime.
        self.dropped_sign: list[str] = []
        while True:
            self.model = LogisticRegression(C=1e6, max_iter=1000).fit(self.woe(df), y)
            coefs = pd.Series(self.model.coef_[0], index=self.selected)
            if (coefs > 0).all() or len(self.selected) == 1:
                return self
            worst = coefs.idxmin()
            self.selected.remove(worst)
            self.dropped_sign.append(worst)

    def woe(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in self.selected:
            woe = self.tables[col]["woe"]
            out[col] = self.binnings[col].transform(df[col]).map(woe).fillna(0.0)   # classe inconnue : WoE neutre
        return pd.DataFrame(out, index=df.index)

    def logit(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.decision_function(self.woe(df))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return 1 / (1 + np.exp(-self.logit(df)))

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Score en points : offset + facteur × ln(cote d'appétence)."""
        return self.offset + self.factor * self.logit(df)

    def points_table(self) -> pd.DataFrame:
        """Points apportés par chaque classe de chaque variable retenue (l'ordonnée à l'origine est répartie)."""
        n = len(self.selected)
        rows = []
        for coef, col in zip(self.model.coef_[0], self.selected):
            for cls, row in self.tables[col].iterrows():
                pts = self.factor * (coef * row["woe"] + self.model.intercept_[0] / n) + self.offset / n
                rows.append({"variable": col, "classe": self.describe(col, cls), "effectif": int(row["evenements"] +
                             row["non_evenements"]), "taux": row["taux"], "woe": row["woe"], "points": pts})
        return pd.DataFrame(rows)

    def describe(self, col: str, cls) -> str:
        b = self.binnings[col]
        if b.kind == "categorical":
            return str(cls)
        lo, hi = b.edges[int(cls)], b.edges[int(cls) + 1]
        if np.isinf(lo):
            return f"≤ {hi:g}"
        return f"> {lo:g}" if np.isinf(hi) else f"]{lo:g} ; {hi:g}]"
