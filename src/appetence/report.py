"""Analyse complète : `python -m appetence.report` écrit `reports/resultats.md` et les figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from . import style
from .data import CATEGORICAL, NUMERIC, ROOT, TARGET, load, time_split
from .metrics import gains_curve, performance, psi
from .scorecard import Scorecard

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
SEED, BLOCKS = 0, 4
LABELS = {"age": "Âge", "balance": "Solde moyen", "campaign": "Contacts (campagne)", "pdays": "Jours depuis le contact",
          "previous": "Contacts antérieurs", "job": "Profession", "marital": "Situation familiale",
          "education": "Études", "default": "Défaut de crédit", "housing": "Prêt immobilier", "loan": "Prêt personnel",
          "contact": "Canal de contact", "month": "Mois du contact", "poutcome": "Résultat campagne précédente"}
VALUES = {"jan": "janvier", "feb": "février", "mar": "mars", "apr": "avril", "may": "mai", "jun": "juin",
          "jul": "juillet", "aug": "août", "sep": "septembre", "oct": "octobre", "nov": "novembre",
          "dec": "décembre", "married": "marié(e)", "single": "célibataire", "divorced": "divorcé(e)",
          "autres": "autres modalités (< 3 %)"}
MODELS = {"scorecard": "Grille de score (WoE + logistique)", "xgboost": "XGBoost"}
COLORS = {"scorecard": style.SERIES[0], "xgboost": style.SERIES[1]}


def fr(x: float, decimals: int = 3) -> str:
    return f"{x:,.{decimals}f}".replace(",", " ").replace(".", ",")


def pct(x: float, decimals: int = 0) -> str:
    return f"{fr(100 * x, decimals)} %"


def one_hot(df: pd.DataFrame, columns: pd.Index | None = None) -> pd.DataFrame:
    X = pd.get_dummies(df[NUMERIC + CATEGORICAL], columns=CATEGORICAL, dtype=float)
    return X if columns is None else X.reindex(columns=columns, fill_value=0.0)


def fit_models(train: pd.DataFrame) -> dict:
    """Grille de score et XGBoost ; l'arrêt précoce de XGBoost utilise les 15 % les plus récents de l'apprentissage."""
    sc = Scorecard(NUMERIC, CATEGORICAL).fit(train, train[TARGET])
    cut = int(len(train) * 0.85)
    X = one_hot(train)
    xgb = XGBClassifier(n_estimators=2000, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                        min_child_weight=5, eval_metric="auc", early_stopping_rounds=100, random_state=SEED)
    xgb.fit(X.iloc[:cut], train[TARGET].iloc[:cut], eval_set=[(X.iloc[cut:], train[TARGET].iloc[cut:])], verbose=False)
    return {"scorecard": sc, "xgboost": xgb, "columns": X.columns}


def proba(models: dict, key: str, df: pd.DataFrame) -> np.ndarray:
    if key == "scorecard":
        return models["scorecard"].predict_proba(df)
    return models["xgboost"].predict_proba(one_hot(df, models["columns"]))[:, 1]


def rolling_evaluation(df: pd.DataFrame, n_train: int, static: dict) -> pd.DataFrame:
    """Découpe la période de test en blocs. Pour chaque bloc : modèle figé contre modèle réentraîné sur
    toutes les données antérieures au bloc (fenêtre croissante)."""
    rows = []
    for k, idx in enumerate(np.array_split(np.arange(n_train, len(df)), BLOCKS), start=1):
        block = df.iloc[idx]
        y = block[TARGET].to_numpy()
        rolled = fit_models(df.iloc[:idx[0]])
        for key in MODELS:
            for mode, models in (("figé", static), ("réentraîné", rolled)):
                p = proba(models, key, block)
                rows.append({"bloc": k, "modèle": key, "mode": mode, "AUC": roc_auc_score(y, p),
                             "taux observé": y.mean(), "taux prédit": p.mean()})
    return pd.DataFrame(rows)


def fig_drift(df: pd.DataFrame, n_train: int, psis: pd.Series, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.1, 1]})
    k, n = 10, len(df)
    rates = [df[TARGET].iloc[i * n // k:(i + 1) * n // k].mean() for i in range(k)]
    centers = np.arange(k) + 0.5
    colors = [style.SERIES[0] if (i + 1) * n // k <= n_train else style.SERIES[1] for i in range(k)]
    ax1.bar(centers, rates, width=0.86, color=colors)
    for c, r in zip(centers, rates):
        ax1.annotate(pct(r), (c, r), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8,
                     color=style.INK_2)
    split = n_train / n * k
    ax1.axvline(split, color=style.INK_2, linewidth=0.9)
    for text, dx, ha in (("apprentissage", -6, "right"), ("test hors temps", 6, "left")):
        ax1.annotate(text, (split, max(rates) * 0.9), xytext=(dx, 0), textcoords="offset points", ha=ha,
                     fontsize=8.5, color=style.INK_2)
    ax1.set_xticks(centers, [str(i + 1) for i in range(k)])
    ax1.set_xlabel("Dixième de la période (mai 2008 → novembre 2010)")
    ax1.yaxis.set_major_formatter(lambda v, _: pct(v))
    ax1.grid(axis="x", visible=False)
    ax1.set_title("Taux de souscription au fil des campagnes", fontsize=10.5)

    s = psis.sort_values()
    ax2.barh([LABELS[c] for c in s.index], s.to_numpy(), color=style.SERIES[0], height=0.7)
    for y, v in enumerate(s.to_numpy()):
        ax2.annotate(fr(v, 2), (v, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
                     color=style.INK_2)
    for x, text in ((0.10, "0,10"), (0.25, "0,25 : dérive forte")):
        ax2.axvline(x, color=style.INK_2, linewidth=0.8)
        ax2.annotate(text, (x, -0.9), xytext=(3, 0), textcoords="offset points", fontsize=8, color=style.INK_2)
    ax2.grid(axis="y", visible=False)
    ax2.set_xlim(0, s.max() * 1.18)
    ax2.xaxis.set_major_formatter(lambda v, _: fr(v, 2))
    ax2.set_title("PSI des variables : apprentissage contre test", fontsize=10.5)
    fig.suptitle("La population et la réponse aux campagnes changent d'une période à l'autre", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_rolling(roll: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, key in zip(axes, MODELS):
        for mode, color in (("figé", style.SERIES[0]), ("réentraîné", style.SERIES[1])):
            r = roll[(roll["modèle"] == key) & (roll["mode"] == mode)]
            ax.plot(r["bloc"], r["AUC"], color=color, marker="o", markersize=5, label=f"Modèle {mode}")
            ax.annotate(fr(r["AUC"].iloc[-1], 2), (r["bloc"].iloc[-1], r["AUC"].iloc[-1]), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=8.5, color=style.INK_2)
        ax.axhline(0.5, color=style.BASELINE, linewidth=0.9)
        ax.annotate("hasard", (1, 0.5), xytext=(0, 4), textcoords="offset points", fontsize=8, color=style.MUTED)
        ax.set_xticks(range(1, BLOCKS + 1), [f"bloc {i}" for i in range(1, BLOCKS + 1)])
        ax.set_title(MODELS[key], fontsize=10.5)
    axes[0].set_ylabel("AUC sur le bloc")
    axes[0].yaxis.set_major_locator(MultipleLocator(0.05))
    axes[0].yaxis.set_major_formatter(lambda v, _: fr(v, 2))
    axes[0].legend(loc="upper center")
    fig.suptitle("Période de test découpée en 4 blocs : un modèle figé se dégrade, le réentraîner le maintient",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_gains(y: np.ndarray, probas: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot([0, 1], [0, 1], color=style.BASELINE, linewidth=1)
    ax.annotate("Ciblage au hasard", (0.62, 0.575), fontsize=8.5, color=style.MUTED, rotation=33)
    for key, p in probas.items():
        x, g = gains_curve(y, p)
        ax.plot(x, g, color=COLORS[key], label=MODELS[key])
    best = max(probas, key=lambda k: np.interp(0.2, *gains_curve(y, probas[k])))
    g20 = float(np.interp(0.2, *gains_curve(y, probas[best])))
    ax.axvline(0.2, color=style.INK_2, linewidth=0.8)
    ax.annotate(f"En appelant 20 % des clients,\n{pct(g20)} des souscripteurs sont touchés", (0.2, g20),
                xytext=(10, -34), textcoords="offset points", fontsize=8.5, color=style.INK_2)
    ax.xaxis.set_major_formatter(lambda v, _: pct(v))
    ax.yaxis.set_major_formatter(lambda v, _: pct(v))
    ax.set_xlabel("Part des clients contactés, du meilleur score au moins bon")
    ax.set_ylabel("Part des souscripteurs touchés")
    ax.set_title("Courbe de gain sur toute la période de test (modèles figés)")
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def fig_iv(sc: Scorecard, path: Path) -> None:
    s = pd.Series(sc.iv).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = [style.SERIES[0] if c in sc.selected else style.BASELINE for c in s.index]
    ax.barh([LABELS[c] for c in s.index], s.to_numpy(), color=colors, height=0.7)
    for y, (c, v) in enumerate(s.items()):
        note = " · retirée : signe incohérent" if c in sc.dropped_sign else ""
        ax.annotate(fr(v, 3) + note, (v, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
                    color=style.INK_2)
    ax.axvline(0.02, color=style.INK_2, linewidth=0.8)
    ax.annotate("seuil : 0,02", (0.02, -0.9), xytext=(4, 0), textcoords="offset points", fontsize=8,
                color=style.INK_2)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, s.max() * 1.45)
    ax.xaxis.set_major_formatter(lambda v, _: fr(v, 1))
    ax.set_title("Information Value des variables (en bleu : retenues dans la grille)")
    fig.savefig(path)
    plt.close(fig)


def perf_table(rows: list[dict]) -> str:
    t = pd.DataFrame(rows)
    for col in ("AUC", "Gini", "KS", "Brier"):
        t[col] = t[col].map(lambda v: fr(v, 3))
    for col in ("capture 10 %", "capture 20 %"):
        t[col] = t[col].map(lambda v: pct(v))
    return t.to_markdown(index=False)


def main() -> None:
    style.apply()
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load()
    train, test = time_split(df)
    y_test = test[TARGET].to_numpy()

    # 1. Validation hors temps (apprentissage sur le passé) contre découpage aléatoire (pratique courante).
    static = fit_models(train)
    probas = {k: proba(static, k, test) for k in MODELS}
    rows = [{"validation": "hors temps", "modèle": MODELS[k], **performance(y_test, p)} for k, p in probas.items()]
    a, b = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df[TARGET])
    random_models = fit_models(a.sort_index())
    rows += [{"validation": "aléatoire", "modèle": MODELS[k],
              **performance(b[TARGET].to_numpy(), proba(random_models, k, b))} for k in MODELS]

    # 2. Dérive entre apprentissage et test.
    psis = pd.Series({c: psi(train[c], test[c]) for c in NUMERIC + CATEGORICAL})

    # 3. Modèle figé contre réentraînement à fenêtre croissante, bloc par bloc.
    roll = rolling_evaluation(df, len(train), static)

    # 4. Exclure d'office les variables instables ? Repérées sans regarder le test : PSI entre la première
    #    et la seconde moitié de l'apprentissage.
    half = len(train) // 2
    intra = {c: psi(train[c].iloc[:half], train[c].iloc[half:]) for c in NUMERIC + CATEGORICAL}
    unstable = tuple(c for c, v in intra.items() if v > 0.25)
    stable_sc = Scorecard(NUMERIC, CATEGORICAL, exclude=unstable).fit(train, train[TARGET])
    auc_stable = roc_auc_score(y_test, stable_sc.predict_proba(test))

    sc = static["scorecard"]
    fig_drift(df, len(train), psis, FIGURES / "derive.png")
    fig_rolling(roll, FIGURES / "reentrainement.png")
    fig_gains(y_test, probas, FIGURES / "gains.png")
    fig_iv(sc, FIGURES / "information_value.png")

    roll_md = roll.assign(**{"modèle": roll["modèle"].map(MODELS), "AUC": roll["AUC"].map(lambda v: fr(v, 3)),
                             "taux observé": roll["taux observé"].map(lambda v: pct(v, 1)),
                             "taux prédit": roll["taux prédit"].map(lambda v: pct(v, 1))})
    points = sc.points_table()
    points = points[points["variable"].isin(sc.selected[:3])].copy()
    points["variable"] = points["variable"].map(LABELS)
    points["classe"] = points["classe"].map(lambda v: VALUES.get(v, v))
    points["taux"] = points["taux"].map(lambda v: pct(v, 1))
    points["woe"] = points["woe"].map(lambda v: fr(v, 2))
    points["points"] = points["points"].map(lambda v: fr(v, 0))
    lines = [
        "# Résultats",
        "",
        f"{fr(len(df), 0)} clients contactés ; apprentissage sur les {fr(len(train), 0)} premiers "
        f"(taux de souscription {pct(train[TARGET].mean(), 1)}), test sur les {fr(len(test), 0)} suivants "
        f"({pct(test[TARGET].mean(), 1)}). Variables exclues : `duration` (fuite d'information) et `day`.",
        "",
        "## Performance : validation hors temps contre découpage aléatoire",
        "",
        perf_table(rows),
        "",
        "Capture 20 % : part des souscripteurs touchés en n'appelant que les 20 % de clients les mieux scorés "
        "(20 % au hasard).",
        "",
        "## Dérive entre apprentissage et test (PSI)",
        "",
        pd.DataFrame({"variable": [LABELS[c] for c in psis.sort_values(ascending=False).index],
                      "PSI": [fr(v, 3) for v in psis.sort_values(ascending=False)]}).to_markdown(index=False),
        "",
        "## Modèle figé contre réentraînement (fenêtre croissante), par bloc de la période de test",
        "",
        roll_md.to_markdown(index=False),
        "",
        "## Exclure les variables instables ?",
        "",
        "Variables au PSI > 0,25 entre les deux moitiés de l'apprentissage : "
        + ", ".join(LABELS[c] for c in unstable) + f". Grille sans ces variables : AUC hors temps "
        f"{fr(auc_stable, 3)}, contre {fr(rows[0]['AUC'], 3)} pour la grille complète.",
        "",
        "## Grille de score",
        "",
        "Variables retenues (IV ≥ 0,02, coefficients positifs), par IV décroissante : "
        + ", ".join(f"{LABELS[c]} ({fr(sc.iv[c], 3)})" for c in sc.selected) + ". "
        + ("Retirée pour signe incohérent (colinéarité) : " + ", ".join(LABELS[c] for c in sc.dropped_sign) + "."
           if sc.dropped_sign else ""),
        "",
        "Points des trois variables les plus informatives (600 points pour une cote de 1 contre 19, "
        "+20 points quand la cote double) :",
        "",
        points[["variable", "classe", "effectif", "taux", "woe", "points"]].to_markdown(index=False),
        "",
    ]
    (REPORTS / "resultats.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
