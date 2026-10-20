"""Jeu « Bank Marketing » (UCI, CC BY 4.0) : téléchargement, variables utilisables et découpage temporel.

Moro, S., Cortez, P. & Rita, P. (2014). Campagnes de télémarketing d'une banque portugaise, de mai 2008
à novembre 2010 ; cible : le client souscrit-il un dépôt à terme ? Les lignes sont dans l'ordre chronologique.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "bank-full.csv"
URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
TARGET = "y"
# `duration` (durée de l'appel) n'est connue qu'une fois l'appel terminé : l'utiliser serait une fuite
# d'information, comme le signalent les auteurs du jeu. `day` (jour du mois) n'a pas de sens prédictif stable.
EXCLUDED = ("duration", "day")
NUMERIC = ["age", "balance", "campaign", "pdays", "previous"]
CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]


def download(path: Path = RAW) -> Path:
    """Télécharge l'archive UCI (zip imbriqué) une fois et en extrait `bank-full.csv`."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=120) as resp:
            outer = zipfile.ZipFile(io.BytesIO(resp.read()))
        inner = zipfile.ZipFile(io.BytesIO(outer.read("bank.zip")))
        path.write_bytes(inner.read("bank-full.csv"))
    return path


def load() -> pd.DataFrame:
    df = pd.read_csv(download(), sep=";")
    df[TARGET] = (df[TARGET] == "yes").astype(int)
    return df.drop(columns=list(EXCLUDED))


def time_split(df: pd.DataFrame, train_share: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apprentissage sur le passé, test sur la période suivante (les lignes sont chronologiques)."""
    cut = int(len(df) * train_share)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
