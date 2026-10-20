import numpy as np
import pandas as pd
import pytest

from appetence import data
from appetence.metrics import capture_rate, gains_curve, ks_statistic, psi
from appetence.scorecard import SMOOTHING, Scorecard, fit_binning, woe_table


def test_leaking_variable_is_excluded():
    assert "duration" in data.EXCLUDED
    assert "duration" not in data.NUMERIC + data.CATEGORICAL


def test_time_split_keeps_order():
    df = pd.DataFrame({"t": range(10), "y": [0] * 10})
    train, test = data.time_split(df, 0.7)
    assert train["t"].max() < test["t"].min()
    assert len(train) == 7


def test_woe_by_hand():
    bins = pd.Series(["a"] * 100 + ["b"] * 100)
    y = pd.Series([1] * 30 + [0] * 70 + [1] * 10 + [0] * 90)
    t = woe_table(bins, y)
    e_a = (30 + SMOOTHING) / (40 + 2 * SMOOTHING)
    n_a = (70 + SMOOTHING) / (160 + 2 * SMOOTHING)
    assert t.loc["a", "woe"] == pytest.approx(np.log(e_a / n_a))
    assert t.loc["a", "woe"] > 0 > t.loc["b", "woe"]
    assert (t["iv"] >= 0).all()


def test_binning_merges_small_classes():
    x = pd.Series(np.r_[np.zeros(970), np.arange(30)], name="x").astype(float)
    b = fit_binning(x, "numeric")
    shares = b.transform(x).value_counts(normalize=True)
    assert shares.min() >= 0.03 or len(shares) <= 2
    cat = fit_binning(pd.Series(["a"] * 98 + ["rare"] * 2, name="c"), "categorical")
    assert cat.transform(pd.Series(["rare", "jamais vu"])).tolist() == ["autres", "autres"]


def test_scorecard_points_follow_pdo():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=4000), "c": rng.choice(["u", "v", "w"], 4000)})
    logit = 1.5 * df["x"] + (df["c"] == "u") - 2
    y = pd.Series((rng.random(4000) < 1 / (1 + np.exp(-logit))).astype(int))
    sc = Scorecard(numeric=["x"], categorical=["c"]).fit(df, y)
    lg, s = sc.logit(df), sc.score(df)
    # +20 points quand la cote d'appétence double.
    assert np.allclose(s - sc.offset, sc.factor * lg)
    assert sc.factor * np.log(2) == pytest.approx(20)
    # La somme des points par classe redonne le score.
    table = sc.points_table()
    first = df.iloc[[0]]
    total = sum(table[(table.variable == col) & (table.classe == sc.describe(col, sc.binnings[col].transform(first[col]).iloc[0]))]
                ["points"].iloc[0] for col in sc.selected)
    assert total == pytest.approx(sc.score(first)[0])


def test_ks_capture_and_gains():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    perfect = -np.arange(10)
    assert ks_statistic(y, perfect) == pytest.approx(1.0)
    assert capture_rate(y, perfect, 0.2) == pytest.approx(1.0)
    contacted, captured = gains_curve(y, perfect)
    assert captured[2] == pytest.approx(1.0) and contacted[-1] == 1.0


def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    ref = pd.Series(rng.normal(size=5000))
    assert psi(ref, pd.Series(rng.normal(size=5000))) < 0.02
    assert psi(ref, pd.Series(rng.normal(1, 1, size=5000))) > 0.25
    cat = pd.Series(["a"] * 900 + ["b"] * 100)
    assert psi(cat, cat) == pytest.approx(0, abs=1e-9)


def test_psi_isolates_a_dominant_value():
    # 93 % de « jamais contacté » (−1) : la dérive vers des clients déjà contactés doit être vue.
    ref = pd.Series(np.r_[np.full(930, -1.0), np.arange(70.0)])
    recent = pd.Series(np.r_[np.full(500, -1.0), np.arange(500.0)])
    assert psi(ref, recent) > 0.25


def test_binning_keeps_dominant_value_apart():
    x = pd.Series(np.r_[np.full(930, -1.0), np.arange(1.0, 71.0)], name="pdays")
    b = fit_binning(x, "numeric")
    classes = b.transform(x)
    assert classes[x == -1].nunique() == 1
    assert not set(classes[x == -1]) & set(classes[x > -1])


def test_scorecard_drops_variable_with_reversed_sign():
    rng = np.random.default_rng(4)
    n = 6000
    z = rng.normal(size=n)
    df = pd.DataFrame({"a": z + rng.normal(0, 0.3, n), "b": z + rng.normal(0, 0.3, n)})
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-(2 * df["a"] - 0.8 * df["b"] - 2)))).astype(int))
    sc = Scorecard(numeric=["a", "b"], categorical=[]).fit(df, y)
    assert (sc.model.coef_[0] > 0).all()
    assert "b" in sc.dropped_sign
