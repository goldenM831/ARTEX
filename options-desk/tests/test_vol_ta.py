import datetime as dt

import numpy as np
import pandas as pd
import pytest

from optdesk.feeds.replay import SyntheticFeed
from optdesk.ta import analyze, macd, rsi, swing_levels
from optdesk.vol import detect_vol_regime, hv_close, hv_yang_zhang, iv_percentile, iv_rank, vol_cone


@pytest.fixture(scope="module")
def bars():
    return SyntheticFeed("TA701", start_price=4800, vol=0.22, seed=3, asof=dt.datetime(2026, 10, 26)).get_bars("TA701")


def test_hv_recovers_true_vol(bars):
    hv = hv_close(bars["close"], 250).dropna().iloc[-1]
    assert 0.15 < hv < 0.30
    yz = hv_yang_zhang(bars, 60).dropna().iloc[-1]
    assert 0.08 < yz < 0.5


def test_hv_constant_series_is_zero():
    s = pd.Series(np.full(100, 4800.0))
    assert hv_close(s, 20).dropna().iloc[-1] == 0


def test_vol_cone_shape(bars):
    cone = vol_cone(bars["close"])
    assert "p50" in cone.columns and "current" in cone.columns
    assert (cone["p5"] <= cone["p95"]).all()


def test_iv_rank_percentile():
    s = pd.Series(np.linspace(0.1, 0.3, 101))
    assert iv_rank(s, 0.2) == pytest.approx(0.5)
    assert iv_percentile(s, 0.31) == pytest.approx(1.0)
    assert iv_percentile(s, 0.2) == pytest.approx(0.5, abs=0.01)


def test_regime_detection_bias(bars):
    sig_cheap = detect_vol_regime(bars, iv_atm=0.10)
    sig_rich = detect_vol_regime(bars, iv_atm=0.45)
    assert sig_cheap.bias == "buyer"
    assert sig_rich.bias == "seller"
    assert sig_rich.regime in ("squeeze", "normal", "expansion", "spike")


def test_spike_detected_on_shock(bars):
    df = bars.copy()
    last = df.index[-1]
    shock = df["close"].iloc[-1] * 0.12
    for i in range(1, 4):
        idx = last + pd.Timedelta(days=i)
        c = df["close"].iloc[-1] * (1 - 0.06)
        df.loc[idx] = [c, c + shock, c - shock, c, 1e6]
    assert detect_vol_regime(df).regime == "spike"


def test_ta_outputs(bars):
    snap = analyze(bars)
    assert -1 <= snap.score <= 1
    assert snap.label in ("看涨", "偏多", "中性", "偏空", "看跌")
    assert 0 <= snap.rsi14 <= 100
    res, sup = swing_levels(bars)
    assert res and sup
    assert len(macd(bars["close"])) == len(bars)
    assert rsi(bars["close"]).dropna().between(0, 100).all()


def test_trend_scores_direction():
    idx = pd.bdate_range(end="2026-10-26", periods=200)
    up = pd.Series(np.linspace(4000, 5200, 200), index=idx)
    df = pd.DataFrame({"open": up, "high": up * 1.003, "low": up * 0.997, "close": up, "volume": 1})
    assert analyze(df).score > 0.3
    dn = df.iloc[::-1].set_index(idx)
    assert analyze(dn).score < -0.3
