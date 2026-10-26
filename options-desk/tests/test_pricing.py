import math

import pytest

from optdesk.pricing import binomial_american_future, black76_greeks, black76_price, implied_vol, prob_itm, strike_for_delta


def test_put_call_parity():
    F, K, T, s = 4800, 4700, 0.2, 0.25
    c = black76_price(F, K, T, s, "C")
    p = black76_price(F, K, T, s, "P")
    assert c - p == pytest.approx(F - K, abs=1e-6)


def test_known_value():
    # Black-76 ATM 近似：价格 ≈ 0.4 * F * sigma * sqrt(T)
    F, T, s = 100.0, 1.0, 0.2
    c = black76_price(F, F, T, s, "C")
    assert c == pytest.approx(0.4 * F * s * math.sqrt(T), rel=0.01)


def test_implied_vol_roundtrip():
    F, K, T = 7200, 7400, 0.15
    for s in (0.12, 0.25, 0.6):
        for cp in ("C", "P"):
            px = black76_price(F, K, T, s, cp)
            assert implied_vol(px, F, K, T, cp) == pytest.approx(s, abs=1e-6)


def test_implied_vol_below_intrinsic_is_none():
    assert implied_vol(50, 4800, 4900, 0.1, "C") is not None
    assert implied_vol(90, 4800, 4700, 0.1, "C") is None  # 低于内在价值 100


def test_greeks_bounds():
    g = black76_greeks(4800, 4800, 0.1, 0.25, "C")
    assert 0.45 < g.delta < 0.6
    assert g.gamma > 0 and g.vega > 0 and g.theta < 0
    gp = black76_greeks(4800, 4800, 0.1, 0.25, "P")
    assert g.delta - gp.delta == pytest.approx(1.0, abs=1e-9)


def test_american_premium_small_but_nonnegative():
    F, K, T, s = 4800, 5200, 0.2, 0.25
    eu = black76_price(F, K, T, s, "P")
    am = binomial_american_future(F, K, T, s, "P", steps=300)
    assert am >= eu - 1e-6
    assert am - eu < 0.02 * eu + 1  # 期货期权提前行权溢价很小


def test_prob_and_delta_strike():
    assert 0.4 < prob_itm(4800, 4800, 0.1, 0.25, "C") < 0.5
    k = strike_for_delta(4800, 0.1, 0.25, "C", 0.25)
    assert black76_greeks(4800, k, 0.1, 0.25, "C").delta == pytest.approx(0.25, abs=0.01)
