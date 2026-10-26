import datetime as dt
import math

import numpy as np
import pytest

from optdesk import strategies as S
from optdesk.contracts import default_underlying, estimate_option_expiry, format_option_code, get_product, parse_contract
from optdesk.feeds.replay import SyntheticFeed
from optdesk.recommender import Intent, Recommender


@pytest.fixture(scope="module")
def chain():
    feed = SyntheticFeed("TA701", start_price=4800, vol=0.22, asof=dt.datetime(2026, 10, 26, 14, 0))
    return feed.get_chain("TA701").compute_iv()


@pytest.fixture(scope="module")
def bars():
    return SyntheticFeed("TA701", start_price=4800, vol=0.22, asof=dt.datetime(2026, 10, 26, 14, 0)).get_bars("TA701")


def test_contract_parsing():
    c = parse_contract("TA701C4800")
    assert (c.product, c.year, c.month, c.cp, c.strike) == ("TA", 2027, 1, "C", 4800.0)
    d = parse_contract("pp2601-P-7000")
    assert (d.product, d.exchange, d.cp, d.strike) == ("PP", "DCE", "P", 7000.0)
    s = parse_contract("SHFE.ru2605C15000")
    assert (s.product, s.year, s.month, s.strike) == ("RU", 2026, 5, 15000.0)
    assert format_option_code("TA", 2027, 1, "P", 4600) == "TA701P4600"
    assert format_option_code("PP", 2026, 1, "C", 7000) == "pp2601-C-7000"


def test_expiry_rules():
    # 2027-01 合约：郑商所 2026-12 第 3 个交易日 = 12/3（周四）；大商所第 5 个 = 12/7；上期所倒数第 5 个 = 12/25
    assert estimate_option_expiry("TA", 2027, 1) == dt.date(2026, 12, 3)
    assert estimate_option_expiry("PP", 2027, 1) == dt.date(2026, 12, 7)
    assert estimate_option_expiry("RU", 2027, 1) == dt.date(2026, 12, 25)
    u = default_underlying("TA", dt.date(2026, 10, 26))
    assert u == "TA701"


def test_strike_steps():
    spec = get_product("TA")
    assert spec.strike_step(4800) == 50
    assert spec.strike_step(6000) == 100
    assert spec.round_strike(4823) == 4800


def test_chain_iv_recovers_smile(chain):
    atm = chain.atm_iv()
    assert 0.2 < atm < 0.3
    assert not chain.sanity_issues() or all("平价" not in i for i in chain.sanity_issues())
    assert len(chain.liquid()) > 10


def test_bull_call_spread_bounds(chain):
    s = S.bull_call_spread(chain, 10, long_strike=4800, short_strike=5000)
    m = s.evaluate(chain.F, chain.T, 0.25)
    width = 200 * 5 * 10
    assert m.net_premium_total < 0                       # 借方
    assert m.max_loss == pytest.approx(m.net_premium_total)
    assert m.max_profit == pytest.approx(width + m.net_premium_total)
    assert m.margin_estimate == 0
    assert 4800 < m.breakevens[0] < 5000
    assert 0 < m.prob_profit < 1


def test_bear_call_spread_credit(chain):
    s = S.bear_call_spread(chain, 10, short_strike=4900)
    m = s.evaluate(chain.F, chain.T, 0.25)
    assert m.net_premium_total > 0
    assert m.max_profit == pytest.approx(m.net_premium_total)
    assert m.margin_estimate > 0


def test_covered_call_metrics(chain):
    s = S.covered_call(chain, 5000, 4700, hedge_base="spot", target_price=4900)
    m = s.evaluate(chain.F, chain.T, 0.25)
    short = s.option_legs()[0]
    assert short.qty == -1000 and short.strike == 4900
    assert math.isinf(m.max_loss) and m.max_loss < 0
    assert m.upside_cap == 4900
    assert m.annualized_yield > 0
    # 到期在 4900 以上收益封顶
    pay = s.payoff(np.array([5200.0, 6000.0]))
    assert pay[0] == pytest.approx(pay[1])
    tickets = s.order_tickets()
    assert len(tickets) == 1 and tickets[0]["买卖"] == "卖" and tickets[0]["手数"] == 1000


def test_covered_call_futures_base_margin_is_only_futures(chain):
    s = S.covered_call(chain, 5000, 4700, hedge_base="future", target_price=4900)
    m = s.evaluate(chain.F, chain.T, 0.25)
    spec = chain.product
    assert m.margin_estimate == pytest.approx(1000 * chain.F * spec.multiplier * spec.futures_margin_rate)


def test_collar_locks_range(chain):
    s = S.collar(chain, 5000, 4700, floor_price=4600, cap_price=5000)
    m = s.evaluate(chain.F, chain.T, 0.25)
    pay = s.payoff(np.array([3000.0, 4600.0, 5000.0, 9000.0]))
    assert pay[0] == pytest.approx(pay[1])
    assert pay[2] == pytest.approx(pay[3])
    assert math.isfinite(m.max_loss) and math.isfinite(m.max_profit)


def test_recommender_end_to_end(chain, bars):
    for role in ("inventory", "procurement", "directional"):
        rep = Recommender().recommend(chain, bars, Intent(role=role, tons=2000, spot_cost=4700))
        assert rep.strategies, role
        top = rep.strategies[0]
        assert top.metrics is not None and top.order_tickets()
        d = rep.to_dict()
        assert d["strategies"][0]["legs"]
        txt = rep.to_text()
        assert "委托" in txt and top.name in txt


def test_inventory_role_prefers_hedges(chain, bars):
    rep = Recommender().recommend(chain, bars, Intent(role="inventory", tons=5000, spot_cost=4700))
    assert rep.strategies[0].hedge_tons() > 0


def test_budget_filters(chain, bars):
    rep = Recommender().recommend(chain, bars, Intent(role="inventory", tons=5000, max_premium_per_ton=10))
    assert all(s.metrics.net_premium_per_ton >= -10 for s in rep.strategies)
