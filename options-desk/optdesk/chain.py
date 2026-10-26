"""期权链数据模型：统一各行情源（CTP / 天勤 / AkShare / 回放）的快照格式，并计算 IV 与希腊字母。"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Optional

from .contracts import ProductSpec, year_fraction
from .pricing import Greeks, black76_greeks, implied_vol


@dataclass
class Quote:
    symbol: str
    last: float
    bid: float
    ask: float
    bid_vol: int = 0
    ask_vol: int = 0
    volume: int = 0
    open_interest: int = 0
    ts: Optional[dt.datetime] = None

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return 0.5 * (self.bid + self.ask)
        return self.last

    @property
    def spread(self) -> float:
        return self.ask - self.bid if self.bid > 0 and self.ask > 0 else float("inf")

    def is_stale(self, now: Optional[dt.datetime] = None, max_age_sec: float = 60) -> bool:
        if self.ts is None:
            return False
        now = now or dt.datetime.now(self.ts.tzinfo)
        return (now - self.ts).total_seconds() > max_age_sec


@dataclass
class OptionQuote(Quote):
    cp: str = "C"
    strike: float = 0.0
    expiry: Optional[dt.date] = None
    underlying: str = ""
    iv: Optional[float] = None
    greeks: Optional[Greeks] = None

    @property
    def dte(self) -> int:
        return (self.expiry - dt.date.today()).days if self.expiry else 0

    def spread_pct(self) -> float:
        return self.spread / self.mid if self.mid > 0 and math.isfinite(self.spread) else float("inf")


@dataclass
class OptionChain:
    underlying: str
    product: ProductSpec
    futures: Quote
    expiry: dt.date
    options: list[OptionQuote] = field(default_factory=list)
    asof: dt.datetime = field(default_factory=dt.datetime.now)
    r: float = 0.0

    # ------------------------------------------------------------------ 基础
    @property
    def F(self) -> float:
        return self.futures.mid

    @property
    def T(self) -> float:
        return year_fraction(self.asof.date(), self.expiry)

    @property
    def dte(self) -> int:
        return max((self.expiry - self.asof.date()).days, 0)

    def calls(self) -> list[OptionQuote]:
        return sorted((o for o in self.options if o.cp == "C"), key=lambda o: o.strike)

    def puts(self) -> list[OptionQuote]:
        return sorted((o for o in self.options if o.cp == "P"), key=lambda o: o.strike)

    def strikes(self) -> list[float]:
        return sorted({o.strike for o in self.options})

    def get(self, cp: str, strike: float) -> Optional[OptionQuote]:
        for o in self.options:
            if o.cp == cp and abs(o.strike - strike) < 1e-9:
                return o
        return None

    # ------------------------------------------------------------------ 计算
    def compute_iv(self) -> "OptionChain":
        F, T = self.F, self.T
        for o in self.options:
            px = o.mid
            o.iv = implied_vol(px, F, o.strike, T, o.cp, self.r) if px > 0 else None
            o.greeks = black76_greeks(F, o.strike, T, o.iv, o.cp, self.r) if o.iv else None
        return self

    def atm_strike(self) -> float:
        return min(self.strikes(), key=lambda k: abs(k - self.F))

    def atm_iv(self) -> Optional[float]:
        """ATM 隐波：取平值附近 call/put 均值；若两侧其一缺失则取另一侧。"""
        k = self.atm_strike()
        ivs = [o.iv for o in (self.get("C", k), self.get("P", k)) if o and o.iv]
        return sum(ivs) / len(ivs) if ivs else None

    def skew_25d(self) -> Optional[float]:
        """25 delta 风险逆转：IV(25d put) - IV(25d call)。正值表示下跌保护更贵。"""
        c = self.nearest_delta("C", 0.25)
        p = self.nearest_delta("P", 0.25)
        if c and p and c.iv and p.iv:
            return p.iv - c.iv
        return None

    def nearest_delta(self, cp: str, target_abs_delta: float, liquid_only: bool = True) -> Optional[OptionQuote]:
        pool = self.liquid() if liquid_only else self.options
        cands = [o for o in pool if o.cp == cp and o.greeks]
        if not cands:
            return None
        return min(cands, key=lambda o: abs(abs(o.greeks.delta) - target_abs_delta))

    def nearest_strike(self, cp: str, price: float, side: str = "nearest",
                       liquid_only: bool = True) -> Optional[OptionQuote]:
        """side: nearest / above / below —— 选择距离 price 最近 / 之上 / 之下的行权价。"""
        pool = self.liquid() if liquid_only else self.options
        cands = [o for o in pool if o.cp == cp]
        if side == "above":
            cands = [o for o in cands if o.strike >= price]
        elif side == "below":
            cands = [o for o in cands if o.strike <= price]
        if not cands:
            return None
        return min(cands, key=lambda o: abs(o.strike - price))

    def liquid(self, max_spread_pct: float = 0.15, min_oi: int = 50, max_abs_spread_ticks: int = 20) -> list[OptionQuote]:
        """流动性过滤：点差 / 持仓量门槛，避免推荐无法成交的合约。参数需按品种调。"""
        out = []
        for o in self.options:
            if o.mid <= 0:
                continue
            sp_ok = (o.spread_pct() <= max_spread_pct) or (o.spread <= max_abs_spread_ticks * self.product.option_tick)
            if sp_ok and o.open_interest >= min_oi:
                out.append(o)
        return out

    def sanity_issues(self) -> list[str]:
        """数据质量检查：行情陈旧、买卖价倒挂、Put-Call Parity 偏离过大。"""
        issues = []
        if self.futures.is_stale(self.asof):
            issues.append("期货行情陈旧")
        if self.futures.bid > self.futures.ask > 0:
            issues.append("期货买卖价倒挂")
        for o in self.options:
            if o.bid > o.ask > 0:
                issues.append(f"{o.symbol} 买卖价倒挂")
        # parity: C - P = F - K
        for k in self.strikes():
            c, p = self.get("C", k), self.get("P", k)
            if c and p and c.mid > 0 and p.mid > 0:
                dev = (c.mid - p.mid) - (self.F - k)
                if abs(dev) > max(3 * self.product.price_tick, 0.004 * self.F):
                    issues.append(f"K={k:g} 平价关系偏离 {dev:+.1f}")
        return issues
