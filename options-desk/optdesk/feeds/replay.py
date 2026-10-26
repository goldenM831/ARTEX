"""回放 / 合成行情源：无需任何账号即可跑通全流程（开发、测试、演示、策略回归）。

两种模式：
  1. SyntheticFeed：按 GBM 生成日线，按带偏斜的波动率曲面生成期权链报价（含买卖价差、持仓量）。
  2. CsvReplayFeed：读取本地 CSV（如 AkShare 下载的日线 + 期权日行情 / 自己录制的 tick 快照）逐日回放，
     用于离线回归"策略引擎在历史上每天会给什么建议"。
"""
from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..chain import OptionChain, OptionQuote, Quote
from ..contracts import estimate_option_expiry, format_option_code, get_product, parse_contract
from ..pricing import black76_price
from .base import MarketFeed


class SyntheticFeed(MarketFeed):
    name = "synthetic"

    def __init__(self, underlying: str = "TA601", start_price: float = 4800.0, vol: float = 0.22,
                 drift: float = 0.0, days: int = 300, seed: int = 7, skew: float = 0.08,
                 iv_premium: float = 0.03, asof: Optional[dt.datetime] = None):
        self.cid = parse_contract(underlying)
        self.spec = get_product(self.cid.product)
        self.underlying = underlying
        self.vol = vol
        self.skew = skew
        self.iv_premium = iv_premium
        self.asof = asof or dt.datetime.now()
        rng = np.random.default_rng(seed)
        n = days
        idx = pd.bdate_range(end=self.asof.date(), periods=n)
        rets = rng.normal((drift - 0.5 * vol * vol) / 244, vol / math.sqrt(244), n)
        path = np.exp(np.cumsum(rets))
        close = start_price * path / path[-1]          # 以 start_price 作为"当前价"锚定路径末端
        close = np.round(close / self.spec.price_tick) * self.spec.price_tick
        intraday = np.abs(rng.normal(0, vol / math.sqrt(244) * 0.6, n)) * close
        open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.002, n))
        high = np.maximum(open_, close) + intraday
        low = np.minimum(open_, close) - intraday
        volu = rng.integers(200_000, 900_000, n)
        self._bars = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volu}, index=idx)

    @property
    def last_price(self) -> float:
        return float(self._bars["close"].iloc[-1])

    def get_quote(self, symbol: str) -> Quote:
        p = self.last_price
        return Quote(symbol, p, p - self.spec.price_tick, p + self.spec.price_tick, 50, 50, 300_000, 1_200_000, self.asof)

    def get_bars(self, symbol: str, period: str = "1d", n: int = 300) -> pd.DataFrame:
        return self._bars.tail(n).copy()

    def main_contract(self, product: str) -> str:
        return self.underlying

    def smile_iv(self, K: float, F: float) -> float:
        """简单偏斜曲面：OTM put 更贵（商品期权常见为 call 偏贵，可用负 skew 模拟）。"""
        m = math.log(K / F)
        return max(self.vol + self.iv_premium - self.skew * m + 0.6 * m * m, 0.05)

    def get_chain(self, underlying: str, expiry: Optional[dt.date] = None) -> OptionChain:
        F = self.last_price
        expiry = expiry or estimate_option_expiry(self.cid.product, self.cid.year, self.cid.month)
        T = max((expiry - self.asof.date()).days, 1) / 365
        fut = self.get_quote(underlying)
        opts = []
        tick = self.spec.option_tick
        for K in self.spec.strikes_around(F, 10):
            iv = self.smile_iv(K, F)
            dist = abs(K - F) / F
            oi = int(max(30_000 * math.exp(-dist * 25), 10))
            for cp in ("C", "P"):
                px = black76_price(F, K, T, iv, cp)
                half = max(tick, px * (0.01 + dist * 0.2)) / 2
                bid = max(math.floor((px - half) / tick) * tick, 0.0)
                ask = math.ceil((px + half) / tick) * tick
                code = format_option_code(self.cid.product, self.cid.year, self.cid.month, cp, K)
                opts.append(OptionQuote(code, round(px, 2), bid, ask, 20, 20, oi // 10, oi, self.asof,
                                        cp=cp, strike=K, expiry=expiry, underlying=underlying))
        return OptionChain(underlying, self.spec, fut, expiry, opts, self.asof)


class CsvReplayFeed(MarketFeed):
    """按日回放本地 CSV。

    futures_csv : date,open,high,low,close,volume
    options_csv : date,symbol,cp,strike,expiry,bid,ask,last,volume,open_interest   （可选；无则按 HV 合成）
    """
    name = "csv-replay"

    def __init__(self, underlying: str, futures_csv: str | Path, options_csv: Optional[str | Path] = None):
        self.underlying = underlying
        self.cid = parse_contract(underlying)
        self.spec = get_product(self.cid.product)
        self.bars = pd.read_csv(futures_csv, parse_dates=["date"]).set_index("date").sort_index()
        self.opts = None
        if options_csv:
            self.opts = pd.read_csv(options_csv, parse_dates=["date", "expiry"])
        self.cursor = len(self.bars) - 1

    def seek(self, date: dt.date) -> None:
        self.cursor = int(self.bars.index.searchsorted(pd.Timestamp(date), side="right") - 1)

    def step(self) -> bool:
        if self.cursor + 1 >= len(self.bars):
            return False
        self.cursor += 1
        return True

    @property
    def asof(self) -> dt.datetime:
        return self.bars.index[self.cursor].to_pydatetime()

    def get_quote(self, symbol: str) -> Quote:
        p = float(self.bars["close"].iloc[self.cursor])
        return Quote(symbol, p, p, p, ts=self.asof)

    def get_bars(self, symbol: str, period: str = "1d", n: int = 300) -> pd.DataFrame:
        return self.bars.iloc[max(0, self.cursor + 1 - n): self.cursor + 1].copy()

    def main_contract(self, product: str) -> str:
        return self.underlying

    def get_chain(self, underlying: str, expiry: Optional[dt.date] = None) -> OptionChain:
        fut = self.get_quote(underlying)
        if self.opts is None:
            synth = SyntheticFeed(underlying, asof=self.asof)
            synth._bars = self.get_bars(underlying)
            return synth.get_chain(underlying, expiry)
        day = self.opts[self.opts["date"] == self.bars.index[self.cursor]]
        if expiry is not None:
            day = day[day["expiry"].dt.date == expiry]
        if day.empty:
            raise LookupError(f"{self.asof.date()} 无期权数据")
        exp = day["expiry"].iloc[0].date()
        opts = [OptionQuote(r.symbol, r.last, r.bid, r.ask, 0, 0, int(r.volume), int(r.open_interest), self.asof,
                            cp=r.cp, strike=float(r.strike), expiry=r.expiry.date(), underlying=underlying)
                for r in day.itertuples()]
        return OptionChain(underlying, self.spec, fut, exp, opts, self.asof)
