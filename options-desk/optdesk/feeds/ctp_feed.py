"""CTP 行情源（期货公司生产环境的标准接口，最终上线建议以此为主源、天勤为备源）。

依赖：pip install openctp-ctp   （上期技术 CTP 的 Python 封装，支持 Windows / Linux / macOS）
测试：SimNow（http://www.simnow.com.cn）或 openctp 的 TTS 7x24 环境；生产行情前置地址向期货公司索取。

CTP 行情特点：
  - 推送为 500ms 快照（CThostFtdcDepthMarketDataField），包含期货与期权，字段含 BidPrice1/AskPrice1/Volume/OpenInterest；
  - 行情接口只需 BrokerID/UserID/Password（部分前置允许空账号），交易接口才需穿透式监管 AppID/AuthCode；
  - 没有"期权列表查询"：期权合约清单需通过交易接口 ReqQryInstrument 拉取，或用 AkShare/交易所网站每日导入，
    本模块通过 `instruments` 参数注入（symbol -> 元数据），由上层维护。

本模块把 SPI 回调转换为线程安全的快照缓存，并把每笔快照追加写入 parquet/CSV（可选）供回放。
"""
from __future__ import annotations

import datetime as dt
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from ..chain import OptionChain, OptionQuote, Quote
from ..contracts import get_product, parse_contract
from .base import MarketFeed


@dataclass
class OptionMeta:
    symbol: str
    underlying: str
    cp: str
    strike: float
    expiry: dt.date


class CtpFeed(MarketFeed):
    name = "ctp"

    def __init__(self, front: str, broker_id: str = "", user_id: str = "", password: str = "",
                 instruments: Optional[dict[str, OptionMeta]] = None,
                 bars_provider: Optional[Callable[[str, str, int], pd.DataFrame]] = None,
                 recorder: Optional[Callable[[dict], None]] = None):
        self.front, self.broker_id, self.user_id, self.password = front, broker_id, user_id, password
        self.instruments: dict[str, OptionMeta] = instruments or {}
        self.bars_provider = bars_provider      # CTP 不提供 K 线，交由 AkShare/天勤/自建 tick 聚合提供
        self.recorder = recorder
        self._lock = threading.Lock()
        self._snap: dict[str, Quote] = {}
        self._connected = threading.Event()
        self._logged_in = threading.Event()
        self._api = None
        self._spi = None
        self._subscribed: set[str] = set()
        self._req_id = 0

    # ------------------------------------------------------------------ 生命周期
    def start(self) -> None:
        try:
            from openctp_ctp import mdapi
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("未安装 openctp-ctp：pip install openctp-ctp") from e

        feed = self

        class Spi(mdapi.CThostFtdcMdSpi):
            def __init__(self, api):
                super().__init__()
                self.api = api

            def OnFrontConnected(self):
                feed._connected.set()
                req = mdapi.CThostFtdcReqUserLoginField()
                req.BrokerID, req.UserID, req.Password = feed.broker_id, feed.user_id, feed.password
                self.api.ReqUserLogin(req, feed._next_id())

            def OnFrontDisconnected(self, nReason):
                feed._connected.clear()
                feed._logged_in.clear()

            def OnRspUserLogin(self, pRspUserLogin, pRspInfo, nRequestID, bIsLast):
                if pRspInfo is None or pRspInfo.ErrorID == 0:
                    feed._logged_in.set()
                    if feed._subscribed:
                        self.api.SubscribeMarketData([s.encode() for s in feed._subscribed], len(feed._subscribed))

            def OnRtnDepthMarketData(self, d):
                feed._on_tick(d)

        self._api = mdapi.CThostFtdcMdApi.CreateFtdcMdApi("./ctp_md_flow/")
        self._spi = Spi(self._api)
        self._api.RegisterSpi(self._spi)
        self._api.RegisterFront(self.front)
        self._api.Init()
        if not self._logged_in.wait(30):
            raise TimeoutError("CTP 行情登录超时，检查前置地址 / 交易时段 / 账号")

    def stop(self) -> None:
        if self._api is not None:
            self._api.Release()

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def subscribe(self, symbols: list[str]) -> None:
        new = [s for s in symbols if s not in self._subscribed]
        self._subscribed.update(symbols)
        if new and self._logged_in.is_set():
            self._api.SubscribeMarketData([s.encode() for s in new], len(new))

    # ------------------------------------------------------------------ 回调
    def _on_tick(self, d) -> None:
        def f(x):
            return 0.0 if x is None or x >= 1e300 else float(x)   # CTP 空值为 DBL_MAX
        try:
            ts = dt.datetime.strptime(f"{d.ActionDay} {d.UpdateTime}", "%Y%m%d %H:%M:%S") + dt.timedelta(milliseconds=d.UpdateMillisec)
        except Exception:
            ts = dt.datetime.now()
        q = Quote(d.InstrumentID, f(d.LastPrice), f(d.BidPrice1), f(d.AskPrice1), int(d.BidVolume1), int(d.AskVolume1),
                  int(d.Volume), int(f(d.OpenInterest)), ts)
        with self._lock:
            self._snap[q.symbol] = q
        if self.recorder:
            self.recorder({"ts": ts, **{k: getattr(q, k) for k in ("symbol", "last", "bid", "ask", "volume", "open_interest")}})

    # ------------------------------------------------------------------ 接口
    def get_quote(self, symbol: str) -> Quote:
        if symbol not in self._subscribed:
            self.subscribe([symbol])
        with self._lock:
            q = self._snap.get(symbol)
        if q is None:
            raise LookupError(f"{symbol} 尚未收到行情")
        return q

    def get_bars(self, symbol: str, period: str = "1d", n: int = 300) -> pd.DataFrame:
        if self.bars_provider is None:
            raise NotImplementedError("CTP 不提供历史 K 线，请注入 bars_provider（如 AkShareHistory.get_bars）")
        return self.bars_provider(symbol, period, n)

    def get_chain(self, underlying: str, expiry: Optional[dt.date] = None) -> OptionChain:
        metas = [m for m in self.instruments.values() if m.underlying == underlying and (expiry is None or m.expiry == expiry)]
        if not metas:
            raise LookupError(f"{underlying} 没有注入期权合约清单（instruments）")
        exp = expiry or max({m.expiry for m in metas}, key=lambda e: sum(1 for m in metas if m.expiry == e))
        metas = [m for m in metas if m.expiry == exp]
        self.subscribe([underlying] + [m.symbol for m in metas])
        fut = self.get_quote(underlying)
        opts = []
        with self._lock:
            for m in metas:
                q = self._snap.get(m.symbol)
                if q is None:
                    continue
                opts.append(OptionQuote(**q.__dict__, cp=m.cp, strike=m.strike, expiry=m.expiry, underlying=underlying))
        spec = get_product(parse_contract(underlying).product)
        return OptionChain(underlying, spec, fut, exp, opts, dt.datetime.now())

    def health(self) -> dict:
        return {"feed": self.name, "ok": self._logged_in.is_set(), "connected": self._connected.is_set(),
                "symbols": len(self._snap)}
