"""天勤 TqSdk 实时行情源（推荐作为第一阶段实时接入，最快跑通）。

安装：pip install tqsdk        账号：https://account.shinnytech.com 免费注册（实时行情免费，历史数据/专业功能需专业版）
合约代码：天勤风格 "CZCE.TA601" / "DCE.pp2601" / "SHFE.ru2601" / "INE.sc2601"；主连 "KQ.m@CZCE.TA"。

用法：
    feed = TqFeed(user, password)
    feed.start()
    chain = feed.get_chain("CZCE.TA601")
    ...
    feed.stop()

注意：
  - TqApi 不是线程安全的；本实现把 api 放在后台线程里循环 wait_update()，对外提供最近一次快照（线程锁保护）。
  - 期权列表通过 api.query_options(underlying) 获取，到期日用 quote.expire_datetime（真实值，优先于规则估算）。
"""
from __future__ import annotations

import datetime as dt
import threading
import time
from typing import Optional

import pandas as pd

from ..chain import OptionChain, OptionQuote, Quote
from ..contracts import get_product, parse_contract
from .base import MarketFeed


def _product_from_tq(symbol: str) -> str:
    return parse_contract(symbol).product


class TqFeed(MarketFeed):
    name = "tqsdk"

    def __init__(self, user: str, password: str, underlyings: Optional[list[str]] = None, kline_days: int = 300):
        self.user, self.password = user, password
        self.underlyings = underlyings or []
        self.kline_days = kline_days
        self._api = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._quotes: dict = {}          # symbol -> tq quote 对象（引用，天勤会原地更新）
        self._klines: dict = {}          # symbol -> DataFrame 引用
        self._options: dict = {}         # underlying -> list[symbol]
        self._ready = threading.Event()
        self._last_err: Optional[str] = None

    # ------------------------------------------------------------------ 生命周期
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="tqsdk-feed", daemon=True)
        self._thread.start()
        if not self._ready.wait(60):
            raise TimeoutError(f"TqSdk 连接超时: {self._last_err}")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _run(self) -> None:
        try:
            from tqsdk import TqApi, TqAuth
        except ImportError as e:  # pragma: no cover
            self._last_err = f"未安装 tqsdk: {e}"
            return
        try:
            api = TqApi(auth=TqAuth(self.user, self.password))
            self._api = api
            with self._lock:
                for u in self.underlyings:
                    self._subscribe_underlying(u)
            self._ready.set()
            while not self._stop.is_set():
                api.wait_update(deadline=time.time() + 1)
        except Exception as e:  # pragma: no cover
            self._last_err = repr(e)
        finally:
            if self._api is not None:
                try:
                    self._api.close()
                except Exception:
                    pass

    def _subscribe_underlying(self, u: str) -> None:
        api = self._api
        self._quotes[u] = api.get_quote(u)
        self._klines[u] = api.get_kline_serial(u, 86400, data_length=self.kline_days)
        syms = api.query_options(u)
        self._options[u] = syms
        for s in syms:
            self._quotes[s] = api.get_quote(s)

    def _ensure(self, u: str) -> None:
        if u not in self._options:
            with self._lock:
                self._subscribe_underlying(u)
            # 等待首批数据到齐
            for _ in range(50):
                q = self._quotes[u]
                if q.last_price == q.last_price and q.last_price > 0:  # not NaN
                    break
                time.sleep(0.1)

    # ------------------------------------------------------------------ 接口
    @staticmethod
    def _to_quote(q, symbol: str) -> Quote:
        ts = dt.datetime.fromisoformat(q.datetime[:26]) if q.datetime else None
        f = lambda x: float(x) if x == x else 0.0  # NaN -> 0
        return Quote(symbol, f(q.last_price), f(q.bid_price1), f(q.ask_price1), int(q.bid_volume1 or 0),
                     int(q.ask_volume1 or 0), int(q.volume or 0), int(q.open_interest or 0), ts)

    def get_quote(self, symbol: str) -> Quote:
        if symbol not in self._quotes:
            with self._lock:
                self._quotes[symbol] = self._api.get_quote(symbol)
        return self._to_quote(self._quotes[symbol], symbol)

    def get_bars(self, symbol: str, period: str = "1d", n: int = 300) -> pd.DataFrame:
        self._ensure(symbol)
        k = self._klines[symbol]
        df = pd.DataFrame({
            "open": k["open"].values, "high": k["high"].values, "low": k["low"].values,
            "close": k["close"].values, "volume": k["volume"].values,
        }, index=pd.to_datetime(k["datetime"].values, unit="ns"))
        return df.dropna().tail(n)

    def main_contract(self, product: str) -> str:
        spec = get_product(product)
        q = self._api.get_quote(f"KQ.m@{spec.exchange}.{spec.code if spec.exchange == 'CZCE' else spec.code.lower()}")
        return q.underlying_symbol

    def get_chain(self, underlying: str, expiry: Optional[dt.date] = None) -> OptionChain:
        self._ensure(underlying)
        spec = get_product(_product_from_tq(underlying))
        fut = self.get_quote(underlying)
        opts: list[OptionQuote] = []
        exp_counter: dict[dt.date, int] = {}
        for s in self._options[underlying]:
            q = self._quotes[s]
            exp = dt.datetime.fromtimestamp(q.expire_datetime).date() if q.expire_datetime else None
            if exp is None or (expiry and exp != expiry):
                continue
            exp_counter[exp] = exp_counter.get(exp, 0) + 1
            base = self._to_quote(q, s)
            opts.append(OptionQuote(**base.__dict__, cp="C" if q.option_class == "CALL" else "P",
                                    strike=float(q.strike_price), expiry=exp, underlying=underlying))
        if not opts:
            raise LookupError(f"{underlying} 无期权行情（检查是否已挂牌 / 账号是否有权限）")
        exp = expiry or max(exp_counter, key=exp_counter.get)
        opts = [o for o in opts if o.expiry == exp]
        return OptionChain(underlying, spec, fut, exp, opts, dt.datetime.now())

    def health(self) -> dict:
        return {"feed": self.name, "ok": self._ready.is_set() and not self._stop.is_set(), "err": self._last_err}
