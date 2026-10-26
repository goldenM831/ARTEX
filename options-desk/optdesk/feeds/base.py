"""行情源抽象接口。所有数据源（CTP / 天勤 / AkShare / 回放）实现同一接口，策略层不感知来源。

设计要点：
  - get_bars 返回 pandas.DataFrame(columns=open/high/low/close/volume, index=时间)
  - get_chain 返回统一的 OptionChain（已填好 futures 快照与期权报价，未计算 IV）
  - 实时源用 start()/stop() 维护后台连接；离线源 start() 可为空
"""
from __future__ import annotations

import abc
import datetime as dt
from typing import Optional

import pandas as pd

from ..chain import OptionChain, Quote


class MarketFeed(abc.ABC):
    name: str = "base"

    def start(self) -> None:  # noqa: B027
        pass

    def stop(self) -> None:  # noqa: B027
        pass

    @abc.abstractmethod
    def get_quote(self, symbol: str) -> Quote: ...

    @abc.abstractmethod
    def get_bars(self, symbol: str, period: str = "1d", n: int = 300) -> pd.DataFrame: ...

    @abc.abstractmethod
    def get_chain(self, underlying: str, expiry: Optional[dt.date] = None) -> OptionChain: ...

    def main_contract(self, product: str) -> str:
        """主力合约代码。离线源可用持仓量最大者；天勤可用 KQ.m@CZCE.TA。"""
        raise NotImplementedError

    def health(self) -> dict:
        return {"feed": self.name, "ok": True}
