"""AkShare 历史数据适配（免费，用于 HV / 波动率锥 / IV 历史 / 回测，不用于实时）。

安装：pip install akshare
注意：AkShare 接口签名随上游网站改版而变化较频繁，本模块把调用集中封装，并建议把结果落盘（parquet/CSV）后再用。
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from ..contracts import parse_contract

_EXCHANGE_OPTION_FUNC = {
    "CZCE": ("option_czce_daily", "symbol"),
    "DCE": ("option_dce_daily", "symbol"),
    "SHFE": ("option_shfe_daily", "symbol"),
    "INE": ("option_shfe_daily", "symbol"),
}


class AkShareHistory:
    def __init__(self, cache_dir: str | Path = ".cache/akshare"):
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)

    def _ak(self):
        try:
            import akshare as ak
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("未安装 akshare：pip install akshare") from e
        return ak

    def get_bars(self, symbol: str, period: str = "1d", n: int = 300) -> pd.DataFrame:
        """期货日线（新浪源）。symbol 如 'TA2601'（新浪对郑商所也用 4 位月份）。"""
        cid = parse_contract(symbol)
        sina = f"{cid.product}{cid.year % 100:02d}{cid.month:02d}"
        f = self.cache / f"bars_{sina}.parquet"
        if f.exists() and (dt.datetime.now() - dt.datetime.fromtimestamp(f.stat().st_mtime)).seconds < 6 * 3600:
            return pd.read_parquet(f).tail(n)
        ak = self._ak()
        df = ak.futures_zh_daily_sina(symbol=sina)
        df = df.rename(columns={"date": "date", "open": "open", "high": "high", "low": "low", "close": "close",
                                "volume": "volume"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")[["open", "high", "low", "close", "volume"]].astype(float)
        try:
            df.to_parquet(f)
        except Exception:
            df.to_csv(f.with_suffix(".csv"))
        return df.tail(n)

    def get_main_bars(self, product: str, n: int = 600) -> pd.DataFrame:
        """主力连续日线（新浪 'TA0' 风格），用于长周期 HV / 波动率锥。"""
        ak = self._ak()
        df = ak.futures_zh_daily_sina(symbol=f"{product.upper()}0")
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")[["open", "high", "low", "close", "volume"]].astype(float).tail(n)

    def option_daily(self, product: str, trade_date: dt.date) -> pd.DataFrame:
        """交易所期权日行情（结算价 / 持仓 / 成交 / 隐波（部分交易所提供））。"""
        ak = self._ak()
        cid_ex = parse_contract(f"{product}2601").exchange
        func, _ = _EXCHANGE_OPTION_FUNC[cid_ex]
        return getattr(ak, func)(symbol=_PRODUCT_CN.get(product.upper(), product), trade_date=trade_date.strftime("%Y%m%d"))

    def trade_calendar(self) -> set[dt.date]:
        ak = self._ak()
        df = ak.tool_trade_date_hist_sina()
        return set(pd.to_datetime(df["trade_date"]).dt.date)

    def holidays(self, start: dt.date, end: dt.date) -> set[dt.date]:
        cal = self.trade_calendar()
        d, out = start, set()
        while d <= end:
            if d.weekday() < 5 and d not in cal:
                out.add(d)
            d += dt.timedelta(days=1)
        return out


# AkShare 期权日行情接口按中文品种名取数
_PRODUCT_CN = {
    "TA": "PTA期权", "MA": "甲醇期权", "PF": "短纤期权",
    "PP": "聚丙烯期权", "L": "聚乙烯期权", "V": "聚氯乙烯期权", "EG": "乙二醇期权", "EB": "苯乙烯期权", "PG": "液化石油气期权",
    "SC": "原油期权", "BU": "石油沥青期权", "RU": "天然橡胶期权",
}
