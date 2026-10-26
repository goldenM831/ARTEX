"""波动率监测：历史波动率、波动率锥、IV 分位、波动状态识别。

输入统一为 pandas.DataFrame，列：open/high/low/close（日线或分钟线均可，年化因子自行传入）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 244  # 国内商品期货年均交易日


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1)).dropna()


def hv_close(close: pd.Series, window: int = 20, ann: int = TRADING_DAYS) -> pd.Series:
    return log_returns(close).rolling(window).std(ddof=1) * np.sqrt(ann)


def hv_parkinson(df: pd.DataFrame, window: int = 20, ann: int = TRADING_DAYS) -> pd.Series:
    x = np.log(df["high"] / df["low"]) ** 2
    return np.sqrt(x.rolling(window).mean() / (4 * np.log(2))) * np.sqrt(ann)


def hv_garman_klass(df: pd.DataFrame, window: int = 20, ann: int = TRADING_DAYS) -> pd.Series:
    hl = 0.5 * np.log(df["high"] / df["low"]) ** 2
    co = (2 * np.log(2) - 1) * np.log(df["close"] / df["open"]) ** 2
    return np.sqrt((hl - co).rolling(window).mean()) * np.sqrt(ann)


def hv_yang_zhang(df: pd.DataFrame, window: int = 20, ann: int = TRADING_DAYS) -> pd.Series:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    on = np.log(o / c.shift(1))
    oc = np.log(c / o)
    rs = np.log(h / o) * np.log(h / c) + np.log(l / o) * np.log(l / c)
    v_on = on.rolling(window).var(ddof=1)
    v_oc = oc.rolling(window).var(ddof=1)
    v_rs = rs.rolling(window).mean()
    return np.sqrt(v_on + k * v_oc + (1 - k) * v_rs) * np.sqrt(ann)


def vol_cone(close: pd.Series, windows=(5, 10, 20, 40, 60, 120),
             pcts=(0.05, 0.25, 0.5, 0.75, 0.95)) -> pd.DataFrame:
    """波动率锥：各窗口下 HV 的分位数，用于判断当前 HV / IV 在历史中的位置。"""
    rows = {}
    for w in windows:
        s = hv_close(close, w).dropna()
        if len(s) < 5:
            continue
        rows[w] = {f"p{int(p*100)}": float(s.quantile(p)) for p in pcts}
        rows[w]["current"] = float(s.iloc[-1])
    return pd.DataFrame(rows).T


def iv_rank(iv_series: pd.Series, current: Optional[float] = None) -> float:
    s = iv_series.dropna()
    if s.empty:
        return float("nan")
    cur = s.iloc[-1] if current is None else current
    lo, hi = s.min(), s.max()
    return float((cur - lo) / (hi - lo)) if hi > lo else 0.5


def iv_percentile(iv_series: pd.Series, current: Optional[float] = None) -> float:
    s = iv_series.dropna()
    if s.empty:
        return float("nan")
    cur = s.iloc[-1] if current is None else current
    return float((s < cur).mean())


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


@dataclass
class VolSignal:
    hv20: float
    hv60: float
    hv_percentile_1y: float          # 当前 HV20 在近一年 HV20 中的分位
    atr_ratio: float                 # 今日 ATR(14) / ATR(14) 60 日均值
    bb_width_pct: float              # 当前布林带宽在近 120 日的分位（低=压缩）
    iv_atm: Optional[float] = None
    iv_minus_hv: Optional[float] = None
    iv_rank_1y: Optional[float] = None
    regime: str = "normal"           # squeeze / normal / expansion / spike
    bias: str = "neutral"            # seller / buyer / neutral  —— 期权买卖方倾向
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"HV20={self.hv20:.1%}", f"HV60={self.hv60:.1%}",
                 f"HV分位={self.hv_percentile_1y:.0%}", f"ATR比={self.atr_ratio:.2f}",
                 f"带宽分位={self.bb_width_pct:.0%}"]
        if self.iv_atm is not None:
            parts.append(f"IV(ATM)={self.iv_atm:.1%}")
        if self.iv_minus_hv is not None:
            parts.append(f"IV-HV={self.iv_minus_hv:+.1%}")
        if self.iv_rank_1y is not None:
            parts.append(f"IV Rank={self.iv_rank_1y:.0%}")
        return f"[{self.regime} / 倾向:{self.bias}] " + " ".join(parts)


def detect_vol_regime(df: pd.DataFrame, iv_atm: Optional[float] = None,
                      iv_history: Optional[pd.Series] = None) -> VolSignal:
    """综合 HV / ATR / 布林带宽 / IV 给出波动状态与期权买卖方倾向。

    规则（可迭代调参）：
      - spike     : ATR 比 > 1.8 或 HV20 分位 > 0.9 ——> 波动已爆发，慎做买方追高波动率
      - expansion : ATR 比 > 1.25 或 HV20 > HV60 * 1.2
      - squeeze   : 布林带宽分位 < 0.15 且 ATR 比 < 0.8 ——> 蓄势，利于买方 / 价差买方
      - 倾向      : IV-HV > +4 个点 且 IV Rank > 0.6 → seller；IV-HV < -3 个点 或 IV Rank < 0.25 → buyer
    """
    close = df["close"]
    notes: list[str] = []
    hv20_s = hv_close(close, 20)
    hv60_s = hv_close(close, 60)
    hv20 = float(hv20_s.iloc[-1]) if hv20_s.notna().any() else float("nan")
    hv60 = float(hv60_s.iloc[-1]) if hv60_s.notna().any() else hv20
    hv_pct = float((hv20_s.dropna().tail(TRADING_DAYS) < hv20).mean()) if hv20_s.notna().sum() > 20 else 0.5

    atr_s = atr(df, 14)
    atr_ratio = float(atr_s.iloc[-1] / atr_s.tail(60).mean()) if atr_s.notna().sum() > 14 else 1.0

    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std(ddof=0)
    bbw = (4 * sd20 / ma20).dropna()
    bb_width_pct = float((bbw.tail(120) < bbw.iloc[-1]).mean()) if len(bbw) > 20 else 0.5

    regime = "normal"
    if atr_ratio > 1.8 or hv_pct > 0.9:
        regime = "spike"
        notes.append("波动率已处于高位 / 急剧放大，卖方权利金丰厚但需防尾部，买方不宜追高")
    elif atr_ratio > 1.25 or (not np.isnan(hv60) and hv20 > hv60 * 1.2):
        regime = "expansion"
        notes.append("波动率扩张中，顺势策略优先，价差结构控制 vega 风险")
    elif bb_width_pct < 0.15 and atr_ratio < 0.8:
        regime = "squeeze"
        notes.append("布林带收口 + ATR 低位：波动压缩，关注突破；适合低成本买方或价差买方布局")

    bias = "neutral"
    iv_minus_hv = None
    ivr = None
    if iv_atm is not None and not np.isnan(hv20):
        iv_minus_hv = iv_atm - hv20
        if iv_history is not None and len(iv_history.dropna()) > 20:
            ivr = iv_rank(iv_history, iv_atm)
        if iv_minus_hv > 0.04 and (ivr is None or ivr > 0.6):
            bias = "seller"
            notes.append(f"IV 比 HV 高 {iv_minus_hv*100:.1f} 个点，期权偏贵 → 备兑 / 卖方价差 / 领口中的卖腿更划算")
        elif iv_minus_hv < -0.03 or (ivr is not None and ivr < 0.25):
            bias = "buyer"
            notes.append(f"IV 低于 HV {abs(iv_minus_hv)*100:.1f} 个点或处于低分位，期权偏便宜 → 保护性买权 / 买方价差成本低")
        else:
            notes.append("IV 与 HV 基本匹配，策略选择以方向观点和现货需求为主")
    return VolSignal(hv20=hv20, hv60=hv60, hv_percentile_1y=hv_pct, atr_ratio=atr_ratio,
                     bb_width_pct=bb_width_pct, iv_atm=iv_atm, iv_minus_hv=iv_minus_hv,
                     iv_rank_1y=ivr, regime=regime, bias=bias, notes=notes)
