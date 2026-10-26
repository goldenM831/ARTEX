"""行情技术指标与方向打分。

目标不是"预测"，而是把常用指标压成一个可解释的方向分数 [-1, 1]，
供策略引擎决定：牛市价差 / 熊市价差 / 中性收租（备兑、领口）以及行权价选择（支撑 / 阻力位）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def macd(close: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    dif = ema(close, fast) - ema(close, slow)
    dea = ema(dif, signal)
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": 2 * (dif - dea)})


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n).std(ddof=0)
    return pd.DataFrame({"mid": mid, "upper": mid + k * sd, "lower": mid - k * sd,
                         "pct_b": (close - (mid - k * sd)) / (2 * k * sd)})


def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    dn = -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return pd.DataFrame({"pdi": pdi, "mdi": mdi, "adx": dx.ewm(alpha=1 / n, adjust=False).mean()})


def swing_levels(df: pd.DataFrame, lookback: int = 60, order: int = 3) -> tuple[list[float], list[float]]:
    """近 lookback 根 K 线内的摆动高 / 低点，作为阻力 / 支撑候选（用于备兑卖 call、领口买 put 的行权价选择）。"""
    d = df.tail(lookback)
    highs, lows = d["high"].values, d["low"].values
    res, sup = [], []
    for i in range(order, len(d) - order):
        if highs[i] == max(highs[i - order:i + order + 1]):
            res.append(float(highs[i]))
        if lows[i] == min(lows[i - order:i + order + 1]):
            sup.append(float(lows[i]))
    return sorted(set(res)), sorted(set(sup))


@dataclass
class TASnapshot:
    close: float
    ma5: float
    ma20: float
    ma60: float
    rsi14: float
    macd_hist: float
    adx14: float
    pct_b: float
    resistance: float | None
    support: float | None
    score: float                     # [-1, 1]
    label: str                       # 看涨 / 偏多 / 中性 / 偏空 / 看跌
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        r = f"{self.resistance:.0f}" if self.resistance else "-"
        s = f"{self.support:.0f}" if self.support else "-"
        return (f"[{self.label} score={self.score:+.2f}] close={self.close:.0f} "
                f"MA5/20/60={self.ma5:.0f}/{self.ma20:.0f}/{self.ma60:.0f} RSI={self.rsi14:.0f} "
                f"MACD柱={self.macd_hist:+.1f} ADX={self.adx14:.0f} %B={self.pct_b:.2f} 阻力={r} 支撑={s}")


def analyze(df: pd.DataFrame) -> TASnapshot:
    """多指标投票 → 方向分数。权重可按品种回测后调整。"""
    close = df["close"]
    c = float(close.iloc[-1])
    ma5, ma20, ma60 = (float(sma(close, n).iloc[-1]) for n in (5, 20, 60))
    r = float(rsi(close).iloc[-1])
    m = macd(close)
    mh = float(m["hist"].iloc[-1])
    mh_prev = float(m["hist"].iloc[-2]) if len(m) > 1 else 0.0
    a = adx(df)
    adx14, pdi, mdi = (float(a[k].iloc[-1]) for k in ("adx", "pdi", "mdi"))
    bb = bollinger(close)
    pct_b = float(bb["pct_b"].iloc[-1])
    res_lv, sup_lv = swing_levels(df)
    resistance = min((x for x in res_lv if x > c), default=None)
    support = max((x for x in sup_lv if x < c), default=None)

    score, reasons = 0.0, []
    # 1) 均线排列（权重 0.35）
    if c > ma5 > ma20 > ma60:
        score += 0.35; reasons.append("均线多头排列")
    elif c < ma5 < ma20 < ma60:
        score -= 0.35; reasons.append("均线空头排列")
    else:
        t = 0.0
        t += 0.12 if c > ma20 else -0.12
        t += 0.12 if ma20 > ma60 else -0.12
        score += t
        reasons.append("均线交织" if abs(t) < 0.2 else ("价格在 MA20 上方" if t > 0 else "价格在 MA20 下方"))
    # 2) MACD（权重 0.25）
    if mh > 0 and mh > mh_prev:
        score += 0.25; reasons.append("MACD 红柱放大")
    elif mh > 0:
        score += 0.10; reasons.append("MACD 红柱收敛")
    elif mh < 0 and mh < mh_prev:
        score -= 0.25; reasons.append("MACD 绿柱放大")
    else:
        score -= 0.10; reasons.append("MACD 绿柱收敛")
    # 3) RSI（权重 0.2，带超买超卖反向修正）
    if r > 70:
        score -= 0.10; reasons.append(f"RSI {r:.0f} 超买，追涨需谨慎")
    elif r < 30:
        score += 0.10; reasons.append(f"RSI {r:.0f} 超卖，杀跌需谨慎")
    elif r > 55:
        score += 0.15; reasons.append("RSI 强势区")
    elif r < 45:
        score -= 0.15; reasons.append("RSI 弱势区")
    # 4) ADX 趋势强度：强趋势放大分数，弱趋势压缩分数（利于中性收租）
    if adx14 > 25:
        score *= 1.2; reasons.append(f"ADX {adx14:.0f} 趋势明确")
    elif adx14 < 18:
        score *= 0.6; reasons.append(f"ADX {adx14:.0f} 震荡市，利于备兑 / 领口收租")
    score = float(max(-1.0, min(1.0, score)))
    label = ("看涨" if score > 0.5 else "偏多" if score > 0.15 else
             "看跌" if score < -0.5 else "偏空" if score < -0.15 else "中性")
    return TASnapshot(c, ma5, ma20, ma60, r, mh, adx14, pct_b, resistance, support, score, label, reasons)
