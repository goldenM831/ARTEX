"""期货期权定价、希腊字母、隐含波动率。

- Black-76：国内商品期权的行业基准（交易所结算价 / 多数行情软件的 IV 口径均基于 Black 模型）。
- 国内商品期权为美式，但期货期权的提前行权价值很小；需要精确时可用 binomial_american_future。
- 不依赖 scipy，正态分布用 math.erf 实现，便于在任何环境部署。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

CP = Literal["C", "P"]

SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


@dataclass
class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float      # 每 1 个波动率点（1%）的价格变化
    theta: float     # 每自然日的时间价值衰减（负数）
    rho: float

    def as_dict(self) -> dict:
        return {k: round(v, 6) for k, v in self.__dict__.items()}


def _d1_d2(F: float, K: float, T: float, sigma: float) -> tuple[float, float]:
    vs = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / vs
    return d1, d1 - vs


def intrinsic(F: float, K: float, cp: CP) -> float:
    return max(F - K, 0.0) if cp == "C" else max(K - F, 0.0)


def black76_price(F: float, K: float, T: float, sigma: float, cp: CP, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        return intrinsic(F, K, cp) * math.exp(-r * T) if T > 0 else intrinsic(F, K, cp)
    d1, d2 = _d1_d2(F, K, T, sigma)
    df = math.exp(-r * T)
    if cp == "C":
        return df * (F * norm_cdf(d1) - K * norm_cdf(d2))
    return df * (K * norm_cdf(-d2) - F * norm_cdf(-d1))


def black76_greeks(F: float, K: float, T: float, sigma: float, cp: CP, r: float = 0.0) -> Greeks:
    if T <= 0 or sigma <= 0:
        p = intrinsic(F, K, cp)
        d = (1.0 if F > K else 0.0) if cp == "C" else (-1.0 if F < K else 0.0)
        return Greeks(p, d, 0.0, 0.0, 0.0, 0.0)
    d1, d2 = _d1_d2(F, K, T, sigma)
    df = math.exp(-r * T)
    sq = math.sqrt(T)
    pdf = norm_pdf(d1)
    price = black76_price(F, K, T, sigma, cp, r)
    if cp == "C":
        delta = df * norm_cdf(d1)
        theta_y = (-F * df * pdf * sigma / (2 * sq)
                   + r * F * df * norm_cdf(d1) - r * K * df * norm_cdf(d2))
    else:
        delta = -df * norm_cdf(-d1)
        theta_y = (-F * df * pdf * sigma / (2 * sq)
                   - r * F * df * norm_cdf(-d1) + r * K * df * norm_cdf(-d2))
    gamma = df * pdf / (F * sigma * sq)
    vega = F * df * pdf * sq / 100.0
    rho = -T * price
    return Greeks(price, delta, gamma, vega, theta_y / 365.0, rho)


def implied_vol(price: float, F: float, K: float, T: float, cp: CP, r: float = 0.0,
                lo: float = 1e-4, hi: float = 5.0, tol: float = 1e-8, max_iter: int = 100) -> float | None:
    """Black-76 隐含波动率：Newton 迭代 + 二分法兜底。价格低于内在价值或 T<=0 时返回 None。"""
    if T <= 0 or price <= 0:
        return None
    df = math.exp(-r * T)
    if price < intrinsic(F, K, cp) * df - 1e-12:
        return None
    sigma = max(min(math.sqrt(2 * math.pi / T) * price / F, hi), lo)
    for _ in range(20):
        g = black76_greeks(F, K, T, sigma, cp, r)
        diff = g.price - price
        if abs(diff) < tol:
            return sigma
        v = g.vega * 100.0
        if v < 1e-12:
            break
        step = diff / v
        new = sigma - step
        if not (lo < new < hi):
            break
        sigma = new
    f_lo = black76_price(F, K, T, lo, cp, r) - price
    f_hi = black76_price(F, K, T, hi, cp, r) - price
    if f_lo * f_hi > 0:
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = black76_price(F, K, T, mid, cp, r) - price
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def binomial_american_future(F: float, K: float, T: float, sigma: float, cp: CP,
                             r: float = 0.0, steps: int = 200) -> float:
    """期货标的的美式期权（CRR 二叉树，期货漂移为 0）。用于校验美式提前行权溢价。"""
    if T <= 0:
        return intrinsic(F, K, cp)
    dt_ = T / steps
    u = math.exp(sigma * math.sqrt(dt_))
    d = 1.0 / u
    p = (1.0 - d) / (u - d)
    disc = math.exp(-r * dt_)
    values = [intrinsic(F * (u ** j) * (d ** (steps - j)), K, cp) for j in range(steps + 1)]
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            cont = disc * (p * values[j + 1] + (1 - p) * values[j])
            ex = intrinsic(F * (u ** j) * (d ** (i - j)), K, cp)
            values[j] = max(cont, ex)
    return values[0]


def prob_itm(F: float, K: float, T: float, sigma: float, cp: CP) -> float:
    """到期落在实值的风险中性概率 N(±d2)。"""
    if T <= 0 or sigma <= 0:
        return 1.0 if intrinsic(F, K, cp) > 0 else 0.0
    _, d2 = _d1_d2(F, K, T, sigma)
    return norm_cdf(d2) if cp == "C" else norm_cdf(-d2)


def prob_above(F: float, level: float, T: float, sigma: float) -> float:
    """对数正态下到期价格高于 level 的概率。"""
    if level <= 0:
        return 1.0
    if T <= 0 or sigma <= 0:
        return 1.0 if F > level else 0.0
    _, d2 = _d1_d2(F, level, T, sigma)
    return norm_cdf(d2)


def strike_for_delta(F: float, T: float, sigma: float, cp: CP, target_delta: float) -> float:
    """给定目标 delta（绝对值）反推行权价。"""
    target = abs(target_delta)
    target = min(max(target, 0.01), 0.99)
    # 反解 N(d1)=target（看跌为 N(-d1)=target），Acklam 逆正态近似精度足够用于选档
    d1 = _inv_norm(target if cp == "C" else 1 - target)
    return F * math.exp(0.5 * sigma * sigma * T - d1 * sigma * math.sqrt(T))


def _inv_norm(p: float) -> float:
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r_ = q * q
    return (((((a[0]*r_+a[1])*r_+a[2])*r_+a[3])*r_+a[4])*r_+a[5])*q / (((((b[0]*r_+b[1])*r_+b[2])*r_+b[3])*r_+b[4])*r_+1)
