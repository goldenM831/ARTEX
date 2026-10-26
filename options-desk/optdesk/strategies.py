"""策略结构：腿、组合、到期损益、风险指标、保证金估算、委托单生成。

约定：
  - 价格单位为 元/吨（原油为 元/桶），qty 单位为 手（期权 / 期货）或 吨（现货）。
  - qty > 0 为多头（买入），qty < 0 为空头（卖出）。
  - 组合的损益均为"整单"口径（元），同时给出 元/吨 口径方便业务沟通。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .chain import OptionChain, OptionQuote
from .contracts import ProductSpec
from .pricing import Greeks, black76_greeks


@dataclass
class Leg:
    kind: str                     # spot / future / call / put
    qty: float                    # 手（现货为吨）
    price: float                  # 成交 / 参考价（期权为权利金，现货为成本）
    symbol: str = ""
    strike: float = 0.0
    iv: Optional[float] = None
    multiplier: float = 1.0       # 现货为 1（qty 已是吨）

    @property
    def tons(self) -> float:
        return self.qty * self.multiplier

    @property
    def side(self) -> str:
        return "买" if self.qty > 0 else "卖"

    def payoff(self, prices: np.ndarray) -> np.ndarray:
        if self.kind == "spot" or self.kind == "future":
            return (prices - self.price) * self.tons
        if self.kind == "call":
            return (np.maximum(prices - self.strike, 0.0) - self.price) * self.tons
        if self.kind == "put":
            return (np.maximum(self.strike - prices, 0.0) - self.price) * self.tons
        raise ValueError(self.kind)

    def greeks(self, F: float, T: float, r: float = 0.0) -> Greeks:
        if self.kind in ("spot", "future"):
            return Greeks(F, 1.0, 0.0, 0.0, 0.0, 0.0)
        cp = "C" if self.kind == "call" else "P"
        return black76_greeks(F, self.strike, T, self.iv or 1e-6, cp, r)


@dataclass
class StrategyMetrics:
    net_premium_per_ton: float      # >0 收取权利金，<0 支付权利金
    net_premium_total: float
    max_profit: float               # 元；inf 表示无上限
    max_loss: float                 # 元；负数；-inf 表示无下限
    breakevens: list[float]
    prob_profit: Optional[float]
    expected_pnl: Optional[float]
    delta_tons: float               # 组合净 delta 折算为吨
    gamma_tons_per_pt: float        # 价格每变动 1 元，delta 吨数变化
    vega_total: float               # 波动率每 +1 个点的盈亏（元）
    theta_total: float              # 每自然日时间价值变化（元）
    margin_estimate: float          # 元（近似）
    annualized_yield: Optional[float] = None   # 备兑 / 卖方：权利金年化收益
    protection_pct: Optional[float] = None     # 下行保护幅度（相对标的价）
    upside_cap: Optional[float] = None         # 收益封顶价
    scenarios: dict = field(default_factory=dict)   # 到期 ±1σ/±2σ 情景损益（元）

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["breakevens"] = [round(x, 1) for x in self.breakevens]
        d["scenarios"] = {k: round(v) for k, v in self.scenarios.items()}
        return d


@dataclass
class Strategy:
    name: str
    product: ProductSpec
    legs: list[Leg]
    rationale: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metrics: Optional[StrategyMetrics] = None
    score: float = 0.0

    # ------------------------------------------------------------------ 损益
    def payoff(self, prices: np.ndarray) -> np.ndarray:
        return sum(leg.payoff(prices) for leg in self.legs)

    def option_legs(self) -> list[Leg]:
        return [l for l in self.legs if l.kind in ("call", "put")]

    def hedge_tons(self) -> float:
        return sum(l.tons for l in self.legs if l.kind in ("spot", "future"))

    def evaluate(self, F: float, T: float, sigma: Optional[float], r: float = 0.0) -> StrategyMetrics:
        prices = np.linspace(F * 0.5, F * 1.5, 2001)
        pnl = self.payoff(prices)
        opt_tons = max((abs(l.tons) for l in self.option_legs()), default=1.0) or 1.0
        prem_total = -sum(l.price * l.tons for l in self.option_legs())
        prem_per_ton = prem_total / opt_tons

        # 极值：在 (0, 3F] 的宽网格上取极值；两端仍有斜率的方向视为"无上限 / 无下限"（含现货敞口的情形）
        wide = np.linspace(F * 1e-3, F * 3.0, 3001)
        pnl_wide = self.payoff(wide)
        slope_hi = (pnl_wide[-1] - pnl_wide[-2]) / (wide[-1] - wide[-2])
        slope_lo = (pnl_wide[1] - pnl_wide[0]) / (wide[1] - wide[0])
        has_linear = any(l.kind in ("spot", "future") for l in self.legs)
        max_profit = math.inf if slope_hi > 1e-9 else float(pnl_wide.max())
        max_loss = -math.inf if (slope_hi < -1e-9 or (slope_lo > 1e-9 and has_linear)) else float(pnl_wide.min())
        # 情景损益：±1σ / ±2σ（按到期波动），比"最大亏损"更贴近业务沟通
        scen = {}
        if sigma and T > 0:
            sd = sigma * math.sqrt(T)
            for label, k in (("-2σ", -2), ("-1σ", -1), ("0", 0), ("+1σ", 1), ("+2σ", 2)):
                px = F * math.exp(k * sd)
                scen[f"{label}({px:.0f})"] = float(self.payoff(np.array([px]))[0])

        bes = []
        sign = np.sign(pnl)
        for i in np.where(np.diff(sign) != 0)[0]:
            x0, x1, y0, y1 = prices[i], prices[i + 1], pnl[i], pnl[i + 1]
            bes.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x0))

        prob, exp_pnl = None, None
        if sigma and T > 0:
            # 对数正态密度数值积分
            lnF = math.log(F) - 0.5 * sigma * sigma * T
            sd = sigma * math.sqrt(T)
            dens = np.exp(-0.5 * ((np.log(prices) - lnF) / sd) ** 2) / (prices * sd * math.sqrt(2 * math.pi))
            w = dens * np.gradient(prices)
            w /= w.sum()
            prob = float(w[pnl > 0].sum())
            exp_pnl = float((w * pnl).sum())

        d = g = v = th = 0.0
        for leg in self.legs:
            gk = leg.greeks(F, T, r)
            d += gk.delta * leg.tons
            g += gk.gamma * leg.tons
            v += gk.vega * leg.tons
            th += gk.theta * leg.tons

        margin = estimate_margin(self, F)
        m = StrategyMetrics(prem_per_ton, prem_total, max_profit, max_loss, bes, prob, exp_pnl,
                            d, g, v, th, margin, scenarios=scen)
        hedge = self.hedge_tons()
        shorts = [l for l in self.option_legs() if l.qty < 0]
        if hedge > 0 and shorts and T > 0:
            m.annualized_yield = prem_total / (F * hedge) * 365 / max(T * 365, 1)
            m.protection_pct = prem_per_ton / F
            calls = [l for l in shorts if l.kind == "call"]
            if calls:
                m.upside_cap = min(l.strike for l in calls)
        elif hedge == 0 and shorts and prem_total > 0 and margin > 0 and T > 0:
            m.annualized_yield = prem_total / margin * 365 / max(T * 365, 1)
        self.metrics = m
        return m

    # ------------------------------------------------------------------ 执行
    def order_tickets(self) -> list[dict]:
        """生成可直接下到交易终端的委托清单（开仓）。"""
        out = []
        for leg in self.legs:
            if leg.kind == "spot":
                continue
            out.append({
                "合约": leg.symbol,
                "买卖": leg.side,
                "开平": "开仓",
                "手数": int(abs(leg.qty)),
                "参考价": round(leg.price, 2),
                "折合吨": abs(leg.tons),
                "类型": {"future": "期货", "call": "看涨期权", "put": "看跌期权"}[leg.kind],
            })
        return out

    def describe(self) -> str:
        legs = "；".join(
            f"{l.side}{int(abs(l.qty))}手 {l.symbol}@{l.price:.1f}" if l.kind != "spot"
            else f"现货{l.tons:.0f}吨(按{l.price:.0f}计价)" for l in self.legs)
        return f"{self.name}: {legs}"


# ---------------------------------------------------------------------- 保证金
def estimate_margin(strategy: Strategy, F: float) -> float:
    """交易所保证金近似估算（上线前请按经纪商实际参数核对）。

    - 期权空头单腿：max(权利金 + 期货保证金 - 0.5*虚值额, 权利金 + 0.5*期货保证金) * 乘数
    - 期货多头 + 卖 call（备兑组合）：只收期货保证金，期权空头不额外收
    - 借方价差（买低卖高 call / 买高卖低 put）：最大亏损即权利金，组合保证金 0
    - 贷方价差：行权价差 * 乘数
    - 现货 + 卖 call：现货不在交易所，期权空头按单腿收取（这是现货企业备兑与期货备兑的核心区别）
    """
    spec = strategy.product
    mult = spec.multiplier
    fut_margin_per_lot = F * mult * spec.futures_margin_rate
    legs = strategy.legs
    futs = [l for l in legs if l.kind == "future"]
    calls = [l for l in legs if l.kind == "call"]
    puts = [l for l in legs if l.kind == "put"]
    margin = 0.0

    def short_single(l: Leg) -> float:
        otm = max(l.strike - F, 0.0) if l.kind == "call" else max(F - l.strike, 0.0)
        per_ton = max(l.price + F * spec.futures_margin_rate - 0.5 * otm,
                      l.price + 0.5 * F * spec.futures_margin_rate)
        return per_ton * abs(l.tons)

    long_fut_lots = sum(l.qty for l in futs if l.qty > 0)
    short_fut_lots = -sum(l.qty for l in futs if l.qty < 0)
    margin += (long_fut_lots + short_fut_lots) * fut_margin_per_lot

    for group in (calls, puts):
        longs = sorted([l for l in group if l.qty > 0], key=lambda l: l.strike)
        shorts = sorted([l for l in group if l.qty < 0], key=lambda l: l.strike)
        covered_lots = long_fut_lots if group is calls else short_fut_lots
        for s in shorts:
            lots = abs(s.qty)
            cov = min(lots, covered_lots)
            covered_lots -= cov
            lots -= cov
            if lots <= 0:
                continue
            paired = None
            for lg in longs:
                if abs(lg.qty) >= lots and lg.multiplier == s.multiplier:
                    paired = lg
                    break
            if paired is None:
                margin += short_single(s) * lots / abs(s.qty)
                continue
            if (group is calls and paired.strike <= s.strike) or (group is puts and paired.strike >= s.strike):
                continue  # 借方价差
            margin += abs(paired.strike - s.strike) * mult * lots
    return margin


# ---------------------------------------------------------------------- 构建器
def _lots_for_tons(tons: float, spec: ProductSpec) -> int:
    return max(int(round(tons / spec.multiplier)), 1)


def _pick(chain: OptionChain, cp: str, *, price: Optional[float] = None, delta: Optional[float] = None,
          side: str = "nearest") -> Optional[OptionQuote]:
    if price is not None:
        o = chain.nearest_strike(cp, price, side)
        if o:
            return o
    if delta is not None:
        return chain.nearest_delta(cp, delta)
    return None


def _spot_leg(spec: ProductSpec, tons: float, mark: float) -> Leg:
    """现货腿按当前期货价计价：损益表示"相对什么都不做"的增量效果；历史成本只用于说明文字。
    现货与期货之间的基差风险不在此处建模，见 README 的基差说明。"""
    return Leg("spot", tons, mark, f"{spec.code}现货", multiplier=1.0)


def _leg_from_quote(o: OptionQuote, lots: int, spec: ProductSpec, is_long: bool) -> Leg:
    # 买方按卖一价、卖方按买一价作参考，保守估计成交成本
    px = (o.ask if is_long else o.bid) if (o.bid > 0 and o.ask > 0) else o.mid
    return Leg("call" if o.cp == "C" else "put", lots if is_long else -lots, px, o.symbol, o.strike, o.iv, spec.multiplier)


def covered_call(chain: OptionChain, tons: float, spot_cost: float, *, hedge_base: str = "spot",
                 target_price: Optional[float] = None, target_delta: float = 0.25) -> Optional[Strategy]:
    """备兑：持有现货（或期货多头）+ 卖出虚值看涨。target_price 优先（如库存目标出货价 / 阻力位）。"""
    spec = chain.product
    lots = _lots_for_tons(tons, spec)
    o = _pick(chain, "C", price=target_price, delta=target_delta, side="above")
    if not o:
        return None
    legs = []
    if hedge_base == "spot":
        legs.append(_spot_leg(spec, tons, chain.F))
    else:
        legs.append(Leg("future", lots, chain.F, chain.underlying, multiplier=spec.multiplier))
    legs.append(_leg_from_quote(o, lots, spec, is_long=False))
    why = [f"卖出行权价 {o.strike:g} 看涨，收权利金 {legs[-1].price:.1f} 元/吨，相当于把库存目标出货价锁在 {o.strike:g}",
           f"若到期价格 < {o.strike:g}，权利金全部落袋，增厚库存收益；若 > {o.strike:g}，按 {o.strike:g} 价格出货（含权利金等效 {o.strike + legs[-1].price:.0f}）"]
    if hedge_base == "future":
        why.append("期货多头 + 卖 call 为交易所认可的备兑组合，期权空头不另收保证金")
    else:
        why.append("现货备兑：期权空头需缴纳保证金，注意资金占用与追保")
        if spot_cost:
            why.append(f"相对现货成本 {spot_cost:.0f}，封顶出货价含权利金为 {o.strike + legs[-1].price:.0f}，"
                       f"对应库存利润 {(o.strike + legs[-1].price - spot_cost):+.0f} 元/吨")
    return Strategy("备兑看涨（Covered Call）", spec, legs, why, ["收租", "库存", "中性偏多"])


def protective_put(chain: OptionChain, tons: float, spot_cost: float, *, floor_price: Optional[float] = None,
                   target_delta: float = 0.30) -> Optional[Strategy]:
    spec = chain.product
    lots = _lots_for_tons(tons, spec)
    o = _pick(chain, "P", price=floor_price, delta=target_delta, side="below")
    if not o:
        return None
    legs = [_spot_leg(spec, tons, chain.F), _leg_from_quote(o, lots, spec, is_long=True)]
    why = [f"买入行权价 {o.strike:g} 看跌，为库存设定保底价 {o.strike:g}（扣权利金后实际保底 {o.strike - legs[-1].price:.0f}）",
           "保留全部上涨空间，适合波动率不高、下行风险事件临近时使用"]
    if spot_cost:
        why.append(f"相对现货成本 {spot_cost:.0f}，最差情形库存损益被锁定在 {(o.strike - legs[-1].price - spot_cost):+.0f} 元/吨")
    return Strategy("保护性看跌（Protective Put）", spec, legs, why, ["保险", "库存", "买方"])


def collar(chain: OptionChain, tons: float, spot_cost: float, *, floor_price: Optional[float] = None,
           cap_price: Optional[float] = None, zero_cost: bool = True) -> Optional[Strategy]:
    """领口：现货 + 买 put（保底）+ 卖 call（融资）。zero_cost=True 时在满足 floor 的前提下寻找最接近零成本的 call。"""
    spec = chain.product
    lots = _lots_for_tons(tons, spec)
    p = _pick(chain, "P", price=floor_price, delta=0.30, side="below")
    if not p:
        return None
    put_leg = _leg_from_quote(p, lots, spec, is_long=True)
    c = None
    if cap_price is not None:
        c = _pick(chain, "C", price=cap_price, side="above")
    if c is None and zero_cost:
        cands = [o for o in chain.liquid() if o.cp == "C" and o.strike > chain.F]
        if cands:
            c = min(cands, key=lambda o: abs(o.bid - put_leg.price))
    if c is None:
        c = _pick(chain, "C", delta=0.25)
    if not c:
        return None
    call_leg = _leg_from_quote(c, lots, spec, is_long=False)
    legs = [_spot_leg(spec, tons, chain.F), put_leg, call_leg]
    net = call_leg.price - put_leg.price
    why = [f"买 {p.strike:g} 看跌保底 + 卖 {c.strike:g} 看涨融资，净{'收' if net >= 0 else '付'}权利金 {abs(net):.1f} 元/吨",
           f"库存价值被锁定在区间 [{p.strike:g}, {c.strike:g}]，适合有保值要求但不愿付高额保费的库存",
           "交易所通常不对现货领口给予组合保证金，若以期货多头替代现货则可享受备兑优惠"]
    if spot_cost:
        why.append(f"相对现货成本 {spot_cost:.0f}：最差 {(p.strike + net - spot_cost):+.0f} 元/吨，最好 {(c.strike + net - spot_cost):+.0f} 元/吨")
    return Strategy("领口（Collar）", spec, legs, why, ["保值", "库存", "区间"])


def cash_secured_put(chain: OptionChain, tons: float, *, target_buy_price: Optional[float] = None,
                     target_delta: float = 0.30) -> Optional[Strategy]:
    """采购端：卖出看跌，等价于"愿意在 K 价位接货并先收权利金"。"""
    spec = chain.product
    lots = _lots_for_tons(tons, spec)
    o = _pick(chain, "P", price=target_buy_price, delta=target_delta, side="below")
    if not o:
        return None
    leg = _leg_from_quote(o, lots, spec, is_long=False)
    why = [f"卖出 {o.strike:g} 看跌收权利金 {leg.price:.1f}：若跌破 {o.strike:g} 则以 {o.strike:g} 接货（实际采购成本 {o.strike - leg.price:.0f}），否则权利金降低采购成本",
           "适合有刚性采购计划、且接受在更低价位接货的场景；需预留行权后接期货多头的资金"]
    return Strategy("卖出看跌降低采购成本（Cash-Secured Put）", spec, legs=[leg], rationale=why, tags=["采购", "卖方", "中性偏多"])


def _vertical(chain: OptionChain, lots: int, cp: str, long_k: float, short_k: float, name: str,
              why: list[str], tags: list[str]) -> Optional[Strategy]:
    lo, so = chain.get(cp, long_k), chain.get(cp, short_k)
    if not lo or not so or lo.mid <= 0 or so.mid <= 0:
        return None
    spec = chain.product
    legs = [_leg_from_quote(lo, lots, spec, True), _leg_from_quote(so, lots, spec, False)]
    return Strategy(name, spec, legs, why, tags)


def bull_call_spread(chain: OptionChain, lots: int, *, long_strike: Optional[float] = None,
                     short_strike: Optional[float] = None, target_price: Optional[float] = None) -> Optional[Strategy]:
    lk = long_strike or chain.atm_strike()
    so = _pick(chain, "C", price=target_price, delta=0.25, side="above") if short_strike is None else chain.get("C", short_strike)
    if not so or so.strike <= lk:
        return None
    why = [f"买 {lk:g} 看涨 / 卖 {so.strike:g} 看涨：看涨至 {so.strike:g} 附近，用卖腿把成本降下来",
           "最大亏损 = 净权利金，最大盈利 = 行权价差 - 净权利金；交易所对借方价差不收保证金"]
    return _vertical(chain, lots, "C", lk, so.strike, "牛市看涨价差（Bull Call Spread）", why, ["方向", "看涨", "借方"])


def bull_put_spread(chain: OptionChain, lots: int, *, short_strike: Optional[float] = None,
                    support_price: Optional[float] = None, width_steps: int = 2) -> Optional[Strategy]:
    so = _pick(chain, "P", price=support_price, delta=0.30, side="below") if short_strike is None else chain.get("P", short_strike)
    if not so:
        return None
    step = chain.product.strike_step(so.strike)
    lk = so.strike - width_steps * step
    lo = chain.nearest_strike("P", lk, "below", liquid_only=False)
    if not lo or lo.strike >= so.strike:
        return None
    why = [f"卖 {so.strike:g} 看跌 / 买 {lo.strike:g} 看跌：只要到期不跌破 {so.strike:g}（支撑位）即赚取全部净权利金",
           "IV 偏高时优于牛市看涨价差；最大亏损 = 行权价差 - 净权利金，保证金 ≈ 行权价差 × 乘数"]
    return _vertical(chain, lots, "P", lo.strike, so.strike, "牛市看跌价差（Bull Put Spread，收权利金）", why, ["方向", "看涨", "贷方"])


def bear_put_spread(chain: OptionChain, lots: int, *, long_strike: Optional[float] = None,
                    short_strike: Optional[float] = None, target_price: Optional[float] = None) -> Optional[Strategy]:
    lk = long_strike or chain.atm_strike()
    so = _pick(chain, "P", price=target_price, delta=0.25, side="below") if short_strike is None else chain.get("P", short_strike)
    if not so or so.strike >= lk:
        return None
    why = [f"买 {lk:g} 看跌 / 卖 {so.strike:g} 看跌：看跌至 {so.strike:g} 附近，卖腿降低成本",
           "适合库存阶段性看空但不想减现货：价差盈利对冲库存跌价，最大亏损锁定为净权利金"]
    return _vertical(chain, lots, "P", lk, so.strike, "熊市看跌价差（Bear Put Spread）", why, ["方向", "看跌", "借方"])


def bear_call_spread(chain: OptionChain, lots: int, *, short_strike: Optional[float] = None,
                     resistance_price: Optional[float] = None, width_steps: int = 2) -> Optional[Strategy]:
    so = _pick(chain, "C", price=resistance_price, delta=0.30, side="above") if short_strike is None else chain.get("C", short_strike)
    if not so:
        return None
    step = chain.product.strike_step(so.strike)
    lo = chain.nearest_strike("C", so.strike + width_steps * step, "above", liquid_only=False)
    if not lo or lo.strike <= so.strike:
        return None
    why = [f"卖 {so.strike:g} 看涨 / 买 {lo.strike:g} 看涨：只要到期不突破 {so.strike:g}（阻力位）即赚取全部净权利金",
           "IV 偏高时的看空 / 中性偏空收租结构；最大亏损 = 行权价差 - 净权利金"]
    return _vertical(chain, lots, "C", lo.strike, so.strike, "熊市看涨价差（Bear Call Spread，收权利金）", why, ["方向", "看跌", "贷方"])
