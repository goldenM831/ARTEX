"""策略推荐引擎：行情 + 波动状态 + 技术面 + 业务意图 → 排序后的可执行策略组合。

流程：
  1. 期权链计算 IV / Greeks，做数据质量检查
  2. 波动率监测（HV/ATR/布林 + IV-HV 比较）→ 买方 / 卖方倾向
  3. 技术指标 → 方向分数 + 支撑 / 阻力
  4. 按业务角色生成候选（库存 / 采购 / 方向交易）
  5. 评估（损益、概率、Greeks、保证金）并打分排序
  6. 输出：推荐理由 + 关键指标 + 可直接下单的委托清单
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from . import strategies as S
from .chain import OptionChain
from .strategies import Strategy
from .ta import TASnapshot, analyze
from .vol import VolSignal, detect_vol_regime


@dataclass
class Intent:
    role: str = "inventory"             # inventory / procurement / directional
    tons: float = 1000.0
    spot_cost: Optional[float] = None
    view: str = "auto"                   # auto / bull / bear / neutral
    target_price: Optional[float] = None  # 库存目标出货价 / 采购目标价
    floor_price: Optional[float] = None   # 库存保底价
    max_premium_per_ton: Optional[float] = None   # 愿意为保护支付的最高权利金
    margin_budget: Optional[float] = None         # 可用保证金（元）
    min_oi: int = 50
    max_spread_pct: float = 0.15


@dataclass
class Report:
    asof: dt.datetime
    underlying: str
    F: float
    expiry: dt.date
    dte: int
    atm_iv: Optional[float]
    skew_25d: Optional[float]
    vol: VolSignal
    ta: TASnapshot
    view: str
    intent: Intent
    data_issues: list[str]
    strategies: list[Strategy]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "asof": self.asof.isoformat(timespec="seconds"), "underlying": self.underlying, "F": self.F,
            "expiry": self.expiry.isoformat(), "dte": self.dte, "atm_iv": self.atm_iv, "skew_25d": self.skew_25d,
            "vol": self.vol.__dict__, "ta": self.ta.__dict__, "view": self.view, "data_issues": self.data_issues,
            "strategies": [{
                "name": s.name, "score": round(s.score, 3), "tags": s.tags, "legs": s.order_tickets(),
                "metrics": s.metrics.as_dict() if s.metrics else None, "rationale": s.rationale,
            } for s in self.strategies],
            "notes": self.notes,
        }

    def to_text(self, top: int = 5) -> str:
        L = []
        L.append(f"=== {self.underlying} 期权策略建议  @ {self.asof:%Y-%m-%d %H:%M}  期货={self.F:.1f}  到期={self.expiry} ({self.dte}天) ===")
        iv = f"{self.atm_iv:.1%}" if self.atm_iv else "-"
        sk = f"{self.skew_25d:+.1%}" if self.skew_25d is not None else "-"
        L.append(f"ATM IV={iv}  25dSkew(P-C)={sk}")
        L.append("波动: " + self.vol.summary())
        for n in self.vol.notes:
            L.append("   · " + n)
        L.append("技术: " + self.ta.summary())
        L.append("   · " + "；".join(self.ta.reasons))
        L.append(f"方向观点: {self.view}   业务: {self.intent.role}  数量: {self.intent.tons:.0f} 吨")
        if self.data_issues:
            L.append("数据告警: " + "；".join(self.data_issues[:6]) + (" …" if len(self.data_issues) > 6 else ""))
        for n in self.notes:
            L.append("提示: " + n)
        L.append("")
        for i, s in enumerate(self.strategies[:top], 1):
            m = s.metrics
            L.append(f"#{i} [{s.score:.2f}] {s.describe()}")
            if m:
                mp = "无上限" if math.isinf(m.max_profit) else f"{m.max_profit:,.0f}"
                ml = "无下限" if math.isinf(m.max_loss) else f"{m.max_loss:,.0f}"
                pp = f"{m.prob_profit:.0%}" if m.prob_profit is not None else "-"
                be = "/".join(f"{b:.0f}" for b in m.breakevens) or "-"
                L.append(f"    净权利金 {m.net_premium_per_ton:+.1f} 元/吨 ({m.net_premium_total:+,.0f} 元)  最大盈利 {mp}  最大亏损 {ml}  盈亏平衡 {be}  盈利概率 {pp}")
                extra = []
                if m.annualized_yield is not None:
                    basis = "权利金/库存市值" if s.hedge_tons() > 0 else "权利金/保证金"
                    extra.append(f"年化({basis}) {m.annualized_yield:.1%}")
                if m.protection_pct is not None:
                    extra.append(f"下行缓冲 {m.protection_pct:.1%}")
                if m.upside_cap is not None:
                    extra.append(f"收益封顶 {m.upside_cap:.0f}")
                L.append(f"    Δ {m.delta_tons:+.0f} 吨  Γ {m.gamma_tons_per_pt:+.2f} 吨/元  Vega {m.vega_total:+,.0f} 元/点  Theta {m.theta_total:+,.0f} 元/日  保证金≈{m.margin_estimate:,.0f}  " + "  ".join(extra))
                if m.scenarios:
                    L.append("    到期情景: " + "  ".join(f"{k} {v:+,.0f}" for k, v in m.scenarios.items()))
            for r in s.rationale:
                L.append("    - " + r)
            L.append("    委托: " + " | ".join(f"{t['买卖']}{t['开平']} {t['合约']} x{t['手数']} @{t['参考价']}" for t in s.order_tickets()))
        return "\n".join(L)


def _far_enough(level: Optional[float], F: float, min_pct: float) -> Optional[float]:
    if level is None or abs(level - F) / F < min_pct:
        return None
    return level


class Recommender:
    def __init__(self, r: float = 0.0):
        self.r = r

    # ------------------------------------------------------------------ 主入口
    def recommend(self, chain: OptionChain, bars: pd.DataFrame, intent: Intent,
                  iv_history: Optional[pd.Series] = None) -> Report:
        chain.r = self.r
        chain.compute_iv()
        issues = chain.sanity_issues()
        atm_iv = chain.atm_iv()
        vol = detect_vol_regime(bars, atm_iv, iv_history)
        ta = analyze(bars)
        view = intent.view if intent.view != "auto" else ("bull" if ta.score > 0.15 else "bear" if ta.score < -0.15 else "neutral")
        F, T = chain.F, chain.T
        sigma = atm_iv or (vol.hv20 if not math.isnan(vol.hv20) else 0.25)
        notes: list[str] = []
        if chain.dte < 7:
            notes.append("距到期不足 7 天，Gamma 风险大、权利金极薄，建议换到下一到期月")
        if not chain.liquid(intent.max_spread_pct, intent.min_oi):
            notes.append("当前期权链没有满足流动性门槛的合约，建议只做主力月份或放宽门槛")

        cands = self._candidates(chain, intent, view, vol, ta)
        out: list[Strategy] = []
        for s in cands:
            if s is None:
                continue
            m = s.evaluate(F, T, sigma, self.r)
            if intent.max_premium_per_ton is not None and -m.net_premium_per_ton > intent.max_premium_per_ton:
                continue
            if intent.margin_budget is not None and m.margin_estimate > intent.margin_budget:
                s.rationale.append(f"保证金 {m.margin_estimate:,.0f} 超过预算 {intent.margin_budget:,.0f}，已降权")
                s.score -= 0.5
            s.score += self._score(s, intent, view, vol, chain)
            out.append(s)
        out.sort(key=lambda s: s.score, reverse=True)
        return Report(chain.asof, chain.underlying, F, chain.expiry, chain.dte, atm_iv, chain.skew_25d(),
                      vol, ta, view, intent, issues, out, notes)

    # ------------------------------------------------------------------ 候选
    def _candidates(self, chain: OptionChain, it: Intent, view: str, vol: VolSignal, ta: TASnapshot) -> list[Optional[Strategy]]:
        F = chain.F
        spec = chain.product
        lots = S._lots_for_tons(it.tons, spec)
        cost = it.spot_cost or F
        # 技术位离现价太近时不作为行权价依据（否则会选出几乎平值、保护/增强意义很弱的腿），改用 delta 选档
        res = _far_enough(ta.resistance, F, 0.015)
        sup = _far_enough(ta.support, F, 0.015)
        target = it.target_price or res
        floor = it.floor_price or sup or cost * 0.95
        c: list[Optional[Strategy]] = []
        if it.role == "inventory":
            c.append(S.covered_call(chain, it.tons, cost, hedge_base="spot", target_price=target))
            c.append(S.covered_call(chain, it.tons, cost, hedge_base="future", target_price=target))
            c.append(S.collar(chain, it.tons, cost, floor_price=floor, cap_price=target))
            c.append(S.protective_put(chain, it.tons, cost, floor_price=floor))
            if view == "bear":
                c.append(S.bear_put_spread(chain, lots, target_price=sup))
            if view != "bear":
                c.append(S.bull_put_spread(chain, lots, support_price=sup))
        elif it.role == "procurement":
            c.append(S.cash_secured_put(chain, it.tons, target_buy_price=it.target_price or sup))
            c.append(S.bull_call_spread(chain, lots, target_price=res))
            if view == "bull":
                c.append(S.bull_put_spread(chain, lots, support_price=sup))
            if view == "bear":
                c.append(S.bear_call_spread(chain, lots, resistance_price=res))
        else:  # directional
            if view == "bull":
                c.append(S.bull_call_spread(chain, lots, target_price=res))
                c.append(S.bull_put_spread(chain, lots, support_price=sup))
            elif view == "bear":
                c.append(S.bear_put_spread(chain, lots, target_price=sup))
                c.append(S.bear_call_spread(chain, lots, resistance_price=res))
            else:
                c.append(S.bull_put_spread(chain, lots, support_price=sup))
                c.append(S.bear_call_spread(chain, lots, resistance_price=res))
                c.append(S.covered_call(chain, it.tons, F, hedge_base="future", target_price=res))
        return c

    # ------------------------------------------------------------------ 打分
    def _score(self, s: Strategy, it: Intent, view: str, vol: VolSignal, chain: OptionChain) -> float:
        m = s.metrics
        sc = 0.0
        # 1) 方向匹配
        if view == "bull" and "看涨" in s.tags or view == "bear" and "看跌" in s.tags:
            sc += 0.6
        if view == "neutral" and ("收租" in s.tags or "区间" in s.tags or "贷方" in s.tags):
            sc += 0.4
        if view == "bull" and "看跌" in s.tags or view == "bear" and "看涨" in s.tags:
            sc -= 0.8
        if view == "bear" and "中性偏多" in s.tags:
            sc -= 0.4
        # 2) 波动率倾向：IV 偏贵→偏好收权利金的结构；IV 偏便宜→偏好买方
        credit = m.net_premium_total > 0
        if vol.bias == "seller":
            sc += 0.5 if credit else -0.3
        elif vol.bias == "buyer":
            sc += 0.4 if not credit else -0.2
        if vol.regime == "spike" and not credit and "买方" in s.tags:
            sc -= 0.3
        if vol.regime == "squeeze" and credit and it.role == "directional":
            sc -= 0.3
        # 3) 盈利概率与盈亏比
        if m.prob_profit is not None:
            sc += 0.6 * (m.prob_profit - 0.5)
        if math.isfinite(m.max_profit) and math.isfinite(m.max_loss) and m.max_loss < 0:
            rr = m.max_profit / abs(m.max_loss)
            sc += min(rr, 3.0) * 0.1
        if m.expected_pnl is not None and it.tons > 0:
            sc += max(min(m.expected_pnl / (chain.F * it.tons) * 20, 0.3), -0.3)
        # 4) 库存场景：优先真正对冲库存的结构；纯期权增强结构（不含现货/期货腿）降权
        if it.role == "inventory":
            if s.hedge_tons() > 0:
                if m.annualized_yield is not None:
                    sc += min(m.annualized_yield, 0.3)
            else:
                sc -= 0.6
                s.rationale.append("该结构不对冲库存下跌，只作为增强收益的补充，已在库存场景下降权")
        # 5) 流动性惩罚
        for leg in s.option_legs():
            q = chain.get("C" if leg.kind == "call" else "P", leg.strike)
            if q is not None:
                sp = q.spread_pct()
                if not math.isfinite(sp) or sp > it.max_spread_pct:
                    sc -= 0.4
                elif sp > it.max_spread_pct / 2:
                    sc -= 0.15
                if q.open_interest < it.min_oi:
                    sc -= 0.4
        return sc
