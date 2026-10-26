"""Streamlit 看板：交易员日常使用界面（第一版）。

    streamlit run optdesk/dashboard.py

左侧填业务参数（品种 / 角色 / 吨数 / 成本 / 观点），右侧展示：波动监测、技术面、期权链、推荐策略（含委托单与到期损益图）。
行情源默认为合成数据；配置 TQ_USER / TQ_PASSWORD 环境变量后可切换天勤实时。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optdesk.contracts import default_underlying, get_product, list_products  # noqa: E402
from optdesk.feeds.replay import SyntheticFeed  # noqa: E402
from optdesk.recommender import Intent, Recommender  # noqa: E402

st.set_page_config(page_title="期权策略台", layout="wide")
st.title("商品期权策略台 · optdesk")

with st.sidebar:
    st.header("业务参数")
    product = st.selectbox("品种", list_products(), index=0)
    spec = get_product(product)
    feed_kind = st.radio("行情源", ["合成(演示)", "天勤实时"], index=0, horizontal=True)
    underlying = st.text_input("合约", default_underlying(product) if feed_kind == "合成(演示)" else f"{spec.exchange}.{default_underlying(product)}")
    role = st.selectbox("业务角色", ["inventory", "procurement", "directional"],
                        format_func=lambda x: {"inventory": "库存保值/增强", "procurement": "采购", "directional": "方向交易"}[x])
    tons = st.number_input("数量（吨）", 0.0, 1e7, 5000.0, step=float(spec.multiplier * 10))
    cost = st.number_input("现货成本（元/吨，0=忽略）", 0.0, 1e6, 0.0)
    view = st.selectbox("方向观点", ["auto", "bull", "bear", "neutral"],
                        format_func=lambda x: {"auto": "自动（技术面）", "bull": "看涨", "bear": "看跌", "neutral": "中性"}[x])
    target = st.number_input("目标价（0=按阻力位）", 0.0, 1e6, 0.0)
    floor = st.number_input("保底价（0=按支撑位）", 0.0, 1e6, 0.0)
    if feed_kind == "合成(演示)":
        price = st.number_input("当前价（合成）", 1.0, 1e6, 4800.0)
        vol = st.slider("真实波动率（合成）", 0.05, 0.8, 0.22)
    refresh = st.button("刷新 / 重新计算", type="primary")

intent = Intent(role=role, tons=tons, spot_cost=cost or None, view=view,
                target_price=target or None, floor_price=floor or None)


@st.cache_resource(show_spinner=False)
def _tq_feed(user: str, pwd: str, u: str):
    from optdesk.feeds.tqsdk_feed import TqFeed
    f = TqFeed(user, pwd, [u])
    f.start()
    return f


if feed_kind == "天勤实时":
    user, pwd = os.environ.get("TQ_USER"), os.environ.get("TQ_PASSWORD")
    if not user or not pwd:
        st.error("请设置环境变量 TQ_USER / TQ_PASSWORD")
        st.stop()
    feed = _tq_feed(user, pwd, underlying)
else:
    feed = SyntheticFeed(underlying, start_price=price, vol=vol)

bars = feed.get_bars(underlying)
chain = feed.get_chain(underlying)
report = Recommender().recommend(chain, bars, intent)

c1, c2, c3, c4 = st.columns(4)
c1.metric("期货价", f"{report.F:.1f}")
c2.metric("ATM IV", f"{report.atm_iv:.1%}" if report.atm_iv else "-", f"IV-HV {report.vol.iv_minus_hv:+.1%}" if report.vol.iv_minus_hv is not None else None)
c3.metric("HV20 / HV60", f"{report.vol.hv20:.1%} / {report.vol.hv60:.1%}")
c4.metric("到期", f"{report.expiry}  ({report.dte}天)")

st.subheader("波动监测 & 技术面")
colA, colB = st.columns(2)
with colA:
    st.markdown(f"**{report.vol.regime} / 倾向 {report.vol.bias}** — " + report.vol.summary())
    for n in report.vol.notes:
        st.caption("· " + n)
with colB:
    st.markdown(f"**{report.ta.label} ({report.ta.score:+.2f})** — " + report.ta.summary())
    st.caption("· " + "；".join(report.ta.reasons))
if report.data_issues:
    st.warning("数据告警：" + "；".join(report.data_issues[:8]))
for n in report.notes:
    st.info(n)

st.line_chart(bars[["close"]].assign(MA20=bars["close"].rolling(20).mean(), MA60=bars["close"].rolling(60).mean()))

st.subheader("推荐策略")
for i, s in enumerate(report.strategies[:5], 1):
    m = s.metrics
    with st.expander(f"#{i}  [{s.score:.2f}]  {s.describe()}", expanded=(i == 1)):
        if m:
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("净权利金 元/吨", f"{m.net_premium_per_ton:+.1f}")
            k2.metric("最大盈利", "无上限" if np.isinf(m.max_profit) else f"{m.max_profit:,.0f}")
            k3.metric("最大亏损", "无下限" if np.isinf(m.max_loss) else f"{m.max_loss:,.0f}")
            k4.metric("盈利概率", f"{m.prob_profit:.0%}" if m.prob_profit is not None else "-")
            k5.metric("保证金≈", f"{m.margin_estimate:,.0f}")
            st.caption(f"Δ {m.delta_tons:+.0f} 吨 · Vega {m.vega_total:+,.0f} 元/点 · Theta {m.theta_total:+,.0f} 元/日 · 盈亏平衡 {', '.join(f'{b:.0f}' for b in m.breakevens) or '-'}")
            if m.scenarios:
                scen = pd.DataFrame({"情景": list(m.scenarios.keys()),
                                     "到期损益(元)": [f"{v:,.0f}" for v in m.scenarios.values()]})
                st.dataframe(scen, hide_index=True, use_container_width=True)
        for r in s.rationale:
            st.markdown("- " + r)
        st.markdown("**委托清单**")
        st.dataframe(pd.DataFrame(s.order_tickets()), hide_index=True, use_container_width=True)
        xs = np.linspace(report.F * 0.85, report.F * 1.15, 200)
        st.line_chart(pd.DataFrame({"到期损益": s.payoff(xs)}, index=np.round(xs, 0)))

st.subheader("期权链")
rows = [{"合约": o.symbol, "类型": o.cp, "行权价": o.strike, "买": o.bid, "卖": o.ask, "IV": o.iv,
         "Delta": o.greeks.delta if o.greeks else None, "持仓": o.open_interest, "点差%": o.spread_pct() * 100}
        for o in sorted(chain.options, key=lambda o: (o.cp, o.strike))]
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
