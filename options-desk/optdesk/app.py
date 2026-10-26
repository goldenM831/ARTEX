"""命令行入口。

    python -m optdesk demo   --product TA --role inventory --tons 5000 --cost 4800
    python -m optdesk live   --feed tqsdk --underlying CZCE.TA601 --role inventory --tons 5000 --cost 4800 --interval 30
    python -m optdesk replay --underlying TA601 --futures-csv data/TA601.csv --role directional --tons 500
    python -m optdesk chain  --product TA        # 打印合成期权链（IV / Greeks / 流动性）

实时模式支持 --webhook 推送（企业微信 / 钉钉 机器人），当首选策略或波动状态变化时推送摘要。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from .contracts import default_underlying
from .feeds.replay import CsvReplayFeed, SyntheticFeed
from .recommender import Intent, Recommender, Report


def _intent_from_args(a) -> Intent:
    return Intent(role=a.role, tons=a.tons, spot_cost=a.cost, view=a.view, target_price=a.target,
                  floor_price=a.floor, max_premium_per_ton=a.max_premium, margin_budget=a.margin_budget)


def _add_intent_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--role", choices=["inventory", "procurement", "directional"], default="inventory",
                   help="inventory=库存保值/增强, procurement=采购, directional=方向交易")
    p.add_argument("--tons", type=float, default=1000)
    p.add_argument("--cost", type=float, default=None, help="现货成本（元/吨）")
    p.add_argument("--view", choices=["auto", "bull", "bear", "neutral"], default="auto")
    p.add_argument("--target", type=float, default=None, help="目标出货价 / 采购价")
    p.add_argument("--floor", type=float, default=None, help="保底价")
    p.add_argument("--max-premium", type=float, default=None, help="可接受最高净付权利金（元/吨）")
    p.add_argument("--margin-budget", type=float, default=None, help="保证金预算（元）")
    p.add_argument("--json", action="store_true", help="输出 JSON")
    p.add_argument("--top", type=int, default=5)


def _emit(report: Report, as_json: bool, top: int) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=str))
    else:
        print(report.to_text(top))


def _push_webhook(url: str, text: str) -> None:
    import urllib.request
    body = json.dumps({"msgtype": "text", "text": {"content": text[:3800]}}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=5).read()
    except Exception as e:  # pragma: no cover
        print(f"[webhook] 推送失败: {e}", file=sys.stderr)


def cmd_demo(a) -> None:
    underlying = a.underlying or default_underlying(a.product)
    feed = SyntheticFeed(underlying, start_price=a.price, vol=a.vol, seed=a.seed, drift=a.drift)
    bars = feed.get_bars(underlying)
    chain = feed.get_chain(underlying)
    report = Recommender().recommend(chain, bars, _intent_from_args(a))
    _emit(report, a.json, a.top)


def cmd_chain(a) -> None:
    underlying = a.underlying or default_underlying(a.product)
    feed = SyntheticFeed(underlying, start_price=a.price, vol=a.vol, seed=a.seed)
    chain = feed.get_chain(underlying).compute_iv()
    print(f"{underlying} F={chain.F:.1f} T={chain.T:.3f} 到期={chain.expiry} ATM IV={chain.atm_iv():.2%}")
    print(f"{'合约':<16}{'买':>8}{'卖':>8}{'中':>8}{'IV':>8}{'Delta':>8}{'OI':>9}{'点差%':>7}")
    for o in sorted(chain.options, key=lambda o: (o.cp, o.strike)):
        iv = f"{o.iv:.1%}" if o.iv else "-"
        d = f"{o.greeks.delta:+.2f}" if o.greeks else "-"
        print(f"{o.symbol:<16}{o.bid:>8.1f}{o.ask:>8.1f}{o.mid:>8.1f}{iv:>8}{d:>8}{o.open_interest:>9}{o.spread_pct()*100:>7.1f}")


def cmd_replay(a) -> None:
    feed = CsvReplayFeed(a.underlying, a.futures_csv, a.options_csv)
    feed.seek(dt.date.fromisoformat(a.start)) if a.start else None
    rec = Recommender()
    intent = _intent_from_args(a)
    rows = []
    while True:
        bars = feed.get_bars(a.underlying)
        if len(bars) >= 80:
            try:
                rep = rec.recommend(feed.get_chain(a.underlying), bars, intent)
                top = rep.strategies[0] if rep.strategies else None
                rows.append({"date": feed.asof.date().isoformat(), "F": rep.F, "view": rep.view, "regime": rep.vol.regime,
                             "bias": rep.vol.bias, "top": top.name if top else None,
                             "prob": top.metrics.prob_profit if top and top.metrics else None})
                if not a.json:
                    print(f"{rows[-1]['date']} F={rep.F:.0f} {rep.view:<7} {rep.vol.regime:<9} {rep.vol.bias:<7} -> {rows[-1]['top']}")
            except LookupError as e:
                print(f"{feed.asof.date()} 跳过: {e}")
        if not feed.step():
            break
    if a.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))


def _build_live_feed(a):
    if a.feed == "tqsdk":
        from .feeds.tqsdk_feed import TqFeed
        user = a.tq_user or os.environ.get("TQ_USER")
        pwd = a.tq_password or os.environ.get("TQ_PASSWORD")
        if not user or not pwd:
            sys.exit("请提供 --tq-user/--tq-password 或环境变量 TQ_USER/TQ_PASSWORD")
        return TqFeed(user, pwd, [a.underlying])
    if a.feed == "ctp":
        sys.exit("CTP 模式需要在代码中注入期权合约清单与 K 线提供者，见 feeds/ctp_feed.py 文档")
    return SyntheticFeed(a.underlying)


def cmd_live(a) -> None:
    feed = _build_live_feed(a)
    feed.start()
    rec = Recommender()
    intent = _intent_from_args(a)
    last_key: Optional[tuple] = None
    log_dir = Path(a.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            try:
                bars = feed.get_bars(a.underlying)
                chain = feed.get_chain(a.underlying)
                rep = rec.recommend(chain, bars, intent)
            except Exception as e:
                print(f"[{dt.datetime.now():%H:%M:%S}] 刷新失败: {e}", file=sys.stderr)
                time.sleep(a.interval)
                continue
            top = rep.strategies[0] if rep.strategies else None
            key = (top.name if top else None, tuple(t["合约"] for t in top.order_tickets()) if top else (), rep.vol.regime, rep.vol.bias)
            os.system("cls" if os.name == "nt" else "clear") if not a.json else None
            _emit(rep, a.json, a.top)
            with open(log_dir / f"signals_{dt.date.today():%Y%m%d}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rep.to_dict(), ensure_ascii=False, default=str) + "\n")
            if key != last_key and a.webhook:
                _push_webhook(a.webhook, rep.to_text(top=2))
            last_key = key
            time.sleep(a.interval)
    except KeyboardInterrupt:
        pass
    finally:
        feed.stop()


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="optdesk", description="国内商品期权策略工具")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="合成行情跑一遍完整推荐流程")
    d.add_argument("--product", default="TA")
    d.add_argument("--underlying", default=None)
    d.add_argument("--price", type=float, default=4800)
    d.add_argument("--vol", type=float, default=0.22)
    d.add_argument("--drift", type=float, default=0.0)
    d.add_argument("--seed", type=int, default=7)
    _add_intent_args(d)
    d.set_defaults(func=cmd_demo)

    c = sub.add_parser("chain", help="打印期权链（合成）")
    c.add_argument("--product", default="TA")
    c.add_argument("--underlying", default=None)
    c.add_argument("--price", type=float, default=4800)
    c.add_argument("--vol", type=float, default=0.22)
    c.add_argument("--seed", type=int, default=7)
    c.set_defaults(func=cmd_chain)

    r = sub.add_parser("replay", help="按日回放 CSV，检验引擎历史上会给出什么建议")
    r.add_argument("--underlying", required=True)
    r.add_argument("--futures-csv", required=True)
    r.add_argument("--options-csv", default=None)
    r.add_argument("--start", default=None)
    _add_intent_args(r)
    r.set_defaults(func=cmd_replay)

    lv = sub.add_parser("live", help="实时模式")
    lv.add_argument("--feed", choices=["tqsdk", "ctp", "synthetic"], default="tqsdk")
    lv.add_argument("--underlying", required=True, help="天勤风格代码，如 CZCE.TA601")
    lv.add_argument("--interval", type=int, default=30, help="刷新间隔（秒）")
    lv.add_argument("--tq-user", default=None)
    lv.add_argument("--tq-password", default=None)
    lv.add_argument("--webhook", default=None, help="企业微信 / 钉钉机器人 webhook")
    lv.add_argument("--log-dir", default="signals")
    _add_intent_args(lv)
    lv.set_defaults(func=cmd_live)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
