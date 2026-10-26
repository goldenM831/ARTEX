"""品种规格、合约代码解析、到期日估算。

国内三大商品交易所的期权代码风格不同：
  CZCE : TA601C4800        (品种 + yym m(3位) + C/P + 行权价)
  DCE  : PP2601-C-7000     (品种 + yymm + -C/-P- + 行权价)
  SHFE : ru2601C15000      (小写品种 + yymm + C/P + 行权价)
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import yaml

_CONFIG_PATH = Path(__file__).parent / "config" / "products.yaml"


@dataclass(frozen=True)
class ProductSpec:
    code: str
    name: str
    exchange: str
    multiplier: float
    price_tick: float
    option_tick: float
    option_style: str
    main_months: tuple
    strike_steps: tuple
    futures_margin_rate: float
    verified: bool = False

    def strike_step(self, price: float) -> float:
        for lo, hi, step in self.strike_steps:
            if price >= lo and (hi is None or price < hi):
                return step
        return self.strike_steps[-1][2]

    def round_strike(self, price: float) -> float:
        step = self.strike_step(price)
        return round(round(price / step) * step, 6)

    def strikes_around(self, price: float, n_each_side: int = 8) -> list[float]:
        """围绕当前价生成一串行权价（仅 demo / 回放使用；真实行权价以行情为准）。"""
        atm = self.round_strike(price)
        out = [atm]
        k = atm
        for _ in range(n_each_side):
            k = k + self.strike_step(k)
            out.append(round(k, 6))
        k = atm
        for _ in range(n_each_side):
            k = k - self.strike_step(k - 1e-9)
            if k <= 0:
                break
            out.append(round(k, 6))
        return sorted(set(out))


@dataclass
class ContractId:
    product: str
    exchange: str
    year: int
    month: int
    cp: Optional[str] = None        # 'C' / 'P' / None(期货)
    strike: Optional[float] = None
    raw: str = ""

    @property
    def is_option(self) -> bool:
        return self.cp is not None

    @property
    def delivery_month(self) -> dt.date:
        return dt.date(self.year, self.month, 1)

    def futures_code(self) -> str:
        return format_futures_code(self.product, self.year, self.month)


@lru_cache(maxsize=1)
def _load_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=None)
def get_product(code: str) -> ProductSpec:
    cfg = _load_config()
    key = code.upper()
    if key not in cfg["products"]:
        raise KeyError(f"未配置的品种: {code}，请在 config/products.yaml 中补充")
    p = cfg["products"][key]
    return ProductSpec(
        code=key,
        name=p["name"],
        exchange=p["exchange"],
        multiplier=float(p["multiplier"]),
        price_tick=float(p["price_tick"]),
        option_tick=float(p["option_tick"]),
        option_style=p.get("option_style", "american"),
        main_months=tuple(p.get("main_months", [])),
        strike_steps=tuple(tuple(x) for x in p["strike_steps"]),
        futures_margin_rate=float(p.get("futures_margin_rate", 0.1)),
        verified=bool(p.get("verified", False)),
    )


def list_products() -> list[str]:
    return list(_load_config()["products"].keys())


def exchange_info(exchange: str) -> dict:
    return _load_config()["exchanges"][exchange.upper()]


# ----------------------------------------------------------------------------
# 代码解析 / 生成
# ----------------------------------------------------------------------------
_RE_CZCE = re.compile(r"^([A-Z]{1,2})(\d{3})(?:([CP])(\d+(?:\.\d+)?))?$")
_RE_DCE = re.compile(r"^([a-zA-Z]{1,2})(\d{4})(?:-([CP])-(\d+(?:\.\d+)?))?$")
_RE_SHFE = re.compile(r"^([a-zA-Z]{1,2})(\d{4})(?:([CP])(\d+(?:\.\d+)?))?$")


def parse_contract(code: str, exchange: Optional[str] = None) -> ContractId:
    """解析合约代码。允许带交易所前缀，如 'CZCE.TA601C4800'（天勤风格）。"""
    raw = code.strip()
    if "." in raw:
        exchange, raw = raw.split(".", 1)
    body = raw.replace(" ", "")
    m = _RE_DCE.match(body)
    style = None
    if m and "-" in body:
        style = "dce"
    else:
        m = _RE_CZCE.match(body.upper())
        if m and len(m.group(2)) == 3:
            style = "czce"
        else:
            m = _RE_SHFE.match(body)
            style = "shfe" if m else None
    if not m:
        raise ValueError(f"无法解析合约代码: {code}")
    prod, ym, cp, strike = m.group(1).upper(), m.group(2), m.group(3), m.group(4)
    spec = get_product(prod)
    if style == "czce":
        year = 2020 + int(ym[0])  # 郑商所 1 位年份：按 2020 年代解析，2030 年后需调整
        # 处理十年循环：选择距今最近的年份
        now_y = dt.date.today().year
        cands = [y for y in range(now_y - 5, now_y + 6) if y % 10 == int(ym[0])]
        year = min(cands, key=lambda y: abs(y - now_y)) if cands else year
        month = int(ym[1:])
    else:
        year = 2000 + int(ym[:2])
        month = int(ym[2:])
    return ContractId(
        product=prod,
        exchange=exchange.upper() if exchange else spec.exchange,
        year=year,
        month=month,
        cp=cp.upper() if cp else None,
        strike=float(strike) if strike else None,
        raw=code,
    )


def format_futures_code(product: str, year: int, month: int) -> str:
    spec = get_product(product)
    style = exchange_info(spec.exchange)["option_code_style"]
    if style == "czce":
        return f"{spec.code}{year % 10}{month:02d}"
    if style == "dce":
        return f"{spec.code.lower()}{year % 100:02d}{month:02d}"
    return f"{spec.code.lower()}{year % 100:02d}{month:02d}"


def format_option_code(product: str, year: int, month: int, cp: str, strike: float) -> str:
    spec = get_product(product)
    style = exchange_info(spec.exchange)["option_code_style"]
    k = int(strike) if float(strike).is_integer() else strike
    cp = cp.upper()
    if style == "czce":
        return f"{spec.code}{year % 10}{month:02d}{cp}{k}"
    if style == "dce":
        return f"{spec.code.lower()}{year % 100:02d}{month:02d}-{cp}-{k}"
    return f"{spec.code.lower()}{year % 100:02d}{month:02d}{cp}{k}"


# ----------------------------------------------------------------------------
# 交易日历 / 到期日估算
# ----------------------------------------------------------------------------
_HOLIDAYS: set[dt.date] = set()


def set_holidays(days: Iterable[dt.date]) -> None:
    """注入交易所节假日（可用 akshare.tool_trade_date_hist_sina 生成）。不注入时只按周末估算。"""
    _HOLIDAYS.clear()
    _HOLIDAYS.update(days)


def is_trading_day(d: dt.date) -> bool:
    return d.weekday() < 5 and d not in _HOLIDAYS


def trading_days_in_month(year: int, month: int) -> list[dt.date]:
    d = dt.date(year, month, 1)
    out = []
    while d.month == month:
        if is_trading_day(d):
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def estimate_option_expiry(product: str, year: int, month: int) -> dt.date:
    """按交易所规则估算期权最后交易日（= 到期日）。真实到期日以行情 expire_datetime 为准。"""
    spec = get_product(product)
    rule = exchange_info(spec.exchange)["last_trade_rule"]
    py, pm = (year, month - 1) if month > 1 else (year - 1, 12)
    days = trading_days_in_month(py, pm)
    if rule == "czce":
        return days[2]
    if rule == "dce":
        return days[4]
    if rule == "shfe":
        return days[-5]
    raise ValueError(rule)


def default_underlying(product: str, asof: Optional[dt.date] = None, min_days: int = 35) -> str:
    """选择最近的主力月份合约：期权到期日距 asof 至少 min_days 天。用于 demo / 默认订阅。"""
    asof = asof or dt.date.today()
    spec = get_product(product)
    months = spec.main_months or tuple(range(1, 13))
    y, m = asof.year, asof.month
    for _ in range(30):
        if m in months:
            exp = estimate_option_expiry(product, y, m)
            if (exp - asof).days >= min_days:
                return format_futures_code(product, y, m)
        m += 1
        if m > 12:
            m, y = 1, y + 1
    raise RuntimeError("无法确定默认合约")


def year_fraction(start: dt.date | dt.datetime, end: dt.date | dt.datetime, basis: int = 365) -> float:
    if isinstance(start, dt.datetime):
        start = start.date()
    if isinstance(end, dt.datetime):
        end = end.date()
    return max((end - start).days, 0) / basis


def trading_days_between(start: dt.date, end: dt.date) -> int:
    n, d = 0, start
    while d < end:
        d += dt.timedelta(days=1)
        if is_trading_day(d):
            n += 1
    return n
