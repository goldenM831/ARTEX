# optdesk —— 国内商品期权策略工具（规划 + 可运行 MVP）

面向石化现货贸易 / 产业客户的期权策略支持工具：**接国内期货期权实时行情 → 波动监测 + 技术面 → 结合库存/采购/方向意图生成可执行的策略组合（备兑、领口、保护性看跌、牛/熊市价差、卖出看跌降采购成本）→ 输出理由、风险指标、保证金、委托清单**。

本目录是一个独立项目（与仓库其余部分无关），可直接拷出单独建仓。

```
cd options-desk
pip install -r requirements.txt
python -m optdesk demo --product TA --role inventory --tons 5000 --cost 4700     # 合成行情跑通全流程
python -m optdesk chain --product PP --price 7200                                # 看期权链 IV / Greeks / 流动性
streamlit run optdesk/dashboard.py                                               # 交易员看板
python -m pytest tests                                                           # 回归测试
```

---

## 1. 目标与边界

| 业务场景 | 对应策略 | 工具要回答的问题 |
| --- | --- | --- |
| 库存保值 / 增强（持有 PTA、MEG、PP、甲醇等现货） | 备兑看涨、领口、保护性看跌、熊市看跌价差 | 卖哪个行权价？年化多少？封顶价在哪？下行缓冲多少？保证金占多少？ |
| 采购端（未来有买货需求） | 卖出看跌（愿意低位接货）、牛市看涨价差（锁采购上限） | 在哪个价位接货合算？权利金能降多少采购成本？ |
| 方向交易 / 自营 | 牛市看涨 / 看跌价差、熊市看跌 / 看涨价差 | 借方还是贷方价差？行权价怎么选？盈亏比、盈利概率、Greeks 暴露？ |
| 波动率判断（所有场景的前置） | IV vs HV、IV Rank、波动率锥、ATR / 布林带宽状态 | 现在期权贵还是便宜？该做买方还是卖方？ |

**边界（第一阶段明确不做）**：全自动下单、场外期权 / 含权贸易定价、跨品种套利、做市报价。这些在路线图第 4 阶段。

---

## 2. 总体架构

```
┌──────────────┐  ┌──────────────┐  ┌───────────────┐
│ CTP 行情前置  │  │ 天勤 TqSdk    │  │ AkShare / Wind │   ← 数据源层（可插拔，同一接口 MarketFeed）
│ (生产主源)    │  │ (实时备源/MVP) │  │ (历史/日频)    │
└──────┬───────┘  └──────┬───────┘  └──────┬────────┘
       └────────┬────────┴─────────────────┘
        feeds/  │  统一快照：Quote / OptionQuote / OptionChain / K 线 DataFrame
                ▼
┌─────────────────────────────────────────────────────┐
│ 分析层                                                │
│  pricing.py  Black-76 定价 / Greeks / IV / 美式二叉树校验 │
│  vol.py      HV(多估计量) / 波动率锥 / IV Rank / 状态识别  │
│  ta.py       MA/MACD/RSI/BOLL/ADX / 支撑阻力 / 方向打分    │
│  chain.py    期权链 IV 曲面 / 偏斜 / 流动性过滤 / 数据校验  │
└──────────────────────────┬──────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────┐
│ 策略层                                                │
│  strategies.py  腿/组合 → 到期损益、极值、盈亏平衡、概率、   │
│                 Greeks、情景损益、交易所保证金估算、委托单  │
│  recommender.py 业务意图(Intent) + 行情状态 → 候选生成 →    │
│                 打分排序 → Report(文本 / JSON)             │
└──────────────────────────┬──────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────┐
│ 呈现 / 落地层                                          │
│  app.py       CLI：demo / chain / replay / live(+webhook)│
│  dashboard.py Streamlit 看板                            │
│  signals/*.jsonl  每次刷新的建议留痕（复盘用）              │
└─────────────────────────────────────────────────────┘
```

设计原则：

1. **规则配置化**：品种乘数、行权价间距、到期日规则、保证金率在 `config/products.yaml`；交易所改规则只改配置。
2. **数据源可插拔**：策略层只认 `OptionChain` / `DataFrame`，CTP、天勤、回放三种源切换不改一行策略代码。
3. **可回放、可回归**：合成源 + CSV 回放源 + 单测，策略逻辑每次迭代都能离线验证"过去每天会给什么建议"。
4. **建议必须可执行**：每条建议带委托清单（合约 / 买卖 / 开平 / 手数 / 参考价），参考价按买方吃卖一、卖方吃买一的保守口径。
5. **先半自动再自动**：建议 → 人确认 → 交易终端下单；留痕后用真实成交回头校准模型。

---

## 3. 行情接入方案（国内实际可用）

| 方案 | 适用 | 优点 | 限制 | 本项目支持 |
| --- | --- | --- | --- | --- |
| **CTP**（`openctp-ctp`） | 生产主源 | 期货公司标配、免费、含全部商品期权、500ms 快照、可录 tick 回放 | 无 K 线、无期权合约列表查询（需交易接口 `ReqQryInstrument` 或每日导入）；交易接口需穿透式监管 AppID/AuthCode | `feeds/ctp_feed.py` |
| **天勤 TqSdk** | MVP 实时源 / 备源 | pip 即用、免费实时、`query_options` 直接拿期权链、自带 K 线与真实到期日 | 历史数据 / 回测需专业版；API 单线程 | `feeds/tqsdk_feed.py`（已实现，需账号验证） |
| **AkShare** | 历史日线 / 交易所期权日行情 / 交易日历 | 免费 | 接口随上游改版漂移，需落盘缓存 | `feeds/akshare_feed.py` |
| **Wind / iFinD / Choice** | 若公司已购 | IV 历史、持仓结构、现货价格 | 付费、需终端 | 预留接口位（实现 `MarketFeed` 即可） |
| 现货价格（卓创 / 隆众 / 公司内部） | 基差与现货成本 | — | 多为人工 / Excel | 通过 `Intent.spot_cost` 输入；后续做 CSV/Excel 导入 |

**落地顺序**：先用天勤跑通实时（一天内可见效果）→ 向期货公司申请 CTP 行情前置 + SimNow 测试 → CTP 为主、天勤为备，两源价格互校。

---

## 4. 核心模块说明

### 4.1 定价（`pricing.py`）
- Black-76：国内商品期权 IV 口径的事实标准，不依赖 scipy。
- Greeks：delta / gamma / vega(每 1 vol 点) / theta(每自然日) / rho。
- `implied_vol`：Newton + 二分兜底，低于内在价值返回 `None`（脏数据过滤）。
- `binomial_american_future`：美式校验（期货期权提前行权溢价很小，测试里断言 < 2%）。

### 4.2 波动监测（`vol.py`）
- HV：收盘 / Parkinson / Garman-Klass / Yang-Zhang 四种估计量，20/60 日。
- 波动率锥 `vol_cone`、`iv_rank` / `iv_percentile`。
- `detect_vol_regime` → `VolSignal`：
  - 状态：`squeeze`（布林收口 + ATR 低位）/ `normal` / `expansion` / `spike`
  - 倾向：`seller`（IV-HV > 4 点且 IV Rank 高）/ `buyer`（IV 低于 HV 3 点或 IV Rank < 25%）/ `neutral`
  - 阈值全部是常量，回测后调。

### 4.3 技术面（`ta.py`）
- MA5/20/60 排列、MACD 柱变化、RSI 区间、ADX 趋势强度、布林 %B。
- 投票合成 `score ∈ [-1,1]` 与标签（看涨/偏多/中性/偏空/看跌），附可读理由。
- `swing_levels` 摆动高低点 → 阻力 / 支撑，直接用作备兑卖 call、领口买 put、价差卖腿的行权价依据（离现价 < 1.5% 时自动放弃改用 delta 选档）。

### 4.4 期权链（`chain.py`）
- 统一 `OptionQuote`（买卖价、量、持仓、到期、IV、Greeks）。
- `liquid()`：点差比例 / 绝对点差 / 持仓量三重门槛，**不推荐成交不了的合约**。
- `sanity_issues()`：行情陈旧、买卖倒挂、Put-Call Parity 偏离 → 报告里直接显示"数据告警"。
- `atm_iv()`、`skew_25d()`（25d 风险逆转）。

### 4.5 策略（`strategies.py`）
- `Leg` / `Strategy`：到期损益向量化计算；`evaluate()` 给出：净权利金（元/吨、整单）、最大盈亏（含"无上限/无下限"判定）、盈亏平衡点、对数正态盈利概率与期望损益、组合 Greeks（Δ 折合吨、Vega 元/点、Theta 元/日）、±1σ/±2σ 到期情景、保证金估算、年化（库存口径或保证金口径）、下行缓冲、封顶价。
- `estimate_margin()`：按交易所规则近似——期权空头单腿公式、**期货多头 + 卖 call 备兑组合只收期货保证金**、借方价差 0、贷方价差按行权价差；**现货备兑的期权空头要缴保证金**（现货企业与期货备兑的核心差异，报告中已明确提示）。
- 构建器：`covered_call`（现货 / 期货两种底仓）、`collar`（可自动找零成本 call）、`protective_put`、`cash_secured_put`、`bull_call_spread`、`bull_put_spread`、`bear_put_spread`、`bear_call_spread`。
- `order_tickets()`：委托清单。

### 4.6 推荐引擎（`recommender.py`）
`Intent`（角色 / 吨数 / 现货成本 / 观点 / 目标价 / 保底价 / 权利金预算 / 保证金预算）→ 候选生成 → 打分：方向匹配、IV 贵贱与借贷方匹配、盈利概率、盈亏比、期望损益、库存角色优先真正对冲库存的结构、流动性惩罚、预算约束。输出 `Report.to_text()` / `to_dict()`。

> 现货腿按当前期货价计价，损益代表"相对什么都不做的增量"；现货成本只用于说明文字（锁定利润多少）。**基差风险（现货 vs 期货）暂不建模**，报告中按品种基差经验值人工考虑，第 3 阶段纳入。

---

## 5. 实施路线（按里程碑，不按日历）

### 阶段 0：规则与资源准备
- 确认覆盖品种清单（PTA/MEG/PP/L/PVC/甲醇/苯乙烯/LPG/原油/沥青/橡胶…），逐项核对 `products.yaml` 并把 `verified` 置为 true。
- 申请：天勤账号（立刻）、期货公司 CTP 行情前置 + SimNow、公司是否有 Wind。
- 注入真实交易日历（`contracts.set_holidays`，可由 AkShare `tool_trade_date_hist_sina` 生成）。

**验收**：`python -m optdesk demo` 在所有目标品种上输出合理建议；`pytest` 全绿。

### 阶段 1：实时跑通（天勤）
- `python -m optdesk live --feed tqsdk --underlying CZCE.TA701 --role inventory --tons 5000 --cost 4700 --interval 30 --webhook <企业微信机器人>`
- 用交易所公布 IV / 期货公司软件对照 `chain` 的 IV 与 Greeks，误差 < 0.5 vol 点。
- 每次刷新写入 `signals/*.jsonl` 留痕。

**验收**：盘中连续运行一个交易日无异常；首选策略或波动状态切换时企业微信收到推送。

### 阶段 2：历史数据与回归
- AkShare 拉取主力连续日线 + 期权日行情落盘（parquet），构建 IV 历史 → `iv_rank` 生效。
- `python -m optdesk replay` 对过去 1–2 年逐日跑建议，统计：各状态下建议策略的到期胜率 / 平均损益 / 最大回撤；据此调 `vol.py`、`recommender._score` 的阈值与权重。
- 增加品种维度参数（PTA 与原油的流动性门槛、波动阈值不同）。

**验收**：形成一份"策略回测报告"（胜率、盈亏比、按波动状态分组），阈值有数据支撑。

### 阶段 3：组合与风控
- 持仓簿：现货 + 期货 + 期权统一录入（CSV/Excel 导入或 CTP 交易接口查询），组合 Greeks、到期情景、压力测试（±5%/±10%、IV ±5 点）。
- 基差建模：现货价格导入，领口 / 备兑的"实际保底 / 封顶"按基差修正。
- 保证金精确化：接期货公司保证金参数；组合保证金申报提醒。
- 到期与移仓管理：DTE < 7 天告警、滚动建议（roll up / roll out）。
- 信号复盘：`signals/` 与真实成交对齐，统计"建议 vs 实际"。

### 阶段 4：半自动执行与拓展
- CTP 交易接口（openctp）：建议 → 一键生成委托 → 人工确认后发送；穿透式监管认证。
- 含权贸易 / 场外期权：累购累沽（Accumulator）、远期含权合同定价与对冲参数，用于和场外报价比价。
- 波动率策略：跨式 / 宽跨式 / 蝶式 / 日历价差；IV 曲面建模（SVI）。
- Web 服务化：FastAPI + 前端，多人使用与权限。

---

## 6. 可靠性设计（接地气的工程细节）

- **双源互校**：CTP 与天勤同一合约价差超过 2 跳告警。
- **行情陈旧检测**：`Quote.is_stale`，非交易时段自动降级为"参考建议"。
- **IV 求解失败 / 平价偏离**：对应合约不进入候选，报告列出数据告警。
- **流动性门槛**：默认点差 ≤ 15% 或 ≤ 20 跳、持仓 ≥ 50 手，按品种配置。
- **到期保护**：DTE < 7 天提示换月。
- **留痕**：每次刷新 JSON 落盘；回放可复现任意一天的建议。
- **单测覆盖**：定价（平价、IV 往返、Greeks 边界、美式溢价）、策略（价差极值、备兑封顶、领口区间、保证金）、波动 / 技术（HV 收敛、状态识别、趋势打分）、端到端推荐。

---

## 7. 部署与合规

- 运行环境：公司 Windows / Linux 任意一台能访问期货公司行情前置的机器；Python 3.11+。
- 账号 / 密码走环境变量（`TQ_USER` / `TQ_PASSWORD`、CTP 账号），不进代码库。
- 推送：企业微信 / 钉钉机器人 webhook（已实现）。
- 合规：期权空头保证金与追保、交易所持仓限额、套保额度申请（产业客户可申请套保持仓）、交易接口穿透式监管认证。
- 本工具输出为**决策支持**，不构成自动交易；所有参数（保证金率、行权价间距、到期规则）上线前须对照交易所最新规则核对。

---

## 8. 目录

```
options-desk/
├── optdesk/
│   ├── config/products.yaml   品种 / 交易所规则
│   ├── contracts.py           代码解析、到期估算、默认合约
│   ├── pricing.py             Black-76 / Greeks / IV / 美式二叉树
│   ├── vol.py                 HV / 波动率锥 / IV Rank / 状态识别
│   ├── ta.py                  技术指标 / 支撑阻力 / 方向打分
│   ├── chain.py               期权链、流动性、数据校验
│   ├── strategies.py          策略构建、评估、保证金、委托单
│   ├── recommender.py         意图 → 候选 → 打分 → 报告
│   ├── app.py                 CLI（demo / chain / replay / live）
│   ├── dashboard.py           Streamlit 看板
│   └── feeds/                 base / replay(合成+CSV) / tqsdk / ctp / akshare
├── tests/
├── requirements.txt
└── pyproject.toml
```

## 9. 下一步最小动作清单

1. 注册天勤账号，`pip install tqsdk`，跑 `live --feed tqsdk`，对照行情软件验证 IV。
2. 把公司实际覆盖品种填进 `products.yaml` 并核对规则。
3. 用 AkShare 落盘 2 年日线 + 期权日行情，跑 `replay`，出第一份回测报告。
4. 向期货公司申请 CTP 行情前置与 SimNow，接 `CtpFeed`。
