# 自主投资系统：架构审查、开源模型交叉研究与实施建议

研究日期：2026-09-10

范围：当前 `AI_Investor_Test` 代码库、用户提供的 41 节设计稿、相关开源量化/LLM 项目及基础论文

状态：仅设计与研究；没有下载或运行外部交易机器人，没有提交任何订单

## 执行结论

这份设计稿的主方向是正确的，而且明显优于“让 LLM 看 60 只股票并直接挑一只”的常见方案。最值得保留的五个原则是：

1. 将 LLM 限定为证据研究组件，而不是最终交易权威。
2. 将预测、组合构建、风险审批和订单执行拆开。
3. 要求点时（point-in-time）数据、预测留档和事后校准。
4. 让现金、持仓和新候选共同竞争下一单位资本。
5. 用目标组合驱动确定性订单计划，而不是让语言模型直接输出订单。

但在投入真钱前，设计稿需要四项关键修正：

- `$1,000 → $100,000` 必须被定义为带期限的到达目标，而不能与“最大化长期期望对数财富”混为一个目标。
- Kelly 不能直接使用点估计；应采用收缩后的、分布稳健或受回撤约束的 Kelly。
- 单一 `$1,000` 账户产生的实盘样本远远不足以训练和校准这些模型；主要统计证据必须来自点时历史横截面数据和影子决策。
- “不设任意仓位上限”是投资哲学，不等于现在就删除现有 20% 硬上限。硬风险政策在修改前应继续保持，且模型永远不能自行修改它。

因此，推荐路线不是导入某个 GitHub 机器人，而是：以 **Qlib 的模块边界**作参考、以 **skfolio/相关论文的验证方法**作参考、以 **Riskfolio-Lib 或透明的 CVXPY 模型**作组合研究工具、以 **LEAN/NautilusTrader 的执行一致性原则**作工程参考，再从 TradingAgents/FinMem 中只抽取结构化反方审查和记忆分类。[^5][^7][^9][^18][^19][^14][^15]

明天可以做完整的市场日端到端测试，但按当前代码状态必须是 `DRY_RUN` 或 `SHADOW`，不能是 `LIVE`。现在还没有收益预测、组合优化、风险审批、订单幂等和成交对账模块；把配置改成 live 不能弥补这些缺失。

## 1. 如何正确表达 `$1,000 → $100,000`

100 倍是清楚的愿景，但它本身还不是一个可优化的数学目标，因为没有期限和可接受的破产概率。不同期限要求的复合年增长率如下：

| 到达期限 | 所需 CAGR |
|---:|---:|
| 3 年 | 364.2% |
| 5 年 | 151.2% |
| 7 年 | 93.1% |
| 10 年 | 58.5% |
| 15 年 | 35.9% |
| 20 年 | 25.9% |
| 30 年 | 16.6% |

在不使用保证金、期权、做空或杠杆 ETF 的前提下，5–10 年完成 100 倍需要极其罕见且持续的超额收益，无法被模型诚实地视为高概率基准。市场判断可以帮助捕捉非对称机会，但不能把低概率机会制造成高概率事件。

建议采用双层目标：

### 投资引擎的主目标

在明确的回撤/永久损失约束下，最大化经过估计误差、交易成本和模型不确定性调整后的期望对数财富：

\[
\max_w \; \mathbb{E}[\log(1+r^\top w)] - C(w-w_0) - U(w)
\]

其中 `C` 是交易与执行成本，`U` 是参数不确定性惩罚，现金是组合中的显式资产。Kelly 的理论吸引力在于它对应长期几何增长，但原始 Kelly 对概率/收益估计误差极其敏感。风险约束 Kelly 研究表明，可以在增长目标之外显式约束财富跌破某一水平的概率。[^1][^2]

### 战略层的到达目标

单独跟踪：

\[
P(\tau_{100000} \leq T), \quad
P(\min_{t\leq T} W_t < W_{floor})
\]

也就是“在期限 `T` 内到达 10 万美元的概率”和“期间跌破生存底线的概率”。这个指标用于评估路径，而不是在落后时强迫系统加仓、提高频率或追涨。

在用户没有指定 `T` 与最低可接受财富之前，系统可以保留 100 倍愿景，但不能声称已经针对它完成最优设计。

## 2. 对原设计稿的专业评价

### 设计正确的地方

- **经济目标合理**：期望几何增长比单期 PnL、胜率或固定年收益更接近长期财富目标。
- **数据先于模型**：对 `source / as_of / ingestion_time / market_time / revision` 的要求是实盘量化的核心。
- **宇宙与 alpha 分离**：被观察不等于值得买入，避免将预筛选误当作预测。
- **多信号、多周期、保留分歧**：避免一个任意加权的 0–100 分掩盖模型冲突。
- **预测分布而非标签**：期望收益、概率、尾部损失、区间和半衰期才是组合输入。
- **当前持仓与现金参与机会成本比较**：这是比“买入筛选器”成熟得多的投资决策方式。
- **目标组合与执行分离**：执行应确定、可重试、可对账。
- **廉价监控与昂贵推理解耦**：符合 `$1,000` 账户的成本尺度。
- **预测先冻结，结果后到达**：这是校准、归因和防止自我欺骗的必要条件。

### 需要修正或补足的地方

1. **目标函数冲突**：最大化长期期望对数财富，不等于最大化某一期限前到达 100 倍的概率。必须分层表达。
2. **浓度逻辑不完整**：高置信度不应直接等于高仓位。仓位还必须取决于置信度是否经过独立样本校准、尾部是否可识别、模型是否共同依赖同一数据。
3. **“独立 alpha”可能是假独立**：动量、突破、相对强度和板块轮动经常共享同一价格趋势风险。需要估计信号残差相关性，而不是按模块名称认为它们独立。
4. **制度条件元模型容易维度爆炸**：`regime × signal × sector × horizon × volatility` 在有限样本中会迅速过拟合。权重必须向全局基准强烈收缩。
5. **校准假设不足**：普通 conformal prediction 依赖交换性，而金融时间序列非平稳。MAPIE 可以帮助实现区间和覆盖率度量，但必须用滚动/自适应方案并持续检查覆盖率，不能照搬静态教程。[^11]
6. **LLM 置信度不能直接数值化**：语言模型说“80% confidence”没有概率含义。只能把结构化判断作为特征，再用事后样本学习其映射。
7. **事件预期数据是关键瓶颈**：`actual vs expected vs priced-in` 需要历史分析师预期、修订时间和公告精确时间。免费的当前快照无法可靠重建历史预期。
8. **实盘账户样本不足**：即使每天产生一次交易，一年也只有约 250 个高度相关观察。应同时保存所有候选与拒绝候选的影子预测，以扩充横截面校准样本。
9. **研究成本的单位经济学**：每月 `$10` 相当于初始本金的 1%；每月 `$99` 的数据订阅相当于初始本金的 9.9%。对 `$1,000` 账户，基础设施成本必须先通过影子研究证明有足够边际价值。

## 3. GitHub 开源项目交叉分析

以下结论基于项目说明、关键源码/接口和论文，而不是展示出来的回测收益。没有任何仓库应被直接接入实盘。

| 项目 | 与设计稿的相似处 | 值得抽取 | 主要风险/不采纳部分 | 结论 |
|---|---|---|---|---|
| [Microsoft Qlib](https://github.com/microsoft/qlib) | 数据→特征→预测→策略→回测→在线服务的松耦合全链路 | 数据处理器、实验记录、预测与组合策略分离、离线/在线接口 | 自带数据和模型结果不能外推到美国实盘；表达式 `Ref(..., N<0)` 支持未来引用，若没有特征/标签命名空间隔离可造成泄漏[^6] | **架构参考，暂不整体导入** |
| [skfolio](https://github.com/skfolio/skfolio) | 组合优化、风险度量、成本、约束、WalkForward、CombinatorialPurgedCV、不确定集 | 时间序列验证、协方差/均值收缩、bootstrap 不确定集、成本与换手约束 | 方法选择空间很大，本身也可形成“优化器动物园”；不能用验证集反复挑赢家 | **优先评估的研究依赖** |
| [Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib) | Kelly/log-mean-risk、CVaR/回撤、稳健组合和风险预算 | 场景式 Kelly、尾部/回撤风险、稳健优化的参考实现 | 功能极广，默认范例不等于适合本账户；需要用自己的收益分布与成本 | **组合层 challenger** |
| [cvxportfolio](https://github.com/cvxgrp/cvxportfolio) | 预测、协方差、交易成本、持仓与多期目标统一进优化器 | 目标组合、现金、交易成本、风险预测误差、滚动仿真 | GPL-3.0 需单独做许可审查；公开数据默认值不是本系统的点时数据或真实成交模型[^8] | **优先借鉴数学与接口；依赖待审** |
| [PyPortfolioOpt](https://github.com/PyPortfolio/PyPortfolioOpt) | 模块化预期收益、风险模型、约束、Black–Litterman、HRP | 更轻量的收缩协方差与基准优化器 | 经典均值—方差对均值估计敏感，不能把 `max_sharpe` 当增长最优[^10] | **适合 MVP 基线** |
| [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) | 模型无关预测区间和风险控制 | 区间覆盖率、rolling/adaptive conformal、校准诊断 | 非平稳和小样本会破坏名义覆盖率；近期历史也出现过时间序列泄漏修正 | **校准工具，不是 alpha** |
| [River](https://github.com/online-ml/river) | 在线学习、渐进验证、漂移检测 | 数据/预测分布漂移报警，渐进指标 | 在线更新容易追逐噪声；漂移报警不能自动触发模型替换[^12] | **仅作监测与 challenger** |
| [OpenBB](https://github.com/OpenBB-finance/OpenBB) | 多供应商数据统一接口，可供 Python/API/MCP 使用 | Provider adapter、统一结果对象、可替换数据源 | 聚合层不保证上游准确性，也不自动提供点时语义；供应商许可不同 | **借鉴 data-provider 抽象** |
| [TradingAgents](https://github.com/TauricResearch/TradingAgents) | 基本面/新闻/技术研究、bull/bear 辩论、风险经理 | 结构化研究状态、一次反方审查、可恢复工作流 | 源码图包含多轮 bull/bear 和 aggressive/neutral/conservative LLM 讨论，成本高且观点并不统计独立；最终仍偏交易提案[^14] | **只取一次 red-team 思想** |
| [FinMem](https://github.com/NathanDDDD/finmem-llm-stocktrading) | 分层金融记忆、决策历史、反思 | 区分短期事件、长期论文点、持仓 thesis 和记忆有效期 | 单股/研究型实验，不证明实盘可校准收益；自我反思容易从噪声编故事 | **只取记忆分类** |
| [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) | 金融情绪、标题、实体与关系抽取 | 本地或便宜模型做新闻分类的 challenger | NLP benchmark 不是收益预测；情绪正确也不等于超出市场预期[^16] | **研究特征 challenger** |
| [FinRL](https://github.com/AI4Finance-Foundation/FinRL) | RL 环境、成本、动作与资产价值状态 | 仿真环境接口和基准实验 | 示例环境把动作乘 `hmax` 后转整数，并可按 turbulence 阈值全部卖出；这类简化会把模拟器假设学成策略[^17] | **MVP 不用 RL** |
| [LEAN](https://github.com/QuantConnect/Lean) | 事件驱动、模块化、回测与 live、数据/券商插件 | 交易时钟、事件、订单生命周期、适配器边界 | 体系庞大，Robinhood MCP 没有现成受审适配；嵌入本项目会喧宾夺主 | **执行语义参考** |
| [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) | 确定性事件驱动，同一策略代码用于仿真/live，broker adapter | 时间模型、状态缓存、消息总线、对账和 adapter 设计 | 当前没有 Robinhood 股票 MCP adapter；Rust/Python 体系对 15 分钟、60 标的系统过重。项目也明确说明 live 仍有仿真无法复现的差异[^19] | **工程模式参考，不集成** |
| [ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) | 多“投资者人格”、risk/portfolio manager、回测入口 | 可读的 agent 编排示例 | 官方明确称其为 proof of concept、教育用途且不实际交易；人格投票不是可校准 alpha[^20] | **拒绝作为投资引擎** |
| [R&D-Agent-Quant](https://github.com/microsoft/RD-Agent) | 自动提出并评估量化研究想法 | 将实验代码、假设和结果做成 challenger 流水线 | 自主生成研究会放大多重检验；不得自动晋升生产策略 | **后期研究助手，永不直连 live** |

### 最相似的组合

没有单个仓库完整实现设计稿。最接近的组合是：

```text
Qlib 模块边界与实验记录
    + skfolio 的时间序列验证和不确定集
    + Riskfolio/CVXPY 的受约束增长型组合
    + LEAN/Nautilus 的确定性执行语义
    + TradingAgents 的一次性反方审查
    + FinMem 的有限期 thesis memory
```

这与设计稿的最大相似点，是不让 LLM 同时承担数据、预测、仓位和执行。最大改进点，是把模型验证、点时数据和模型晋升做成一等公民，而不是写在 prompt 中。

## 4. 推荐的修正版系统结构

```text
公共/券商数据
  → 双时间数据账本（event_time + known_at）
  → 数据质量门（缺失/陈旧/冲突即 abstain）
  → 观察宇宙（与 alpha 完全分离）
  → 特征快照（版本化、只向后引用）
  → 独立 alpha 家族（各自周期与失败模式）
  → 预测分布 + 区间 + 校准状态
  → 量化候选 + 所有持仓
  → LLM 证据研究（事实、变化、反证、来源）
  → 一次 adversarial review
  → 收缩后的 meta synthesis
  → 场景/稳健 fractional Kelly + 现金
  → 目标组合
  → 不可绕过的硬风险审批
  → 确定性订单计划
  → DRY/SHADOW/LIVE broker adapter
  → 成交对账、预测到期、归因与 challenger 更新
```

### 必须增加的四道边界

#### 4.1 双时间数据边界

每条记录至少有：

- `event_time`：经济事件发生时间；
- `published_at`：来源公布时间；
- `known_at`：系统最早可使用时间；
- `ingested_at`：系统收到时间；
- `revision_id` / `is_restatement`；
- `source`、`source_record_id`、`payload_hash`；
- `quality_status`、`freshness_seconds`。

回测查询必须使用 `known_at <= decision_time`，而不是只按财报期末日或新闻日期筛选。

#### 4.2 特征—标签隔离边界

Qlib 的表达式系统允许负数 `Ref` 指向未来，这对生成标签很方便，但如果和特征共用表达式权限会泄漏。[^6] 本项目应强制：

- `features/` 只能调用 `lag >= 0`；
- `labels/` 可以读取未来，但永远不进入实时依赖图；
- 每个 feature 保存 `max_lookback`、`availability_lag` 和 lineage；
- CI 构造人为未来冲击，确认实时特征不改变。

#### 4.3 预测—研究边界

LLM 不直接输出“未来涨 12%”。它输出：新事实、预期差、方向、持续时间、反证、信息质量和来源。一个在历史影子样本上训练并校准的映射器，才可以把这些字段转换为预测分布的增量；在样本不足时，LLM 增量应高度收缩甚至只记录、不参与仓位。

#### 4.4 投资风险—工程安全边界

投资层可以提出高度集中的目标组合；硬风险层仍可拒绝它。当前 `max_position_fraction = 0.20` 等限制是用户先前确认的安全政策，除非用户显式修改，否则不应因新设计稿的“非任意集中”原则而自动删除。

## 5. 当前代码库 Gap Analysis（A–J）

### A. 已经存在

- 官方 Robinhood Agentic MCP OAuth/PKCE、刷新令牌和 macOS Keychain 凭据存储。
- 只读 MCP 工具 allowlist；监控代码不能调用下单、取消、期权或 crypto 工具。
- 精确识别唯一 active、`agentic_allowed` 账户并在日志中掩码。
- 12 ETF + 33 core + 10 event + 最多 5 持仓保留位的 60 标的观察框架。
- 美东常规交易时段 15 分钟轮询、3×20 报价批次、调用预算、缓存和重试。
- 价格变化触发器、本地 JSONL 日志和零 LLM 的廉价监控路径。
- `DRY_RUN` 硬锁；当前配置无法通过环境变量开启 live，代码中也没有下单实现。
- OpenAI API 的只读账户快照、token/cost 记录和单次成本上限。

这些工作属于可靠的“传感器和安全外壳”，不是投资决策引擎。

### B. 尚不存在

- 点时历史数据仓库、公司行动、退市证券和数据修订账本；
- 特征 registry、标签生成、历史重放和数据 lineage；
- 市场状态、alpha engines、预测分布、校准与 meta model；
- LLM 基本面/事件研究 schema、来源验证、red team；
- 协方差/尾部场景、机会成本和目标组合优化器；
- 持仓 thesis state、卖出模型和候选 counterfactual；
- 独立、不可绕过且真正被调用的 risk engine；
- broker 抽象、order planner、review、idempotency、partial fill、reconciliation；
- shadow portfolio/fill simulator；
- 月度 LLM 预算的持久化执行（配置存在，但没有总账）；
- Delta 表、Databricks workflow/secrets 和云端无人值守运行；
- strategy/model/prompt registry 与 champion/challenger promotion gate。

### C. 需要重构

1. `universe.py::_rank_core` 的分数包含 15% 当日绝对涨跌幅。这将预期收益信息混入 core membership，直接违背“宇宙构建与 alpha 分离”。应移到 event/alpha 层。
2. Core 目前每日重建；应改成每周正式调仓、每日资格/异常检查，并加入进入/退出 hysteresis。
3. `quotes_are_fresh` 只检查一批报价中最新的时间戳。只要 20 条中有一条新鲜，其他陈旧报价也可能通过。应逐条验证，并在全部三批完成后再次验证一致性。
4. weekday + 固定 9:30–16:00 不是完整交易日历。应使用 NYSE calendar 处理假日、提前收盘和特殊停市；现有 stale guard 只能兜底。
5. universe cache 只有日期和 `cache_version`，还应包含策略版本、配置 hash、数据源、生成时间和有效期。
6. event bucket 目前主要是价格变化 × 相对成交量，不包含 earnings/news/revision 事件；应先称为 `price_volume_event`，避免语义夸大。
7. `settings.toml` 中的 risk 配置目前只是静态文本，没有执行路径；要变成纯函数审批和不可变审计记录。
8. Robinhood 客户端被直接引用；应放到 `BrokerAdapter` 与 `MarketDataProvider` 后面，研究数据和执行数据不能共用隐式接口。
9. 运行日志目前是本地 JSONL；代码与 schema 可以进 GitHub，但真实账户、OAuth、订单、持仓和高频行情日志不能进入公开仓库。

### D. 当前较弱的假设

- 默认 Robinhood scanner 的候选集合、排序和字段覆盖足够稳定，但尚未验证分页、停牌、证券类型、退市与 corporate action 语义。
- 用当前报价和当日变化触发深度研究，但没有按资产波动率、行业或事件类型做标准化；2% 对不同股票不是同等异常。
- sector 缺失会进入 `unknown`，可能让 sector cap 产生意外排除。
- 当前报价可用于实时监控，不代表 Robinhood 历史 fundamentals 能支持无泄漏回测。
- 60 个观察标的是计算/成本预算，不是完整机会集合；发现层必须在更宽的可交易集合上先用便宜规则扫描。
- 单次 LLM 输出成本在调用后才核算，不能防止一次异常长输入先发生费用；需要调用前估算、响应硬上限和月度 ledger 三层控制。

### E. 所需数据源

| 数据 | DRY MVP | 严格研究/晋升前要求 | 说明 |
|---|---|---|---|
| 实时报价、账户、持仓、订单、tradability | Robinhood MCP | Robinhood MCP + 对账快照 | 官方支持 quotes、historicals、fundamentals、earnings、订单 review/place 等工具，但仍需自行做时间与质量验证[^21] |
| OHLCV、拆股、分红、退市 | 可先用 Robinhood 验证流程 | 有 point-in-time corporate actions 与 delisted history 的供应商 | 不能用今天仍存活的股票列表回测过去 |
| 财务报表与公告时间 | SEC EDGAR submissions/XBRL | SEC 原文 + 规范化 PIT vendor | SEC API 免费、实时更新 submissions/XBRL；XBRL 数值仍需和原始 filing 交叉检查[^22] |
| 分析师预期与修订 | MVP 暂不作为 alpha | 带 forecast vintage 的付费数据 | `actual vs expected` 不能用当前网页回填历史 |
| Earnings calendar/results | Robinhood 作实时触发 | 带公告精确时间和历史预期版本 | 区分盘前/盘后和何时可交易 |
| 宏观数据 | FRED | ALFRED vintage | ALFRED 保留原始发布与后续修订，适合无修订泄漏回测[^23] |
| 新闻 | SEC/公司 IR/官方公告优先 | 许可清晰、含 published_at/updated_at 的供应商 | 去重、修订、时区和来源信誉必须显式化 |
| 利率/信用/波动/跨资产 | 官方指数和公开源 | 可复现的历史供应商 | 作为 regime context，不占 60 个股票位 |

OpenBB 可以统一多个 provider 的调用接口，但它是路由层而不是数据真相层；每条记录仍要保留原 provider 与许可。[^13]

### F. 各层推荐数学模型

| 层 | 第一版 | 后续 challenger | 不建议起步使用 |
|---|---|---|---|
| 数据质量 | schema、范围、单调时间、交叉源差异、逐条 freshness | 异常检测 + 数据漂移 | LLM 补数字 |
| Regime | 透明的连续状态向量：趋势、宽度、波动、相关、利率/信用；EWMA + shrinkage | Bayesian/HMM probabilities；ruptures/River 仅作变点报警 | 单一 bull/neutral/bear 标签；按近期 PnL 切换 |
| Trend/RS | 行业/指数残差动量，5/20/60 日，多期限 | 正则化 GBDT/linear panel model | 直接套 Alpha158 后挑最好回测 |
| Breakout/event | 波动率标准化 gap、range breakout、relative volume、event tag | 条件事件模型/生存模型 | “涨跌超过 2% 就买/卖” |
| Mean reversion | 1/5 日残差、波动/流动性/事件条件化 | 状态空间残差模型 | 无条件抄底 |
| Fundamental | 财务增速、利润/现金流质量、行业相对值，按 `known_at` | 分层 Bayesian panel / monotonic GBDT | 用今日财务数据库回填历史 |
| Revisions/PEAD | 数据齐备后再启用；surprise 与 revision vintage | 事件横截面模型 | 没有历史预期版本时伪造信号 |
| Forecast | regularized linear/GBDT quantiles；sign probability 后校准 | Bayesian model averaging、adaptive conformal | LLM 自报概率 |
| Meta | 全局先验 + regime 交互的强收缩线性/层级模型 | mixture-of-experts challenger | 每个 regime/sector/horizon 单独自由调权 |
| Covariance | EWMA + Ledoit-Wolf shrinkage + factor residual | regime-conditioned shrinkage / stress scenarios | 60×60 无收缩样本协方差 |
| Portfolio | 现金显式化；成本与不确定性惩罚的 quadratic/fractional Kelly | scenario log-utility + risk-constrained/distributionally robust Kelly[^2][^24] | raw full Kelly；只最大化预测均值 |
| Calibration | rolling reliability、Brier、ECE、quantile coverage | MAPIE adaptive intervals + block bootstrap | 随机 K-fold |

### G. 必须保持确定性的组件

- 数据获取、时间戳规范化、schema/质量门；
- 交易日历、宇宙资格、特征计算；
- 所有数值信号和收益标签；
- 预测校准与权重收缩；
- 协方差、场景、成本和组合求解；
- risk gate、订单计划、取整、幂等、提交、取消和对账；
- token/cost ledger、版本 registry、日志和评估；
- challenger promotion 测试。

### H. 真正适合 LLM 的组件

- 从 10-Q/10-K/8-K、earnings transcript 和公司公告中抽取发生了什么；
- 对比此前指引、市场预期和最新事实；
- 识别竞争格局、监管、管理层口径和商业模式变化；
- 标注量化关系可能失效的结构性变化；
- 生成有来源的 bull/bear、falsification 和未解决问题；
- 对重大目标仓位做一次独立 red-team。

LLM 不应：扫描全部市场、计算指标、生成价格、决定目标权重、绕过 risk engine、下单或修改自身生产策略。

### I. 推荐实施顺序

1. **冻结目标和安全契约**：主目标、100 倍跟踪指标、现有 hard risk、模式与预算。
2. **建立双时间数据 contract**：先定义表和验证，再拉更多数据。
3. **修复当前监控缺陷**：core 混入涨跌幅、hysteresis、逐条 freshness、交易日历。
4. **构建历史重放与 shadow ledger**：同一 feature/decision 函数可用于历史和当前时点。
5. **实现 3 个简单独立 alpha**：趋势/相对强度、事件后延续、短期残差反转；先不碰 RL。
6. **实现概率预测与校准报告**：预测在 outcome 前不可修改。
7. **加入 LLM 结构化研究**：只针对触发后的少数候选与全部持仓；1 主分析 + 1 反方上限。
8. **实现收缩的 synthesis 与目标组合**：现金、成本、协方差、不确定性。
9. **实现确定性 risk/order planner 与 SHADOW fill**。
10. **运行完整市场日 SHADOW**，故障注入并检查日志。
11. **只有满足 live gate 后，代码评审式地加入 Robinhood write adapter**；不是改一个 boolean。

### J. 能证明架构的最小端到端 DRY_RUN

第一版不需要 12 个 alpha、HMM、RL 或完整元模型。建议：

1. 读取真实账户、现金和持仓；
2. 生成修正后的 60 标的观察宇宙；
3. 拉取 60–120 日 OHLCV，并为每个数据点记录 `known_at`；
4. 计算 market context：SPY/QQQ/RSP/IWM 趋势、宽度代理、实现波动和 sector dispersion；
5. 运行三个互相可解释的引擎：
   - `residual_trend_v0`（20/60 日）；
   - `price_volume_event_v0`（gap/relative volume/突破）；
   - `short_residual_reversal_v0`（1/5 日，事件过滤）；
6. 每个引擎输出 5/20 日 excess-return distribution 的基线估计、样本数和不确定性；
7. 选 3 个新候选 + 所有持仓进入 LLM；
8. LLM 只生成结构化 evidence object，随后一次反方审查；
9. LLM 信号第一版只在报告中展示，不影响目标权重，直到有 shadow 校准样本；
10. 用 shrinkage covariance + fractional Kelly 二次近似产生 `cash + assets` 目标组合；
11. hard risk engine 审批并生成 hypothetical orders；
12. 保存完整 decision report、数据 hash、模型/提示版本、成本、拒绝候选未来表现。

这条最小路径能证明整个思想是否贯通，同时避免为了“高级”而先引入最难验证的层。

## 6. 验证与反过拟合协议

漂亮回测不是晋升依据。每个新信号/模型必须先注册假设、测试次数和接受标准：

1. **点时宇宙**：包含当时可交易且后来退市/并购的证券；
2. **滚动 walk-forward**：训练、校准、测试按时间前进；
3. **purge + embargo**：当 20/60 日标签跨越分割边界时，删除重叠样本；skfolio 已提供相关模型选择工具，可作为实现参考。[^7]
4. **固定成本情景**：spread/slippage 使用保守分位数，并做 2×/3× 压力测试；
5. **多重检验台账**：记录所有失败实验而非只保存赢家；
6. **PBO/CSCV**：评估被选择的最佳策略在测试切片变差的概率。[^3]
7. **Deflated Sharpe Ratio**：对选择偏差、非正态和尝试次数做修正。[^4]
8. **Bootstrap block**：保留时间依赖与横截面共同冲击；
9. **参数平台**：接受在邻近参数仍有效的宽平台，不接受尖锐最优点；
10. **跨制度稳定性**：牛市、熊市、高通胀、低波动、危机与 sector rotation；
11. **概率校准**：按预测桶比较命中率、Brier、区间覆盖和收益分布；
12. **Shadow champion/challenger**：只有预先定义的证据达标才晋升，LLM 无权自行晋升。

## 7. 关于实时部署和 LIVE 的硬结论

Robinhood 官方 Agentic MCP 当前确实提供账户、组合、实时报价、historicals、fundamentals、earnings、tradability、order review/place/cancel 等工具，并允许 agent 在没有逐笔确认时下单；Robinhood 同时明确提示 AI 可能误解、不完整或过时的信息，最终风险由账户持有人承担。[^21]

它可以作为 **执行与当前账户真相接口**，但不应成为所有历史研究的唯一数据仓库。MCP 是工具协议，不自动提供：

- 点时财务/预期数据库；
- 可重放的退市证券宇宙；
- 数据版本和修订语义；
- 幂等订单业务键；
- Databricks 长期 OAuth 运维保证；
- 模型正确性或风险审批。

在当前项目中，以下 live gate 均未完成：

- decision engine 和 target portfolio；
- risk config 的实际执行；
- broker adapter/write-tool 最小权限；
- `review_equity_order → place_equity_order` 两阶段状态机；
- 重试后不重复下单的 idempotency key；
- partial fill / reject / cancel / expired 状态处理；
- broker positions、orders、cash 与内部账本对账；
- stale/partial quote 全量拒绝；
- kill switch 在每个写调用前复查；
- 模拟 job retry、网络中断和账户错配的故障测试。

因此，明天的“完整测试”应定义为完整的 **SHADOW 市场日**：真实数据、真实账户读取、完整决策、完整 hypothetical order、模拟重试和对账，但零提交。等这些门通过后，再设计小额 canary live，而不是直接让整个 `$1,000` 进入自治执行。

## 8. 数据与 GitHub 的边界

项目的代码、配置模板、schema、模型卡、策略版本、测试和这份研究报告可以提交到 GitHub。

以下内容不应进入当前公开仓库，即使用户希望“所有文件和数据都写到 GitHub”：

- OpenAI key、Robinhood OAuth token、cookie/session；
- 完整 account number、持仓/订单/税务明细；
- 原始新闻全文或受许可限制的数据；
- 高频市场快照和带账户身份的决策日志；
- Databricks secrets。

运行数据应进入私有 Delta/object storage；GitHub 只保存脱敏 schema、小型合成 fixture 和不可逆聚合研究结果。公开 Git 不是账户数据库或 secret manager。

## 9. 最终建议

设计稿可作为 `architecture_v1` 的基础，但应批准以下修订后再编码：

1. 把 100 倍目标改成带期限和生存底线的跟踪指标；主优化仍是风险约束的长期增长。
2. 保留当前 hard risk，直到用户单独审查并版本化修改；模型不能修改。
3. 第一版只做 3 个 alpha 和透明的收缩模型，不使用 RL、不自动挖策略、不让 LLM 影响仓位。
4. 先建立 point-in-time 与 replay，再谈复杂模型。
5. 用 Qlib/skfolio/Riskfolio 等提取接口和数学思想，不复制其默认数据、默认参数或回测结论。
6. 将一次 structured LLM research + 一次 red-team 作为上限，只有事件触发才调用。
7. 明天做 SHADOW E2E；LIVE 延后到执行/风险/幂等/对账 gates 全部可验证。

这条路线没有承诺 100 倍，但它最大限度地避免两件更危险的事：用复杂度伪装 alpha，以及用一次漂亮回测把 `$1,000` 变成不可复现的实验损失。

## Sources

[^1]: J. L. Kelly Jr., [“A New Interpretation of Information Rate”](https://onlinelibrary.wiley.com/doi/abs/10.1002/j.1538-7305.1956.tb03809.x), Bell System Technical Journal 35(4), 917–926 (1956).
[^2]: Busseti, Ryu, Boyd, [“Risk-Constrained Kelly Gambling”](https://web.stanford.edu/~boyd/papers/pdf/kelly.pdf), Stanford (2016), with [official code](https://github.com/cvxgrp/kelly_code).
[^3]: Bailey, Borwein, López de Prado, Zhu, [“The Probability of Backtest Overfitting”](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf) (2015).
[^4]: Bailey and López de Prado, [“The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality”](https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID2460551_code87814.pdf?abstractid=2460551) (2014).
[^5]: Microsoft, [Qlib repository](https://github.com/microsoft/qlib) and [portfolio strategy documentation](https://github.com/microsoft/qlib/blob/main/docs/component/strategy.rst).
[^6]: Microsoft Qlib, [`Ref` expression implementation](https://github.com/microsoft/qlib/blob/main/qlib/data/ops.py), whose documentation states `N < 0` retrieves future data.
[^7]: skfolio, [official repository](https://github.com/skfolio/skfolio) and [model-selection documentation](https://github.com/skfolio/skfolio/blob/main/docs/user_guide/model_selection.rst).
[^8]: Boyd et al., [“Multi-Period Trading via Convex Optimization”](https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf) (2017), and [cvxportfolio](https://github.com/cvxgrp/cvxportfolio).
[^9]: Dany Cajas, [Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib).
[^10]: PyPortfolio, [PyPortfolioOpt](https://github.com/PyPortfolio/PyPortfolioOpt).
[^11]: scikit-learn-contrib, [MAPIE](https://github.com/scikit-learn-contrib/MAPIE).
[^12]: online-ml, [River](https://github.com/online-ml/river).
[^13]: OpenBB Finance, [OpenBB data platform](https://github.com/OpenBB-finance/OpenBB).
[^14]: Tauric Research, [TradingAgents](https://github.com/TauricResearch/TradingAgents) and its [agent graph source](https://github.com/TauricResearch/TradingAgents/blob/main/tradingagents/graph/setup.py).
[^15]: Yu et al., [FinMem official repository](https://github.com/NathanDDDD/finmem-llm-stocktrading).
[^16]: AI4Finance Foundation, [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) and [benchmark tasks](https://github.com/AI4Finance-Foundation/FinGPT/blob/master/fingpt/FinGPT_Benchmark/readme.md).
[^17]: AI4Finance Foundation, [FinRL stock trading environment source](https://github.com/AI4Finance-Foundation/FinRL/blob/master/finrl/meta/env_stock_trading/env_stocktrading.py).
[^18]: QuantConnect, [LEAN](https://github.com/QuantConnect/Lean).
[^19]: NautilusTrader, [official repository and live/backtest architecture](https://github.com/nautechsystems/nautilus_trader).
[^20]: Virat Singh, [AI Hedge Fund](https://github.com/virattt/ai-hedge-fund).
[^21]: Robinhood, [Trading with your agent: tools, execution and disclosures](https://robinhood.com/us/en/support/articles/trading-with-your-agent/).
[^22]: U.S. SEC, [EDGAR Application Programming Interfaces](https://www.sec.gov/search-filings/edgar-application-programming-interfaces).
[^23]: Federal Reserve Bank of St. Louis, [ALFRED](https://fred.stlouisfed.org/docs/api/fred/alfred.html) and [real-time periods](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html).
[^24]: Sun and Boyd, [“Distributional Robust Kelly Gambling: Optimal Strategy under Uncertainty in the Long-Run”](https://arxiv.org/abs/1812.10371) (2018).
