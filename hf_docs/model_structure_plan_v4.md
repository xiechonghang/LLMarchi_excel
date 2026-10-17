# 模型结构阅读工具：项目计划 v4（垂直切片 · 流程精简）

更新日期：2026-10-13。本计划由用户在 2026-10-13 明确发起："去掉繁重的流程，快速得到可实用的工程"，并调整分工。

**同日分工修订：Codex 恢复任务分发与最终核验，开发通过 Claude CLI，独立测试通过 opencode CLI + `huoshanai/deepseek-v4.1-flash`。** 最新执行入口为[项目推进说明书](project_execution_handbook_v4.md)，包含实际工程快照、当前红测和新对话启动提示。本文 §0 复盘及 §4 初始盘点保留当时背景；与最新分工/状态冲突处以说明书为准。

**v3 的需求与验收标准仍然有效**（[model_structure_plan_v3.md](model_structure_plan_v3.md) 是需求权威）；v4 改变的是**执行顺序**（垂直切片优先，第一个可见成果提前）和**协作流程**（逐微任务仪式收敛为工作包推进+里程碑门禁）。三方职责保留，分发通道改为终端 CLI。v3 的 S0–S8 阶段编号保留作对照，不再作为执行单位。

---

## 0. v4 的改进背景：v1–v3 执行中的结构性问题（2026-10-13 复盘）

v4 不是推翻 v3 的设计——v3 的需求、轴字典与验收条款被证明是扎实的。问题全部出在**执行流程与里程碑结构**上。以下每条弊端都有可核对的量化证据，改进方式在对应章节落地。

### 0.1 弊端一：流程开销倒挂（协调产物 ≈ 产品代码的 11 倍）

v1–v3 期间磁盘实况：产品源码约 **3,900 行**（6 个模块），测试约 11,400 行，而 `outputs/_coordination/` 下的任务单/交接报告/审核记录达 **349 个文件、约 44,600 行**。每开发一个微任务要完成"任务单 → 开发报告 → 交付哈希固定 → 独立验证报告 → 修复单 → 复验 → Codex 审核"的完整闭环；为开发 515 行的图引用检查器，产生了 46+56 项测试、853 项全量回归反复执行、200 个合成图 SCC 交叉对照——全部发生在**任何真实模型数据进入系统之前**。单位产品代码背负约 10 倍流程文档，而这些文档不产生任何用户可见价值。

**v4 改进**（§2.3）：取消 per-micro-task 三件套，过程事实收敛到单一 `progress.md`；正确性主防线从"三方互检"改为"机器契约 + 受控错误注入"（自动化、可重复、随代码永久保留）；独立测试从逐微任务收敛为每里程碑一次门禁。

### 0.2 弊端二：切片过碎，收尾仪式吞掉交付本身

v1 期间的图 IR 开发被切成约 10 个"探针"小单（node/port/tensor/axis/edge/guard/alias/state_identity/fold_boundary/graph_document），每个各自走一遍全流程。到 2026-10-13，代码快照已经 56/56 及全量 853/853 通过，但阶段仍无法关闭——因为**一份报告里"红转绿计数是 5 还是 6"需要校正、一份正式开发汇总缺失**。流程文件的完整性成了比代码正确性更高的门槛。多轮迭代止步于"局部 IR 有进展"，用户看不到阶段性成果。

**v4 改进**（§2.3、§3）：以用户可感知的交付物定义里程碑（V1 = 能打开的 structure.html），代码测试绿即算该步完成；文档只保留结论页；不再为形式完整性反复收尾。

### 0.3 弊端三：里程碑放得太靠后，基础设施在空转状态下打磨

原 v3 把第一个用户可见成果压在 S3，而 S0–S2 全是基础设施。S1 内部的场景求值、源码证据、JSON Schema 等能力，本质是**只有对着真实模型内容才有意义的东西**——在空转状态下反复独立验证它们，投入产出比极低。且验证强度（合成图穷举、SCC 交叉对照）放在了错误的位置：应该对"提取出的真实图"严格，而不是对"空图数据结构"严格。

**v4 改进**（§3）：垂直切片优先——V1 直接冲 CSA·TopK 代表层闭环（用户核心原则"每类结构只存一份代表模板"的最小完整体现），S1 剩余基础设施降级为随用随补；折叠连通性等价证明不再作为前置（折叠改为纯渲染分组，图数据层不删边——这本来就更贴合"展开是渲染策略，不是存储策略"的用户原则）；核验强度集中在真实模型契约上。

### 0.4 弊端四：三方工具依赖叠加了协作单点与网络故障

v3 分工依赖三个外部工具的持续可用性：Codex 做分发与验收、DeepSeek Harness 做独立测试、Claude 做开发。期间多轮出现 Claude 502、Harness TRANSPORT 断连，每次断连后"代码与测试证据保留、但汇总与定版交接未完成"，恢复成本高。任务分发与最终验收集中在 Codex 一方，也使其成为流程瓶颈。

**v4 改进**（§2）：任务分发收归开发方（Claude Code，减少一轮跨工具交接）；独立测试改为 opencode 无头调用 deepseek-v4.1-flash（2026-10-13 冒烟通过，可自动化、不依赖 GUI 应用在线）；Codex 退出常规流程。单次门禁失败的重试成本从"恢复三方会话"降为"重跑一条命令"。

### 0.5 弊端五：分工结构让用户承担了流程成本

v3 中用户除了阅读验收，还实际承担了授权确认、工具连通性恢复、会话观察等流程性事务；每个微任务的三方协作本质上都以用户的时间和注意力为隐性预算。用户反馈"进度太慢，多轮迭代始终没有阶段性成果"的根源即在于此。

**v4 改进**（§2.3）：用户参与压缩到每个里程碑一次阅读验收，工程正确性完全由自动化测试与门禁承担。

### 0.6 保留项：v1–v3 中的真实资产

以下内容经复盘确认有效，v4 明确保留、不推倒重来：
- **S0 全部交付**（输入扫描/注册/inspect，137 项回归）。
- **IR 库三件**：`graph_ir.py` / `graph_document.py` / `checks/graph_contracts.py`（853 项测试基线，接口清晰）——直接复用。
- **诚实记录原则**：不伪称通过、unknown 不当 false、失败记录保留、测试数不冒充完成度。
- **轴字典与核验在出图前的设计原则**。
- v3 计划文档本身（作为需求权威继续引用）。

## 1. 需求（继承 v3，不变项摘要）

- 输入模型配置与源码，输出**有证据的**离线数据流阅读页 `structure.html`；file:// 可读、无 CDN、无裸数字 shape。
- **每类结构只存一份代表模板**（用户核心原则："展开是渲染策略，不是存储策略"）；层序/数量在 layer_plan 中。
- 主案例 DeepSeek-V4-Flash-0731 参考实现：SWA/CSA/HCA、低秩 Q/O、共享 KV、mHC、Hash/Top-K MoE、Compressor/Indexer、缓存状态、DSpark。
- 轴字典：每条轴带 symbol/name/含义/来源；`hidden_states [batch_size=1, seq_len=1, hc_streams=4, hidden_size=4096]` 样式。
- 证据上限 = 源码核对 + 配置/场景推导；不伪称运行过模型；unknown 不当 false；核验在出图前。
- 同一语义图投影出 CSV 表格，不各自维护事实；Qwen3-0.6B 迁移回归；最后固化为 Skill。

完整维度字典与场景清单见 v3 §4.2/§4.3，验收条款见 v3 各阶段"验收"段——v4 节点验收直接引用它们。

## 2. 分工与流程（v4 变更，针对 §0.1/§0.4/§0.5 的弊端）

### 2.1 角色调整

| 角色 | v3 | v4 |
|---|---|---|
| 开发 | Claude Code | **Claude Code，经 Claude CLI 执行开发与修复** |
| 任务分发 | Codex | **Codex，通过终端调用 CLI 派发工作包**（不再逐微任务写三件套文档） |
| 独立测试 | DeepSeek Harness（逐微任务） | **DeepSeek（deepseek-v4.1-flash，经 opencode 无头调用），只在阶段门禁（V1/V2/V3 各一次）** |
| 最终验收 | Codex | **DeepSeek 独立门禁 → Codex 最终核验；用户阅读验收单独记录** |

### 2.2 通道现实（2026-10-13 实测，两次更新）

- **独立测试通道 = opencode 无头调用 DeepSeek 模型（已验证可用）**：用户配置了 `~/.config/opencode/opencode.jsonc`（火山引擎网关，huoshanai provider），本机 `~/.opencode/bin/opencode` 1.18.4 支持 `opencode run -m huoshanai/deepseek-v4.1-flash "<任务>"` 无头执行，冒烟测试通过（2026-10-13）。门禁任务由此通道执行，**不需要用户中转**。API key 只存在于 opencode 配置文件中，不得写入任何项目文档/记忆/产物。
- **DSH 应用仍是用户侧工具**：其 `dsh` CLI 仅能启动 GUI profile，无 headless 接口；用户如想继续用 DSH 会话参与阅读反馈，属可选补充，不再是门禁必需。
- **Codex CLI 被用户明确排除**（2026-10-13：不要使用 codex cli；当日报 `workspace routing discovery failed`）。
- Claude Code 子代理与主会话同模型，**不构成跨模型独立性**——不用它冒充独立测试。

### 2.3 仪式削减规则（v4 生效，针对 §0.1/§0.2 的弊端，取代 development_workflow.md 的逐单闭环）

1. **不再产生** per-micro-task 的 dispatch/results/audit 三件套文档。过程事实收敛到 `structure_reader/docs/progress.md`（真实文件、真实退出码、四态状态）与代码内测试。
2. **正确性主防线 = 机器契约**：每个提取/渲染模块配 unittest（标准库，`python3 -m unittest`），关键语义配**受控错误注入**——构造已知错误变体（删边/换方向/拆别名），断言检查器必须报错。
3. **阶段门禁一次独立测试**：每个里程碑（V1/V2/V3）完成时，Codex 将一份自包含门禁单经 `opencode run -m huoshanai/deepseek-v4.1-flash` 发给 DeepSeek 执行；缺陷由 Codex 交 Claude CLI 修复，按影响范围复验，最后由 Codex 核验。门禁单、结果及最终结论存 `outputs/_coordination/v{N}/gate/`。
4. 用户只做两件事：V1 后的阅读验收反馈；（可选）继续用 DSH 会话做阅读体验反馈。
5. 诚实记录原则不变：不写"已验证"而未验证的内容；测试数不冒充完成度；失败记录保留。

## 3. 里程碑（垂直切片优先）

v4 里程碑结构直接针对 §0.3 的弊端：原 v3 把第一个可见成果压在 S3，v4 把 S1 的剩余基础设施**降级为随用随补**，直接冲代表层闭环。折叠连通性等价证明、合成图 SCC 交叉对照等高开销项**不再作为前置**（折叠改为纯渲染分组，图数据层不删边；核验强度集中到真实模型契约上）。

### V1：CSA·TopK 代表层闭环（对应 v3 S2+S3 精简版）

交付一条命令产出首个可打开、有证据、有核验的交付物：

- `layer_plan.json`：四类主干分组（2/1/20/20，合计 43，并集校验）+ 实例绑定（40/41/42 DSpark 读出差异保留）。
- `semantic_graph.json`：CSA·TopK 代表层完整图——双 mHC、Attention（低秩 Q、共享 KV 别名、窗口）、Compressor（overlap 压缩+状态）、Indexer（含 qr 输入、Top-K select 边、独立缓存状态）、MoE（Gate/专家模板/shared expert），全部节点/边带 model.py 行号证据。
- 场景求值：v3 §4.3 场景集（prefill 3/4/5/127/128/129、decode P=2/127/128），guard 三态激活表。
- 核验：通用引用检查（既有 853 项测试基线）+ DeepSeek 模型契约 + 受控错误注入 ≥8 项。
- `structure.html`：自包含离线页，L0 概览（层序条带+分组）+ L1 代表层展开 + 轴详情面板 + 源码证据行号 + 场景切换。
- `connections.csv` / `tensor_axes.csv` 同源导出。
- CLI：`build --model deepseek_v4_flash_0731 --scenario-set standard` 一次跑通 `classify → extract → validate → render → export`；核验失败不产正式页。
- 验收：DeepSeek 门禁通过 + Codex 最终核验；用户打开 structure.html 的阅读反馈单独记录，未反馈不代签。

**明确不做**（V1 内）：SWA·Hash/CSA·Hash/HCA·TopK 提取、DSpark 子图、fold 连通性等价证明、JSON Schema、source_contracts AST 断言引擎（证据先以"行号+表达式原文+人工核对声明"承载）。

### V2：其余变体 + DSpark + HF 差异（对应 v3 S4+S5 精简版）

- 补 SWA·Hash、CSA·Hash、HCA·TopK 模板（HCA 无学习式 Indexer、Hash 的 token-ID 依赖可见）。
- DSpark：首 stage 投影/草稿输入、末 stage head/Markov/confidence、46 项 compress_ratios 的归属、prefill 不画不存在的草稿路径。
- HF 实现具名差异报告；完整阅读体验打磨（L2 展开、状态面板、阅读指南）。
- 受控错误扩充（v3 §5.3 全列表）。

### V3：表格齐套 + Qwen 迁移 + Skill（对应 v3 S6+S7+S8）

- 六类 CSV 齐 + 可选 xlsx；可重复 analysis 目录（内容寻址 analysis_id）。
- Qwen3-0.6B：独立 Q/K/V、GQA、SwiGLU、普通残差；DeepSeek 回归不破。
- Skill 固化（模型/版本/场景选择、build/validate、阅读交付步骤）；未知模型/基线失配/缺依赖的失败路径。

## 4. 当前状态盘点（2026-10-13，与磁盘核对）

**有效资产：**
- S0 正式通过（输入扫描/注册/inspect，137 项回归），继续有效。
- `graph_ir.py`（1175 行）+ `graph_document.py`（273 行）+ `checks/graph_contracts.py`（515 行）：IR/序列化/引用检查器，853 项回归基线健康——**全部复用，不重写**。
- v3 计划的需求/轴字典/验收条款：继续权威。

**本轮新落盘、未测试**（垂直切片第一批，写于 v4 定稿前）：
- `src/structure_reader/scenario_evaluator.py`：白名单 AST 有限表达式求值器（整数/布尔、三态、min/max、防深防大）。
- `src/structure_reader/rules/deepseek_v4_flash_0731.py`：轴字典（含 config 交叉核对）、层分组推导（并集/重叠校验）、守卫目录（8 条，白名单谓词）、场景集（9 个）。
- `rules/` 尚缺 `__init__.py`；两者均无测试。

**仍然缺失（= V1 工作量）**：`extract/` 提取器、`checks/deepseek_v4_contracts.py`、build 编排、渲染器、CSV 导出、CLI `build` 命令、全部单测。

**事实核对已完成**（V1 建图的依据，行号在 model.py）：层分组 compress_ratios 映射（459,472-477）、共享 KV 单投影（466,508）、qr→Indexer（502,517）、inverse RoPE 先于 O 投影（539-547）、mHC 方向契约（690-693）、hash 门控查表（581-582）、DSpark 读出（917-926）。

## 5. 下一步动作（V1 执行清单，顺序即计划）

1. `rules/__init__.py` + 两个新模块的 unittest（求值器边界、轴字典↔config 交叉核对、分组并集、守卫白名单）。
2. `extract/deepseek_v4_reference.py`：CSA·TopK 代表层（layer 4）建图——节点/端口/张量/轴/边/别名/守卫/状态版本，每条带行号证据。
3. `checks/deepseek_v4_contracts.py`：方向契约（mHC comb、共享 KV 单投影、qr→Indexer、inverse RoPE 顺序）+ 受控错误注入 ≥8 项。
4. `build_pipeline.py`：classify → extract → validate → render → export 编排；失败不落正式产物。
5. `render/`：自包含 HTML（无 CDN；内嵌图数据/轴图例/证据行号/场景切换）。
6. `export/semantic_tables.py`：connections.csv、tensor_axes.csv（从图投影）。
7. CLI `build` 命令挂接。
8. 全量测试跑绿 → Codex 产出 V1 门禁单并经 opencode 调 deepseek-v4.1-flash → 缺陷交 Claude CLI 修复 → Codex 最终核验（结果存 `_coordination/v1/gate/`）。
9. 用户：打开 structure.html 做阅读验收。

工作量粗估：**3–5 个工作会话**（原 v3 对应段估 4–7 人日，且不含仪式开销）。

## 6. 文档与记录约定（v4）

- 本文件保留 V1/V2/V3 执行计划；[项目推进说明书](project_execution_handbook_v4.md)是最新分工与接续入口；v3 保留为需求与验收条款权威，v1/v2/archive 不再作为执行依据。
- §0 复盘的量化证据可核对：`structure_reader/outputs/_coordination/`（349 个文件，v1–v3 协调产物原样保留作历史，不再新增）、`docs/progress.md`（阶段状态与实际测试数）、[2026_10_13_v3_project_checkpoint.md](stage_reports/2026_10_13_v3_project_checkpoint.md)（v3 末期状态快照）。
- `development_workflow.md` 的三方逐单流程**被本文件 §2 取代**（文件保留作历史）。
- 进度唯一入口：`structure_reader/docs/progress.md`；里程碑门禁单放 `outputs/_coordination/v{N}/gate/`。
- 阶段快照：每个 V 里程碑完成后在 `hf_docs/stage_reports/` 加一份，含实际文件、测试结果、DeepSeek 门禁结论、未覆盖范围。
