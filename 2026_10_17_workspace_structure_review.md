# LLMarchi_excel 工作区结构与工程目的分析

分析日期：2026-10-17（Asia/Shanghai）。分析范围：`LLMarchi_excel/` 全工作区；`models/`、`hflib/` 只作为只读输入清点，不精读。
分析者：pi（只读核查 + 本地复跑，未启动任何外部模型通道，未修改产品代码）。

本文不是阶段验收记录，不替代 `hf_docs/project_execution_handbook_v4.md`（分工与接续入口）与 `structure_reader/docs/progress.md`（滚动进度）。它是一份**外部视角的结构体检**，所有数字都可用附录 A 的命令重跑复核。

---

## 1. 工程目的

**把 DeepSeek-V4-Flash-0731 官方参考推理源码做"静态取证"，用一条命令编译成一份可离线打开、每个节点/边都带 `model.py` 行号证据的模型结构阅读页（外加同源 JSON/CSV）。全程不加载权重、不 `import torch`、不跑推理、不声称数值等价。**

动机是逐层收敛出来的，磁盘上留有完整的化石层：

| 阶段 | 证据文件 | 目的演进 |
| --- | --- | --- |
| ① HF 生态调研 | `hf_docs/README.md`(371 行)：Transformers 仓库与 HF 模型仓库的依赖三问、模型文件构成、`scripts/download_hf_model.py` | 先搞清"下载了什么、谁依赖谁、离线能否用" |
| ② 方法论选型 | `hf_docs/model_structure_visualization.md`(190 行)：torchview / TorchLens / PyTorch FX / Google Model Explorer / torch.export+ONNX 评估 | 判定现有工具不够，必须自建 IR + 行号证据 + 分层阅读 |
| ③ 工程落地 | `model_structure_plan_v1→v2→v3→v4` + `project_execution_handbook_v4.md` + `structure_reader/` | v1 大而全 → v2 收敛静态阅读 → v3 加模板/实例与出图前核验 → v4 砍流程、垂直切片 |

**不可削减的产品信条**（v3 为需求权威，v4 只改执行顺序）：

- 证据上限 = 源码核对 + 配置/场景推导；`unknown` 必须带原因，不得当 `false`；`P>0` 不自动等于存在有效 cache。
- 核验在出图前；检查器必须用**受控错误注入**证明自己能拒绝坏图，而不是只证明正常图能过。
- 展开是渲染策略，不是存储策略 → 每类重复结构只存一份代表模板，43 层实例与差异放 `layer_plan` / `instance_bindings`。
- 图是唯一事实，CSV 与 HTML 都只是同源投影（`export/semantic_tables.py` 仅 68 行，因为它真的只是投影）。
- 首版终态（V3）：六类 CSV + Qwen3-0.6B 迁移回归 + 固化为项目 Skill。

---

## 2. 工作区四层结构

```
LLMarchi_excel/                      ← 非 git 工作区根
├─ README.md               [治理层] 工作区入口 + 状态快照（已滞后）
├─ hf_docs/                [治理层] 需求/计划/方法论/阶段快照，236KB / 2,609 行
│   ├─ README.md                        HF 资料与依赖关系主线
│   ├─ model_structure_plan_v3.md       需求与验收条款权威（433 行）
│   ├─ model_structure_plan_v4.md       垂直切片 V1/V2/V3 + 流程精简（159 行）
│   ├─ project_execution_handbook_v4.md 接续入口、分工、门禁模板（390 行）
│   ├─ model_structure_visualization.md 方法论与工具评估
│   ├─ development_workflow.md          v3 逐单流程，已被 §2 取代（保留历史）
│   ├─ archive/                         v1、v2、v2_review、旧说明入口
│   └─ stage_reports/                   2026_10_13_v3_project_checkpoint.md
├─ models/   22MB          [输入层] 只读。DeepSeek-V4-Flash-0731（分析对象，无权重）
│                          + DeepSeek-V4.1-Flash（未被任何 configs 引用、未使用）
├─ hflib/    815MB         [输入层] 只读。transformers（自带 git，HEAD 536ecc00）+ diffusers（未用）
│                          实际只消费 2 个文件：modeling_deepseek_v4.py / configuration_deepseek_v4.py
├─ scripts/download_hf_model.py        [工具层] 工作区级下载器，不是产品入口
├─ .claude/settings.local.json         [工具层] 仅白名单一条工程解释器的 Bash
└─ structure_reader/  200MB            [产品层] 唯一开发根（见 §5）
```

**治理权威链是显式写死的**（handbook §1）：用户当前指令 > `handbook_v4` > `plan_v4` > `plan_v3`；v1/v2/archive 不作执行依据。这是本工作区最特别的结构特征 —— **文档即治理**：角色分工、产物目录 schema、调用命令模板、甚至"不得用删除红测凑绿"的红线都写在 markdown 里并被当作约束执行。

**执行分工（v4 修订 + 用户 10-13 覆盖）**：Codex 派单与最终核验；Claude CLI 开发；opencode CLI + `huoshanai/deepseek-v4.1-flash` 做每里程碑一次独立门禁；用户只做里程碑后的阅读验收。被分析模型（V4-Flash-0731）与测试模型（V4.1 Flash）不得混淆。

---

## 3. 端到端数据流（已实测跑通）

```
model.py/kernel.py/generate.py + config.json(根+inference)              ← models/
modeling_deepseek_v4.py + configuration_deepseek_v4.py                  ← hflib/
   │ 8 个输入文件 SHA256 冻结于 configs/source_baselines/（失配→unreviewed / failed）
   ▼
inventory.py ──▶ inspect 扫描包  outputs/_inventory/<model>/<content_id>/   (现存 24 个)
   ▼
rules/deepseek_v4_flash_0731(轴字典/8 守卫/9 场景) → scenario_evaluator(白名单 AST 三态求值)
   ▼
extract/deepseek_v4_reference  ── 主图 107 节点 / 173 边
   → deepseek_v4_variants      ── csa_topk / csa_hash / hca_topk / swa_hash
   → deepseek_v4_dspark → dspark_stage → endpoints/exit → markov → confidence → integration
                                ── 集成图 276 节点 / 364 边 / 264 张量 / 18 状态版本 / 13 转换
   ▼
checks/graph_contracts(通用引用) + deepseek_v4_contracts + dspark_contracts(2,939 行)
   ▼
layer_plan / instance_bindings / scenarios → render/structure_page(单文件 HTML) + export/semantic_tables
   ▼
outputs/deepseek_v4_flash_0731/<content_id>/{viewer,data,tables,reports,logs,run_manifest.json}
   .work 暂存 → os.replace 原子提交 → 同输入幂等复用 → 内容不一致拒覆盖 → 失败落 outputs/_failed/
```

用户命令只有两条：`inspect`、`build`（`validate/render/export-tables` 未独立暴露，能力内含于 build）。
最新候选：`outputs/deepseek_v4_flash_0731/6dede2fc…/`（3.8MB，2026-10-17 11:17）。

---

## 4. 体量分布：工程的真实"结构形状"

| 类别 | 规模 | 磁盘 |
| --- | --- | --- |
| 产品源码 `structure_reader/src/` | 26,404 行 / 30 文件 | ~1MB |
| 回归测试 `structure_reader/tests/` | 23,992 行 / 41 个测试文件 / **1,538 个 test 方法** | 6.0MB |
| 计划与方法论 `hf_docs/` | 2,609 行 | 236KB |
| **过程证据 `structure_reader/.work/coordination/`** | **1,079 文件 / 327,489 行** | 39MB |
| **历史协调档案 `structure_reader/outputs/_coordination/`** | **349 文件 / 26,411 行** | 5.4MB |
| 生成产物 `structure_reader/outputs/` | 67 个候选（各≈4MB）+ **541 个 `_failed`** + 24 个扫描包 | 152MB |
| 只读输入 `models/ + hflib/` | — | 837MB |

结论：**技术资产 ≈ 5.0 万行，过程记录 ≈ 35.4 万行，比值 ≈ 7×**。而 `plan_v4` §0.1 当初开火的正是"协调产物 349 文件 / 44,600 行、倒挂 11 倍"，并承诺"取消逐微任务三件套、过程事实收敛到单一 `progress.md`"。实际发生的是：**过程文件从 349 → 1,079（3 倍增长），只是从 `outputs/_coordination` 搬到了 `.work/coordination`。** 工作区 90% 以上体积是输入副本 + 生成产物 + 过程记录，而非可携带源码。

---

## 5. `structure_reader` 内部结构

### 5.1 分层与依赖（自底向上）

```
L0 基元（0 内部依赖）: graph_ir(1,175)  model_registry(311)  scenario_evaluator(348)  render/structure_page(2,556)
L1 文档/清单        : graph_document(273)←graph_ir ; inventory(889)←model_registry
L2 规则/证据        : rules/deepseek_v4_flash_0731(475)←scenario_evaluator ; extract/evidence(256)←rules
L3 主图提取         : extract/deepseek_v4_reference(1,901) → deepseek_v4_variants(822) → deepseek_v4_dspark(963)
L4 DSpark 六组件链  : dspark_stage(2,273) → endpoints(1,406)/exit(1,090) → markov(1,649) → confidence(1,153)
                      → integration(1,683)
L5 校验             : checks/graph_contracts(515) → deepseek_v4_contracts(1,727) / dspark_contracts(2,939)
L6 编排/落盘        : build_pipeline(400) + cli(1,164) + layer_plan(321) + export/semantic_tables(68)
```

### 5.2 顶层目录职责

| 路径 | 职责 | 备注 |
| --- | --- | --- |
| `scripts/read_model_structure.py` | 薄用户入口（版本检查后转发 `cli.main`） | 同目录另有 6 个一次性 `b4_fix0X_{smoke,collect_evidence}.py` |
| `src/structure_reader/` | 库本体，`dependencies = []`，仅标准库 | 唯一真正的可携带资产 |
| `configs/models/` | 模型注册 JSON，实际被消费的单一数据源 | 仅 1 个模型 |
| `configs/source_baselines/` | 8 输入文件 SHA256，运行时不自动更新 | 冻结 |
| `tests/` | 1,538 项回归 + `fixtures/*mutant*.json` 受控错误 | 全绿（24.96s） |
| `docs/` | user_guide / output_reference / developer_guide / progress | progress 151 行 / 50KB |
| `outputs/` | 正式产物根 + `_inventory` + `_failed` + `_coordination` | 152MB，无 GC |
| `.work/` | 声明为"可丢弃暂存" | 实际存着门禁唯一证据 + 被测试读取 |
| `results/` | 空目录残留 | 死目录 |

---

## 6. 问题清单

严重度：`高` = 影响可复现性/可信度或已自相矛盾；`中` = 影响可维护性与下一步扩展；`低` = 卫生问题。

### A. 工作区级（6 项）

| # | 严重度 | 问题 | 证据 | 影响 |
| --- | --- | --- | --- | --- |
| A1 | 高 | **`.work` 语义自相矛盾**：文档定义它"可丢弃、不作交付"，但 V1 独立门禁 round01–04 的 `freeze_manifest.json / result.json / 独立测试日志 / 页面探针` 唯一副本就在这里；handbook §9.1 要求门禁落 `outputs/_coordination/vN/gate/`，未执行 | `plan_v4` §6、`handbook` §9.1 vs `.work/coordination/v1/gate/round0{1..4}/` | 一次误清理即销毁唯一独立验证证据；审计链不可靠 |
| A2 | 高 | **无版本控制**。工作区与 `structure_reader/` 均非 git 仓库（`git rev-parse` → fatal），一致性靠手写 SHA 快照文件（`codex_before_*_hashes.json`、`before_fix0X_products.json`）与 541 个 `_failed` 目录代替 diff/回滚 | 磁盘核查；唯一 git 是 `hflib/transformers` | 无分支、无回滚点、无变更可追溯；A1 风险被放大 |
| A3 | 高 | **文档层全面滞后**：工作区 README 与 handbook §3.4/§3.5/§10 仍写"892 项、1 failure 2 errors、CLI 只有 inspect、未找到正式 structure.html、V1 pending"；实际 1,538 项全绿、67 个 HTML、B4 已到补修05 | `head -20 hf_docs/project_execution_handbook_v4.md` §3.5 vs 本次复跑 | 新对话按 handbook 接续会重做已完成工作；权威链名义 4 层实则互相矛盾 |
| A4 | 中 | **执行通道退回人肉观察**。v4 §0.4 的改进点（独立测试=一条无头命令）被用户 10-13 覆盖为 VS Code 可见终端 + `/clear` 派发 + 每 20 分钟 automation 轮询；handbook §5.3/5.4 两个调用模板从未按原样使用，progress 里连"独立报告 headless 文字与实际可见 UI 派发不符"都被记录 | `progress.md` 大量条目记录 spinner 名/Terminal 编号/IDE 连接/权限模式 | 流程监控本身重新成为主要产出，A 类问题（§4 的 7× 倒挂）未被真正削减 |
| A5 | 中 | **阶段编号 4 套并存**：S0–S8（v3）/ V1–V3（v4）/ B1–B4 + 补修01–05（v2/dspark）/ R1–R2a–R2b（reader_recovery），且状态唯一入口是 151 行 / 50KB 的单行老化日志 | `docs/progress.md` | 状态不可查询、不可机读；"到底什么通过了"必须通读全文 |
| A6 | 中 | **V2 从未走自己规定的独立门禁**。V1 走过 round01–04；V2 把代码从约 8K 行扩到 26K 行并新增 364 边集成图，`progress.md` 自陈"完整 V2 独立门禁与用户阅读验收待完成" | `README.md` DSpark 边界段、`progress.md` 10-17 条目 | 产品主体处于"自测绿 + 未经独立验证"的中间态 |

### B. 工程级（11 项）

| # | 严重度 | 问题 | 证据 | 影响 |
| --- | --- | --- | --- | --- |
| B1 | 高 | **测试绿灯不可移植**。6 个页面测试按 mtime 挑 `outputs/` 最新 HTML（`_latest_output_dir()`），硬编码 `/opt/homebrew/bin/node`，并把探针日志写进 `.work/coordination/**/logs`；`test_page_topology_r2b.py:60` 与 `test_page_js_execution.py:20` 直接依赖 `.work` 内文件。`outputs/`、`.work/` 均在 `.gitignore` | `tests/test_page_topology_r1/r2a/r2b.py`、`test_b4_fix01_page.py`、`test_page_js_execution.py`、`test_b4_fix02_defense.py` | 干净环境无法复现"1538 全绿"；部分断言对象是历史产物而非当次构建 |
| B2 | 高 | **渲染器 = 110KB Python 内嵌字符串**。`render/structure_page.py` 2,556 行中 110,423/111,801 字符在三引号模板内（HTML+CSS+JS 混于一文件），无模板分离、无独立 lint/语法检查 | 正则统计三引号块 | progress 反复出现"页面 JS SyntaxError / 按钮逻辑错位 / 适配 k 值算错"的结构性根因；只能靠 Node VM+DOM 桩间接验证 |
| B3 | 高 | **产品可读性无验证通道**。V1 用户阅读验收在 10-15 被明确改记"未通过（不再仅 pending）"，此后一切推进都在"工程 passed、阅读未签"下继续；progress 反复自陈"Node VM + DOM 桩 ≠ 浏览器""file:// 限制不绕过""不代签" | `progress.md` 10-15 条目、`README.md` "完整 V2 独立门禁与用户阅读验收待完成" | 最像产品的缺口：真实浏览器行为目前没有任何实测覆盖 |
| B4 | 中 | **文档/代码自述漂移 4 处**：`__init__.py:1`"当前仅 S0 输入扫描"、`cli.py:1`"当前仅公开 S0 inspect 入口"、`cli.py:348`"尚无模型图/HTML/CSV"、`pyproject.toml:8` description 同样陈旧 | grep `仅 S0` 与 `后续阶段` 两类字样 | 使用者读到的能力边界与实际相反 |
| B5 | 中 | **入口职责膨胀**：README 声称"`scripts/read_model_structure.py` 唯一薄用户入口"，实际 `scripts/` 有 6 个一次性 `b4_fix0X_*` 取证/冒烟脚本，且会随补修编号继续增殖 | `ls structure_reader/scripts/` | 用户/新对话无法分辨正式入口与研发残骸 |
| B6 | 中 | **分层倒挂**：`cli.py`(1,164 行) 中 `_build_documents` 单函数跨约 380 行，实际承担图装配/校验编排/落盘；`build_pipeline.py` 反而只有 400 行 | `cli.py:645`、`build_pipeline.py` | CLI 与业务编排边界名不副实，`build` 逻辑不可复用/难测 |
| B7 | 中 | **模型耦合面过宽**：7 个 `extract/dspark_*.py` 全部依赖 `rules/deepseek_v4_flash_0731` + `extract/deepseek_v4_reference` + `model_registry`（每模块 10–13 个内部 import），组件间链式复用（confidence→exit/endpoints→stage→dspark→reference），`checks/dspark_contracts.py` 2,939 行与该模型契约互绑 | §5.1 依赖图 | "唯一已注册模型"实为"唯一可实现模型"；V3 的 Qwen3-0.6B 迁移（S7）成本被显著低估 |
| B8 | 中 | **测试文件命名 = 研发流程历史，不是模块边界**：`test_b4_fix01~05_defense`（92 个方法，主题高度重叠）、`test_gate_repairs01/02`、`test_page_topology_r1/r2a/r2b` | `ls tests/` | 回归资产随轮次单调增殖（892 → 1,538，约 25s），无人去重；改一处语义要碰 5 个"补修"文件 |
| B9 | 中 | **过程体积倒挂未解决**（见 §4）：`.work/coordination` 1,079 文件 / 327,489 行 vs 产品+测试 50,396 行 ≈ 7×；`outputs/_coordination` 旧 349 文件原样留着 | §4 体量表（附录 A 命令可重算） | v4 的核心自我改进目标（§0.1）名义完成、实际反向 |
| B10 | 低 | **无工程化基础设施**：无 pytest/tox、无 ruff/mypy、无 CI、无 conftest、无 Makefile；README 快速开始要求用户手填 `~/.cache/codex-runtimes/...` 绝对解释器路径；`.claude/settings.local.json` 只白名单该一条路径 | 工作区内上述配置文件与 `.github` 目录均不存在 | 离开这台机器就跑不起来；换人/换环境即断链 |
| B11 | 低 | **产物/临时区无回收**：`outputs/` 152MB（67 候选 + 541 `_failed` + 24 扫描包）、`.work/` 39MB 持续增长；`results/` 是空目录残留 | `du -sh` | 磁盘与认知噪声；`_failed` 541 个目录淹没真正失败信号 |

---

## 7. 应保留的强项（重构时不要连婴儿带水倒掉）

1. **治理分层干净**：需求权威（v3 轴字典 / 场景清单 / §5.3 错误注入全清单）与执行流程（v4）分离，冲突优先级显式声明。少见且正确。
2. **交付不变量真的实现了**：内容寻址 `content_id` + `.work` 暂存 → `os.replace` 原子提交 + 同输入幂等复用 + 既存内容不一致**拒覆盖** + 失败隔离到 `_failed` + 固定输入 SHA 基线失配即 `unreviewed/failed`（`cli.py::_materialize/_verify_existing`）。这比任何流程文档都更有长期价值。
3. **受控错误注入是真防线**：`tests/fixtures/integration_mutants/*.json`、`markov_mutant_*.json` + `test_b4_fix0X_defense`；progress 中多次靠它推翻自己的绿灯（"60 全绿不证明缺口消失"、"测试还点不存在 norms>indexer 只判标题，覆盖断言过浅"）。
4. **证据上限被物理执行**：`pyproject.toml` `dependencies = []`，输入目录只读，不下载权重、不 import torch。
5. **诚实记录原则**：四态（pending/in_progress/passed/failed）、不写"已验证"而未验证、测试数不冒充完成度、失败记录保留、工程通过 ≠ 阅读验收、不替用户签阅读验收。
6. **IR 三件套（`graph_ir` / `graph_document` / `checks/graph_contracts`）** 零业务耦合、可独立测试，v4 已确认复用不重写 —— 这是整个工作区里最干净的模块边界。

---

## 8. 改造清单（按依赖顺序，工作区级优先）

| 步 | 动作 | 关闭问题 | 判据 |
| --- | --- | --- | --- |
| 1 | `git init`（`LLMarchi_excel` 或至少 `structure_reader`），ignore `models/ hflib/ outputs/ .work/`，跟踪 `hf_docs/` + `structure_reader/{src,tests,configs,docs,scripts}` | A2 | `git log` 有首个提交；`git status` 只剩工作产物 |
| 2 | 过程资产单向流动：门禁 `freeze_manifest/result/logs` 从 `.work` 搬到 `outputs/_coordination/vN/gate/`（照手册执行）；`.work/coordination` 1,079 文件归档为单个 `tar.zst`（仅保留 `task*.md` + 最终 result）；**禁止测试读 `.work`** | A1, B1, B9 | 删掉 `.work` 后 `python -m unittest discover` 仍全绿 |
| 3 | 新增机器可读 `status.json`（V1/V2/V3/B*/R* 状态 + 当前 candidate SHA + 测试计数 + 阅读验收状态），所有文档改为引用它；顺手修 4 处"仅 S0"自述与 README/handbook §3 快照 | A3, A5, B4 | 一处更新，处处一致；`docs_delivery` 测试断言 `status.json` |
| 4 | 开真实浏览器验收通道（用户 `file://` 打开 + 记录反馈/截图到 `outputs/_coordination/*/reading/`），并按 handbook §5.4 原样跑一次 **V2 无头独立门禁** | A6, B3 | 页面在浏览器中实测记录在案；V2 gate 有独立 result |
| 5 | 从 `structure_page.py` 拆出模板资产（`templates/structure.html` + `static/page.js` + 注入 payload），使 JS 能被独立 lint/语法检查 | B2 | renderer 只拼数据，不再内含 110KB 字符串 |
| 6 | 把 `cli._build_documents` 下沉到 `build_pipeline`，`cli` 只留 argparse + 原子落盘；`b4_*` 脚本归档 `scripts/dev_archive/` 或删除 | B5, B6 | `cli.py` 显著变薄；`build` 可被测试直接调用而不经 subprocess |
| 7 | 页面测试改为自包含：测试内 `run_build()` 到 `tempfile` 目录，取消 mtime 选最新；node 缺失统一 `skipUnless`；探针 `.cjs` 移入 `tests/probes/` | B1 | 清空 `outputs/` 后仍可完整复现全绿 |
| 8 | 按模块边界重组测试（合并 `test_b4_fix0X_defense` 为 `test_dspark_integration_defense` 等），配 `make test` / ruff / mypy + 最小 CI | B8, B10 | 测试名可映射到被测模块；CI 一条命令可复现 |
| 9 | 加产物 GC 策略（保留最近 N 个候选 + 门禁引用到的候选永久保留，其余移 `outputs/_archive`），删 `results/` 空目录 | B11 | `du -sh outputs/` 可控且审计链完整 |

**建议先做 1–3 步**：它们不触碰任何产品逻辑，风险最低，却能一次性关闭 6 个高危问题中的 4 个（A1/A2/A3/B1 的部分）。

---

## 附录 A：本次实测命令与结果

```bash
cd /Users/xiechonghang/Workspace/LLMarchi_excel

# 全量回归（2026-10-17 11:44，实测）
PYTHONDONTWRITEBYTECODE=1 \
  /Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m unittest discover -s structure_reader/tests -q
# → Ran 1538 tests in 24.960s / OK / 原始 exit 0（无管道掩盖）

# 版本控制状态
git rev-parse --show-toplevel                       # → fatal: not a git repository（工作区根亦无）
(cd hflib/transformers && git rev-parse --short HEAD) # → 536ecc0073（只读输入自带检出）

# 体量
wc -l $(find structure_reader/src -name '*.py' -not -path '*__pycache__*') | tail -1   # 26,404
wc -l $(find structure_reader/tests -name '*.py')                          | tail -1   # 23,992
grep -c "def test" structure_reader/tests/*.py | awk -F: '{s+=$2}END{print s}'          # 1,538
find structure_reader/.work/coordination -type f | wc -l                    # 1,079
find structure_reader/outputs/_coordination -type f | wc -l                 # 349
ls structure_reader/outputs/deepseek_v4_flash_0731 | wc -l                  # 67 候选
ls structure_reader/outputs/_failed | wc -l                                 # 541
du -sh models hflib structure_reader hf_docs                                # 22M / 815M / 200M / 236K

# 图规模（最新候选）
python3 -c "import json;d=json.load(open('structure_reader/outputs/deepseek_v4_flash_0731/6dede2fc214e02afd2765e18714624ad4bf75472ad5a71b08d967ffbb2723248/data/semantic_graph.json'));print(len(d['nodes']),len(d['edges']))"   # 107 173
python3 -c "import json;d=json.load(open('structure_reader/outputs/deepseek_v4_flash_0731/6dede2fc214e02afd2765e18714624ad4bf75472ad5a71b08d967ffbb2723248/data/dspark/semantic_graph.json'));print(len(d['nodes']),len(d['edges']))"                                                                                # 276 364
```

## 附录 B：未核查项（本文不声称）

- 未启动 Claude / opencode / DeepSeek 任何外部模型通道；未读取 `~/.config/opencode/opencode.jsonc` 或凭据。
- 未在真实浏览器打开任何 `structure.html`；页面 JS 行为结论仅来自仓库内测试与 progress 记录的自述。
- 未精读 `hflib/`、`models/` 内容（除确认被注册消费的 8 个输入文件与 `inference/model.py` 存在性）。
- 未运行 `inspect` / `build` 的新产物构建（只做只读核查与既有产物统计）；`progress.md` 中记录的候选 SHA 未逐一重算。
- Qwen3-0.6B 迁移输入是否存在未核查（handbook §3.1 记为"本次未找到本地 Qwen3-0.6B 配置资料"）。
