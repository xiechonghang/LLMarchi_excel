> **历史执行说明，已被2026-10-17新需求替换。** 当前接续请读[需求书](../structure_reader/docs/requirements_alignment_2026_10_17.md)、[实施计划v5](project_execution_plan_v5.md)和[progress顶部](../structure_reader/docs/progress.md)。下文旧V1/V2/V3、B4、门禁、无头/可见终端及S0快照仅保留历史，不再决定下一任务。新需求不要求先补完旧参考实现DSpark。

# 模型结构阅读工具：项目推进说明书（v4 分工修订）

更新时间：2026-10-13，Asia/Shanghai。整理者：Codex。

**当前对话后续修订：用户要求派单改为 VS Code 页面操作，在项目文件夹新建可见集成终端启动 Claude/opencode，保留中间输出并定期查看完成/错误/权限提示及收集公开回复。此要求覆盖本文下方“不得使用 VS Code 屏幕派单”和后台无头调用模板；角色、默认 Claude 配置、指定 DeepSeek 模型、V1→V3 和里程碑门禁保持不变。不得隐藏运行过程。用户已明确授权两个 CLI 当前配置模型服务处理本项目必要文档/源码/配置/测试，排除凭据、其他项目、权重。**

**新对话从本文开始：Codex 负责派单与最终核验，通过终端调用 Claude CLI 开发、opencode CLI + DeepSeek 独立测试；按 V1 → V2 → V3 的垂直切片推进到首版完成。** 保留 v3 产品需求和 v4 流程精简，不恢复逐微任务三方交接仪式。

工作区：`/Users/xiechonghang/Workspace/LLMarchi_excel`。工程根：`structure_reader/`。下文未写绝对路径的文件均相对工作区。

本文是接续执行入口和组织规则，不是项目完成报告。编写本文时仅核对文件、历史记录、本地 CLI 帮助和当前回归；未启动 Claude/DeepSeek 模型任务，未修复产品代码，未产出结构阅读页。

## 1. 执行依据与用户最新决定

冲突时按以下顺序处理：

1. 用户在当前及后续对话中的明确指令。
2. 本说明书：2026-10-13 用户调整后的分工、启动步骤、状态与衔接规则。
3. [计划 v4](model_structure_plan_v4.md)：V1/V2/V3 顺序、减少仪式、按需补基础设施。
4. [计划 v3](model_structure_plan_v3.md)：产品范围、模型事实、轴字典、场景与最终验收要求；执行顺序和协作规则由上述文件覆盖。

[development_workflow.md](development_workflow.md) 和 [v3 阶段快照](stage_reports/2026_10_13_v3_project_checkpoint.md) 保留历史证据。其中“VS Code 可见终端”“Harness GUI 测试”“先补齐 S1 汇总和 schema 才能进入 S2”等要求不再适用。v4 原先的“Claude 自行派单、Codex 退出常规流程、DeepSeek 门禁即最终验收”也被用户本次分工覆盖。

用户最新分工原意：

- 任务分发与最终核验由 Codex 执行，通过终端调用工具，不做屏幕操作。
- 代码开发由 Claude Code 执行，必须使用 Claude CLI。
- 代码测试与验证由 DeepSeek 执行，使用 opencode CLI + DeepSeek V4.1 Flash。

不调用 Codex CLI，不恢复 VS Code/DSH/Harness 的屏幕派单。Codex 桌面当前对话可以执行本机终端命令，并亲自复核结果。被分析模型是 **DeepSeek-V4-Flash-0731**；测试模型是 **deepseek-v4.1-flash**，两者不得混淆。

## 2. 产品目标与不可削减的边界

从模型配置与对应源码生成有依据、可分层展开的离线数据流阅读页。用户一条命令拿到明确的结果目录和 `viewer/structure.html`，无需手工维护图或 JSON。

### 2.1 首版交付

- DeepSeek-V4-Flash-0731 官方参考实现完整静态结构：SWA/CSA/HCA、低秩 Q/O、共享 KV、双 mHC、Hash/Top-K MoE、Compressor/Indexer、缓存/压缩状态和 DSpark。
- 每类重复结构一份代表模板；43 层层序、分组数量、实际实例参数/状态与跨层读出单独保存。
- 自包含离线页：L0 概览、L1 代表结构、L2 模块详情，轴含义/来源、源码片段、场景切换与状态说明。file:// 可读，无 CDN；信息不能只靠颜色或悬停。
- 同源图数据、实例绑定、场景、输入/运行清单；阅读、核验、源码证据、覆盖和 HF 具名差异报告。
- 六类语义 CSV：节点、连接、轴、层实例、源码证据、检查结果；xlsx 可选。支持代表/全实例投影，表格不另行维护事实。
- 固定版本的 Qwen3-0.6B 迁移案例及 DeepSeek 回归，最终将已实现命令固化为项目 Skill。

### 2.2 全程底线

- 证据上限为源码核对与配置/场景推导；不导入/执行模型参考程序，不声称推理、运行追踪或数值等价已验证，不下载全量权重。
- `unknown` 有原因，不能当作 false。P>0 不自动证明有效 cache 存在；历史起点与连续调用假设显式保存。
- 每条轴区分张量名、轴名、符号、值/表达式、中文含义和来源；逻辑/物理 shape、布局、dtype、全局/本 rank 含义分开。首版 world_size=1。
- 通用引用检查通过不等于模型连接正确；源码行号存在不等于语义核对通过。固定输入基线失配时不得继续发布成功页。
- `inspect → classify → extract → validate → render → export`；先核验后出图。失败写入 `_failed`，不污染正式成功目录。
- 展开是渲染策略：折叠仅隐藏/分组呈现，不从图数据删除边，不将通用折叠等价证明重新设置为 V1 前置。
- 机器契约和受控错误是长期防线；独立期望不能从新生成输出自动刷新成“正确答案”。

### 2.3 模型核验重点

- 主干分组 `swa_hash=2`、`csa_hash=1`、`hca_topk=20`、`csa_topk=20`，并集为 0–42，无重叠。40/41/42 的 DSpark 外部读出差异保留。
- 概览为 Embedding → HC Expand → 43 层 → HC Head → Final RMSNorm → LM Head，并保留 DSpark 跨层读出。
- Block 两次 `hc_pre → norm → module → hc_post`。comb 方向是 `sum_j comb[...,j,k] * residual[...,j,d]`；不能凭长度同为 4 的 shape 判断方向。
- 共享 KV 单投影及同张量 K/V 别名；qr→Indexer；Indexer 产选择索引而非 Value；CSA 学习式 Indexer 与独立状态，HCA 无学习式 Indexer。
- inverse RoPE 先于分组低秩 O 投影；保留 sink、部分 RoPE、量化模拟和不透明 kernel 边界。
- Hash 依赖 token ID；Top-K 路由、权重、token 分派/汇合和 shared expert 可见；真实专家选择及每专家 token 数保持未知。
- DSpark 使用公共计算模板和首尾角色绑定，3 stages 不能画成三个完整同构实例；46 项 compress_ratios 的前 43 项属主干、尾 3 项属草稿。参考配置 3 stages 与根配置 nextn=1 分开解释。
- HF 对照无对应完整 DSpark 支持，不将其标成完整 0731 图。

## 3. 本次实际核对的工程状态

这是 2026-10-13 的快照，新对话先比较文件变化，不把本文的测试数作为永远有效的绿灯。

### 3.1 工作区职责与现有输入

- `hf_docs/`：计划、方法说明、历史记录和本文。
- `models/`：V4-Flash-0731 与 V4.1-Flash 原始资料；本期图分析消费前者。后者的存在不意味着本工程支持它。
- `hflib/`：第三方源码输入，包括 Transformers/Diffusers；静态分析不要求将其安装为运行依赖。
- `scripts/download_hf_model.py`：已有工作区下载工具，不是结构阅读工程入口。
- `structure_reader/`：唯一开发根。产品在 src/configs/docs/tests，入口在 scripts，正式产物在 outputs，临时内容在 .work。
- 工作区根当前不是 Git 仓库，不能假定可用根目录 git diff/status 或自动创建 worktree。第三方 `hflib/transformers` 有自己的 Git 检出；本次 HEAD 为 `536ecc007387a50e77603bb5d92100e9b07514cc`，不得改动输入仓库来修工程。
- 本次未找到本地 Qwen3-0.6B 配置资料；HF 的 `models/qwen3/` 源码已存在。V3 仍需准备并固定具体 Qwen 配置版本，不把已有 HF 类误当完整第二模型输入。

### 3.2 可以继承的资产

- S0 已正式批准：注册、输入扫描、raw/effective config 与来源、固定基线、inspect CLI、环境/权重能力报告、内容寻址扫描包。
- S0 历史证据：Harness 137/137；Codex 28 项及 137 项回归。批准只覆盖输入扫描基础。
- `graph_ir.py`：轴、端口、张量、节点、四类边、guard、别名、状态版本和折叠表示。
- `graph_document.py`：SemanticGraph、规范化序列化/反序列化；当前图文档只有 template_id，真实实例绑定和结构化证据仍需补。
- `checks/graph_contracts.py`：通用引用、方向、状态环和诊断检查。当前 SHA 与旧 853 项通过快照一致（见第 10 节），直接复用。
- 旧 S1 报告曾因缺正式汇总而 pending；不为形式收尾重启旧闭环。事实与失败记录保留，工程正确性在新 V 门禁中验证。

历史依据：

- [S0 最终审核](../structure_reader/outputs/_coordination/s0/2026_10_12_round_02/audit/final_acceptance.md)。
- [S1 独立复验及历史失败](../structure_reader/outputs/_coordination/s1/2026_10_12_round_01/results/verification_01b_repair_rerun.md)。
- [v3 状态快照](stage_reports/2026_10_13_v3_project_checkpoint.md)。

### 3.3 比原 v4 盘点更新的实际文件

以下文件现在均存在，不应重复创建或从零重写：

- `src/structure_reader/scenario_evaluator.py`。
- `src/structure_reader/rules/deepseek_v4_flash_0731.py`。
- `src/structure_reader/rules/__init__.py`。
- `tests/test_scenario_evaluator.py`、`tests/test_rules_deepseek.py`。

它们属于已落盘准备工作，尚未通过当前全量检查，更没有独立 DeepSeek 门禁结论。

### 3.4 当前测试真实结果及首批问题

本次 Codex 为状态核对运行了一次现有全量 unittest（命令见第 5 节）：Python 3.12.14，**892 项，1 failure、2 errors，退出码 1**。这是本地状态核验，不冒称 DeepSeek 独立测试，也不是 V1 最终验收。

三个实际红项：

1. `test_rules_deepseek.TestLayerGroups.test_unknown_ratio_rejected`：`derive_layer_groups()` 在 `notes[(att, gate)]` 抛裸 `KeyError: ('swa', 'topk')`。测试构造的未知 ratio 位于尾部，先引入了未支持主干组合；修复时分别检验未知主干 ratio、未知组合与尾部 stage 配置，不要只吞掉异常。
2. `test_scenario_evaluator.TestEvaluateTri.test_bad_syntax_unknown`：测试调用 `evaluate_tri("import os")` 缺必填 env，抛 TypeError。由 Claude 明确 API 契约并修正实现/测试的一致性；DeepSeek 按契约独立核对。
3. `test_scenario_evaluator.TestWhitelistRejections.test_reject_excess_depth`：30 层括号未触发异常。括号不形成对应 AST 深度；要测试真实深 AST，若需要文本长度/括号上限则显式定义，不能通过弱化要求凑绿。

另有阅读代码发现的状态问题：`derive_layer_groups()` 当前将 `csa_topk` 写成 `template_status="extracted"`，但没有提取器和真实图。开始 V1 时纠正为真实未提取状态；构建成功之后才能标 extracted。

后续需针对真实使用补查求值器：环境变量类型、布尔与整数边界、整数规模、min/max 参数、三态布尔组合和非法表达式处理。当前代码有 AST 深度/节点数限制，但不能照抄 v4“防大”自述为整数规模限制已交付。这些是待核查项，不是本次全部实测缺陷。

### 3.5 尚未交付

本次实际 CLI `--help` 只有 inspect。未找到产品提取器、模型专用契约、build 编排、render、export、layer_plan 独立模块、完整来源证据模块/schema、Qwen 适配或项目 Skill。未找到正式模型 `structure.html`、`semantic_graph.json`、`connections.csv`。

原 v4 的“新模块未测试且缺 rules/__init__.py”已过时；旧 progress/README 的执行链也滞后。当前应记：**S0 历史 passed；V1 pending（准备代码已有，当前基线红）；V2/V3 pending；尚无用户可读交付。** 不因本文完成而标记 V1 已启动或通过。

## 4. 分工与精简执行闭环

### 4.1 Codex：派单、协调、最终核验

- 在本对话控制推进顺序、需求边界、依赖和实际状态。
- 通过终端启动 Claude CLI，把一个能形成可检查成果的工作包交给 Claude；按接口/交付边界拆，不按每个 IR 字段拆单。
- 每个里程碑完成时通过 opencode 启动指定 DeepSeek，提供自包含门禁任务；读取其结果，将缺陷交回 Claude 修复。
- 最终亲自读实际文件、复现核心命令、检查图/报告/表格和页面行为，核对门禁对应的源码版本，裁定工程状态。
- 更新简短进度和里程碑验收结论；不替用户勾选阅读体验。Codex 本地复现属于最终核验，不替代 DeepSeek 的独立测试职责。
- 常规产品开发不由 Codex 接手；外部通道失败时保留结果和断点，恢复指定通道，不暗中改模型/改开发者。

### 4.2 Claude Code CLI：产品开发与修复

- 对派发工作包实现产品代码、配套开发自测和必要文档。
- 复用现有 IR、扫描和基线逻辑；按实际模型内容补接口，避免为了完整目录而创建空包/占位产物。
- 返回实际改动文件、运行命令/退出码、成果路径、未覆盖项和明确下一断点。
- 根据 DeepSeek/Codex 的具名缺陷修复；不能自己将里程碑标 passed。

### 4.3 DeepSeek 经 opencode：独立测试与验证

- 指定 `huoshanai/deepseek-v4.1-flash`，不自动换模型。
- 独立阅读需求、源模型依据和工程实现；执行测试及真实 build，对关键连接/场景/状态建立独立期望。
- 可新增/修正独立测试与 fixture；产品源码问题交给 Claude，不直接修产品代码。不能以删除红测、降低期望或复制生成结果来使门禁通过。
- 返回命令、实际退出码、缺陷定位、证据、未覆盖和门禁结论。程序退出 0、模型回复“通过”和工程实测通过分别核对。
- 每个 V 里程碑一次完整独立门禁；发现缺陷后重测受影响范围，核心/共享接口修改再做必要全量回归。不开逐小单独立门禁。

### 4.4 用户：范围与阅读体验

- 用户负责需求调整和产品阅读反馈，不负责工具中转、环境恢复或工程排错。
- V1 提供首次实际页面供阅读；V2/V3 继续交付可读成果，但不为每个小任务要求用户确认。
- 工程 passed 与阅读 pending 分别记录。等待 V1 阅读反馈时可以继续不依赖反馈的 V2 工作；不能将沉默当作阅读通过。首版最终报告须明确尚待反馈项。

### 4.5 每个工作包如何闭环

`Codex 定义工作包 → Claude 实现并自测 → Codex 查看实际成果 → 到 V 门禁时 DeepSeek 独立测试 → Claude 修缺陷 → Codex 最终核验并更新进度`。

每包只需要一个简短任务文本和可恢复输出；放 `.work/coordination/vN/<package>/`。不新建逐包 dispatch/results/audit 三件套，不强制补旧开发汇总，不反复用全量回归证明未变化部分。正式长期记录只保留 progress、每 V 门禁、最终核验结论及必要阶段快照。

同一时刻产品文件只有一个开发执行者。DeepSeek 门禁时冻结待测产品修改；用相关源码/测试 SHA256 对应版本，完成后再交给 Claude。没有根 Git 时用文件哈希和必要的局部备份即可，不创建整套重复工作区。

## 5. 终端通道与可执行命令

### 5.1 本次已核对的本机条件

- Claude CLI：`/opt/homebrew/bin/claude`，版本 `2.1.287`。
- opencode CLI：`/Users/xiechonghang/.opencode/bin/opencode`，版本 `1.18.4`。
- 工程 Python：`/Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`，版本 `3.12.14`。
- 系统 `python3` 指向 `/usr/bin/python3`，不要把它自动当作工程解释器。
- opencode 配置位置在原 v4 中记录为 `~/.config/opencode/opencode.jsonc`；历史 v4 记载指定模型冒烟通过。本次仅检查版本/help，**未重测远程连通或认证，未读取配置凭据**。

新对话先确认上述路径仍可用；本地帮助优先于过时示例。不要升级 CLI、更改账户配置或安装模型推理依赖来推进静态工程。

### 5.2 本地基线核对

```bash
cd /Users/xiechonghang/Workspace/LLMarchi_excel
READER_PYTHON=/Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3

PYTHONDONTWRITEBYTECODE=1 "$READER_PYTHON" -m unittest discover -s structure_reader/tests -q
"$READER_PYTHON" structure_reader/scripts/read_model_structure.py --help
```

第一条当前预期是第 3.4 节的红项，修复后应全绿；新对话以实际结果为准，不用历史 853 强制匹配当前计数。完整 build/validate/render/export 命令尚未实现，不能用计划示例冒充当前命令。

### 5.3 Claude 开发调用模板

以下是未来工作包调用形式，本文编写时未执行。先写实际 `task.md`，明确模型、需求、边界、成果、验证命令和停止条件，然后在工作区调用：

```bash
cd /Users/xiechonghang/Workspace/LLMarchi_excel
READER_TASK_DIR=structure_reader/.work/coordination/v1/bootstrap
mkdir -p "$READER_TASK_DIR"
# Codex 先把具体任务写到 "$READER_TASK_DIR/task.md"。
/opt/homebrew/bin/claude -p --permission-mode acceptEdits --output-format json \
  < "$READER_TASK_DIR/task.md" \
  > "$READER_TASK_DIR/claude_result.json" \
  2> "$READER_TASK_DIR/claude_stderr.log"
READER_CLAUDE_EXIT=$?
```

`acceptEdits` 允许工作区编辑，但不保证所有命令自动获准。Codex 根据本机支持的允许工具选项，只放行本任务必要的读取、编辑和具名测试命令；遇到工具拒绝按实际原因处理。不默认使用跳过权限参数，不为便利改全局权限。保留默认 Claude 模型，不擅自覆盖用户配置。

实际 orchestration 用可轮询的终端进程，不必等待长命令阻塞整段对话；定期向用户说明真实进展。记录本次实际会话 ID，续接用本次 ID，不盲目 `--continue` 其他项目会话。没有会话恢复能力时用简短断点任务重启。

### 5.4 DeepSeek 门禁调用模板

```bash
cd /Users/xiechonghang/Workspace/LLMarchi_excel
READER_GATE_DIR=structure_reader/outputs/_coordination/v1/gate
mkdir -p "$READER_GATE_DIR"
# Codex 先写好 "$READER_GATE_DIR/task.md"。
/Users/xiechonghang/.opencode/bin/opencode run \
  --dir /Users/xiechonghang/Workspace/LLMarchi_excel \
  --model huoshanai/deepseek-v4.1-flash \
  --file "$READER_GATE_DIR/task.md" -- \
  "读取附件，按门禁单独立验证本项目。允许维护独立测试，不修改产品源码。报告真实命令、退出码、缺陷和未覆盖范围。" \
  > "$READER_GATE_DIR/deepseek_result.txt" \
  2> "$READER_GATE_DIR/deepseek_stderr.log"
READER_DEEPSEEK_EXIT=$?
```

使用已有本项目工作区权限；不默认加 `--auto`。若配置阻止必要本地测试，先检查实际拒绝原因，再按已有任务授权限定项目路径/命令处理。不得读取或转发 API key、账户配置、其他项目或模型权重。用户本次明确指定两个远程模型通道用于本项目开发/测试；不重复请求同范围普通派单许可。

不启用 `--thinking`，正式日志只保留任务、公开结果和执行证据；排除凭据与模型内部思考。CLI 进程退出码和工具内部命令退出码都要记录；若进程成功但测试未实际执行，门禁不能通过。本地帮助确认该模型选择参数可用，不代表远端保证在线。

### 5.5 工具失败恢复

超时、502、传输错误：确认进程是否仍运行、文件是否变化，保留已完成工作；对临时故障有限重试（最多两次），用明确会话/断点恢复。认证、模型不可用或权限拒绝不盲目重试，也不切换到 GUI 或另一个模型冒充结果。可继续不依赖失败通道的只读核对和文档整理；不可宣布阶段通过。若自动审批明确拒绝，说明具体动作与理由，只有必要时才向用户提出实际阻断项。

## 6. V1：先交付 CSA·TopK 代表层闭环

目标：一条 build 命令生成真实、已核验、可离线打开的代表层页面。V1 不等待完整通用 schema/AST 断言引擎/折叠证明，不提取其余三类内部图或 DSpark 子图。

### 工作包 A：现有准备模块收敛（第一个派单）

Claude 读取现有模块和测试，修复第 3.4 节三项红测、虚假的 extracted 状态，补与实际使用相关的求值边界；复用输入基线。Codex 本地复现红转绿和文件变化后直接进入建图，不增加独立小门禁或历史补报告任务。

交付：现有回归全绿、真实状态、明确 API 和仍未覆盖项。不能借此宣布 V1/S1 完整通过。

### 工作包 B：真实代表图与模型核验

主要文件：`extract/deepseek_v4_reference.py`、`checks/deepseek_v4_contracts.py`，按需要补 `layer_plan.py`、来源证据表示与 tests/configs。

- 代表 layer 4：双 mHC、Attention 低秩 Q/共享 KV/窗口、overlap Compressor 及前后状态、含 qr 输入的 Indexer/Top-K select/独立缓存、MoE Gate/专家模板/shared expert。
- 真实 layer_plan 四组与全部 43 层实例绑定；40/41/42 读出只记录外部关系，DSpark 内部 V2 实现。
- 每个节点/边有对应源码/config 依据；证据先采用固定基线 + qualified symbol/行号 + 原文片段/表达式 + 核对结论。无需通用 Python 理解器，但必须区别 located/source_checked/derived/uncovered。
- 通用引用检查 + 模型语义契约；至少 8 项有意义的受控错误，如 comb 反向、删 qr 边、拆 KV 别名、逆 RoPE/O 顺序交换、错误压缩 guard、状态版本丢失、轴/布局错误、层分区缺失。

### 工作包 C：可打开的完整交付

主要文件：`build_pipeline.py`、render 及 assets、`export/semantic_tables.py`、CLI、用户文档和真实产物。

- build 连通 classify/extract/validate/render/export，失败不落正式页。
- L0 层序/分组与整体范围；未提取变体显示“未覆盖”，不伪装为已有完整结构。
- L1 代表层可展开；轴详情、嵌入源码片段和行号、guard 三态/场景切换；图/表同源。
- V1 两类 CSV：connections、tensor_axes；图、层规划、实例、场景、证据/核验/覆盖报告及结果导航齐备。
- 从 V1 起就检查固定输入基线、规范化图 ID、失败隔离和输入版本区分。优先复用 inspect 的内容寻址/原子提交方法，V3 再完善全套可重复交付验收；不得先用容易覆盖不同输入的固定目录凑首屏。

### V1 场景与门禁

首批沿 v4：prefill T=3/4/5/127/128/129，decode T=1、P=2/127/128，默认显示 P=128。进入 V1 门禁时补齐代表 CSA 所需 P=3/4；说明历史 cache 前提，未给前提时状态未知。HCA 的 P=126 和 DSpark 场景随 V2 补齐。

Indexer C=0、少于 Ki、达到/超过 Ki 的有效选择与因果可见性需独立测试；容量不足以实现的场景不要虚构成功路径，记录适用参数/容量前提。不能将槽位数量或 Ki 上限直接当作每 query 的有效选择数量。

DeepSeek 执行 V1 独立门禁后，Codex 复现真实 build、查看代表图及 CSV、检查 file:// 离线阅读/场景切换/轴与源码、确认失败路径。界面验收可使用本地浏览器或工具检查，但任务派发始终用 CLI。最后返回实际 HTML 路径供用户阅读；体验状态 pending，等待反馈，不代签。

计划 build 入口（实现后才可用）：

```bash
"$READER_PYTHON" structure_reader/scripts/read_model_structure.py build \
  --model deepseek_v4_flash_0731 --scenario-set standard
```

## 7. V2：完整 DeepSeek 变体、DSpark 与阅读体验

Claude 复用 V1 模板/接口，补 SWA·Hash、CSA·Hash、HCA·TopK。契约检查 HCA 无学习式 Indexer、Hash token-ID 路由、变体参数与真实实例绑定；不能只换图标题复用错误拓扑。

补 DSpark 公共 stage、首入口/末出口、共享参数、40/41/42 读出、Markov/confidence、配置消费方和 prefill/decode 差异。角色首尾可以同时成立，以兼容单 stage 表示。

补 v3 完整标准场景：prefill 3/4/5/127/128/129；decode P=2/3/4、126/127/128；Indexer 可见性与数量边界；DSpark prefill 建缓存和 decode 草稿。区分 capacity、valid_length、remainder、overlap、窗口槽位与 Indexer 状态。

完成 L2 模块展开、状态面板、源码/轴全名可见性、阅读指南和 HF 具名差异报告。对 source/config hash 与来源锚点保持失配拒绝；JSON Schema/通用 AST 断言引擎继续按实际需要实现，不重新形成独立前置长阶段。

受控错误覆盖 v3 §5.3 全清单。DeepSeek 门禁需验证真实全模型产物与场景，Codex 最终核对：全部目标变体/DSpark 关系可见、共享模板与实例差异正确、展开不丢边、HF 差异诚实、页面可读。按用户 V1 反馈修阅读问题，保留未收到反馈状态。

## 8. V3：表格齐套、迁移、Skill 与首版结束

- 六类 CSV 与可选 xlsx；代表/全实例投影、唯一 ID、联查一致、图表同源。缺 xlsx 依赖时 CSV 可交付，并给明确降级说明。
- 完善 analysis_id 对输入、规则、场景、工具/模板版本的内容寻址；同输入稳定复用、不同版本隔离、冲突检测、完整 manifest 和产物清单；失败不覆盖成功结果。
- 准备 Qwen3-0.6B 配置及对应 HF 源码基线，固定 revision/hash；表达独立 Q/K/V、GQA、SwiGLU、普通残差与 cache；完成第二模型完整交付并回归 DeepSeek。只下载必要配置/源码，不下载权重。
- 工程入口稳定后，Claude 按技能创建规则编写 `structure_reader/skills/model-structure-reader/SKILL.md` 及 reading_format。Skill 只编排已实现命令，不复制提取逻辑；个人技能目录安装另行处理。
- 实现/核对 build、validate、render、export-tables 的最终接口及重新验证约束；工程 README、user_guide、developer_guide、output_reference 与命令一致。
- DeepSeek 独立门禁：两个模型真实 build、关键语义、全部必要回归、未知模型/输入基线失配/缺依赖/重复运行/失败隔离等路径。
- Codex 最终复现两个成功交付、核对数据与页面、确认失败/降级行为，给出首版范围与限制。首版工程通过与用户阅读验收分别报告，未反馈不能称“全部验收完成”。

首版不额外扩展完整第二套 DeepSeek HF 图、ATen/ONNX、运行追踪、CUDA 数值验证、多模态/SSM 或通用任意模型支持。工作量采用实际包完成情况估计，不将原 v4 的“3–5 个工作会话”当交付承诺。

## 9. 记录、产物与完成状态

### 9.1 只维护必要入口

- 本说明书：组织规则和启动方法；范围/分工改变时更新。
- `structure_reader/docs/progress.md`：唯一滚动进度。以 V1/V2/V3 为执行状态，S0–S8 仅保留历史对照。每次记录当前包、实际文件、命令/退出码、结果路径、缺陷/断点、下一动作。
- `outputs/_coordination/vN/gate/`：每 V 一份门禁任务、DeepSeek 结果、测试日志、受检文件哈希及 Codex 最终核验结论。复验可追加具名 attempt，不复制成一整套新仪式。
- `hf_docs/stage_reports/`：每 V 完成后至多一份简短阶段快照；旧报告不改写成新通过证据。
- `.work/coordination/`：可恢复任务/调用输出，不作语义图输入或独立 golden。

门禁单最低内容：目标与 v3/v4 条款、受检范围和源输入、文件/哈希、准确 Python 和命令、成功产物/失败预期、允许修改独立测试而禁止改产品源码、证据边界、报告位置。无需把几百份旧协调文档全部转给模型。

状态四态 pending/in_progress/passed/failed；工程结论由 Codex 给出。产品验证报告的 success 是技术检查结果，不等于用户体验通过。遇到外部调用失败记录 tool_error/未执行事实，不冒称代码测试失败或门禁通过。

### 9.2 正式模型交付目录

```text
structure_reader/outputs/<model_key>/<analysis_id>/
  README.md
  run_manifest.json
  viewer/structure.html
  data/input_manifest.json
  data/layer_plan.json
  data/semantic_graph.json
  data/instance_bindings.json
  data/scenarios.json
  reports/reading_guide.md
  reports/validation_report.json
  reports/source_evidence.json
  reports/coverage_report.json
  reports/hf_implementation_diff.md      # DeepSeek 适用
  tables/semantic_nodes.csv
  tables/connections.csv
  tables/tensor_axes.csv
  tables/layer_instances.csv
  tables/source_evidence.csv
  tables/checks.csv
  tables/semantic_tables.xlsx           # 可选
  logs/build.log
```

V1 只交约定子集并明确未覆盖，V3 齐套；不创建空文件凑完整树。失败放 `outputs/_failed/<run_id>/`，不产生看似成功的正式阅读页。保留历史协调/模型分析，不自动清理用户输入或历史成果。

## 10. 当前文件指纹与测试证据

本次 2026-10-13 实际计算的 SHA256（用于比较快照，不是永久批准基线）：

```text
src/structure_reader/checks/graph_contracts.py
2bf288741fc061695ef1e8f2811a810fdf1807fa84cf7d9a44264487361cbc39
src/structure_reader/scenario_evaluator.py
55e5cd9e091923a521f0acd8f755beebd1ac324472153bbfcc240abad8626bac
src/structure_reader/rules/deepseek_v4_flash_0731.py
4af7b358544ad216c075833c4ff4580d6ed30179b75365d82c00ddb66929e291
tests/test_scenario_evaluator.py
180284e9335088e853488d21f099daa8d318b724e9c78bea5f464e9b27549da6
tests/test_rules_deepseek.py
5cbe32cc3ea941635d1ba172d32d542e16ac8d5ef6400ced66713f0c71f372a6
```

上述路径相对 `structure_reader/`。本次原始 unittest 结果摘要：

```text
Ran 892 tests in 4.919s
FAILED (failures=1, errors=2)
exit_code=1
KeyError: ('swa', 'topk')
TypeError: evaluate_tri() missing 1 required positional argument: 'env'
AssertionError: ExpressionError not raised
```

尚无本轮 Claude 开发日志或 opencode 独立验证记录，不得凭历史可用性写成已执行。首个动作是第 6 节工作包 A，再建真实图和可读页面。

## 11. 新对话接续步骤及启动提示

新对话执行顺序：

1. 读本文、v4、v3；读取 progress 和本文列出的关键代码。只在遇到具体问题时打开相关旧审核，不全量回放历史仪式。
2. 核对当前文件/CLI/Python 和实际变化；确认是否已有后续任务正在运行，避免重复编辑；文件有变化时重建快照事实。
3. 简短说明当前状态，然后通过 Claude CLI 派工作包 A。修复后进入 B、C，按 V1→V2→V3 推进。
4. 每 V 调 opencode + 指定 DeepSeek 做独立门禁，缺陷回 Claude 修复，Codex 最终核验。用户可读产物及时交付，不等所有版本做完才展示。
5. 跨对话/中断前更新 progress 中的实际文件、会话/进程、测试结果、产物路径和下一包；不留“继续上一任务”这种无具体断点的记录。

可直接复制给新对话：

> 请读取 `/Users/xiechonghang/Workspace/LLMarchi_excel/hf_docs/project_execution_handbook_v4.md`，并按其列出的 v3 产品需求、v4 垂直切片和最新分工继续推进本项目到首版完成。你负责通过终端派单与最终核验；产品开发使用 Claude CLI；独立测试使用 opencode CLI + `huoshanai/deepseek-v4.1-flash`。不要用 VS Code/DSH/Harness 屏幕操作派单，也不要调用 Codex CLI。先核对最新实际状态，从说明书的第一个未完成工作包接续，不重写已有 IR、不补旧逐单报告仪式。允许发送本项目必要源码、配置和测试给上述开发/测试通道，不包含凭据、其他项目或权重。逐里程碑交付实际离线阅读页并记录阅读验收待反馈；持续修复到工程检查和独立门禁通过，最终由你核验。

启动后不需要再次询问是否开始普通开发/派单。若用户后续改范围、要求暂停或实际出现不可自行解决的权限/通道阻断，按新指令处理并说明具体原因。
