# 模型结构阅读项目：v3 阶段性计划、进度与后续安排

整理日期：2026-10-13，Asia/Shanghai。整理者：Codex。

**当前总状态：S0已正式通过；S1开发中、尚未正式通过；S2–S8未开始。当前工程仍不能生成供用户阅读的DeepSeek模型结构页。**

本文件按本次实际读取的代码、报告和日志整理。原计划不改版、不扩大范围；开发交付、快照测试通过、阶段最终验收分别记录。后续恢复工作可先读本文件，再进入具体任务与证据。

## 1. 原本项目计划与用户目标

正式执行依据为[项目计划 v3](../model_structure_plan_v3.md)。[v1](../archive/model_structure_plan_v1.md)、[v2](../archive/model_structure_plan_v2.md)和[v2审查记录](../archive/model_structure_plan_v2_review.md)保留历史；后续按v3推进。

目标是从模型配置与对应源码建立有证据的语义数据流图，让阅读者快速理解关键模块、连接与条件，而不必依靠繁杂ONNX低层计算或单独逐行读源码。

用户要求对应的首版范围：

- 主案例：**DeepSeek-V4-Flash-0731官方参考实现**。覆盖SWA/CSA/HCA、低秩Q/O、共享KV、mHC、Hash/Top-K MoE、Compressor/Indexer、缓存状态及DSpark；HF实现交付具名差异报告，完整第二套HF图为扩展。
- 轴标签同时包含数值、参数名、含义和来源，例如 `hidden_states [batch_size=1, seq_len=1, hc_streams=4, hidden_size=4096]`。张量名与维度参数名分别保存；未知值说明原因。
- 概览保留层序、数量与实例绑定，重复结构用代表模板；模块可展开，不能丢失旁路、状态边与端口连接。
- 同一语义图生成离线页面、连接表和证据报告，避免图与表各自维护事实。
- 工程承担解析、构建、核验和导出；Skill在工程命令稳定后固化流程。
- 核验必须在正式出图前执行；证据上限为源码核对与配置/场景推导，不伪称已经运行模型或验证数值等价。

首版最终交付：离线 `structure.html`、统一图数据、层规划/实例绑定/场景、阅读和核验报告、六类CSV及可选xlsx、Qwen3-0.6B迁移案例、可重复使用的Skill。

固定流程：`inspect → classify → extract → validate → render → export`。原v3粗估约**12–20人日**，属于工作量估计，不是交付日期承诺；S3完成后按实际成本重估。

## 2. 原定S0–S8阶段与节点

以下文件为原计划的主要预期位置，除第3节明确列为已实现的文件外，均不能视为已交付。产品路径以 `structure_reader/` 为根。

### S0：工程布局、入口与输入基线（原估1–2人日）

- **任务**：工程布局、模型注册、输入清单、原始/有效配置及来源、解释器/依赖探测、固定审查基线。
- **主要文件**：README/pyproject/.gitignore；`scripts/read_model_structure.py`；`cli.py`、`inventory.py`、`model_registry.py`；模型注册/基线JSON；用户、开发、输出和进度文档。
- **产出**：可用inspect命令、输入和能力报告、清楚的目录与使用入口。
- **验收**：真实检出缺权重，配置消费方和默认值有来源，基线可比对，帮助只列实际命令，临时/产物正确落位。
- **当前状态**：passed，Codex已批准。

### S1：IR、轴、条件与通用核验（原估2–3人日）

- **任务**：统一图契约、结构化轴、Guard三态、状态版本/别名、来源和模板/实例表示、有限表达式求值与通用检查。
- **主要文件**：`graph_ir.py`、图文档、`scenario_evaluator.py`、`source_evidence.py`、`schemas/semantic_graph.schema.json`、图/源码检查器及独立测试。
- **产出**：可序列化、可核验的语义图；错误精确到节点、边、引用或字段。
- **验收**：缺端口/轴来源、错误版本及表达式失配能检出；unknown不当false；同输入稳定；同数值不同语义轴不混淆。
- **当前状态**：in_progress。局部IR与检查器有进展，完整S1未验收。

### S2：主干分类与实例绑定（原估1人日）

- **任务/依赖**：S1通过后生成四类主干分组、配置消费声明、层/stage实例身份、读出及DSpark角色、结构等价检查。
- **主要文件**：`layer_plan.py`、DeepSeek rules、`checks/layer_equivalence.py`、层分区测试。
- **产出**：layer_plan、instance_bindings与等价报告。
- **验收目标**：2/1/20/20合计43主干层且无重叠缺失；DSpark不混入主干；参数/状态ID唯一；40/41/42读出差异保留。以上是待验证目标，尚非当前提取成果。
- **当前状态**：pending。

### S3：CSA·TopK代表闭环（原估2–3人日）

- **任务/依赖**：S2后完成代表层4的双mHC、Attention、Compressor、Indexer、MoE与源码连接依据，接入validate和最小阅读页。
- **主要文件**：`extract/deepseek_v4_reference.py`、`checks/deepseek_v4_contracts.py`、最小renderer/assets、结构测试与独立fixture。
- **产出**：首个真实模型analysis目录、语义图、证据/核验/覆盖报告和可打开的代表层页面。
- **验收**：Q/KV并行、qr→Indexer、选择/状态边、inverse RoPE与O顺序有依据；错误方向、删边、交换顺序被检出；压缩边界场景差异可见；失败不发布正式页面。
- **当前状态**：pending；这是第一次实际阅读效果里程碑。

### S4：其余主干变体与完整DSpark（原估2–4人日）

- **任务/依赖**：S3后补SWA·Hash、CSA·Hash、HCA·TopK、DSpark首尾适配/主干读出/Markov与confidence/共享参数及HF差异报告。
- **主要文件**：完整DeepSeek rules/extractor/contracts、scenarios、结构测试/fixtures、`export/reading_reports.py`。
- **产出**：主案例全部目标特征的源码语义图、实现差异与未覆盖范围。
- **验收**：HCA无学习式Indexer；Hash的token ID依赖可见；DSpark首尾和目标层连接完整；prefill不画不存在的完整草稿路径；46项配置不误画为46主干层。
- **当前状态**：pending。

### S5：完整阅读体验（原估1–2人日）

- **任务/依赖**：S3后可准备布局，完整验收依赖S4；实现概览/分组/实例切换、L1/L2展开、轴/源码/场景/状态面板和阅读指南。
- **主要文件**：renderer/assets、HTML阅读测试、user_guide/output_reference。
- **产出**：正式自包含离线页面与阅读说明。
- **验收**：file://可读、不依赖CDN；无裸数字shape；全名/含义不靠悬停；展开连接守恒；阅读者能回答CSA选择来源、缓存边界、mHC流向、层2/4路由、Q形状和DSpark首尾差异。
- **当前状态**：pending；用户阅读反馈单独记录。

### S6：同源表格与可重复交付（原估1–2人日）

- **任务/依赖**：S4/S5后导出六类CSV及可选xlsx，提供代表/全实例投影，补受控错误、目录隔离与幂等测试。
- **主要文件**：`export/semantic_tables.py`、表格及contracts回归、输出清单与使用文档。
- **产出**：同源tables、可重复analysis目录、完整核验和覆盖报告。
- **验收**：按ID联查、实例ID唯一、表/图一致、错误被捕获、同输入重跑稳定、不同版本隔离、失败不污染成功目录。
- **当前状态**：pending。

### S7：Qwen3-0.6B迁移回归（原估1–2人日）

- **任务/依赖**：S6后固定Qwen配置revision/HF源码版本，复用流程表达独立Q/K/V、GQA、SwiGLU、残差与cache，运行DeepSeek回归。
- **主要文件**：Qwen注册/scenario/基线、`rules/qwen3_0_6b.py`、`extract/qwen3_hf.py`、模型回归fixture。
- **产出**：第二模型的完整静态交付包和实际支持矩阵。
- **验收**：公用IR不靠DeepSeek特例，GQA映射正确，层数按固定config，DeepSeek不回退，支持范围不外推到未测模型。
- **当前状态**：pending。

### S8：Skill固化与首版发布（原估0.5–1人日）

- **任务/依赖**：S7及工程命令稳定后编写Skill，固化模型/版本/场景选择、build/validate、阅读与交付步骤。
- **主要文件**：`skills/model-structure-reader/SKILL.md`、reading_format及完整工程文档。
- **产出**：两模型可重复使用的工程与Skill。
- **验收**：两模型各完成成功生成，未知模型/基线失配/缺xlsx依赖的失败或降级正确；不跳过核验；用户不手工改JSON也能使用。
- **当前状态**：pending。Skill尚未创建或安装。

原里程碑：**S0–S3单个代表层闭环 → S4–S6完整DeepSeek页面与表格 → S7–S8迁移验证与Skill首版**。

## 3. 当前实际整体进度

### 3.1 S0正式完成的范围

注册、inspect入口、配置消费/来源记录、原始输入哈希、固定基线、环境/权重实况与工程使用文档已交付。

- Harness最终137/137回归通过；Codex核心/产物28项及137项本地回归通过。
- [S0最终审核](../../structure_reader/outputs/_coordination/s0/2026_10_12_round_02/audit/final_acceptance.md)正式批准该阶段。
- 批准范围是输入扫描与工程基础；未批准模型提取、页面或推理能力。

### 3.2 S1已落盘的代码能力

- [graph_ir.py](../../structure_reader/src/structure_reader/graph_ir.py)：NamedAxis、Port、Tensor、Node、Edge、StateVersion/Transition、Guard/Activation、AliasRelation、FoldBoundaryMapping/Group。
- [graph_document.py](../../structure_reader/src/structure_reader/graph_document.py)：SemanticGraph、规范化JSON、受控反序列化与重复键检查；保留轴/端口/证据等内部顺序，显式记录world_size=1分析假设。
- [graph_contracts.py](../../structure_reader/src/structure_reader/checks/graph_contracts.py)：轴/张量/端点引用、四类边方向与tensor一致性、状态版本/环、guard/alias引用及诊断稳定性检查。
- Tensor已有logical_shape/physical_shape派生接口；Node与图文档JSON输入边界已返修。

这些是库接口。**CLI当前仍只提供inspect，build/validate/render/export-tables尚未实现。**实际用法见[用户指南](../../structure_reader/docs/user_guide.md)。

### 3.3 检查器本轮证据与尚未闭环部分

1. 原版`d3be1523`：独立46项中7失败，涉及state边漏检、跨方向同名Port误报、环诊断顺序漂移、重复ID定义的先到先得漂移。[原始失败报告](../../structure_reader/outputs/_coordination/s1/2026_10_12_round_01/results/verification_01b_graph_reference_checker.md)。
2. Edge小步`c3cb83b1`：正式交接；Harness与Codex的V3连接测试16/16通过，完整46项44通过、剩2失败。[开发交接](../../structure_reader/outputs/_coordination/s1/2026_10_12_round_01/results/repair_01b_edge_endpoint_slice.md)。实际红转绿5项；该开发报告曾误写6项，计数校正仍需收尾。
3. 第二小步当前代码快照`2bf28874`：Harness已记录56/56与全量853/853通过，日志已落盘；Codex独立56项通过，并用不同的可达关系算法对照200个合成图、116个含环强连通分量，0失败。[复验过程报告](../../structure_reader/outputs/_coordination/s1/2026_10_12_round_01/results/verification_01b_repair_rerun.md)、[853项日志](../../structure_reader/outputs/_coordination/s1/2026_10_12_round_01/evidence/verification_01b_rerun_scc_live_discover.log)、[Codex交叉对照证据](../../structure_reader/outputs/_coordination/s1/2026_10_12_round_01/evidence/codex_01b_scc_crosscheck.json)。
4. **第二小步仍缺正式开发汇总**：本次核对未找到`results/repair_01b_graph_reference_checker.md`；Harness报告保留定版pending，不能写成完整01b/S1通过。

本次核对代码SHA256：`2bf288741fc061695ef1e8f2811a810fdf1807fa84cf7d9a44264487361cbc39`。
独立检查器测试SHA256：`5ebf7e76bca59024c2d8e6cd20e3ddcbfb8ff8312c54fd4521f19b53aaff5a17`。

853项是797项既有回归加56项检查器测试；测试数不代表项目完成比例。上述合成图验证也不证明已经正确提取真实DeepSeek结构。

### 3.4 用户目标当前可达程度

- **带含义的维度**：数据模型部分具备，尚无阅读页呈现。
- **关键模块和连接**：通用库可存图、检查器当前快照有测试依据；实际模型连接还未提取或作源码语义核验。
- **CSA/HCA/MoE/mHC/DSpark**：尚无已生成的真实模型图和代表层页面。
- **条件与KV cache**：已有三态及状态身份表示；谓词/形状求值、容量、有效长度、remainder、overlap和Indexer状态尚未完整实现。
- **源码证据**：当前主要为字符串引用；结构化证据对象、等级与断言语义核验尚未完成。
- **离线页面、CSV、Qwen迁移和Skill**：均未交付。

[当前效果直测](../../structure_reader/outputs/_coordination/s1/2026_10_12_round_01/audit/current_user_effect_check.md)记录CLI仅inspect、无structure.html/semantic_graph.json/connections.csv模型产物。[详细过程审核](../../structure_reader/outputs/_coordination/s1/2026_10_12_round_01/audit/retry_2026_10_13_01b.md)保留缺陷、返修和复验轨迹。

### 3.5 工具分工与最近执行状态

按[执行规范](../development_workflow.md)：Claude在工作目录可见CLI开发；DeepSeek Harness在应用内工作区对话独立测试；Codex亲自复验并裁定、记录能力状态；用户参与实际阅读反馈。

本轮主要使用VS Code可见Terminal9与Harness的LLMarchi_excel会话。默认模型/设置未主动修改。用户对本项目测试必要源码/配置/测试发送Harness远程模型的明确授权继续有效，排除凭据、其他项目与权重。

两端早间恢复连通并实际执行了开发/测试，随后在收尾阶段再次观察到Claude 502和Harness TRANSPORT。代码及测试证据保留；汇总与定版交接未完成。本次阶段整理不重新测试连通性，也不推断网络根因或永久不可用。

## 4. 后续执行顺序：先收齐S1，再进入S2

### 下一节点A：关闭本轮01b检查器返修交接

- **Claude**：读取已有56项/全量日志，校正Edge报告的红转绿数量，补正式第二小步汇总、文件清单、SHA与未覆盖范围。任务见[诊断稳定性小单](../../structure_reader/outputs/_coordination/s1/2026_10_12_round_01/dispatch/repair_01b_diagnostic_stability_slice.md)。
- **Harness**：对已交接的最终SHA复跑独立检查器套件及全量；保留原红测和中间SCC缺陷，补正式定版结果，不放宽期望。见[独立复跑单](../../structure_reader/outputs/_coordination/s1/2026_10_12_round_01/dispatch/verification_01b_repair_rerun.md)。
- **Codex**：核对三方版本和报告计数，独立复跑最终完整回归；记录该局部契约是否可接受。完整S1仍需以下节点，不能由01b单项替代。
- **关闭标准**：实际文件/报告齐备、同SHA复跑通过、四类缺陷及SCC边界无遗漏，日志说明模型提取仍未做。

### 下一节点B：折叠边界与剩余图契约

已准备[折叠边界检查小单](../../structure_reader/outputs/_coordination/s1/2026_10_12_round_01/dispatch/development_01b_fold_boundary_checker.md)，尚未分发。

先补成员/端口存在、跨组边的data/state/select/control映射覆盖、孤立映射、合法fan-out和重叠组支持边界。边界覆盖通过后，还需独立证明实际折叠前后连接等价；两者不能混为一个已完成能力。rank/layout等完整图契约继续对照v3逐项核对。

### 下一节点C：有限表达式与场景/状态

已准备[有限表达式核心02a](../../structure_reader/outputs/_coordination/s1/2026_10_12_round_01/dispatch/development_02a_finite_expression_core.md)及[完整场景求值单](../../structure_reader/outputs/_coordination/s1/2026_10_12_round_01/dispatch/development_task_02_scenario_evaluator.md)，尚未分发。

先实现AST白名单、整数/布尔类型边界、unknown原因与三态逻辑、长度/深度/整数规模限制；整棵表达式先校验，不因短路放过非法节点。然后接场景、轴与Guard结果及缓存历史假设；P>0不自动等于存在有效cache。补capacity/valid_length/remainder/overlap等表达，不改模板并集拓扑。

### 下一节点D：来源证据、模板/实例模型与schema

按[源码证据开发单](../../structure_reader/outputs/_coordination/s1/2026_10_12_round_01/dispatch/development_task_03_source_evidence.md)实现qualified_symbol、固定基线、AST表达式/调用断言、located与source_checked分级；锚点存在不能替代语义正确。

补模板/实例表示（S2再做真实逐层绑定）、结构化证据和状态属性；在接口稳定后交付匹配实现的JSON Schema、公开API与使用说明。合成fixture独立维护，不能把新生成输出自动当golden。

### 下一节点E：S1完整验收门禁

Harness对合并接口独立验证，Codex检查实际图构造/求值/来源失配/折叠错误与产物，逐项对照v3 S1标准。完整证据充分后才写S1正式通过裁定并启动S2。

**S1尚未关闭的清单**：开发/定版交接、完整折叠与图契约、有限表达式/场景、缓存属性、结构化源码证据、模板/实例模型、JSON Schema、公开接口/文档及合并验收。

## 5. S2以后用户能看到什么

- **S2完成**：正确的层分组和实例绑定；还不能代替正式阅读页。
- **S3完成**：第一个有源码证据和核验的CSA·TopK代表层页面，可直接开始阅读效果验收。
- **S4–S6完成**：完整DeepSeek目标特征、离线阅读体验、同源连接/轴/实例/证据表格及可重复交付。
- **S7–S8完成**：经Qwen3-0.6B迁移回归的工程、真实支持矩阵与Skill。

后续正式产物按v3落位：

```text
structure_reader/outputs/deepseek_v4_flash_0731/<analysis_id>/
  README.md
  viewer/structure.html
  data/semantic_graph.json
  data/layer_plan.json
  data/instance_bindings.json
  data/scenarios.json
  reports/validation_report.json
  reports/source_evidence.json
  reports/coverage_report.json
  tables/connections.csv
  tables/tensor_axes.csv
  ...
```

上树是**后续交付约定**，本次未创建占位模型产物。实际analysis_id由输入/规则/场景/工具版本内容生成，不能直接使用本记录日期替代。

## 6. 文件分区与后续更新方式

- `hf_docs/model_structure_plan_v3.md`：正式原计划；本文件是阶段快照。
- `hf_docs/stage_reports/`：阶段性计划/整体进度/下一步记录。
- `structure_reader/src/`、`configs/`、`schemas/`：产品与分析契约；未实现目录不为好看而建空包。
- `structure_reader/scripts/`：用户可执行入口。
- `structure_reader/tests/`：Harness独立测试与审查fixture。
- `structure_reader/docs/`：实际使用、开发、产物字段和持续进度。
- `structure_reader/outputs/_coordination/`：任务单、角色报告、证据与Codex审核；不是模型语义图输入。
- `structure_reader/outputs/<model>/<analysis_id>/`：通过核验后的正式模型交付。
- `structure_reader/.work/`：可丢弃暂存，不作为交付或正确答案来源。

后续阶段整理新增日期/轮次文件并更新[阶段索引](README.md)。每次说明实际SHA、产物、测试与未覆盖范围；测试绿灯和最终阶段批准分开；原错误记录保留。下一次恢复先关闭节点A，再依上述S1节点推进，之后依S2–S8门禁执行。
