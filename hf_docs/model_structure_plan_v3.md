> **历史需求计划。** 用户2026-10-17已重新确认[当前需求](../structure_reader/docs/requirements_alignment_2026_10_17.md)，执行使用[实施计划v5](project_execution_plan_v5.md)。本文历史条款不再补充或扩大当前范围。

# 模型特征结构阅读工具：项目计划 v3（当前执行）

更新日期：2026-10-13（执行状态更新，计划范围不变）。状态：**S0 passed（Codex批准），S1 in_progress。** 分工见[开发执行规范](development_workflow.md)，S0裁定见[最终审核](../structure_reader/outputs/_coordination/s0/2026_10_12_round_02/audit/final_acceptance.md)，S1进展与实际缺陷见[最新过程审核](../structure_reader/outputs/_coordination/s1/2026_10_12_round_01/audit/retry_2026_10_13_01b.md)。局部IR/统一图文档已有797项回归通过；新增图级引用检查器原版46项独立测试中7项失败，正在返修，不能以旧绿灯批准S1。模型结构阅读页仍未实现；首轮失败/审批阻断记录保留历史。

本版整合 v2 审查结论，保留静态分析与特征结构代表路线，补齐结构修正、条件/状态表示、源码证据、核验和工作区规范。v1/v2 与审查记录归入 `archive/`，仅作历史参考。本计划自包含，开发不必到旧版本拼接规范。

## 1. 目标、范围与交付

输入模型配置、对应源码和分析场景，输出可分层展开的数据流阅读页。每类已核对的局部结构只保存一份模板；整体概览保留层序、数量、实例绑定与跨层读出。重复层不逐层铺开内部结构。

主案例为 **DeepSeek-V4-Flash-0731 官方参考实现**，覆盖 SWA/CSA/HCA、低秩 Q/O、共享 KV、mHC、Hash/Top-K MoE、缓存/压缩/Indexer 状态及 DSpark。HF 实现本期交付具名差异报告；完整第二套 HF 图列为扩展任务。

首版交付包括：

1. 自包含离线阅读页 `structure.html`，含概览、代表层、模块内部、轴图例、源码片段与条件路径。
2. 统一语义图数据、层型规划、实例绑定、场景与输入清单。
3. 阅读说明、结构核验、源码证据、覆盖与实现差异报告。
4. 同源语义节点/连接/轴/实例/证据 CSV；xlsx 为可选同源格式。
5. 固定型号的 Qwen3 迁移案例及 DeepSeek 回归结果。
6. 调用已实现工程入口的项目 Skill。

本期证据上限是**源码核对与配置/场景推导**。运行追踪、数值等价验证、真实推理、全量权重下载、底层 ATen/ONNX 算子表、多模态/SSM 等通用支持另列后续阶段。语义节点表不冒称底层算子表。

## 2. 工作区与命名规则

### 2.1 顶层职责

```text
LLMarchi_excel/
  README.md                    # 当前入口：目录用途、计划与使用文档导航
  hf_docs/                     # HF 资料、当前计划与历史记录
    README.md                  # HF 资料索引与依赖关系
    model_structure_plan_v3.md # 唯一当前计划
    model_structure_visualization.md  # 方法说明与远期工具评估
    archive/                   # v1/v2、审查记录及旧说明入口
  hflib/                       # 第三方软件源码输入（Transformers、Diffusers）
  models/                      # 原始模型配置、参考代码、分词及权重输入
  scripts/                     # 现有工作区级工具，例如下载模型
  structure_reader/            # 本工程唯一开发根目录；在 S0 创建
```

本次只建立/整理文档入口与历史目录。以下工程树为开发规划，不创建空包、空脚本或伪造产物。

### 2.2 工程目录

```text
structure_reader/
  README.md                    # 工程用途、环境、快速使用、能力与限制
  pyproject.toml               # 项目与版本信息，核心不依赖 torch
  .gitignore                   # outputs/.work/.venv/缓存的忽略规范
  scripts/
    read_model_structure.py    # 唯一用户命令入口；薄封装，调用 cli.py
  src/structure_reader/
    __init__.py
    cli.py                     # 参数、命令与错误退出码
    inventory.py               # 输入扫描、版本/哈希和能力报告
    layer_plan.py              # 模板分组、计数和实例绑定
    graph_ir.py                # 统一节点/端口/边/轴/状态结构
    model_registry.py          # 模型标识 → 文件/适配器/场景入口
    source_evidence.py         # 符号、表达式、行号与证据定位
    scenario_evaluator.py      # 有限条件与形状表达式求值
    rules/
      deepseek_v4_flash_0731.py # 声明适配范围与结构规则
      qwen3_0_6b.py             # Qwen3-0.6B 的适配声明
    extract/
      deepseek_v4_reference.py # 构建参考实现模板与绑定
      qwen3_hf.py               # 构建 Qwen3 HF 模板与绑定
    checks/
      graph_contracts.py       # 公用 IR/端口/轴/状态与展开检查
      source_contracts.py      # 审查基线与表达式证据检查
      layer_equivalence.py     # 适配范围内的结构等价与层分区检查
      deepseek_v4_contracts.py # 独立于提取器的模型契约
    render/
      html_renderer.py         # 把通过核验的数据嵌入离线页
      assets/reader.html       # 阅读页模板；不是运行产物
      assets/reader.css
      assets/reader.js
    export/
      semantic_tables.py       # 同源 CSV 和可选 xlsx
      reading_reports.py       # 阅读说明、覆盖与证据报告
  configs/
    models/deepseek_v4_flash_0731.json
    models/qwen3_0_6b.json
    scenarios/deepseek_v4_flash_0731.json
    scenarios/qwen3_0_6b.json
    source_baselines/
      deepseek_v4_flash_0731_reference.json    # 参考实现审查基线
      deepseek_v4_flash_0731_hf_comparison.json # HF 对照审查基线
      qwen3_0_6b_hf.json        # Qwen3 HF 审查基线
  schemas/semantic_graph.schema.json
  docs/
    user_guide.md              # 用户：怎么运行、打开什么、怎么读结果
    developer_guide.md         # 开发者：怎么适配、校验与维护
    output_reference.md        # 每个产物的字段、用途与证据等级
    progress.md                # S0–S8 状态、实际产物与验收记录
  skills/model-structure-reader/
    SKILL.md
    references/reading_format.md
  tests/
    fixtures/                 # 小型、已审查的期望数据，属于源码资产
    test_graph_contracts.py
    test_source_contracts.py
    test_layer_partition.py
    test_deepseek_structure.py
    test_scenarios.py
    test_semantic_tables.py
    test_html_reader.py
    test_model_regression.py
  outputs/                    # 唯一正式生成产物根目录
  .work/                      # 可丢弃临时文件；不得成为输入或正式交付
```

普通 Python/JSON/Markdown 功能文件使用 `snake_case`；README.md、SKILL.md 等标准入口名沿用约定。模型键固定为 `deepseek_v4_flash_0731` 和 `qwen3_0_6b`，展示名称另存正式 model ID。不要使用 `model.py`、`utils.py`、`test.py`、`temp_final.py` 等含糊命名。

`scripts/` 只放可执行入口；`src/` 是库代码；`configs/` 是分析设置和已审查基线，不能覆盖原始模型 config；`tests/fixtures/` 是版本化验收资产；`outputs/` 和 `.work/` 不反向作为适配规则或测试期望。

### 2.3 正式产物目录

```text
structure_reader/outputs/deepseek_v4_flash_0731/<analysis_id>/
  README.md                   # 自动生成的交付导航：先打开什么、对象与证据范围
  run_manifest.json           # 输入/规则/场景/工具版本、状态与文件清单
  viewer/structure.html       # 用户直接打开的自包含阅读页
  data/
    input_manifest.json
    layer_plan.json
    semantic_graph.json
    instance_bindings.json
    scenarios.json
  reports/
    reading_guide.md
    validation_report.json
    source_evidence.json
    coverage_report.json
    hf_implementation_diff.md
  tables/
    semantic_nodes.csv
    connections.csv
    tensor_axes.csv
    layer_instances.csv
    source_evidence.csv
    checks.csv
    semantic_tables.xlsx      # 仅启用 xlsx 时生成
  logs/build.log              # 诊断记录，不参与图的事实依据
```

`analysis_id` 由规范化输入内容、分析设置、规则/基线、schema 和影响输出的工具/模板版本生成，不能只用时间戳。可使用内容哈希前缀，完整哈希写入 manifest，检测前缀冲突。同输入重跑复用同目录并进行一致性检查；新配置或版本生成新目录，不静默覆盖不同分析。

有效正式产物只在核验成功后提交；失败记录写入 `outputs/_failed/<run_id>/`，manifest 明确 `failed`，不生成看似成功的正式阅读页。`_failed` 也是产物，`.work` 只是暂存。

不产生 `final/final2/new/latest_copy` 等重复目录。命令结束返回本次 analysis_dir、阅读页和核验报告路径；如实现 outputs 索引，只有一个 `outputs/index.html`，不再复制一套结果。

暂存与日志不提交为工程源码；正式产物按需保留/分享，运行后不得散落到根目录、models 或 hflib。清理限定 `.work` 或具名生成目录，不默认删除历史分析或模型输入。

## 3. 正确的案例结构

### 3.1 对象与版本

参考输入为 `models/deepseek-ai/DeepSeek-V4-Flash-0731/` 下的根配置、`inference/config.json`、`inference/model.py` 与相关 kernel 源码。HF 对照输入为 `hflib/transformers/src/transformers/models/deepseek_v4/` 的 modeling/configuration 源码。

本地清单声明模型 revision 为 `7872f01b1d1fe23eabc4c98b48bffcef5a386062`；HF checkout 当前核对的 commit 为 `536ecc007387a50e77603bb5d92100e9b07514cc`。S0 记录并核对实际状态；清单声明 revision 与本地内容哈希分别存放，未核对线上一致性时不声称它们已经一致。

原始根配置与参考运行配置分别解释：参考实现使用 `dim/n_layers/n_mtp_layers` 等，根配置使用 `hidden_size/num_hidden_layers/num_nextn_predict_layers` 等。S0 输出 raw → effective 映射、覆盖值与 dataclass 默认值来源，不假定名称相似就等价。

当前已核对主干分组：

- `swa_hash`：层 0/1，共 2。
- `csa_hash`：层 2，共 1。
- `hca_topk`：层 3/5/…/41，共 20。
- `csa_topk`：层 4/6/…/42，共 20。

组内完整层号由程序生成；并集必须等于 0–42，互不重叠。CSA 总计 21，主干合计 43。源码核对依据：[参考实现](../models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/model.py) 与 [原始配置](../models/deepseek-ai/DeepSeek-V4-Flash-0731/config.json)。

### 3.2 主干与 mHC

模型概览必须包含 `Embedding → HC Expand → 43 层主干 → HC Head → Final RMSNorm → LM Head`，并保留主干目标层到 DSpark 的跨层读出。

每个 Block 有两次完整路径：

```text
streams → hc_pre → attention_norm → Attention → hc_post（汇入原 streams）
        → hc_pre → ffn_norm       → MoE       → hc_post（汇入本段输入 streams）
```

streams 为 `[B,T,N_hc,H]`，pre 汇聚输出为 `[B,T,H]`；post 系数为 `[B,T,N_hc]`，comb 矩阵为 `[B,T,N_hc_src,N_hc_dst]`，hc_post 输出仍为 `[B,T,N_hc_dst,H]`。方向契约：

```text
out[...,k,d] = post[...,k]*module_output[...,d]
             + sum_j comb[...,j,k]*residual[...,j,d]
```

源流轴与目标流轴长度都可为 4，方向错误不能仅靠 shape 发现；须保存并核对索引/归约轴与表达式。

### 3.3 Attention、MoE 与 DSpark

- 输入并行产生 Q、窗口共享 KV、压缩分支和 Indexer 的相关输入。Indexer 还读取低秩查询状态 qr；它产生选择索引，不是 Value。
- CSA 有学习式 Indexer、Top-K 选择、带 overlap 的压缩与独立 Indexer 状态。HCA 没有学习式 Indexer，但仍有压缩条目的因果可见位置选择；两者都结合窗口 KV。
- 共享 KV 单投影/同一张量作为 K 与 V，保留 alias；不能画成独立 K/V 两个 Linear。
- Attention 输出先对其 RoPE 切片反向旋转，再做分组低秩 O 投影。保留 sink、部分 RoPE、量化模拟和布局说明；kernel 内部未分析时标为不透明节点。
- MoE 分开表示 Hash 与得分 Top-K 选择、路由权重、专家模板、token 分派/汇合和 shared expert。实际专家选择与每专家 token 数静态时为未知。
- DSpark 使用公共 stage 计算模板 + 实例绑定 + 独立入口/出口子图：首 stage 的主干投影/草稿输入、末 stage 的 head/Markov/confidence 可见；3-stage 不被误标为三个完整同构实例。首尾角色可同时成立，以兼容未来单 stage 配置。
- 参考配置 3 个 stage、根配置 `num_nextn_predict_layers=1`、长度 46 的 compress_ratios 分别记录消费方；前 43 项属于主干，尾部 3 项属于草稿 stage。
- 当前 HF 实现无对应 DSpark 类且忽略 `mtp.*`；差异报告不得暗示其已覆盖完整 0731。

## 4. 阅读层级、轴与场景

### 4.1 四层界面

- L0：首尾组件、主干层序条带、分组计数、DSpark 入口/出口与跨层读出。
- L1：4 类主干代表结构；选择具体层时展示真实实例绑定。
- L2：Attention、Compressor、Indexer、MoE、mHC、DSpark 入口/出口；折叠不丢端口、旁路与状态边。
- L3：预留底层子图与粒度接口；首版不承诺 ATen/ONNX 逐算子解析。重要布局操作可通过边注释展开说明。

颜色辅助类型识别，层号/类型文字始终可访问；信息不依赖颜色或悬停。源码片段与来源在离线页内嵌，不能要求浏览器联网或直接读取模型目录才能阅读。

### 4.2 轴规范

完整展示例：`hidden_states [batch_size=1, seq_len=1, hc_streams=4, hidden_size=4096]`。短标签可为 `[B=1,T=1,N_hc=4,H=4096]`，常驻图例与端口详情给全名、中文含义和来源。hidden_states 是张量名称，hidden_size 是轴长度，二者分别存储。

每个轴保存 symbol、name、meaning、value/表达式、origin 与证据。每个张量另存 dtype、布局、逻辑/物理 shape。具体值来自配置、场景或有条件的源码推导；未知含原因。

维度字典：

- B=batch_size；T=本次 seq_len；P=调用前 past_len；L=P+T=逻辑总长度，均与场景有关。
- H=hidden_size=4096；N_hc=hc_streams=4；N_hc_src/dst 区分流混合方向。
- Hq=query_heads=64；Hkv=kv_heads=1；D=head_dim=512；D_rope=64；D_nope=448。D 不由 H/Hq 推断。
- Rq=q_lora_rank=1024；G=output_groups=8；Ro=o_lora_rank=1024。
- W=window_size=128；r=compress_ratio（CSA 4/HCA 128）；C=已完成 compressed_entries；R=待压缩 remainder_tokens，三个概念不得混用。
- Hi=index_heads=64；Di=index_head_dim=128；Ki=index_topk=512 是上限，不是每条 query 都有 512 个有效选择。
- E=routed_experts=256；K=experts_per_token=6；Eshared=1；I=expert_intermediate_size=2048；N_tokens=B×T；N_e=路由到专家 e 的 token 数，静态时未知。

参考代码可能使用 `[B,T,Hq,D]`，HF 路径可能使用 `[B,Hq,T,D]`；端口保留实际布局。涉及并行时区分全局/本 rank 维度，首版场景固定 world_size=1 并记录这是分析假设。缓存容量还受参考 ModelArgs 默认/覆盖的 max_seq_len 影响，不能直接用根配置 max_position_embeddings 代替。

### 4.3 静态条件模板

保存带 guard 的模板，场景切换计算已知条件并突出适用节点/边。`active_for_scenario` 为 true/false/unknown；数据相关条件未知时保持 unknown。模板的并集拓扑与某场景实际激活路径分别表示，不能宣称场景只改变 shape。

标准场景至少包括：

- prefill：P=0，T=3/4/5/127/128/129，覆盖未凑满块、压缩边界与尾部。
- decode：T=1；P=2/3/4 覆盖 CSA，P=126/127/128 覆盖 HCA 和窗口回绕。默认先显示 P=128 的非压缩边界 decode，再提供 P=127 的边界对照。
- Indexer：C=0、少于 Ki、达到/超过 Ki 的形状/可见性条件；有因果约束的条目数量与存储槽位数量分开。
- DSpark：主干读出、prefill 建缓存与 decode 草稿计算分别展示。

场景含历史起点/连续调用假设、有效参数和已知状态条件。P>0 不能自动当作存在有效历史 cache；缺少初始条件时保留未知。状态记录是源码推导，不称为真实运行快照。参考实现不支持的多 token decode 不能借 HF 路径补成已支持。

## 5. 图数据、证据与核验

### 5.1 主数据是图

IR 统一保存：template_id、局部调用 node_id、端口/tensor ID、边、结构化轴、条件、来源、状态前后版本、别名、折叠端口映射和 granularity。表格只是同一数据的投影。

模板与实例分开：instance_binding 记录实际 layer/stage、参数路径/state_id、角色、逐层配置和外部读出。40/41/42 层可共用相应局部模板，但保留 DSpark 读出差异。结构签名纳入实现版本、Attention/MoE 变体及影响结构的尺寸/选项。

等价检查限定于适配器识别过的字段与源码条件；未覆盖的结构分支不默认为同构。非结构字段可记为“不影响当前图”，而不是所有未知 config 字段一律失败或静默忽略。

边类别包括 data/state/select/control。状态使用 state_id 与 before/after version 表示跨调用，区分 capacity、valid_length、压缩尾部、overlap、Indexer 与 slice/view/alias；不能把缓存更新画成同一时刻无版本的循环。

### 5.2 每条结论有证据

规则基线保存已审查的 source/config 哈希、符号、表达式匹配与适配版本。当前文件与规则基线比较；不能仅与本次刚生成的 inventory 比较。

证据包含 source_id、qualified_symbol、解析出的行区间、源码片段/表达式摘要、支持的断言与状态。对边记录生产表达式、消费调用/参数、必要的 guard 与别名依据。行号存在和 AST 解析成功都不是语义核验通过。

仅在适配范围内做 AST 符号/赋值/参数/分支匹配；复杂连接经源码审查形成具名规则，不建设任意 Python 理解器，不执行/导入模型参考代码。形状和条件表达式采用受限算术/比较求值，不调用任意 Python 表达式。

### 5.3 核验始终在出图前

固定流程：`inspect inputs → classify → extract → validate → render → export`。默认 build 完成全过程。

核验包括层分区、模板/实例身份、端口存在、轴和来源、连接证据、guard、状态/alias、折叠端口映射，以及模型特有方向/顺序契约。提取器和契约分别维护；独立审查 fixture 不从提取器输出自动刷新为“正确答案”。

受控错误至少覆盖：重复/缺失层；DSpark 首尾丢失；comb 方向反转；给 HCA 添加 Indexer；共享 KV 拆为双投影；删除 qr→Indexer；inverse RoPE 移到 O 投影后；错误压缩 guard；丢失 state version；错误布局轴；源码内容与规则基线失配。

每个节点/边区分源码核对、配置/场景推导、未覆盖。报告预留形状采集/执行/数值证据字段，但首版不填伪运行证据。独立 AI 阅读作为补充审核，结论须落到源码表达式，不能代替事实证据或自动提升等级。

## 6. 开发阶段与责任

阶段编号统一为 **S0–S8**，避免与历史 M/T 编号混用。职责可以由同一个开发者依次承担：工程负责人负责实现，模型审查负责源码依据，验证负责人负责独立期望/失败检查，阅读验收人负责产品可读性。用户只需参与阅读验收，不承担手工补图、维护 JSON 或环境排错。

每阶段具备任务、文件、产出/功能、开发方式、验收与分工；通过时在 `docs/progress.md` 写实际文件、检查结果及未覆盖范围。自 2026-10-12 起，开发任务由 Claude Code CLI 执行，独立校验/验证/测试由 DeepSeek Harness 执行，最终审核/验收/成果测试由 Codex 完成；详见执行规范。当前S0=passed，S1=in_progress（IR小单已向可见Claude分发），S2–S8=pending。首轮失败记录保留；授权后的Harness完整定版已执行。

### S0：工程布局、命令入口与输入基线（约 1–2 人日）

- **依赖/分工**：起点；工程负责人主责，模型审查核对输入。
- **任务**：创建非空工程入口和文档；建立模型注册与输入清单；分别探测解释器/依赖；记录 raw/effective config、默认值、版本和已审查基线。
- **预期文件**：工程 README/pyproject/.gitignore；`scripts/read_model_structure.py`；`cli.py`、`inventory.py`、`model_registry.py`；DeepSeek model config 与 source_baseline；`docs/user_guide.md`、`developer_guide.md`、`output_reference.md`、`progress.md`。
- **产出/功能**：inspect 命令、input_manifest/capability report；目录职责与可用命令清晰。
- **开发方式**：核心标准库、Python ≥3.11，优先使用已有 3.12；torch 不作为静态前置。输入源码只读，不安装模型运行依赖。其他版本兼容性实测再标记。
- **验收**：真实检出无权重；原始与运行配置消费方清楚；所有输入/默认值有来源；源码与规则审查基线可比较；CLI 帮助只列实际命令；临时/产物均在指定目录。

### S1：IR、轴、条件与通用核验（约 2–3 人日）

- **依赖/分工**：S0；工程负责人主责，验证负责人审查契约。
- **任务**：定义 schema；实现结构化轴、guard 三态、state version/alias、来源与模板/实例模型；建立有限表达式求值及最小核验。
- **预期文件**：`graph_ir.py`、`scenario_evaluator.py`、`source_evidence.py`、schema；`checks/graph_contracts.py`、`source_contracts.py`；`tests/test_graph_contracts.py`、`test_source_contracts.py`、`test_scenarios.py`。
- **产出/功能**：语义图可校验、可序列化；错误报告精确到节点/边/字段。
- **开发方式**：用已审查的小图分别测试别名、前后状态、折叠映射、布局和未知值；条件不依赖运行模型。
- **验收**：缺端口/轴来源、错误状态版本、失配表达式被检出；unknown 不被当作 false；相同输入规范化图稳定；源流/目标流轴不因数值相等而混淆。

### S2：主干分类与实例绑定（约 1 人日）

- **依赖/分工**：S1；模型适配负责人主责，模型审查/验证配合。
- **任务**：生成四类主干模板分组；实现配置消费声明、实例身份、目标层读出、DSpark 角色绑定和结构等价检查。
- **预期文件**：`layer_plan.py`、DeepSeek rules；`checks/layer_equivalence.py`；`tests/test_layer_partition.py`。
- **产出/功能**：layer_plan、instance_bindings 与等价检查报告；概览所需计数从同一数据生成。
- **开发方式**：检查适配范围内初始化/forward/stage 条件；局部模板与外部读出分别编码。
- **验收**：2/1/20/20 合计 43、层分区无重叠缺失；DSpark 不归入主干；实例参数/状态 ID 唯一；40/41/42 的读出保留；未处理结构差异导致明确失败或拆组。

### S3：CSA·TopK 代表的端到端闭环（约 2–3 人日）

- **依赖/分工**：S2；模型适配/工程负责人主责，验证负责人独立核对。
- **任务**：先完成代表层 4 的双 mHC、Attention/Compressor/Indexer/MoE；对连接和条件给源码依据；接入 validate、最小阅读页和报告。
- **预期文件**：`extract/deepseek_v4_reference.py` 的 CSA 路径；`checks/deepseek_v4_contracts.py`；render 模板/renderer 的最小实现；`tests/test_deepseek_structure.py`；独立 fixtures。
- **产出/功能**：首个可核验的 analysis_dir，含语义图、核验/证据/覆盖报告、可打开的代表层阅读页。
- **开发方式**：按源码表达式核对 Q/KV 并行、qr→Indexer、选择/状态边、inverse RoPE 与 O 顺序；源码边和形状推导分别标记。
- **验收**：关键节点/边都有语义匹配的锚点；反转 comb、删 qr 边、交换 RoPE/O 顺序均失败；非边界/边界场景有 guard 差异；核验失败不发布正式阅读页。

### S4：其余主干变体与完整 DSpark（约 2–4 人日）

- **依赖/分工**：S3；模型适配负责人主责，模型审查复核。
- **任务**：补 SWA·Hash、CSA·Hash、HCA·TopK；实现 DSpark 公共模板、首尾适配、主干读出、Markov/confidence 与共享参数；形成 HF 差异报告。
- **预期文件**：DeepSeek rules/extractor/contracts 的完整适配；DeepSeek scenarios；扩充 structure/scenario 测试与 fixtures；`export/reading_reports.py`。
- **产出/功能**：主案例所有目标特征的源码语义图；HF 差异说明；具名未覆盖范围。
- **开发方式**：复用公共模块模板，保留阶段/入口方法/场景差异；自定义 kernel 按已核对接口呈现，未读内部不虚构内部算子。
- **验收**：HCA 无学习式 Indexer；Hash 使用 token ID 选专家的依赖可见；DSpark 首尾和目标层连接完整，prefill 不显示不存在的完整草稿路径；46 项配置不误画为 46 主干层。

### S5：完整阅读体验与使用文档（约 1–2 人日）

- **依赖/分工**：S3 后可做布局，完成验收依赖 S4；展示负责人主责，用户参与阅读验收。
- **任务**：概览/分组/实例切换、L1/L2 展开、端口轴详情、源码片段、场景突出与状态面板；生成交付导航与阅读指南。
- **预期文件**：render assets/renderer；`tests/test_html_reader.py`；更新 user_guide/output_reference。
- **产出/功能**：正式自包含 structure.html 与 reading_guide；命令清楚指出打开哪个文件。
- **开发方式**：界面只读通过核验的 IR；无需外部 CDN/服务器；关键关系直接标注，更多细节按需查看。
- **验收**：file:// 离线可读；无裸数字 shape；源码/轴全名不依赖悬停；桌面与窄屏可用；展开前后端口连接守恒。阅读人能回答 CSA 选择来源、缓存边界、mHC 流向、层 2/4 路由差异、Q 中间形状、DSpark 首尾差异。

### S6：同源表格、独立核验与可重复交付（约 1–2 人日）

- **依赖/分工**：S4/S5；验证负责人主责，工程负责人补失败路径。
- **任务**：导出六类 CSV、可选 xlsx；代表/全实例两种投影；扩充受控错误、稳定 ID、输入失配、目录隔离和重复运行测试。
- **预期文件**：`export/semantic_tables.py`；`tests/test_semantic_tables.py` 与各 contracts 回归；输出清单生成；相关 user_guide。
- **产出/功能**：tables、可重复分析目录和完整核验/覆盖报告。
- **开发方式**：表格从 IR/实例绑定投影，不重新解析源码；xlsx 可用 openpyxl，缺少时 CSV 仍可正常交付。静态检查不命名为模型数值验证。
- **验收**：表格可按 ID 联查；全实例 node ID 唯一且保留模板 ID；表/图连接一致；全部受控错误被捕获；重跑规范化 data/报告一致（运行时间/日志等元数据除外）；不同版本产物隔离，失败不污染成功目录。

### S7：固定 Qwen3 型号的迁移回归（约 1–2 人日）

- **依赖/分工**：S6；适配负责人主责，验证负责人验收。
- **任务**：选择 Qwen3-0.6B，固定 config revision/哈希和 HF source commit；复用流程表达独立 Q/K/V、GQA、SwiGLU、普通残差与 cache；运行 DeepSeek 回归。
- **预期文件**：Qwen model/scenario/source_baseline；`rules/qwen3_0_6b.py`、`extract/qwen3_hf.py`；`tests/test_model_regression.py` 与 Qwen fixtures。
- **产出/功能**：第二模型的完整静态交付包；实际支持矩阵。
- **开发方式**：只准备配置/源码资料，不下载权重；28 层等值以固定配置为准。优先新增适配器，必要的公用接口修订具名记录并回归。
- **验收**：不靠 DeepSeek 特例表达 Q/K/V 与残差；GQA 头映射正确；按 config 分类，不写死整个系列层数；DeepSeek 所有验收仍通过；支持范围不外推到未测试模型。

### S8：Skill 固化与首版发布（约 0.5–1 人日）

- **依赖/分工**：S7 与工程命令稳定；Skill 集成负责人主责，验证负责人核对失败/降级行为。
- **任务**：用技能创建流程编写项目 Skill；固化模型/版本/场景选择、build/validate、阅读与交付说明；补完整用户和开发者文档。
- **预期文件**：`skills/model-structure-reader/SKILL.md`、reading_format；工程 README、user_guide、developer_guide、output_reference、progress。
- **产出/功能**：两模型可重复使用的工程与 Skill；一次请求给出明确结果路径、证据与未覆盖范围。
- **开发方式**：Skill 只编排已实现命令，不复制图提取逻辑；个人技能目录安装另作使用步骤；独立审核按会话授权与可用能力组织。
- **验收**：DeepSeek/Qwen 各生成一次成功；未知模型、源码基线失配、缺 xlsx 依赖等失败/降级正确；不跳过 validate、不伪造执行证据；用户不手工修改 JSON 也能使用。

## 7. 命令与用户使用方式

本节是 **S0–S8 的接口契约**。S0脚本与inspect已实现，实际用法见[工程用户指南](../structure_reader/docs/user_guide.md)。build/validate/render/export-tables仍未实现，以下相关示例只属计划；帮助文本只展示inspect。

核心 Python 使用 ≥3.11。本机已知可用的应用 Python 3.12 路径如下，执行前 S0 再确认；也可以指定用户自己的兼容解释器。xlsx 可选依赖只在 export-tables 时检查。

```bash
cd /Users/xiechonghang/Workspace/LLMarchi_excel
READER_PYTHON="/Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"

# 识别输入、版本与当前能力；不生成正式阅读页
"${READER_PYTHON}" structure_reader/scripts/read_model_structure.py inspect \
  --model deepseek_v4_flash_0731

# 计划命令，尚未实现：提取、核验、阅读页、CSV 与报告一次生成
"${READER_PYTHON}" structure_reader/scripts/read_model_structure.py build \
  --model deepseek_v4_flash_0731 --scenario-set standard
```

用户拿到输出后：先打开返回路径下的 `README.md` 或 `viewer/structure.html`；阅读源码证据/场景解释看 reports；进一步做表格处理看 tables。模型输入、工程库代码和 HTML 模板都不是用户要打开的最终结构图。

高级命令在工程实现后提供：

```bash
# 使用 build 返回的真实 analysis_dir，不照抄这个占位路径
READER_ANALYSIS_DIR="/absolute/path/from-build-output"

"${READER_PYTHON}" structure_reader/scripts/read_model_structure.py validate \
  --analysis-dir "${READER_ANALYSIS_DIR}"

"${READER_PYTHON}" structure_reader/scripts/read_model_structure.py render \
  --analysis-dir "${READER_ANALYSIS_DIR}"

"${READER_PYTHON}" structure_reader/scripts/read_model_structure.py export-tables \
  --analysis-dir "${READER_ANALYSIS_DIR}" --format xlsx \
  --granularity semantic --instance-mode representatives
```

render/export-tables 重新确认核验状态与适用输入，不允许绕过检查。默认输出根为工程的 outputs，基于工程位置解析而非任意工作目录；analysis-dir 使用 build 返回路径。CLI 失败退出非零并报告原因/失败记录位置；用户无需从日志猜是否完成。

`user_guide.md` 必须给环境确认、两模型命令、产物导航、场景读法和常见失败解决方式；`developer_guide.md` 给新增适配/基线更新/测试；`output_reference.md` 给 JSON/表格字段与证据。三份文档各负其责，工程 README 只做快速入口与链接。

## 8. 节点推进、工作量与完成标准

```text
S0 → S1 → S2 → S3 → S4 → S5 → S6 → S7 → S8
                    核验从 S1 起持续，不另等“可信阶段”
```

- 第一里程碑 S0–S3：单个 CSA·TopK 代表闭环，目录/接口/证据/核验已具备。
- 第二里程碑 S4–S6：完整 DeepSeek 特征、可读页面与同源表格。
- 第三里程碑 S7–S8：经 Qwen3 迁移验证的首版与 Skill。

粗略工作量约 **12–20 人日**，包含 v2 遗漏的 DSpark 角色、条件/状态 IR、语义证据与独立核验；不是交付日期承诺。S3 完成后按实际提取/审查/阅读成本重估。运行追踪和目标 CUDA 数值验证不计入此估算。

阶段状态为 pending/in_progress/passed/failed；只有实测检查与实际产物齐全才 passed。用户阅读反馈到来前，相关节点保留“工程检查已通过、阅读验收待完成”的记录，不替用户勾选。

首版完成必须同时满足：目录职责/命名清楚、一个入口可复现、43 层分区和 DSpark 角色正确、关键连接有证据、轴含义/来源完整、条件和缓存状态诚实、展开不丢边、CSV/可选 xlsx 同源、未知模型正确提示适配、两模型回归通过、文档与实际命令一致。

## 9. 后续扩展与资料

后续运行证据可接入 FX/运行采集、shape 检查与目标设备数值验证，沿用同一 IR 与来源标识，分别生成新的证据产物；不会把已有源码图直接升级成“已运行”。L3 底层算子/ONNX、多模态与其他机制按独立适配矩阵推进。

- [工作区入口](../README.md)：当前目录用途。
- [HF 资料与依赖关系](README.md)：下载模型与软件代码的关系。
- [方法说明与工具评估](model_structure_visualization.md)：通用方法和远期采集工具。
- [v2 审查记录](archive/model_structure_plan_v2_review.md)：本次修正依据。
- [Qwen3-0.6B 官方配置](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)：S7 固定具体 revision 后使用。

执行起点为 S0。后续新增脚本、工程文件与工作产物均按本版路径和职责落位。
