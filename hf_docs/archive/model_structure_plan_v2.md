# 模型特征结构阅读工具：项目计划书 v2（历史归档）

更新日期：2026-10-11。状态：**历史归档，已被 [计划 v3](../model_structure_plan_v3.md) 替代。** 本文保留当时方案，包含后来在审查中发现的问题，不作为开发或验收依据。

本文是当时的执行计划（v2），曾替代 [model_structure_plan_v1.md](model_structure_plan_v1.md)（v1）的里程碑结构与工程分解。替代原因是一次需求澄清：**不需要完整模型的逐层展示，而是特征结构的展示——例如 100 层 decoder 只展示一层的内部，不循环渲染 100 层**。相关规则现已整合进 v3，不再需要引用本版执行。

## 1. 需求定义

### 1.1 需求清单

| 编号 | 需求 | 说明 |
|---|---|---|
| R1 | 特征结构展示（本期核心） | 展示模型整体与内部结构；模型由重复层组成时，**每类结构只渲染 1 个代表实例**，绝不逐层重复 |
| R2 | 整体概览不丢信息 | 紧凑表示必须仍传达：共多少层、每层什么类型、顺序如何；不允许因为"只画一层"而丢失层序与数量 |
| R3 | 连接可溯源 | 每个节点/连接能定位到源码位置；连接依据来自数据流证据，不用模块注册顺序或 AI 推测充当 |
| R4 | 维度可读 | shape 带轴名与来源（沿用 v1 第 2 节规范），禁止裸数字 `[1,1,4,4096]` |
| R5 | 算子级 Excel（下一期） | 结构按算子导出为表格；本期数据格式必须表优先设计，避免事后返工 |

### 1.2 明确不在本期范围

- **运行追踪与数值验证**：本机实测无 Python ≥3.10 环境、无 torch；且 0731 参考实现 `from kernel import sparse_attn, ...` 依赖 CUDA kernel，在这台 Mac 上不可能被运行追踪。本期证据等级止步于"源码核对 + 配置推导"。
- **全量权重下载与真实推理**（约 510 GB）。
- **自研图编辑器/前端框架**：用零依赖自包含 HTML 起步。
- **V4.1（deepseek_v41）展示**：HF 库未注册 `deepseek_v41`（已核实仅有 `deepseek_v4`），且 V4.1 config 是多模态结构（含 `vision_config`、`image_token_id`），`text_config` 已含 `dspark_*`/`engram_*` 字段。列为后续分析对象，本期只在差异记录中说明。

### 1.3 需求对设计的直接推论

1. **展开是渲染策略，不是存储策略。** 图数据只存每类结构 1 份；"×21 层"是层型规划里的一个计数字段，不生成 21 份图拷贝。要看第 17 层？渲染层按 layer_plan 把代表实例套上层号即可。
2. **数据量收益是实打实的。** 0731 若逐层展开约 43 层 × 每层 40 节点 ≈ 1700+ 节点；按特征结构约 4 个代表 × 40 ≈ 160 节点，渲染、校验、阅读成本同比例下降。
3. **同构必须声明并核验，不能默认。** 只有经核验"计算图相同"（权重值不同没关系，结构相同即可）的层才允许归入同一代表。核验机制见 T4。
4. **节点即表行（为 R5 预留）。** 每个节点从第一天起就是扁平记录：算子名、语义组、父层级、带轴 shape、源码锚点全部是顶层字段。Excel 导出只是格式转换，不需要重新解析。

## 2. 展示模型：四层阅读界面

```text
L0 模型概览        ← R2：层型条带 + 分组计数，一眼看清整体
 └─ L1 特征层代表  ← R1：每种结构类型 1 个实例，完整内部展开
     └─ L2 模块内部 ← Attention / MoE / mHC / 压缩分支
         └─ L3 算子级 ←（下一期）对应 Excel 导出的粒度
```

### 2.1 L0 模型概览

以 DeepSeek-V4-Flash-0731 为例（层数与分布已对照 config 核实）：

**层型条带**：43 个刻度排成一行，颜色代表类型，悬停显示"层 17 · CSA·TopK"，点击跳转到该类型的代表层。条带保序，重复但不铺开内部结构——100 层同类模型时就是 100 个同色刻度，视觉上退化为一条带计数标注的横条。

**分组清单**（与条带同数据源）：

| 层号 | 结构类型 | 数量 |
|---|---|---|
| 0–1 | SWA · Hash-MoE | 2 |
| 2 | CSA · Hash-MoE | 1 |
| 3, 5, …, 41（奇数） | HCA · TopK-MoE | 20 |
| 4, 6, …, 42（偶数） | CSA · TopK-MoE | 21 |

外加首尾组件：Embedding → 主干（43 层）→ Final RMSNorm → LM Head。DSpark 的 3 个 draft stage 同样按"1 代表 + ×3"处理。

### 2.2 L1 特征层代表

0731 共 4 个代表实例（上表 4 行各 1 个）+ 1 个 DSpark stage 代表。每个代表展示完整数据流：mHC pre 汇聚 → Attention（含低秩 Q、共享 KV、窗口/压缩分支）→ mHC post/comb → MoE → 残差。每个节点旁显示带轴符号的 shape。

### 2.3 L2 模块内部

Attention 展开为：Q 低秩投影 + 归一化 + 头展开 → 部分 RoPE → KV 共享投影（一个投影产出 K 和 V，不画成两个独立 Linear）→ 滑窗/压缩缓存分支 → Indexer（仅 CSA）→ Top-K 选择 → 稀疏注意力 → 输出分组低秩投影 → 反向旋转。MoE 展开为路由（Hash / TopK 两种）、专家模板（不铺满 256 个）、shared expert、加权汇合。mHC 展开为 pre/post/comb 三段与流方向。

### 2.4 L3 算子级（下一期）

展开 RMSNorm 为 pow/mean/rsqrt/mul、布局边为 reshape/transpose 等。本期图格式已按此粒度可扩展（节点记录含语义组字段），L3 只是增加更深的代表子图，不需要改架构。

## 3. 技术路线

**静态优先，源码 + 配置驱动，规则文件制。**

1. **事实依据**（2026-10-11 实测）：本机系统 Python 为 Xcode 3.9.6，无 3.11/venv/torch；Node v26 可用。0731 参考实现依赖 CUDA kernel 不可本地运行；HF `deepseek_v4` 实现可读可（远期）追踪。因此第一版 100% 静态：纯标准库（Python 3.9 兼容）解析 config 与源码，产出图数据与 HTML。
2. **两个分析对象，分开标注**：官方参考实现 `inference/model.py`（含 DSpark）与 HF `modeling_deepseek_v4.py`（无 DSpark，对 `mtp.*` 有加载忽略规则）是两个带版本标签的图来源，不合并。本期以参考实现为主线（结构最完整），HF 实现作为第二来源出对照差异表。
3. **规则文件制，不做通用 Python 理解器**：每个模型一份规则文件（层型分类规则 + 代表层提取规则 + 语义契约），源码锚点（file:line）写死在规则里；源码更新导致锚点失配时显式报错，不静默出图。
4. **环境**：工程核心保持 3.9 兼容零依赖；另建 uv 环境（Python 3.11 + torch CPU）仅供 T5 的 xlsx 导出与远期运行追踪使用，不阻塞主线。
5. **场景固定**：shape 为符号推导，默认场景 decode（B=1, T=1, P=127）；场景值只影响轴数值展示，不影响图拓扑。场景文件随 inventory 一起版本化。

## 4. 数据设计

### 4.1 layer_plan.json（层型规划，T1 产出）

```json
{
  "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
  "source": {"impl": "reference", "commit": "<hash>", "config_sha": "<hash>"},
  "groups": [
    {"group_id": "swa_hash",   "layers": [0, 1],              "representative": 0,
     "type_key": "SWA+HashMoE"},
    {"group_id": "csa_hash",   "layers": [2],                 "representative": 2,
     "type_key": "CSA+HashMoE"},
    {"group_id": "csa_topk",   "layers": [4, 6, "...", 42],   "representative": 4,
     "type_key": "CSA+TopKMoE"},
    {"group_id": "hca_topk",   "layers": [3, 5, "...", 41],   "representative": 3,
     "type_key": "HCA+TopKMoE"}
  ],
  "dspark": {"stages": 3, "representative": 0},
  "isomorphism_evidence": {"method": "config_crosscheck+source_branch", "report": "isomorphism_report.json"}
}
```

`type_key` 的取值维度就是分类规则的输出：注意力变体（SWA/CSA/HCA）× 路由变体（Hash/TopK）。任何使计算图分支的层号条件（如 `layer_idx < num_hash_layers`）都必须进入 type_key，否则不许归组。

### 4.2 structure_graph.json（代表层图，T2 产出，每类型一份）

节点是扁平记录（表优先，直接对应 R5）：

```json
{
  "node_id": "csa_topk.att.q_lora_proj",
  "op": "Linear",
  "semantic_group": "attention.q_path",
  "parent": "csa_topk",
  "in_shapes":  [{"name": "hc_merged_input", "axes": ["B=1", "T=1", "H=4096"]}],
  "out_shapes": [{"name": "q_a", "axes": ["B=1", "T=1", "Rq=1024"]}],
  "axis_sources": {"Rq": "config.q_lora_rank", "H": "config.hidden_size"},
  "anchor": "inference/model.py:412-418",
  "evidence": "source_review"
}
```

边记录生产者端口、消费者端口、类别（data / state / select），状态类边（缓存读写、Indexer 状态、overlap 状态）单独标类别，不混入数据流。轴命名与来源沿用 v1 计划第 2 节的维度字典（`Hq=64, Hkv=1, D=512, D_rope=64, R=128, Ki=512, E=256, K=6…`），不再重复。

### 4.3 算子表 schema（与 4.2 同源，T5 消费）

每行 = 一个节点：

```
group_id, rep_layer, node_id, op, semantic_group, parent,
input_axes, output_axes, axis_sources, anchor_file, anchor_lines, evidence
```

导出时可选"展开模式"：按 layer_plan 把代表行乘以组内层号（`rep_layer` 换成实际层号），满足将来"全量算子表"的需求——这再次体现"展开是渲染策略"。

## 5. 任务拆解

估时为单人（含 AI 协作）专注开发的粗略人日，供排序，不是交付承诺。

### T0 案例固定与清单（0.5 人日）

- **开发内容**：`inventory.py`，纯标准库；扫描模型目录与两份源码，记录文件哈希、config 字段、源码锚点候选。
- **产出**：`artifacts/deepseek_v4_0731/inventory.json`；能力报告（无 torch、无权重，证据上限=源码核对）。
- **验收**：重跑输出稳定；所有输入有路径与哈希；检出权重缺失与 kernel 依赖（`from kernel import …`）。

### T1 层型分类器（1 人日）

- **开发内容**：`layer_plan.py` + `rules/deepseek_v4_0731.py`；解析 config → 生成 layer_plan（分组、代表层、type_key）。分类规则显式消费 `compress_ratios`、`num_hash_layers` 等字段。
- **产出**：`layer_plan.json`。
- **验收**：分组与手工核对一致（2 SWA + 21 CSA + 20 HCA；Hash 层 = 0/1/2）；未知字段报错不猜；对"每层可变配置数组"（长度 = 层数的字段）逐个检查是否被分类规则消费。

### T2a 代表层端到端（1.5 人日，先打通一条线）

- **开发内容**：只做 `csa_topk` 一个代表：`graph_ir.py`（dataclass）+ `extract/deepseek_v4_0731.py` 的 CSA 部分 + `render/html.py` 最小渲染。这是全流程第一次贯通。
- **产出**：`graphs/csa_topk.json` + 可打开的 HTML（仅此一层）。
- **验收**：每个节点有源码锚点且锚点真实存在（机器校验 file:line 命中）；每条边端口有带轴名 shape；CSA 的 Indexer/压缩分支/Top-K 选择/共享 KV 全部在场。

### T2b 全部代表与模块深度（1.5–2 人日）

- **开发内容**：补齐其余 3 个主干代表 + DSpark stage 代表；L2 模块展开（Attention 内部、MoE、mHC 三段）；HF 实现对照差异表（DSpark 有无、Cache 类差异）。
- **产出**：`graphs/{swa_hash,csa_hash,hca_topk,dspark}.json`、`impl_diff.md`、完整 HTML。
- **验收**：HCA 代表无 Indexer；mHC pre/post/comb 方向与流维度正确（[B,T,4,4096] ↔ [B,T,4096]）；Hash 与 TopK 路由节点不同；共享 KV 是单投影双输出；46 项 compress_ratios 不被画成 46 层。

### T3 阅读页完整体验（1 人日）

- **开发内容**：L0 条带 + 分组清单 + 跳转；代表层切换；轴图例常驻；端口点击显示轴来源；窄屏可用。
- **产出**：`artifacts/deepseek_v4_0731/structure.html`（自包含，file:// 直接打开）。
- **验收**：见第 6.3 节用户阅读问题清单，全部可答。

### T4 同构核验（1 人日）

- **开发内容**：`checks/isomorphism.py`：核验每个分组的"结构相同"声明——(a) config 交叉校验（分类规则消费的字段与层数数组矛盾即报错）；(b) 源码分支核验（使 forward 分支的层号条件必须已进入 type_key）；(c) 远期补充：权重 shape 校验（读 safetensors 头，需权重或 HTTP Range，本期不做）。
- **产出**：`isomorphism_report.json`，HTML 每个分组显示"×21 · 同构依据：config 交叉 + 源码分支条件 x,y,z"。
- **验收**：受控错误（见 6.1）全部被捕获。

### T5 算子表导出（1 人日）

- **开发内容**：`export/table.py`：图 → CSV（标准库）；xlsx 经 uv 环境用 openpyxl（可选）。含展开模式。
- **产出**：`operators.csv`（每代表一节或加 group 列）。
- **验收**：每个节点恰好一行；与 graph 按 node_id 可互相检索；展开模式行数 = Σ组内层数×每层节点数。

### T6 第二模型泛化：Qwen3（1 人日）

- **开发内容**：`rules/qwen3.py`：36 层同构 → 1 代表；标准 GQA + SwiGLU。只写规则文件，不改 IR/渲染/校验核心。
- **产出**：`artifacts/qwen3/…` 全套。
- **验收**：核心代码零改动（diff 证明）；Q/K/V 三分支、GQA 头映射正确展示；0731 回归不变。

### T7 Skill 固化（0.5–1 人日）

- **开发内容**：`skills/model-structure-reader/SKILL.md`：固化"inventory → layer_plan → extract → render"调用流程与阅读规范；只引用已实现 CLI；无规则文件的模型输出"需要适配"而非臆造图。
- **产出**：项目级 skill；新会话一句"展示 <model> 的结构"可走通。
- **验收**：对 0731 与 Qwen3 各跑一次成功；对未适配模型（如 V4.1）正确输出适配需求。

## 6. 核验与审核

### 6.1 机器核验（每次出图自动执行）

1. **锚点校验**：每个 `anchor` 的 file:line 必须命中当前源码文件；源码哈希与 inventory 不符则整体拒绝出图（防源码升级后规则失配静默出错）。
2. **端口校验**：每条边两端端口存在；shape 轴数量与 rank 一致；每个轴有 axis_sources 或显式 `unknown` 标注。
3. **契约校验**（模型语义契约，写在规则文件里，机器逐条检查）：CSA 代表必须含 Indexer 节点而 HCA 不得含；K/V 两端口的生产者必须是同一节点（共享 KV）；mHC comb 的输出 shape 必须是 [B,T,H] 且输入含流轴；Hash 组层号集与 `num_hash_layers` 一致。
4. **受控错误注入**（测试用例，非运行时）：预置 4 个变异图——comb 方向反转、给 HCA 加 Indexer、共享 KV 改双投影、把 [0,1] 标成 CSA——契约校验必须全部报错。这保证"校验器真的能抓错"，而不是只会放行。

### 6.2 AI 自审核（每阶段一次）

独立会话（或子代理）只拿源码与 config，不拿图，回答固定问题清单（Indexer 输出接去哪、压缩条目何时新增、mHC 汇聚方向…），与图数据比对；不一致处生成差异报告，人工裁决后修规则。此步骤防"规则写错了又自己校验自己"的循环论证。

### 6.3 用户阅读审核（T3/T7 的最终验收）

用户不读源码，仅凭 HTML 回答：

- CSA 的 Top-K 选择索引由哪个模块产生？依赖什么状态？
- KV 存在哪里？窗口多大？压缩条目什么时候新增？
- mHC 如何把 [B,T,4,4096] 变回 [B,T,4096]？方向是什么？
- 层 2 与层 4 的 MoE 路由差别在哪？
- Q 从 4096 维到 64×512 经过了哪些中间形状？

全部可答 = 阅读验收通过。答不上来的地方就是图或标签的缺陷，回 T2/T3 修。

### 6.4 证据分级与诚实标记

沿用 v1 的四级证据（源码核对 / 形状追踪 / 运行追踪 / 数值验证）。本期全部产物标注为"源码核对 + 配置推导"；HTML 页脚与每个分组摘要常驻显示该级别；未覆盖项（如量化模拟分支的实际数值行为、自定义 kernel 内部）显式标注"未覆盖"，不用推测补齐。

### 6.5 可重复性

固定 revision/commit 哈希 → 产物含图 ID 与版本字段 → 重跑 diff 为空（除时间戳）。规则或源码任一变化，图 ID 变化，进回归清单。

## 7. 风险与降级

| 风险 | 影响 | 对策 |
|---|---|---|
| 同构声明出错（某层有隐藏的层号分支） | 图错误且被伪装成"×21 正确" | T4 的源码分支核验 + 6.2 独立自审；不确定的层单独成组（多一个代表，可接受） |
| 规则文件写错连接 | 图语义错误 | 契约校验 + 受控错误注入 + AI 自审三道防线 |
| 源码/模型升级 | 锚点失配 | 哈希不符即拒绝出图，提示重新适配 |
| 参考实现与 HF 实现结构分歧被混用 | 图来源不明 | 两来源分别出图、分别标注，差异表强制交付 |
| 纯静态无法表达的机制（数据依赖路由的实际选择） | 展示不完整 | 节点标"数据依赖，静态仅示拓扑"，实际行为留待远期运行追踪 |

## 8. 里程碑与工作量汇总

```text
阶段一（可见结果）  T0 → T1 → T2a → T2b → T3     约 5.5–6.5 人日
阶段二（可信）      T4 → T5                       约 2 人日
阶段三（可复用）    T6 → T7                       约 1.5–2 人日
合计                                              约 9–10.5 人日
```

与 v1 的 25–38 人日相比，缩减来自：不做运行追踪与数值框架（v1 的 M3/M5 大部）、不做 React UI（v1 M4 大部）、不做多角色分工文档；质量由三道核验防线而非流程重量保证。v1 的 M3/M5/M6（运行追踪、数值验证、DSpark 目标硬件验证）作为远期阶段保留在原文档中，待 uv 环境与目标设备就绪后按需启动。

## 9. 与既有文档的关系

- [model_structure_visualization.md](../model_structure_visualization.md)：方法论与工具评估，继续有效；其中 FX/TorchLens/Model Explorer 部分属远期运行阶段。
- [model_structure_plan_v1.md](model_structure_plan_v1.md)（v1）：**里程碑结构由本文替代**；其第 2 节维度字典与第 7 节场景清单继续作为本计划 T2/T4 的规范引用。
- [README.md](../README.md)：工作区文档索引。
- 工程建议新建在 `structure_reader/`（计划路径，尚未创建）：

```text
structure_reader/
  pyproject.toml
  README.md
  src/structure_reader/
    cli.py
    inventory.py            # T0
    layer_plan.py           # T1
    graph_ir.py             # T2 数据结构（扁平、表优先）
    rules/
      deepseek_v4_0731.py   # T1/T2：分类 + 提取 + 语义契约
      qwen3.py              # T6
    render/html.py          # T2a/T3
    checks/{isomorphism.py, contracts.py}   # T4 + 6.1
    export/table.py         # T5
  artifacts/                # 运行产物（inventory/graphs/html/报告）
```
