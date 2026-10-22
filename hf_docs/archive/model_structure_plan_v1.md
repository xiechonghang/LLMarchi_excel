# 模型结构阅读工程：项目计划书 v1（DeepSeek-V4-Flash-0731 案例）

更新日期：2026-10-11。状态：**历史归档。曾被 [计划书 v2](model_structure_plan_v2.md) 替代，当前执行 [计划 v3](../model_structure_plan_v3.md)；维度与场景规范已整合进 v3。**

v1 的建设顺序是"全模型逐层可展开"：以 DeepSeek-V4-Flash-0731 为主案例（替代更早的"先从 Qwen3 开始"方案），8 个里程碑 M0–M8，25–38 人日。后被需求澄清修正为**特征结构展示**（每类结构 1 个代表实例，不逐层循环），见 v2。

## 1. 项目目标与交付边界

输入模型目录、对应源码和一个分析场景，输出可逐层展开、含张量轴含义、可回溯源码的数据流图。默认展示语义结构；RMSNorm 的内部算术、布局变换和量化细节按需展开。

第一版完整梳理 **SWA、CSA、HCA、共享 KV、低秩投影、mHC、Hash/Top-K MoE、Cache 状态和 0731 的 DSpark 分支**。这里的“完整”指这组目标机制及其源码连接覆盖，运行证据按实际能力单独标记。

实现分工：**Skill 组织分析步骤和交付规范；Python 工程产生与验证图数据；图浏览器负责层级阅读；AI 辅助解释与适配。** 每条连接均须携带证据，不能用模块注册顺序或 AI 推测代替数据依赖。

分两种发布等级：

- **结构阅读版**：原始配置的源码结构覆盖、轴与形状推导、状态更新规则；小规模测试提供局部证据。没有执行的原始模型路径仍标为源码核对。
- **目标运行验证版**：在匹配的软件、设备和算子环境中，补充目标配置下的执行与数值证据。真实 checkpoint 验证还需要相应权重；不能把结构阅读版说成已验证训练权重推理。

本计划涵盖推理结构。训练优化器、生产部署性能和完整分布式服务平台另立范围。全量权重下载不是结构阅读版的前置任务。

## 2. 展示规范：让数值有名字、有来源

`hidden_states` 是张量名称，`hidden_size` 是它的一个轴的长度；这两个概念应分别展示。

### 2.1 统一格式

详细标签示例：

```text
层间输入：hidden_states
shape = [batch_size=1, seq_len=1, hc_streams=4, hidden_size=4096]
轴顺序 = [B, T, N_hc, H]
来源：B/T 来自场景；N_hc ← config.hc_mult；H ← config.hidden_size
证据：源码推导；dtype/device 的运行值尚未采集

mHC 汇聚后：attention_input
shape = [batch_size=1, seq_len=1, hidden_size=4096]
变换：沿 hc_streams 轴加权汇聚，保留 batch/sequence/hidden 轴
```

节点与关键边默认显示 `hidden_states [B=1,T=1,N_hc=4,H=4096]`，图旁常驻轴图例；点击端口显示完整英文名、中文解释和来源。仅裸数字 `[1,1,4,4096]` 不满足验收。窄屏可换行，完整含义不能只放在鼠标悬停中。

每个轴记录：符号、规范名称、中文含义、值或表达式、来源配置/场景字段、推导关系、证据等级。每个张量另记录名称、dtype、device、布局与逻辑/物理形状。

### 2.2 DeepSeek 的重点轴

- `B / batch_size`：本次调用的批大小，来自场景输入。
- `T / seq_len`：本次输入 token 数，不是总历史长度。
- `P / past_len`：调用前已处理的历史 token 数；参考实现对应 `start_pos`。
- `L=P+T / total_len`：调用后的逻辑总长度，不等于每个缓存的物理槽位数。
- `H / hidden_size=4096`、`N_hc / hc_streams=4`：隐藏维和 mHC 并行流数。
- `Hq / query_heads=64`、`Hkv / kv_heads=1`、`D / head_dim=512`：查询头、共享 KV 头、每头维度。**不能用 H/Hq 推断 D。**
- `D_rope=64`、`D_nope=448`：位置旋转与非旋转部分；切片位置以具体实现核对，不能只照类注释判断。
- `Rq / q_lora_rank=1024`、`G / output_groups=8`、`Ro / o_lora_rank=1024`：查询低秩维、输出组数及每组低秩维。
- `W / window_size=128`：滑动窗口长度；同时记录缓存容量、有效长度、环形写入位置/视图，不直接画成 `[B,P+T,D]`。
- `r / compress_ratio`：CSA 为 4，HCA 为 128；`C / compressed_entries` 是已完成压缩条目数；`R / remainder_tokens` 是待压缩尾部 token 数。
- `Hi / index_heads=64`、`Di / index_head_dim=128`、`Ki / index_topk=512`：Indexer 参数；Ki 是上限，有效/可见条目数还受历史长度和因果条件限制。
- `E / routed_experts=256`、`K / experts_per_token=6`、`Eshared=1`、`I / expert_intermediate_size=2048`：专家总数、每 token 选中数、共享专家数和专家中间维。
- `N_tokens=B×T`、`N_e / tokens_for_expert_e`：展平 token 数和某专家接收的 token 数；N_e 依赖路由数据，应显示符号或实际观测值。

参考实现的 Q 布局可为 `[B,T,Hq,D]`，Transformers 可为 `[B,Hq,T,D]`；单个共享 KV 头在参考实现中可省略单例轴。浏览器展示实际布局与轴映射，不强制改造成同一种物理形状。

## 3. 已核对的案例基线

### 3.1 输入与版本

- 模型目录：[DeepSeek-V4-Flash-0731](../../models/deepseek-ai/DeepSeek-V4-Flash-0731/README.md)。本地清单声明 revision 为 `7872f01b1d1fe23eabc4c98b48bffcef5a386062`；后续 M0 应核对文件内容与该 revision 的一致性。
- [根配置](../../models/deepseek-ai/DeepSeek-V4-Flash-0731/config.json) 与 [参考运行配置](../../models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/config.json) 分别保留，不能混用字段名或默认值。
- 官方参考实现：[inference/model.py](../../models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/model.py)，用于梳理完整 0731，包括 DSpark。
- HF 实现：[modeling_deepseek_v4.py](../../hflib/transformers/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py) 与 [configuration_deepseek_v4.py](../../hflib/transformers/src/transformers/models/deepseek_v4/configuration_deepseek_v4.py)。本地 commit 为 `536ecc007387a50e77603bb5d92100e9b07514cc`，版本 `5.19.0.dev0`。
- 当前模型目录实际 `.safetensors` 数为 0；尚未做目标模型执行。当前检查到的系统 Python 为 3.9.6，不能直接作为要求 Python ≥3.10 的这份 HF 源码环境；其他环境须在 M0 实测。

### 3.2 必須分别展示的结构

下面层号均从 0 开始，依据本地配置与上述源码整理：

- 主干 43 层：0–1 为纯 SWA；2、4、…、42 共 21 层 CSA；3、5、…、41 共 20 层 HCA。
- 每个主干 Block 都有 MoE；前 3 层使用 Hash 路由，后续为得分 Top-K 路由。因此层 2 是 CSA + Hash MoE，不能仅按 Attention 类型合并所有 CSA 层。
- 主干层间 hidden state 是 `[B,T,4,4096]`；Attention/MoE 接收 mHC 汇聚后的 `[B,T,4096]`。两次 mHC 包含 pre 汇聚、post 扩展及 comb 残差流混合，不能简化成普通 Residual Add。
- Attention 的 KV 由同一个投影产生，并作为 K 和 V 使用；Q 有低秩投影、归一化、展开到查询头的路径，O 有分组低秩投影。这里不能套用 Q/K/V 三个独立 Linear 的通用图。
- CSA：滑动窗口 KV + 带重叠的压缩 KV + 独立 Indexer 压缩状态 + Top-K 选择。Indexer 输出是选择索引/掩码依赖，不是新的 Value 特征。
- HCA：滑动窗口 KV + 更高比率的压缩 KV，没有 CSA 的学习式 Indexer；压缩条目仍需满足因果可见性。参考实现虽然也调用 `sparse_attn` kernel，但逻辑上读取可见的压缩条目，不能因此把 HCA 标成学习式 Top-K CSA。
- 压缩分支有待完成块的 KV/score 状态；CSA 还有 overlap 状态。达到压缩边界才新增完整条目，不是每次 decode 都新增一个条目。
- 共享 KV 的部分 RoPE、Attention 输出的反向旋转、sink、量化模拟均需保留为节点或明确属性。

### 3.3 0731 的 DSpark 与实现差异

官方模型卡说明 0731 附带 DSpark 推测解码模块。[官网模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)。本地参考实现里：

- `compress_ratios` 长度为 46，前 43 项用于主干，最后 3 项为 0，对应运行配置中 3 个 DSpark stage。
- 根配置 `num_nextn_predict_layers=1`，参考配置 `n_mtp_layers=3`；必须记录两份配置各自的消费方，不能将两者直接视为等价字段。
- DSpark 读取主干目标层 40、41、42 的隐藏状态，按 hc 流求均值后拼接；包含草稿输入、3 个 stage、共享 embed/head、Markov 和 confidence 分支。
- 本地 HF 实现对 `mtp.*` 设置加载忽略规则，源码中没有上述 DSpark 类。因此 HF 主干图和官方参考实现的完整 0731 图必须是两个带版本标签的分析对象，不能合并后宣称 HF 已实现 DSpark。

DeepSeek 官方报告用于解释 CSA/HCA/mHC 概念；0731 的具体 DSpark 结构以该版本配置与参考源码为准。[官方技术报告](https://arxiv.org/html/2606.19348v1)。

## 4. 工程架构与分工

```text
Skill（选择对象、场景、执行顺序、证据和交付规范）
  ↓ 调用统一 CLI
配置/版本探测 → 模型 Adapter → 源码/追踪采集 → 统一图 IR
                                               ↓
                                  语义归组 + 轴注释 + 状态建模
                                               ↓
                                  验证报告 + 浏览器 + 阅读说明
```

角色按职责分配，单人也可以按顺序完成；以下不表示已安排多人或已启动子代理：

- **模型适配负责人**：核对配置、源码、分支及公式，维护 DeepSeek adapter 和实现差异记录。
- **图与采集负责人**：维护 IR、FX/运行采集、tensor 版本/别名、状态端口和追踪能力检测。
- **展示负责人**：层级浏览、端口与轴标签、来源侧栏、状态交互、折叠映射和离线输出。
- **验证负责人**：维护边界场景、独立数学参考、跨场景状态验证、迁移回归和覆盖报告。
- **Skill 集成负责人**：稳定后封装调用流程、失败降级和阅读模板。
- **用户/阅读验收人**：在 M4/M8 阅读产物，判断是否能快速回答结构问题；不承担手工补图或调试采集的责任。

工程建议新建在 `model_structure_reader/`。下面所有工程路径均为**计划文件**，尚未创建；`artifacts/` 是运行时生成内容。

```text
model_structure_reader/
  pyproject.toml
  README.md
  src/model_structure_reader/
    cli.py
    inventory.py
    ir/{graph.py,tensor.py,provenance.py}
    adapters/{base.py,deepseek_v4_reference.py,deepseek_v4_hf.py,qwen3.py}
    capture/{capabilities.py,source.py,runtime.py}
    semantics/{axes.py,layout.py,grouping.py,deepseek_v4.py}
    validation/{graph_contracts.py,shape_contracts.py,state_checks.py,numerics.py}
    export/{model_explorer.py,report.py}
  schemas/semantic_graph.schema.json
  scenarios/deepseek_v4_0731.yaml
  ui/{package.json,src/AxisLegend.tsx,src/PortDetail.tsx,src/StatePanel.tsx}
  skills/model-structure-reader/{SKILL.md,references/reading-format.md}
  tests/{unit,integration,fixtures,ui}
  artifacts/deepseek_v4_0731/<implementation>/<scenario>/
```

展示优先试用 Model Explorer 的层级/端口能力，`ui/` 只补齐轴解释、状态和证据阅读能力；M0 确认集成形式。如果外部浏览器无法满足核心阅读要求，再用同一 IR 构建轻量浏览页，不把数据格式绑定到某个前端。

## 5. 开发节点与验收

每阶段通过验收才进入依赖它的阶段。工期是单人专注开发的粗略人日估算，供排序，**不是交付日期承诺**；M0 技术试验后应重估。

### M0：固定案例、环境与技术路线（1–2 人日）

- **依赖/分工**：起点；模型适配负责人主责，采集与展示负责人配合。
- **开发任务**：建立版本/文件哈希清单；区分两份配置与实现；探测 Python、PyTorch、设备和自定义 kernel；在一个小 Attention/状态 wrapper 上试 FX 或一个运行采集后端；验证 Model Explorer 自定义图与端口元数据可用。TorchLens/torchview 作为试验候选，第一版不同时维护多个运行后端。
- **预期文件**：`pyproject.toml`、`README.md`、`inventory.py`、`capture/capabilities.py`、`scenarios/deepseek_v4_0731.yaml`、`tests/fixtures/deepseek_v4_0731/inventory.json`。
- **开发产出/功能**：环境能力报告、输入清单、实现差异表和采集/前端选择记录；CLI 能识别对象与缺失能力。
- **开发方式**：先离线读取配置与源码；单独环境安装经过试验的依赖，避免改动 hflib 原仓；记录兼容版本与降级方式。
- **验收方式**：自动报告 43 个主干层与 3 个 DSpark stage 的不同来源；检出无权重；所有输入有路径与内容哈希。选定后端产生带输入输出端口的最小样例；无法追踪处有具名限制，不能把“导入成功”算采集成功。

### M1：图 IR 与轴语义规范（2–3 人日）

- **依赖/分工**：M0；图与采集负责人主责，验证负责人审查。
- **开发任务**：定义图、节点、调用 ID、端口、tensor/状态版本、alias、轴、来源、证据与覆盖范围；区分数据/状态/选择依赖；规范符号 shape 与实际 shape。
- **预期文件**：`ir/graph.py`、`ir/tensor.py`、`ir/provenance.py`、`schemas/semantic_graph.schema.json`、`semantics/axes.py`、`tests/unit/test_graph_contracts.py`、`tests/unit/test_axis_annotations.py`。
- **开发产出/功能**：可校验的 `semantic_graph.json` 与统一维度字典；提供长标签、短标签和轴来源查询。
- **开发方式**：用独立小图测试重复调用、布局变化、KV 别名、状态读写；未知轴输出 `unknown` 与原因，不猜名字。
- **验收方式**：轴数量与 rank 一致；维度数值/表达式均有来源；每条边端口有效；状态版本可跨调用关联。错误连接、缺轴、错误 merge 在契约测试中被检出。未知不等于已验证。

### M2：DeepSeek 源码适配与语义结构（3–4 人日）

- **依赖/分工**：M1；模型适配负责人主责，验证负责人独立核对。
- **开发任务**：实现官方参考/HF 两个 adapter；识别 SWA/CSA/HCA × Hash/Top-K MoE 层型；建立低秩 Q/O、共享 KV、mHC、Compressor、Indexer、专家和状态的来源映射。先完成主干；记录 DSpark 接口，内部展开在 M6 完成。
- **预期文件**：`adapters/base.py`、`adapters/deepseek_v4_reference.py`、`adapters/deepseek_v4_hf.py`、`capture/source.py`、`semantics/deepseek_v4.py`、`tests/fixtures/deepseek_v4_0731/source_contracts.json`、`tests/integration/test_deepseek_structure.py`。
- **开发产出/功能**：主干概览、代表层 0/2/3/4 的源码语义图；配置映射和源码锚点；显示主干到 DSpark 的待展开接口。
- **开发方式**：显式模型规则配合源码符号定位；人工核对函数内部生产/消费关系后固化规则，保存证据与适用版本。不是建设一个能理解任意 Python 的静态编译器。
- **验收方式**：层数与 2 SWA/21 CSA/20 HCA 分布正确；层 2 的 Hash 属性正确；mHC 的 pre/post/comb 与流混合方向正确；CSA 有 Indexer，HCA 无 Indexer；HF 图明确 DSpark 未实现。关键节点和连接有代码区间/符号锚点，源码变化导致规则失配时显式报错。

### M3：形状、数据连接与 Cache 追踪（4–6 人日）

- **依赖/分工**：M2；图与采集负责人主责，模型适配/验证负责人配合。
- **开发任务**：接入 M0 选定的一个运行后端；对函数式操作、in-place 更新和 Cache 提供 wrapper；采集 shape/dtype/producer-consumer；实现逻辑缓存、容量、有效条目、尾部与 overlap 状态映射。
- **预期文件**：`capture/runtime.py`、`semantics/layout.py`、`validation/shape_contracts.py`、`validation/state_checks.py`、`tests/integration/test_compression_boundaries.py`、`tests/integration/test_cache_aliases.py`。
- **开发产出/功能**：prefill/decode 分场景的图与状态快照；原始配置的符号形状推导；可执行子模块/缩小配置的观测形状和证据覆盖报告。
- **开发方式**：先单模块/单层，预算允许再扩大；meta/fake 可用就采集形状，自定义 kernel 无支持则保持显式语义节点并输出限制。缩小配置与原始配置生成不同 artifact，禁止混用数值。
- **验收方式**：覆盖第 7 节的压缩边界与窗口回绕；压缩条目只在应完成的边界增加；识别共享 KV 的别名与 Indexer 独立状态；可核对选中位置与因果掩码。采集失败必须降级证据并保留原因，不能填补伪造运行边。

### M4：可读浏览器与标签体验（3–4 人日）

- **依赖/分工**：M1 后可开始界面骨架，完整验收依赖 M3；展示负责人主责，用户参与阅读验收。
- **开发任务**：适配 Model Explorer 或 M0 确认的浏览页；整体→层→Attention/MoE/mHC→底层四级展开；轴图例/端口细节/来源入口；按场景显示状态与证据；折叠重复层和专家。
- **预期文件**：`export/model_explorer.py`、`ui/package.json`、`ui/src/AxisLegend.tsx`、`ui/src/PortDetail.tsx`、`ui/src/StatePanel.tsx`、`tests/ui/reader.spec.ts`。
- **开发产出/功能**：DeepSeek 主干阅读页面；点击节点可解释输入来源和输出去向；点击布局边可见轴变换；源码与运行证据分层呈现。
- **开发方式**：以 M2/M3 的真实 IR 驱动界面；节点展开前后保存端口和连线映射；MoE 默认显示专家模板与路由扇出，不铺满 256 个重复专家。
- **验收方式**：节点和关键边没有裸数字 shape；常驻图例和完整轴含义可访问；可区分 T/P/L、容量/有效长度；折叠不丢旁路/状态/选择边；桌面与窄屏无文字遮挡。阅读人不读源码即可回答“CSA 的选择索引从哪来、KV 存哪里、mHC 如何汇合、Hash 与 Top-K 如何区别”。

### M5：独立验证与差异报告（3–5 人日）

- **依赖/分工**：M3；验证负责人主责，模型适配负责人处理失败。
- **开发任务**：建立独立数学参考与状态验证；验证 mHC 方向、压缩 gate/softmax 轴、CSA overlap、MoE gather/scatter 与共享专家汇合；给每个节点/边标记证据等级和场景覆盖。
- **预期文件**：`validation/graph_contracts.py`、`validation/numerics.py`、`export/report.py`、`tests/unit/test_mhc_reference.py`、`tests/integration/test_moe_reference.py`、`tests/integration/test_prefill_decode_equivalence.py`。
- **开发产出/功能**：`validation_report.json`、`coverage.json`、`reading.md`；明确实际执行对象、误差指标、测试输入与未覆盖范围。
- **开发方式**：固定输入/种子及有效路由表；BF16/FP8/FP4 分别记录容差与实现范围；核对 prefill 后续 decode 与可比较的分块执行状态。跨实现比较先对齐权重、布局、位置编码与量化策略，再判断误差。
- **验收方式**：错误 comb 方向、错误压缩 axis、提前新增 cache 条目、删除 shared expert 等受控错误可被测试检出。数值报告有 dtype 对应的事先规定阈值，NaN/Inf 或超阈值失败；纯 shape 测试不得标成数值通过。局部数值通过不能提升未执行的全模型证据。

### M6：0731 DSpark 完整结构与硬件验证接口（4–6 人日）

- **依赖/分工**：M2/M3/M5；模型适配负责人主责，采集和验证负责人配合。
- **开发任务**：展开主干 40/41/42 到 DSpark 的跨层读出、草稿 token/noise、3 个 stage、主干 KV 输入、Markov 与 confidence 分支；区分 prefill 建缓存与 decode 草稿计算；建立可选目标设备执行入口。
- **预期文件**：`adapters/deepseek_v4_reference.py` 的 DSpark 部分、`scenarios/deepseek_v4_0731.yaml` 的 DSpark 场景、`tests/integration/test_dspark_structure.py`、`tests/integration/test_target_runtime.py`、`README.md` 的硬件验证说明。
- **开发产出/功能**：完整 0731 源码图与实现差异报告；可展开 DSpark，主干共享参数/跨层连接可见；设备具备条件时补充目标模块运行报告。
- **开发方式**：源图先完成，子模块执行随后验证。目标自定义 kernel 要求 CUDA 等匹配环境；本机缺条件时记录为待目标设备验证，保持结构阅读产物可用。
- **验收方式**：46 项压缩配置不被误画成 46 层主干；参考配置 3-stage 与根配置字段差异有解释；prefill 不产生不存在的草稿执行路径；共享 embed/head 不复制成不相关参数。完整结构版验收必须含 DSpark 来源和跨层连接；目标执行子项无硬件时明确“未通过/未执行”，不得伪装成完整运行版通过。

### M7：将成熟流程封装成 Skill（2–3 人日）

- **依赖/分工**：M4/M5/M6 与 CLI 行为稳定；Skill 集成负责人主责，验证负责人审查失败路径。
- **开发任务**：固化输入识别、adapter/场景选择、工程调用、图与报告输出、证据检查和阅读说明；只引用已实现的命令；缺 adapter 时输出适配需求而非臆造图。
- **预期文件**：`skills/model-structure-reader/SKILL.md`、`skills/model-structure-reader/references/reading-format.md`、`cli.py`、`tests/integration/test_skill_workflow.py`。
- **开发产出/功能**：项目内可复用 Skill；一次请求生成模型概览、场景图、来源、覆盖和局限说明。
- **开发方式**：工程提供 `inventory / analyze / validate / render` 等职责清楚的接口，具体语法在实现时确定；Skill 指向工程入口，不复制模型计算逻辑。个人技能目录安装另作为使用步骤。
- **验收方式**：在新输出目录重跑成功；同时测试无权重、无 CUDA、版本失配和不支持模型。结果如实标明证据等级，不把源码图包装成自动运行图；无需用户手工编辑中间 JSON 才能完成流程。

### M8：迁移验收、文档与发布（3–5 人日）

- **依赖/分工**：M7；验证负责人主责，模型适配/展示/Skill 集成负责人配合，用户做最终阅读验收。
- **开发任务**：迁移到 Qwen3、一个常规稠密 Decoder 和一个非 DeepSeek 的 MoE；选带不同 Cache/Attention 机制的案例；检验公用 IR/采集/浏览器能否复用。
- **预期文件**：`adapters/qwen3.py`、其他已选案例 adapter、`tests/integration/test_model_matrix.py`、`tests/fixtures/<model>/`、`README.md`、`hf_docs/model_structure_visualization.md`、`artifacts/*/coverage.json`。
- **开发产出/功能**：支持矩阵、可复现示例、使用说明和首版发布包；保留 DeepSeek 回归产物。
- **开发方式**：新模型主要补 adapter/规则，公用代码有变动需回归；可使用小规模测试，但报告原始与测试配置差异。
- **验收方式**：新增模型不用改 IR schema 或浏览器核心即可表达关键机制；不同 Q/K/V、GQA、普通残差、MoE 均能正确展示；DeepSeek 测试仍通过。对非 Decoder、多模态、SSM 等未测机制明确未支持，不能仅因 DeepSeek 通过就宣称通用。

## 6. 里程碑、依赖和工作量

```text
M0 → M1 → M2 → M3 → M5 → M6 → M7 → M8
       └── M4 界面骨架 ──┘     ↑
              M3 后验收 ──────┘
```

- **里程碑 A，M0–M2**：输入与证据规范稳定，DeepSeek 主干源码图可生成。
- **里程碑 B，M3–M5**：主干可交互阅读、轴含义与 Cache 路径可核对，局部追踪/数值报告可生成。
- **里程碑 C，M6–M7**：完整 0731 包括 DSpark，Skill 可重复调用工程。
- **里程碑 D，M8**：经其他模型验证的可复用首版。

串行估算合计 **25–38 人日**。允许多个职责并行时，M4 和部分验证工作可穿插；实际日历工期取决于人员、环境、上游适配难度与硬件可用性。该估算不含全量权重下载、GPU 采购或完整多卡服务部署。

当前完成的是计划和源码预核对，**M0 仍未验收**。每阶段应记录 `pending / in_progress / passed / failed`；阶段通过需要测试/产物路径，不能凭文字描述勾选。

## 7. 必须覆盖的场景

以下 P 是调用前历史长度。历史状态须由连续输入构建，不能只设置 `start_pos` 却没有对应缓存。

- **SWA**：P=0 的短 prefill；累计长度经过 128；P=127/128 后 T=1，验证窗口边界和环形位置。
- **CSA**：P=0，T=3/4/5/8/9；P=2/3/4 后 T=1，验证压缩闭合与 overlap；不同 chunk 划分下对齐有效状态。
- **HCA**：P=0，T=127/128/129/256；P=126/127/128 后 T=1，验证不完整块与新增条目。小规模代理测试可另选比率，但原始比率 128 的场景必须保留。
- **Indexer**：压缩条目少于 512 的动态有效 Top-K；达到/超过 512 的选择行为；每个 query 的因果可见范围；无可见压缩条目时的空选择。
- **MoE**：层 0/2 的 Hash 与层 3/4 的 Top-K；同一专家收多 token、某专家零 token、有效 tid2eid 表、6 路加权汇合、shared 分支。
- **mHC**：N_hc=4 输入；pre/post/comb 形状、Sinkhorn 行列归一化和 comb 方向；相同隐藏维但不同流值以避免对称输入掩盖方向错误。
- **DSpark**：主干采集→prefill cache→decode 草稿→Markov/confidence；阶段数、草稿长度和采样策略按当前配置/入口分别记录。
- **批量/布局**：B=1 与 B=2；T=1 与 T>1；布局折叠展开后轴顺序一致。参考实现不支持的多 token decode 要标为不支持，不能按 HF 路径强行套用。

## 8. 最终验收清单

1. **可读性**：读者可回答一个张量来自哪里、去哪里、各轴什么意思、哪些值由 config 决定；无需先看 ONNX 算术碎片。
2. **结构正确**：SWA/CSA/HCA、mHC、两类 MoE 和 DSpark 的关键节点、分支、汇合与状态均有来源。
3. **维度正确**：每个关键端口和边含轴名、值/表达式、布局与推导来源；未知值明确标识。
4. **状态正确**：缓存逻辑长度与容量分开；压缩尾部、overlap、Indexer 状态和 in-place/alias 均可追溯。
5. **折叠可逆阅读**：归组保存原子节点与端口映射，展开不丢边；RMSNorm/SwiGLU/布局不能靠外观误识别。
6. **证据诚实**：源码核对、shape 推导/采集、实际执行、数值验证分别记录到节点/边；缩小模型、替代 kernel、模拟量化均可见。
7. **可重复**：同一输入/版本/场景生成可比较的稳定图；图 ID 和测试用例能定位变化。
8. **可扩展**：M8 的不同模型用同一 IR 和展示流程；支持范围写入实际测试矩阵。

## 9. 参考资料与已有文档

- [HF 仓库与模型关系](../README.md)：文件来源、软件依赖与本项目下载情况。
- [模型结构可视化方法](../model_structure_visualization.md)：源码、追踪、语义归组及证据分层的方法。
- [DeepSeek 0731 官网配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/config.json) 与 [官方参考代码目录](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/main/inference)。线上 main 可变，本工程基线以固定 revision/哈希为准。
- [Model Explorer 官方适配器说明](https://github.com/google-ai-edge/model-explorer/wiki/6.-Develop-Adapter-Extension)：层级 namespace、输入输出端口和元数据。
- [PyTorch FX 官方文档](https://docs.pytorch.org/docs/2.14/fx.html)、[meta device 官方说明](https://docs.pytorch.org/docs/2.14/meta.html)：采集能力与限制，在 M0 中对目标环境实测。

DeepSeek 主案例能锻炼共享 KV、压缩、选择、专家路由和复杂状态建模能力；是否足以适配某类新模型，最终由 M8 的迁移证据决定。
