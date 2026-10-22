# 项目计划 v2 审查：修正项与完善建议

审查日期：2026-10-11。对象：[历史计划 v2](model_structure_plan_v2.md)，并对照 [v1](model_structure_plan_v1.md)、[方法说明](../model_structure_visualization.md)、[资料索引](../README.md) 与本地 DeepSeek 官方参考源码。意见已整合到 [当前计划 v3](../model_structure_plan_v3.md)。本次执行的是文档与静态源码核对，没有运行模型。本文是历史审查记录，不是工程已完成的声明。

## 1. 判断

**保留 v2 的主路线：源码与配置驱动、特征结构代表、轻量离线 HTML、工程稳定后封装 Skill。开工前应修正四类结构错误，并补齐静态证据、条件分支与图数据规范。**

把运行追踪和数值验证后移符合当前阅读目标。重复层不铺开也合理。但“代表层较少”只减少重复处理和展示，不会同比例减少对分支、状态和源码的核对工作。当前计划中的“同构核验”“自动提取”和“可信”需要更精确地限定能力。

## 2. 必须修正的结构问题（P0）

### P0-1：CSA · TopK 数量应为 20

v2 第 2.1 节把层 4、6、…、42 写成 21 层。实际为 `(42-4)/2+1=20`，当前表相加为 44，与主干 43 不一致。

本次用根配置的 `compress_ratios[:num_hidden_layers]` 与 `num_hash_layers` 独立计算，得到：

- SWA · Hash：0、1，共 2。
- CSA · Hash：2，共 1。
- HCA · TopK：3、5、…、41，共 20。
- CSA · TopK：4、6、…、42，共 20。
- 合计 43；CSA 总数为 1+20=21。

**建议验收**：每个主干层恰好属于一个组；组集合不重叠、无缺失、并集等于 `range(43)`；计数从 layers 列表生成，不手填。示例 JSON 中的字符串 `"..."` 只能用于说明，真实产物必须展开成完整整数列表。

### P0-2：DSpark 不能把完整 stage 统一成“1 代表 ×3”

v2 第 2.1/2.2/4.1 节与 T2b 将 3 个 DSpark stage 归为一个代表。源码存在 stage 位置条件：

- `stage_id == 0` 才创建 `main_proj/main_norm`，并由首 stage 提供 `forward_embed` 的主干特征投影与草稿输入。
- 最后 stage 才创建最终 norm、Markov、confidence、hc_head 参数，并提供出口计算。
- `Transformer.forward_spec` 明确先调用首 stage 的 `forward_embed`，再遍历 stage，最后调用末 stage 的 `forward_head`。
- 主干目标层 40/41/42 到草稿入口存在额外跨层读出；不是仅顺序通过三个相同节点。

依据：[DSparkBlock](../../models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/model.py) 的 822–874 行与 `Transformer.forward_spec` 的 929–936 行。

**推荐表示**：共享的 `DSparkBlock.forward` 内核模板可以复用；外层保留入口适配、stage 绑定与末端 head 差异，并显示主干读出和共享参数。或者分成 first/middle/last 三种角色代表。若未来仅有 1 stage，必须允许首尾角色同时成立，不能用互斥分类丢掉出口。

**建议验收**：首末组件的条件与调用位置匹配；共享 embed/head 有同一参数标识；主干读出层号可见。判定同构时必须纳入入口/出口方法及 stage 条件，不能只比较 Attention × MoE。

### P0-3：mHC 契约把汇聚与残差混合混淆了

v2 第 6.1 节要求“mHC comb 的输出 shape 必须是 [B,T,H]”。无论 comb 指矩阵，还是其作用后的残差流结果，这个要求都不正确：

```text
输入 streams                         [B,T,N_hc,H]
pre 汇聚后的模块输入                 [B,T,H]
post 系数                            [B,T,N_hc]
comb 矩阵                            [B,T,N_hc_src,N_hc_dst]
comb 混合后的 residual streams       [B,T,N_hc_dst,H]
post 扩展输出 + 混合 residual         [B,T,N_hc_dst,H]
```

`hc_post` 的方向是 `out[...,k,d] = post[...,k]*x[...,d] + sum_j comb[...,j,k]*residual[...,j,d]`。其中 source/destination 流轴都有长度 4；反转它们后 shape 仍相同，因此仅检验 shape 无法捕获“comb 方向反转”。

依据：[参考源码](../../models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/model.py) 的 `hc_pre`（680–688）、`hc_post`（690–693）和 `Block.forward`（695–707）。

**建议修正图和契约**：Attention 与 MoE 各有一次完整的 hc_pre → Norm → 模块 → hc_post；在 comb 的轴标注与边属性中保存求和轴和映射方向。第 2.2 节目前的“mHC post/comb → MoE → 残差”省略了第二次 hc_pre、ffn_norm 与 hc_post，应补齐。

L0 也应显示 `Embedding → HC Expand → 主干 → HC Head → Final RMSNorm → LM Head`，避免从三维 Embedding 输出直接进入四维主干，又从四维主干直接进入普通 LM Head。

### P0-4：Attention 的连接与反向 RoPE 顺序需要改正

v2 第 2.3 节写成一条串联路径，容易把 Q→KV→Indexer 误画成顺序依赖；并把反向旋转写在输出投影之后。

正确阅读关系依据 `Attention.forward`（490–548 行）：

- 同一个输入 x 分支产生 Q、窗口 KV、主压缩分支和 Indexer 的相关输入；Indexer 还读取 Q 的低秩中间状态 qr。
- Q 和供注意力消费的共享 KV 并行输入注意力；Indexer 输出选择索引，不输出 Value。
- HCA 没有学习式 Indexer Top-K，但仍有因果可见压缩位置选择；不能将“仅 CSA 的 Indexer”泛化成“所有变体都经过学习式 Top-K”。
- 注意力输出 **先在 RoPE 切片上做反向旋转（539 行），再 reshape/分组低秩 O 投影（542–547 行）**。

**建议验收**：检查相应生产者—消费者端口与边序，不只检查“这些节点存在”。删除 qr→Indexer、把 KV 接在 Q 输出之后、把 inverse RoPE 移到 O 投影之后，都应被受控错误测试检出。

## 3. 静态工程最需要补齐的能力（P1）

### P1-1：场景不是只换 shape 数字

v2 第 3 节称“场景值不影响图拓扑”。这只能描述保存了所有条件的静态模板，不能描述实际走过的路径。

参考源码存在：prefill/decode 分支；Compressor 是否凑满一块并提前 return；窗口回绕；DSpark prefill 仅建 KV 而不走完整草稿计算。默认 P=127、T=1 恰好同时到达 CSA 的 4 和 HCA 的 128 压缩边界，不能代表普通 decode。

**建议二选一**：

- 首版保存静态模板与条件，在图中明确 `guard`；场景切换只计算已知条件并突出相应路径。总模板可不变，激活节点/边不同。
- 或分别产出 prefill、decode 非压缩边界、decode 压缩边界的场景图。

至少给节点/边增加 `guard`、`active_for_scenario`（true/false/unknown）、条件来源。Unknown 不是 false；不执行模型时，MoE 的具体专家选择和数值必须保持未知。

状态信息也需区分：历史逻辑长度、已完成压缩条目、待压缩尾部、overlap、Indexer 状态、物理容量。可从条件与已知合法历史推导出的值应标为“推导”，不能叫“运行快照”；缺初始状态时不强行补数值。

### P1-2：锚点存在不等于锚点支持结论

v2 示例的 `q_lora_proj` 锚点 `model.py:412-418` 实际落在 Indexer 路径附近，并不是 Attention 的 q_a 投影。行号都存在，但依然指错计算。

哈希机制也有缺口：如果 inventory 每次记录当前源码，仅比较“源码与刚生成的 inventory 相同”，更新源码再重扫会通过；仍需规则声明它**审查过的源码基线**。

**建议证据字段**：`source_id`、预期内容哈希、`qualified_symbol`（如 `Attention.forward`）、实际行区间、表达式匹配/摘要、该证据支持的断言。由符号定位生成行号，减少写死行号的维护成本。Python 标准库 AST 已提供源码行范围等信息，适合做有限匹配。[Python AST 官方文档](https://docs.python.org/3.9/library/ast.html)。

第一版不必实现通用 Python 数据流分析：在适配范围内识别赋值、调用参数与关键分支，并人工审查复杂连接，然后固化具名规则。边本身也需要证据，例如 `q` 和 `kv` 在 `sparse_attn(q,kv,...)` 中分别作为哪个参数被消费。

出图前比较当前 source/config 与规则的预期基线；失配则报告重新核对需求。不能把行号存在或 AST 解析成功当成语义验证成功。

### P1-3：同构签名应区分共享模板与实例绑定

“图只存代表、显示时套层号”对普通重复层成立，但还有层相关副作用：状态所属层、权重路径、主干读出目标、共享组件、stage 首尾角色。DeepSeek 的 40/41/42 层存在 DSpark 读出；不能只换画面标题后沿用代表层的状态/参数身份。

建议拆成：

- `template_id`：已核对的局部计算结构。
- `instance_binding`：实际 layer/stage ID、参数路径、state ID、配置属性、角色和跨层读出连接。
- `structure_signature`：实现版本、Attention/MoE 变体及影响局部结构的尺寸/选项。

局部结构相同而模型外部读出不同，可以共用局部模板，并把差异保存在实例绑定；无需因此复制全部内部节点。影响局部结构的差异才拆模板。

T4 建议采用“适配器已识别的结构等价检查”作为能力名称。任意 Python 中的隐含层号条件自动发现没有当前实现保障；未处理结构字段/源码分支应 fail 或单独分组，不应宣称完成通用同构证明。

### P1-4：图是主数据，表是投影

表优先是方便导出的设计目标，但只有“扁平节点行”不足以表达数据流。建议以稳定 IR 保存：

- 节点：调用身份、模块/参数绑定、语义类型、层级、配置属性、来源和条件。
- 端口与 tensor：张量名称、结构化轴、dtype、布局、shape 表达式和来源。
- 边：生产/消费端口、data/state/select/control 类别、对应源码依据。
- 状态：state_id、调用前后版本、slice/view/alias 和读写条件。
- 模板展开：父子图、折叠前后端口映射、granularity。

轴不能只存 `"B=1"` 字符串。建议保存 `symbol=B, name=batch_size, meaning=批大小, value=1, origin=scenario.batch_size`，由渲染器生成短标签；H/Rq 等值保存配置来源，B/T/P 保存场景来源，未知值给原因。

此外，v2 第 4.2 节的 `R=128` 与 v1 的 `R=remainder_tokens` 冲突：窗口用 W、压缩比用 r、尾部用 R，保持唯一字典。

### P1-5：提前定义算子级和 Excel 的含义

v2 将 L3 延后，同时 T5 输出 `operators.csv`。如果当前节点是整个 Attention 或 RMSNorm，一行节点并不是一行底层算子。后续将一个节点展开成子图，还需边、端口映射与粒度标识，不能仅增加 semantic_group 就保证不返工。

建议本期明确：CSV 导出的是**语义节点表**；为未来 L3 预留 `semantic / source_op / aten_op` 等粒度标识与父子映射。实际算子表列入后续能力。

为保留连接，表格至少有 Nodes、Edges、TensorAxes、LayerInstances、Sources/Checks 几张逻辑表；CSV 可分别输出多个文件。展开全层时生成唯一 `instance_node_id`，同时保留 `template_node_id`，不能只改 rep_layer 造成跨层 node_id 重复。

### P1-6：把核验移到每次生成流程内

T4 放在 T3 之后才成为“可信阶段”，容易让错误结构先进入阅读页。建议 T1 已检查分组覆盖，T2a 已检查来源/连接/shape，后续 T4 是增强检查，不是首次开始检查。

生成流程改为：

```text
inventory → classify → extract → validate → render → export
```

Skill 必须调用 validate；未通过时输出错误报告，未审查的中间图可明确标为预览，不作为验收交付。

契约最好与提取器分文件；关键期望由独立源码审查形成 fixture。AI 独立阅读可以补充发现问题，但不是形式证明；第二次 AI 回答也不能自动提升证据等级。若组织独立审查，应保留审查结论的源码依据及版本，图与源码冲突时追到具体表达式裁决。

## 4. 计划中的事实、范围与执行细节（P2）

### 4.1 更新环境事实

系统默认 Python 3.9.6 的检查不能推导“本机没有 Python ≥3.10”。本次通过应用的依赖查询得到可用的 Python 3.12.14，并检查到 openpyxl 可发现、torch 不可发现。未安装或创建环境。

解释器：`/Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`。

静态主线保持零依赖可以作为轻量交付选择；不应把它建立在“只有 3.9”的事实前提上。可默认支持现代 Python，系统 3.9 作为兼容选项；每种解释器分别记录解析能力，现代模型源码可能采用新版语法。当前两份 DeepSeek 主源码本次可由默认 3.9 的 AST 解析，这不是所有模型的保证。

xlsx 使用 openpyxl 即可，不需要为此引入 torch CPU 或先建立 uv 环境。[openpyxl 官方教程](https://openpyxl.readthedocs.io/en/stable/tutorial.html)。原版 CUDA kernel 在当前 Mac 不能原样执行的限制仍成立，不能把可用的 Python 3.12 当成已有模型运行环境。

### 4.2 不要混用 V4.1 大小与 Qwen3 子型号

- v2 第 1.2 节的约 510 GB，在已有资料中来自 V4.1 索引 `total_size=510286023000`，不是已经核对的 V4-0731 checkpoint 大小。当前 0731 无本地权重索引，建议写“全量权重不纳入本期；体积以该模型固定 revision 的文件元数据为准”。
- T6 的“Qwen3：36 层”未指定子型号。此前示例 Qwen3-0.6B 的官方配置为 28 层。[官方配置](https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json)。T6 应选定 model ID/revision 并从 config 取层数；不能给整个 Qwen3 系列固定层数。
- 本期 HF 对照到底是“差异报告”还是“第二套完整图”，第 3 节、T2b 与风险表描述不一致。建议主交付为参考实现图与 HF 差异报告；完整 HF 图单独列扩展工作量。

### 4.3 对任务、分工和工期做小幅修订

- T2 使用 `extract/deepseek_v4_0731.py`，目录规划却没有 extract；增加该目录，并明确 rules 负责声明、extract 负责按规则构图、checks 负责核验，避免三者全塞进 rules 文件。
- 补上 schemas、scenarios、tests 与 CLI 失败报告的预期文件；测试不仅是模型契约，也应包括折叠端口映射、布局、来源和图版本。
- 每个任务补主责、协助与验收责任；可以是同一个开发者的不同职责，不必恢复 v1 的多人组织重量，但也不应删除用户已要求的分工。
- T6 将“核心代码零改动”改为“优先通过新增规则适配；若公用接口缺失允许具名修订，并回归 DeepSeek”。零改动不应迫使适配器绕过公用契约。
- v2 逐项工期相加为约 9–10 人日，表中上界 10.5 包含未说明余量。更重要的是 DSpark 角色、条件/状态 IR、语义锚点和独立检查未明确计入；建议把 9–10.5 保留为原初估算，T2a 完成后按真实图与审核成本重估，不再精确承诺一个新数字。
- 方法文档仍把运行采集 + Model Explorer 写成默认主线，README 将 v1 概括为“全模型逐层展开”；v1 实际已有重复层折叠与代表层。建议准确描述版本差异：v2 主要收缩执行范围、存储/展示粒度与依赖，不把运行证明的成本归因于是否画 43 份图。

## 5. 推荐的修订顺序

1. **开工前**：修正 43 层分组、DSpark 角色、mHC 形状/方向、Attention 反向 RoPE 顺序与并行分支。
2. **T0/T1**：锁定规则审查基线；定义结构签名、模板/实例绑定、分组分区校验与能力报告。
3. **T2a**：保留当前 CSA·TopK 端到端起点，同时实现最小边证据、结构化轴、guard、state 版本与 validate。
4. **T2b/T3**：补主干变体及 DSpark 首尾结构；生成源码与条件可查的离线页；用固定阅读问题验收。
5. **T4/T5**：扩充独立 fixture/变异测试；输出节点、边、轴、实例与证据的同源表格。
6. **T6/T7**：固定一个 Qwen3 子型号做迁移，再固化包含 validate 的 Skill。

这组修改保持 v2 的静态、轻量路线，主要补强正确性边界与可维护性，不需要把完整运行追踪重新拉回本期。
