# T12 正式交接：Decoder 数据流图 vNext

用户已授权开启本版本，并要求实际产品开发交给 dsh。你在安装版 DeepSeek Harness 的新 LLMarchi_excel 会话中执行 **T12 / GitHub #13**，本票完成后等待 Codex 复核。

## 1. 接手与保护基线

先执行并核对实际 cwd `/Users/xiechonghang/Workspace/LLMarchi_excel`、HEAD `947ae78f120279403ec0a4c539e99d3d8869e1d6`，读取项目 AGENTS.md 和它指向的领域文档。仓库已有大量未提交开发，HEAD 不是当前实现的完整身份。

接手快照：`/var/folders/cf/v8s2g_0n0_5c9x9brbcy1nw00000gn/T/llmarchi-T12-decoder-vnext-baseline-20261030-g8bu9m2t/manifest.json`，2026-10-30 18:03:26 +08:00，816 个跟踪/未忽略文件，14814 个受保护历史产物，当前五模型候选共 40 个文件另有副本。先核对 manifest、文件哈希、已有缺失路径和工作区状态，将结果落入本票证据目录。对差异区分本票接手前改动与快照后的外部协调更新；不恢复/覆盖用户改动。

`docs/tasks/model-structure-2026-10-28/{execution.json,tickets.json,monitor-state.json,README.md,executions/}`、任务单及本交接单由 Codex 协调，本票只读，快照后外部更新属于正常协调。自己的报告与 T12-decoder-verify 目录可写。保留既有删除、旧候选、五模型已验收工作簿、单算子模板和已认可独立讨论稿；历史文件按哈希保护。不要 commit/push、重置/清理工作区、关闭 Issue、派下票、恢复旧会话或自动化。

## 2. 读取范围与实施

按顺序读：

1. `docs/tasks/model-structure-2026-10-28/12.md`：本票唯一完整验收条目，GitHub https://github.com/xiechonghang/LLMarchi_excel/issues/13 。
2. `docs/requirements/decoder_dataflow_refinement_2026_10_30.md`：用户认可的粒度、参数全称、单向紧凑布局及保守化简边界。
3. 既有实现和相关测试，优先复用 `extract/deepseek_v4_hf.py`、已有模型事实/构建/来源身份/详情和 `display_names`；产品入口与 Excel 导出本票保持旧行为。

先说明复用/扩展/新增接口，再实施模型无关的 raw/normalized 语义投影和有前提的保守化简，覆盖 DeepSeek 四种真实层变体。`kv_proj` 和 RMSNorm 分离；保留具名 Linear、全称维度、分支/缓存/类型语义与旧节点/源码对应。不能证明的化简保留并解释；规则范围以 12.md 为准。

底层预览使用 **/archify**。读取可用 Archify 技能后，从同源语义数据程序生成 KV 与完整 Attention 的阶段产物，遵循已认可的垂直主链、短 cos/sin 旁枝，无空的 01/02 分区；原讨论稿仅作样式参照，不是输入模型事实。不要修改安装技能包或把本机个人路径硬编码进产品；工具可发现/配置，工具/规则/语义数据哈希进入产物身份。

来源参考：本地 `hflib/transformers` 是实际 HF 检出，commit `536ecc007387a50e77603bb5d92100e9b07514cc`。DeepSeek 源码在其 `src/transformers/models/deepseek_v4/modeling_deepseek_v4.py`，kv_proj/kv_norm 787–788，路径 807–822，RoPE 339–356，同一 KV 送 K/V 846–850；实施时再核对实际文件与 SHA。不要用根 HEAD 假称未提交解析/渲染改动已提交，也不要伪造 Archify repo 证据。

## 3. 验证与交付停止点

每条启用规则有独立正反例与确定性小张量/索引映射验证，保持静态模型分析，无需实例化/权重。按本票和受影响模块做有意义回归；旧 ONNX 缺依赖、旧布局失败单列，不重复昂贵全量。五模型既有 Excel 字节须与基线相同，旧层组/43层/单层及来源拒绝保持。

交付 `docs/tasks/model-structure-2026-10-28/T12-decoder-result.md` 和 `T12-decoder-verify/`，包含：接手校验、改动列表及复用、新增接口、raw/normalized 数据、规则应用/拒绝、KV/Attention Archify 入口、来源/节点映射、工具身份、测试、历史产物/工作簿哈希保护及限制。原始/化简后要能定位对照。

完成 T12 后停止生成，明确“等待 Codex 复核”；T13 接产品浏览、T14 其他模型属于后续票，不提前实施。本票产物是阶段交付，不能宣称五模型新版已经完成。
