# T12 结果（第三轮收尾）：Decoder 语义图与可追溯等价化简

状态：**第三轮收尾完成，停止生成，等待 Codex 复核**。未关闭
[#13](https://github.com/xiechonghang/LLMarchi_excel/issues/13)，未 commit/push，未改协调文件与自动化
（保持 PAUSED/45 分钟），未恢复旧会话，未进入 T13/T14。第一、二轮证据与 Codex 复核目录原样保留；
本轮产物写入 [`T12-decoder-verify/round-3/`](T12-decoder-verify/round-3/)。

第二轮复核两项 P1（拒绝改写破坏原图、缓存输入向上回指）已修复；此前七项按第二轮结论保持。
返修对照见 [round-3/fix-matrix.json](T12-decoder-verify/round-3/fix-matrix.json)，第一轮七项见
[round-2/fix-matrix.json](T12-decoder-verify/round-2/fix-matrix.json)。

## 1. 基线（第三轮，逐文件哈希）

- 工作区 `/Users/xiechonghang/Workspace/LLMarchi_excel`；HEAD `947ae78f120279403ec0a4c539e99d3d8869e1d6`
  （本票未提交，HEAD 未变）。快照与 round-2 相同。
- 脚本 [round-3/verify_baseline_r3.py](T12-decoder-verify/round-3/verify_baseline_r3.py)，结果
  [round-3/baseline-check.json](T12-decoder-verify/round-3/baseline-check.json)：基线 816 文件中 812 一致、
  4 个 Codex 协调文件只读登记；14814 个受保护历史产物全部一致；五模型 40 个候选文件全部一致；
  两处既有删除保持；协调目录外无未说明差异。

## 2. 本轮两项 P1 修复

### P1-1 拒绝改写不得留下部分修改

- 根因：`semantic/rules.py::_rewrite` 在逐个消费者收集 pending 时已删除前面通过检查的边；后续消费者
  guard 冲突抛 `_RewriteRejected`，调用方只记录 rejected，没有回滚。
- 修复：`_rewrite` 改为**两阶段**——先只读预检全部消费者/端口/guard（任何冲突在修改前抛出），再提交；
  提交阶段用 `_Mutable.snapshot()`/`restore()` 事务保护，任何异常回滚。R7 的手工合并同样快照保护。
  拒绝路径不改变节点、端口、边、张量、guard 与映射。
- 独立反例：Codex 的
  [rejected-rewrite-probe.py](../T12-decoder-review/round-2/rejected-rewrite-probe.py) 重跑
  [round-3/rejected-rewrite-probe-r3.txt](T12-decoder-verify/round-3/rejected-rewrite-probe-r3.txt)：
  拒绝前后 `before`/`after` 三条边完全一致，`validator ()`（不再是 reachability-changed）。
- 回归：`TestRejectedRewriteLeavesGraphUnchanged` 2 例（“先可接受、后冲突”的多消费者；拒绝后
  normalized 图与 raw 边完全一致）。

### P1-2 缓存状态输入不得向上回指

- 根因：`render/archify_preview.py::_layout` 把“调用前状态”固定锚在 `cache_update` 附近，而
  compressor/indexer 锚在 `kv_concat` 上方更靠上的位置，导致 dashed 状态边向上。
- 修复：`_layout` 改为按**全部真实 state 消费者**的拓扑放置：前态位于全部消费者上方，后态位于全部生产者
  下方；冲突时只移动旁枝节点并保持顺序（不删边、不改写状态语义）。布局层加硬约束：任何 state 边源顶部
  必须高于目标顶部，否则拒绝生成。
- 最终 SVG 路线核对：新增 `state_route_violations`，从交付 HTML 的
  `data-edge-id` / `data-composition-points` 读取每条虚线路由，出现任何 y 下降段即违规。
  [round-3/state-route-checks.json](T12-decoder-verify/round-3/state-route-checks.json) 覆盖 10 份预览
  （四变体 raw/normalized、KV/Attention），全部 **0 violations**。
- 实际路线（csa-topk attention normalized）：
  `cache_in(1200) → compressor(1439)`、`→ indexer(1294)`、`→ cache_update(1690)`；
  hca：`1345 → 1439 / 1690`；swa：`1596 → 1690`。全部向下，缓存后态在生产者下方输出。
- 回归：`TestStateEdgesDownward` 2 例（四变体 × raw/normalized × KV/Attention 的布局方向；SVG 路线
  违规检测器正反例）。

## 3. 前两轮结论保持

前轮已解决的 7 项继续成立（浏览器失败窄分类、206 文件工具依赖闭包、R5 索引映射与 viewed→target 条件、
guard 单边/冲突、R7 操作数角色、Attention 全依赖与 edges trace、mHC post/comb）。本轮未改动这些语义，
只让 `_rewrite` 事务化并重排状态节点；语义数据与四变体规则结果保持一致。

## 4. 语义数据（四组真实层）

| 层组 | 代表层 | raw 节点/张量 | normalized | 证据 |
| --- | ---: | ---: | ---: | --- |
| swa-hash | 0 | 41 / 41 | 37 / 37 | [raw](T12-decoder-verify/round-3/semantic/swa-hash-raw.json) · [normalized](T12-decoder-verify/round-3/semantic/swa-hash-normalized.json) |
| csa-hash | 2 | 45 / 50 | 41 / 46 | [raw](T12-decoder-verify/round-3/semantic/csa-hash-raw.json) · [normalized](T12-decoder-verify/round-3/semantic/csa-hash-normalized.json) |
| hca-topk | 3 | 44 / 47 | 40 / 43 | [raw](T12-decoder-verify/round-3/semantic/hca-topk-raw.json) · [normalized](T12-decoder-verify/round-3/semantic/hca-topk-normalized.json) |
| csa-topk | 4 | 45 / 50 | 41 / 46 | [raw](T12-decoder-verify/round-3/semantic/csa-topk-raw.json) · [normalized](T12-decoder-verify/round-3/semantic/csa-topk-normalized.json) |

四变体 `validate_bundle` / `legacy_coverage` / `validate_normalization` 均为 0 issue
（[round-3/validation/](T12-decoder-verify/round-3/validation/)）；R1/R3/R5/R7 应用、R2/R4/R6 拒绝并给出
候选原因；每条布局改写带 index-map 探针与来源。

## 5. Archify 预览（round-3 新 receipt）

| 层组 | KV | 完整 Attention |
| --- | --- | --- |
| csa-topk（raw） | [KV 原始](T12-decoder-verify/round-3/previews/csa-topk-kv-raw.html) | [Attention 原始](T12-decoder-verify/round-3/previews/csa-topk-attention-raw.html) |
| csa-topk（normalized） | [KV 化简后](T12-decoder-verify/round-3/previews/csa-topk-kv-normalized.html) | [Attention 化简后](T12-decoder-verify/round-3/previews/csa-topk-attention-normalized.html) |
| csa-hash | [KV](T12-decoder-verify/round-3/previews/csa-hash-kv-normalized.html) | [Attention](T12-decoder-verify/round-3/previews/csa-hash-attention-normalized.html) |
| hca-topk | [KV](T12-decoder-verify/round-3/previews/hca-topk-kv-normalized.html) | [Attention](T12-decoder-verify/round-3/previews/hca-topk-attention-normalized.html) |
| swa-hash | [KV](T12-decoder-verify/round-3/previews/swa-hash-kv-normalized.html) | [Attention](T12-decoder-verify/round-3/previews/swa-hash-attention-normalized.html) |

- 10 份预览 `validate`/`deliver`/`check` 全部 pass；`browser-check` 在本会话仍是
  `viewer/browser-check-runtime`（Chrome 沙箱/GPU 环境），按窄规则分类 environment 并记录；**未声称**
  浏览器几何或视觉审查通过，Codex 可在可用 Chrome 复跑。第二轮 Codex 已对 round-2 十图做过真实
  browser-check（[round-2-browser/summary.json](../T12-decoder-review/round-2-browser/summary.json)），
  本轮图已按新路径重新生成并等待复跑。
- 每份图仍带 edges trace（`previews/edges/*.json`：张量 id/标签/全称 shape/guard/state transition）
  与“分支/状态关系”卡片；KV 原始 8/7、化简 7/6；Attention 原始 41/46、化简 37/42（csa-topk）。

## 6. 来源与工具身份

- `hflib/transformers` @ `536ecc007387a50e77603bb5d92100e9b07514cc`（origin
  `https://github.com/huggingface/transformers.git`，工作区 clean）；组件来源为仓库相对路径，
  `finalize --repo-root` 核验。
- [round-3/manifest.json](T12-decoder-verify/round-3/manifest.json)：Archify 3.2.0、CLI `3cebcee3…`、
  206 文件依赖闭包摘要 `87726d62…`；产品文件哈希（rules `1e044a63…`、archify_preview `9ced486b…`、
  layout_probe `e4767cd3…`、适配器 `e15f4f38…`）与语义/预览/edges 哈希均在 manifest；根 HEAD 只作仓库身份。

## 7. 测试

新增/更新 61 项全部通过（[round-3/tests-final.json](T12-decoder-verify/round-3/tests-final.json)）：

- `test_semantic_rules` 25 项（含拒绝不改图 2 项、guard/R5/R7/探针等 round-1 反例）。
- `test_deepseek_semantic` 15 项。
- `test_archify_preview` 21 项（含四变体状态方向、SVG 路线违规检测、浏览器分类、包闭包、视图依赖）。

回归通过：`test_hf_decoder_reading` 61、`test_hf_layer_levels` 41、`test_hf_network_overview` 21、
`test_operator_workbook_v2` 35、`test_rules_deepseek` 20、`test_scenario_evaluator` 45、
`test_v2_variants` 15、`test_deepseek_reference_graph` 70、`test_hf_build_missing_xlsx` 2。
单列既有问题：`test_hf_decoder_diagram` 6 项失败（未跟踪夹具旧标签/缩写，产品代码未被本票修改）、
`test_hf_decoder_onnx` 无可执行用例（缺 onnx/onnxruntime）。未跑昂贵全量。

## 8. 保护与限制

- 五模型 40 个候选文件（含工作簿）与 14814 个历史产物逐字节一致；旧候选、第一/二轮证据、Codex 复核目录
  未被覆盖；协调文件只读；未 commit/push、未关闭 Issue、未派单、未恢复自动化。
- 逻辑语义图不等于设备执行图；未折叠权重、未改精度、未删缓存副作用；未证明的化简保留并记录原因。
- 本会话 Chrome 仅环境失败，浏览器几何/视觉审查未完成；10 份图的真实 browser-check 由 Codex 复核。
- 产品浏览（T13）与其他模型（T14）未提前实施；本票是阶段交付。

## 9. 停止点

第三轮两项 P1 已修，产物与证据写入 `T12-decoder-verify/round-3/`，前两轮证据与复核目录保留。
**停止生成，等待 Codex 复核**；未进入 T13/T14。
