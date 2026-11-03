# D02 四项阻塞返修结果

日期：2026-11-03（Asia/Shanghai）。仅返修 D02/#18，未启动 D03–D05，
未关票、未 commit/push、未恢复自动化、未访问旧 T14。状态：**代码、候选、
原生静态门禁和显示几何返修完成；真实浏览器 P2/P3 仍需 Codex 授权环境补跑**。

## 1. 新候选

| 项 | 值 |
| --- | --- |
| 候选 | `structure_reader/outputs/deepseek_v4_flash_0731_hf/c47633db4bec814108f664efb4a0898f862c97f49eeaa860291868fa872b2486` |
| 正式入口 | `viewer/structure.html`（2,032,933 bytes，sha256 `5a9337a70f80e589495bdab9a9b1097ffc10839049a968db2db0f64c44d09e60`） |
| 整网原生图 | sha256 `dda361a661c6ed426e6c2c3f49bd47813f48a8c8086d091dac3f0c4d2543bc8b` |
| 证据 | `D02-rework-verify/` |

## 2. 四项逐项响应

### P1 主图符号维度回归

- 新增统一可见文本：Decoder/mHC 语义边显示 **`身份｜全称 shape`**，
  例如 `collapsed｜batch_size,seqlen,hidden_size`、
  `combᵀ｜batch_size,seqlen,hc_mult,hc_mult`、`branch_output｜batch_size,seqlen,hidden_size`。
- 四组 Decoder 的 collapsed/post/comb/branch_output/原始多流/combᵀ 与四组 mHC
  的 collapsed/post/comb 返回全部带 shape；`display-audit.json` 的
  `labels_missing_shape` / `returns_missing_shape` 均为空。
- 没有用弹窗/源码/完整文字面板替代主图维度；标签随文字扩宽后由统一布局器
  扩画布/错层，未删信息。

### P2 四组 mHC 标签碰撞

- `separate_edge_labels` 的碰撞预算改为 Archify 实际标签盒
  `max(实际字符宽, textUnits*4.8+10)`、高 14px，并扩大水平/垂直让位范围。
- 四组 mHC 的 `f-7-o-in-o-attn-hc-collapse` 与
  `f-15-o-in-o-ffn-hc-collapse` 标签盒水平间隔约 **88.2px**
  （`label-collision-audit.json`）；不再依赖单一 4.16px/字符估算。
- **限制**：这是离线盒计算，不是 Chrome `getBBox`；最终真实 bbox 需 Codex
  在授权环境用原 `verify-final.mjs` 复核，本报告不宣称已通过。

### P3 两张 CSA 原生可读性

- Decoder 布局收紧列距/侧栏距，并约束完整标签不越出组件画布右缘；
  `decoder-csa-hash` / `decoder-csa-topk` 的 viewBox 从 **1812 降到 1772**。
- 按既有 Reader 规则推算：1364 宽时连线字约 **6.158px**、1479 宽时约
  **6.677px**（`native-projection.json`）。
- **未完成**：dsh 仍无法启动 Chrome；对最终 CSA 的 `browser-check` 只得到
  `viewer/browser-check-runtime` 环境失败回执（`native-attempt/`），**不能**
  写成原生可读性已通过。Codex 必须重跑原生 gate，确认不再出现
  `viewer/projected-text-readability` / `viewer/viewport-overflow`。

### P4 mHC 局部参数与 Sinkhorn 循环

- mHC candidate 参数卡只保留 `B/seqlen/H/Nhc/Sit/Eps`：
  `hc_mult=4`、`hc_sinkhorn_iters=20`、`hc_eps=1e-06`，不再显示
  Nh/Rq/G/W 等无关前缀。
- 页面 mHC 图题下新增“mHC 局部参数（全称/来源）”块，列出
  `config[hc_mult]`、`config[hc_sinkhorn_iters]`、`config[hc_eps]` 与
  modeling 926-938、976-979 来源。
- comb 节点主文字 `comb 混合（softmax+Sinkhorn）`、副文字
  `1 次初始列归一 + (Sit-1) 次 [行归一→列归一]`；详情写明
  `hc_sinkhorn_iters-1=19`、不复制 20 份节点、无向上回流。
- 保留具名算法粒度；未把循环藏到源码面板。

## 3. 测试与原生

- **547/547 测试通过**（17 文件，`tests.log`）；新增/扩展：
  - 全 shape 身份标签（Decoder + mHC 三路）；
  - mHC 参数卡只含局部参数；
  - Sinkhorn 循环文字/19 次/无 20 份节点；
  - 标签不压节点/标题、mHC 几何外沿非负。
- **29/29** 图 `validate`/`deliver`/`check` 通过；29/29 重建 HTML 与交付字节一致。
- **browser-check 未在本机完成**（环境失败）；不能据此声称 29/29 浏览器通过。
- 29 图离线标签重叠 **0**、SVG 可见文字重叠 **0**、省略号 **0**；
  33 图位短 iframe **0**。
- `geometry-audit.json`：Decoder/mHC 节点、via 路由、标签外沿 min x/y ≥0，
  标签不压节点。

## 4. 浏览器限制

dsh 当前环境：

- `Failed to initialize sandbox. Operation not permitted`
- `GPU process exited unexpectedly: exit_code=6`
- `Timeout Page.enable`

已实际尝试并保留：

- `D02-rework-verify/probe-attempt/`（`--geometry-only`，exit 1，0 截图）
- `D02-rework-verify/probe-attempt-full/`（完整探针，exit 1，0 截图）
- `D02-rework-verify/native-attempt/decoder-csa-hash|decoder-csa-topk/`
  （原生 browser-check 环境失败）
- `D02-rework-verify/visual-attempt/`（decoder/mHC visual-check 失败）
- `native-evidence.json`（validate/deliver/check 29/29；browser-check 环境失败）

因此最终候选的 1440×900/1920×1080 截图、真实标签 bbox、CSA 原生字号门禁、
查看器缩放与长图滚动、29 图逐张目检均 **未完成**；AC18/AC19 保持未验收。

## 5. 历史保护

- 继承 22242 项历史产物：22242/22242 未变；
- 上轮候选 248：248/248 未变；
- D01 已接受 `f1ea240e…` 124：124/124 未变；
- D02 返修前 `0eb1ca64…` 124：124/124 未变；
- 两处既有删除保持；
- Excel 最新两候选逐字节一致；
- 协调文件 1845/1847 未变；`tickets.json`、`D01-execution.json` 为 Codex 派发
  D02 时外部更新，只读未改；
- 源文件快照 202：198 未变、4 修改（archify_layers、build_hf_decoder、
  hf_reading_page、test_deepseek_reading_page）。

## 6. 限制与停止点

- P2/P3 的**真实浏览器证据**仍未完成；不能在报告中写成通过。
- P1/P4 的候选/页面/参数/循环已按新候选绑定。
- 未做 D03–D05，未改 Codex 证据/协调 JSON/handoff，未关 Issue、未 commit/push、
  未恢复自动化。

完成 D02 返修，停待 Codex 复核。
