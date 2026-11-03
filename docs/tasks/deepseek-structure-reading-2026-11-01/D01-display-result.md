# D01 显示返修结果报告：整网布局/连线可读性 + 最终显示证据

日期：2026-11-02（Asia/Shanghai）。范围：只做 D01 / #17 显示返修；未启动 D02–D05、
未继续旧 T14、未关票、未 commit/push、未恢复自动化。状态：**实施完成，停待 Codex 复核**。

## 1. 新候选与入口

| 项 | 值 |
| --- | --- |
| 候选 | `/Users/xiechonghang/Workspace/LLMarchi_excel/structure_reader/outputs/deepseek_v4_flash_0731_hf/f1ea240e2c0c328b075863a963526a1e64bdb717e1584b50f3b6d36821279614` |
| analysis_id | `f1ea240e2c0c328b075863a963526a1e64bdb717e1584b50f3b6d36821279614` |
| **正式入口** | `viewer/structure.html`（2,019,910 bytes，sha256 `de632e2f727e85fee5ab28671647da59715753b9018a9d7b444ec60ea9c8582d`） |
| 整网原生图 | `viewer/archify/network.html`（sha256 `e720f01aa5e7933891c27130a411fb94f79c24bb3de32666da1a293a27524c9d`） |
| 新候选前的 `1bc0ea5c` | 保留未动；28 张非 network 图与它**逐字节一致** |
| 报告/证据 | `D01-display-result.md`、`D01-display-verify/` |

构建命令（未改）：

```bash
PYTHONPATH=structure_reader/src \
  /Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  structure_reader/scripts/read_model_structure.py \
  build-hf --model deepseek_v4_flash_0731_hf --all-variants --archify-layers --json
```

## 2. 整网布局与连线文字（目标 1）

### 问题根因

- 旧整网把 `position_ids / mask_build / rope_main / rope_compress / hc_expand`
  并列成 4 列，层循环继承最右列：下半部主链偏右、左下大块空白。
- 画布 1834 → viewBox 1922，在 1440×900 下 Reader 只能缩到 1346 宽：
  8px 连线字被投影为 **5.602497px**，低于 6px；主节点同样被缩小。

### 生成器修复（`render/archify_layers.py`）

- **主链单列**：`position_ids`、两组 `RoPE` 改为 `role="branch"` 的左侧辅助预计算，
  锚定在真实消费者附近；`mask_build` 与 `hc_expand` 同处 rank 2，但
  `hc_expand` 排在前面，层循环继承主链列（col0），不再被挤到最右列。
- **真实节点/边全部保留**：19 节点、27 条边不变；每条边的 tensor id、
  tensor label、全称 shape 仍在 connections/relations 中；未删边、未用省略号。
- **失败边恢复全称文字**：`f-0-n-net-embed-n-net-mask-build` 重新显示
  `batch_size,seqlen,hidden_size`（位于两列之间的行间空隙，不压节点）；
  `mask → loop` 也显示 `batch_size,1,seqlen,mask_sequence_length`。
- **侧栏标签防叠字**：整网侧栏同行的短标签按 18px 逐条错开；其他模型/其他图
  不做无谓改动（28 图字节不变）。
- 画布 1834×1694 → **1094×1694**，viewBox **1158×1762**；主链 x=382，
  画布中心 547，主链中心 553（基本居中）；左侧辅助列与右侧 mask 分支平衡。

### 可读性推导（不是浏览器测量）

Archify Reader 规则（`assets/template.html`）与 SVG 元数据：

| 项 | 值 |
| --- | --- |
| viewBox / 字体 | 1158×1762；node 11/9px，edge 8px，primary 14px |
| readable width / primary width | 977 / 1486 → `minWidth = 1364`（视口上限） |
| 投影 scale | 1364 / 1158 ≈ **1.178**（browser-check 对 scale 取 `min(1)`） |
| 连线文字（门槛口径） | **8px**（≥6px），不再出现 5.6px |
| 连线文字（CSS 实际放大） | ≈ **9.42px**；节点最小 ≈ **10.6px** |
| 纵向滚动 | 页面高于 900px，但 readabilityOk + `intrinsic-height` + authored overflow → 预期 `readable-vertical-scroll`，符合“允许可读纵向滚动” |

上述是源码规则推导，**必须**由授权环境 `browser-check` 确认；本机无法启动 Chrome
（见 §5），未写成已通过。

### 全图库核对

- 29 张公开图 candidate JSON：`…` **0 处**（上一轮已清零，本轮保持）。
- 28 张非 network 图与 Codex 已测 28/28 通过的 `1bc0ea5c` **逐字节一致**
  （`graph-byte-parity.json`：28 same / 1 changed）。
- 只有 network 需要重新测量；其节点、边、来源、全称参数均未删除。

## 3. 最终页面与交互验证（目标 2）

### 已完成的自动/逻辑验证

- 交付页：2 阶段、9 个页签组、22 按钮/22 面板、33 个 `?embed=1` iframe、
  0 个旧 SVG；路由 595、锚点 1901、相对文件 168，失效/缺失 0；外链 0。
- 交付页实际 JS 的最小 DOM 桩（非浏览器）：
  - 点击整网 `head` 后 block/ops 组可见性不变；
  - Decoder → 模块关系 → 内部算子 → 整网往返后整网仍选 `head`；
  - `reveal` 只激活所在组；组内无可见图时恢复第一张；
  - 参数锚点、源码面板开关、辅助输入/权重边在静态链路中均存在。
- 页面嵌入宽度下限从 960 提高到 1200：窄图（整网/Embedding/Attention）内嵌时
  不再被压到 720/960；network 内嵌 1200×1866。

### 未完成的浏览器证据

- dsh 环境仍无法完成 Chrome 渲染（见 §5），**未保存最终候选任何截图**，
  未做 1440×900 / 1920×1080、100% 缩放下的章节/页签/返回/滚动/查看器缩放实测。
- 已提供授权环境可执行的
  [`verify_structure_page.mjs`](D01-display-verify/verify_structure_page.mjs) 与
  [`capture_all_graphs.sh`](D01-display-verify/capture_all_graphs.sh)：
  - CDP 驱动系统 Chrome（不加 `--no-sandbox`），两视口、四层章节、页签、
    跨层返回、源码/参数联查、长图滚动、iframe 查看器缩放，逐状态 PNG + `result.json`；
  - 29 图逐张 `browser-check` + `visual-check` 截图。
- 28 图虽与已测候选逐字节一致，仍须在最终候选绑定重跑；network 必须新测。
- AC18/AC19 与运行时 AC14–AC17 **保持未验收**；不能用 28 图自动通过、也不能用
  旧截图冒充最终全图人工目检。

## 4. 验证结果（通过）

- **测试**：17 个文件、**534 项全部通过**（`D01-display-verify/tests.log`）。
  新增/扩展整网测试：主链单列、画布宽度上限、mask 分支、侧栏锚点、数据边向下、
  27 条边/全称 shape 保留、失败边全称文字、侧栏标签不重叠、候选 0 省略号。
- **原生回执**：29/29 `validate`/`deliver`/`check` 通过；29/29 重建 HTML 与交付
  字节一致；29/29 delivery sidecar 绑定候选 JSON/HTML。
- **历史保护**：继承 22242 项 **全部未变**；派发前冻结的两个 D01 候选 248 个文件
  **全部未变**；两处既有删除保持；新候选 Excel 与上一候选逐字节一致
  （`2dabd3af…365b05e`）。
- **源文件差异**：派发前 4 个源文件中 3 个被本轮修改（archify_layers、
  hf_reading_page、test_deepseek_reading_page），build_hf_decoder 未变。

## 5. 失败 / 跳过 / 不可用 / 未目检

| 类别 | 内容 |
| --- | --- |
| **不可用** | dsh 文件系统沙箱内 Chrome 子进程无法初始化：`Failed to initialize sandbox. Operation not permitted`、GPU exit_code=6、DevTools read pipe failed。network/页面 `browser-check`、`visual-check`、CDP 页面脚本均 exit 1；原始回执在 `attempts-final/`。未关闭/绕过 Chrome sandbox，未改工具或候选 HTML。 |
| **未目检** | 最终候选 0 张截图、0 张人工查看；1440×900 / 1920×1080 未测量。 |
| **待 Codex 补跑** | 新 network 的 browser-check（推导预期通过，须实测）；28 图在最终候选上的绑定重跑；最终页两视口与交互、29 图截图逐张查看。 |
| **跳过** | 无测试 `skipUnless` 跳过。 |
| **失败** | 无最终构建/测试/图生成失败；历史/Excel 无变更。 |

## 6. 已知限制

1. 浏览器/视觉验收仍缺；本机环境限制已如实记录，不把“工具已安装”写成“验收通过”。
2. 整网采用左侧辅助列 + mask 右侧分支的单主链布局；主链中心接近画布中心，
   但两侧下方仍会有空白，需在截图中确认观感。
3. 可读性数字是 Reader 规则推导；实际 `browser-check` 可能因 Chrome 版本/
   字体度量略有差异，以授权环境回执为准。
4. 29 图主图文字完整，但标签距离、遮挡与配色仍须逐张目检。
5. 未提前实现 D02–D04 的语义细分。

## 7. 停止点

- 未关 #17/#16，未评论/改标签；
- 未启动 D02–D05，未继续旧 T14；
- 未 commit/push/reset/clean，未恢复自动化；
- 未修改 `D01-rework-codex-*`、`D01-execution.json`、`tickets.json`、
  `D01-display-handoff.md` 等外部协调文件。

停待 Codex 在授权环境补跑浏览器证据并复核。
