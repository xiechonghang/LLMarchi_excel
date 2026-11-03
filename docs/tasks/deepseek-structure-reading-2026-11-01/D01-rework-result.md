# D01 返修结果报告：页签作用域、端点权重依赖、来源与公开图文字

日期：2026-11-02（Asia/Shanghai）。范围：只做 D01 / GitHub #17 返修；
未启动 D02–D05、未继续旧 T14、未关票、未 commit/push、未恢复自动化。
状态：**返修实施完成，停待 Codex 复核**。

## 1. 返修后候选与入口

| 项 | 值 |
| --- | --- |
| 候选根 | `/Users/xiechonghang/Workspace/LLMarchi_excel/structure_reader/outputs/deepseek_v4_flash_0731_hf/1bc0ea5cfffadc870c53296d17a73e17a35da37f44a68dfaa2889b4ce5340e80` |
| analysis_id | `1bc0ea5cfffadc870c53296d17a73e17a35da37f44a68dfaa2889b4ce5340e80` |
| **正式入口** | `viewer/structure.html`（2,019,506 bytes，sha256 `fbc6306b33b68edb49cff97ffe47932d8c25cc928731a7b82a585bd861d274ef`） |
| 图 | 29 张 Archify 原生 HTML（`viewer/archify/`） |
| 交付清单 | `run_manifest.json`（123 个文件）、`reports/source_manifest.json` |
| 首轮候选 | `417846d5…` 保留未动；新候选为新身份 |
| 证据目录 | `docs/tasks/deepseek-structure-reading-2026-11-01/D01-rework-verify/` |

构建命令与首轮一致（不实例化模型、不读权重）：

```bash
PYTHONPATH=structure_reader/src \
  /Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  structure_reader/scripts/read_model_structure.py \
  build-hf --model deepseek_v4_flash_0731_hf --all-variants --archify-layers --json
```

## 2. 复核项逐条修复

### ① P1 页签/reveal 全局选择器导致跨章节清空

**修复**

- `_endpoints_block` / `_block_section` / `_ops_section` 为每组页签加
  `data-tab-group`：整网 `network-endpoints`、每组 `block-<gid>` 与 `ops-<gid>`，
  共 **9 组**。
- `activateTab(btn)` 只在 `btn.closest("[data-tab-group]")` 内切换 `[data-tab]`
  与 `[data-tab-btn]`；`reveal()` 同样按组激活；删除整页 `[data-tab]` 选择。
- `show()` 进入任意层时调用 `ensureActiveTabs(section)`：若该组没有可见图，
  恢复第一张，保证跨层往返不出现空白页签。
- 各层保留自己的选中页签：整网选 `head` 后进入 Decoder/内部通路再返回，整网仍是
  `head`；block/ops 组互不影响。

**证据**

- 交付页：9 个 `data-tab-group`、22 个按钮/22 个面板、整页 `[data-tab]` 选择器
  0 处、`closest("[data-tab-group]")` 与 `ensureActiveTabs(...)` 均存在
  （`display-audit.json → tab_audit`）。
- 用**交付页里的实际 JS**（不是重抄函数）跑最小 DOM 桩：
  点击 `head` 后 block/ops 组可见性不变；点击 block 页签后整网 `head` 与 ops 组不变；
  `reveal('ops-kv')` 只影响 ops 组；组内全隐藏时恢复第一张；
  `show(block)` → `show(network)` 往返后 `active=lv-network` 且整网仍选 `head`
  （`tab_probe`，状态 `ok`）。
- 新增 2 项回归测试（静态组契约、实际 JS 组切换/跨层往返）；reveal 与
  组内恢复在同一 JS 探针中断言。

### ② P1 Embedding、HyperHead 投影与 lm_head 权重节点缺消费边

**修复**

- 端点图边生成改为按 `(权重张量, 消费算子)` 去重：`op.weights` 与作为输入消费的
  权重都会创建 `aux` 边，不再因“节点已存在”跳过。
- `embed_tokens_weight → net_embed`、`hc_head_fn → net_hc_head_project`、
  `hc_head_base/scale → net_hc_head_pre`、`final_norm_weight → net_final_norm`、
  `lm_head_weight → net_lm_head` 全部有且仅有真实消费边。

**证据**

- 交付 JSON：embedding 5 节点/4 边（首轮 3 边）、head 14 节点/14 边（首轮 12）；
  三个缺边节点 incident edges 均为 1（`display-audit.json → weight_edge_audit`，
  `missing_edges=[]`）。
- 新增 2 项测试：每个权重节点至少一条消费边；三个缺陷节点连到正确消费者。

### ③ P2 外部张量来源统一指向首个算子

**修复**

- 端点外张量按真实身份取来源：
  - 有生产算子 → 使用生产算子来源（如 `streams_after_layers` 指向
    `net_layer_loop`，行 1304/1357/1161）；
  - 权重或纯输入 → 使用消费算子来源，并在节点详情写明“消费算子：模块.名称（op_id）”；
  - 输出边界 → 使用生产算子来源。
- 来源行覆盖定义与消费位置：`hc_fn` 995/1001、`hc_base/scale` 1002、
  `norm.weight` 1368/1307/54–60、`lm_head.weight` 1451/1511/1528–1536、
  `embed_tokens.weight` 1302/1332/1328。

**证据**

- `display-audit.json → weight_edge_audit.missing_source_lines = []`；
- 页面来源联查 `all_source_references_present = true`（逐行在交付 HTML 中命中）；
- 新增 1 项测试：权重来源必须包含定义/消费行，且输入/上游张量带生产/消费说明。

### ④ P1 公开 12 图 243 处主图省略号

**修复**

- 完整文字层级从 `network/embedding/head/group` 扩展到
  `decoder/mhc/moe`；节点宽度按文字预算自动加宽（主链上限 520、侧栏上限 760），
  不再用 `…` 换紧凑。
- 纵向连线标签预算 260→720、行内标签 80→300、跨秩标签 260→720，使全称 shape /
  表达式完整显示；标签仍落在行间空隙。
- 侧栏加宽后主链净距从 40 调到 64，避免“辅助输入/段输入”压到主链节点。

**证据**

- 交付 29 张 candidate JSON：**0 处 `…`**（首轮 243 处，集中在
  decoder/mhc/moe 12 图）；原 12 图逐图 0 截断。
- 29/29 `validate` / `deliver` / `check` 通过（完整文字加宽后的几何、标签线路净距、
  画布包容均过门禁）。
- 新增 2 项测试：12 图候选无 `…`；节点宽度有上限（≤760）。

## 3. 验证结果（通过）

### 3.1 测试

- 17 个测试文件、**527 项全部通过**（`D01-rework-verify/tests.log`）。
- 新增/扩展 `tests/test_deepseek_reading_page.py`（15→21 项）：
  页签组静态契约、实际 JS 组切换与跨层往返、reveal/恢复、权重消费边、
  权重来源、输入/上游来源、12 图无截断、宽度上限、端点完整性等。

### 3.2 交付 HTML/JSON

- 页面：2,019,506 bytes，0 个 `<svg>`、33 个 `?embed=1` iframe、2 个阶段按钮、
  9 个页签组；路由 595 / 锚点 1901 / 相对文件 168，失效/缺失 0，外链 0；
  完整文字 696 行、`…` 0。
- 图库：29 图、553 节点、715 条关系；主图截断 0；投影主字号 10.79–16.91px
  （最宽画布 1894px，仍高于 6px 下限）。
- 辅助输入：各图 aux/dashed 边保留（decoder 2、attention 1–3、endpoint 权重 1/5 等）；
  `aux_input_audit` 记录逐图连接数。
- 来源联查：权重/输入/输出节点来源行逐条在页面命中。

### 3.3 原生 Archify 回执

- 29/29 `validate` / `deliver` / `check` 通过；
- 29/29 由候选 JSON 重建的 HTML 与交付 HTML 逐字节一致；
- 29/29 delivery sidecar 绑定候选 JSON 与交付 HTML。

### 3.4 历史保护

- 22242 个受保护历史产物：**22242 未变 / 0 缺失 / 0 变更**；
- 两处既有删除保持；
- 新候选 Excel 与首轮候选逐字节一致（`2dabd3af…365b05e`）；
- 首轮候选 `417846d5…` 及更早候选、旧证据目录全部保留。

## 4. 失败 / 跳过 / 不可用 / 未目检

| 类别 | 内容 |
| --- | --- |
| **不可用** | 浏览器门禁：29/29 图 `browser-check` 失败（Chrome sandbox 初始化被拒，`Operation not permitted`）；正式页 `browser-check` exit 1；两次 `visual-check` 均失败、0 截图。未关闭 sandbox、未绕过限制。AC18/AC19 及运行时 AC14–AC17 **仍未验收**。 |
| **未目检** | 1440×900 / 1920×1080、100% 缩放、长图滚动、查看器缩放均未测量；无逐张截图与人工查看。 |
| **跳过** | 无测试被 `skipUnless` 跳过。 |
| **失败** | 无最终构建/测试/图生成失败。 |
| **已消除** | 首轮的 243 处主图省略号、3 个权重缺边节点、统一错误来源、跨章节页签清空，均在最终候选字节上复核为 0/已修复。 |

## 5. 已知限制

1. 浏览器与视觉验收仍缺失（环境拒绝启动 Chrome）；不得以静态/JS 桩证据替代。
2. 完整文字通过加宽节点实现，侧栏最宽 760、主链 520；图更宽但仍受页面最大内容
   宽度约束，实际投影字号 10.79–16.91px 为估算，非浏览器实测。
3. 公开图文字已完整，但标签之间的实际观感（距离/遮挡）仍需浏览器目检确认。
4. 本轮未改 D02–D04 的语义细分（mHC 分支细节、Q/KV/O 通路、Hash/TopK 与专家内部
   仍维持现有粒度）。
5. `run_manifest.structure.archify.levels` 仍列 7 个层级名（端点图以 entries/README/
   页面入口体现），未另造层级名。

## 6. 停止点

返修完成，**停止等待 Codex 复核**：

- 未关闭 #17/#16，未评论或改标签；
- 未启动 D02–D05，未继续旧 T14；
- 未 commit / push / reset / clean；
- 未恢复自动化；
- 未修改 `D01-codex-*`、`D01-execution.json`、`tickets.json`、`D01-rework-handoff.md`
  等外部协调文件。
