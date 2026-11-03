# D03 rework02 结果：阶段事实与缓存累计/全称来源

日期：2026-11-03（Asia/Shanghai）。只续做 D03 剩余两类技术问题；未重做已通过布局/
入口，未启动 D04/D05，未关票、未 commit/push、未恢复自动化、未操作 E01/旧 T14。

## 1. 新候选

| 项 | 值 |
| --- | --- |
| 候选 | `structure_reader/outputs/deepseek_v4_flash_0731_hf/8136e2132e06c2c84c8e97a6fde9900ac1e613121636fdd6543a95e7091d8531` |
| 正式入口 | `viewer/structure.html`（2,234,067 bytes，sha256 `fa0f2dd3d0f534de75520bf754516e27df907e680cabc186c9e855066e213d81`） |
| 图 | 51 张 Archify 图、55 个嵌入图位 |
| 证据 | `D03-rework02-verify/` |

构建命令：

```bash
PYTHONPATH=structure_reader/src \
  /Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -c "from structure_reader.cli import main; import sys; sys.exit(main(['build-hf','--model','deepseek_v4_flash_0731_hf','--all-variants','--archify-layers','--json']))"
```

## 2. P1 阶段图数据流

`_stage_graph` 不再接收/信任手工 inputs/outputs；只接收阶段 `op_ids`，从 block
真实 facts 推导：

- 阶段输入 = 本阶段 op 的真实 inputs 中、非本阶段内部产出、非权重的张量；
- 未被任何阶段生产的输入连 `s-in`，保留 cache/overlap/entry_count/cos/sin 等真实依赖；
- 阶段输出 = 真实 outputs；被后续阶段消费则连阶段边，否则连 `s-out`；
- 每个张量必须有真实 shape；`shape=[]` 才显示标量；未知 op/tensor 抛
  `ArchifyLayersError`，不再 `or '标量'` 兜底；
- 阶段内部 op_ids/details/来源/条件保留。

具体修复：

- **CSA compressor**：去掉不存在的 `c_mask_extend`/`block_bias`；真实使用
  `c_block_bias` 和该层 `mask_extend`；`c_block_bias` 的
  `[batch_size,1,seqlen,num_compressed]` 不再是标量。
- **HCA compressor**：保留真实 `hca_block_bias` 与 `c_mask_extend`；位置/None 分支不变。
- **CSA/Indexer 窗口**：pool 改为接收真实 `c_windows_kv/gate` / `i_windows_kv/gate`；
  补 `cache_overlap_*`、`cache_entry_count_*` 输入；不再跨阶段传 chunk。
- **Indexer**：`i_q_proj` 补 `cos_q/sin_q`；`i_scorer` 补 `attn_normed`，不使用不存在的
  `i_weights`；`i_topk` 补自身 `cache_compressed_i_post`。
- 最终 candidate/relations 中上述跨阶段/边界输入均存在；测试逐阶段检查真实
  op inputs 与 candidate/edges，并覆盖未知 op/tensor 拒绝、CSA bias rank、
  scorer 输入、windows 形状、全部边界输入。
- 恢复的完整 trace 断言保留，没有减 expected。

## 3. P2 缓存累计长度与全称来源

- 新增 `cache-main-cumlen-post` 阅读投影节点/边：`kvcachelen+seqlen`，注明
  “阅读投影，不新增底层 tensor”；页面缓存表同步新增行。
- 保留既有新 KV、`cache.update`、本次可读 `kv_full`/`repeat_kv` 和分图。
- 缓存逐项表改用 `expand_display_text` 展开全称符号（`batch_size/head_dim/
  sliding_window/buffer_length/...`），不再暴露 `B/Dh/W/Tbuf` 等旧别名。
- 来源列从真实读写算子追溯：写/读 op id 与 source_refs；调用前状态无 producer 时
  显示“上一调用持久状态”，不伪造来源，也不再全“—”。
- `cache_cumlen_post` 是阅读投影；旧 Excel 与历史事实未改。

## 4. 保留与验证

保留：55 图位布局/导航、HCA 压缩器原生已通过、Attention block_bias/HCA 区分、
Q/O 独立入口、完整标签布局、644 项旧显示门禁成果（新候选需复跑）。

- **558/558 测试通过**（17 文件；新增阶段/缓存回归）。
- **51/51** 图 native validate/deliver/check；51/51 重建 HTML 一致。
- 51 图标签重叠 0、SVG 文字重叠 0、省略号 0；55 图位短 iframe 0。
- 新增图 viewBox 推算：嵌入 1200px / 原生 1440px 全部 ≥6px；
  CSA 压缩器 1025、HCA 1015、Indexer 1498、主缓存 1252。
- 阶段审计：CSA 无 `c_mask_extend`/`i_weights`，`c_block_bias` 存在且非标量；
  HCA 有真实 `c_mask_extend`；scorer/windows/overlap/entry 输入全在。

## 5. 浏览器/用户审查

本机 Chrome 仍因 sandbox/GPU 初始化失败；最终候选页面探针一次尝试 exit 1、0 截图，
HCA 压缩器 browser-check 一次尝试 exit 1。回执：
`D03-rework02-verify/probe-attempt/`、`native-attempt-hca/`。

这不否定 Codex 已通过的 644 项显示门禁和 HCA 原生结果；新候选需按
`browser-runbook.md` 在授权 Chrome 复跑，不能沿用旧候选截图。用户按
`review-guide.md` 看精简入口；视觉项标**待用户审查**。

## 6. 保护

- 输出保护：rework02 212 + 上轮 160 + D03 继承 22641 并集 **23013/23013 未变**；
- `8cfa…` 及历史候选只读；D01/D02 接受候选 124/124；两处删除保持；
- 22242：22240/22242，2 个动态例外 `.DS_Store`、`monitor-state.json`，未回滚；
- 协调 2827：2826/2827，`tickets.json` 外部派发更新；
- Excel 最新两候选逐字节一致；
- E01 隔离树未触碰；当前 dirty 126，外部 `E01-verify/`、`dpskperfexcel草稿.xlsx` 未纳入；
- 源文件 222：219 未变、3 修改（archify_layers、hf_reading_page、测试）。

## 7. 停止点

未做 D04/D05；未关 Issue、未 commit/push、未恢复自动化。停待 Codex 技术复核与用户
视觉审查；不把阶段事实修复扩大为通用重构。
