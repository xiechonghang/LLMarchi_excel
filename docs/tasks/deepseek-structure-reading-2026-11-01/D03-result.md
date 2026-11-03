# D03 Attention 总览、Q/KV/O 与三类缓存结果

日期：2026-11-03（Asia/Shanghai）。仅 D03/#19，未启动 D04/D05；未关票、未
commit/push、未恢复自动化、未访问旧 T14。状态：**候选、语义/显示/原生静态门禁
完成；真实浏览器截图与用户视觉审查待补**。

## 1. 新候选

| 项 | 值 |
| --- | --- |
| 候选 | `structure_reader/outputs/deepseek_v4_flash_0731_hf/7c724c14554c404fc838785a956819f85a0f3d8039a353d9a9792f981274fe01` |
| 正式入口 | `viewer/structure.html`（2,112,492 bytes，sha256 `fd107e6d1b8cc3cfeee58101acbd60ee38b90a69291ee7886b240bf3241cb4db`） |
| 整网图 | sha256 `dda361a661c6ed426e6c2c3f49bd47813f48a8c8086d091dac3f0c4d2543bc8b`（保留 D02 字节） |
| 图 | 38 张 Archify 图、42 个嵌入图位 |
| 证据 | `D03-verify/` |

构建命令：

```bash
PYTHONPATH=structure_reader/src \
  /Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -c "from structure_reader.cli import main; import sys; sys.exit(main(['build-hf','--model','deepseek_v4_flash_0731_hf','--all-variants','--archify-layers','--json']))"
```

## 2. 源码/config → 图事实

复用既有 `hf_layer_structure.v1` 事实、DeepSeek semantic bundle、`build_*_graph` 和
Archify 生成器，没有另建手工坐标或第二套事实。

### Attention 总览

- 4 个真实层组各有一张 Attention 总览；Q/KV 从 `attn_normed`/hidden_states 并行派生，
  **没有 Q→KV 边**。
- 保留 hidden_states、q_a_proj/q_a_norm（q_residual）、q_b_proj/q_view/q_transpose/
  q_b_norm/q_rope、共享 KV 的 kv_proj/kv_norm/view/transpose/RoPE、mask、scale、
  sinks、QKᵀ/AV、逆 RoPE、分组输出与 attn_output。
- Indexer 输入为 `attn_normed`、`q_a_normed`（q_residual）与自己的 indexer_cache；
  **不是**外层 compressed_kv 的唯一消费者。
- 新增/修正 `attn.block_bias`：compressor 的 compressed + Indexer topk 只生成
  0/-inf 可见性 mask；topk 不裁剪主 KV/压缩 KV，也不是 compressor 的数据输入。
- CSA/HCA 的 `c_block_bias`/`hca_block_bias` 都走该节点；SWA 无该节点。

### Q/KV/O

- 现有 Attention 图（33–38 节点，32–41 条边）包含 Q、KV、O 的具名 Linear/Norm/
  View/Transpose/部分 RoPE/逆 RoPE/分组输出；KV 另有独立新投影图。
- 连线标签改为显示真实张量身份与全称 shape（semantic candidate 不再隐藏非竖直边
  标签；共享 `separate_edge_labels` 避免重叠）。
- 辅助 cos/sin 作为局部短边接入 Q/KV/逆 RoPE，不伪造成主链顺序。

### 三类缓存

- `cache-<gid>` 图：主 KV 调用前/更新/调用后；压缩器状态（buffer/overlap/
  compressed/entry_count）；Indexer 独立状态；每组一条汇总关系，逐 tensor
  id/name/shape 保留在关系的 `members` 与页面关系 JSON。
- SWA：仅主 KV，**无压缩器/Indexer**（3 节点/2 边）。
- CSA：主 KV + 压缩器 + Indexer（9/6 边），Ca/Cb 重叠与各自状态分开；
- HCA：主 KV + HCA 自身压缩器（6/4 边），无 Indexer。
- `kvcachelen` 只作逻辑历史；主 KV/压缩条目/缓冲余量/调用前后分别表达。

### 压缩器与 Indexer

- `compressor-<gid>`：从真实 `c_*` 算子生成（CSA 11 节点/23 边；HCA 12 节点），
  含 kv/gate 投影、窗口、Ca/Cb 重叠、softmax 汇聚、RoPE、状态更新、block_bias。
- `indexer-<gid>`：从真实 `i_*` 算子生成（10 节点/26 边），含 q_proj、
  KV/gate 投影、自身 buffer/overlap/条目、RoPE、scorer、topk；不把外层
  compressed KV 作为唯一输入。
- 未展开的 D04 内容（Hash/TopK 专家、专家内部）在页面/图注中明确保留。

### 页面

- `_ops_section` 新增 5 个页签：Attention 总览/Q·KV·O、KV 新投影、
  SWA/CSA/HCA 缓存状态、压缩器（CSA/HCA）、Lightning Indexer（CSA）；
  42 个嵌入图位，`?embed=1`。
- D03 banner/图注更新；D04 未交付明确声明；D02 mHC、D01 两阶段/整网端点保留。

## 3. 用户精简审图入口

见 [`D03-verify/review-guide.md`](D03-verify/review-guide.md)：

- 正式页：`viewer/structure.html`；
- 4 组：Attention 总览、缓存、压缩器、Indexer 的 representative links；
- 用户只需按 guide 看新增/变化图；不要求逐张读长截图。

## 4. 验证

- **551/551 测试通过**（17 文件，`tests.log`）；新增：
  - Attention 并行 Q/KV、无 Q→KV、block_bias、Indexer q_residual 输入；
  - SWA/CSA/HCA 缓存类型与调用前后；
  - 压缩器/Indexer 具名算子、无外层压缩 KV 唯一输入；
  - 新图几何非负。
- **38/38** 图 validate/deliver/check 通过；38/38 重建 HTML 与交付字节一致。
- 38 图显式标签重叠 **0**、SVG 可见文字重叠 **0**、省略号 **0**；
  42 图位短 iframe **0**。
- `geometry-audit`：cache/compressor/indexer 节点、via 外沿 min x/y ≥0。
- Attention 语义审计：block_bias 3 组（CSA/HCA）、Q→KV 0、Indexer→Compressor 0、
  q_a_norm→Indexer 标签带 shape。

## 5. 浏览器限制

dsh 当前无法启动 Chrome：

- `Failed to initialize sandbox. Operation not permitted`
- `GPU process exited unexpectedly: exit_code=6`
- `Timeout Page.enable`

已保留原始回执：

- `D03-verify/probe-attempt/`（geometry-only，exit 1，0 截图）
- `D03-verify/probe-attempt-full/`（完整，exit 1，0 截图）
- `D03-verify/native-attempt/`（attention/cache/compressor/indexer 原生检查环境失败）
- `D03-verify/visual-attempt/`（4 张 visual-check 失败，0 截图）
- `D03-verify/native-evidence.json`（38 图 validate/deliver/check 通过；
  browser-check 环境失败）

因此未完成最终候选原生/嵌入 bbox、1440/1920 截图、查看器缩放/长图滚动实测和
逐图视觉判断。用户按 `review-guide.md` 审查；Codex 按
`browser-runbook.md` 补跑。本报告不宣称视觉通过。

## 6. 历史与 E01 保护

- 继承 22242：**22240/22242 未变**；2 个动态例外
  `structure_reader/outputs/.DS_Store`、`docs/tasks/model-structure-2026-10-28/monitor-state.json`
  为外部元数据/监控，未回滚，也不声称全无变化；
- 22641 输出文件：**22641/22641 未变**，硬变更 0；
- 协调 2827：2826/2827 未变；`tickets.json` 为 Codex 派发 D03 的外部更新，只读；
- D01/D02 已接受候选 124/124 未变；两处删除保持；Excel 最新两候选逐字节一致；
- E01 隔离：未改 `outputs/excel-performance-style-2026-11-03`、
  `docs/tasks/excel-performance-style-2026-11-03` 或 Excel 公式/求值器；当前未跟踪的
  `E01-verify/`、`dpskperfexcel草稿.xlsx` 为外部新增，未回滚/纳入 D03。
- 源文件快照 222：214 未变、8 修改（semantic/archify/preview/page/build/tests）。
- 接手 git status 124 行；当前 126 行，差额为外部 `E01-verify/` 与
  `dpskperfexcel草稿.xlsx`，不是 D03 产品改动。

## 7. 限制与停止点

1. 真实浏览器/截图/用户视觉审查未完成；不宣称 AC17/19 浏览器通过。
2. Attention/MoE 总览仍复用既有语义事实；D04 的 Hash/TopK 与专家内部未实施。
3. 缓存图为状态组汇总，逐 tensor 身份/形状在关系 members；没有复制第二套事实。
4. 未做 D04/D05，未关 Issue、未 commit/push、未恢复自动化；停待 Codex 复核与用户审查。
