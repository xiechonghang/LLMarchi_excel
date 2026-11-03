# D03 返修结果：Attention、Q/KV/O、缓存、压缩器/Indexer

日期：2026-11-03（Asia/Shanghai）。仅返修 D03/#19；未启动 D04/D05；未关票、
未 commit/push、未恢复/新建自动化、未访问旧 T14。状态：**五项代码/布局/静态门禁
已收尾；真实浏览器截图与用户视觉审查待补**。

## 1. 新候选

| 项 | 值 |
| --- | --- |
| 候选 | `structure_reader/outputs/deepseek_v4_flash_0731_hf/8cfa840daf2cefb28ad298670faff9ae45765da15ddebf3af8e280ae57d03613` |
| 正式入口 | `viewer/structure.html`（2,214,640 bytes，sha256 `8966274700688fde8ffeec4aaa3977aac0adac94f3c88c4ecb115def370fb384`） |
| 图 | 51 张 Archify 图、55 个嵌入图位 |
| 证据 | `D03-rework-verify/` |

构建命令：

```bash
PYTHONPATH=structure_reader/src \
  /Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -c "from structure_reader.cli import main; import sys; sys.exit(main(['build-hf','--model','deepseek_v4_flash_0731_hf','--all-variants','--archify-layers','--json']))"
```

## 2. 五项逐项响应

### P1 block_bias 真实输入/输出

- `attn.block_bias` 不再是 `kind=aux`；改为真实 **composite 计算节点**，
  由 `compressor` 的 compressed、CSA 的 `indexer.topk`（HCA 的 `position_ids`）
  输入，输出 `block_bias` 给 `mask_extend`。
- 三组实际交付的 Attention（CSA×2、HCA）最终 candidate 均有：
  `compressor→block_bias`、CSA 的 `indexer→block_bias`、`block_bias→mask_extend`；
  HCA 另有 `position_ids→block_bias`。
- 恢复 `test_archify_preview.py` 完整 trace 断言，删除此前
  `expected -= {e.compressor.block_bias2, e.indexer.block_bias}` 的掩盖；
  新增测试直接检查最终 candidate connections，不只看 bundle 字段。
- 保留 `scale→mask_add` 等原有直连，冻结旧投影职责。

### P2 HCA 不套 CSA

- HCA `attn.block_bias` 类型为 `DeepseekV4HCACompressor.block_bias`，
  condition 为 `heavily_compressed_attention`；输入
  `compressed_kv + position_ids`，参数仅 `compress_rate`；
  无 `index_topk`、无 Indexer/TopK。
- 明确 `seqlen==1` 或 `compressed_len==0` 返回 `None`，`mask_extend`
  走 `F.pad` 零扩展分支；没有把 HCA 伪装成 CSA。
- HCA 压缩器细图包含真实 `hca_block_bias` 与 `c_mask_extend`；
  不再只取 `c_*` 前缀而遗漏 `hca_*`。

### P3 缓存逐项身份/shape、更新输入与参数

- 缓存拆为 **9 张真实子图**：
  `cache-main-*`（4 组）、`cache-compressor-*`（CSA×2/HCA）、
  `cache-indexer-*`（CSA×2）；主图不再用 6 项只显示第一项 shape 的汇总。
- `cache-main-*` 增加：
  - 本调用新 KV（`kv_rope`）；
  - `cache.update`；
  - 本次可读 `kv_full` 与 `repeat_kv` 消费者；
  - 调用前/后 keys/values/累计长度。
- 页面新增**缓存逐项表**（每项 id、阶段、完整符号 shape、名称、来源），
  逐项可见；同时显示 cache.update 的 full/cumlen 规则。
- cache/compressor/indexer 只保留就近参数；Indexer 专属
  `index_n_heads/index_head_dim/index_topk` 可见，避免 o_groups 等块通用参数占满。

### P4 原生/嵌入字号

- 取消 `wide_d03` 的 1.8 倍/2000px 膨胀；压缩器/Indexer 改为 6–7 个逻辑阶段，
  缓存拆为主 KV/压缩器/Indexer 三张窄图；节点只显示阶段名，逐项 shape 在边/表。
- 新图 viewBox 宽度：
  - HCA 压缩器 **1140**（修复前 4616）；
  - CSA 压缩器 1121；CSA 缓存 1112；Indexer 缓存 1140；
  - 主 KV 缓存 1564；Indexer 1570；Q 785；O 1100。
- 推算（页面嵌入 1200px、原生 1440px）新增图全部 ≥6px；
  页面 55 图位无短 iframe。**真实 Chrome bbox/两视口截图未在本机完成**，
  不把推算冒充实测。

### P5 Q/O 独立入口

- 新增 `q-<gid>` 4 张、`o-<gid>` 4 张独立图，页面新增
  “Q 通路（独立）”“O 通路（独立）”页签，与 KV 新投影并列。
- Q：q_a_proj/q_a_norm→q_b_proj→View/Transpose 轴布局→q_b_norm→部分 RoPE。
- O：AV/逆 RoPE/轴布局→分组 Reshape→o_a_proj→o_b_proj。
- Attention 总览仍保留 Q/KV 并行与关联，没有伪造 Q→KV 顺序；
  各图有返回总览入口。

## 3. 验证

- **554/554 测试通过**（17 文件）；新增/恢复：
  - 最终 candidate 的 compressor/indexer→block_bias→mask 连接；
  - HCA 类型/condition/position_ids/None 分支；
  - 缓存逐项 id/shape、新 KV/full/cumlen、Q/O 独立通路；
  - 完整 trace 断言恢复，不再减 expected。
- **51/51** 图 validate/deliver/check 通过；51/51 重建 HTML 与交付字节一致。
- 51 图显式标签重叠 **0**、SVG 可见文字重叠 **0**、省略号 **0**；
  55 图位短 iframe **0**；新图几何非负。
- Attention 语义审计：block_bias 3 组（CSA×2/HCA）、Q→KV 0、
  Indexer→Compressor 数据边 0。

## 4. 浏览器/用户审查

- 本机 Chrome 仍报 sandbox/GPU 初始化失败；最终候选：
  - 新版 55 图位探针一次尝试 exit 1、0 截图（`probe-attempt/`）；
  - HCA 压缩器原生 browser-check 一次尝试 exit 1（`native-attempt-hca/`）；
  - 51 图 native validate/deliver/check 通过，browser-check 环境失败。
- 这不是“浏览器不可用”的通用结论；Codex 已确认授权环境 Chrome 可用。
  按 `D03-rework-verify/browser-runbook.md` 补跑 1440/1920 页面与原生新增图。
- 用户按 `D03-rework-verify/review-guide.md` 看 Q/O/三类缓存/HCA 压缩器/Indexer 的
  精简入口；无需逐张长截图。视觉项标 **待用户审查**，未宣称通过。

## 5. 保护

- 返修基线 160 文件 + 继承 D03 22641 输出 = **22801/22801 未变**，硬变更 0；
- D01/D02 接受候选 124/124 未变；旧候选 `7c724…` 未覆盖；
- 22242：**22240/22242 未变**，2 个动态例外 `.DS_Store`、`monitor-state.json`，
  如实记录、未回滚；
- 协调 2827：2826/2827 未变；`tickets.json` 为外部派发更新；
- 两处删除保持；Excel 最新两候选逐字节一致；
- E01：未改其输出/文档/Excel 公式/求值器；外部未跟踪
  `E01-verify/`、`dpskperfexcel草稿.xlsx` 未触碰/未纳入；
- 接手 dirty 124，当前 126，差额为上述外部项；源文件 222 中 214 未变、8 修改。

## 6. 限制/停止点

1. 真实浏览器 bbox、两视口截图、页签/缩放/长图滚动与视觉判断未完成；
   P4 的真实字号需 Codex 授权环境复核。
2. 缓存逐项 shape 在页面表和分图边/`members` 中；没有把 30 项塞回一张宽图。
3. 未做 D04/D05；未关 Issue、未 commit/push、未恢复自动化；停待 Codex 技术复核
   与用户视觉审查。
