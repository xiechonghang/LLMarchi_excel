# D03 rework03 结果：输出扇出边界

日期：2026-11-03（Asia/Shanghai）。只修 `_stage_graph` 同一 tensor 同时供图内/图外消费时
丢失段出口的规则；未重做布局、输入推导、HCA/Attention/Q-O/缓存。未启动 D04/D05，
未关票、未 commit/push、未恢复自动化、未操作 E01/旧 T14。

## 1. 新候选

| 项 | 值 |
| --- | --- |
| 候选 | `structure_reader/outputs/deepseek_v4_flash_0731_hf/73034fda94f87647a270cee4e88035c8e1fee83c927dde6c3feb5f6019bc8ca7` |
| 正式入口 | `viewer/structure.html`（2,234,822 bytes，sha256 `83f5492ccaae8838c155113d84c65890f7ad942393d2475b83149342ccc8a362`） |
| 图 | 51 张 Archify 图、55 个嵌入图位 |
| 证据 | `D03-rework03-verify/` |

构建命令：

```bash
PYTHONPATH=structure_reader/src \
  /Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -c "from structure_reader.cli import main; import sys; sys.exit(main(['build-hf','--model','deepseek_v4_flash_0731_hf','--all-variants','--archify-layers','--json']))"
```

## 2. P1 共享输出规则

内部消费和外部消费分别判断：

- 阶段输出先看是否被**任意图内阶段**消费，已有阶段边保留；
- 再看真实 block ops 中是否有**图外消费者**；有则即使图内也消费，仍连 `s-out`，
  condition 记录外部消费者（如 `i_q_proj`、`kv_concat`）；
- 纯图内中间张量（`c_chunk_kv`、`c_windows_kv` 等）没有外部消费者，不新增段出口；
- 只有持久状态写回（`cache_*_post`）才保留“真实末端”出口；普通中间产物
  `first_window_position`/`c_positions` 不冒充输出；
- 张量身份、完整 shape、外部消费者与来源均由真实 facts 推导，没有新 fact。

五张受影响图最终 candidate/relations：

| 图 | 内部边 | 图外出口 |
| --- | --- | --- |
| `q-csa-hash` | q-lowrank→q-b q_a_normed | q-lowrank→s-out，i_q_proj |
| `q-csa-topk` | 同上 | 同上 |
| `compressor-csa-hash` | s-state→s-bias cache_compressed_c_post | s-state→s-out，kv_concat |
| `compressor-hca-topk` | 同上 | 同上 |
| `compressor-csa-topk` | 同上 | 同上 |

回归覆盖内部+外部双消费、纯内部不泄漏，并检查 candidate/relations 的五图真实出口。

## 3. 保留

- 阶段输入由真实 op inputs 推导、未知 id 报错、只有真实 `[]` 标量；
- HCA 类型/位置/None、Attention block_bias、Q/O 入口、缓存累计/全称/来源、分图与布局；
- 既有 53 项阅读测试和全部旧断言不降；不删/不排除差异。

## 4. 验证

- **559/559 测试通过**（17 文件）。
- **51/51** 图 native validate/deliver/check；51/51 重建 HTML 一致。
- 51 图标签重叠 0、SVG 文字重叠 0、省略号 0；55 图位短 iframe 0。
- 五图 fanout 审计：boundary_edges 2/2/7/5/7，外部消费者均为
  `i_q_proj` 或 `kv_concat`，`leaked_internal=false`。
- 新增图 viewBox 推算（嵌入 1200px/原生 1440px）全部 ≥6px；Q 785、
  compressor 1052、Indexer 1498、主缓存 1252。
- 既有新候选 644/644 显示门禁已通过；本候选新增扇出边绑定后需 Codex 复跑。

## 5. 浏览器/用户审查

本机 Chrome 仍因 sandbox/GPU 初始化失败；最终候选页面探针一次尝试 exit 1、0 截图，
HCA 压缩器 browser-check 一次尝试 exit 1。回执：
`D03-rework03-verify/probe-attempt/`、`native-attempt-hca/`。

按 `browser-runbook.md` 在授权 Chrome 复跑 55 图位和 HCA 原生；用户按
`review-guide.md` 看五图输出扇出及精简入口。视觉标 **待用户审查**，未宣称通过。

## 6. 保护

- 输出保护并集 **23225/23225 未变**，硬变更 0；`8136…` 只读保留；
- 22242：22240/22242，2 个动态例外 `.DS_Store`、`monitor-state.json`，未回滚；
- 协调 2827：2826/2827，`tickets.json` 为外部派发更新；
- D01/D02 接受候选 124/124；两处删除保持；Excel 最新两候选逐字节一致；
- E01 隔离树未触碰；当前 dirty 126，外部 `E01-verify/`、`dpskperfexcel草稿.xlsx` 未纳入；
- 源文件 222：220 未变、2 修改（`archify_layers.py`、测试）。

## 7. 停止点

未做 D04/D05；未关 Issue、未 commit/push、未恢复自动化；停待 Codex 技术复核与用户
视觉审查。
