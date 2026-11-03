# E01 layout-v2 实施结果（第二轮布局反馈）

状态：**layout-v2 完成，停在待 Codex 工程复核与用户阅读**。只实施第二轮布局反馈，
不进入 E03；未改 HTML/结构提取/CLI/共享 schema/求值器/旧候选，未关票、未 commit/push、
未恢复自动化、未进入旧 HTML 会话。

旧 v1 与 rework 交付、报告、root `E01-verify/` 均保留；新产物在
`outputs/excel-performance-style-2026-11-03/E01/layout-v2/`，证据在
`docs/tasks/excel-performance-style-2026-11-03/E01-layout-v2-verify/`。

## 新交付

| 交付 | 路径 | sha256 |
|---|---|---|
| layout-v2 xlsx | `outputs/excel-performance-style-2026-11-03/E01/layout-v2/deliverables/deepseek_v4_flash_perf_layout.layout-v2.xlsx` | `2d511eddfa670fd3328410f01c35b3a931c4db7d9e7fbc18b07fd9787bd30d92` |
| 重算副本 | `.../layout-v2/deliverables/deepseek_v4_flash_perf_layout.layout-v2.recalc.xlsx` | `03456b628ec07aa5566d7f4abf310f541c4e5ee2bf61659ce06b3f0ae50b7c0b` |
| 新 patch（累积当前 E01 Excel 文件） | `.../layout-v2/patch/E01-layout-v2-source.patch` | `8cfbbef355182664c3633b0c39b5f74d7a00bd50eb02ba35f91df69bd323be41` |
| 验证 JSON | `docs/tasks/.../E01-layout-v2-verify/verification.json` | `ba6cb7d3fefa6ed96884201e726cf238b405d2d1f32f9a3b17add7bd0440c43b` |
| 清单 | `.../layout-v2/MANIFEST.json` | `53a4b0aecd9682f508a0ae041eb94df9b2cba45018267e90aa45f85304633497` |
| 复现命令 | `.../layout-v2/reproduce.md` | — |

patch 含 v1 适配模块（v2 依赖的公共公式/参数/稳定 ZIP 辅助）、v2 模块、v2 生成脚本与
v2 测试；`git apply --check` 通过，临时副本应用后重建 sha256 与交付件一致。

## 五项反馈如何落地

1. **主表减冗余**：原 291 个算子实例 → 101 个共用/差异模板；主表 101 个模板行 +
   21 个区块首行 + 21 个块小计 + 1 个整网总计 = 144 行（不含表头），Prefill/Decode
   各 191 行。删除了顶部 revision/实现/冻结输入/草稿说明/长教学段落和表尾规则清单；
   场景说明缩成短句。来源、限制、完整映射在辅助区。
2. **结构只在 A 列标一次**：每个区块只在 A 列首行写一次（如“共用 mHC（×86 次）”），
   模板行 A 列留空；不再有逐行结构标签或跨 12 列层组横幅。块小计仍保留 A 列标签，
   用于模板成本×使用次数。
3. **mHC 共用模板**：attention 前/FFN 前 7 个算子按算法/规则、维度、参数作用域、
   条件、排除计费与 repeat 判定同构，合成一份 7 行模板，使用次数 86（每层 2 次×43 层）；
   块小计 ×86，实际执行成本没有只算一次。
4. **Attention Q/KV 前置共用**：q_a_proj/q_a_norm/q_b_proj/q_b_norm/q_rope_apply/
   kv_proj/kv_head_view/kv_rope_apply 共 8 行，使用次数 43；SWA/CSA/HCA 的 cache、
   compressor/Indexer、qk/av 核心和条件差异仍分开。
5. **复杂维度提到参数层**：新增派生参数
   `M_attn/M_og/M_route/M_res/K_og/K_hc/N_hc/N_qkv/N_gu/N_slice/N_comb/N_disp/
   Kv_vis(CSA/HCA)/Nhit_ok(Hash/TopK)/H_bias_on(HCA)`。主表 n/T/D/M/K/N 主要为
   单个绝对参数引用或 `=D行*E行`、`=F行` 这类短算术；D:I 中无 `IF(`/`ISNUMBER`，
   最长 18 字符。吞吐/合法性检查留在参数与耗时公式。

## 同构判定与模板覆盖

等价签名包含：`family`（mHC 去 attn/ffn 前缀，残差同理）、CUBE/VEC class、全部
terms(b/m/k/n)、repeat、conditional、charge/excluded、guards，以及递归参数签名
（参数 id/formula/value/别名，含 CSA/HCA 作用域的 M/Ncomp）。额外语义上下文：
`cache_update` 按 attention_type 分 SWA/CSA/HCA，`moe_router` 按 moe_type 分 Hash/TopK；
不能仅凭 op 名合并。

- 原实例 291：可计费 282、excluded 8、container 1。
- 模板 101；使用次数合计 2979。
- 共用：mHC 7 行×86；RMSNorm 1×86；QKV 8×43；O/后处理 4×43；残差 3×86。
- 差异：SWA cache 1×2、核心 5×2；CSA cache 1×21、compressor/Indexer 16×21、
  核心 7×21；HCA cache 1×20、压缩器 9×20、核心 7×20。
- MoE：路由/专家模板 11×43；Hash 路由 2×3；TopK 路由 1×40；路由专家切片
  Hash 1×3、TopK 1×40；共享专家 2×43。
- 整网入口 6×1、出口 7×1。
- 逐实例映射 291 行在 `模板映射` 工作表，覆盖 `共用/差异模板=282`、`excluded=8`、
  `container=1`，无“未映射”。

## 成本对照（无重复/无丢项）

`verification.json` 中 `nhit6_totals` 与按原规则直接求和逐项一致（N_hit=6，合法输入）：

| 场景 | 口径 | 前轮参考 | layout-v2 | 差额 |
|---|---|---|---|---|
| Prefill | cube | 25,975,584 | 25,975,584 | 0 |
| Prefill | vec | 103,636,557.5 | 103,636,557.5 | 0 |
| Prefill | op | 129,612,141.5 | 129,612,141.5 | 0 |
| Decode | cube | 3,726,064 | 3,726,064 | 0 |
| Decode | vec | 116,127,598.71875 | 116,127,598.71875 | 0 |
| Decode | op | 119,853,662.71875 | 119,853,662.71875 | 0 |

同一对照也写入 `模板映射` 辅助表，cost_diffs 全 0。默认 N_hit 留空时 cube 总计仍可读，
vec/op 总计为 `未知*`；N_hit=6 时三列均可算。

## 验证

`verify_workbook_v2.py`（`passed=true`、`recalc=true`、`errors=[]`）：

- 覆盖：291/282/8/1/101；逐实例映射 282+8+1；cost_diffs 全 0。
- batch：C8 改为 2 后所有成本与 baseline 完全一致（`mismatches=0`）；计算源 B=1。
- 派生联动：seqlen=16、kvcachelen=4096、N_hit=6 后，两场景整网总计与重新按规则求和
  一致；`hca_block_bias` 为 8（有历史且 seqlen≠1）。
- 非法吞吐：CUBE J6/VEC J7 × 负数/0/空白/数字文本/普通文本/布尔，12/12 case 通过；
  相关模板行与整网总计保持 `未知*`；Decode 条件不执行的 `hca_block_bias` 仍为 0。
- 单位：Prefill/Decode 的 K6/K7 均为 `element/ms`。
- 复合/重复：`i_scorer` 主表 J 引用 `项分解` 合计（2228224/4096=544）；
  `shared_gate_up` 等 repeat 行主表 J 行内 ×2。
- 旧导出：重建 sha256 `2dabd3af336e08b59cc847b7cc8d5c89fc8bc46244abf3a685a0077cc365b05e`，
  与候选 `decoder_variants.xlsx` 一致。

测试（`E01-layout-v2-verify/test-logs/`）：

- layout-v2：8 tests OK。
- v1 布局回归：14 tests OK。
- 性能引擎：12 tests OK。
- 旧导出：57 tests OK。
- v2 旧测试：35 tests OK（14 个依赖本机 HF 输入的用例按设计跳过）。

## 渲染与草稿对照

新 xlsx 用 LibreOffice Kit 重算后渲染，证据在
`E01-layout-v2-verify/render/`：

- `prefill_top.png`：顶部短标题/场景/吞吐/紧凑参数/第一段模板与区块；
- `prefill_main.png`：mHC、QKV、SWA/CSA/HCA 差异、O/残差、MoE、汇总；
- `decode_top.png`：Decode 顶部；
- `mapping_usage.png` / `mapping_instances.png`：模板使用次数、逐实例映射；
- `params.png`：派生参数与来源；
- `terms.png`：复合/重复/excluded 项分解；
- `draft_reference.png`：根目录草稿的渲染对照。

对照草稿：同样采用顶部左侧参数、右侧吞吐、主表 12 列、A 列通路标识；v2 不沿用草稿
示意数值/层数，且用共用模板替代重复展开。LibreOffice 报缺 `Calibri`/`宋体` 并回退；
**原生 Microsoft Excel 未安装，未做原生公式缓存/冻结/打印/字体验收**。

## Patch 与边界

- patch 4 文件：v1 公共模块、v2 模块、v2 脚本、v2 测试；apply-check 通过，临时副本
  重建 sha256 66c0b7…c421，v2 测试 8 OK。
- 原 `structure_reader/src`、`tests`、HTML、extract/render、CLI/构建编排、共享
  schema/求值器、旧 Excel、候选、D01—D05 协调文件未由本票修改。
- 并行 HTML 外部变化 5 个文件已单独归因，未纳入 patch、未回滚；哈希见
  `E01-layout-v2-verify/protected-paths.sha256`。
- 未 `git apply` 到原工作区、未 commit/push、未关闭 Issue、未恢复自动化、未进入旧
  HTML 会话、未启动 E03。

## 验收对照（第二轮反馈）

- [x] 主表无重复层组横幅/逐行结构标签；A 列每结构只标一次；仍为 12 列。
- [x] 同构 mHC/QKV 不按四组重复；辅助映射 291 行覆盖原实例，无丢项/多计。
- [x] 相同合法输入下逐类成本、条件、真实层数与整网总计与前轮一致（差额 0）。
- [x] 修改 seqlen/kvcachelen/吞吐后派生参数与汇总同步；新增参数有含义/作用域，
  主表维度主要是直接引用/短算术。
- [x] 非法吞吐、batch 固定、未知传播、合法 0、复合 repeat、旧导出保护回归通过。
- [x] 新 xlsx 渲染并对照草稿；截图来自新 xlsx；原生 Excel 未测已明确。
- [x] 工程正确性与用户布局确认分开；停待 Codex 与用户阅读，不进入 E03。

## 停点

layout-v2 实施完成，等待 Codex 工程复核与用户阅读。不自行接下一票。
