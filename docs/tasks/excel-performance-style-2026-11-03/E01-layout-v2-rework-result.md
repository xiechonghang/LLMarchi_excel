# E01 layout-v2-rework 结果（P1 输入校验 + P2 标题）

状态：**定点返修完成，停在待 Codex 工程复核与用户阅读**。保留 layout-v2 共用模板布局，
不进入 E03；未改 HTML/结构提取/CLI/共享 schema/候选/旧版，未关票、未 commit/push、
未恢复自动化。

新产物在 `outputs/excel-performance-style-2026-11-03/E01/layout-v2-rework/`，证据在
`docs/tasks/excel-performance-style-2026-11-03/E01-layout-v2-rework-verify/`。旧
layout-v2 文件、报告、证据原样保留。

## 新交付

| 交付 | 路径 | sha256 |
|---|---|---|
| rework xlsx | `outputs/excel-performance-style-2026-11-03/E01/layout-v2-rework/deliverables/deepseek_v4_flash_perf_layout.layout-v2-rework.xlsx` | `953592e594cdccfa7aed8914f5fed791cc10a0e972e6bc585ee4b5c3be751699` |
| 重算副本 | `.../layout-v2-rework/deliverables/deepseek_v4_flash_perf_layout.layout-v2-rework.recalc.xlsx` | `1dffb7c4640240a7dc33a2bcf0d763e7ce6a03ddd24985ec0dd1614327a27619` |
| 累积 patch | `.../layout-v2-rework/patch/E01-layout-v2-rework-source.patch` | `9f49f41ffb5a60fb09b2076e7f0ad8cc954281a0aff081e5347a1315d1d203d6` |
| 验证 JSON | `docs/tasks/.../E01-layout-v2-rework-verify/verification.json` | `0f352ce90ff71795f2d47c9b5744f4ae345bcd55b966eae811862ce9c06ed17e` |
| 清单 | `.../layout-v2-rework/MANIFEST.json` | `ecc869c05e8f5c2d9d3bb236c0161458feaf29ef40ec8a970d256a97829c9f8a` |

`MANIFEST.json` 另列 build summary `f4365a…`。所有新文档使用上述最终哈希；
复现命令已补回仓库根目录 `cd /Users/xiechonghang/Workspace/LLMarchi_excel`。

## P1：输入校验回到参数层

问题根因是维度格直接引用原始输入，成本只检查 `ISNUMBER`，导致负数/小数/布尔/文本被
当作数值。返修不恢复逐行长公式，而是建立参数层校验值：

- 场景输入新增 `V_seqlen/V_kvcach/V_Nhit`，公式显式检查
  `ISNUMBER + NOT(ISLOGICAL) + INT=自身 + >=0`，非法返回 `未知*`。
- 基础配置（H/Nh/G/Dh/Rq/Rk/Ro/W/Nhc/E/K/Es/Id/Ih/Ki/M/Vocab 等）增加
  `V_*` 校验副本；基础配置格本身也改为校验公式，原始值放入隐藏 `原始参数` 表。
- 派生参数公式（Lkv/Ncomp/Kv_vis/Nhit_ok/H_bias_on 等）重写为引用 `V_*`，
  主表维度只引用校验格，保持短公式 `=D*E`/`=F`/`IFERROR(D*E,"未知*")`。
- 正数域：head/group/层宽/专家数/压缩率等要求 ≥1；seqlen/kvcachelen/N_hit 允许 0；
  N_hit 空白=未知。合法 0 与条件不执行 0 分开：seqlen=0 时 q_a_proj 成本为 0；
  Decode 无条件执行的 `hca_block_bias` 仍为 0。

`verification.json` 的 `invalid_dimensions` 12 个 case 全通过：

| 输入 | 值 | 观察 |
|---|---|---|
| seqlen | -1 / 空白 / 1.5 / TRUE / "8" / "abc" | q_a_proj E/G/J/L 全 `未知*`，整网总计 `未知*` |
| seqlen | 0 | q_a_proj J/L=0（合法 0） |
| H | -1 | q_a_proj J/L=`未知*` |
| G | 0 | o_a_proj J/L=`未知*` |
| Nh | 0 | qk_scaled J/L=`未知*` |
| kvcachelen | -1 | Decode `hca_block_bias`=`未知*` |
| N_hit | -1 | `moe_expert_slice` K=`未知*` |

非法依赖没有被 MAX/SUM/直接引用吞成 0；12/12 case 通过，失败 0。

## P2：A 列标题与行高

- A 列宽 17→**28**；区块首行高 20→**28**，块小计行高 **24**；不缩小字体、不重加
  横向横幅。
- 路由专家切片用短而明确的标签：**Hash 专家切片（×3 次）**、**TopK 专家切片
  （×40 次）**；`mHC（×86 次）`、`Q/KV 前置（×43 次）`、`HCA 压缩器（×20 次）` 等
  同级标题也完整可见。
- `block_titles.failures=[]`：21 个区块的完整名称/调用次数/A 列唯一标识全部通过；
  渲染见 `render/prefill_blocks_1.png` 与 `prefill_blocks_2.png`。

## 保留的模板布局与成本

- 主表仍为 101 模板行 + 21 区块首行 + 21 块小计 + 1 整网总计 = 144 行（表头在第 58 行，
  Prefill/Decode 各 202 行）；仍为 12 列。
- mHC 7 行×86、QKV 8 行×43、291 实例→101 模板、覆盖 282+8+1、使用次数 2979 均保留。
- N_hit=6 合法输入下总成本与前轮一致（cost_diffs 全 0）：

| 场景 | cube | vec | op |
|---|---|---|---|
| Prefill | 25,975,584 | 103,636,557.5 | 129,612,141.5 |
| Decode | 3,726,064 | 116,127,598.71875 | 119,853,662.71875 |

- 旧导出重建 sha256 `2dabd3af…b05e`，与候选 `decoder_variants.xlsx` 一致。

## 验证与测试

- `verification.json`：`passed=true`、`recalc=true`、`errors=[]`。
- 覆盖：291/282/8/1/101；逐实例映射 282+8+1；cost_diffs 全 0。
- 非法吞吐 12/12 通过；batch C8 改动不改成本；seqlen=16、kvcach=4096 联动总计正确。
- 单位 K6/K7=`element/ms`；复合 `i_scorer` SUM、repeat ×2 保留。
- 测试：layout-v2 **9**、v1 布局 14、性能引擎 12、旧导出 57、旧 v2 35（14 skipped）
  全部 OK。
- 标题检查 `block_titles.failures=[]`；P1 `invalid_dimensions.failures=[]`。

## Patch 与边界

- patch 含 v1 公共模块、v2 模块、v2 生成脚本、v2 测试 4 个文件；apply-check 通过，
  临时副本重建 sha256 `953592…1699`，v2 测试 9 OK。
- 只新增 layout-v2-rework、本报告和 rework 证据；原工作区 src/tests/HTML/extract/
  render/CLI/schema/候选、旧 Excel 未改。
- 并行 HTML 外部变化 5 个文件单独归因，未纳入 patch；哈希见
  `protected-paths.sha256`。
- 未 apply 到原工作区、未 commit/push、未关 Issue、未恢复自动化、未进入 E03。

## 渲染与限制

最终重算副本渲染了顶部、区块上/下段、完整主表、Decode、模板使用、参数、项分解和
草稿对照，共 9 张；`render_manifest.json`/`render.sha256` 绑定哈希。LibreOffice 报缺
`Calibri`/`宋体` 并回退；**原生 Microsoft Excel 未安装，未做原生公式缓存/冻结/打印/
字体验收**。

## 停点

P1/P2 定点返修完成，保留已通过的共用模板布局与合法成本。停止等待 Codex 与用户阅读，
不自行接下一票。
