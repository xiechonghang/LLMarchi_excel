# E01 / GitHub #22 实施结果

状态：**E01 实施完成，停在待 Codex 复核与用户阅读**。未启动 E02/E03，未接入正式
CLI/构建入口，未 commit/push，未关闭 Issue，未恢复自动化，未进入旧 HTML/T14 会话。

本票只在隔离快照中实现 Excel 新布局；原工作区只新增本任务 outputs 子树、
本报告与 `E01-verify/`。原 `structure_reader/src`、`tests`、HTML、结构提取、
render、CLI/构建编排、共享 schema/求值器、旧 Excel、历史候选和 D01—D05 协调文件
均未由本票修改。

## 交付物

| 交付 | 路径 | sha256 |
|---|---|---|
| 可打开样表 | `outputs/excel-performance-style-2026-11-03/E01/deliverables/deepseek_v4_flash_perf_layout.xlsx` | `7abe374d3978b0d3cc0579787dafa054b2a84a03ba22d852937f70d7c135a48e` |
| LibreOffice 重算副本（证据） | `.../E01/deliverables/deepseek_v4_flash_perf_layout.recalc.xlsx` | `a7eb0a914a1b2781bfe216e574ea4435a72125c8540aefb67973cb86082500e4` |
| 源码 patch（待集成） | `.../E01/patch/E01-source.patch` | `f2e0725fe45cfbf0a51132532abe6ce33411722ae89577c65729c2a4d8da3e82` |
| 构建摘要 | `.../E01/deliverables/build_summary.json` | `cc3b3361f8d99ae61074cf0bc9b544d8e7fb6bc803be9a083122311a9c507c28` |
| 快照清单（逐文件来源/哈希） | `.../E01/workspace/MANIFEST.json` | `ab006156a0774260a7f1ad8a32da3d8e85022470bc51600880467dfc01004f1c` |
| 验证结果 | `E01-verify/verification.json` | `52919a7d85808c26d66686e6c428feced027d7b77a665a84c8902e337893ed20` |
| 复现命令 | `.../E01/reproduce.md` | — |

xlsx 为稳定字节：相同冻结输入与源码重复构建得到同一 sha256，ZIP 条目时间固定为
1980-01-01（复用 `operator_workbook._normalize_zip` 的同一实现）。

## 隔离快照与真实输入

- 快照：`outputs/excel-performance-style-2026-11-03/E01/workspace/`。
- 源码：当前工作区 `structure_reader/src/structure_reader` 的完整副本（排除
  `__pycache__`），因此包含未提交/未跟踪模块，不是 Git HEAD 的完整源码；逐文件状态
  `tracked-clean/tracked-modified/untracked` 与 sha256 见 `workspace/MANIFEST.json`。
- 测试：`hf/test_operator_workbook.py`、`test_operator_workbook_v2.py`、
  `test_perf_engine_models.py` 及 `hf/fixtures/`。
- 冻结真实结构：候选
  `c47633db4bec814108f664efb4a0898f862c97f49eeaa860291868fa872b2486` 的
  `data/layer_structures.json`（sha256 `f9c444115ecdd4333847df77d4aa440b5090f3ce5f64eb68fd6a30ce9e2f236c`）
  与 `data/network_overview.json`（sha256 `2255b767cc124ad33500ac0e08ecedddf51fc486c3fc17ae4f2dfd29485f7b6a`）。
- 布局参考：根目录 `dpskperfexcel草稿.xlsx` 只读，sha256
  `b62109a15875c894cd7dfd892e336a5d82eeeeeae04a5dd6d4148104408a3de1`；其模型名、
  层型、层数、参数和维度均未作为事实使用。
- 未复制原 `structure_reader/outputs/**`、`.git/`、虚拟环境或缓存。

## 实现

新增三个文件（均在 patch 中，未改既有模块）：

1. `structure_reader/src/structure_reader/export/deepseek_perf_layout.py`
   — Excel 专属 12 列布局适配器；消费
   `hf_workbook.build_structures_workbook_spec`，复用 `operator_workbook` 的公式编译、
   输入/参数守卫、未知文本与稳定 ZIP 辅助函数；不建第二套模型事实或求值语义。
2. `structure_reader/scripts/build_deepseek_perf_layout.py`
   — 只读两个冻结 JSON 的独立生成入口。
3. `structure_reader/tests/test_deepseek_perf_layout.py`
   — 11 个 stdlib unittest。

### 12 列主表

每页主表严格 12 列：

`结构/通路 | op | op type | n | T seqlen or N | hiddensize or D | M | K | N | cube(ms) | vector(ms) | op(ms)`

每行一个算子；Prefill/Decode 各 291 行（VEC 214、CUBE 66、CUBE×2 8、CUBE 复合 2、
容器 1）。顺序按真实串行：整网入口 6 项 → 43 层循环容器 → 四层组（SWA-hash 58、
CSA-hash 76、HCA-topk 68、CSA-topk 75）→ 整网出口 7 项 → 整网串行总计。

### 计算口径

- `batch=1` 固定；`n` 只表示 head 数或 1，不承载 batch/层数/专家数。
- CUBE：`cube(ms)=有效M×K×N÷CUBE吞吐`。`b` 中的 batch/head/分组因子只折入有效 M
  一次；`n` 只用于说明 head。因此注意力批量矩阵乘是 `M=Nh×T`，J 不再乘 `n`。
- VEC：`vector(ms)=n×T×D÷VEC吞吐`；`n` 为 head 或 1，`T` 为 VEC 行数，`D` 为列数。
- 纯 CUBE 的 vector=0；纯 VEC 的 cube=0；`op(ms)=MAX(cube,vector)`，任一未知时
  op 保持 `未知*`，不按 0 计。
- 复合项：`i_scorer` 两个矩阵项先 SUM 再入 CUBE（明细逐项展开）；同形重复
  `shared_gate_up`、`c_kv_gate_proj`、`i_kv_gate_proj` 乘 repeat=2，主表 op type 标
  `CUBE（×2）`，J 公式可见 `*(2)`。
- 条件：HCA `block_bias` 在 `seqlen==1` 或 `N_comp==0` 时合法 0；条件输入未定仍为
  `未知*`。
- excluded：层组内 `mask_model_build`/`rope_main`/`rope_compress`（每页 8 行）只
  标 `整网一次`，不计入单层小计；整网入口/出口各收一次。
- container：`net_layer_loop` 只标 `容器不计`，成本由四个层组总计表达。
- 汇总：单层小计 = 本组计费算子逐列和，先 `COUNT(...)=N` 再 `SUM`，未知不被吞成 0；
  层组总计 = 单层小计 × 真实层数（2/1/20/20）；整网总计 = 入口小计 + 四组总计 + 出口
  小计，各只引用一次，父小计不重复加总。
- 顶部：左侧集中参数（场景/真实层数/基础维度/Attention-Indexer/MoE-FFN/mHC/整网），
  右侧独立 CUBE/VEC 吞吐（默认 4096 MAC/ms、64 元素/ms，黄色可改），参数公式全部
  绝对引用（`$D$23` 等），复制行不漂移。`未知*` 的完整原因在单元格批注与辅助明细。

### 辅助明细

`辅助明细` 工作表 24 列、650 行：参数来源与说明；逐算子完整 shape、源码定位、
模块路径、参数引用、逐 term 的 M/K/N、repeat、条件/守卫/排除、主表行号。复合/重复
项的分解在此可追溯，不改变主表成本。

## 验证

运行命令见 `outputs/.../E01/reproduce.md`；证据见 `E01-verify/`。

### 1. 逐算子重算对照

`E01-verify/verify_workbook.py` 用同一 spec/规则在 Python 中重算每个算子的
CUBE/VEC 耗时，与 LibreOffice 重算后的缓存值比较：

- Prefill/Decode：`errors=0`。
- `op=MAX` 逐行检查通过；纯 VEC 的 J=0、纯 CUBE 的 K=0。
- 单层小计、层组总计、整网总计在 N_hit 给定时逐列相加一致（见下）。

### 2. 不等吞吐联动

把 CUBE 吞吐改为 2048、VEC 改为 128 后重算：528 个非零数值检查，
CUBE 行全部×2、VEC 行全部÷2，`op(ms)` 仍为 MAX，`mismatches=0`；CUBE/VEC 引用
未串线。

### 3. N_hit 未知 → 填写

- 默认 `N_hit` 空白：`moe_expert_slice` 的 K/L 为 `未知*`，层组 vector/op 小计与
  整网 vector/op 总计保持 `未知*`；CUBE 小计/总计仍为已知数值。
- 填 `N_hit=6` 后重算：所有 `moe_expert_slice` 行变为数值，vector/op 小计与整网
  总计可算；`unknown_slice_rows=0`。

### 4. 条件 0、非法输入、合法 0

- Decode（seqlen=1）`hca_block_bias` = 0；Prefill 把 kvcachelen 改为 4096 后
  `hca_block_bias` = 4 ms（`8×32÷64`），条件分支方向正确。
- Prefill seqlen 改为文本 `abc` 后，`q_a_proj` cube 与 op 均为 `未知*`，不是 0；
  层组与整网汇总同步保持 `未知*`。
- seqlen=0、N_hit=0 时，`q_a_proj` cube/op = 0（合法 0 不被误判为未知）。

### 5. 旧导出与保护

- `export_structures_workbook` 两次重建字节相同；sha256
  `2dabd3af336e08b59cc847b7cc8d5c89fc8bc46244abf3a685a0077cc365b05e`，与既有候选
  `tables/decoder_variants.xlsx` **完全一致**（旧路径未改变）。
- 新布局重复构建字节稳定：`7abe374...a48e`；ZIP 时间戳固定 1980。
- 候选数据、候选表、草稿哈希记录在 `E01-verify/protected-paths.sha256`。

### 6. 测试

在快照中运行：

- 新布局 `test_deepseek_perf_layout.py`：11 tests OK。
- `test_perf_engine_models.py`：12 tests OK。
- `hf/test_operator_workbook.py`：57 tests OK。
- `test_operator_workbook_v2.py`：35 tests OK（14 个依赖本机 HF 输入的集成用例按
  设计跳过，未伪造）。

日志在 `E01-verify/test-logs/`。

## 渲染检查

用 LibreOffice Kit 重算并渲染以下区域，逐张目检：

- `E01-verify/render/prefill_top.png`：顶部标题、场景/吞吐区、集中参数、主表表头与
  SWA 组前段。
- `E01-verify/render/prefill_mid.png`：复合 `i_scorer`、repeat、`moe_expert_slice`
  未知项、excluded 行与单层/层组小计。
- `E01-verify/render/decode_top.png`：Decode 场景输入、参数、入口与 SWA 组前段。
- `E01-verify/render/detail_params.png` / `detail_ops.png`：辅助明细参数与逐算子区。

检查结果：12 列表头、n/T/D/M/K/N 数字、cube/vector/op 单位、`未知*`、灰色 excluded、
分组横幅、冻结列 A/B、紧凑行高与细边框均可读；长说明在批注/辅助明细，不挤主表。
LibreOffice 报缺 `Calibri`/`宋体` 并回退；**原生 Microsoft Excel 未安装，未做原生
联动/打印/字体验收**，需 E02 或用户在目标环境复核。

## Patch 与集成

- patch 只新增上述 3 个文件；`git apply --check` 在当前工作区通过。
- 在 `/tmp/e01_patch_test`（当前工作区 `structure_reader` src/scripts/tests 副本）应用
  patch 后重建，xlsx sha256 与交付件一致 `7abe374...a48e`；新测试 11 tests OK。
  记录见 `E01-verify/patch-integration.txt`。
- 未 `git apply` 到原工作区、未 commit/push、未改 CLI/构建入口。

## 并行 HTML 外部变化归因

快照建立后，原工作区下列文件被并行 HTML 会话修改（与本 Excel patch 无关，未纳入、
未回滚、未在实施中消费）：

- `structure_reader/src/structure_reader/build_hf_decoder.py`
- `structure_reader/src/structure_reader/extract/deepseek_v4_semantic.py`
- `structure_reader/src/structure_reader/render/archify_layers.py`
- `structure_reader/src/structure_reader/render/archify_preview.py`
- `structure_reader/src/structure_reader/render/hf_reading_page.py`

快照/当前哈希对照见 `workspace/MANIFEST.json` 与 `E01-verify/protected-paths.sha256`。
这些变化不影响本新模块的导入链（只依赖 `hf_workbook`/`deepseek_perf_rules`/
`operator_workbook`/`hf_scenario`/`scenario_evaluator`）。

## 验收对照（plan.md）

- [x] 参数来自真实配置，实际层型/层数有依据；草稿示意未混入事实。
- [x] 12 列主表，每行一个算子；依赖、分支、复合、专家路由计数有既有规则依据。
- [x] batch=1；head/分组矩阵乘与既有规则对照，head 与层数均只乘一次。
- [x] CUBE/VEC 吞吐独立；不等吞吐 528 项检查通过。
- [x] 线性投影、head 批量矩阵乘、Norm/Softmax、MoE 代表项、整网汇总；Prefill/Decode、
  条件 0、缺失/非法输入分别检查。
- [x] 参数固定引用；op 使用 MAX；未知与合法 0 分开可见。
- [x] 渲染两主表与辅助区；原生 Excel 未测已明确。
- [x] 旧导出字节稳定并与候选一致；历史候选/HTML/原始输入保护哈希记录。
- [x] 交付可打开 xlsx 与复现命令；未替换正式入口；工程测试通过不等于用户已认可布局。

## 结论与停点

E01 的独立样表、公式、测试、渲染和 patch 已交付，等待 Codex 复核与用户阅读。
不自行进入 E02/E03、不关闭 Issue、不 commit/push、不恢复自动化、不继续旧 HTML 会话。
