# E01 返修结果（Codex 复核四项）

状态：**E01 返修完成，停在待 Codex 复核**。只返修 Codex 复核中的四项；未启动 E03，
未改 HTML/结构提取/CLI/共享 schema/求值器，未关票、未 commit/push、未恢复自动化。

旧交付、旧报告与根目录旧证据均保留：

- 旧 xlsx：`outputs/excel-performance-style-2026-11-03/E01/deliverables/deepseek_v4_flash_perf_layout.xlsx`
  （sha256 `7abe374d…a48e`）
- 旧报告：`docs/tasks/excel-performance-style-2026-11-03/E01-result.md`
- 旧根目录证据：`E01-verify/`（不删除、不覆盖）

返修新产物在 `outputs/excel-performance-style-2026-11-03/E01/rework/`；返修证据在
派单约定的 `docs/tasks/excel-performance-style-2026-11-03/E01-rework-verify/`。

## 新交付

| 交付 | 路径 | sha256 |
|---|---|---|
| 返修 xlsx（可打开） | `outputs/excel-performance-style-2026-11-03/E01/rework/deliverables/deepseek_v4_flash_perf_layout.rework.xlsx` | `6d376d94088a5f0b280526fb8e7a93be2c6cacbec218186c46aa9ffa7a745335` |
| LibreOffice 重算副本 | `.../rework/deliverables/deepseek_v4_flash_perf_layout.rework.recalc.xlsx` | `8ee448a8d7b22c036022fbcc9e129c127222ef919366dc9ec0edbadc70235a45` |
| 返修 patch | `.../rework/patch/E01-rework-source.patch` | `d0b616c8c60498f2ca73ee8f3f8547130ca8397236a57e5b2718e26a0a69662b` |
| 返修验证 JSON | `docs/tasks/.../E01-rework-verify/verification.json` | `ed4b71e4d21105d623da95a83918183b0aa14ed8d63a02692b590de689dd3767` |
| 返修清单 | `.../rework/MANIFEST.json` | `ef74e151118caf8244ab3686da31910832865172ec034e4f85ed85dfb2e05db3` |
| 复现命令 | `.../rework/reproduce.md` | — |

新 xlsx 仍为稳定字节；patch 只新增 3 个文件，`git apply --check` 通过，临时副本应用后
重建 sha256 与交付件一致。旧导出入口未改，重建 sha256 仍为
`2dabd3af336e08b59cc847b7cc8d5c89fc8bc46244abf3a685a0077cc365b05e`，与候选
`decoder_variants.xlsx` 完全一致。

## 四项修复

### 1. 吞吐守卫（P1）

- `_cost_formula` 现在把 `thru_guard` 并入实际 guard：
  `AND(维度/规则 guard, ISNUMBER(吞吐), NOT(ISLOGICAL(吞吐)), 吞吐>0)`；CUBE 与 VEC
  各自引用自己的吞吐单元格。
- 负数、0、空白、数字文本、普通文本、布尔均不能通过正数数值守卫；相关 op 的 J/K、
  `op(ms)`、单层小计与整网总计保持 `未知*`，不会出现负耗时被 `MAX` 掩成 0。
- 条件明确不执行（Decode `hca_block_bias`，seqlen=1）仍返回合法 0；即使此时吞吐非法，
  条件 0 不被改写为未知。
- 额外处理了 LibreOffice 的布尔输入问题：吞吐单元格不再套整数格式，避免布尔 TRUE 在
  重算时被转成数值 1；公式同时用 `ISLOGICAL` 显式拒绝。
- 回归：`invalid_throughput` 共 12 个 case（CUBE/VEC × 负数/0/空白/数字文本/文本/布尔），
  `failures=[]`；每个 case 在 Prefill/Decode 两场景检查 q_a_proj、q_a_norm、Decode 条件 0、
  单层小计与整网总计。

### 2. batch 固定 1（P2）

- 不再把可见的 `C12` 引用为 `B`；计算源直接使用常量 `1`。
- `C12` 改为文本 `固定 1`，注明“计算源直接使用常量 1，不引用本格；改本格不影响成本”。
- 公式扫描确认没有 `$C$12` 引用；`C12=2` 重算后所有 J/K/L 与 baseline 完全一致
  （`batch_fixed.mismatches=0`），CUBE/VEC 不会静默不一致。

### 3. 单位统一（P2）

- Prefill/Decode 的 `K7`/`K8` 都改为 `element/ms` 开头，不再出现 `MAC/ms` 或 `元素/ms`；
  默认值仍为 4096/64，乘积口径仍为 CUBE `M×K×N`、VEC `n×T×D`，不乘 2。
- 旧导出未改，历史 `decoder_variants.xlsx` 字节稳定。

### 4. 证据路径（P2）

- 返修证据全部写入 `docs/tasks/excel-performance-style-2026-11-03/E01-rework-verify/`：
  `verify_workbook.py`、`verification.json`、`README.md`、`test-logs/`、`render/`、
  `render_manifest.json`、`protected-paths.sha256`、`patch-integration.txt`、
  `deliverables.sha256`。
- 新报告、`reproduce.md`、`MANIFEST.json` 与 README 的命令全部使用上述路径及
  `outputs/.../E01/rework/` 新 xlsx/patch 路径。
- 根目录旧 `E01-verify/` 原样保留，作为旧证据，不删除、不冒充返修证据。

## 验证结果

`docs/tasks/.../E01-rework-verify/verification.json`：`passed=true`、`recalc=true`、
`errors=0`，绑定新 xlsx `6d376d…5335` 与 patch `d0b616…662b`。

- baseline Prefill/Decode：逐算子误差 0；`op=MAX`、纯 CUBE/VEC 的 0 项、汇总
  COUNT 完整性通过。
- 不等吞吐：CUBE 2048/VEC 128，528 个非零数值检查 `mismatches=0`。
- `N_hit=6`：`moe_expert_slice` 未知行 0，向量/op 小计与整网总计可算。
- 条件：Prefill 有历史时 `hca_block_bias=4`，Decode 为 0。
- 非法 seqlen：q_a_proj 为 `未知*`；合法 0（seqlen=0、N_hit=0）仍为 0。
- 非法吞吐：12/12 通过；batch 编辑：`mismatches=0`；单位：`failures=[]`。
- 旧导出：字节稳定且与候选 sha256 一致。

测试（`E01-rework-verify/test-logs/`）：

- 新布局：14 tests OK（新增单位、batch 显示、吞吐守卫回归）。
- 性能引擎：12 tests OK。
- 旧导出：57 tests OK。
- v2：35 tests OK（14 个依赖本机 HF 输入的集成用例按设计跳过）。

## 渲染

新 xlsx 用 LibreOffice Kit 重算后渲染并目检：

- `E01-rework-verify/render/prefill_top.png`
- `E01-rework-verify/render/prefill_mid.png`
- `E01-rework-verify/render/decode_top.png`
- `E01-rework-verify/render/detail_params.png`
- `E01-rework-verify/render/detail_ops.png`

检查点：K7/K8 均为 `element/ms`；batch 行显示 `固定 1`；12 列表头、n/T/D/M/K/N、
`未知*`、excluded/container、分组横幅、冻结列 A/B、紧凑行高与细边框均可读。LibreOffice
报缺 `Calibri`/`宋体` 并回退；**原生 Microsoft Excel 未安装，未做原生联动/打印/字体
验收**，需 Codex 或用户目标环境复核。

## 保护与边界

- 原 `structure_reader/src`、`tests`、HTML、extract/render、CLI/构建编排、共享
  schema/求值器、旧 Excel、候选、D01—D05 协调文件未由本票修改。
- 冻结输入仍为同候选：layer_structures
  `f9c444115ecdd4333847df77d4aa440b5090f3ce5f64eb68fd6a30ce9e2f236c`，network_overview
  `2255b767cc124ad33500ac0e08ecedddf51fc486c3fc17ae4f2dfd29485f7b6a`。
- 并行 HTML 会话在快照后修改的 5 个 build/extract/render 文件已单独归因，未纳入 patch、
  未回滚；哈希见 `E01-rework-verify/protected-paths.sha256` 与快照 `workspace/MANIFEST.json`。
- 未 `git apply` 到原工作区、未 commit/push、未关闭 Issue、未恢复自动化、未进入旧 HTML
  会话、未启动 E02/E03。

## 停点

E01 返修完成，等待 Codex 复核与用户阅读。不自行接下一票。
