# E03 结果：新版 Excel 正式接入（GitHub #23）

状态：**主工作区已完成接入，停在待 Codex 技术复核**。不关票、不 commit/push、不恢复自动化、
不推广其他模型；E01 隔离 workspace、历史候选与已完成 HTML 内容只读保留。

日期：2026-11-03（Asia/Shanghai）。依据最新派单
[E03-dispatch.md](E03-dispatch.md)：HTML 已到停止节点，本轮直接改主工作区，不再只交隔离补丁；
未执行旧 E01 中误发并被用户停止的 D04 消息，未进入旧 HTML/T14 会话。

## 正式交付

| 交付 | 路径 | sha256 |
|---|---|---|
| 正式新候选 | `structure_reader/outputs/deepseek_v4_flash_0731_hf/ab454866942b38e5f01b55ad91fa792bda03cc5d4bd70e40a3174bf18da7ce82/` | — |
| 性能表（主交付） | 同候选 `tables/decoder_performance.xlsx` | `953592e594cdccfa7aed8914f5fed791cc10a0e972e6bc585ee4b5c3be751699` |
| 兼容旧表 | 同候选 `tables/decoder_variants.xlsx` | `2dabd3af336e08b59cc847b7cc8d5c89fc8bc46244abf3a685a0077cc365b05e` |
| 分层入口 | 同候选 `viewer/structure.html` | `5786b7320a1218aa7975c40a67eaf1173eede0361a8228b673dbf1e18f8295b4` |
| 候选 README | 同候选 `README.md` | `2796f4cea7e42b83d91364cf0e27b9af26b14374ff2d83a1af920b47b5b4f19f` |
| 结构数据 | `data/layer_structures.json` / `data/network_overview.json` | `f9c444115ecd…236c` / `2255b767cc12…7b6a` |
| 身份清单 | `reports/source_manifest.json` / `run_manifest.json` | `14d9fd3f689b…7177` / `c737d75c9f6e…dcd9` |

新表与 E01 已批准样表（`953592e5…1699`，见
[E01-layout-v2-rework-result.md](E01-layout-v2-rework-result.md)）**字节相同**；本次正式构建
真实读取的 structures/network 哈希（`f9c444…`/`2255b7…`）也与 E01 冻结输入逐字节一致，
因此不是“为相同哈希伪造旧事实”，而是同一份真实输入的同源结果。旧表字节同时保持
`2dabd3af…b05e`。

## 复现命令

```bash
cd /Users/xiechonghang/Workspace/LLMarchi_excel
python structure_reader/scripts/read_model_structure.py build-hf \
  --model deepseek_v4_flash_0731_hf --all-variants
```

CLI 会显式打印两张表的绝对路径（`表: tables/decoder_performance.xlsx -> …`、
`表: tables/decoder_variants.xlsx -> …`）。完整验证/渲染命令见
[E03-verify/reproduce.md](E03-verify/reproduce.md)。

## 接入方式（唯一源码差异）

只改 Excel 模块、必要 build/CLI、相关测试与入口文档（+94/−14 行，见
[E03-verify/E03-source-delta.patch](E03-verify/E03-source-delta.patch)）。`extract/`、
`render/`、模型事实/config/schema/场景求值、其他模型适配未改（见“保护核对”）。

1. **已批准 patch 原样应用**（不整树拷贝 E01 源码）：patch sha256
   `9f49f41f…203d6`，`git apply --check` 通过后 `git apply`，新增 4 个文件——
   `export/deepseek_perf_layout.py`（v1 公共事实/公式/ZIP 辅助）、
   `export/deepseek_perf_layout_v2.py`（layout-v2-rework 布局）、
   `scripts/build_deepseek_perf_layout_v2.py`（冻结输入入口，保留）、
   `tests/test_deepseek_perf_layout_v2.py`。4 个文件与 E01 workspace 同名文件哈希一致
   （`E03-verify/patch-apply.json`）。
2. [`build_hf_decoder.py`](../structure_reader/src/structure_reader/build_hf_decoder.py#L172-L175)
   `_Adapter` 增加可选 `export_performance`（默认 `None`，其他模型不产出新表）。
3. [`build_hf_decoder.py`](../structure_reader/src/structure_reader/build_hf_decoder.py#L313-L314)
   DeepSeek `tool_sources` 加入两个布局模块 → 进入 `analysis_id`。
4. [`build_hf_decoder.py`](../structure_reader/src/structure_reader/build_hf_decoder.py#L432-L442)
   DeepSeek 适配器用惰性导入包装正式导出器：只有真正生成性能表时才加载布局模块链
   （含 openpyxl），其他模型/路径不加载。
5. [`build_hf_decoder.py`](../structure_reader/src/structure_reader/build_hf_decoder.py#L1128-L1146)
   `_export_performance_workbook`：与旧表同一隔离口径，缺 `openpyxl` / 导出器拒绝输入均
   具名 `HfBuildError`；不降级 CSV、不只遗漏新表后仍发布成功候选。
6. [`build_hf_decoder.py`](../structure_reader/src/structure_reader/build_hf_decoder.py#L1775-L1796)
   `_run_variants` 用同一份 `structures` + `network` 生成表并写入
   `documents["tables/decoder_performance.xlsx"]`（仅 `--all-variants`；单层旧入口不产出）。
7. [`build_hf_decoder.py`](../structure_reader/src/structure_reader/build_hf_decoder.py#L1829-L1841)
   `run_manifest.tables` 明确记录 `performance`（layout/exporter/来源/场景）与
   `variants_legacy`（旧阅读接口）；交付清单由统一 `_finalize_documents` 自动收录新表哈希。
8. [`cli.py`](../structure_reader/src/structure_reader/cli.py#L1257-L1280)
   CLI 文本与 `--json` 摘要显式列出每张表路径（`tables` 字段）。
9. 候选 README（DeepSeek `readme_variants`）新增“性能表/兼容表”两条；
   [根 README](../README.md#L37-L56) 推荐入口改为 `--all-variants` 并说明单层旧表仍是旧阅读接口；
   [structure_reader/README.md](../structure_reader/README.md#L90-L101) 同步产物说明。
10. 新增 [`tests/test_hf_build_performance_table.py`](../structure_reader/tests/test_hf_build_performance_table.py)
    覆盖正式构建产物/清单/身份/缓存/缺依赖/其他模型隔离/archify 路径。

不改 `hf_workbook.py` / `operator_workbook.py` / `deepseek_perf_layout_v2.py` 的公式语义：
新布局复用 E01 已复核实现，未建第三套公式。

## 身份、清单与缓存

- `analysis_id = sha256(canonical(identity))`，identity 含 `tool_sources`（31 项，新增两个布局
  模块）。测试逐项复算 sha256 并确认 `tool_sources` 变化会产生新 `analysis_id`；性能表内容
  仍只由 structures/network 决定，布局源码改动不改表字节。
- `run_manifest.deliverables` 收录 `tables/decoder_performance.xlsx` 哈希；
  `run_manifest.tables` 记录布局与来源；`source_manifest.tool_sources` 记录两个模块哈希。
- 缓存完整性沿用 `cli._verify_existing`（按 documents 全量核对）：新表**丢失**报
  `problem: missing`、**损坏**报 `expected_sha256/actual_sha256`，均 `OutputConsistencyError`
  拒绝复用（测试覆盖，含未损坏时 `reused`）。
- 缺 `openpyxl`：新表路径抛具名 `HfBuildError`（含“性能表 … openpyxl …
  decoder_performance.xlsx”），无候选落盘；既有整链缺依赖回归仍通过。

## 验证

| 检查 | 结果 |
|---|---|
| 两次正式构建 | run1 `created`、run2 `reused`；analysis_id/全部文件哈希一致（[logs/build-run1.txt](E03-verify/logs/build-run1.txt)、[build-run2.txt](E03-verify/logs/build-run2.txt)） |
| E01 验证脚本对正式表 | `passed=true`、`recalc=true`、`errors=[]`；全部 check 与 E01 批准样表**逐项相同**（[verification.json](E03-verify/verification.json)） |
| 覆盖/模板/使用次数 | 291 实例 = 282 计费 + 8 excluded + 1 container → 101 模板；mHC 7×86、QKV 8×43；`cost_diffs` 全 0 |
| N_hit=6 总成本 | Prefill cube/vec/op = 25,975,584 / 103,636,557.5 / 129,612,141.5；Decode = 3,726,064 / 116,127,598.71875 / 119,853,662.71875（与批准样表一致） |
| 非法输入/吞吐 | `invalid_dimensions` / `invalid_throughput` failures 全空；合法 0 仍为 0 |
| 旧表字节 | `stable_bytes=true`，重建 = 候选 = `2dabd3af…b05e` |
| 单位/标题 | `units.failures=[]`、`block_titles.failures=[]` |
| 测试 | 15 组共 **528** 项全 OK（含新增 12 项 E03、layout-v2 9、旧导出 57、v2 35、CLI 54、其他四模型 220+） |
| HTML 绑定 A/B | `PASS`：归一化 analysis_id 后与 pre-E03 身份渲染**字节一致**（2,694,852 B），只多出新表；data/旧表字节一致（[html_binding_diff.json](E03-verify/html_binding_diff.json)） |
| 渲染 | 6 张；`prefill_top`、`prefill_main` 首页、`decode_top`、`mapping_usage`、`params` 与 E01 批准样表渲染**字节相同**；样式/真实使用次数保持（[render_manifest.json](E03-verify/render_manifest.json)） |

测试日志全部在 [E03-verify/logs/](E03-verify/logs/)。

## 保护核对（接手基线 → 当前）

- `baseline.json` 记录接手时 dirty 树、拟修改文件哈希、只读范围 91 个文件、157 个历史候选
  键文件、E01 隔离 workspace/证据树聚合哈希；既有删除（`CONTEXT.md`、
  `2026_10_17_workspace_structure_review.md`）保留。
- `protected_check.json`：91 个只读源码/输入**全部未变**；157 个历史候选的
  `run_manifest.json`/`README.md`/`decoder_variants.xlsx`/`decoder_reading.xlsx`/
  `structure.html` **全部未变**；候选数 157 → 158（只新增本票一个）。
- `source_delta.json`：把本票编辑反向还原后，`build_hf_decoder.py`、`cli.py`、
  `structure_reader/README.md`、根 `README.md` 的哈希**全部等于接手基线**——即上面的
  patch 就是 E03 的全部源码/文档改动，没有夹带其他会话的修改。
- `git status` 与基线相比只多出本票的 5 个新增文件 + 根 README；
  `hf_docs/`、`structure_reader/docs/` 等既有改动与删除原样保留。
- 已知测试副作用：`tests/test_cli_inspect.py`（既有用例）在失败分支回归时按产品行为写入
  3 条 `structure_reader/outputs/_failed/<run-id>/` 失败回执（本次 21:49，共 712 条）；
  不覆盖任何候选、不影响本票产物。

## 未做与限制

- 原生 Microsoft Excel 未安装：未验证原生公式缓存、冻结窗格、打印与字体；LibreOffice
  渲染报缺 `Calibri`/`宋体` 并回退，不能替代原生 Excel 验收。
- 未重跑昂贵 Archify 图层：`--archify-layers` 路径由测试桩确认用同一份 structures/network
  生成同源新表，且不触碰图布局/图身份；真实 Archify 图字节未重建验证。
- 未更新 `structure_reader/docs/model_delivery_index.md`（其“当前候选”是 T14 vNext 图入口，
  属 HTML/D 系列范围）；本票只在候选内绑定自己的 analysis_id，不推广其他模型。
- 新候选不含 Archify/ONNX（推荐命令即 `--all-variants`），因此与历史 `--archify-layers`
  候选的 `candidate id` 不同是预期的；HTML 差异只在 `analysis_id` 绑定。
- 未 commit/push、未关 Issue、未改 D01–D05 状态或协调 JSON、未恢复自动化。

## 停点

E03 主工作区接入与验证完成，停待 Codex 技术复核。复核通过前不再自行推广其他模型或
接下一票。
