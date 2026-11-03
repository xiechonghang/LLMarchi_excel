# T01 结果：统一既有 HF 构建，保留单层与四变体交付

状态：**完成（P2 产品修复已通过 Codex 独立复验；末次仅补齐回归测试的输入就绪保护），等待最终复核**。仅执行 T01；未 commit/push、未关闭 GitHub #2、未进入 T02。

## 执行基线

- 工作区：`/Users/xiechonghang/Workspace/LLMarchi_excel`；HEAD `947ae78f120279403ec0a4c539e99d3d8869e1d6`（与交接一致）。
- 接手快照：`/var/folders/.../llmarchi-T01-baseline-20261028-7qzfihg4/files`（含 283 个文件）。
- 接手时 `git status` 54 项（含两项删除、未提交修改与未跟踪文件）；本轮未 reset/clean、未恢复删除、未覆盖历史候选，未改 `docs/` 下 Codex 派遣材料。
- 相对接手快照，本轮实际只改 3 个文件：
  - `structure_reader/src/structure_reader/build_hf_decoder.py`（统一构建 + 复核 P2 修复）
  - `structure_reader/tests/test_hf_decoder_reading.py`（仅两处测试夹具）
  - `structure_reader/tests/test_hf_build_missing_xlsx.py`（新增缺 openpyxl 回归）
- 新增证据：`docs/tasks/model-structure-2026-10-28/T01-verify/`（两个一次性验证脚本，非产品测试）；`T01-review/missing-xlsx-reproduction.json` 为 Codex 复核证据，未改动。
- 运行测试生成的 `__pycache__` 已清理（`structure_reader/{src,tests,scripts}`）；`outputs/` 为既有忽略目录。

## 改动清单

### 复用（未修改，继续由两条路径共同消费）

| 模块 | 复用内容 |
| --- | --- |
| `rules/deepseek_v4_hf.py` | `SOURCE_FILES`、`verify_source_fingerprints`、`load_hf_model_values`、`effective_parameter_records` |
| `extract/deepseek_v4_hf.py` | `build_reading_document`（单层；四变体经容器复用） |
| `checks/deepseek_v4_hf_contracts.py` | `validate_hf_reading_document`（逐文档/逐变体） |
| `render/decoder_page.py` | `render_decoder_page`、`DecoderPageError` |
| `render/m2_variants.py` | `build_variants_container`（四类层归属、代表层、43 层覆盖检查）、`render_variants_page`、`export_variants_workbook` |
| `export/operator_workbook.py` | `export_operator_workbook` |
| `onnx_module/build.py` | `build_all`（仅 `--onnx` 时惰性导入） |
| `cli.py` | `build-hf` 入口、`_materialize`/`_verify_existing`/`_fail_exit`（未改动） |

### 原位重构（`build_hf_decoder.py`）

新增一组共用辅助函数，`run_hf_build`（单层）与 `_run_variants`（四变体）都只调用它们：

| 共用函数 | 替代的重复步骤 |
| --- | --- |
| `_require_supported_model` | 单层原有模型键检查；四变体原无此检查（现两条路径一致具名拒绝） |
| `_verified_input_shas` | 两条路径各自的来源指纹复算 + `SOURCE_FILES` 输入 SHA 循环 |
| `_read_source_snapshot` | 两条路径各自的源码快照循环 |
| `_tool_source_shas` | 两条路径各自的工具源码 SHA（统一为缺失即拒绝，不再有 `MISSING` 占位） |
| `_validate_document` | 两条路径各自的 `validate_hf_reading_document` 调用与错误拼装 |
| `_load_onnx_stack` | 两条路径各自的 ONNX/onnxsim 依赖导入与版本身份 |
| `_build_onnx_document` | 两条路径各自的模块失败/数量/checker/shape/simplify 校验 |
| `_render_html` | HTML 渲染错误统一转为 `HfBuildError`（四变体原先直接抛出） |
| `_export_workbook` | **导出器 import 与调用**、缺依赖/拒绝输入统一转为 `HfBuildError` |
| `_source_manifest_document` | 两份几乎相同的 `reports/source_manifest.json` 装配 |
| `_finalize_documents` | 两份 `doc_hashes` + `run_manifest.json` 装配 |

由此删除的重复实现（同一文件内的内联代码块，非独立模块）：

- `_run_variants` 内的来源核验、输入 SHA 推导、source snapshot、工具身份、ONNX 导入/构建/校验、manifest/哈希装配共 6 段重复逻辑；
- `run_hf_build` 内对应的同类内联逻辑改为调用共用函数。

行为收敛（均为**加强**，未降低既有检查）：

- 单层工具源缺失从写入 `"MISSING"` 改为具名 `HfBuildError`；四变体原有的严格身份不变。
- 单层 `--onnx` 现在同样执行 checker/shape/onnxsim 逐模块校验（原仅检查 failed）；四变体的逐组模块数（16/18/17/18）与强校验保持。
- 未知模型键在 `--all-variants` 路径上也于构建前拒绝。
- 四变体 `reports/source_manifest.json` 增补 `source_revision` 字段（与单层共用装配；值与该候选 `data/reading_variants.json` 的 `model.source_revision` 一致，均为 `536ecc007387a50e77603bb5d92100e9b07514cc`）。这是唯一新增的产物字段。

### 复核 P2 修复：导出器 import 纳入统一错误隔离

- **回归**：`export/operator_workbook.py` 在模块顶层导入 openpyxl。重构后 `from structure_reader.export.operator_workbook import export_operator_workbook` 位于 `_export_workbook` 的 `try` 之外，缺 openpyxl 时单层路径直接抛 `ModuleNotFoundError`，绕过具名失败；四变体因导出器在包装内调用而仍抛 `HfBuildError`。Codex 证据：`T01-review/missing-xlsx-reproduction.json`（before：`HfBuildError`；after：`ModuleNotFoundError`）。
- **修复**：`_export_workbook(source, *, variants: bool = False)` 把导出器 import 与实际调用放进同一个 `try`；单层/四变体仅以 `variants` 选择既有导出函数，未新增平行导出器，未改公共构建接口（`run_hf_build` 参数与返回不变，`build-hf` CLI 不变）。
- **本机复现（修复前）**：单层 `ModuleNotFoundError: import of openpyxl halted; None in sys.modules`；四变体 `HfBuildError`（与 Codex 证据一致）。
- **本机复现（修复后）**：两条路径均为 `HfBuildError: xlsx 导出必需依赖缺失（openpyxl）: ...；不降级为 CSV，不发布假完整候选`。
- **CLI 级验证**：`T01-verify/missing_xlsx_named_failure.py` 在临时工程根上阻断 openpyxl 并调用真实 `build-hf`，单层与四变体 **10/10 PASS**：rc=1、`error.kind=hf_build_error`、消息含“xlsx 导出必需依赖缺失/openpyxl”、`outputs/<model_key>/` 无候选目录、失败诊断记录保留。
- **保留回归**：新增 `tests/test_hf_build_missing_xlsx.py`（2 tests），对两条既有构建入口断言缺 openpyxl 时为 `HfBuildError` 且不落盘；不依赖 mock 内部函数。
- **测试就绪保护（末次补齐）**：该回归按既有 HF 测试约定检查只读输入（`models/.../config.json`、`hflib/.../modeling_deepseek_v4.py`）及本测试所需的工程源码/配置；缺失时在 `setUp` 明确 `skipTest`，不再因缺分析输入抛出 `HfConfigError` 而误报为缺 openpyxl 分支。模拟空临时 `REPO_ROOT`（运行期改常量、改写常量后导入两种方式）均为 **2 skipped / 0 errors**；有输入时 2/2 通过。本次只改该测试与报告，未改产品代码。

### 删除 / 退役

- 无文件删除、无模块退役；未新增第三套构建器或第二套导出器。
- 仅移除上述 `_run_variants`/`run_hf_build` 内被共用函数替代的重复代码块。
- 旧 `build`/reference/DSpark 路径未触碰。

### 测试夹具迁移（`tests/test_hf_decoder_reading.py`）

`test_full_build_atomic_output` 与 `test_analysis_id_stable_and_sensitive` 原先只给临时工程根放一份 `configs/models/*.json`，依赖单层工具身份允许 `MISSING`。统一为严格工具身份后，假工程根改为镜像 `src` 与 `configs` 符号链接（与既有独立门禁 `tests/hf_independent/gate_round01` 的 `_temp_project` 做法一致），仍不写真实 `outputs/`。断言未改。

## 行为保持证据（重构前字节 vs 重构后字节）

重构前用同一工作树代码对真实输入构建，保存全部字节到 `/tmp/t01-before/`；P2 修复后用同一命令保存到 `/tmp/t01-after2/` 并逐文件比较（比较脚本为一次性检查，未入库）：

- 单层 7 个交付文件：`data/reading_document.json`、`data/effective_config.json`、`viewer/decoder.html`、`tables/decoder_reading.xlsx` **逐字节相同**；`README.md` 仅 `analysis_id` 不同；`reports/source_manifest.json`、`run_manifest.json` 仅 `analysis_id` 与其派生 hash 不同。
- 四变体 6 个交付文件：`data/reading_variants.json`、`viewer/decoder_variants.html`、`tables/decoder_variants.xlsx` **逐字节相同**；`README.md` 仅 `analysis_id` 不同；两个 manifest 仅身份字段与派生 hash 不同。
- `identity.tool_sources` 的唯一差异是 `src/structure_reader/build_hf_decoder.py` 自身 SHA（本次改动导致）；7 个输入 SHA、模型键、层集合、ONNX 关闭状态、变体归属、manifest 的 `schema/status` 均不变。
- 新候选的 `run_manifest.json.deliverables` 与目录内实际文件 SHA 全部一致。
- P2 修复未改变任何正向产物字节：修复后候选的 HTML/Excel/reading JSON 与首轮候选（`55219...`/`52df...`）逐字节相同。
- 四类层归属与代表层保持：`swa-hash` 2 层（rep 0）、`csa-hash` 1 层（rep 2）、`hca-topk` 20 层（rep 3）、`csa-topk` 20 层（rep 4），合计 43 层；容器覆盖检查未改。
- 单层与四变体读取的模型语义/算子/张量/缓存事实未改（`reading_document.json` 与 `reading_variants.json` 字节相同即为直接证据）。

## 构建命令与产物入口（P2 修复后当前候选）

解释器（工程实测 3.12）：

```bash
PY=/Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
cd /Users/xiechonghang/Workspace/LLMarchi_excel
```

单层样板（layer 4）：

```bash
"$PY" structure_reader/scripts/read_model_structure.py build-hf \
  --model deepseek_v4_flash_0731_hf --json
```

- analysis_id：`e859305c6d8338137bd838abf42857076d0716efcdf0ceffb0e9608a7d3de8a7`
- HTML：[decoder.html](../../../structure_reader/outputs/deepseek_v4_flash_0731_hf/e859305c6d8338137bd838abf42857076d0716efcdf0ceffb0e9608a7d3de8a7/viewer/decoder.html)
- Excel：[decoder_reading.xlsx](../../../structure_reader/outputs/deepseek_v4_flash_0731_hf/e859305c6d8338137bd838abf42857076d0716efcdf0ceffb0e9608a7d3de8a7/tables/decoder_reading.xlsx)
- 7 个交付文件，1.7 MB；HTML 1,311,290 B，Excel 63,757 B。

四变体（SWA+Hash / CSA+Hash / HCA+TopK / CSA+TopK）：

```bash
"$PY" structure_reader/scripts/read_model_structure.py build-hf \
  --model deepseek_v4_flash_0731_hf --all-variants --json
```

- analysis_id：`9313cf765a69684607c9899388a1100f5cd6097d6eeaf2b641bb57f251587803`
- HTML：[decoder_variants.html](../../../structure_reader/outputs/deepseek_v4_flash_0731_hf/9313cf765a69684607c9899388a1100f5cd6097d6eeaf2b641bb57f251587803/viewer/decoder_variants.html)
- Excel：[decoder_variants.xlsx](../../../structure_reader/outputs/deepseek_v4_flash_0731_hf/9313cf765a69684607c9899388a1100f5cd6097d6eeaf2b641bb57f251587803/tables/decoder_variants.xlsx)
- 6 个交付文件，6.2 MB；HTML 4,854,466 B，Excel 187,684 B。

可选 ONNX 仍为同一入口 `--onnx`；本机缺依赖时两条路径均 rc=1、不发布候选（见下）。

重复执行两条命令均为同一 analysis_id 且 `reused: true`。旧候选目录未变（含首轮本轮候选 `55219...`/`52df...` 与更早的 `cfa518...`）；候选目录数由 74 增至 78（4 个为本轮首轮+P2 修复的新身份）。

## 实际测试结果

| 命令 | 结果 |
| --- | --- |
| `PYTHONPATH=src "$PY" -m unittest tests.test_hf_decoder_reading tests.test_m2_toc tests.test_hf_build_missing_xlsx` | **66 tests，1 failure**（`test_wrong_layer_rejected`，接手前既存）；单层构建链两个用例、目录/导航 3 项、缺 openpyxl 回归 2 项均通过 |
| `PYTHONPATH=src "$PY" -m unittest tests.test_m2_guard` | 2 errors：`mock.patch("structure_reader.onnx_module.build.build_all")` 需导入 onnx，而本机未安装；**接手前基线同样 2 errors**，非本次引入 |
| `PYTHONPATH=src "$PY" ../docs/tasks/model-structure-2026-10-28/T01-verify/onnx_guard_stub_check.py` | **19/19 PASS**（桩替换 ONNX 工具链，验证两条路径的 checker/shape/onnxsim/模块失败/模块数/未知键阻断及分组产物前缀/身份） |
| `PYTHONPATH=src "$PY" ../docs/tasks/model-structure-2026-10-28/T01-verify/missing_xlsx_named_failure.py` | **10/10 PASS**（CLI 级：单层与四变体缺 openpyxl 均 rc=1、`hf_build_error`、消息具名、未发布候选、失败记录保留） |
| 空临时 `REPO_ROOT` 模拟（缺 models/hflib/工程源码；运行期改常量与改写常量后导入两种方式） | **2 skipped，0 errors**（就绪保护生效，不抛 `HfConfigError`） |
| 重复执行两条 `build-hf` 命令 | 同 analysis_id，第二次 `reused: true` |
| `--onnx` 两条路径（缺依赖） | 均 rc=1，具名 `ONNX 依赖不可用 ... No module named 'onnx'`，未新增候选目录、未产生 HTML |

`test_wrong_layer_rejected` 在接手前已失败：它断言 layer 0/2 仍被拒绝，而工作树中四层型已支持；属 T01 之前的既存断言漂移，本票未修改（不在统一构建范围）。

产物静态核验（本次实际读取 P2 修复后候选）：两 HTML 内部 `href="#..."` 目标缺失均为 0、id 无重复（单层 1082 个 id；四变体 3922 个 id）；两个 xlsx 均为 `Prefill`/`Decode` 两表且 openpyxl 可读；`run_manifest.deliverables` 与文件 SHA 全部一致。

## 未验证项

- **ONNX/onnxsim 实际构建未在本机端到端运行**：本机解释器无 onnx/onnxsim/onnxruntime（接手前亦然），未下载/安装依赖。逐组阻断与身份仅由桩脚本与既有 `test_m2_guard`（待有依赖环境执行）覆盖；本机未生成真实 ONNX 产物。真实 ONNX 环境补验由 Codex 负责；Codex 已在隔离环境独立复验单层 72 / 四变体 276 个原简模型字节与派单前基线相同、逐模块检查通过（此处只记录复核结论，不代签）。
- 浏览器实际渲染、键盘/窄屏导航、用户可读性未签；Excel 原生 GUI 未打开；Netron GUI 未启动。
- 未运行 HF 模型、未加载/下载权重、未声明数值等价。
- 未运行依赖历史候选的旧全仓门禁；额外跑过的 `tests.test_hf_decoder_diagram` 存在 6 个既存失败（该模块不导入 `build_hf_decoder`，与本票改动无调用关系），未在本票修复，也不计入 T01 结果。

## 结论

T01 的重复构建流程已收敛为一条共用流水线：来源核验、输入/工具身份、源码快照、语义检查、ONNX 逐模块校验、HTML/xlsx 错误隔离（含导出器 import）、source/run manifest 装配均由单层与四变体共用；入口、四类层归属、代表层、单层与四变体的 HTML/Excel 交付能力保持，检查强度只增不减。复核发现的 P2（导出器 import 绕过具名失败）已在本票内修复，并由保留回归测试与 CLI 级检查覆盖；该回归在缺只读输入时明确跳过，不再误报。无阻塞项，等待最终复核。
