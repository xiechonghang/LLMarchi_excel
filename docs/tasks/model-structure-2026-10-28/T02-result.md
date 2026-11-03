# T02 结果：DeepSeek 整网总览连接真实层组与已有 Block 详图

状态：**首轮复核两项 P2 已在本票内修复，等待再次复核**。本票只做 T02：在既有 HF 构建与四类层型/代表层之上补齐 token→logits 的整网外围、真实层序与四组归属，并提供总览↔既有 Block/算子详图的离线往返入口。未 commit、未 push、未关闭 GitHub #3、未进入 T03、未改自动化/协调状态文件。

## 首轮复核修复（P2）

### Spec：`logits_to_keep` 越界长度（`slice(-Lk,None)`）

- 问题：原文档把正整数 `Lk` 写成 `S_sel = Lk`。官方源码 `modeling_deepseek_v4.py:1510-1511` 为 `slice(-logits_to_keep, None)`，负起点越界会被裁剪到 0，实际长度为 `min(T, Lk)`（如 T=1、Lk=5 仍只输出 1 个位置）。
- 修复：`extract/deepseek_v4_network.py` 新增可测试的 `logits_slice_length(T, Lk)`（`Lk=0 → T`；`Lk>0 → min(T, Lk)`；负整数明确拒绝），并据此改写 `Lk`/`Ssel` 参数式与注记、`lm_input`/`net_logits_slice`/`logits`/`net_lm_head` 说明，页内示例改为 `T=1、Lk=5 → S_sel=1，不是 5`。覆盖注记明确：只分析非负 int 路径；**负整数与 tensor 型 `logits_to_keep` 未分析**，不声明其输出长度。
- 门禁：`validate_hf_network_overview` 新增检查——`net_logits_slice` 注记与 `Ssel` 参数式/注记必须表达 `min(T, Lk)` 越界裁剪；回退为 `S_sel = Lk` 时构建具名拒绝。
- 回归：新增 `test_logits_slice_length_clips_at_sequence_length`（1/5→1、5/3→3、5/5→5、5/9→5、7/0→7、负数/布尔拒绝）、`test_document_expresses_clipped_logits_length`、`test_unclipped_logits_length_claim_rejected`（门禁拒绝旧写法）。

### Standards：共享详情渲染，不再复制第二份长模板

- 问题：`render/network_page.py` 的参数行、张量行、算子卡/源码面板复制了 `decoder_page._render_phase` 的通用详情模板；且两版已经漂移（旧卡合并所涉张量的参数引用，新卡只列算子显式参数）。
- 修复：新增 `render/detail_render.py`，统一提供 `render_parameter_row` / `render_tensor_row` / `render_operator_card`（含输入/输出/权重、**所涉张量参数引用合并**、源码快照面板、`toggleSource`、`locateInDiagram` 图内定位）/ `render_notes_list` / `source_snapshot_lines` / `value_repr`。`decoder_page._render_phase` 与 `network_page` 都调用同一实现，各自只保留页面外壳、章节编排与结构事实。整网页通过 `extra_links` 钩子保留层循环卡片的“层顺序与四组”专属入口；详情页不传该钩子，输出不变。
- 身份：`detail_render.py` 已纳入 `_TOOL_SOURCE_RELPATHS`（工具身份 21→22 个文件），缺失即具名拒绝。
- 回归：单层 `decoder.html`/Excel/JSON 与 T01 候选 `e859305c…` 仍**逐字节相同**；四变体 `decoder_variants.html`/`reading_variants.json`/`decoder_variants.xlsx` 与首轮交付 `bc3cd4fd…` 仍**逐字节相同**（共享重构未改既有页面）；整网页因合并参数引用/新增说明而变化，双向锚点与源码面板由测试与核对脚本覆盖。

## 执行基线

- 工作区：`/Users/xiechonghang/Workspace/LLMarchi_excel`；HEAD `947ae78f120279403ec0a4c539e99d3d8869e1d6`（main，与派单一致）。
- 接手快照：`/var/folders/cf/v8s2g_0n0_5c9x9brbcy1nw00000gn/T/llmarchi-T02-baseline-20261028-qned8nmo/manifest.json`（311 个文件；接手 `git status` 55 项，含两项根目录删除、未提交修改与未跟踪候选）。
- 相对快照，本票实际改动 **12 个文件**：8 个修改、4 个新增。其中 `build_hf_decoder.py`、`checks/deepseek_v4_hf_contracts.py`、`cli.py`、`render/diagram_source_labels.py`、`render/m2_variants.py`、`render/display_names.py`、`render/decoder_page.py` 在接手前已有未提交修改（本票只做增量）；`rules/deepseek_v4_hf.py` 在快照中是干净文件。
- 未 reset/clean、未恢复两项删除、未覆盖历史候选、未批量格式化；`docs/` 下 Codex 派单/复核材料未被改写。
- 接手前 `structure_reader/outputs/deepseek_v4_flash_0731_hf/` 下 78 个历史候选全部保留；本票最终新增 2 个候选：首轮交付 `bc3cd4fd…`（按复核要求保留，供证据使用）与修复后最终候选（下方 analysis_id），目录总数 78→80。修复过程中还曾产生三个中间候选（`ea412c7e…`、`1f44e7e7…`、`18a61432…`），因参数命名/死代码清理等修正被取代，已删除并核对为目标路径；不涉及任何接手前候选或用户文件。
- 测试生成的 `structure_reader/{src,tests,scripts}/__pycache__` 已清理（`structure_reader/.work/tools/` 下既有 pycache 未触碰）。

## 改动清单

### 新增（承担的新职责）

| 文件 | 职责 |
| --- | --- |
| `extract/deepseek_v4_network.py` | 整网总览事实提取：embedding、默认 position_ids、模型级滑动窗口 mask、main/compress 两组 RoPE、mHC 流扩展、真实层循环（当前配置 43 层；真实层序与四组）、HyperHead 收拢、Final RMSNorm、logits 位置切片（`min(T,Lk)` 越界裁剪）、lm_head→logits、`past_key_values` 读/写；输出 `hf_network_overview.v1`。含可测试的 `logits_slice_length`。只静态读源码/配置，不导入/实例化/运行模型。 |
| `render/detail_render.py` | 共享详情渲染：参数行、张量行、算子卡（输入/输出/权重、所涉张量参数引用合并、源码快照面板、`toggleSource`、`locateInDiagram`）、注记列表。单层/四变体详情页与整网总览页共用，避免第二份长模板。 |
| `render/network_page.py` | 整网总览单文件离线 HTML：复用 `diagram.py` 的真实张量引用连接与 SVG、共享详情渲染；保留整网专属外壳（主路径图、43 层序带、四组表、来源身份），生成到既有四变体详图的真实锚点链接。 |
| `tests/test_hf_network_overview.py` | 21 项行为测试：身份与 schema、四组恰好覆盖与真实层序、外围/终点源码锚点、边界 shape 与数据流闭合、mHC/循环连接、`logits_to_keep` 越界裁剪与门禁回归、契约检查的具名拒绝（缺 SHA/缺终点算子/破坏 mHC shape/覆盖缺口/缺 mask 依赖/缺详情锚点/未裁剪长度）、缺失来源拒绝、总览↔详图往返链接与离线约束。 |

### 原位重构（复用既有实现，未另造平行产品）

| 文件 | 改动 |
| --- | --- |
| `rules/deepseek_v4_hf.py` | 新增 `LAYER_GROUP_SPECS`（四组 id/标签/层型组合/代表层）与 `layer_group_assignments()`：按 `layer_types`/`mlp_layer_types` 逐层推导归属，并校验“四组恰好覆盖全部主干层、无空组、代表层在组内、无未归属组合”。它是四变体与整网总览共用的**唯一**层组事实来源。 |
| `render/m2_variants.py` | 四组与层归属统一由 `rules.layer_group_assignments` 提供；删除被替代的 `VARIANTS` 元组与 `_layer_groups`。`render_variants_page` 新增可选 `overview_href`：提供时在页首与每组末尾生成“返回整网总览”链接；不提供时输出与既有页面一致。 |
| `render/decoder_page.py` | `_render_phase` 的参数行、张量行、算子卡/源码面板改为调用 `detail_render` 共享实现；删除重复的 `_KIND_LABELS`/`_source_snapshot_lines`/`_value_repr`/`_notes_list`（已迁入共享模块），既有页面输出逐字节不变。 |
| `checks/deepseek_v4_hf_contracts.py` | 抽出 `_check_unit`/`_check_flow_units`，供阅读文档与整网总览共用同一套骨架/引用闭合/数据流核查（阅读文档行为不变）；新增 `validate_hf_network_overview`：schema/身份、单路径、四组覆盖与层序、外围/终点源码锚点、mHC 非普通残差、循环真实依赖、embed/lm_head 权重独立、详情锚点齐备、输入 SHA 完整、`min(T,Lk)` 越界裁剪。 |
| `build_hf_decoder.py` | 复用 T01 统一流水线：`_run_variants` 在同一候选内构建/校验/渲染/发布 `data/network_overview.json` 与 `viewer/network.html`，并给四变体页传入 `overview_href="network.html"`；`_render_html` 统一捕获 `NetworkPageError` 为具名 `HfBuildError`；`_TOOL_SOURCE_RELPATHS` 纳入 `deepseek_v4_network.py`、`network_page.py`、`detail_render.py`（缺失仍具名拒绝）；`run_manifest` 增补 `network_overview` 摘要；README 与入口说明更新。单层路径与 CLI 契约未变。 |
| `cli.py` | `--all-variants` 帮助、阶段摘要与入口打印更新为“四变体 + 整网总览”，并打印 `viewer/network.html`；命令/参数/退出码不变。 |
| `render/diagram_source_labels.py` | 为整网页新增算子源码名/行号（`net_*`）、网络级张量源码变量（`inputs_embeds`、`causal_mask`、`hidden_streams`、`hc_head_*`、`logits` 等）与计算类型词（Embedding/Unsqueeze/Contiguous/Loop）；既有层内映射未改。 |
| `render/display_names.py` | 新增整网参数显示名 `Nlay`（num_hidden_layers）、`Ssel`（logits_sequence_length）、`Lk`（logits_to_keep）。为避免与既有文档正文中单独出现的 `S`/`N` 冲突而使用专用 id；已用 T01 单层候选做逐字节回归确认——首版直接用 `S` 曾导致既有页面把正文 "S" 展开为 `logits_sequence_length`，已在本票内修正。 |

### 删除 / 退役

- 删除 `render/m2_variants.py` 的 `_layer_groups` 与 `VARIANTS`：职责迁到 `rules/deepseek_v4_hf.LAYER_GROUP_SPECS` / `layer_group_assignments`，模板分组与覆盖校验不再有两份实现。
- 删除 `render/decoder_page.py` 中被共享模块替代的重复详情渲染辅助（`_KIND_LABELS`、`_source_snapshot_lines`、`_value_repr`、`_notes_list` 及内联算子卡/参数行/张量行），由 `render/detail_render.py` 承接。
- 删除本票构建产生三个中间候选目录（见“执行基线”），不涉及任何接手前候选、首轮交付候选或用户文件。
- 无其他文件删除；旧 `build`/reference/DSpark 路径未触碰。

### 复用（未修改）

| 模块 | 复用内容 |
| --- | --- |
| `extract/deepseek_v4_hf.py`、`extract/m2_hca.py`、`extract/m2_swa_hash.py` | 四类真实 Block/算子事实（层内结构不重复实现） |
| `render/diagram.py` | `build_module_graph` 的真实张量引用连接、`render_svg`、`diagram_css`、`legend_html` |
| `render/decoder_page.py` | `_PAGE_HEAD` 样式与页面骨架；详情单元现由 `detail_render` 提供 |
| `rules/deepseek_v4_hf.py` | `SOURCE_FILES`、`verify_source_fingerprints`（来源漂移/缺失具名拒绝）、`load_hf_model_values`、指纹身份 |
| `build_hf_decoder.py` 共用步骤 | `_verified_input_shas`、`_read_source_snapshot`、`_tool_source_shas`、`_validate_document`、`_render_html`、`_export_workbook`、`_finalize_documents`、`_materialize` 内容寻址落盘与失败隔离 |

## 来源身份与事实

- 模型键：`deepseek_v4_flash_0731_hf`；实现：本地官方 Transformers 检出，`DeepseekV4ForCausalLM` / `DeepseekV4Model` / `DeepseekV4DecoderLayer`，eager 路径。
- 源码版本：`536ecc007387a50e77603bb5d92100e9b07514cc`；根配置：`models/deepseek-ai/DeepSeek-V4-Flash-0731/config.json`。
- 7 个分析输入的实际 SHA256 已写入候选 `reports/source_manifest.json` 与总览页“来源身份核对”表；构建前由 `verify_source_fingerprints` 复算已分析指纹，缺失/漂移即具名失败（不下载、不沿用旧结论）。关键前缀：`config.json 6c8f3d2d…`、`modeling_deepseek_v4.py d6eb834f…`、`configuration_deepseek_v4.py 49c81840…`、`cache_utils.py 2bef552b…`、`masking_utils.py c1d0e1e6…`、`modeling_rope_utils.py 6a072707…`、`activations.py 5b20c0a3…`。
- 工具身份 22 个文件（首轮 21 个 + `render/detail_render.py`），缺失即拒绝，不填 `MISSING`。
- 静态事实依据（`modeling_deepseek_v4.py` 行号）：`embed_tokens` :1302-1303/:1332-1333；默认 `position_ids` :1334-1337；`create_sliding_window_causal_mask` :1344-1350；mHC 流扩展 :1351；main/compress RoPE :1352-1355；层循环 :1357-1366（ModuleList :1304-1306，循环体 :1161-1186）；`self.norm(self.hc_head(...))` :1368；HyperHead :999-1003/:993-997；`slice_indices` :1510（`slice(-Lk,None)`，长度 `min(T,Lk)`）；`lm_head` :1451/:1511；输出 :1528-1536。mHC 是真实残差机制，未被简化成普通残差模板；`tie_word_embeddings=false`，`embed_tokens` 与 `lm_head` 是两份独立权重。
- 四组真实归属：`swa-hash` 2 层（0,1）、`csa-hash` 1 层（2）、`hca-topk` 20 层（3,5,…,41）、`csa-topk` 20 层（4,6,…,42），合计 43；真实顺序交错（0 SWA、1 SWA、2 CSA+Hash、3 HCA、4 CSA+TopK、5 HCA…），页内 43 格逐层展示并链接到所属组详图。
- 未运行模型、未导入 torch/transformers、未加载/下载权重，不声明数值等价。

## 可重复构建命令与落盘入口

解释器（本机实测 3.12，openpyxl 3.1.5）：

```bash
PY=/Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
cd /Users/xiechonghang/Workspace/LLMarchi_excel
```

本票入口（整网总览 + 四组详图，同一候选）：

```bash
"$PY" structure_reader/scripts/read_model_structure.py build-hf \
  --model deepseek_v4_flash_0731_hf --all-variants --json
```

- 最终（修复后）analysis_id：`3626bd75818f57db1faf646be046a04f4e1a57315445fa8afb04950b876ed3c1`
- **整网总览 HTML**：`structure_reader/outputs/deepseek_v4_flash_0731_hf/3626bd75818f57db1faf646be046a04f4e1a57315445fa8afb04950b876ed3c1/viewer/network.html`（169,328 B）
- 四变体详图 HTML（总览链接目标）：同候选 `viewer/decoder_variants.html`（4,855,601 B，与首轮交付逐字节相同）
- 两表 Excel：同候选 `tables/decoder_variants.xlsx`（187,684 B，与首轮交付逐字节相同）
- 整网数据：同候选 `data/network_overview.json`（53,839 B）；四变体数据 `data/reading_variants.json`（1,397,493 B，与首轮交付逐字节相同）
- 来源/运行清单：`reports/source_manifest.json`、`run_manifest.json`（7 个交付文件 SHA 与实际一致）
- 首轮交付候选（保留供复核）：`structure_reader/outputs/deepseek_v4_flash_0731_hf/bc3cd4fd460f0af660108e7d663c4722c3920aaf1b63eee05feb3e9cd248ac69/`（`viewer/network.html` 167,114 B）
- 重复执行同一命令：analysis_id 不变、`reused: true`，候选目录数不增。

单层样板入口保持可用（T01 行为不变）：

```bash
"$PY" structure_reader/scripts/read_model_structure.py build-hf \
  --model deepseek_v4_flash_0731_hf --json
```

当前代码对 layer 4 的内存构建与 T01 候选 `e859305c…` 的 `viewer/decoder.html`、`tables/decoder_reading.xlsx`、`data/reading_document.json`、`data/effective_config.json` **逐字节相同**（仅 identity 中的工具源码 SHA 因本票改动而变化）。本票未为单层再发布新候选，避免重复候选；T01 候选仍可打开。

可选 `--onnx` 仍是同一入口，未启用时行为不变；本票未改 ONNX 构建/门禁，也未生成整网 ONNX（整网总览不宣称是 ONNX 图）。

## 实际测试与静态核对结果

| 命令 / 检查 | 结果 |
| --- | --- |
| `PYTHONPATH=src "$PY" -m unittest tests.test_hf_network_overview` | **21/21 通过**（含新增 `logits_to_keep` 越界 3 项） |
| `PYTHONPATH=src "$PY" -m unittest tests.test_hf_decoder_reading tests.test_m2_toc tests.test_hf_build_missing_xlsx tests.test_hf_network_overview` | **87 tests，86 通过、1 失败**：`test_wrong_layer_rejected` 为接手前既存断言漂移（断言 SWA/Hash 应被拒绝，而四层型已实现；T01 已记录），本票未修改该测试 |
| `PYTHONPATH=src "$PY" -m unittest tests.test_m2_guard` | 2 errors：本机缺 onnx 导致 mock 目标模块不可导入；接手前同样不可运行，本票未改 ONNX 路径，未用桩冒充通过 |
| `PYTHONPATH=src "$PY" -m unittest tests.test_hf_decoder_diagram` | 54 tests，6 failures（既有显示名断言漂移，T01 已记录为与本票改动无调用关系；未修复） |
| `T02-verify/verify_network_roundtrip.py <最终候选>` | **26/26 PASS**：7 个交付 SHA 一致；总览 id 唯一 113、内部锚点无缺失、无外部资源；总览→详图 20 条链接全部命中详图 id（3923 个 id 唯一）；详图返回总览 9 条（页首 1 + 8 组）；43 层格；四组齐备；来源身份表含全部 7 个输入 SHA；`Loop(×43)`；`tie_word_embeddings=false`；`min(sequence_length, logits_to_keep)` 越界裁剪；14 张算子卡源码面板（68 处）与回图定位 |
| 重复执行 `--all-variants` 构建 | 同 analysis_id、`reused: true` |
| 既有详情页回归（共享重构后） | 单层 `decoder.html`/Excel/JSON 与 T01 `e859305c…` 逐字节相同；四变体 `decoder_variants.html`/`reading_variants.json`/`decoder_variants.xlsx` 与首轮交付 `bc3cd4fd…` 逐字节相同 |
| 整网页共享重构核对 | 14 张算子卡各含源码面板与回图定位；`hc_head_pre` 等卡片已合并所涉张量参数引用（如 `Nhc/H/Eps`）；层循环卡片保留“层顺序与四组”入口 |
| 工具身份 | 22 个工具源与当前工作树逐文件 SHA 一致（`reports/source_manifest.json` 复算，漂移为空） |
| 新测试缺输入保护 | 与既有 HF 测试一致：缺 `models/` 或 `hflib/transformers` 时明确 `skipUnless` 跳过；缺失来源另由 `HfConfigError`/`HfBuildError` 具名拒绝（有测试覆盖） |

`T02-verify/verify_network_roundtrip.py` 为一次性只读证据脚本（非产品测试、不参与构建），可对任意 T02 候选重复执行。

## 已知限制与未验证范围

- **静态链接核验不等于浏览器阅读验收**：本次只验证了 HTML 锚点存在、id 唯一、离线无外部资源、源码面板/回图定位链接可解析；未实际操作浏览器点击、缩放、窄屏/键盘导航，也未签用户可读性。
- 整网总览随 `--all-variants` 候选发布（与四变体详图同目录，链接为同候选相对路径）；单层候选按设计不包含 `network.html`——该候选没有四组详图可进入，伪造跨候选链接会更差。单层入口本身保持可用且内容逐字节回归通过。
- 总览只到模块/端点粒度：层内 Attention/压缩器/Indexer/MoE 与双 mHC 混合仍由既有代表层详图承担；完整逐级浏览、默认只显示当前层级、统一 `seqlen/kvcachelen` 场景属 T03，本票未提前实现。
- `logits_to_keep` 只分析非负 int 路径；负整数（`slice` 正起点）与 tensor 型索引未分析，页面/文档明确标注，不声明其输出长度。
- Excel 仍只有 Prefill/Decode 两表分组结构，未加入整网行、性能列或用户公式（性能与 Excel 收敛属后续 T04+）；本票未新增任何性能公式或耗时。
- 未运行 HF 模型、未加载权重、未声明数值等价；量化执行（FP8/FP4 kernel）明确未覆盖。
- 本机无 onnx/onnxsim/onnxruntime，未做真实 ONNX 构建；T01 的 ONNX 补验证据仍有效，本票未改 ONNX 代码路径。
- 图规模：总览主路径图沿用既有 SVG 布局（宽约 3500px、可横向滚动），未做人工视觉打磨；阅读体验待用户/复核确认。
- `test_wrong_layer_rejected` 与 `tests.test_hf_decoder_diagram` 的既存失败按派单要求未在本票修复，也未计入本票回归通过。
- 未重跑依赖历史候选的独立门禁/镜像实现；未改自动化、协调状态、任务单或 GitHub issue。

## 结论

T02 在产品代码中补齐了 DeepSeek-V4-Flash 的 token→logits 整网事实：来源身份与漂移拒绝沿用 T01 机制；四类层组归属由配置逐层推导并机器校验恰好覆盖 43 层；mHC 入口扩展/出口收拢、层循环真实依赖、mask/位置/RoPE/cache 边界与 `min(T,Lk)` 的 logits 终点均按 HF 源码行号表达。详情渲染已收敛为 `detail_render` 单一实现，单层/四变体与整网总览共用，既有页面逐字节不变。总览页可直接进入既有四组 Block/算子详图并返回，构建/门禁/落盘仍走同一套统一流水线。等待再次复核；不自动提交/推送，不关闭 issue，不转入 T03。
