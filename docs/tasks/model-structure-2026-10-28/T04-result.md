# T04 结果：统一 Excel 导出并提供可填写的分析场景

状态：**首轮复核 4 项 + 二轮窄范围 2 项已在本票修复，等待 Codex 再次复核**。本票只做 T04：单层/分组
Excel 收敛到一处通用导出实现，交付同源两页、四组、可填写性能配置、**活公式**、
真实循环归属与“成本待提供”的真实工作簿。未 commit/push、未关闭 GitHub #5、
未进入 T05、未改自动化或协调状态。

## 首轮复核修复（T04-review/findings-round-1.md）

| 复核项 | 修复 |
| --- | --- |
| Standards P3：通用导出写死“四组层数合计” | 改为 `{len(network["groups"])} 组层数合计 {total}`；新增单组 network fixture 回归，不再出现“四组”。 |
| Spec P2：清空输入后未知变成 0，未知也不能恢复 | 参数输入引用与派生式全部改为**带 `IF`/`ISNUMBER`/`IFERROR` 守卫的活公式**。空白、文本、负数、小数都返回“未知（原因）”，恢复/补填后由同一工作簿重算回数值；不再把未知折成静态文本。 |
| Spec P2：缺少实际复合维度 | 新增 3 列“实际输入/输出/权重 shape（按本表配置重算）”。符号 shape 和原参数联查保留；实际 shape 由轴表达式编译为活公式。`B×seqlen` 在 B=2、seqlen=4 时为 8，输出 `[8,4096]`；未知/不支持/歧义写原因，不猜值。 |
| Spec P2：通用分组强制 Decoder 层归属 | `layer_indices`/`representative_layer_index` 改为**可选元数据**；单页 fixture 不再声明层号。DeepSeek 适配仍严格校验四组非空、代表层属于本组、层索引不重叠且恰好覆盖连续层集合。 |

首轮候选 `dab11c8963702e47427b8b6440a48e1ed917ace02b4bdedee1003c0d77e334f7`
按用户要求保留，供复核对照；本轮最终候选为 `c3fd4296…`。首轮候选不修改。

### 二轮窄范围补充修复

| 复核补充 | 修复与证据 |
| --- | --- |
| 非数字文本使 `INT` 先报 `#VALUE`（AND 不短路） | 整数输入守卫改为整体 `IFERROR(AND(ISNUMBER(ref),INT(ref)=ref,ref>=0),FALSE)`。真实候选副本在 `Decode!D12` 填 `abc` 后重算：`seqlen`、`L_kv`、HCA `block_bias`、实际 shape 均为“未知（原因）”，无 `#VALUE`；同一文件改回 1 后恢复 `L_kv=128`。见 `workbook-check.json` 的 `textabc`/`textabc_recovered`。 |
| 实际 shape 的轴绑定忽略显式 `parameter_refs` | 新增 `_axis_symbol_row`：同名候选先用本算子/张量显式引用消歧，恰好一个明确候选才绑定；零个或多个才写“符号存在歧义”。为 fixture 增加未被引用的同名 `H_other=16`，回归证明实际 shape 绑定显式 `H=8`；去掉显式引用、或同时显式引用两个候选时写歧义原因、不猜值。 |


## 接手确认

- 工作区：`/Users/xiechonghang/Workspace/LLMarchi_excel`；cwd 同上。
- HEAD：`947ae78f120279403ec0a4c539e99d3d8869e1d6`（main，与派单一致）；本票未产生 commit。
- 接手快照：`/var/folders/cf/v8s2g_0n0_5c9x9brbcy1nw00000gn/T/llmarchi-T04-baseline-20261028-c13mnff8/manifest.json`
  （362 项，16:43:14 采集；同目录 `files/` 为内容副本）。实际改动与删除均对照该快照。
- T03 已验收候选保留：`d496600d0c5ec471695e9f08afe6014b60a83197d4a55d3c53e3a1984a3df342`
  与首轮候选 `b20ac96b…`；T03 的 `data/layer_structures.json`、
  `data/network_overview.json` 与本票最终候选逐字节相同。

## 交付入口

```bash
PY=/Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
cd /Users/xiechonghang/Workspace/LLMarchi_excel
PYTHONPATH=structure_reader/src "$PY" structure_reader/scripts/read_model_structure.py \
  build-hf --model deepseek_v4_flash_0731_hf --all-variants --json
```

- 最终 analysis_id：`c3fd4296377dcf42ef682ebbebd00f013595a17cde0ca0c3e6f9a9c472e22ff9`
- **真实 Excel**：[tables/decoder_variants.xlsx](../../../structure_reader/outputs/deepseek_v4_flash_0731_hf/c3fd4296377dcf42ef682ebbebd00f013595a17cde0ca0c3e6f9a9c472e22ff9/tables/decoder_variants.xlsx)
  （262,368 B；SHA256 `2da0caf15976f058cd46bc9dc2f8c2143a9eef8bd59ca04db34b309cd534fff0`）
- 分层页（T03 能力保持）：同候选 [viewer/structure.html](../../../structure_reader/outputs/deepseek_v4_flash_0731_hf/c3fd4296377dcf42ef682ebbebd00f013595a17cde0ca0c3e6f9a9c472e22ff9/viewer/structure.html)
  （2,694,048 B；页面代码未改）。
- 同候选数据：`data/layer_structures.json`（793,560 B）、
  `data/network_overview.json`（53,152 B）、`reports/source_manifest.json`、
  `run_manifest.json`。
- 首轮候选（保留，不修改）：
  `dab11c8963702e47427b8b6440a48e1ed917ace02b4bdedee1003c0d77e334f7/`。
- 单层入口仍可用：`build-hf --model deepseek_v4_flash_0731_hf --layer-index 4`；已在内存
  构建核对 `tables/decoder_reading.xlsx`，同样为可填写配置 + 活公式。
- 重复执行同一命令：analysis_id 不变、`reused: true`，候选目录数不增。

## 输入身份与构建方式

- 模型键 `deepseek_v4_flash_0731_hf`；implementation `transformers`；eager。
- 源码版本 `536ecc007387a50e77603bb5d92100e9b07514cc`；配置
  `models/deepseek-ai/DeepSeek-V4-Flash-0731/config.json`。
- `reports/source_manifest.json` 记录 7 项官方输入 SHA、24 项工具源码 SHA、身份与
  analysis_id。工具清单已移除 `render/m2_variants.py`、加入 `export/hf_workbook.py`。
- 候选中 6 个交付文件 SHA、24 项工具源、7 项输入全部复算一致。
- 真实构建不运行/实例化模型、不读取权重；未使用网络。

## 复用 / 重构 / 删除 / 新增

### 保留复用

- `export/operator_workbook.py`：参数行绑定、显式 `parameter_refs` 消歧、联查公式、
  样式、具名异常、固定工作簿时间与稳定 ZIP 全部沿用。
- `extract/hf_layer_structures.py` 的四组同源容器、`extract/deepseek_v4_hf.py` 的
  Block 结构、`extract/hf_scenario.py` 的场景预设/派生式、
  `extract/deepseek_v4_network.py` 的整网外围与四组归属：未新建第二份结构事实。
- `scenario_evaluator.py` 的白名单 AST 与三态语义：Excel 公式编译复用同一白名单，
  没有引入 `eval` 或任意 Python 解释器。
- `build_hf_decoder.py` 的来源核验、输入/工具身份、失败拒绝、内容寻址发布、ONNX 门禁
  保持；T03 分层页与参数/张量/图/源码联查未重写。

### 原位重构

| 文件 | 改动 |
| --- | --- |
| `export/operator_workbook.py` | 新增通用 `export_workbook(spec)`：一处创建 Workbook、按声明建 sheet/分组、写可填写输入、活公式、参数引用、实际 shape、保存与稳定 ZIP；`export_operator_workbook(v1 doc)` 为薄兼容适配。通用检查支持可选层元数据、动态组数汇总、输入类型、引用闭合与未知/歧义。 |
| `scenario_evaluator.py` | 新增 `expression_names` 与 `excel_formula`：把白名单表达式编译为 Excel 公式（`//`→`INT(a/b)`、`%`→`MOD`、`min/max`→`MIN/MAX`、`and/or`→`AND/OR`）。 |
| `build_hf_decoder.py` | 单层路径改用 `build_layer_documents` 同次取得结构 + v1 文档；`_export_workbook` 统一调用 `export/hf_workbook.py`；工具身份清单同步。 |
| `cli.py`、`structure_reader/README.md`、`docs/hf_reading_variants_contract_v1.md` | 更新为“两页、可填写配置、活公式、实际 shape、成本待提供”的入口说明与契约增量。 |
| `tests/hf/test_operator_workbook.py` | 静态导入白名单加入 `structure_reader`（复用 `scenario_evaluator`），其余 v1 断言保持。 |
| `tests/test_hf_layer_levels.py` | T03“导出仍在 render”断言改为 T04 事实：`render.m2_variants` 已删除、通用导出在 `export`。 |

### 新增

| 文件 | 职责 |
| --- | --- |
| `export/hf_workbook.py` | DeepSeek 模型适配：固定两页/四组/示例场景；把 Block 参数与 `derivations` 映射成输入引用/派生公式/未知；严格校验真实层归属；生成整网归属摘要。不创建 Workbook、不保存文件。 |
| `tests/test_operator_workbook_v2.py` | 21 项 T04 回归：单页/双页声明、无层号单页 fixture、动态组数汇总、可填写输入、活公式、实际 shape、跨组隔离、HCA 轴别名、独立期望值、未知不默认 0、循环归属、成本待提供、稳定字节、具名拒绝。 |
| `T04-verify/verify_workbook.py` | 独立验证：真实候选 + LibreOffice 重算 + 编辑链 + 实际 shape。 |
| `T04-verify/workbook-check.json` | 上述验证的实际结果与候选 SHA。 |
| `T04-verify/changed-files.json`、`T04-verify/incremental.patch` | 相对接手快照的改动/新增/删除证据。 |

### 删除 / 退役

- 删除 `render/m2_variants.py`：其 `export_variants_workbook` 另建 Workbook、调用导出器
  私有函数，正是收敛目标；调用者已迁移，无残留引用。
- 本票中间候选 `3d08d9ee…`、`afe019d0…`、`fbec4a65…`、`2069d733…` 在最终候选发布后按精确路径
  删除（记录于 `changed-files.json`）；首轮候选 `dab11c89…` 按要求保留，
  T03/历史候选未动。

## 工作簿设计（真实 Excel）

### 两页、四组、同一事实

- `Prefill`、`Decode` 两页；每页四个层组：SWA+Hash（0/1）、CSA+Hash（2）、
  HCA+TopK（3/5…41，20 层）、CSA+TopK（4/6…42，20 层）。
- 每页 277 个算子行（58/76/68/75），来自与 HTML 同一份 `hf_layer_structure.v1`；
  不机械复制 43 层，也不逐专家复制行。
- 每组参数区保留参数 id/符号/含义/值/值类型/表达式/来源类型/来源定位/说明；
  算子区保留符号 shape、模块路径、参数引用（可点击定位）、联查公式、源码位置、
  分析状态，并新增：
  - 循环归属（层组/真实层/次数）；
  - 成本状态（用户公式待提供）；
  - 实际输入/输出/权重 shape（按本表配置重算）。
- 每页末尾“整网归属”列出网络外围 14 项、四组真实层索引/次数 2/1/20/20，并注明
  共 43 层、模板复用不表示共享权重、Hash MoE 层 0/1/2 消费 `input_ids`、
  父组摘要与子项不重复计入整网成本。组数/合计文字按实际声明动态生成，不写死四组。

### 可填写性能配置

每页顶部黄色输入区（真实数值/文本单元格，可改）：

| 配置 | Prefill 示例 | Decode 示例 | 说明 |
| --- | --- | --- | --- |
| `seqlen` | 8 | 1 | 本次新 token 数；示例可改 |
| `kvcachelen` | 0 | 4096 | 调用前逻辑历史 token 总数；不是物理保留长度/字节数 |
| `batch_size` | 1 | 1 | 示例可改 |
| 计算精度 | FP32 | FP32 | 来自本工程 FP32 阅读特化，可改 |
| 权重/激活存储位宽 | 32/32 bit | 32/32 bit | 示例性能配置，可改 |
| 路由专家命中数 | 未知（未给定） | 未知（未给定） | 数据相关，不根据权重推断，不默认 0/K |

页面说明明确：Prefill(8,0)、Decode(1,4096) 只是可改场景，结构 shape 仍保留符号；
用户性能公式与 XPU 能力未提供。

### 活公式：数值 → 空白/非法 → 恢复

- `B`/`seqlen`/`kvcachelen` 参数行的值是指向本页输入单元格的**守卫公式**，例如：
  `=IF(AND(ISNUMBER($D$12),INT($D$12)=$D$12,$D$12>=0),$D$12,"未知（输入 seqlen 未给定或非法）")`。
- 派生参数是带依赖守卫的真实 Excel 公式，例如 `L_kv`：
  `=IF(AND(ISNUMBER($D$13),...,ISNUMBER($D$34)),IFERROR((MIN($D$34,($D$43-1))+$D$33),"未知（公式求值失败…）"),"未知（依赖参数未给定或非法：W, kvcachelen, seqlen）")`。
- 输入空白、文本、负数、小数时，相关单元格返回“未知（原因）”，不是 0；用户恢复
  原值或补填新值后，同一现成工作簿由 Excel/LibreOffice 重算恢复数值。未知不再写死
  成静态文本，也不靠修改 spec 重新导出来恢复。
- HCA `block_bias` 在 seqlen 空白时返回未知文本，不会误判为 `TRUE`；恢复 seqlen=1
  后重算回 `FALSE`。
- 结构性未知（表达式超出白名单、缺少参数、符号歧义）仍写静态“未知（原因）”，
  因为不是用户可填输入能恢复的情况。

### 实际 shape 与符号 shape 并存

- 原符号 shape 列（输入/输出/权重）和参数联查列保持，不替换为数值文本。
- 新增 3 列实际 shape，按轴编译为活公式：
  - `B×seqlen` 在 B=2、seqlen=4 时为 8；
  - `moe_flatten` 输出实际为 `[8,4096]`（输入 `[2,4,4096]`）；
  - `L_kv+N_comp`、`min(kvcachelen,W-1)`、`2Es`、`Rk÷2`、`Tbuf'` 等轴
    通过归一化/别名绑定到参数单元格；
  - `N_hit`、`N_e` 等数据相关量写“未知（轴未命中参数）”，不猜 0；
  - `⌈T/2⌉`、`B T`、`sqrt(H)`、`Nh××Dh` 等超白名单写法写“超出白名单”原因。
- 实际 shape 公式同样带守卫：输入空白/非法时整条 shape 显示“未知（依赖参数…）”，
  恢复后重算回 `[...]`。轴符号使用算子/张量显式 `parameter_refs` 消歧：同名候选
  恰好一个明确引用时计算，零个或多个才写歧义原因。

### 成本与汇总边界

- 每个算子行成本列写“待提供；不填猜测耗时、不把缺失成本记 0”。
- 整网归属表成本列同样“待提供”；全簿没有成本数值、没有 `SUM` 成本公式。
- 不发明 Roofline/通用成本公式；算子→模块/循环组→整网串行汇总留给用户公式提供后的
  后续任务（T10）。本票不宣称整网性能已实现。

## 循环归属与专家模板

- 四组循环次数来自真实 `layer_indices`：2、1、20、20，共 43；不是 43 行复制。
- 每组只保留一份代表层 Block 模板；说明写明同组每层是独立实例、模板复用不代表
  共享权重、`tie_word_embeddings=false` 使 embed_tokens 与 lm_head 也独立。
- MoE 专家仍是单份模板，不按 256 个专家展开；`E`（专家参数规模）与 `K`（每 token
  选择数）区分；实际命中数保持未知。
- 通用单页 fixture 不声明 `layer_indices`/代表层；DeepSeek 适配在
  `build_structures_workbook_spec` 中严格校验真实层归属后才导出。

## 实际验证证据

### 相关测试（本票环境）

| 测试 | 数量 | 结果 |
| --- | --- | --- |
| `tests/hf/test_operator_workbook.py`（v1 兼容） | 57 | OK |
| `tests/test_operator_workbook_v2.py`（T04 新增） | 24 | OK |
| `tests/test_hf_layer_levels.py` | 41 | OK |
| `tests/test_hf_decoder_reading.py` | 61 | OK |
| `tests/test_hf_network_overview.py` | 21 | OK |
| `tests/test_hf_build_missing_xlsx.py` | 2 | OK |
| `tests/test_m2_toc.py` | 5 | OK |
| `tests/test_scenario_evaluator.py` | 45 | OK |
| `tests/test_page_topology_r2b.py` | 30 | OK |

合计 256 项直接相关 + 30 项页面拓扑 = **286 项通过**。另：
- `test_hf_decoder_onnx.py` 在当前 Python 无 `onnx` 时按既有约定跳过（exit 0）；
- `test_m2_onnx.py` 因当前环境缺 `onnx` 报 4 个导入错误，**不计为通过**。本票未改
  ONNX/onnxsim 构建路径，T03 的隔离 ONNX 证据保留；未在本次环境重跑 ONNX。
- 未运行依赖历史产物的全仓门禁。

### 真实工作簿重算（LibreOffice Kit）

用安装版 LibreOffice Kit CLI 对最终候选真实 xlsx 重算，`data_only=True` 读取缓存值；
证据见 [`workbook-check.json`](T04-verify/workbook-check.json)：

- 两页输入分别为 8/0、1/4096，B=1，FP32，位宽 32；未知专家命中数保持未知。
- 原场景重算值与独立期望一致（节选）：

| 场景 | 组 | L_kv | N_win | N_comp | 其他 |
| --- | --- | --- | --- | --- | --- |
| Prefill(8,0) | CSA+Hash / CSA+TopK | 8 | 2 | 2 | `L_kv_post=8`、`Kisel=2` |
| Prefill(8,0) | HCA+TopK | 8 | 0 | 0 | `block_bias=FALSE` |
| Decode(1,4096) | CSA+Hash / CSA+TopK | 128 | 0 | 1024 | `Tcc=1024`、`Fut=1024`、`Kisel=512` |
| Decode(1,4096) | HCA+TopK | 128 | 0 | 32 | `Tcc=32`、`Fut=32`、`Kisel=32`、`block_bias=FALSE` |

- **同一现成工作簿编辑链**（不重新导出 spec）：
  - 清空 `Decode!D12` 后重算：`seqlen` 行 = “未知（输入 seqlen 未给定或非法）”，
    `L_kv` = “未知（依赖参数…）”，HCA `block_bias` = “未知（依赖参数…）”，
    实际 shape 也显示未知，不是 0 / 127 / TRUE；
  - 补填 8：`seqlen=8`、`L_kv=135`，公式恢复；
  - 填非数字文本 `abc`：`seqlen`/`L_kv`/实际 shape 均为未知，**无 `#VALUE`**；
    同一文件改回 1 后恢复 `L_kv=128`；
  - 填 -1：重新返回未知；恢复 1 后回到原值。
- 修改输入为 `seqlen=4, kvcachelen=4092, B=2` 后重算：四组 `L_kv=131`；
  CSA `N_win=1, N_comp=1024`，HCA `N_win=1, N_comp=32, Tbuf=124, Tcc=31,
  Fqt=31, block_bias=TRUE`；算子联查显示 `B=2; seqlen=4`。
- **实际复合 shape**：同一候选 `moe_flatten` 输入/输出为
  `[1,8,4096]` / `[8,4096]`；修改 B=2、seqlen=4 后为 `[2,4,4096]` / `[8,4096]`。
- 两页/四组/277 算子行/动态组数汇总/成本“待提供”/无 `SUM`/整网 2/1/20/20 共 43
  均由脚本从落盘工作簿核对，不只检查公式字符串存在。
- 重算后端为 LibreOffice native，缺字体提示 Calibri（不影响数值）；**未做原生
  Microsoft Excel 打开验证**。

### 身份与稳定字节

- 候选 6 个交付文件 SHA 与 `run_manifest.json` 全部一致；24 项工具源、7 项输入
  SHA 全部复算一致。
- 同一命令重复执行 `reused: true`、analysis_id 不变。
- v1/v2 导出均有“相同输入相同字节”回归；固定属性时间与 ZIP 时间戳/权限保持。

## 未验证 / 明确限制

- **性能公式与 XPU 能力尚未提供**：没有算子耗时、组小计或整网串行汇总；没有猜测
  成本、没有把缺失成本记 0；不宣称整网性能已实现。
- **HiDream 未接入**：单页能力只由无层号/无 HF 字段的测试 fixture 验证，不声称
  HiDream 已实现，也未建立通用插件框架；GLM/Qwen 未触及。
- **未做原生 Excel GUI 阅读验证**：只做 LibreOffice Kit 重算与 openpyxl 读取；用户
  阅读体验/可读性未代签。
- **T04 未重验浏览器交互**：分层页代码未改，沿用 T03 的浏览器证据；本票只改同候选
  身份与 Excel。窄屏/键盘/缩放/Netron GUI 未验。
- **ONNX 未在本环境复跑**（缺 onnx/onnxsim）；本票未改 ONNX 构建，T03 隔离证据保留。
- 未实现架构编辑器、修改 hidden size 后的重建/校验、并行/融合/通信/Profiler 或
  完整生成耗时；这些不属于本票。

## 保护与停止条件

- 接手前未提交改动、根两项删除、`docs/`、T03/T02 历史候选与证据均保留；未
  reset/clean、未批量格式化、未恢复删除。
- 首轮候选 `dab11c89…` 按用户要求保留，不修改；本票只删除自己的中间候选
  `3d08d9ee…`、`afe019d0…`、`fbec4a65…`、`2069d733…` 与已迁移的 `render/m2_variants.py`。
- 快照之后由派单/协调流程写入的 `docs/handoffs/dsh_structure_batch_01_2026_10_28.md`、
  `docs/tasks/model-structure-2026-10-28/{README.md,execution.json,monitor-state.json,tickets.json}`
  为外部更新；本票只读，未编辑、未改自动化或协调状态。
- 完成即停止，等待 Codex 复核；不 commit/push、不关闭 #5、不进入 T05。
