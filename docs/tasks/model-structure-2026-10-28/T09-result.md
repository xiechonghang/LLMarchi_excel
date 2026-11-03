# T09 结果：五模型结构交付整合与替换代码收尾

状态：**实施完成，等待 Codex 独立复核**。未关闭 GitHub #10，未进入 T10/T11，未改协调/自动化文件，未 commit/push，自动化仍为 PAUSED。

## 接手核对

- 工作区：`/Users/xiechonghang/Workspace/LLMarchi_excel`；分支 `main`；HEAD `947ae78f120279403ec0a4c539e99d3d8869e1d6`（本票前后未变，未产生提交）。
- 接手快照：`/var/folders/cf/v8s2g_0n0_5c9x9brbcy1nw00000gn/T/llmarchi-T09-baseline-20261029-fc8cchru/manifest.json`，`598` 文件，内容副本在 `files/`；
  接手时工作树 `git status` 共 102 项（含根两项删除）。
- 根 `CONTEXT.md`、`2026_10_17_workspace_structure_review.md` 的既有删除保持；所有既有修改/新文件、官方来源与正式候选均未 reset/clean/覆盖。
- 基线内文件本次只增量修改，未做批量格式化；交接快照后由协调流程写入的文件（execution/tickets/monitor 等）保持只读。

## 统一入口与五模型当前交付

五模型共用同一入口（同一产物布局、同一身份算法）：

```sh
cd /Users/xiechonghang/Workspace/LLMarchi_excel
PYTHONPATH=structure_reader/src \
  /Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  structure_reader/scripts/read_model_structure.py \
  build-hf --model <model_key> --all-variants --json
```

完整索引（产物/SHA/来源/覆盖/显式拒绝/历史候选/限制）见 [五模型结构交付索引](../../../structure_reader/docs/model_delivery_index.md)。索引附带机器可读 JSON 块，由 `T09-verify/verify_candidates.py` 对磁盘实际候选逐项复核。

| 模型 | `model_key` | 当前 `analysis_id` | 覆盖 |
| --- | --- | --- | --- |
| DeepSeek-V4-Flash-0731 | `deepseek_v4_flash_0731_hf` | `61b74593da1e9fe4f84cd0384035a42b0c356c9063d26470c6ca5f9ebba3309d` | 43 层 / 4 组（swa-hash、csa-hash、hca-topk、csa-topk） |
| GLM-5.3 | `glm_5_3_hf` | `5fb3633254e03fd5001eeda4319c03b561cec3bf69f3e5ecc7c9aeddc74fad86` | 78 层 / 3 组（full-dense、shared-moe、full-moe） |
| GLM-5.3-Flash | `glm_5_3_flash_hf` | `ee00e2cc1826f445042f474fd834ece43633145cac8cf2287eb082f7fd412391` | 45 层 / 3 组（kda-dense、dsa-moe、kda-moe） |
| Qwen3.8-Flash-Next | `qwen_3_8_flash_next_hf` | `50303d62214148d4a69e6329219a8cdb824e8d7ba1245a984d12e3a941737a76` | 48 层 / 3 组（gdn-moe、gdn-ple-moe、qsa-moe） |
| HiDream-O1-Image | `hidream_o1_image_hf` | `8af3d408ef8feea26c06e7564de37953557fc397bf5a93098f00ad133771b564` | 单次 forward：decoder 36 层 + 条件视觉塔 27 blocks（单 sheet） |

| 模型 | 官方来源 | 入口 HTML | 工作簿 |
| --- | --- | --- | --- |
| DeepSeek-V4-Flash-0731 | checkpoint `models/deepseek-ai/DeepSeek-V4-Flash-0731`；transformers `536ecc00…` | [`structure.html`](../../../structure_reader/outputs/deepseek_v4_flash_0731_hf/61b74593da1e9fe4f84cd0384035a42b0c356c9063d26470c6ca5f9ebba3309d/viewer/structure.html) `aa6144f7f706902f…` | [`decoder_variants.xlsx`](../../../structure_reader/outputs/deepseek_v4_flash_0731_hf/61b74593da1e9fe4f84cd0384035a42b0c356c9063d26470c6ca5f9ebba3309d/tables/decoder_variants.xlsx) `2da0caf15976f058…` |
| GLM-5.3 | `zai-org/GLM-5.3` @ `ba32f8c1…`（glm_moe_dsa）；transformers `536ecc00…` | [`structure.html`](../../../structure_reader/outputs/glm_5_3_hf/5fb3633254e03fd5001eeda4319c03b561cec3bf69f3e5ecc7c9aeddc74fad86/viewer/structure.html) `6ec8f5cb204f8e77…` | [`decoder_variants.xlsx`](../../../structure_reader/outputs/glm_5_3_hf/5fb3633254e03fd5001eeda4319c03b561cec3bf69f3e5ecc7c9aeddc74fad86/tables/decoder_variants.xlsx) `135105af6bb85fe6…` |
| GLM-5.3-Flash | `zai-org/GLM-5.3-Flash` @ `46a050be…`（glm5_next）；transformers `536ecc00…` | [`structure.html`](../../../structure_reader/outputs/glm_5_3_flash_hf/ee00e2cc1826f445042f474fd834ece43633145cac8cf2287eb082f7fd412391/viewer/structure.html) `d406023c9e6883db…` | [`decoder_variants.xlsx`](../../../structure_reader/outputs/glm_5_3_flash_hf/ee00e2cc1826f445042f474fd834ece43633145cac8cf2287eb082f7fd412391/tables/decoder_variants.xlsx) `f60e631ec3326480…` |
| Qwen3.8-Flash-Next | `Qwen/Qwen3.8-Flash-Next` @ `de4b8e4d…`（qwen4_exp）；transformers `536ecc00…` | [`structure.html`](../../../structure_reader/outputs/qwen_3_8_flash_next_hf/50303d62214148d4a69e6329219a8cdb824e8d7ba1245a984d12e3a941737a76/viewer/structure.html) `7f082c36a1556119…` | [`decoder_variants.xlsx`](../../../structure_reader/outputs/qwen_3_8_flash_next_hf/50303d62214148d4a69e6329219a8cdb824e8d7ba1245a984d12e3a941737a76/tables/decoder_variants.xlsx) `b6450ca35d8dad38…` |
| HiDream-O1-Image | `HiDream-ai/HiDream-O1-Image` @ `0b0901d9…`；GitHub `2c2d29ff…` | [`structure.html`](../../../structure_reader/outputs/hidream_o1_image_hf/8af3d408ef8feea26c06e7564de37953557fc397bf5a93098f00ad133771b564/viewer/structure.html) `6e639cf84a040c26…` | [`decoder_variants.xlsx`](../../../structure_reader/outputs/hidream_o1_image_hf/8af3d408ef8feea26c06e7564de37953557fc397bf5a93098f00ad133771b564/tables/decoder_variants.xlsx) `078b0bc34a02f4da…` |

每个候选目录同时包含 `README.md`、`run_manifest.json`、`data/layer_structures.json`、`data/network_overview.json`、`reports/source_manifest.json`；除 DeepSeek 分层入口沿用历史清单外，另有 `data/effective_config.json`。DeepSeek 单层入口 `--layer-index 4` 保持兼容并仍可用。

## 本票改动：复用 / 重构 / 删除 / 新增

### 复用与保留（未改语义）

- 五模型各自的 `rules/*.py`、`extract/*.py`、`extract/*_network.py`、`checks/*_contracts.py`、`render/*_narrative.py`、`render/*_source_labels.py` 与 `render/hf_model_profile.py` 的模型事实、层组/场景/守卫、源码证据与显示名均保留；本票只改公共编排与导出层的重复包装。
- 公共链仍只有一条：`build_structures_container` → 各模型 `build_layer_documents` → `network_overview` → `render_structure_page` → `operator_workbook.export_workbook` → 内容寻址落盘。
- 旧 reference/DSpark 实现与入口（`extract/deepseek_v4_reference.py`、`extract/dspark_*.py`、`checks/dspark_contracts.py`、`build_hf_decoder._deepseek_adapter` 的 `supports_single`/`--onnx`）未被整包删除；历史候选、官方来源文件与 `models/`、`hflib/` 输入保持只读。

### 重构（同形闭包收敛到声明 + 公共函数）

`build_hf_decoder.py` 从 1492 行降到 1320 行；模型适配现在只声明差异字段与文案：

- `_Adapter` 改为声明式：`page_item_extra` / `identity_extra` / `manifest_group_fields` / `manifest_network_fields` / `manifest_fixed` / `effective_value_fields` / `effective_extra_fields` / `effective_layer_fields` / `effective_parameter_records` / `variant_effective_config`。
- 新增公共函数 `_page_items`、`_shipped_structures`、`_identity_structures`、`_run_manifest_structure`、`_effective_config`、`_structure_builder`（[build_hf_decoder.py:166](../../../structure_reader/src/structure_reader/build_hf_decoder.py#L166)、[:196](../../../structure_reader/src/structure_reader/build_hf_decoder.py#L196)、[:217](../../../structure_reader/src/structure_reader/build_hf_decoder.py#L217)、[:232](../../../structure_reader/src/structure_reader/build_hf_decoder.py#L232)）。
- 四个 Decoder 模型共用 `_structure_builder`；HiDream 保留自己的 `build_variant_fn` 闭包，因为 decoder 0..35 与 ViT 0..26 是两个独立索引空间（[build_hf_decoder.py:809](../../../structure_reader/src/structure_reader/build_hf_decoder.py#L809)）。
- 单层入口与分层入口改为调用同一组公共函数；`analysis_id` 仍由 input/tool/identity 的规范 JSON 计算，未改字段语义（本次工具源码变化后自动产生新身份）。

### 删除（有调用证据的重复/无调用包装）

删除前逐个核对调用点：这些符号只作为 `_Adapter` 字段被 `run_hf_build`/`_run_variants` 使用，`tests/`、`scripts/`、其它 `src/` 模块均无引用（`grep -rn` 无结果）；导出模块中的私有包装在`__all__` 与测试中均无入口。

| 删除项 | 位置 | 替代 |
| --- | --- | --- |
| 5 份 `page_items` 闭包 | 五个 `_*_adapter()` | `_page_items(adapter, variants)` |
| 5 份 `shipped` 闭包 | 同上 | `_shipped_structures(adapter, container)` |
| 5 份 `identity_structures` 闭包 | 同上 | `_identity_structures(adapter, variants)` |
| 5 份 `run_structure` 闭包 | 同上 | `_run_manifest_structure(...)` |
| 4 份同形 `structures` 闭包 | DeepSeek/GLM-5.3/GLM-5.3-Flash/Qwen | `_structure_builder(...)`（HiDream 保留） |
| 5 份 `effective_config` 外层 dict | 同上 | `_effective_config(adapter, values)` + 字段元组 |
| 4 份 `extra_variant_documents` 闭包 | GLM/GLM-Flash/Qwen/HiDream | `variant_effective_config` 声明 |
| 5 份无调用的私有 `_parameter_rows` | 五个 `export/*_workbook.py` | 已由 `hf_workbook_common.group_from_block → parameter_rows` 组装 |
| 4 份无调用的私有 `_network_section` | `hf_workbook.py`、`glm53_workbook.py`、`glm53flash_workbook.py`、`qwen38flash_workbook.py` | `_sheet()` 已直接传 `net_label`/`net_notes`（HiDream 的 `_network_section` 仍被使用，保留） |
| 无使用导入 | 五个 export 模块的 `parameter_rows` 别名等 | 删除随包装失效的导入 |

公共导出接口（`export_structures_workbook`、`export_single_structure_workbook`、`SHEET_SCENARIOS`、`DERIVATION_TO_PARAM`、`INPUT_IDS`）全部保留。

### 新增

- [model_delivery_index.md](../../../structure_reader/docs/model_delivery_index.md)：五模型当前候选、真实产物/SHA、来源、覆盖、支持/拒绝入口、历史候选与限制的单一索引（含机器可读 JSON 块）。
- [structure_reader/README.md](../../../structure_reader/README.md) 顶部增加指向该索引的统一入口说明。
- `T09-verify/`：候选静态核验、跨模型回归、浏览器点击、Qwen 原生重算、测试清单、截图与改动清单脚本。
- `T09-verify/five-model-delivery.dataflow.html`：可选 Archify 数据流讲解图（仅数据产物来源，见下）。

### Qwen 旧说明更正（精确范围）

`extract/qwen38flash_hf.py` 两处旧文案按 T07 验收要求更正：

1. `Bq` 示例由“S4/P4092 时依次为 1023,1023,1023,1024（未达 512 上限）”更正为“N_q=1023,1023,1023,1024，B_q 均为 512（min(Btop=512, N_q) 按 block 预算截断）”。公式本身早已是 `B_q = min(Btop, N_q)`，旧示例把未截断的 N_q 当成了 B_q。
2. pooled key Norm 说明由“与逐 token key 共用 k_layernorm”更正为“只对 pooled key 做 RMSNorm(128)；indexer 不做逐 token key 归一化，因此不存在与之共权重”。源码已无逐 token indexer K Norm，旧备注不能继续作为结构事实。

精确影响范围：`data/layer_structures.json` 两处 `note` 叶子；Prefill/Decode 两页各 2 个说明单元格（`I330`、`N374`）；HTML 两处说明区域。其余 Qwen 结构、算子、张量、权重维度、依赖、公式单元格与场景取值不变。同时修正了首版文案中裸 `K` 被显示名替换为 `num_experts_per_tok` 的问题。

## 新候选与身份

工具源码在本票发生变化，五个模型都重建为真实新候选（不是沿用旧指纹）：

| 模型 | 相对接手工作树的工具源变化 | 新 `analysis_id` |
| --- | --- | --- |
| DeepSeek-V4-Flash-0731 | build_hf_decoder.py、export/hf_workbook.py | `61b74593da1e9fe4f84cd0384035a42b0c356c9063d26470c6ca5f9ebba3309d` |
| GLM-5.3 | build_hf_decoder.py、export/glm53_workbook.py | `5fb3633254e03fd5001eeda4319c03b561cec3bf69f3e5ecc7c9aeddc74fad86` |
| GLM-5.3-Flash | build_hf_decoder.py、export/glm53flash_workbook.py | `ee00e2cc1826f445042f474fd834ece43633145cac8cf2287eb082f7fd412391` |
| Qwen3.8-Flash-Next | build_hf_decoder.py、export/qwen38flash_workbook.py、extract/qwen38flash_hf.py | `50303d62214148d4a69e6329219a8cdb824e8d7ba1245a984d12e3a941737a76` |
| HiDream-O1-Image | build_hf_decoder.py、export/hidream_o1_workbook.py | `8af3d408ef8feea26c06e7564de37953557fc397bf5a93098f00ad133771b564` |

五个候选均通过实际 CLI 构建（退出码 0、`status: success`、`reused: false`），构建命令与索引一致。T05–T08 验收候选 `c3fd4296…`、`abefb101…`、`29f82a3e…`、`c9217122…`、`5423a22f…` 及各自返修候选全部保留，在当前索引中列为历史身份，不冒充当前实现。

## 实际验证

### 1. 受影响测试

`T09-verify/tests.json`：596 项通过，0 失败/错误/跳过（Python 3.12.14）。

| 模块 | 通过 |
| --- | --- |
| `tests/test_hidream_o1_structure.py` | 79 |
| `tests/test_qwen38flash_structure.py` | 49 |
| `tests/test_glm53flash_structure.py` | 62 |
| `tests/test_glm53_structure.py` | 30 |
| `tests/test_hf_decoder_reading.py` | 61 |
| `tests/test_hf_layer_levels.py` | 41 |
| `tests/test_hf_network_overview.py` | 21 |
| `tests/test_operator_workbook_v2.py` | 24 |
| `tests/hf/test_operator_workbook.py` | 57 |
| `tests/test_hf_build_missing_xlsx.py` | 2 |
| `tests/test_scenario_evaluator.py` | 45 |
| `tests/test_m2_guard.py` | 2 |
| `tests/test_page_topology_r2b.py` | 30 |
| `tests/test_page_topology_r2a.py` | 27 |
| `tests/test_page_topology_r1.py` | 17 |
| `tests/test_page_js_execution.py` | 2 |
| `tests/test_model_registry.py` | 15 |
| `tests/test_docs_delivery.py` | 32 |

未计为通过：`tests/test_hf_decoder_diagram.py` 的 6 项旧布局断言（T05 接手时已失败，本票未改该历史测试）；ONNX 相关测试（本机缺 `onnx/onnxsim/onnxruntime`）。

### 2. 交付索引与候选静态核验

`T09-verify/candidates-check.json`（passed）：五候选的产物集合/SHA/字节、`run_manifest.deliverables` 闭合、`analysis_id = SHA256(规范 identity)` 复算、工具源码与输入文件 SHA 复算、层组归属与 `run_manifest.structure` 对照、HTML id 唯一/普通锚点闭合、工作簿 sheet/层组/层数、Qwen 两处旧说明消失且新说明存在、历史候选仍存在且不等于当前候选。

显式拒绝重新触发 9 条且全部为具名 `HfBuildError`：未知模型键；GLM-5.3 / GLM-5.3-Flash / Qwen / HiDream 缺 `--all-variants`；四模型显式 `--onnx`。

### 3. 对照各票验收候选的回归

`T09-verify/cross-model-regression.json`（passed）。差异只允许两类并已逐项核对：

- 候选身份（HTML 中 `analysis_id`、README 的 analysis 行）；DeepSeek/GLM/Flash/HiDream 另有渲染器既有中性 CSS 注释差异（非 DOM/行为）。
- Qwen 仅两处说明：`layer_structures.json` 2 个 `note` 叶子、工作簿 4 个说明单元格、HTML 2 处说明区域。

| 模型 | layer_structures | network_overview | effective_config | 工作簿 | HTML |
| --- | --- | --- | --- | --- | --- |
| DeepSeek-V4-Flash-0731 | same | same | both-missing | same | same-after-candidate-id+css-comment |
| GLM-5.3 | same | same | same | same | same-after-candidate-id+css-comment |
| GLM-5.3-Flash | same | same | same | same | same-after-candidate-id+css-comment |
| Qwen3.8-Flash-Next | 2 leaf diff（仅 note） | same | same | differs（仅 4 个说明单元格） | same-after-candidate-id+notes |
| HiDream-O1-Image | same | same | same | same | same-after-candidate-id+css-comment |

DeepSeek/GLM/GLM-Flash/HiDream 的结构 JSON、整网 JSON、有效配置与工作簿与验收候选**字节相同**；HTML 归一候选身份与 CSS 注释后相同。Qwen 的 2 处 note 为 T07 验收要求的文案更正，公式与结构未动。

### 4. 可见浏览器实际点击

`T09-verify/browser-navigation.json`（passed）：headless Chrome + 同源 iframe 对五个候选实际点击，覆盖整网→每个真实层组→Block→模块→张量→参数→源码面板→返回层组/整网、场景切换与恢复，并检查 id 唯一与普通锚点闭合。

| 模型 | 实际层组 | 点击步骤 | 结果 |
| --- | --- | --- | --- |
| DeepSeek-V4-Flash-0731 | csa-hash, csa-topk, hca-topk, swa-hash | 27 | True |
| GLM-5.3 | full-dense, full-moe, shared-moe | 21 | True |
| GLM-5.3-Flash | dsa-moe, kda-dense, kda-moe | 21 | True |
| Qwen3.8-Flash-Next | gdn-moe, gdn-ple-moe, qsa-moe | 21 | True |
| HiDream-O1-Image | uit-decoder, vit-tower | 15 | True |

截图：`T09-verify/t09-*.png`（每模型整网/两组 Block/算子页）。

### 5. 工作簿原生重算

`T09-verify/formula-recalc-check.json`（passed）：对 Qwen **已交付**工作簿的临时副本改输入，用安装版原生 LibreOffice kit 重算，22 组场景通过，工作簿 SHA `b6450ca35d8dad38fd6e633b0ec129353f130eda04130fa39cc23da95be2c005`，整表无 Excel 错误值。关键场景：

- Prefill/Decode 默认；S=4/P=4092/B=2；逐 query `j=0..3` 的 `Vq/Nq/Bq/Rtail/Ksel`，其中 `Bq` 均为 512（验证更正后的说明，而不是 1023/1024）。
- `j` 负数/小数/越界/文本/空白 → `Vq/Nq/Bq/Ksel` 保持未知，`Ksel_last` 仍由全长给出；零完整块分支 `Bq=0`、tail 仍执行。
- 长历史 131072、seqlen 空白/文本/0、恢复合法输入；每例整表错误值扫描。

其余四模型工作簿与验收候选字节相同，可直接绑定已有原生重算证据（SHA 见交付索引）：DeepSeek `2da0caf1…`、GLM-5.3 `135105af…`、GLM-5.3-Flash `f60e631e…`、HiDream `078b0bc3…`；本票未重复跑其昂贵全量/ONNX。

### 6. 可选 Archify 图

- 交付：[五模型交付数据流图](T09-verify/five-model-delivery.dataflow.html)（Archify dataflow，`finalize` 的 validate/deliver/check/browser-check 全通过，证据在 `T09-verify/archify/`；同一 DOM 的截图见 `archify-five-model-dataflow.png`）。
- 复核后文案微调：`network_overview` 节点说明改为 `JSON：logits / x_pred`，并在“读图边界”卡注明“文本模型 token→logits、HiDream 单次 forward→x_pred，不把 x_pred 说成 logits”；节点/关系/覆盖/其它说明未变，仅重新 finalize 同一输出路径并更新回执与截图。
- 来源与限制：图只使用已落盘候选数据（交付索引 JSON、run/source manifest）与已实现的构建行为；**未**使用 `--repo-root`，未为出图 commit，未把当前未提交工作树当作 HEAD `947ae78f` 的源码证据；图内已注明“T09 当前工作树候选数据（未提交，不是 HEAD 源码证据）”。
- 未另建产品 viewer、未新增第二套源码解析器；页面内导航仍以 `viewer/structure.html` 为准。Archify 工作副本在仓库根 `.archify/`（隐藏、未跟踪，供回执路径追溯），`T09-verify/archify/` 是其证据副本。

## 未验证 / 剩余限制

- DeepSeek 的模块级 ONNX 本票未重建（本机缺 `onnx/onnxsim/onnxruntime`）；能力与代码入口保留，历史含 ONNX 候选保留，但当前候选不含 `onnx/`，不把旧 ONNX 算作当前交付。
- 无 Microsoft Excel GUI 验收；原生 LibreOffice 重算通过不等于 Excel GUI 或用户阅读体验通过。
- 静态分析限制保持：不下载/加载权重、不实例化或运行模型、不做数值等价；GLM/GLM-Flash/Qwen 只做语言主干，HiDream 只做单次 forward，视觉/MTP/采样等边界按各模型 README 与交付索引。
- 用户性能公式与 XPU 能力参数仍未提供：五份工作簿成本列“待提供”，无整网耗时；T10/T11 继续等待，不猜成本。
- Archify 图做了自动门禁与一次截图查看，未做逐区视觉审查；不代签用户阅读验收。

## 改动清单

相对接手快照（`changed-files.json` / `incremental.patch`）：修改 14（产品 8、快照后协调流程写入 6），新增 122（产品 1、证据 42、`T09-dispatch.md/png` 2 为协调创建、`models/` 下 77 个接手前已存在的官方输入），删除 0；根两项既有删除保持。接手前已脏的文件标 `preexisting_dirty`，不归本票。

- 源码：`build_hf_decoder.py`；`export/hf_workbook.py`、`export/glm53_workbook.py`、`export/glm53flash_workbook.py`、`export/qwen38flash_workbook.py`、`export/hidream_o1_workbook.py`；`extract/qwen38flash_hf.py`（仅两处说明）。
- 文档：`structure_reader/README.md`、`structure_reader/docs/model_delivery_index.md`、本文件。
- 证据：`T09-verify/`（脚本、JSON、截图、Archify 候选与回执、changed-files/incremental.patch）。
- 未改：协调/自动化（execution.json、tickets.json、monitor-state.json、任务 README、总交接）、`models/`、`hflib/`、旧候选与官方来源。

停止点：等待 Codex 复核。不关闭 #10，不进入 T10/T11，不 commit/push，不恢复任何暂停的自动化。
