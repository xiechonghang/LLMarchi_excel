# T03 结果：同一份参数化 DeepSeek 结构的逐级浏览

状态：**首轮复核 3 项 P2 已在本票修复，等待再次复核**。本票只做 T03：把四类层组
的层内事实收敛为**一份**参数化 Block 结构（shape 保留 `seqlen`/`kvcachelen` 符号），
用场景预设表达不同调用；在既有图/详情/源码能力上交付「整网 → 层组 → Block →
内部语义算子」的分层离线页并替换旧默认长页装配。未 commit/push、未关闭 GitHub #4、
未进入 T04、未改自动化/协调状态文件。

## 首轮复核修复（P2）

复核记录见 [T03-review/findings-round-1.md](T03-review/findings-round-1.md)。三项均在
本票内修复，首轮候选 `b20ac96b…` 保留，修复后发布新候选（下方 analysis_id）。

### P2-1 普通锚点与分层路由统一

- 问题：`parseRoute` 只识别 `#/group/...`，既有 `#p-…`/`#t-…`/`#op-…`/`#figop-…`
  联查会落回整网并隐藏目标。
- 修复：路由同时解析层级路由与普通锚点。锚点按作用域 id 判定所属层
  （`p-block-<gid>-*` → 该组 Block 参数区；`t/op/figop/mod-block-<gid>-*` → 该组算子层；
  `p/t/op/figop-network-*` 与 `net-*` → 整网层），并以「目标实际所在 `[data-lv]`
  section」兜底。`reveal()` 展开目标的所有祖先 `<details>`、滚入视口并短暂高亮；
  `locateInDiagram` 先切到目标所属层级再展开/定位，不再只在当前 DOM 里找元素。
- 证据（真实 Chrome，非静态检查）：[browser_nav_check.py](T03-verify/browser_nav_check.py)
  覆盖参数联查 `#p-block-csa-topk-Nhc`（进入 Block 层、展开
  `block-params-csa-topk`、目标 top=180/高亮）、浏览器后退回算子模块、张量联查
  `#t-…`（进入算子层张量索引、top=177）、图节点 `#op-…`（展开
  `mod-block-csa-topk-res_attn`、目标可见）、回图定位 `#figop-…`（top=164、details 开）、
  参数深链与初始深链 `#p-…`（top=0/180、`block-params-csa-topk` 开）、张量深链、返回整网；
  [browser-nav.json](T03-verify/browser-nav.json) 记录 14 个交互步骤 + 2 个深链步骤。

### P2-2 压缩条目因果可见边界

- 问题：新增场景说明写 `w ≤ (t+1)//M`；官方源码 `modeling_deepseek_v4.py:437-439`
  与 `:574-581` 实际 `masked_fill(entry >= threshold, -inf)`，正确可见条件是
  `w < (t+1)//M`（`w` 等于阈值即屏蔽）。
- 修复：CSA/HCA 的 `future_threshold` 说明统一为严格小于，并新增
  `first_query_threshold = (kvcachelen + 1) // M`（首查询边界，便于核对
  「等于阈值即屏蔽」）；同步更正 CSA 的历史张量注记（`c_block_bias`）、Indexer
  `future_mask`/top-k 注记、缓存注记与 HCA `block_bias`/条目注记。契约门禁新增：
  非 SWA 结构必须同时有 `future_threshold`/`first_query_threshold`，说明必须含
  `w < (t+1)//M` 与「等于阈值」，出现 `w ≤` 即具名拒绝。
- 边界例（复核给出）：HCA `kvcachelen=4092, seqlen=4, M=128`，首查询 t=4092 阈值
  `(4092+1)//128=31`；条目 31 覆盖 3968..4095 含未来 token，必须屏蔽（推导值
  `first_query_threshold=31`）；末查询 t=4095 阈值 32，条目 31 可见。测试同时覆盖
  `visible(31,4092,128)=False`、`visible(30,4092,128)=True` 与
  `visible(31,4095,128)=True`。

### P2-3 结构构造移出页面层

- 问题：`build_structures_container`/`build_variants_container` 位于
  `render/m2_variants.py`，仍读配置、推导层组并调用提取器，违反「变体构造移出页面模块」。
- 修复：结构容器构造迁入
  [extract/hf_layer_structures.py](structure_reader/src/structure_reader/extract/hf_layer_structures.py)
  （`build_structures_container`，`hf_layer_structures.v1`）；`build_hf_decoder` 与三个
  真实测试改用 extract 入口；`render/m2_variants.py` 只保留 `export_variants_workbook`
  （Excel 导出留待 T04 收敛，本票不重写），删除无实际调用者的
  `build_variants_container` 兼容入口。工具身份加入新提取模块，回归测试断言 render
  不含 `layer_group_assignments`/`load_hf_model_values`/`build_layer_documents`。

## 执行基线

- 工作区：`/Users/xiechonghang/Workspace/LLMarchi_excel`；HEAD `947ae78f120279403ec0a4c539e99d3d8869e1d6`（main，与派单一致）；本票未产生 commit。
- 接手快照：`/var/folders/cf/v8s2g_0n0_5c9x9brbcy1nw00000gn/T/llmarchi-T03-baseline-20261028-ditww4rw/manifest.json`（332 项，15:55:11 采集）。
- 接手前未提交改动、两项根目录删除（`2026_10_17_workspace_structure_review.md`、`CONTEXT.md`）、`docs/`、历史候选全部保留；未 reset/clean、未格式化、未恢复删除。
- 快照之后由派单/协调流程更新了 `docs/handoffs/dsh_structure_batch_01_2026_10_28.md`、`docs/tasks/model-structure-2026-10-28/{README.md,execution.json,monitor-state.json,tickets.json}`（mtime 15:57–15:58，均为外部写入）。本票对这些文件只有读取，没有编辑，也没有改自动化或协调状态。
- 相对快照，本票实际改动/新增见下；`structure_reader/outputs/deepseek_v4_flash_0731_hf/` 接手前 80 个候选。T03 首轮候选 `b20ac96b…` 按要求保留；首轮复核后发布 1 个修复后最终候选。构建过程中被取代的 12 个候选与 1 个 `_failed` 诊断目录已按精确路径删除（未涉及任何接手前候选/用户文件）。

## 交付入口

```bash
PY=/Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
cd /Users/xiechonghang/Workspace/LLMarchi_excel
PYTHONPATH=structure_reader/src "$PY" structure_reader/scripts/read_model_structure.py \
  build-hf --model deepseek_v4_flash_0731_hf --all-variants --json
```

- 最终 analysis_id：`d496600d0c5ec471695e9f08afe6014b60a83197d4a55d3c53e3a1984a3df342`
- **分层浏览入口（真实 HTML）**：`structure_reader/outputs/deepseek_v4_flash_0731_hf/d496600d0c5ec471695e9f08afe6014b60a83197d4a55d3c53e3a1984a3df342/viewer/structure.html`（2,694,048 B；单文件离线，默认只显示整网层）
- 同候选：`data/layer_structures.json`（793,560 B，四份参数化结构 + 场景派生）、`data/network_overview.json`（53,152 B）、`tables/decoder_variants.xlsx`（188,720 B，Prefill/Decode 两表，同一结构的场景投影）、`reports/source_manifest.json`、`run_manifest.json`。
- 首轮候选（按要求保留，供复核对照）：`structure_reader/outputs/deepseek_v4_flash_0731_hf/b20ac96bdbc44fedb41794f97e6dc49ff64c23cf7eb5d5d030aaea5fdd03606b/`。
- 重复执行同一命令：analysis_id 不变、`reused: true`，候选目录数不增。
- 单层入口保持可用：`build-hf --model deepseek_v4_flash_0731_hf`（layer 4）仍产出 `viewer/decoder.html`、两表 `decoder_reading.xlsx` 与阅读 JSON（同一结构的场景投影；已内存构建核对，未另发布候选）。

## 结构：一份参数化事实 + 场景预设

### 唯一结构来源

`extract/deepseek_v4_hf.py` 的 `BlockStructureBuilder` 及 `m2_hca.py`/`m2_swa_hash.py`
子类为每个真实层组构建**一份** `hf_layer_structure.v1`：

- 参数/张量/算子/来源锚点各一份；`block.id == "block"`，文档**不含 `phases`**；
- 所有 shape 使用 `seqlen`（本次新 token 数）与 `kvcachelen`（调用前逻辑历史 token 数），
  旧 `T`/`P` 轴已从结构、整网文档与图标签中移除；`B×T` 改为 `B×seqlen`；
- 缓存调用前/调用后仍是两个张量（`cache_keys` / `cache_keys_post`），残差仍是
  `post⊙Out + combᵀ·In`，共享 KV（`Nkv=1`、K=V 同存储）、压缩器 overlap、Indexer
  top-k/-1 哨兵、block_bias、MoE 单专家模板与 `index_add` 汇合全部保留；
- 新增 `derivations`（纯表达式，交给 `scenario_evaluator` 白名单求值）：
  `L_kv`、`L_mask`、`L_kv_post`、`Tbuf`、`Tbuf_post`、`N_win`、`N_comp`、
  `N_win_i`、`N_comp_i`、`first_window_position`、`future_threshold`、
  `first_query_threshold`（首查询可见边界）、`Ki_sel`（HCA 另有 `block_bias` 条件）。

### 场景预设（只绑定取值，不改结构）

| 场景 id | seqlen | kvcachelen | 用途 |
| --- | --- | --- | --- |
| `prefill-empty`（默认） | 8（示例） | 0 | 空缓存 Prefill |
| `prefill-symbolic` | 符号 | 0 | 验证未知项如实标注 |
| `decode-nonzero` | 1 | 4096 | 非零历史单 token Decode；窗口未闭合 |
| `decode-closed` | 1 | 4095 | 与 M 对齐：本调用闭窗 |
| `decode-multi` | 4 | 4092 | 非零历史多 token，检查没有偷留 `+1`/`T=1` |

派生前提固定写在结构与页面里：**固定配置 + 空态起步、连续正常执行**；不覆盖裁剪/
回滚后的自定义快照。`kvcachelen` 是逻辑历史总数，不等于物理保留长度
（`min(kvcachelen, W-1)`），也不是缓存字节数。

关键推导（源码依据：`cache_utils.py:276-286` 的 `get_mask_sizes`、
`modeling_deepseek_v4.py:198-224` 的 `update`/`store_compression_weights`、
`:226-237` 的 `update_compressor_states`、`:432-439` 的 HCA `block_bias`）：

- `L_kv = L_mask = min(kvcachelen, W-1) + seqlen`（`is_full` 两分支在 `min(cum, W-1)` 下等价）；
- `Tbuf = kvcachelen % M`、`N_win = (Tbuf + seqlen) // M`、`N_comp = kvcachelen // M + N_win`；
- `first_window_position = (kvcachelen // M) · M`；因果可见边界为
  `w < (t+1)//M`（`w` 等于阈值即屏蔽）：末查询阈值 `(kvcachelen+seqlen)//M`，
  首查询阈值 `(kvcachelen+1)//M`；
- HCA `block_bias` 物化条件：`seqlen != 1 and N_comp > 0`（源码 `:432-439` 的 None 分支）。

### 场景对照（实测派生值）

| 场景 | CSA+TopK `L_kv / N_win / N_comp` | HCA+TopK `L_kv / N_win / N_comp / block_bias` | SWA+Hash `L_kv` |
| --- | --- | --- | --- |
| prefill-empty (8, 0) | 8 / 2 / 2 | 8 / 0 / 0 / False | 8 |
| prefill-symbolic | 未知（缺 seqlen） | 未知（缺 seqlen） | 未知（缺 seqlen） |
| decode-nonzero (1, 4096) | 128 / 0 / 1024 | 128 / 0 / 32 / False | 128 |
| decode-closed (1, 4095) | 128 / 1 / 1024 | 128 / 1 / 32 / False | 128 |
| decode-multi (4, 4092) | 131 / 1 / 1024 | 131 / 1 / 32 / True | 131 |

`decode-nonzero` 的 `L_kv=128` 同时说明「物理保留长度 ≠ 逻辑历史」；HCA
`decode-multi` 的首查询阈值 31 / 末查询阈值 32 是可见边界的核对点（条目 31 在首查询
屏蔽、末查询可见）；`decode-multi`
的 `L_kv=131` 说明结构按符号 `seqlen` 表达而不是固定 `+1`。未知场景的
`N_comp`/`Ki_sel` 会沿依赖链保持 unknown（不默认成 0）。

### 已修正的旧阶段假设

- 删除 `_PhaseBuilder` 的 prefill/decode 双构建与 `is_prefill` 分支；缓存注记、
  coverage 注记、标题不再按阶段各写一份；
- 删除 Decode 中固定 `+1`/`T=1`/`seq_len==1` 的结构性表述（`seq_len==1` 现在只作为
  源码原文出现在 `source_refs` 证据里，或作为 `block_bias` 场景条件由求值器判定）；
- `L_kv`/`L_mask`/`L_kv_post`/`Tbuf'`/`N_comp` 等参数表达式与 derivations 共用同一
  字符串常量，避免公式与派生值两处漂移。

## 分层页面（默认只显示当前层级）

`render/hf_structure_page.py`（新）消费整网总览 + 四份结构，输出单文件离线页：

- **整网** `#/network`（默认）：token→logits 主路径图、43 层真实顺序格、四类层组卡片；
  网络级参数/张量/算子/来源身份收进默认折叠的 `<details>`。
- **层组** `#/group/<gid>`：真实层索引/代表层/同构模板说明、场景派生面板、Block 模块链。
- **Block** `#/group/<gid>/block`：Block 总览图（真实分支/残差/缓存载体）、模块卡片、
  块边界/缓存载体清单（折叠）、参数表（折叠）、按实际层型生成的压缩器选择说明、场景派生（折叠）。
- **内部算子** `#/group/<gid>/ops[/<module>]`：按模块折叠的模块图 + 算子卡（参数/权重/
  源码按需展开）、张量索引、缓存与覆盖注记；`/ops/<module>` 直接展开对应模块。
- 每层都有静态面包屑与「↑ 返回」普通链接；场景条在页首，切换只换派生面板与说明，
  结构 shape 不变。页面脚本只做路由/场景渲染，不派生结构事实。
- 复用：`diagram.py` 的 `build_overview_graph`/`build_module_graph`/`build_csa_composite_graph`/
  `render_svg`/`legend_html`，`detail_render` 的参数行/张量行/算子卡/源码面板/回图定位，
  `module_narrative` 的模块讲解，`deepseek_v4_network` 的整网事实与四组归属。
- 可选 ONNX/Netron：启用 `--onnx` 且存在 bundle 时，每个模块显示 prefill/decode 两份
  阅读特化 ONNX 的本地 Netron 链接与下载链接；未启用时不出现、不伪造。

## 复用 / 重构 / 删除 / 新增清单

### 新增（新职责或提炼来源）

| 文件 | 职责 |
| --- | --- |
| `structure_reader/src/structure_reader/extract/hf_scenario.py` | 场景预设（seqlen/kvcachelen）、派生量三态求值、把同一结构投影成既有 prefill/decode 文档（`project_legacy_phase`）的唯一入口。 |
| `structure_reader/src/structure_reader/render/hf_structure_page.py` | 四级分层离线页（整网/层组/Block/算子）、场景面板、路由与面包屑；只消费已有结构事实。 |
| `structure_reader/extract/hf_layer_structures.py` | 四类层组的 `hf_layer_structures.v1` 容器构造（每层组一份结构 + 同源 prefill/decode 投影）；从 render 页面层迁入。 |
| `structure_reader/tests/test_hf_layer_levels.py` | 41 项语义回归：单结构/场景参数身份/无残留 T、P 与 +1/闭窗与未知/阈值边界（含等于屏蔽）/依赖保留/页面层级/锚点路由/构造分层/契约门禁。 |

### 原位重构（复用既有实现）

| 文件 | 改动 |
| --- | --- |
| `extract/deepseek_v4_hf.py` | `_PhaseBuilder` → `BlockStructureBuilder`：单次构建、符号 seqlen/kvcachelen、derivations、中性条件注记；`build_layer_documents` 一次构建返回（结构 + 两阶段投影）；`build_layer_structure`/`build_reading_document` 成为薄包装；`logits_slice_length`、四层型分派等保留。 |
| `extract/m2_hca.py`、`extract/m2_swa_hash.py` | 构建器改为单结构子类；HCA/SWA 各自派生量与条件（HCA `block_bias`、SWA 无压缩）；删除 `build_*_document` 双阶段包装。 |
| `extract/deepseek_v4_network.py` | 整网文档统一 `seqlen`/`kvcachelen`；`detail` 改为 `structure.html` 的分层路由锚点（`group`/`block`/`operators`/`tensors`）。 |
| `render/m2_variants.py` | 只保留 `export_variants_workbook`（Excel 导出，T04 收敛）；结构容器构造迁入 extract，删除旧长页渲染与 `build_variants_container` 兼容入口。 |
| `build_hf_decoder.py` | `--all-variants` 改为发布分层页与结构 JSON；每层组同时做结构契约检查；工具身份改用新模块；身份/清单/manifest 更新；单层路径保持。 |
| `checks/deepseek_v4_hf_contracts.py` | 新增 `validate_hf_layer_structure`（单结构/参数语义/shape 词表/派生表达式/场景 honesty）；整网检查改 `min(seqlen, Lk)` 与新 detail 锚点。 |
| `render/module_narrative.py` | 总览讲解按场景参数（结构级/空缓存/非零历史）如实表述，不再写死 prefill/decode 文案。 |
| `render/detail_render.py` | `render_tensor_row` 增加 `anchor=False`（同一结构多处列出时不产生重复 id）。 |
| `cli.py`、`structure_reader/README.md` | 入口/摘要/帮助改为分层浏览；README 说明新旧两个 `structure.html` 的区别。 |

### 删除 / 退役

- 删除 `render/network_page.py`：整网层内容迁入分层页 L0（`_network_graph`/层序带/组表/来源表在同一模块中复用），不再保留第二套整网页装配。
- 删除 `render/m2_variants.py` 的 `render_variants_page`、`_toc`、`_namespace`、`_chapter`、`_page_css`：旧「Prefill 四组 → Decode 四组」长页被分层页取代。
- 删除 `extract/m2_hca.py`/`m2_swa_hash.py` 的 `build_hca_document`/`build_swa_document`/`build_csa_hash_document`/`_document`：双阶段构建入口被「单结构 + 投影」取代。
- 删除 `render/m2_variants.py` 的 `build_structures_container`/`build_variants_container`：结构构造迁入 `extract/hf_layer_structures.py`；页面层不再读取配置或推导层组。
- 不再发布 `viewer/network.html`、`viewer/decoder_variants.html`、`data/reading_variants.json`（后者可由结构 JSON 与投影重建）；历史候选目录保持不动。
- 删除本票产生的 7 个中间候选与 1 个 `_failed` 诊断目录（精确路径核对后删除）；不涉及接手前候选或用户文件。

### 复用（未修改）

`rules/deepseek_v4_hf.py`（来源指纹/四组归属）、`render/diagram.py`（连接与 SVG）、
`render/decoder_page.py`（单层入口与页面骨架，未改）、`export/operator_workbook.py`（两表
Excel 参数/算子/样式）、`scenario_evaluator.py`（白名单三态求值）、
`build_hf_decoder.py` 共用的输入/工具身份、源码快照、ONNX 门禁与内容寻址落盘。
`render/decoder_page.py`、`render/display_names.py`、`render/diagram_source_labels.py`、`render/diagram.py` 本票未修改（它们的未提交改动来自接手前/T02，保持原样）。

## 来源与配置身份

- 模型键 `deepseek_v4_flash_0731_hf`；实现 `DeepseekV4ForCausalLM`/`DeepseekV4Model`/
  `DeepseekV4DecoderLayer`（eager）；源码版本 `536ecc007387a50e77603bb5d92100e9b07514cc`；
  根配置 `models/deepseek-ai/DeepSeek-V4-Flash-0731/config.json`。
- 7 个分析输入的 SHA256 写入候选 `reports/source_manifest.json`，并由
  `verify_source_fingerprints` 在构建前复算；缺失/漂移即具名拒绝。
- 四组真实归属（`rules.layer_group_assignments`）：`swa-hash` 2 层(0,1)、`csa-hash` 1 层(2)、
  `hca-topk` 20 层(3,5..41)、`csa-topk` 20 层(4,6..42)，共 43 层，保持真实交错顺序。
- 未运行/实例化 HF 模型、未导入 torch/transformers、未加载或下载权重，不声明数值等价。

## 实际验证

| 检查 | 结果 |
| --- | --- |
| `unittest`：`test_hf_decoder_reading`(61)、`test_hf_layer_levels`(41)、`test_hf_network_overview`(21)、`test_m2_toc`(5)、`test_hf_build_missing_xlsx`(2) | **130/130 通过**（首轮 125；本轮 +5 项锚点路由/构造分层/阈值边界与门禁回归）（[tests-core.log](T03-verify/tests-core.log)） |
| `test_hf_decoder_diagram` | 54 tests，6 失败：接手前既有的显示名/边标签断言漂移（T01/T02 已记录，未在本票修复、也未计入通过） |
| `test_m2_hca_prod`/`test_m2_swa_hash_prod`/`test_m2_mask_q` | 10 errors，全部为本机缺 onnx/onnxruntime 的环境错误（接手前同样不可运行）；未用桩冒充通过 |
| [verify_structure_candidate.py](T03-verify/verify_structure_candidate.py)（只读静态复核） | **141 PASS / 0 FAIL**：6 个交付文件 SHA、页面 id 唯一/普通锚点均落在层级 section 内/路由与定位实现/Block 参数区稳定 id/四级 section/43 层/5 场景/默认场景/未知项、结构 JSON 每层组一份且无 phases、具体场景无未知、符号场景有原因、因果阈值严格小于且「等于阈值」、四组覆盖 0..42、`min(seqlen,Lk)`、Excel 两表、工具身份含新提取模块/不含退役页面（[verify.log](T03-verify/verify.log)） |
| [browser_nav_check.py](T03-verify/browser_nav_check.py)（**真实 headless Chrome**，注入探针副本，不改候选） | **14 个交互步骤 + 2 个初始深链步骤全通过**：整网→层组（切场景 decode-nonzero=128/符号=未知）→Block→`ops/res_attn` 模块→源码面板展开；参数联查 `#p-block-csa-topk-Nhc` 进入 Block 层、展开 `block-params-csa-topk`、目标可见/高亮；浏览器后退回算子模块；张量联查 `#t-…` 进入算子层并定位；图节点 `#op-…` 展开模块并定位算子卡；回图定位 `#figop-…` 可见；直接 deep-link 参数/tensor 与「以 `#p-…` 初始打开」均正确；返回整网（[browser-nav.json](T03-verify/browser-nav.json)） |
| 截图（headless Chrome 1440×1200） | [整网](T03-verify/screens/network.png) · [层组+场景派生](T03-verify/screens/group-csa-topk.png) · [Block](T03-verify/screens/block-csa-topk.png) · [内部算子](T03-verify/screens/ops-csa-topk.png) |
| 重复构建 | `analysis_id` 不变、`reused: true` |
| 单层入口 | 内存构建成功（`viewer/decoder.html` + `decoder_reading.xlsx` + 阅读 JSON），两阶段来自同一结构投影 |

## 旧断言迁移说明

- `test_hf_decoder_reading`：shape 断言由 `T/P` 迁移为 `seqlen/kvcachelen`；删除
  「SWA/Hash 未实现必须拒绝」的过时断言，改为四类真实层型均可构建 + 越界层拒绝；
  `test_prefill_vs_decode_cache_premise` 改为「场景绑定」测试（两阶段张量/算子/形状完全
  同源，仅 seqlen/kvcachelen 取值不同），并对未知派生量的诚实性做检查。
- `test_hf_network_overview`：`min(T,Lk)` → `min(seqlen,Lk)`；detail 锚点从
  `decoder_variants.html#g-<phase>-<i>` 迁移为 `structure.html#/group/<gid>[...]`；
  原来的 network/variants 双页往返测试改为分层页的四级往返测试。
- `test_m2_toc`：旧侧边目录不再存在；迁移为分层路由/模块锚点「目标唯一、有界、真实存在、
  响应式规则保留」的契约，不保留旧长页断言。
- 首轮复核后新增：锚点归属层/路由实现/目标可见断言（替代只查 id 存在）、阈值边界
  严格性与「等于阈值即屏蔽」门禁、结构构造归 extract（render 不读配置）回归。

## 已知限制与未验证范围

- **静态/浏览器操作不等于用户阅读验收**：headless Chrome 验证了真实加载、层级/锚点
  切换、details 展开、目标滚入与高亮、后退/深链；未验证窄屏/键盘/缩放体验，也未签
  用户可读性。SVG `<a>` 在探针里通过设置同目标 hash 触发导航（SVG 元素无 HTML
  `click()`），页面本身的链接与 `locateInDiagram` 均由真实脚本执行。
- 可选 `--onnx` 路径未在本机运行（无 onnx/onnxsim/onnxruntime）；分层页的 ONNX 联查由
  合成 bundle 的单元测试覆盖，真实 ONNX 重新生成仍需隔离依赖环境。
- Excel 仍是 Prefill/Decode 两表（同一结构的投影），行/列与计算列的统一留给 T04；
  本票只保证两表仍可打开且参数/算子区一致。
- `prefill-empty` 的 `seqlen=8`、`decode-*` 的 4096/4095/4092 是**示例场景取值**，不是
  配置常量；结构 shape 始终保留符号。T04 绑定用户性能配置时可直接替换预设。
- 本票只做 DeepSeek-V4-Flash；其余模型、性能公式/XPU 成本、并行/融合/通信、Profiler、
  Netron GUI、量化执行与数值等价均未涉及。
- 快照后协调文件的外部更新已在「执行基线」说明；本票未改这些文件，也未改自动化。
- 同名 `structure.html`：`outputs/deepseek_v4_flash_0731/viewer/structure.html` 是旧
  reference/DSpark 参考实现页面（未改动）；本票入口在 `..._hf/<analysis_id>/viewer/structure.html`。

## 结论

T03 把 DeepSeek-V4-Flash 四类层组的层内事实收敛为每个层组一份 `hf_layer_structure.v1`
（同一结构 + seqlen/kvcachelen 场景），并交付可离线打开的「整网→层组→Block→算子」
分层页；旧 phase×variant 长页装配与已迁移的重复实现已删除，整网页/模块图/详情卡/源码
面板/可选 ONNX 均复用既有能力。125 项相关测试通过，静态复核 116 项通过，真实 Chrome
完成四级导航/场景/源码展开。等待 Codex 复核；不自动提交/推送、不关闭 issue、不进入 T04。
