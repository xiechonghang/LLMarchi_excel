# output_reference：产物文件与字段说明

适用版本：structure_reader 0.1.0（2026-10-16）。字段描述以实际落盘产物为准：inspect 扫描包（`outputs/_inventory/deepseek_v4_flash_0731/<content_id>/`，schema v2）与 build 阅读产物（`outputs/deepseek_v4_flash_0731/<analysis_id>/`）。后续新增字段时更新本文件。

## 产物布局

```text
outputs/
├── _inventory/<model_key>/<content_id>/   # inspect 的正式扫描包（同输入同目录，幂等复用）
│   ├── run_manifest.json
│   ├── input_manifest.json                # schema: input_manifest_v2
│   ├── capability_report.json             # schema: capability_report_v2
│   └── README.md                          # 本包导航
├── <model_key>/<analysis_id>/             # build 的结构阅读产物（内容寻址，幂等复用）
│   ├── run_manifest.json                  # schema: build_run_manifest_v1
│   ├── README.md                          # 阅读产物导航（四图/场景/边界）
│   ├── viewer/structure.html              # 自包含离线阅读页（唯一完整 HTML）
│   ├── data/
│   │   ├── input_manifest.json            # 同 inspect 输入清单（build 复算）
│   │   ├── layer_plan.json                # 四组层规划 + decode 版本链 + DSpark 读出（有序 concat → stage0 唯一 → main_x 三 stage 共享）
│   │   ├── semantic_graph.json            # 主图（csa_topk 代表模板）
│   │   ├── instance_bindings.json         # 43 层逐实例绑定 + 版本链
│   │   ├── scenarios.json                 # 主图场景实例 + variant_views 四变体场景视图
│   │   ├── variants.json                  # 四变体规格（节点/边计数/代表层/r/门控）
│   │   └── variants/<key>/                # csa_hash / csa_topk / hca_topk / swa_hash
│   │       ├── semantic_graph.json        # 该变体真实图
│   │       ├── scenarios.json             # 该变体场景实例（env/guard/派生/前提）
│   │       ├── connections.csv            # 该变体边表
│   │       └── tensor_axes.csv            # 该变体张量轴表
│   ├── reports/
│   │   ├── reading_guide.md               # 阅读顺序与覆盖边界
│   │   ├── validation_report.json         # 受控错误验证（逐项 detected）
│   │   ├── source_evidence.json           # 合并证据注册表（主图+DSpark 同 ID 一致合并；五图 ev_ 引用闭合）
│   │   └── coverage_report.json           # extracted/pending_extraction/uncovered
│   ├── tables/
│   │   ├── connections.csv                # 主图边表（edge_id 与图 JSON 联查）
│   │   └── tensor_axes.csv                # 主图张量轴表
│   └── logs/build.log
├── _failed/<UTC时间>-<短uuid>/            # failed 运行的具名诊断
│   └── run_manifest.json                  # schema: failed_run_v1
└── _coordination/…                        # 开发协调记录（dispatch/results/evidence/logs），不是业务产物
```

通用约定：所有文件内 path 相对工作区根；产物不含绝对路径、时间戳、解释器安装路径（保证同输入字节级确定）。inspect 的 `content_id` 与 build 的 `analysis_id` 均为完整 64 位 SHA256，其构成见各自 `run_manifest.json` 的 `content_id_basis`。

## inspect 扫描包

### run_manifest.json（schema: run_manifest_v1）

| 字段 | 说明 |
| --- | --- |
| `artifact` | `{kind: input_inventory, stage: S0, note: 不是模型结构图}` |
| `status` / `reviewed` | `success`（reviewed=true，基线全 match）/ `unreviewed`（reviewed=false，缺基线，不表示通过）。failed 不产生本包 |
| `model_key` / `display_model_id` | 模型键与显示名 |
| `content_id` / `content_id_schema` | 完整内容 ID 及其 schema 标签 |
| `content_id_basis` | 身份文档原文：注册 JSON 哈希、inputs（role/path/sha/size）、基线（文件哈希+声明表）、hf_checkout（commit/dirty/dirty_paths/error，已剔除工作区绝对路径）、weights_actual、baselines、tools |
| `files[]` | 四文件清单 + 各自 SHA256；`run_manifest.json` 自身为 null（本文件） |
| `tools` | `structure_reader_version`、`python_version`、影响输出的源码文件 SHA256 |

### input_manifest.json（schema: input_manifest_v2）

| 字段 | 说明 |
| --- | --- |
| `inputs[]` | 每输入 `{role, path, sha256, size_bytes, description}`（含 model_manifest，共 8 项） |
| `revisions.model_manifest` | 清单声明：path/declared_repo_id/declared_revision（声明值，未线上核验） |
| `revisions.hf_checkout` | HF 本地 git 实况：commit/commit_available/dirty/dirty_paths/error |
| `configs.raw_root_config` | 根 config **原样保存**（HF 命名）：`fields`（architectures/attention_bias/hidden_size…）、`hf_consumer`（DeepseekV4Config + 来源）、`naming_convention`、`role` |
| `configs.constructor_state` | 参考实现构造态（44 字段）：`fields[k] = {value, source(dataclass_default|run_config), source_path, line, expression, original_type, order}`；`merge_rule`、`evidence.method`（AST 静态读取，未导入/执行模型代码）、`run_config_keys_not_consumed`、`dataclass_source`、`run_config_path` |
| `configs.entrypoint_overrides` | 入口扫描：`constructor_call`（`ModelArgs(**json.load(f))`，行号）、`config_open_calls`、`entrypoint_parameters`（main 形参默认值，含模型原始 `inference/generate.py` 自带的 `--interactive` 开关——属模型入口参数而非本工具参数）、`overrides[]`（逐条 field/expression/value/is_literal/guards[].condition_source/line/statement） |
| `configs.notes` | 消费规则说明（构造态 = 默认值 + run_config；入口覆盖只记录不并入；路径已规范化） |
| `raw_to_effective_pairs` | `equivalent_pairs[]`（37 条：raw_field/effective_field/raw_value/effective_value/match）+ `non_equivalent_pairs[]`（3 条，含差异说明）+ `notes` |
| `weights` | `actual`（磁盘实况：safetensors_count/shard_pattern_matches/index_present）、`manifest_declared`（清单声明）、`consistency`（status=match/mismatch + note；声明不冒充现存）、`report_basis` |

实测锚点（DeepSeek-V4-Flash-0731）：`constructor_state.fields.max_seq_len = {value: 4096, source: dataclass_default, line: 38}`；`fields.n_layers = {value: 43, source: run_config}`；`entrypoint_overrides.overrides` 含 `max_batch_size=1` 与 `max_seq_len=64*1024`（guards: interactive，L83-85）；`weights` = 磁盘 0 个 safetensors / 索引缺失 / 清单声明 48 / consistency=mismatch。

### capability_report.json（schema: capability_report_v2）

| 字段 | 说明 |
| --- | --- |
| `interpreter` | 真实探测：python_version/implementation/platform（3.12.14/CPython/macOS…） |
| `optional_packages` | torch/safetensors/transformers/openpyxl 的 find_spec 探测 `{available, module_spec}`；available 只表示包在当前解释器**可发现**（find_spec），不证明导入/执行成功；只探测不导入 |
| `core_dependencies` | `{min_python: 3.11, stdlib_only: true, torch_required: false}` |
| `implemented_capabilities` | 当前已实现能力清单（含输入清单/配置消费/基线比对/CLI inspect/build 等；动态清单，以最新扫描包为准） |
| `not_implemented[]` | `{feature, stage, note}` 未实现能力清单（动态，以最新扫描包为准）；清单随阶段推进变化，已交付项不再列出 |
| `environment_note` | 解释器绝对安装路径只在 CLI stdout 出现、不进产物的说明 |

### 扫描包 README.md

包内导航页：内容 ID、模型键、阶段声明（"本目录不是模型结构图"）、审查状态行、阅读顺序（run_manifest → input_manifest → capability_report）、配置消费说明、实际权重状态、"尚无的内容"（无模型图/HTML/CSV，无空占位）、一致性与复现说明。

## build 阅读产物

### run_manifest.json（schema: build_run_manifest_v1）

`status: success`、`artifact: {kind: structure_reading}`、`analysis_id`（内容寻址：输入 SHA + 工具源码 + 场景集）、`content_id_basis`（同 inspect 的 basis 结构 + scenario_set）、`files[]`（全部产物文件与 SHA256，run_manifest 自身 null）、`tools`。

### viewer/structure.html

自包含单文件离线页（file:// 可开，无外网/CDN）：数据以 JSON 内嵌（与 data/、tables/ 同源），JS 只做呈现不派生事实。四图统一切换（工具条"图"按钮：CSA·TopK 主图 / CSA·Hash / HCA·TopK / SWA·Hash）、三视图（模块总览/局部展开/完整图）、场景切换（两套按钮统一入口）、节点/边/模块/投影详情与证据查看。默认适配保证节点标签实际 ≥12px；"定位选中"/"查看全局范围"/"适配视图"三个入口；拖拽平移 + 滚轮缩放（5px 点击阈值）。

### data/scenarios.json（主图场景 + variant_views）

`schema: scenario_instances_v1`。主图 `instances[]`：`scenario_id`、`env`、`guard_activations`（**对象形状** `{guard_id: {state: "true"|"false"|"unknown", reason}}`）、`compression`（压缩路径逐步实例化）、`indexer`（选择边界）、`derived`（N_tokens/N_ctx/N_blk/N_vis/N_sel_c/remainder/decode_trigger）、`history_premise`。`variant_views.<key>[]`：该变体场景视图（**guard_activations 为字符串形状** `{guard_id: "true"|"false"}`，无逐步压缩实例化；derived 字段随变体语义不同，如 SWA 有 N_win/window_*）。页面 JS 对两种形状归一化后呈现，不把字符串形状误判为 unknown。

### data/semantic_graph.json 与 data/variants/<key>/semantic_graph.json

真实有向图：`template_id`、`nodes[]`（stable_id/op_type/端口/证据引用）、`edges[]`（stable_id/src/dst/端口/tensor_id/kind(data|state|select|control)/guard_ref/证据引用）、`tensors[]`（轴定义/逻辑物理布局）、`guards[]`、`state_versions[]`/`state_transitions[]`、`alias_relations[]`（张量别名关系，页面注明非 data 边）、`named_axes[]`。主图 = csa_topk 代表模板（107 节点/173 边）；变体图随变体语义增减（如 SWA 40 节点/51 边，无压缩器/Indexer）。

实测锚点（2026-10-16）：主图 107 节点/173 边（data 156 · state 10 · select 1 · control 6）；csa_hash 107/171（+block.input_ids_src → moe.gate.tid2eid 的 Hash 路由链，-2 条 TopK 边）；hca_topk 69/105（无 Indexer，有固定压缩选择）；swa_hash 40/51（纯滑窗，无压缩区）。四变体共享 12 个场景 ID（prefill_t3/4/5/127/128/129 + decode_p2/3/4/126/127/128）。

### tables/*.csv 与 data/variants/<key>/*.csv

`connections.csv`：edge_id/src_node/src_port/dst_node/dst_port/tensor_id/kind/guard_ref（边 ID 集合与对应图 JSON 精确相等）。`tensor_axes.csv`：张量轴目录（tensor_id/axis_symbol/axis_name/...，轴定义与图 JSON named_axes 联查一致）。变体 CSV 与变体图 JSON 同源。

### reports/

`reading_guide.md`（打开方式、五图/视图/场景/可读性用法、覆盖边界）；`validation_report.json`（受控错误验证：逐变异 detected，`all_detected: true` 才落成功产物）；`dspark_validation.json`（DSpark 集成契约 D1-D15 + 24 项集成受控错误（含 codex_b4_review01 五个原始 mutant 逐字段等价回放），同 fail-fast 策略）；`source_evidence.json`（`schema: source_evidence_v1`，`entries[]` = 合并证据注册表：主图 + DSpark 集成同 ID 逐字段一致合并（冲突具名拒绝），五图全部 `ev_` 证据引用在此闭合，轴引用的 `model.py:NNN` 定位文本不属 registry ID）；`coverage_report.json`（extracted = 四类主干模板 + DSpark 整体集成（`dspark.status: extracted_integrated`）；pending_extraction 为空——DSpark 已整体集成（B4）；uncovered = kernel 数值等）。

## _failed/<run_id>/run_manifest.json（schema: failed_run_v1）

`status: failed`、`run_id`、`recorded_at_utc`、`model_key`、`error: {kind, message, named_structure_reader_error, traceback}`、`tools`。错误 kind 实测例：`UnknownModelModelError`、`ModelConfigNotFoundError`（--workspace-root 指向无注册配置的目录）、`baseline_mismatch`、`baseline_schema_unrecognized`、`output_consistency_conflict`、`build_error`。

## 证据等级

- 文件 SHA256/大小、磁盘权重计数、git commit、解释器/可选包探测：**实测证据**。
- 构造态/入口覆盖来源（行号/表达式）：**AST 静态读取**（未执行模型代码）。
- manifest declared revision、声明文件数：**声明值**（未线上核验），与实测字段分开保存。
- raw↔effective match：扫描时计算的对照结论，原样持久化。
- 图/场景/证据：**源码静态推导**（语义图、guard 谓词、场景派生量均带来源证据引用；cache 实际内容与 kernel 数值 unknown，不冒充已验证）。
