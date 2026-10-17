# developer_guide：开发约定、适配与维护

适用版本：structure_reader 0.1.0（2026-10-16）。面向在本工程内继续开发/适配的开发者（Claude CLI 开发、DeepSeek Harness 独立测试、Codex 最终复验的协作分工见 `hf_docs/development_workflow.md`）。

## 1. 目录与角色边界

| 路径 | 角色 | 修改约束 |
| --- | --- | --- |
| `scripts/read_model_structure.py` | 唯一薄用户入口：Python≥3.11 检查 → `../src` 加入 sys.path → `cli.main`；仅标准库 | 保持"薄"；不要在入口加业务逻辑 |
| `src/structure_reader/model_registry.py` | 注册消费：读取 `configs/models/<model_key>.json`（单一数据源）；具名异常 `ModelConfigNotFoundError/ModelConfigParseError/ModelConfigSchemaError` | 注册 JSON 的 schema 校验在这里 |
| `src/structure_reader/inventory.py` | 输入扫描：`build_inventory(model_key, workspace_root=None)`；AST 读 dataclass 默认值（不导入/不执行模型代码） | 扫描逻辑；异常一律具名（`StructureReaderError` 子类） |
| `src/structure_reader/cli.py` | argparse + 产物落盘：内容寻址、`.work` 暂存原子提交、幂等复用、`_failed` 记录；`_build_documents` 生成 build 全部交付文件（含候选 README/reading_guide） | 退出码契约：0=success/unreviewed，1=failed，2=参数错误 |
| `src/structure_reader/build_pipeline.py` | build 编排：inspect 基线核对 → classify → extract → validate → 场景实例化 → analysis_id（输入 SHA + 工具源码 + 场景集） | 工具源码清单 `_tool_sources_hash` 必须覆盖全部影响产物的源码（新渲染资产必须加入） |
| `src/structure_reader/render/structure_page.py` | 单文件离线页模板（数据 JSON 内嵌，JS 只呈现不派生事实）：四图统一切换/三视图/场景/详情/可读性 | 页面守卫（守恒断言/分层完成性）失败必须显式报错，不静默 |
| `src/structure_reader/extract/` | 语义图/证据/变体提取（源码静态推导，带证据引用） | 只读输入（models/hflib/hf_docs）；不执行模型代码 |
| `src/structure_reader/checks/` | 图引用校验 + 模型契约 + 受控错误验证 | 任何漏检按失败处理（不落成功产物） |
| `configs/models/*.json` | 模型注册（实际被消费的单一数据源） | 改注册即改扫描行为 |
| `configs/source_baselines/*.json` | 输入身份基线（SHA256 一次固定） | **代码与 CLI 都不得自动更新**；更新须走独立开发单 |
| `models/`、`hflib/`、`hf_docs/` | 原始输入与计划文档 | **只读**。任何开发/测试/负例不得写这些路径 |
| `tests/` | 独立测试（fixtures 属源码资产） | 测试与产品代码分离；Harness 原样复跑 |
| `outputs/` | 唯一正式产物根 | `_inventory/`（扫描包）、`<model_key>/<analysis_id>/`（阅读产物）、`_failed/`（诊断）、`_coordination/`（开发协调） |
| `.work/` | 可丢弃暂存 | 不得作为输入或交付；`os.replace` 原子提交后即清 |

多终端协作时的硬规则：**只改自己开发单授权的文件**。其他终端可能同时落盘 `src/`、`cli.py` 或注册 JSON——触碰同文件必须走单据协调，不抢写。

## 2. 输入只读规则（务必保留的提醒）

2026-10-12 第二轮曾发生测试隔离险情，此后确立的守卫规则（已固化在 `tests/_support.py`）：

- 负例测试只作用于**隔离副本**（`models/` 为 `shutil.copytree` 实体副本；`hflib/` 以符号链接只读引用，且任何测试不写 hflib 路径）。
- 写/删辅助函数 `safe_unlink` / `safe_write_text` 先调 `assert_outside_repo`：`resolve()` 后落在真实工作区内的路径直接断言失败，副本路径放行。
- 运行前后对真实输入做 SHA256 快照比对（无新增/删除/改动才允许收尾）。

新增任何测试或负例时沿用这套守卫；不要为图省事直接在真实 `models/`、`hflib/` 上做删除/改写实验。

## 3. 内容寻址产物契约

- inspect 的 `content_id` = 对规范化身份文档（`run_manifest.json` 的 `content_id_basis`）的完整 SHA256：注册 JSON 哈希、各输入 role/path/sha/size、基线文件哈希+声明、HF checkout（commit/dirty/error，剔除 workspace 绝对路径前缀）、磁盘权重实况、环境字段（Python 版本/实现/平台、optional_packages find_spec 探测结果）、工具源码哈希与版本。
- build 的 `analysis_id` = SHA256(输入 SHA + `_tool_sources_hash` + 场景集)；**新增渲染资产/影响产物的源码必须加入 `_tool_sources_hash` 清单**，否则同输入重跑会伪复用旧产物。
- **不得进入** content_id/analysis_id：绝对工作区路径、时间戳、解释器安装路径等偶然字段（解释器绝对路径只允许出现在 CLI stdout 摘要）。
- 同输入 → 同目录。重跑先逐文件核对 SHA256：一致则幂等复用（`reused`），不一致（篡改/不完整）抛 `output_consistency_conflict` → 退出 1 + `_failed` 记录，**拒绝静默覆盖**。
- 写入路径：`<project>/.work/` 临时目录 → 成功后 `os.replace` 原子提交。
- 换解释器/平台会生成新 content_id（有意设计：避免同 ID 不同报告）。

## 4. 状态与失败语义

| status | 条件 | 退出码 |
| --- | --- | --- |
| `success` | 已声明基线全部 match | 0（reviewed=true） |
| `unreviewed` | 声明了基线但有 missing，或完全未声明 | 0（reviewed=false；**不得标 pass**） |
| `failed` | 未知模型/缺输入/坏配置（具名 `StructureReaderError`）、基线 mismatch 或 schema_unrecognized、产物一致性冲突、build 校验/验证失败 | 1 + `_failed/<run_id>/run_manifest.json`（failed_run_v1：错误类别/message/traceback/工具身份） |

- 基线 mismatch → `baseline_mismatch`；基线顶层为非对象 JSON → `baseline_schema_unrecognized`。
- 注册 JSON 缺必需消费角色 → 具名 `InvalidConfigError`，不再裸 KeyError。
- build 内部失败（图契约/受控错误/渲染守卫）→ `build_error`；受控错误未全部检出**不落成功报告**。
- 错误信息必须具名 + 可读 message；traceback 只进 `_failed` 记录，不上 stdout。

## 5. 适配新模型的步骤（S0 范围内）

1. `configs/models/<model_key>.json`：填 `inputs[]`（7 个角色：root_config/run_config/reference_model_source/reference_kernel_source/reference_entrypoint_source/hf_modeling_source/hf_configuration_source）+ `constructor`（dataclass 符号/来源/入口）+ `weights` + `raw_to_effective_pairs`。
   前提：该模型须已具备本扫描的消费契约——必需角色文件存在、`ModelArgs` 式 dataclass（AST 可读字面量默认值）、`json→ModelArgs` 式入口可识别。仅写注册 JSON + 基线**不构成支持**；HF 生态其他架构的模型需等适配层实现后另行适配。
2. `configs/source_baselines/<model_key>_reference.json`（等价对）与 `_hf_comparison.json`：真实复算 SHA256 一次固定；`source_files` 为列表（path/sha256/role/purpose/symbol）。
3. 真实跑 `inspect --model <新键>`：先得 `unreviewed`（若基线名未挂进注册的 `source_baselines` 列表）或 `success`（声明且全 match）；验证构造态来源字段与行号正确。
4. 全程不改 `models/`、`hflib/` 原始输入；不自动生成/更新基线。

## 6. 测试与验收流程

- 全量回归：`python3 -m unittest discover -s tests`（当前 1235 项，含 113 项 DSpark 回归与页面 JS 执行测试）。页面 JS 测试（test_page_js_execution / test_page_topology_r1/r2a/r2b）通过 Node VM + DOM 桩执行**实际生成 HTML** 的脚本——这是 JS 单元执行，**不是浏览器/CSS/真实 pointer 硬件验收**；桩视口固定 1000×640，CSS 视觉与真实交互由真实浏览器核验（接续门禁），测试如实声明此局限。
- 测试执行方：DeepSeek Harness 独立运行；开发者自检**不算独立验收**。
- 开发/测试隔离：隔离副本 + `assert_outside_repo` 守卫 + 前后快照（见第 2 节）。
- 阶段状态只由实测 + 实际产物决定（`docs/progress.md` 维护规则）；任何单份开发报告不得自评阶段 passed。

## 7. 已知限制

- DSpark：六组件（`deepseek_v4_dspark.py` Attention / `dspark_stage.py` 公共 stage×3 / `dspark_endpoints.py` 首入口 / `dspark_exit.py` 末出口 head / `dspark_markov.py` Markov / `dspark_confidence.py` Confidence）已由 `dspark_integration.py` 真实合并为整体集成图（B4），`checks/dspark_contracts.py` 承载集成契约与 12 项受控错误，build/render/CSV/场景状态全部接入；kernel 数值与实际采样值 unknown 边界保持。
- 未实现：`validate / render / export-tables` 独立子命令（能力已含在 build）、运行追踪与数值等价验证（未排期）、Skill（S8）。
- 页面 JS 测试是 DOM 桩执行：不核验 CSS 视觉、字体渲染、真实浏览器交互；"用户已阅读通过"只能由真实浏览器/独立阅读包门禁给出，测试绿灯不等于用户验收。
- `run_manifest.json` 自身 SHA256 在文件列表中为 null（本文件）；其余文件哈希在案。
- 线上 revision 一致性未核验（manifest 声明与本地内容哈希分开保存）。
- 非 3.12 解释器未实测（薄入口只检查 ≥3.11）。
