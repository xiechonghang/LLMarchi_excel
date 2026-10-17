# user_guide：怎么运行、打开什么、怎么读结果

适用版本：structure_reader 0.1.0（2026-10-16）。所有行为以下述**实际落盘 CLI**为准；尚未实现的命令一律不写用法。

## 1. 运行环境

- Python ≥ 3.11；核心仅标准库，不需要 torch、不需要权重文件。
- 本机已实测解释器（3.12.14）：

```bash
READER_PYTHON="/Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
```

其他 ≥3.11 解释器可用但未实测；3.9（如系统 `/usr/bin/python3`）会在薄入口处得到明确版本错误并退出 1。

## 2. 命令

```bash
cd /Users/xiechonghang/Workspace/LLMarchi_excel

# 查看帮助（exit 0）
"${READER_PYTHON}" structure_reader/scripts/read_model_structure.py --help
"${READER_PYTHON}" structure_reader/scripts/read_model_structure.py inspect --help
"${READER_PYTHON}" structure_reader/scripts/read_model_structure.py build --help

# 扫描唯一已注册模型（exit 0；本机实测输出 status: success）
"${READER_PYTHON}" structure_reader/scripts/read_model_structure.py inspect --model deepseek_v4_flash_0731

# JSON 摘要（status/reviewed/content_id/output_dir/baseline_statuses/weights）
"${READER_PYTHON}" structure_reader/scripts/read_model_structure.py inspect --model deepseek_v4_flash_0731 --json
```

参数：

| 参数 | 说明 |
| --- | --- |
| `--model`（必选） | 模型键 = `structure_reader/configs/models/<model_key>.json` 文件名 stem |
| `--workspace-root` | 扫描读取的输入与注册所在工作区；缺省从源码位置向上定位，与 cwd 无关（隔离负例测试用） |
| `--json` | stdout 改为 JSON 摘要；不是整份模型数据 |

`validate / render / export-tables` 作为独立子命令尚未实现（它们的能力已包含在 build 内），帮助中不出现单独条目。

## 2b. build：生成结构阅读产物（已实现）

```bash
"${READER_PYTHON}" structure_reader/scripts/read_model_structure.py build \
  --model deepseek_v4_flash_0731 --scenario-set standard

# JSON 摘要（status/analysis_id/output_dir/page/reused）
"${READER_PYTHON}" structure_reader/scripts/read_model_structure.py build \
  --model deepseek_v4_flash_0731 --scenario-set standard --json
```

一条命令连通：输入扫描与固定基线核对 → 四组 classify → 代表图 + 四类变体图提取 → 通用引用与模型契约校验 → 场景实例化 → 四图统一离线页渲染 → 同源 CSV 导出。**成功后 stdout 的 `page` 字段给出唯一完整 HTML 路径**（`outputs/<model_key>/<analysis_id>/viewer/structure.html`），**双击或以 file:// 离线打开**（单文件自包含：无外网、无 CDN、无服务、无依赖）。

产物目录（`outputs/<model_key>/<analysis_id>/`）：`viewer/structure.html`（四图统一阅读页，默认场景 decode_p128）、`data/`（主图/层规划/实例绑定/场景 + `data/variants/<key>/` 每变体语义图/场景/CSV）、`reports/`（阅读指南/校验/证据/覆盖）、`tables/`（connections.csv、tensor_axes.csv，ID 与图 JSON 可联查）、`logs/build.log`。

内容寻址：`analysis_id` 覆盖输入 SHA256、工具源码与场景集；同输入重跑幂等复用（先核对既有文件，被篡改则拒绝覆盖退出 1）；不同输入/工具版本隔离到不同目录。失败写 `outputs/_failed/<run_id>/`，不落正式页。

### 阅读页怎么用（viewer/structure.html）

- **四图切换**：图工具条"图"按钮在 CSA·TopK（主图，layer 4 代表模板）/ CSA·Hash / HCA·TopK / SWA·Hash 间切换；每次切换按所选图重建全部节点/边/模块/场景（空模块不显示，未知分组进具名"其它"）。
- **三视图**：默认"模块总览"（模块盒 + 模块间投影边，投影边保留全部原始边 ID）→ 点击模块进入**局部展开**（该模块真实节点/边的有向图，跨模块边连端口轨道并标注真实外部端点）→ "完整图"（全部节点/边拓扑，列 = 数据流拓扑深度、行 = 语义分区）。
- **场景**：默认 decode_p128；图工具条与下方"场景"面板两套按钮同步（统一入口），guard 三态（true/false/unknown，unknown 不伪作 false）与压缩路径细节随场景变化；模块间投影边 guard 状态不一致时聚合显示 mixed。
- **可读性**：局部展开与完整图默认适配保证节点标签实际 ≥12px（CSS 像素）；横向溢出可拖拽平移（起点从输入侧开始）；"定位选中"把选中节点/边居中；"查看全局范围"显式整体缩小（想一眼看全图时用，可读性让位于范围）；"适配视图"恢复可读默认。放大/缩小按钮与滚轮缩放、拖拽平移（5px 点击阈值）可用。
- **详情**：点击节点/边/模块/投影边在近图面板看真实端口/张量/轴/证据（行号+原文内嵌）；图下方 L1 节点表/场景面板/状态版本/层规划/证据为补充。长标签截短显示，全名在悬停 title 与详情面板保留。
- **选择同步**：切图后场景 ID 有效则保留、节点/边选择存在于新图则保留；失效选择清空并在图上方说明。切场景后已选详情随场景刷新（guard 三态不陈旧）。

边界（诚实声明）：静态结构分析，不执行模型代码；cache 实际内容、每专家 token 数、Indexer 实际可见条目数 unknown；kernel 内部数值、fast_hadamard_transform 未覆盖；四类主干模板已全部提取（四图均可视阅读），DSpark 已整体集成（B4：首入口/公共 stage×3/末出口 head/Markov/Confidence 真实合并图，工具条 DS 按钮查看；DSpark 图下有"实例角色与参数绑定"面板——mtp.0/1/2 实际层 43/44/45、first/last 角色、公共模板参数 → mtp.i 路径/张量联查、唯一 embed/head 共享对象、last 专有 norm/hc_head/Markov/Confidence 参数的 runtime/checkpoint dtype；场景面板含每实例 cache before/after/published 版本绑定，有效槽位数数值与前提分开、前提未满足明确 unknown，覆盖判定按 P+1 而非 P——P=127 时 P+1=128 恰满 W=128；HF 实现无对应 DSpark 类，实际 token/采样/置信度数值 unknown，页面如实标注）。

## 3. 退出码与三种状态

| 退出码 | 含义 |
| --- | --- |
| 0 | `success`（已声明基线全部 match）或 `unreviewed`（缺基线：可输出诊断扫描，但**未审查、不表示通过**） |
| 1 | `failed`：未知模型/缺输入/坏配置（具名错误）、基线 mismatch 或 schema 无法识别、既存产物与本输入冲突（拒绝覆盖）等 |
| 2 | argparse 参数错误（如缺 `--model`） |

本机实测：`inspect --model deepseek_v4_flash_0731` → 0；未知模型 → 1（stdout 给出可用模型列表与 `_failed` 记录路径）；缺 `--model` → 2。

失败时不产生正式扫描包，并在 `structure_reader/outputs/_failed/<UTC时间>-<短uuid>/run_manifest.json` 留具名诊断（错误类别/message，不裸 traceback）。

## 4. 打开什么

### 4a. inspect 扫描包四文件

成功/unreviewed 运行后，stdout 返回 `output_dir`：

```
structure_reader/outputs/_inventory/deepseek_v4_flash_0731/<64位内容ID>/
├── run_manifest.json        # ① 先读：status/reviewed、完整 content_id 及其构成、文件清单与哈希、工具版本
├── input_manifest.json      # ② 再读：输入清单 + 配置消费 + raw↔effective 对照 + 权重实况
├── capability_report.json   # ③ 已实现/未实现能力、解释器与可选包真实探测
└── README.md                # 本扫描包的导航页
```

推荐阅读顺序 = 上表编号（①→②→③），README.md 是入口导航。**不要期待在本目录找到模型结构图**——包内 README 与 capability_report 都明确写有"尚无模型图/HTML/CSV"；要看结构图请跑 `build`（2b 节）。

### input_manifest.json 里怎么找关键事实（v2 schema 实测字段）

- 各输入文件身份：`inputs[]`（role/path/sha256/size/description，path 为工作区相对路径）。
- **根 config 原样**：`configs.raw_root_config`（HF 命名字段如 `hidden_size=4096`、`num_nextn_predict_layers=1`，以及它的 HF 消费方 DeepseekV4Config）。
- **参考实现构造态**：`configs.constructor_state.fields`（44 字段，逐个含 value/source/source_path/行号）。例：`max_seq_len=4096`/`max_batch_size=4` 来自 model.py dataclass 默认值（L38/L37）；`n_layers=43`、`n_mtp_layers=3`、`n_hash_layers=3`、`hc_mult=4` 来自 `inference/config.json`。
- **入口覆盖**：`configs.entrypoint_overrides.overrides[]`——`interactive` 守卫下 `max_batch_size=1`、`max_seq_len=64*1024`（65536），来源/条件/行号在案。这些**不属于构造态**；且 `--interactive` **不是本工具 inspect 的参数**——它是模型原始 `inference/generate.py` 自带 argparse 的开关，只有按参考实现自己的入口运行时才生效，inspect 只静态记录其存在与条件。
- **raw↔effective 对照**：`raw_to_effective_pairs`（37 等价对含 match 结论；3 条非等价对照，含 `num_nextn_predict_layers=1`（根 config）≠`n_mtp_layers=3`（参考实现消费值）这类"名字像、语义不同"的显式说明）。
- **权重实况**：`weights.actual`（磁盘现存 0 个 safetensors、索引缺失）与 `weights.manifest_declared`（清单声明 48 个）分开保存，`weights.consistency.status=mismatch`。**声明不冒充现存**。

### 两份配置消费方的区别（容易读错的地方）

根 HF `config.json` 与参考实现 `inference/config.json` 是**两个消费方**：前者由 HF `DeepseekV4Config` 消费，后者经 `generate.py main()` 的 `ModelArgs(**json.load(f))` 构造参考模型。位置编码上限 `max_position_embeddings=1048576` 是根 config 的 RoPE 位置上限，与构造态 `max_seq_len=4096`（KV 缓存容量）不是同一个数，后者不得用前者代替。

### 4b. build 阅读产物

成功 build 后按 stdout `page` 路径打开 `viewer/structure.html`（用法见 2b 节"阅读页怎么用"）；数据核对用 `data/semantic_graph.json` 与 `tables/*.csv`（同源：全部边 ID 一致）。扫描包四文件（4a 节）描述的是 `inspect` 产物，不是 build 产物。

## 5. 结果可信度标记

- `run_manifest.json` 的 `status`/`reviewed` 字段：`success` = 全部已声明源码基线比对一致（reviewed）；`unreviewed` = 缺基线，**未审查，不表示通过**；`failed` 包不产生。
- 基线（`configs/source_baselines/*.json`）一次固定 SHA256，运行时**不会自动更新**；输入文件变了会得到 mismatch → 退出 1。这是设计行为：防止静默审查已漂移的输入。
- 同输入重跑：内容 ID 相同，逐文件核对后幂等复用（stdout 标 `result: reused`）；既存内容与本输入不符则拒绝覆盖并退出 1。

## 6. 常见问题

- **想看模型结构图/某层怎么连** —— 跑 `build`（2b 节）并打开 `viewer/structure.html`。`inspect` 扫描包只交付输入与配置证据，不要用扫描包代替最终阅读效果。
- **想加一个模型** —— 仅写注册 JSON + 基线**不等于支持**：当前扫描要求该模型已具备消费契约——注册声明的必需角色文件存在、`ModelArgs` 式 dataclass 构造（AST 可读默认值）与 `json→ModelArgs` 入口可识别。已有此契约的模型按 developer_guide §5 注册；其他模型（HF 生态任意模型）需等适配层实现后另行适配，当前不支持。
- **改了被扫描的输入文件** —— 下次 inspect/build 会基线 mismatch 而失败（退出 1），`_failed` 记录里有逐文件对比；这表示输入已偏离上次审查身份，需要走基线更新流程而不是重跑绕过。
- **扫描要不要下载权重/装 torch** —— 都不要。权重只做磁盘存在性计数；torch/safetensors/transformers/openpyxl 只在 capability_report 里报告 find_spec 探测结果（**只证明包在当前解释器可发现，不证明导入/执行成功**），不安装、不导入运行。
- **阅读页要不要联网/装浏览器插件** —— 不要。单文件 HTML 双击即开（file://），无外网、无 CDN、无服务。
