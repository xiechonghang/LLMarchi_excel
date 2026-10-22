> **当前入口（2026-10-22）：** HF decoder 的 HTML、两表 Excel、模块级 ONNX/onnxsim 及本地 Netron 链接已实现，入口为 `build-hf --model deepseek_v4_flash_0731_hf --layer-index 4`。克隆、依赖安装、固定版本输入准备与运行步骤见[根目录 README](../README.md)。M1 整体用户阅读确认仍待完成。下文保留旧 `inspect` / `build` 参考实现与 DSpark 说明，不是当前 HF 路径的快速开始。

# structure_reader

模型特征结构阅读工具。当前状态：**S0 输入扫描 + 结构阅读产物 build 已实现**（`inspect` / `build` 两条命令；四图统一离线阅读页 + DSpark 整体集成图、代表图/变体图数据、场景实例化、同源 CSV 均已交付。DSpark 六组件（Attention/公共 stage×3/首入口/末出口 head/Markov/Confidence）已真实合并为整体集成图并接入 build/阅读页/CSV。阶段全景见 [docs/progress.md](docs/progress.md)）。

## 它做什么 / 不做什么

现在能做：

`inspect`（S0 输入扫描）：

- 扫描已注册模型的输入文件（角色/工作区相对路径/SHA256）；
- 保存根 config 原样（HF 命名）与参考实现构造态（`ModelArgs` dataclass 默认值 + 运行配置覆盖，逐字段含来源与行号）；
- 记录入口覆盖（如模型原始 `inference/generate.py` 里 `--interactive` 开关分支的 `max_seq_len=65536/max_batch_size=1`；该开关属模型入口而非本工具），只记录不并入构造态;
- 报告实际权重现状（磁盘现存 vs 下载清单声明，分别保存）；
- 与固定源码基线比对输入文件身份（缺失 → `unreviewed`，失配 → `failed`）；
- 产出内容寻址的清单目录（同输入同目录、幂等复用）。

`build`（结构阅读产物）：

- 一条命令连通：输入扫描与基线核对 → 层规划 → 代表图 + 四类变体图提取 → 引用/契约校验 → 场景实例化 → 四图统一离线页渲染 → 同源 CSV 导出；
- 产物 `viewer/structure.html` 双击/file:// 即可打开：CSA·TopK（主图）/ CSA·Hash / HCA·TopK / SWA·Hash 四图切换，模块总览 → 局部展开 → 完整图三视图，场景切换（guard 三态/压缩路径），节点/边/模块/投影详情与证据查看；默认适配保证节点标签实际 ≥12px，可拖拽平移/缩放/定位选中/查看全局范围；
- `data/`（图/层规划/实例绑定/场景）、`reports/`（阅读指南/校验/证据/覆盖）、`tables/`（connections.csv、tensor_axes.csv，ID 与图 JSON 可联查）与页面同源。

不能做（尚未实现，命令也不提供）：

- `validate / render / export-tables` 独立子命令（其能力已包含在 build 内）；
- 运行推理/数值等价验证、运行追踪（未排期）；
- 除唯一已注册模型外的其它模型适配（HF 生态任意模型需适配层，见 user_guide 常见问题）。

DSpark 边界：整体集成图为 reference inference 源码静态取证（HF transformers 实现无对应 DSpark 类，named 差异见产物 `data/dspark/integration.json`）；实际 token/采样/置信度数值、kernel 内部数值、cache 实际内容保持 unknown。完整 V2 独立门禁与用户阅读验收待完成。

**扫描输出不等于最终阅读效果**：`inspect` 扫描包是"输入清单+配置消费"证据，不是用户最终阅读的模型结构图；`build` 产物（`viewer/structure.html`）才是供阅读的模型结构页。

## 快速开始

```bash
cd /Users/xiechonghang/Workspace/LLMarchi_excel
READER_PYTHON="/Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"  # 本机已实测的 3.12

"${READER_PYTHON}" structure_reader/scripts/read_model_structure.py --help
"${READER_PYTHON}" structure_reader/scripts/read_model_structure.py inspect --model deepseek_v4_flash_0731
"${READER_PYTHON}" structure_reader/scripts/read_model_structure.py build --model deepseek_v4_flash_0731 --scenario-set standard
# 成功后 stdout 给出唯一完整 HTML 路径（outputs/<model_key>/<analysis_id>/viewer/structure.html），双击离线打开
```

详细用法与产物说明见 [docs/user_guide.md](docs/user_guide.md)。

## 目录职责

| 目录/文件 | 职责 |
| --- | --- |
| `scripts/read_model_structure.py` | 唯一薄用户入口（版本检查后转发到 `src/structure_reader/cli.py`） |
| `src/structure_reader/` | 库本体（扫描/注册/CLI/提取/校验/渲染/导出），仅标准库 |
| `configs/models/` | 模型注册 JSON，**实际被消费的单一数据源**（改这里即改扫描行为） |
| `configs/source_baselines/` | 固定的输入身份基线；一次固定，**运行时不自动更新** |
| `models/`、`hflib/` | 原始模型/HF 输入，**只读** |
| `tests/` | 独立测试（fixtures 与断言属源码资产） |
| `outputs/` | 唯一正式产物根目录（`_inventory/` 扫描包、`<model_key>/<analysis_id>/` 阅读产物、`_failed/` 失败诊断、`_coordination/` 开发协调记录） |
| `.work/` | 可丢弃临时暂存（原子提交前缓冲） |

## 环境

- Python ≥ 3.11；核心仅标准库，**不需要 torch**（扫描不做张量/权重加载）。
- 本机实测解释器为 3.12（路径见快速开始）；其他版本未实测，不宣称兼容。
- 可选包（torch/safetensors/transformers/openpyxl）只做 find_spec 可发现性探测报告（**不证明导入/执行成功**），不安装、不导入。
- 阅读页为单文件 HTML，浏览器直接打开；无服务、无 CDN、无新依赖。

## 文档

- [docs/user_guide.md](docs/user_guide.md) — 怎么运行、打开什么、怎么读结果（含阅读页使用）
- [docs/output_reference.md](docs/output_reference.md) — 产物文件与字段说明
- [docs/developer_guide.md](docs/developer_guide.md) — 开发者适配/校验/维护
- [docs/progress.md](docs/progress.md) — 阶段状态（由验收流程维护）
