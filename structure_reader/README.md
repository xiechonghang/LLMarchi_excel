> **五模型当前交付索引（2026-10-29，T09）：** 五个 HF 模型键共用同一入口 `build-hf --model <model_key> --all-variants --json`；当前候选、真实产物/SHA、官方来源、覆盖范围与显式拒绝入口见 [docs/model_delivery_index.md](docs/model_delivery_index.md)。索引绑定内容寻址候选（输入+工具源码 SHA），不是 HEAD 提交身份；T05–T08 历史验收候选保留，不冒充当前实现。
>
> **当前入口（2026-10-29，T08）：** 已注册五个 HF 模型键：`deepseek_v4_flash_0731_hf`、`glm_5_3_hf`（GLM-5.3）、`glm_5_3_flash_hf`（GLM-5.3-Flash）、`qwen_3_8_flash_next_hf`（Qwen3.8-Flash-Next）与 `hidream_o1_image_hf`（HiDream-O1-Image；后四者只提供分层入口）。HiDream-O1-Image 构建：`build-hf --model hidream_o1_image_hf --all-variants --json` → `outputs/hidream_o1_image_hf/<analysis_id>/viewer/structure.html` + **单 sheet**（Generation）工作簿；只分析一次网络 forward（文本 + 像素 patch + 时间步 → x_pred），50/28 步采样/scheduler/CFG 在外部不分析；显式 `--onnx` 或缺少 `--all-variants` 会具名拒绝。Qwen3.8-Flash-Next 构建：`build-hf --model qwen_3_8_flash_next_hf --all-variants --json` → `outputs/qwen_3_8_flash_next_hf/<analysis_id>/viewer/structure.html` + 两页三组工作簿；来源为官方 `Qwen/Qwen3.8-Flash-Next` revision 与本地官方 transformers `qwen4_exp` 实现，只做 text_config 语言主干，视觉/MTP 执行与单层入口不在范围；显式 `--onnx` 会具名拒绝。GLM-5.3-Flash 构建：`build-hf --model glm_5_3_flash_hf --all-variants --json` → `outputs/glm_5_3_flash_hf/<analysis_id>/viewer/structure.html` + 两页三组工作簿；来源为官方 `zai-org/GLM-5.3-Flash` revision 与本地官方 transformers `glm5_next` 实现，只做 text_config 语言主干，视觉/MTP/FP8 执行与单层入口不在范围；显式 `--onnx` 会具名拒绝。GLM-5.3 构建：`build-hf --model glm_5_3_hf --all-variants --json` → `outputs/glm_5_3_hf/<analysis_id>/viewer/structure.html` + 两页三组工作簿。
>
> **T03 入口说明：** HF 阅读入口为分层结构浏览 `build-hf --model deepseek_v4_flash_0731_hf --all-variants`，产物 `viewer/structure.html`（整网→层组→Block→内部算子，默认只显示当前层级；同一份参数化结构用 seqlen/kvcachelen 场景表达不同调用），同候选另有两表 Excel、`data/layer_structures.json`/`data/network_overview.json`、可选模块级 ONNX/onnxsim 与本地 Netron 链接；单层入口 `--layer-index 4` 仍保留。克隆、依赖安装、固定版本输入准备与运行步骤见[根目录 README](../README.md)。M1 整体用户阅读确认仍待完成——静态工程通过不代签用户阅读验收。下文保留旧 `inspect` / `build` 参考实现与 DSpark 说明，不是当前 HF 路径的快速开始。

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
- 产物 `viewer/structure.html`（旧 reference/DSpark 模型键 deepseek_v4_flash_0731；HF 阅读另有 <model_key>_hf 下的同名分层页）双击/file:// 即可打开：CSA·TopK（主图）/ CSA·Hash / HCA·TopK / SWA·Hash 四图切换，模块总览 → 局部展开 → 完整图三视图，场景切换（guard 三态/压缩路径），节点/边/模块/投影详情与证据查看；默认适配保证节点标签实际 ≥12px，可拖拽平移/缩放/定位选中/查看全局范围；
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
# HF 分层浏览入口见下文「分层结构浏览入口」（outputs/deepseek_v4_flash_0731_hf/<analysis_id>/viewer/structure.html）
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


## 分层结构浏览入口（整网 → 层组 → Block → 算子；可选 ONNX/Netron）

```bash
python -c "import sys; from structure_reader.cli import main; sys.exit(main([
  'build-hf','--model','deepseek_v4_flash_0731_hf','--all-variants','--onnx','--json']))"
```

产物：`outputs/deepseek_v4_flash_0731_hf/<analysis_id>/viewer/structure.html`
（单文件离线页：默认只显示当前层级；整网 → 四类层组 → Block → 内部算子逐级进入，
面包屑/返回链接回上级；参数、权重、源码面板按需展开）、
`data/layer_structures.json`（每个层组一份参数化 Block 结构 + 场景派生）、
`data/network_overview.json`（token→logits 整网路径）、两页
`tables/decoder_performance.xlsx`（正式性能表：已确认的共用模板布局，Prefill/Decode
两页 + 模板映射/参数/项分解/原始参数辅助表；与旧表同源但内容身份不同）、两页
`tables/decoder_variants.xlsx`（兼容阅读表，旧导出器生成、字节保持，Prefill/Decode
各含四组，同一份结构的场景投影），逐组
`onnx/<group>/<phase>/<module>/{model.onnx,model.sim.onnx,mapping.json,state.json}`。
需项目内隔离 ONNX 依赖（onnx/onnxsim/onnxruntime）；本地 Netron 链接由已注册
`llmarchi-netron` 启动器解析。图/ONNX 为 FP32 阅读特化、无真实权重，非 HF 数值等价。
性能表导出器与布局源码进入 `reports/source_manifest.json` 的 `tool_sources` 与
`analysis_id`；缺 `openpyxl`、新表丢失/损坏都具名失败，不降级 CSV。

Excel 每页顶部是可填写的性能配置（seqlen、kvcachelen、batch size、计算精度、
权重/激活存储位宽；示例 Prefill(8,0)、Decode(1,4096) 只是可改取值）。派生参数
（L_kv、窗口/压缩条目、阈值等）写成指向这些输入/本组参数单元格的 Excel 活公式，
并带 `IF`/`ISNUMBER` 守卫：改输入后由 Excel 重算；空白/非法输入显示“未知（原因）”，
恢复或补填后回到数值，不默认 0、不折成静态文本。算子区保留符号 shape、模块路径、
真实层组循环次数与参数联查，另提供“实际输入/输出/权重 shape”公式列（如
`B×seqlen` 在 B=2、seqlen=4 时为 8）；未知/歧义/超白名单轴写原因，不猜值。
用户性能公式与 XPU 能力尚未提供：成本列写“待提供”，不生成猜测耗时、不把缺失
成本记 0；整网串行汇总留待后续任务。

### GLM-5.3（`glm_5_3_hf`，T05）

```bash
PYTHONPATH=structure_reader/src python structure_reader/scripts/read_model_structure.py \
  build-hf --model glm_5_3_hf --all-variants --json
```

产物：`outputs/glm_5_3_hf/<analysis_id>/viewer/structure.html`（整网 → 三组层型 →
Block → 内部算子）、`data/layer_structures.json`、`data/network_overview.json`、
`data/effective_config.json`、两页 `tables/decoder_variants.xlsx`（Prefill/Decode 各含
三组）。三组是真实层型组合：Full DSA Indexer + Dense MLP（0,1,2）、Shared DSA
Indexer + TopK MoE（57 层）、Full DSA Indexer + TopK MoE（18 层），共 78 层；组内
只保留代表层模板，循环次数来自真实层索引。

模型事实与边界：官方仓库 `zai-org/GLM-5.3` revision
`ba32f8c19c8738e2c57c71798c601d477a7bf844`（config `model_type=glm_moe_dsa`），
实现取本地官方 transformers 检出 `glm_moe_dsa`；`head_dim` 在 `__post_init__` 被指向
`qk_rope_head_dim=64`（根配置保存的 192 不参与 RoPE）。KV 缓存存 512+64 压缩 latent，
用时才 `kv_b_proj` 展开；full indexer 层有 `[B,L,128]` 键缓存，shared 层复用上一层
top-k。MTP 层 78、视觉/图像路径、FP8 量化执行不分析；GLM-5.3 未实现单层长页与可选
ONNX，显式请求会具名拒绝（不虚构支持）。性能公式与 XPU 仍待提供，成本列“待提供”。

场景参数：`seqlen`＝本次新 token 数，`kvcachelen`＝调用前逻辑历史 token 数
（空缓存 Prefill 为 0）。结构 shape 始终保留符号；页面场景按钮只绑定取值并展示
按固定配置与「空态起步、连续正常执行」前提推导的窗口/缓存规模，未知项如实标注。


### GLM-5.3-Flash（`glm_5_3_flash_hf`，T06）

```bash
PYTHONPATH=structure_reader/src python structure_reader/scripts/read_model_structure.py \
  build-hf --model glm_5_3_flash_hf --all-variants --json
```

产物：`outputs/glm_5_3_flash_hf/<analysis_id>/viewer/structure.html`（整网 → 三组层型 →
Block → 内部算子）、`data/layer_structures.json`、`data/network_overview.json`、
`data/effective_config.json`、两页 `tables/decoder_variants.xlsx`（Prefill/Decode 各含
三组）。三组是真实层型组合：KDA 线性注意力 + Dense MLP（0,1,2）、MLA/DSA 稀疏注意力 +
TopK MoE（3,7,…,43，11 层）、KDA 线性注意力 + TopK MoE（其余 31 层），共 45 层；组内
只保留代表层模板，循环次数来自真实层索引。

模型事实与边界：官方仓库 `zai-org/GLM-5.3-Flash` revision
`46a050bec13002ad29ee795c16e61cf4fa3c4a9c`（config `model_type=glm5_next`，
`text_config.model_type=glm5_next_text`；`pipeline_tag=image-text-to-text`），
实现取本地官方 transformers 检出 `glm5_next`（5.19.0.dev0；config 保存版本 5.16.0，
本票只表达本地代码的压缩 latent/NoPE/mHC 语义，不混用 5.16 的展开 K/V 缓存）。
KDA 层缓存固定大小的 conv state（`[B,24576,4]`）与 recurrent state
（`[B,64,128,128]`），不随 kvcachelen 增长；DSA 层缓存 512 维压缩 latent（NoPE，
values 槽 0 维）与 indexer packed `[B,L,257]`，用时 `kv_b_proj` 展开全部 Lkv。
DSA indexer 用 k-pool=4 压缩池 + 池 top-k + 当前不完整池 tail，`K_sel =
min(Lidx, 2048 + Lidx % 4)`，不是固定 2048；稀疏 mask 由 selected counts scatter
得到（True=可见）。每层两处 4 路 mHC 的 post/comb 混合不是普通残差；MoE 288 选 8
+ 共享专家，稠密/专家/共享 MLP 都有 SwiGLU clamp。MTP 层 45、视觉 24 层、FP8 量化
执行不分析；GLM-5.3-Flash 未实现单层长页与可选 ONNX，显式请求会具名拒绝。性能公式
与 XPU 仍待提供，成本列“待提供”。

### Qwen3.8-Flash-Next（`qwen_3_8_flash_next_hf`，T07）

```bash
PYTHONPATH=structure_reader/src python structure_reader/scripts/read_model_structure.py \
  build-hf --model qwen_3_8_flash_next_hf --all-variants --json
```

产物：`outputs/qwen_3_8_flash_next_hf/<analysis_id>/viewer/structure.html`（整网 → 三组
层型 → Block → 算子，默认只显示当前层级）+ `tables/decoder_variants.xlsx`（Prefill/
Decode 两页三组）+ `data/layer_structures.json`、`data/network_overview.json`、
`data/effective_config.json`、`reports/source_manifest.json` 与 `run_manifest.json`。

模型事实与边界：官方仓库 `Qwen/Qwen3.8-Flash-Next` revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`，`model_type=qwen4_exp`
（`text_config=qwen4_exp_text`）；实现取本地官方 transformers 检出 `qwen4_exp`
（5.19.0.dev0；该模块首次随 v5.17.0 发布，config 记录的 5.8.0.dev0 中不存在）。
48 层 = 12×(3×Gated DeltaNet + 1×Qwen Sparse Attention)；GDN 16 QK/48 V 头×128、
conv kernel 4、固定 recurrent state [B,48,128,128]；QSA 24 Q/2 KV 头×256、partial
RoPE 64/256、indexer 4 Q/1 KV 头×128、4 token/micro-block、每 query ≤512 块 + tail；
4 路 Gated Residual（读门逐元素 [4,H]、写门每支路标量 [4]），网络末端 mixer 加权 mean
且没有 Final RMSNorm；layer 1 有 N-gram PLE（16 头×160 维、上下文 [B,2]、dilation-3
短卷积 state 9）；MoE 512 选 10（softmax、无 correction/scale/clamp）+ 共享专家
sigmoid 标量门。三组真实层型 `gdn-moe`(35)、`gdn-ple-moe`(1)、`qsa-moe`(12) 恰好
覆盖 48 层。视觉/图像输入路径、MTP 执行、单层长页与可选 ONNX 未实现；显式 `--onnx`
或缺少 `--all-variants` 会具名拒绝。来源漂移（config/源码 SHA）按指纹具名拒绝。

### HiDream-O1-Image（`hidream_o1_image_hf`，T08，单次 forward）

```bash
PYTHONPATH=structure_reader/src python structure_reader/scripts/read_model_structure.py \
  build-hf --model hidream_o1_image_hf --all-variants --json
```

产物：`outputs/hidream_o1_image_hf/<analysis_id>/viewer/structure.html`（整网 → 两组
重复层组 → Block → 算子，默认只显示当前层级）+ `tables/decoder_variants.xlsx`
（**单 sheet** `Generation`；图像尺寸/prompt/参考图 patch/B/精度/位宽可填写）+
`data/layer_structures.json`、`data/network_overview.json`、
`data/effective_config.json`、`reports/source_manifest.json` 与 `run_manifest.json`。

模型事实与边界：官方仓库 `HiDream-ai/HiDream-O1-Image` revision
`0b0901d99f200389e138c61946af1185f5f49a13`，`model_type=qwen3_vl`
（`architectures=["Qwen3VLForConditionalGeneration"]`）；实现取官方 GitHub commit
`2c2d29ff729e48f33e41f49edfdbd81d5ac103b4` 的 `models/qwen3_vl_transformers.py` /
`pipeline.py` / `utils.py`。它是 Pixel-level Unified Transformer 单次生成网络：
文本 token（chat template + boi + 1×tms）与 32×32×3 像素 patch 在同一个 36 层
decoder 序列里；时间步经正弦 256 + 2 层 MLP 后**写进 tms token 的位置**（不是
adaLN）；注意力是同一层内的行级差异（AR 文本行只对 AR 文本因果，生成 token 行对
全部双向；flash 两趟 + index_copy，eager 4D mask 等价）；输出 `final_layer2` 的
`x_pred`（每生成 token 32×32×3），**不经过 lm_head**；视觉塔 27 个 ViT block 只在
参考图/编辑/IP 任务（K>0）执行，并把 block 8/16/24 的 DeepStack 特征注入 decoder
前 3 层；单次 forward 没有 KV cache。50/28 步采样循环、scheduler（FlowUniPC/
FlowMatch/Flash）、CFG 两次调用、噪声注入与像素反拼都在外部，不乘步数、不算完整
出图耗时；单层长页与可选 ONNX 未实现，显式 `--onnx` 或缺少 `--all-variants` 会具名
拒绝。来源漂移（config/源码 SHA）按指纹具名拒绝。
