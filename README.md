# LLMarchi_excel — 模型结构阅读与理论性能分析

依据固定版本的模型源码与配置进行静态结构分析，生成离线 HTML 分层阅读页、算子依赖与张量形状、源码定位及 Excel 工作簿。当前已适配 DeepSeek-V4-Flash-0731、GLM-5.3、GLM-5.3-Flash、Qwen3.8-Flash-Next 语言主干和 HiDream-O1-Image 单次前向网络；主入口为 `build-hf --all-variants`。

工具不加载真实权重、不实例化模型。理论性能估算依赖分析场景、硬件能力与公式假设，不代表实测性能；可选 ONNX 是依据源码与配置构建的结构图，也不代表已验证 HF 数值等价。

## 当前能力与验收状态（2026-11-04）

- **DeepSeek 阅读改善已整体验收**：整网 → 层组 → 模块关系 → 内部算子，覆盖输入/输出端点、mHC、Attention、Q/KV/O、三类缓存、压缩器、Indexer 和 MoE 路由/专家/汇合；最终冻结版为 67 张图、71 个图位。
- **DeepSeek 正式性能表已接入**：分层构建同时生成新布局 `decoder_performance.xlsx` 与兼容阅读表 `decoder_variants.xlsx`，共用本次结构数据。新布局目前仅接入 DeepSeek。
- **其余四模型已有结构适配，阅读改善正在逐个推广**：GLM-5.3 已生成新候选，待技术复核与用户视觉审查；GLM-5.3-Flash、Qwen3.8-Flash-Next、HiDream-O1-Image 的本轮阅读改善尚未实施。已注册模型不等于已完成同一轮阅读验收。
- 语言模型使用 Prefill/Decode 两个阶段，`seqlen` 表示本次新 token 数，`kvcachelen` 表示调用前逻辑历史长度；HiDream 使用单次 forward 与 `Generation` 工作表，不套用语言模型的阶段或 KV cache。

交付身份与范围见 [DeepSeek 最终验收](docs/tasks/deepseek-structure-reading-2026-11-01/D05-acceptance.md)、[性能表集成复核](docs/tasks/excel-performance-style-2026-11-03/E03-codex-review.md)及[四模型阅读推广计划](https://github.com/xiechonghang/LLMarchi_excel/issues/24)。历史冻结版与当前源码重建的候选应分别识别；共享源码变化后，重建可能产生新的内容身份。

## 克隆与安装

需要 Git、Python 3.11 或更高版本。以下命令从仓库根目录运行；Windows 可使用 Git Bash，或将虚拟环境激活命令改为 `.venv\Scripts\activate`。

```bash
git clone https://github.com/xiechonghang/LLMarchi_excel.git
cd LLMarchi_excel
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e './structure_reader[xlsx]'
python structure_reader/scripts/read_model_structure.py --help
```

## 准备分析输入（不下载权重）

第三方源码和模型资料不打包进本仓库。为保证源码行号与分析结论一致，当前适配固定到以下版本，不能直接换成上游最新版：

```bash
mkdir -p hflib
git clone --filter=blob:none --no-checkout https://github.com/huggingface/transformers.git hflib/transformers
git -C hflib/transformers checkout --detach 536ecc007387a50e77603bb5d92100e9b07514cc

python scripts/download_hf_model.py \
  -m deepseek-ai/DeepSeek-V4-Flash-0731 \
  --revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062 \
  --endpoint https://huggingface.co
```

下载脚本默认跳过权重，请不要加 `-w`。Transformers 检出仅作为文本输入，不需要安装 Transformers、PyTorch 或模型运行环境。上述 DeepSeek 适配消费根 `config.json` 和六个 Transformers 源文件，并校验其 SHA256；来源缺失或漂移会明确报错，不会沿用旧结论。

## 生成 HTML 与 Excel

用分层入口生成 DeepSeek 浏览页与两张 Excel：

```bash
python structure_reader/scripts/read_model_structure.py build-hf \
  --model deepseek_v4_flash_0731_hf --all-variants
```

命令会打印本次输出目录、HTML 入口与每张表的绝对路径。主要结果位于 `structure_reader/outputs/deepseek_v4_flash_0731_hf/<analysis_id>/`：

- `viewer/structure.html`：分层浏览入口（整网 → 层组 → Block → 算子）。
- `tables/decoder_performance.xlsx`：正式性能表，采用已确认的共用模板布局（Prefill/Decode 两页，参数层校验与活公式；共用 mHC、Q/KV 前置、O/残差按模板只展示一次并乘真实使用次数）。
- `tables/decoder_variants.xlsx`：兼容阅读表，由旧导出器生成、字节保持不变；与性能表内容身份不同，两表并存。
- `data/layer_structures.json`、`data/network_overview.json`：两表共用的同一份结构与整网路径。
- `reports/source_manifest.json`、`run_manifest.json`：输入、工具源码（含性能表导出器）与交付文件身份。

### 生成完整分层图库

DeepSeek 已验收的阅读改善通过 `--archify-layers` 启用；省略此参数仍可生成基础分层页和 Excel，但不包含这套完整图库。

```bash
python structure_reader/scripts/read_model_structure.py build-hf \
  --model deepseek_v4_flash_0731_hf --all-variants --archify-layers --json
```

此模式额外需要 Node.js 和可用的 Archify CLI。工具按 `ARCHIFY_CLI` 环境变量、`structure_reader/configs/tools/archify.json` 的 `cli` 字段、常规技能安装位置依次发现 `bin/archify.mjs`；缺失时会明确报错。图页面位于 `viewer/archify/`，图事实与关系位于 `data/archify/`，由 `viewer/structure.html` 统一导航。

五模型使用相同的分层命令，将 `--model` 替换为以下键；各模型需要准备各自固定版本的输入，来源与范围见[模型说明](structure_reader/README.md#各模型来源与覆盖范围)。

- DeepSeek-V4-Flash-0731：`deepseek_v4_flash_0731_hf`
- GLM-5.3：`glm_5_3_hf`
- GLM-5.3-Flash：`glm_5_3_flash_hf`
- Qwen3.8-Flash-Next：`qwen_3_8_flash_next_hf`（语言主干）
- HiDream-O1-Image：`hidream_o1_image_hf`（单次 forward）

后四模型只提供 `--all-variants` 入口，不支持单层入口或 `--onnx`；其现有工作簿为 `decoder_variants.xlsx`，本轮未推广 DeepSeek 新性能表布局。

单层旧入口 `build-hf --model deepseek_v4_flash_0731_hf --layer-index 4` 仍保留，产出 `viewer/decoder.html` 与 `tables/decoder_reading.xlsx`；它是旧阅读接口，不生成新的性能表。

相同输入、工具源码与依赖生成相同内容身份；已有结果会复用，旧候选不会覆盖。新表丢失或损坏时拒绝复用并具名报错，缺 `openpyxl` 时不会降级为 CSV 或悄悄少交付一张表。

## 可选：ONNX 与 Netron

在上述虚拟环境安装 ONNX 工具，再增加 `--onnx`：

```bash
python -m pip install onnx onnxsim onnxruntime
python structure_reader/scripts/read_model_structure.py build-hf \
  --model deepseek_v4_flash_0731_hf --layer-index 4 --onnx
```

输出包含两阶段共 36 个模块的原始和简化 ONNX、名称映射及检查报告；依赖版本参与输出身份。可以在本机 Netron 中直接选择生成的 `.onnx` 文件打开。

HTML 的一键打开本地 Netron 链接需要注册项目启动器，当前仅支持 macOS。安装 Netron 与 Xcode Command Line Tools 后，在仓库根执行：

```bash
bash structure_reader/tools/netron_launcher/setup.sh register --yes
```

这会构建项目内启动器并注册 `llmarchi-netron` URL 协议。浏览器可能询问是否允许打开本地应用；未注册启动器、其他系统或浏览器不支持此协议时，请从 Netron 手动打开 ONNX。HTML/Excel 的生成与阅读不依赖启动器。

## 开发与项目资料

源码、配置、测试与文档位于仓库中；本机最新未提交变更是否已发布，应以远端实际版本为准。`models/`、`hflib/`、虚拟环境、生成结果与本机协调记录由 `.gitignore` 排除；克隆后按上面步骤准备输入并重新构建，不需要历史备份归档。

- [已确认需求](structure_reader/docs/requirements_alignment_2026_10_17.md)
- [历史实施计划 v5](hf_docs/project_execution_plan_v5.md)
- [当前进度与验收边界](structure_reader/docs/progress.md)
- [当前模型范围与旧参考实现](structure_reader/README.md)

`build-hf` 是当前 HF decoder 入口。旧 `inspect` / `build` 参考实现与 DSpark 路径保留供历史研究，不能替代当前 HF 阅读结果；部分旧集成测试依赖本机历史产物，不能将全仓测试视为干净克隆的安装检查。
