# LLMarchi_excel — 模型结构阅读工具

从 Hugging Face Transformers 源码和模型配置静态分析 DeepSeek-V4，生成离线 HTML 结构图、算子流、源码定位、Prefill/Decode 两表 Excel，以及可选的模块级 ONNX/onnxsim 图。当前提供第 4 层的 HF decoder 样板：Attention、CSA/Indexer、KV cache、TopK MoE 与 mHC。

工具不加载权重、不导入或运行 HF 模型。ONNX 是依据源码与配置构建的结构图，权重作为图输入，不代表已验证 HF 数值等价。其他层型和整网分析仍在计划中。

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

下载脚本默认跳过权重，请不要加 `-w`。Transformers 检出仅作为文本输入，不需要安装 Transformers、PyTorch 或模型运行环境。当前 HF 构建实际消费根 `config.json` 和六个 Transformers 源文件，并校验其 SHA256；来源缺失或漂移会明确报错，不会沿用旧结论。

## 生成 HTML 与 Excel

```bash
python structure_reader/scripts/read_model_structure.py build-hf \
  --model deepseek_v4_flash_0731_hf --layer-index 4
```

命令会打印本次输出目录及 HTML 入口。主要结果位于 `structure_reader/outputs/deepseek_v4_flash_0731_hf/<analysis_id>/`：

- `viewer/decoder.html`：离线阅读页，包含纵向结构图、模块包含关系、算子流与源码定位。
- `tables/decoder_reading.xlsx`：Prefill、Decode 两张工作表。
- `data/reading_document.json`、`data/effective_config.json`：页面与表格所用的数据和有效配置。
- `reports/source_manifest.json`、`run_manifest.json`：输入、工具与交付文件身份。

相同输入、工具源码与依赖生成相同内容身份；已有结果会复用，旧候选不会覆盖。

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

源码、配置、测试与文档直接保存在默认分支，可正常 `git clone` 使用。`models/`、`hflib/`、虚拟环境、生成结果与本机协调记录由 `.gitignore` 排除；克隆后按上面步骤准备输入并重新构建，不需要历史备份归档。

- [已确认需求](structure_reader/docs/requirements_alignment_2026_10_17.md)
- [实施计划 v5](hf_docs/project_execution_plan_v5.md)
- [当前进度与验收边界](structure_reader/docs/progress.md)
- [工程说明与旧参考实现](structure_reader/README.md)

`build-hf` 是当前 HF decoder 入口。旧 `inspect` / `build` 参考实现与 DSpark 路径保留供历史研究，不能替代当前 HF 阅读结果；部分旧集成测试依赖本机历史产物，不能将全仓测试视为干净克隆的安装检查。
