# Hugging Face 资料：Transformers 仓库、官网模型与使用依赖

更新日期：2026-10-11。

本文说明 `huggingface/transformers` 软件仓库与 Hugging Face 官网可下载模型之间的关系和依赖，并介绍模型文件、使用流程及本项目的对应情况。结论结合 HF 官方说明、官方软件加载代码与本地检查；示例命令没有在本次整理中执行安装、权重下载或推理。

## 文档索引

**最新执行入口（2026-10-13）：[项目推进说明书](project_execution_handbook_v4.md)。** v3 管产品需求，[v4](model_structure_plan_v4.md)管垂直切片；Codex 终端派单/最终核验，Claude CLI 开发，opencode + DeepSeek 独立测试。当前 892 项本地回归有 1 failure、2 errors，阅读页未交付。下方旧索引中的“唯一 v3”“Harness”“逐单审核”等状态描述保留历史，由推进说明书覆盖。

阶段性项目入口：[2026-10-13原计划、整体进度与后续安排](stage_reports/2026_10_13_v3_project_checkpoint.md)；后续阶段快照统一保存在[stage_reports](stage_reports/README.md)。该记录区分开发、测试与最终验收，不替代v3正式计划。

| 文档 | 定位 | 状态 |
|---|---|---|
| [README.md](README.md)（本文） | 工作区主线：HF 仓库与模型的关系、下载与使用、当前工程状态 | 有效 |
| [模型结构可视化与工程方案](model_structure_visualization.md) | 方法论与工具评估（源码/追踪/归组/证据分层） | 有效；运行期工具属远期 |
| [项目计划 v3](model_structure_plan_v3.md) | **唯一当前执行依据**：特征代表、静态条件/状态、证据核验、清晰目录与使用方式，S0–S8 | S0 passed；S1图IR小单已开始，结构阅读页仍未实现 |
| [开发执行与审核规范](development_workflow.md) | Claude 开发、Harness 测试、Codex 验收与能力日志 | 2026-10-12 起执行 |
| [S0 实际审核日志](../structure_reader/outputs/_coordination/s0/2026_10_12_round_01/audit/final_acceptance.md) | 当前交付、实测结果、能力与缺项 | 10 项通过、4 项未通过；S0 未批准 |
| [S0 第二轮注册回归](../structure_reader/outputs/_coordination/s0/2026_10_12_round_02/results/verification_02_registry_regression.md) | Harness 实际项目测试与注册修复复跑 | 52/52 通过；CLI/基线/文档及完整阶段仍待交接/验收 |
| [S0 第二轮最终审核](../structure_reader/outputs/_coordination/s0/2026_10_12_round_02/audit/final_acceptance.md) | 当前完整交付、本地实测与能力边界 | Harness定版与生命周期回归137项、Codex28项与137回归通过；S0由Codex批准 |
| [历史计划 v1](archive/model_structure_plan_v1.md) | 源码与运行采集的较大建设范围，M0–M8 | 已归档 |
| [历史计划 v2](archive/model_structure_plan_v2.md) | 静态特征结构路线，T0–T7 | 已归档；审查修正已纳入 v3 |
| [v2 审查记录](archive/model_structure_plan_v2_review.md) | v3 的结构修正与工程完善依据 | 历史审查记录 |
| [旧说明入口](archive/transformers_and_hub_models.md) | 关系说明已整合到本文 | 已归档 |

计划沿革：v1 包含较完整的采集/验证建设；v2 收敛为静态特征代表阅读；v3 整合源码审查修正，明确模板/实例、条件/状态、出图前核验以及工程与产物分区。历史版本都在 `archive/`；后续执行 [v3](model_structure_plan_v3.md)，从 S0 开始。

工作区目录入口见 [根 README](../README.md)。未来代码统一位于 `structure_reader/`，正式生成内容统一位于其 `outputs/`；具体脚本、文件和使用步骤见 v3，第 7 节命令尚待工程实现。

## 1. 核心结论

**Hugging Face 官网模型仓库提供模型文件；Transformers 仓库提供使用其中一部分模型的软件代码。两者按兼容的文件和接口约定配合，但官网模型不必全部依赖 Transformers。**

判断依赖时，要区分三个问题：

1. **依赖 Transformers 软件吗？** 使用其模型加载、分词或其他功能时，依赖当前 Python 环境安装的 `transformers` 包；其他软件也可以使用 HF 上的模型文件。
2. **依赖自己克隆的 Transformers 源码目录吗？** 只有当前环境实际引用这份源码时才依赖它。安装发行版通常使用环境中的 `site-packages/transformers/`。
3. **推理时必须访问 HF 官网吗？** 完整模型文件和运行环境已准备好时，可以离线加载。使用 Transformers 不等于每次推理都需要联网。

**下载模型不会自动绑定项目旁边的 Transformers 源码目录；克隆 Transformers 也不会自动下载模型权重或把软件安装进 Python。**

依据：[HF 官方支持的模型库](https://huggingface.co/docs/hub/models-libraries)、[Transformers 官方安装与离线说明](https://huggingface.co/docs/transformers/main/installation)。

## 2. 两类仓库分别提供什么

### 2.1 Transformers 软件源码仓库

官方地址：[huggingface/transformers](https://github.com/huggingface/transformers)。本项目位置：[hflib/transformers](../hflib/transformers)。

这是一个软件项目，提供模型加载与保存、模型执行、分词和预处理、生成及任务封装等代码。安装后通过 `import transformers` 使用。

它既有通用加载代码，也有各模型的执行代码；不是只有下载功能。通常不会把所有模型的训练权重打包进这个软件仓库。[官方软件设计说明](https://huggingface.co/docs/transformers/main/en/philosophy)。

### 2.2 Hugging Face 官网模型仓库

例如 `google-bert/bert-base-uncased`、`Qwen/Qwen3-0.6B` 或本项目的 DeepSeek 模型仓库。

HF Hub 托管、版本化和分发这些模型产物。模型发布者可以是其他公司、机构或个人，不一定是 Hugging Face。

下载模型仓库通常得到权重、配置、分词/预处理文件和使用说明；部分仓库还提供自定义代码、独立推理脚本、适配器或导出格式。**下载这些文件不会自动安装完整的 Transformers 软件。**

HF 支持多个模型库，所以“来自 HF 官网”描述的是分发来源，不是唯一运行依赖。具体开放范围和使用条件以模型自己的文件与许可证为准。[官方模型仓库说明](https://huggingface.co/docs/hub/models)、[支持库列表](https://huggingface.co/docs/hub/models-libraries)、[仓库许可证说明](https://huggingface.co/docs/hub/repositories-licenses)。

## 3. 下载的模型文件是什么

不同发布形式的内容不同。以下是 Transformers 风格模型仓库的常见文件，不要求每个模型都有全部文件。

### 3.1 配置与权重

- `config.json`：模型类型、尺寸及量化等配置，供兼容的模型软件读取。
- `model.safetensors` 或其他权重文件：训练得到的参数张量。
- `model-00001-of-00048.safetensors` 等：拆分保存的权重文件。
- `model.safetensors.index.json`：记录参数位于哪个分片，**索引不是权重本身**。

对于这种发布方式，权重文件需要相应程序加载和执行。只有配置和索引，没有真实权重，不能使用训练好的模型。[官方模型仓库文件说明](https://huggingface.co/docs/hub/transformers)。

### 3.2 分词、预处理与生成文件

- `tokenizer.json`、`tokenizer_config.json` 或其他词表文件：文本编码、特殊 token 等资料。
- processor/image processor 配置：图片、音频或多模态输入的预处理资料。
- `generation_config.json`：默认生成设置。
- 聊天模板或模型自己的消息编码程序：把多轮消息组织成模型要求的输入格式。

分词器可用，不代表模型权重已完整；有 tokenizer，也不代表一定提供通用聊天模板。[官方聊天模板说明](https://huggingface.co/docs/transformers/main/chat_templating)。

### 3.3 代码、说明与其他发布形式

- `README.md`：模型卡与使用说明。
- `modeling_*.py` 等：可能供 Transformers 加载的自定义代码。
- `inference/`：模型作者提供的独立参考推理程序。
- `adapter_model.safetensors` 与 `adapter_config.json`：可能只包含适配器，需要另行准备基础模型。
- GGUF、ONNX 等：对应其他运行时支持的产物。

**模型目录有 Python 文件，不代表 Transformers 会自动执行这些文件；模型目录有权重文件，也不代表它一定是完整基础模型。**

依据：[自定义模型说明](https://huggingface.co/docs/transformers/main/en/custom_models)、[PEFT checkpoint 格式](https://huggingface.co/docs/peft/main/en/developer_guides/checkpoint)、[HF 的 GGUF 支持](https://huggingface.co/docs/hub/gguf)。

## 4. 调用方式决定依赖关系

### 4.1 使用 Transformers 普通加载接口

例如程序调用 `AutoModelForCausalLM.from_pretrained(模型目录)`。

```text
你的程序
  ├─ 导入当前环境里的 transformers 包
  ├─ 通过该包读取下载的配置、权重和配套资料
  └─ 使用该包提供的模型代码执行推理
```

**这条调用路径依赖 Transformers 软件。** 对原生支持的模型，执行代码通常位于安装的包中，模型下载目录不必重复包含它。

同一套软件代码可以使用多个兼容的模型发布版本；模型作者发布新权重，不一定需要为每个版本另建一份 Transformers 软件仓库。但实际兼容性仍需看安装版本、任务、量化和模型要求。[官方模型加载说明](https://huggingface.co/docs/transformers/main/en/models)。

### 4.2 通过 Transformers 加载模型自定义代码

部分模型通过 `auto_map` 指定自定义入口，并按说明使用 `trust_remote_code=True`。

此时依赖包括当前环境的 Transformers、模型仓库的自定义代码，以及相应模型文件。具体执行代码可来自模型仓库，加载接口仍由 Transformers 提供。

因此，**模型自带代码并不自动等于摆脱 Transformers**。`trust_remote_code=True` 也不会把任意 `inference/model.py` 自动接入加载接口。[官方自定义模型说明](https://huggingface.co/docs/transformers/main/en/custom_models)。

### 4.3 使用模型作者的独立推理脚本

独立脚本可以自己读取权重并执行模型，同时使用 Transformers 的部分辅助功能，也可以完全采用其他兼容工具。

应该检查脚本导入和依赖文件，分别判断模型执行、分词、预处理和加载工具的依赖。不能只因为主模型代码在本地，就认定整个程序没有 Transformers 依赖。

本项目 DeepSeek 参考推理属于“模型自己的执行代码 + Transformers 分词器”的组合，具体证据见第 7 节。

### 4.4 使用其他软件库或运行时

HF 官网支持多种调用路径。例如：

- timm 模型可直接通过 timm 从 Hub 加载，使用 timm 的模型代码。[官方 timm/Hub 集成](https://huggingface.co/docs/timm/hf_hub)。
- 兼容的 GGUF 模型可由 llama.cpp 等运行时使用，不需要由 Transformers Python 模型类执行每一步计算。[官方 GGUF 说明](https://huggingface.co/docs/hub/gguf)。
- Diffusers 的 Stable Diffusion pipeline 使用 Transformers 的 CLIP 组件，因此一个完整流程可以依赖多个库。[官方 pipeline 源码](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py)。
- vLLM 提供原生模型实现及 Transformers 模型后端，需要按实际选择的后端判断。[vLLM 官方模型支持说明](https://docs.vllm.ai/en/latest/models/supported_models/)。

**官网模型可以不通过 Transformers 使用，但不能笼统地认为所有其他软件都与 Transformers 无关。** 下载、转换和推理阶段也可能使用不同工具。

## 5. 官方代码怎样证明这层关系

以下只追踪软件加载和依赖，不展开神经网络内部计算。

### 5.1 模型配置把下载文件与软件类连接起来

官网 [Qwen3-0.6B 配置](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json) 中，`model_type` 为 `qwen3`。在本地官方 Transformers checkout 中，配置映射和任务映射分别指向 `Qwen3Config` 与 `Qwen3ForCausalLM`。

```text
官网模型 ID 或下载目录
  → 读取模型配置
  → 找到已安装软件中支持该配置的类
  → 调用具体类的加载方法
  → 读取真实权重并返回模型对象
```

这说明模型目录和软件实现是分开保存、在调用时连接的。配置中的类名是元信息，不会自动生成缺失的程序实现。

本地入口：[配置映射](../hflib/transformers/src/transformers/models/auto/auto_mappings.py)、[AutoConfig](../hflib/transformers/src/transformers/models/auto/configuration_auto.py)、[任务映射](../hflib/transformers/src/transformers/models/auto/modeling_auto.py)。

### 5.2 Auto 加载代码明确区分实现来源

官方 `auto_factory.py` 检查当前任务是否有已注册实现，以及配置是否通过 `auto_map` 指定自定义代码入口；根据具体参数和注册情况选择实际模型类。

- 原生支持时，模型执行代码通常来自当前安装的 Transformers。
- 自定义加载时，执行代码可以来自模型仓库，但加载流程仍使用 Transformers。
- 缺少支持类或有效入口时，不能仅靠修改加载参数让未知模型自动可用。

证据：[官方 Auto 加载源码](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/auto/auto_factory.py)、[本地对应源码](../hflib/transformers/src/transformers/models/auto/auto_factory.py)。

### 5.3 同一个加载器接受官网 ID 与本地路径

通用模型加载代码解析模型文件位置，并处理 checkpoint 及分片等资料；文件可来自 Hub，也可来自本地目录。

因此，“使用 Transformers 软件”和“从官网在线下载”是可分开的两个动作。[官方加载说明](https://huggingface.co/docs/transformers/main/en/models)、[本地加载实现](../hflib/transformers/src/transformers/modeling_utils.py)。

官网 `Transformers` 标签和 `library_name` 是集成元数据，不能替代对当前版本的加载验证。[官方模型卡库信息说明](https://huggingface.co/docs/hub/model-cards#specifying-a-library)。

## 6. 如何下载和使用模型

以下步骤用于说明使用流程，示例路径或版本需按目标模型调整。本次没有执行安装、下载、转换或推理。

### 6.1 下载阶段可以不使用 Transformers

可使用独立的 `huggingface_hub` 包下载模型文件。完整下载示例：

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    revision="main",  # 可复现使用时换成确认过的完整 commit ID
    local_dir="/path/to/Qwen3-0.6B",
)
```

这是下载操作，不会执行模型，也不会自动安装模型的推理依赖。未设置过滤时会下载仓库文件；只想研究资料时可设置过滤，但过滤可能排除实际推理所需内容。[官方 Hub 下载说明](https://huggingface.co/docs/huggingface_hub/main/guides/download)。

本项目的 [download_hf_model.py](../scripts/download_hf_model.py) 使用标准库 HTTP 下载，也不导入 Transformers。默认使用 OpenCSG 镜像、跳过权重；加 `-w` 才包含权重文件。例如：

```bash
cd /Users/xiechonghang/Workspace/LLMarchi_excel
# 只保存非权重资料
python3 scripts/download_hf_model.py -m deepseek-ai/DeepSeek-V4.1-Flash

# 完整下载示例：含数百 GB 级权重，本次未执行
python3 scripts/download_hf_model.py \
  -m deepseek-ai/DeepSeek-V4.1-Flash \
  -mirror hf \
  --revision dba1be0a40aa45a94ad051997016db3960a90277 \
  -w
```

下载结束后仍需检查真实权重、索引引用的分片和其他必要文件；历史清单不等于当前文件完整。

### 6.2 选择软件来源：发行版或本地源码

先准备符合目标软件要求的独立 Python 环境。本地 Transformers checkout 至少要求 Python 3.10。以下以机器上已安装的 Python 3.11 为例：

```bash
cd /Users/xiechonghang/Workspace/LLMarchi_excel
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

接着选择一种 Transformers 来源：

```bash
# 方式一：安装发行版；实际使用时选择支持目标模型的版本
python -m pip install transformers torch safetensors
```

或者：

```bash
# 方式二：让当前环境使用项目中的本地源码
python -m pip install -e ./hflib/transformers
python -m pip install torch
```

发行版一般不依赖项目中的源码目录；editable 安装会引用这份源码目录。具体 PyTorch、设备及额外依赖需要按模型和平台匹配。[官方安装说明](https://huggingface.co/docs/transformers/main/installation)。

确认实际导入位置：

```python
import sys
import transformers

print("Python:", sys.executable)
print("Transformers 版本:", transformers.__version__)
print("Transformers 来源:", transformers.__file__)
```

只有来源实际指向 `hflib/transformers/src/transformers/`，才能说明当前使用这份本地源码。

### 6.3 用 Transformers 加载完整本地模型

下面是适用于完整、兼容且能由 CPU 承载的文本因果语言模型的基础示例。路径需要替换；它不是当前项目两个缺少权重的 DeepSeek 模型可直接运行的示例。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "/path/to/complete-compatible-text-model"
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
model.eval()

# 聊天模型需按自己的模板或消息协议组织输入。
inputs = tokenizer("请用一句话介绍机器学习。", return_tensors="pt")
with torch.inference_mode():
    result = model.generate(**inputs, max_new_tokens=32, do_sample=False)

new_tokens = result[0, inputs["input_ids"].shape[1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
```

大模型、量化模型和多模态模型需要相应设备、组件及加载方式。`device_map="auto"` 等能力可能需要 Accelerate 等依赖，也不会解决软件未支持目标模型的问题。[官方模型加载说明](https://huggingface.co/docs/transformers/main/en/models)。

`local_files_only=True` 可阻止加载器下载缺失文件；`HF_HUB_OFFLINE=1` 可阻止 Hub 请求。二者不会补齐文件，也不约束自定义代码自行访问其他网络资源。[官方离线说明](https://huggingface.co/docs/transformers/main/installation)。

### 6.4 使用模型自己的推理程序

按模型的推理说明安装依赖、转换权重并启动。普通参考脚本的接口不能直接当作 Transformers 的 `from_pretrained()` 接口。

本项目 V4.1 的流程是：**完整 HF 权重 → 转换后的 checkpoint → 自带 generate.py**。在依赖和 CUDA 硬件兼容的服务器上，示例为：

```bash
export PROJECT_DIR=/path/to/LLMarchi_excel
export HF_CKPT_PATH="${PROJECT_DIR}/models/deepseek-ai/DeepSeek-V4.1-Flash"
export SAVE_PATH=/path/to/DeepSeek-V4.1-Flash-TP8
export MP=8

cd "${HF_CKPT_PATH}/inference"
python -m pip install -r requirements.txt
python convert.py \
  --hf-ckpt-path "${HF_CKPT_PATH}" \
  --save-path "${SAVE_PATH}" \
  --model-parallel "${MP}" \
  --expert-dtype fp4 \
  --tokenizer-path "${HF_CKPT_PATH}"

torchrun --nproc-per-node "${MP}" generate.py \
  --ckpt-path "${SAVE_PATH}" \
  --config config.json \
  --interactive
```

这里 `--config config.json` 指 `inference/` 下的运行配置。HF 原始权重目录和转换后的目录不能混用；`MP=8` 是发布说明中的示例，不是本次验证的最低硬件配置。

两个本地参考实现直接使用 `torch.cuda`、TileLang 等，并在分布式场景使用 NCCL，不能原样视为当前 Mac 的运行方案。V4 与 V4.1 的转换参数也不同，应按各自说明使用。

本地入口：[V4 推理说明](../models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/README.md)、[V4.1 推理说明](../models/deepseek-ai/DeepSeek-V4.1-Flash/inference/README.md)。

## 7. 当前工程中的关系和状态

### 7.1 目录怎样对应

- [hflib/transformers](../hflib/transformers)：保存的软件源码；是否参与运行取决于安装和导入路径。
- [hflib/diffusers](../hflib/diffusers)：另一份软件源码；目录存在不会自动形成 DeepSeek 的运行依赖。
- [models/deepseek-ai](../models/deepseek-ai)：下载的模型资料、配置和参考推理代码。
- [scripts/download_hf_model.py](../scripts/download_hf_model.py)：独立下载工具。

### 7.2 DeepSeek 参考脚本怎样依赖 Transformers

两个模型的 `inference/generate.py` 都导入 `transformers.AutoTokenizer`，同时从模型自己的 `model.py` 导入 `Transformer` 与 `ModelArgs`，并用 Safetensors 装载转换后的权重。

所以它们原样运行时，**依赖 Transformers 的分词功能，模型执行使用自带代码**。不能说整个程序完全不依赖 Transformers；也不能据此认为它一定引用了本项目克隆的那份源码。

证据：[V4 生成脚本](../models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/generate.py)、[V4.1 生成脚本](../models/deepseek-ai/DeepSeek-V4.1-Flash/inference/generate.py)、[V4 依赖](../models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/requirements.txt)、[V4.1 依赖](../models/deepseek-ai/DeepSeek-V4.1-Flash/inference/requirements.txt)。

### 7.3 当前模型文件尚未完整

2026-10-11 复查结果：

- **DeepSeek-V4-Flash-0731**：没有实际 Safetensors 权重，也没有本地权重索引。历史清单列出 48 个分片及索引，但这 49 个文件当前均不存在；清单记录不代表当前完整。
- **DeepSeek-V4.1-Flash**：存在权重索引，引用的 48 个分片全部缺失。下载清单的 `include_weights` 为 `false`。
- V4.1 索引记录的权重总大小为 `510286023000` 字节，约 510.29 GB / 475.24 GiB；这是权重元数据，不是已下载大小或推理显存需求。
- 两个模型的 tokenizer 数据已存在，但本地 `tokenizer_config.json` 都没有通用 `chat_template`，应使用各自的 `encoding/` 协议。

证据：[V4 清单](../models/deepseek-ai/DeepSeek-V4-Flash-0731/metadata_manifest.json)、[V4.1 清单](../models/deepseek-ai/DeepSeek-V4.1-Flash/.download_manifest.json)、[V4.1 权重索引](../models/deepseek-ai/DeepSeek-V4.1-Flash/model.safetensors.index.json)、[V4 消息编码](../models/deepseek-ai/DeepSeek-V4-Flash-0731/encoding/README.md)、[V4.1 消息编码](../models/deepseek-ai/DeepSeek-V4.1-Flash/encoding/README.md)。

### 7.4 软件支持与环境状态

此前静态代码检查记录：本地 Transformers 为 `5.19.0.dev0`，HEAD 为 `536ecc007387a50e77603bb5d92100e9b07514cc`；有 `deepseek_v4` 的原生类注册，未找到 `deepseek_v41` 的实现或注册。两个模型的顶层配置都没有 `auto_map`。

因此，本地 V4 有原生加载的代码入口，但未验证完整加载；本地 V4.1 不能只增加 `trust_remote_code=True` 就自动接入自带推理程序。这些结论仅针对检查过的版本，不代表所有版本或其他运行时的支持情况。

本次复查的终端 `python3` 是 Xcode 的 Python 3.9.6；该解释器没有找到 Transformers、PyTorch、Tokenizers、Safetensors 或 Hugging Face Hub 包。这个结果不代表机器上其他 Python 环境的安装情况。

当前工程可用于资料和代码研究，但还不能视为准备完成的推理环境。

## 8. 使用时的判断顺序

1. 先确定使用哪种软件入口：Transformers、模型自带程序或其他工具。
2. 按该入口核对真正的依赖及软件版本，查看实际包导入位置。
3. 核对下载文件是否完整，尤其是权重、分片与预处理资料。
4. 准备兼容的设备与运行依赖，再按模型要求组织输入并调用。

最终要分清：**模型文件的下载来源、使用的软件、引用的软件版本/目录，以及是否联网，是四件不同的事。**

## 9. 资料来源与验证范围

通用结论依据本文各节链接的 HF 官方文档与官方软件源码，当前项目结论依据本地文件、导入语句、依赖清单、类映射和所检查解释器。

本次未进行模型权重加载、推理结果或性能验证。官方在线 `main` 文档与源码可能变化；实际使用应以安装的软件版本和模型 revision 为准。文中的“文档支持”“静态代码存在”都不等于已经在目标硬件上验证成功。

其他入口：[DeepSeek V4 官网模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)、[DeepSeek V4.1 官网模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)、[HF 软件目录约定](../hflib/readme.md)、[本项目模型目录约定](../models/readme.md)。
