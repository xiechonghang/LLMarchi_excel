# HF decoder阅读数据接口v1

状态：Codex为里程碑一样板确定的开发接口，用于Claude主干与pi Excel并行开发；不是模型事实、golden或已验收产物。两个开发者共同只读此文档，变更由Codex协调。需求书和实施计划v5约束优先。

## 输入来源

模型配置与无权重辅助文件来自工作区scripts/download_hf_model.py下载到models/的HF模型包；Transformers实现来自工作区hflib/transformers本地官方源码检出。两者只读，不运行下载、不加载权重、不以模型包inference/model.py作为HF事实来源。

## 公开接口

Excel模块：`structure_reader.export.operator_workbook.export_operator_workbook(reading_document: dict) -> bytes`，返回合法xlsx文件字节，不自行写文件、不读取models/outputs/.work、不导入产品HF提取器。输入可被验证，错误抛ValueError或其本地子类；不偷偷补造缺失事实。

Claude负责生成并核对reading_document及将返回字节纳入真实build。pi只依赖本文，不等待或修改Claude实现。

## 数据形状（字段名固定）

顶层：

- `schema`：固定字符串`hf_decoder_reading.v1`。
- `model`：对象，含`model_key`、`display_name`、`implementation`（`transformers`）、`source_revision`、`config_path`字符串。
- `scope`：对象，含`layer_index`整数、`attention_type`、`moe_type`、`analysis_path`（`eager`）、`notes`字符串数组。
- `phases`：顺序固定的两个对象，`id`分别为`prefill`、`decode`；禁止第三个阶段或重复阶段。

每个phase：

- `id`、`title`字符串。
- `parameters`：参数对象数组，见下方。
- `tensors`：张量对象数组，见下方。
- `operators`：算子对象数组，见下方；保持实际阅读顺序。
- `cache_notes`：字符串数组，说明缓存输入前提、读写和返回/持久状态区别。
- `coverage_notes`：字符串数组，说明本阶段覆盖与未覆盖。

参数对象：

- `id`：阶段内唯一标识，算子引用使用此值；`symbol`、`meaning`字符串。
- `value`：配置或派生值，允许JSON标量、数组、对象或null；运行时长度/数据相关数量可为null。
- `expression`：公式字符串或null；`note`：解释字符串。
- `source_refs`：来源对象数组。

张量对象：

- `id`：阶段内唯一标识；`name`字符串。
- `shape`：按轴顺序的整数/符号表达式字符串数组，例如`["B","T","H"]`；标量shape为`[]`。
- `parameter_refs`：参数id数组，引用本phase.parameters。
- `kind`：`activation`、`weight`、`cache`、`mask`、`index`或`boundary`之一。
- `note`：解释字符串。其他可选字段允许存在，消费者不靠它们补事实。

算子对象：

- `id`：阶段内唯一标识；`module_path`、`name`、`op`字符串。
- `inputs`、`outputs`、`weights`：张量id数组，全部在本phase.tensors可解析。
- `parameter_refs`：参数id数组，引用本phase.parameters。
- `source_refs`：来源对象数组。
- `analysis_status`：`analyzed`或`internal_unanalyzed`。
- `note`：解释字符串。后者在HTML和Excel必须显示中文“内部未分析”。

来源对象：`path`、`locator`、`kind`、`note`字符串；kind为`config`、`default`、`derived`、`source`或`runtime`。locator可为JSON字段位置或源码符号/行区间；runtime不得伪造源码或配置值。

同一权重或激活张量可被多个算子引用，不能把共享引用误判为重复定义。阶段之间id按阶段作用域解释，不强制复制不同id。结构规模用E/K/N等参数表达，不设repeat_count/repetition_count/重复次数字段。

## Excel呈现要求

仅`Prefill`、`Decode`两张表，各自依次有模型/阶段简述、参数符号区、算子区，所有信息可独立阅读。

参数区至少包含参数id、符号、含义、值、表达式和来源；JSON复合值可以确定性字符串化。算子行至少包含id、模块、名称/算子类型、输入名称与shape、输出名称与shape、权重名称与shape、参数引用、源码位置、分析状态和说明。参数引用汇总算子及所用张量的显式parameter_refs。

算子行的参数引用必须可定位本表参数区。可以使用Excel公式引用本表的参数单元格并显示符号和值，或可直接定位的单元格引用；不能只复制一份无法联查的数字。引用shape保留符号，同时可读到对应配置值；不为了模拟运行填B/T/P的固定值。

冻结窗格、筛选、列宽/换行和明显的黑盒标记服务于阅读。不能另建参数sheet、metadata隐藏sheet或逐专家复制行。返回文件在相同输入下应稳定；固定工作簿/ZIP时间等易变元数据，避免破坏内容寻址重用。

## 自包含验证

pi可以自行制作小型合成输入用于单元测试，放在其测试文件或tests/hf/fixtures/workbook_*中，必须明确为接口测试，不能称为真实DeepSeek候选。不要读取.work或真实outputs作为测试前提。测试包括两表、参数引用、shape/权重对应、复用张量、黑盒标记、错误引用拒绝和相同输入稳定性。

真实模型语义由Claude依据HF源码构建，由opencode独立验证；导出器不重新解释或推测模型结构。
