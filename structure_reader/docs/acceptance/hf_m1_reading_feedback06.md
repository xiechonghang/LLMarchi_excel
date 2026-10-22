# 用户阅读反馈06：实际 ONNX / onnxsim 简化图并排阅读（2026-10-21）

> 反馈06工程交付（2026-10-21）：已由原Harness App1017 High完成实际ONNX/onnxsim及原图并排，候选788ee019…。当前模块两阶段已生成真实原/简文件，工程复核通过；无真实权重FP32阅读特化、语义分组与实际简化分开，原SVG/三数据/Excel保持。实际阅读与M1整体确认仍待用户，自动检查PAUSED，旧角色不恢复。详[最终复核与入口](hf_m1_harness_onnx06_review04.md)。

> 安装授权更新：用户已答复允许仅项目内隔离安装ONNX/onnxsim及必要依赖；不下载权重、不执行HF，不改全局环境。继续原Harness App1017 High，先实际Q分支样板再覆盖当前模块，详dpsk_harness_onnx06_implement01.md。

用户认为当前032ea HTML已接近可接受，但原图描述复杂度仍高，难以直观感受结构。要求在当前HTML分析基础上，结合源码、参数和结构理解，生成各模块的实际ONNX图，使用onnxsim优化，并与当前HTML对应框图并排展示。

这超出反馈02的“类似ONNX图形样式”，是新增实际格式转换、简化和对应关系。保当前详图及已正确的包含关系、缓存读写、配置选择、源码及双向定位；不以替换原图、复制SVG或改标题冒称ONNX生成/优化。用户“接近ok”不是M1整体最终验收。

继续由DeepSeek Harness App / DeepSeek-V4.1-Flash 1017 High原会话开发，Codex仅需求映射/派单/验收。先静态核实际模块转换覆盖、标准算子与条件/缓存表达、权重输入和ONNX shape所需事实；目前工作Python未安装onnx/onnxsim/onnxruntime，已有numpy2.3.5，无torch。已向用户询问仅项目隔离安装ONNX/onnxsim及必要依赖；未回复前不安装下载。模型权重不下载、不伪造，HF只静态读取、不导入执行。

ONNX无需权重实值即可把权重作为图输入；这不证明与HF模型数值等价。生成图的标准算子语义/shape/状态前后需逐项有据；不清楚的dtype/分支不猜，不把自定义黑盒称成已标准展开/可数值验证。onnxsim实际返回/校验、未简化或失败应如实记录，不承诺节点数一定下降。

预期交付：当前模块对应的原始及简化ONNX文件、来源/对应映射和工具记录，HTML左原详图/右实际简化ONNX图，可定位到源码/原算子；缺失支持明确到模块与原因。保持完整Prefill后Decode、原reading_document/effective_config/xlsx字节不变；独立派生信息新增文件，不改原分析契约或原认可粒度。所有历史候选证据只读；自动检查PAUSED，不恢复OpenCode。
