# HF M1 ONNX展示独立复验01的Codex复核

2026-10-19。**方向与身份范围认可；独立检查覆盖需有限补足，展示最终工程认可暂缓。** 本轮没有确认新的产品缺陷，不回GLM、不重做布局。

原Terminal22、huoshanai/deepseek-v4.1-flash正常完成onnx_view_gate/round01并空闲，报告passed。Codex已读实际两新检查文件、7正负例、身份脚本、原始日志与报告；实际复跑7项、固定c97候选检查、身份检查均rc0。98冻结/84旧候选保护、开始结束指纹、新两文件SHA与当前一致。原GLM两实际build及字节同源结论保持，未重跑构建。

42模块/缓存图的548实际SVG路径端点，按真实circle/IO/compute几何定位，再对独立旧graph_expect.py推导的JSON引用，missing/extra/reversed/unresolved均0。正例、反向、错终点和旧162px悬空负例确实运行。2张导航总览各40边，仅端点已定位、聚合方向未全量枚举；原报告“4040”是两张40/40拼接表述，不是4040条边。保持单列。

但其余覆盖表述过强，不能直接全量通过：

- `load_aliases`只处理ast.Assign，实际OP_SOURCE为AnnAssign，运行得到映射大小 **0/164/49**；框名expected为空后跳过，具名源码检查里读出的label没有与实际来源比较，只验证固定字符串存在于HF文件。
- 边标签名称/shape查找的是全页`title`全文列表，未比较本条可见tspan，更没有证明条件与名称在该条同时可见。
- 配色检查的pairs直接硬编码，虽然定义了CSS读取函数，却未用实际声明值计算；因此“按实际CSS优先级配对”没有被该脚本证明。

Codex只做三项有限内存反例（不写输入或候选）：当前专家投影框删一个可见字符、第一边标签删可见首行但保留title、实际nlabel CSS改白字，旧检查均error=0。证据 `codex_onnx_view_gate_review01/coverage_counterexamples.json`。这些是检查能力缺口，**不是当前真实HTML已经出现这些错误**；Codex此前针对实际SVG/配色的正面证据保持。

仅向原DeepSeek补足原任务这三处证据与真实源码命名绑定：新证据目录round01_followup01，不改产品、不改原两检查或旧报告、不新建通用框架；复用已正确端点/身份结论，不重跑旧门禁或模型/Excel，不重新构建。原round01文件与两检查保护只读。候选c97和98冻结身份不变。

browser_visual=blocked/not_verified、native_excel=not_verified、user_reading=not_passed。导航总览宽17582及大空隙是已记录静态观察，本轮不据此新增布局任务；等待用户实际阅读反馈。最终展示通过仍需补足检查后记录。
