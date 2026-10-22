# HF M1 ONNX式展示最终静态工程复核与阅读入口

2026-10-19，Codex。**display_static_engineering=passed**，限用户反馈02的两项HTML展示要求。M1整体用户阅读仍pending；browser_visual=blocked/not_verified、native_excel=not_verified，不进入M2/M3。

## 新阅读入口

[打开新版HTML](../../outputs/deepseek_v4_flash_0731_hf/c97d0b5c72a2d6da2936bb74edbb150e9ee29320a89c977ff9a528f277227231/viewer/decoder.html)。同候选tables/decoder_reading.xlsx内容与用户先前认可版本逐字节相同。

图中计算框显示源码调用/表达式和计算类型；tensor/权重放在箭头标注源码名称与shape，非计算端点用圆点、初始hidden_states明确输入。保留原复合算子粒度、分支/残差/共享及缓存时序和闭窗条件。框内深字配固定浅底；边、标签、marker和图例有对应浅暗配色。

## 完成依据

原DeepSeek可见会话正常完成有限补证并空闲。Codex已读实际补充脚本、原始命令/报告和四个有限内存负例，复跑该小补证rc0，真实候选通过，框缺字、边可见行缺失但保留title、实际CSS白字、错误别名四例均被拒绝。没有再运行旧模型/Excel门禁或构建。

原round01的42模块/cache图548条实际SVG起终坐标与JSON producer/consumer引用完全一致，缺失/多余/反向/未定位均0；两张导航总览各40条边只定位端点，聚合方向未全量独立枚举，单列于548之外。

补证读取AnnAssign后实际映射75/164/49；5具名长框名两阶段10框的实际nlabel及计算类型完整。628边标签按每图文档顺序edge↔label配对、检查本条可见tspan；538条有tensor href的名称/JSON shape绑定，50条件与名字/shape同块可见。90无href边实际是80导航聚合边+10缓存下游边，非90条缓存下游；其自身可见全文完整，不宣称这90条均独立绑定JSON shape。

源码语义核查限具名与代表性静态来源，不以75/164覆盖存在冒称所有名字已独立逐个证明。补证把算子id `q_a_norm`误当张量id，空片段被称literal的该项不采信；Codex直接核对真实张量 `q_a_normed`→`q_residual`、HF:813的嵌套归一化赋值，以及Prefill/Decode实际可见名称和JSON `[B,T,Rq]`，补足此具名案例，不改旧checker或原报告。其他具名expert_gu_out、i_scores、expert_down_raw正确来源绑定保持。i_scorer的`.sum(2)`为源码`.sum(dim=2)`的关键字省略，记录为简写。

实际CSS声明和深色覆盖配对计算：框主字14.49、副字7.39；浅暗边标签、条件和flow/weight/cache线对比满足本轮静态判据。补充脚本有限选择器检查与Codex此前未变源码的三类线/marker/图例成对核查合并使用；不把它当通用CSS解释器或真实浏览器视觉。先前43图+44受影响页面rc0、两实际build同c97身份/第二次reused、7交付字节与13工具源同源的结论保留。

## 身份与限制

98文件freeze SHA `4de7e69d3ca469757c493eebcd23674cc38831c4142188f9696fb9c2052bc96d`，84旧候选及16份原round01证据/检查，独立开始结束与Codex当前复算均匹配。补证脚本SHA `21eb5843477083dcda091e57c094a55aedcf18b27817f2d3b2ffe88bc878d96b`与报告一致；JSON/effective_config/xlsx与5411/29d70逐字节相同。产品仍为c97，补证未改产品或旧证据。

真实浏览器字体/CSS/滚动/点击和原生Excel未验证；导航总览宽17582及大间距是已记录静态观察，本轮不新增布局工作，供用户实际阅读判断。阅读确认与静态工程结论分开，等待用户新图反馈，不能自动视为M1最终通过。

独立证据：`structure_reader/.work/coordination/hf_m1/onnx_view_gate/round01/`与`round01_followup01/`；Codex证据：`codex_onnx_view_gate_review01/`与`codex_onnx_view_gate_review02/`。旧报告覆盖过强的记录保留，以上文准确范围为准。
