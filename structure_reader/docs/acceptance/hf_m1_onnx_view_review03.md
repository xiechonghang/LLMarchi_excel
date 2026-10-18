# HF M1 ONNX式展示复核03：源码名收尾认可，待新展示独立复验

2026-10-19，Codex。仅认可收尾02的节点源码名称换行修复；**新展示独立复验待完成，未宣布展示工程最终通过**，真实浏览器与用户阅读均未通过。

原可见Terminal25、huoshanai/glm-5.3-flash正常交付后空闲。候选 `c97d0b5c72a2d6da2936bb74edbb150e9ee29320a89c977ff9a528f277227231`。

修前指纹反向还原：只把当前 `_node_lines` 主行调用/注释及 `_wrap` 签名与返回在内存还原，就匹配修前diagram.py SHA；实际只有该产品文件及对应测试改变。节点名称按需要换行，原高度随行数增长；边标签与类型换行、源码映射、配色、端口几何算法未改。新两检查分别实际解析现样板150个框内nlabel与自包含长名SVG，具名5例×2阶段全部完整。

Codex实际43图、44受影响页面检查rc0。实际44 SVG中节点源码名0截断、类型0截断、628边标签可见全文0截断、50条件同见、324同tensor-href圆点0悬空；暗主题flow/weight/cache线marker图例成对对比仍6.55/8.03/8.75，框主字14.49/副字7.39。最近同href端口口径没有独立证明所有producer→consumer方向；新circle/IO解析与JSON独立期望比较由下一轮负责，旧rect专用checker不直接套用。

原始两实际build日志rc0、同新ID/第二次reused；当前7交付字节与纯builder一致，manifest与13工具源匹配。JSON/effective_config/xlsx与5411/29d70逐字节一致，84历史候选文件保护匹配。没有重新分析模型/Excel或重复旧门禁。

证据：`structure_reader/.work/coordination/hf_m1/codex_onnx_view_review03/`，含 `cleanup_diff.json`、`result.json`、`text_and_ports.json`、`colour_pairs.json`、`raw_log_review.json`及两实际复跑日志。新展示固定身份及受影响独立复验另记 `onnx_view_gate/round01/`；不移植旧5411冻结结论。
