# M1 用户图示反馈02：初版复核01，需修正同一展示任务

2026-10-18，Codex。原可见GLM正常交付新候选`e8505d9e261b00624d15b7c8fbbdfa066b9d574bc43e4992b9e5addfca3ebda3`并停止。仅核本轮两项用户要求，不重审模型或Excel。

正确成果保留：计算框开始使用源码调用名与计算类型，tensor/权重/cache矩形已改圆点端口；模块概览标成导航。浅底计算框固定深色主字和副字，解决原页面dark浅字叠浅框的问题；模型JSON、effective_config、xlsx仍与5411/29d70逐字节相同。Codex36图+44页面检查rc0、两次真实build原始日志同新ID/第二次reused、7当前构建字节及13工具源/manifest匹配、70旧候选保护未变。

## 为什么还未完成用户要求

用户完成点①（配色）：框内主字与浅底静态对比度14.49、副字7.39，改善认可；但flow线和marker改为固定`#1f4e79`且dark未配对，深色card `#1d2126`上静态对比度仅1.87。修文字时不应让表示tensor流的箭头变暗。此为颜色计算证据，不是浏览器实测或完整色彩合规审计。

用户完成点②（计算框/张量箭头/源码名和size）尚有三类具体遗漏：

- 端口圆点改了绘制位置，anchor仍按旧172px矩形边缘和旧高度计算，路径不接实际端点。直接读取候选SVG：Prefill Attention-Q的q_weight圆心`(26,119)`、半径5，而对应权重箭头起点`(188,131.5)`，相距约162px。按同tensor href关联，324个有引用的圆点均存在超过9px的最近路径端点距离。数量只是同一个端口几何缺陷的范围，不拆任务；不能用旧虚拟矩形端点checker证明新圆点已连接。
- `full_label = e.cond or e.label`使50条条件边可见文本只剩闭窗条件，张量源码名和shape不在该箭头上；另12条长边标签实际可见的两行把维度截掉，全文仅在title不足以满足“箭头标size”。应将名称/shape/条件分别安排可见行，有限调整留白，不把全文截断作为避让成功。
- 映射表有条目存在不代表源码名正确。`expert_gu_out[N_e,2*Es]`命名`current`，但HF:1050的current在_apply_gate之后已是`[N_e,Es]`；中间Linear结果应标实际F.linear表达式或_apply_gate的gate_up入参。`i_scores[B,T,...]`命名scores，但HF:453的scores含head轴，:456减head后返回，实际调用方为index_scores，应按真实返回表达式或调用方名称。`q_a_norm`标self.q_a_norm(q_a)，HF:813没有局部q_a变量；`expert_down_raw`标down[e]也不是:1051真实self.down_proj[expert_idx]。这些是本轮展示别名错误，模型数据不变，不回头重建模型。

结论：`changes_required`。只回原GLM完成同一用户任务，保留正确粒度/数据/Excel和已改善的图义/框字配色。暂不派DeepSeek新门禁；修后再固定新身份做受影响展示复验。静态工程与浏览器视觉/用户阅读分开，M2/M3未启。

证据：`.work/coordination/hf_m1/codex_onnx_view_review01/`中的result.json、port_geometry.json、visible_edge_labels.json、source_name_findings.json、colour_pairs.json、原始图页回放；修前before_fix01.json与77份候选保护供下一轮核对。
