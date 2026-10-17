# M1 连接与缓存图：展示静态工程复验03通过

2026-10-18，Codex复核原可见OpenCode / huoshanai/deepseek-v4.1-flash独立交付。对应用户阅读反馈01：保留已认可的算子粒度和Excel内容，为HTML增加直观结构图；冗余留待以后讨论。

固定候选：`5411cee8243a3a361d5d5991c5838d210e216d7d743fff0ba8e5ecfd4438d3be`。本轮96文件freeze SHA `097d74c530bb6ebaf694a3ff7fe6670448dce3219ddffa275c3750161421b778`，63份旧候选保护；独立开始/结束与Codex当前再复算均一致。旧轮报告、冻结及候选保留不动。

## 结论与范围

`display_static_engineering=passed`。新HTML每阶段含Decoder模块总览、模块内部算子连接图、三类缓存的历史/本次读视图/持久回写图；保留原卡片、参数和张量联查。完整Prefill在前、完整Decode在后。图从同一reading_document引用生成，模型事实与Excel未扩改。

本次通过只适用于固定样板及记录的静态覆盖，不代表任意图布局正确、数值等价或真实浏览器视觉通过。`browser_visual=blocked/not_verified`、`native_excel=not_verified`、`user_reading=pending`。用户先前认可粒度和Excel内容的反馈继续有效；新图仍待用户阅读。M2/M3未启动。

## Codex复核依据

- 读取本轮实际独立报告、最小identity_check.py与原始argv/cwd/stdout/stderr/returncode日志；复核复用检查仍为冻结的round02版本，8个自包含正负例能识别反向/错终点与中心在外但尾部遮挡。本轮没有复制一套新检查框架。
- Codex复跑既有8项、显式指定新候选的参数化展示检查、最小新身份检查，均rc0。未重跑模型/Excel旧门禁、无关坏输入探针、全仓或长度矩阵。
- 44图在记录的CJK/Latin文字宽度度量下完整标签块及中心入框均0，节点相交、两行间距和画布越界均0；关键Indexer“追加N_win_i条（可为0）”整块避让保持。此项是静态工程度量，不是浏览器实测字体结论。
- 42模块/缓存图548条实际SVG路径起终点与独立从固定JSON引用推导的有向边相符，缺失/多余/反向/未定位均0。2张总览图不在548端点全量统计内；不据此声称总览每条聚合边已全量证明。
- 三类缓存角色、实际下游与闭窗条件保持，kv_full为本次读视图，持久回写与本次读取有区分。2212本地href有目标，132marker唯一且闭合，无外部依赖，阶段连续。
- helper中无_anchor_free残留，AST无未定义下划线调用，同列推让只接受完整块。实际HTML与b72e逐字节相同；JSON、effective_config、xlsx与原29d70逐字节相同；7份当前构建字节和manifest六交付SHA、12工具源匹配。GLM原始29图+44页面检查rc0及两真实build同新ID/第二次reused已核，不重复构建。

独立证据：`.work/coordination/hf_m1/diagram_gate/round03/`；Codex复核：`.work/coordination/hf_m1/codex_diagram_gate_review03/result.json`及原始回放日志。三角色均完成停止，等用户反馈，不新增任务保持忙碌。

## 用户阅读入口

- [新HTML连接与缓存图](../../outputs/deepseek_v4_flash_0731_hf/5411cee8243a3a361d5d5991c5838d210e216d7d743fff0ba8e5ecfd4438d3be/viewer/decoder.html)
- [Excel（内容与已认可版本相同）](../../outputs/deepseek_v4_flash_0731_hf/5411cee8243a3a361d5d5991c5838d210e216d7d743fff0ba8e5ecfd4438d3be/tables/decoder_reading.xlsx)
