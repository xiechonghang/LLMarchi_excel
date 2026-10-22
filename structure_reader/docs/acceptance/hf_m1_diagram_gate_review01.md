# M1 新图示独立验证01复核：边标签被节点遮挡

2026-10-18，Codex。固定候选32af1e923cca4f8c1d535786e4b612d4ccfdfe736c47aeb3671230f77f2d5733，展示结论changes_required。

DeepSeek原可见验证会话正常交付并空闲。Codex已读实际6个独立检查文件、正负例、报告和原始日志；复跑10项自包含检查rc0、固定候选检查rc1，具名原因label_occluded。84冻结与42旧候选文件重新复算无变化，开始/结束一致，新检查6文件SHA全匹配。同源报告/两build日志保持固定ID/reused；本轮不重复构建或旧模型/Excel门禁。

确认的一个布局缺陷：实际44 SVG内32处边标签基线锚点落在节点矩形内部，且节点在标签之后绘制，不透明背景覆盖标签。Codex另用XML直接复现32例及绘制顺序，而非只接受包围盒启发式。例如Prefill总览cos_q/sin_q标签(1482,36.5)在attn_cache模块矩形(1396,16,172,49)内；moe_route的logits标签落在moe_dispatch上，缓存当前读视图下游标签落在持久状态节点上。不是32个开发任务；另350处估算包围盒交叠仅风险，不逐条升级成确定错误。

已核连线张量多重集、缓存角色/下游/条件、链接/marker、多行间距与画布边界通过；JSON/xlsx/effective_config仍与29d70相同。独立检查的覆盖限制应准确记录：当前HTML图检查主要比较(tid,kind)多重集，尚未从实际路径端点全面验证方向，因此独立报告“无方向错”覆盖表述过强。Codex此前helper真实输入/权重反查及实际候选同源证据仍保持；下一轮展示复核应有限核路径端点，不能只靠多重集宣称方向正确，也不扩全边变异平台。

证据：hf_m1/diagram_gate/round01/只读保留；Codex复核hf_m1/codex_diagram_gate_review01/result.json、occlusion_replay.json、replayed_gate.json及原始两组日志。

只把边标签/节点遮挡交原GLM展示层定点修复，保留数据、粒度、Excel和已修缓存成果；不能仅调前后绘制顺序反过来遮节点文字。真实浏览器视觉仍not_verified，用户阅读未通过，M2/M3不启动。修后新候选按展示变化固定身份再复核。
