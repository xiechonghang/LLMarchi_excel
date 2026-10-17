# M1 展示复验02复核：方向通过，宽文字块避让未完成

2026-10-18，Codex。50f10185f4db393169578ac6ce0f82782ccd34064b8bec9430837a2b99884680，changes_required，仅展示范围。

DeepSeek已正常完成。Codex读实际6检查及8项自包含正负例/报告/原始日志，复跑8项rc0、固定候选诊断rc1；90冻结/49保护再复算不变、开始结束吻合，新6文件SHA匹配。实际SVG路径起终坐标映射到节点锚点，与独立JSON期望对比，42模块/缓存图548边missing/extra/reversed/unresolved全0，上轮方向覆盖不足已补齐。旧32中心遮挡0，当前JSON/xlsx/effective_config与29d70相同；缓存角色/下游/条件、链接/marker、两行间距、画布与同源稳定保持。

同一个标签/节点遮挡问题仍未完成：_block_free失败回退_anchor_free仅保证中心。具名条件“追加 N_win_i 条（可为0）”在Prefill Indexer(1654.33,34)，按渲染器自用文字度量，右端含检测余量1723.58，去掉4px余量仍1719.58；后绘i_scorer矩形从1672开始，文字估计侵入47.58px，包含条件尾部。实际字体须缩至约27%宽度才能脱离，此处不能用“中心不入框”或“既有已认可”豁免。另input_layernorm/token_idx/总览标签亦有具名风险。274相交/208稳健估算仅范围度量，不拆274任务；真实浏览器字形仍未核验。

证据：只读diagram_gate/round02；Codex复核codex_diagram_gate_review02/result.json、key_text_replay.json、replayed_gate.json及原始日志。有限修复只需取消损害完整文字的回退，在必要列/行留白或有边对应的标签空间内安排完整占用；不改模型图数据/Excel，不重建图框架、不重复旧模型门禁。修后按新身份交展示复验，用户图示阅读未通过，M2/M3未启动。
