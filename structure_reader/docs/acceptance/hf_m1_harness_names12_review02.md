# 反馈12修复01复核：保全名与长标签，修正替换语义

2026-10-24，5c846590…的512条可见边标签完整，528条实际SVG边进出端点与14de真实节点/端口/容器几何配对一致、无新悬空。38说明、源码pre原文、三数据、144ONNX、5935旧保护及151交付/23源/7输入/两真实build身份已核。正式仅该范围认可。

仍changes_required：说明文字无上下文替换转置T及压缩键K，实际出现`compressed_kv.sequence_length`、`comb.sequence_length`、`k^sequence_length`与`num_experts_per_tok^sequence_length`，改变了原源码/数学语义。需保.T/^T转置、键K与专家数K区别；合法维度继续全名。另缓存讲解“前态+T”及普通参数表达式Tbuf'后态显示同名映射一起收尾。

证据codex_harness_names12_review01/{verify_final.py,final_result.json}。候选只读，有限修同一完整名称需求，不重新分析模型。
