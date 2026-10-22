# HF M1主干修复复核03

2026-10-17，Codex。候选0e091374fb72ae67cc6e9b3f3d86f57c346dafafcb2f63620f1aca371d3008e7。结论：**changes_required**，仍未派正式独立验证。

上轮4组修复落盘确认：两套overlap为[B,M,D]且注明闭窗条件；i_scores覆盖Tci+N_win_i；routed显式还原[B,T,H]后相加且单专家必要算子已分行；默认position_ids为[1,T]。原断链/错误AV拒绝/HTML联查保留。Codex实际主干44项、Excel57项、独立准备121项rc0，两build-hf rc0同ID/reused；7份字节与当前构建一致，manifest与11tool_sources一致，旧两候选14文件指纹未变。57项修前基线仅主干提取器、检查器及自身测试3文件变化。

单专家拆行后有一处寻址链尚未完整：moe_expert_weight缺expert_token_idx实际输入，却声明top_k_weights[token_idx,top_k_pos,None]；两个Linear权重直接引用[E,2Es,H]/[E,H,Es]整库，未显式绑定当前e选择的二维切片。HF modeling:1045-1052明确expert_idx选择权重片、token_idx/top_k_pos选择路由权重；需在一份模板的真实引用中表达，不能仅说明文字。此修正直接对应需求§3/§5，不增加其它功能或检查平台。

证据.work/coordination/hf_m1/codex_review03/result.json、fix01_facts.json、原始测试/两build/错误AV隔离日志；来源仍仅静态阅读。已交原GLM会话限定修复02。Excel模块认可保留，Qwen与DeepSeek等待。真实HF整体、浏览器/native Excel/用户阅读未通过，M2/M3未启。
