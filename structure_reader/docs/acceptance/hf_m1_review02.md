# HF M1主干修复复核02

2026-10-17，Codex。候选31c183e7ec26ca083c945e4764bc3abd29b11714ad77db9f92760d6527a2d02e。结论：**changes_required**，尚未派正式独立验证。

已关闭：初版producer断开/前向引用/自环（两阶段均零）；HTML参数联查断链（当前零缺目标）；wrong_av原变异具名拒绝、实际CLI隔离rc1/0HTML；临时源码sigmoid→cos及未知模型键两个原路径均rc1/0HTML。主干38项、Excel57项、独立准备121项全部rc0；两次真实build-hf均rc0、同候选reused=true；7份交付字节与当前build一致，manifest各产物和11项tool_sources SHA匹配。Excel已认可模块保留。

仍有四组直接对应需求§§3/4/5/6的呈现事实问题：

1. CSA缓存overlap在两阶段均写成compressor [B,Dh]/indexer [B,Id]，调用前后同误。HF :278-292保存chunk[:, -1, :, :head_dim]，chunk为[B,N_win,M,2D]，所以持久Ca尾是[B,M,Dh]/[B,M,Id]。闭窗数为0时不应画成无条件覆盖overlap。
2. i_scorer消费调用后compressed_i [B,Tci+N_win_i,Id]，却产i_scores [B,T,Tci]。HF :556-566明确用追加后compressed_kv计分/TopK；prefill旧Tci=0但本次可有新条目，当前图漏了它们。分数轴应覆盖Tci+N_win_i。
3. MoE routed_out为[B×T,H]，shared_out为[B,T,H]，moe_merge仅标add，缺HF :1133实际view(batch,seq_len,hidden_dim)。此外单专家仍一个gather+linear+chunk+clamp+silu+mul+linear+mul大算子、多个输出同卡/同Excel行，不能按需求§3读出各矩阵乘与中间连接；index_add输入没有token_idx，反而重复列topk_weights。需要一份专家模板内的必要算子行和选token索引流，避免逐专家复制。
4. 默认position_ids在当前图为[B,T]，cos/sin却固定[1,T,Rk/2]。HF :1335-1337默认unsqueeze(0)确为[1,T]；应在当前选定路径保持一致，并区分默认与可选自定义批次位置，不扩输入组合矩阵。

以上是实际候选事实错误与阅读粒度修正，不要求建设通用形状推理、全边变异或更多长度测试。原始源码仅静态阅读，未导入/执行模型。证据.work/coordination/hf_m1/codex_review02/result.json、source_shape_findings.json、flow.json、wrong_av_input.json、source_probes.json及各原始命令/输出。旧输入与候选只读保留。

GLM同会话定点修复；Qwen/DeepSeek均等待，冻结接口不变。工程、浏览器/native Excel和用户阅读尚未通过；M2/M3未启动。
