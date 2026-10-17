# HF M1主干修复复核04

2026-10-17，Codex。固定候选29d70d15971b524bc41cf334fa3768ab851186f68005549628ea638dd51abbd4。结论：**ready_for_independent_review**，已知单专家寻址修复范围认可；尚非M1正式工程/用户阅读通过。

静态对照HF modeling:1045-1052：moe_expert_weight现在消费expert_token_idx与expert_topk_pos双索引；moe_expert_slice从整库与命中专家选择二维gate_up_e[2Es,H]/down_e[H,Es]，两个Linear真实绑定对应片，仍只一份模板。两阶段与上轮比较仅此4算子和2张量变化，前4组缓存overlap/闭窗、Indexer score新增长度、routed_view与必要分行、默认position维度均保留。修前57文件仅提取器/检查器/自身测试3文件变化，Excel模块未变，旧三候选21文件未变。

Codex实际44/57/121项均rc0；两真实build同固定ID/reused，7交付字节与当前构建一致、manifest及11tool_sources匹配。HTML联查0断链、JSON/xlsx两表一致。wrong_av原始语义拒绝，真实CLI临时隔离rc1/0HTML；源码sigmoid改cos及未知模型键同样rc1/0HTML。测试数量仅记录，不代替源码核查。

独立诊断0error/18warning/4manual，其中Indexer三输入与专家二维片使旧启发式不能定位，两类manual及未使用声明需交正式独立验证逐项源码核实；未忽略为通过，未据此另扩开发。详细证据.work/coordination/hf_m1/codex_review04/result.json、fix02_facts.json、diagnostic.json及原始日志。新联合冻结在hf_m1/gate/round01/freeze.json，正式任务task.md；仅当前M1样板验证。浏览器/native Excel/用户阅读未通过，M2/M3未启。
