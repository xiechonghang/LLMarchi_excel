# 反馈11收尾01复核：主要事实修正保留，最后三处文字清理

2026-10-24，候选 `f451b7a12f5a691936e54fcf152e9056291210616bbd3fcbfd18c299df51c4bd`。Codex读实际 helper/接入/有限断言及新报告/原始日志。源码可见名、三个无效缓存 key、mHC 参数派生和转义、权重 shape 与三返回/本层来源、MoE bias 只选择及 gate/up 截断、主要路由/专家/Indexer 维度均已修正，保留。

仍 changes_required，仅同一说明文字：

- `overview_narrative` 写“第三次残差回写”，实际每层仅 Attention 与 MoE 后两次。需要改为第二次/最后一次，不能暗示第三段。
- `_indexer` lead 说隐藏状态和 q_residual 一起生成自己的压缩条目。实际隐藏状态经 kv/gate 分支生成条目，q_residual 经 q_b_proj 生成 query，hidden_states 还经 scorer.weights_proj 得到头权重；这些随后进入打分。应明确三个输入角色，避免新说明反而串错支路。
- 专家段把已带括号的 `c.sh` 再包一层，出现 `[[N_e,H]]`；用参数链接 `Lim=...` 直接拼进 `clamp(-Lim=...,Lim=...)` 形成不合法公式。统一正文为单层 shape、gate 上界 Lim、up 范围 [-Lim,Lim]，参数值单列。M=M=.../W=W=... 同类重复符号一并小范围清理。

fix01 identity.json 沿用初版保护计数2483且freeze_count=0，不能作为本次2635保护的证据；以实际保存的保护清单复算为准，不修改原证据。产品事实已正确的部分不重做，不扩大模型/图/Excel/ONNX或门禁。
