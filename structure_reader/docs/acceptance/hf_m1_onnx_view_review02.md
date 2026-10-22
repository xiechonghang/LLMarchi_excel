# HF M1 ONNX式展示复核02

2026-10-19，Codex。结论：**changes_required**，仅剩已要求的源码名称完整可见问题；暂不派展示独立复验。用户阅读反馈02仍未关闭，M2/M3未启。

GLM在原可见主干会话正常完成收尾01并空闲，交付候选 `31ac27674ab4d4c4285b9b4cbb24d03f86e6b3766b4c3ae202ec26df16bc2ad4`。相对修前6个拥有路径，仅diagram.py、diagram_source_labels.py、图测试改变；其余不变。

已复核的正确部分：

- Codex实际41图测试、44受影响页面测试均rc0；原始GLM两实际build日志rc0，同新ID，第二次reused。当前7交付字节与纯builder一致，manifest、13工具源匹配；三数据文件与旧认可版本逐字节相同，77旧候选文件保护无变化。
- 实际SVG中324个有引用圆点的最近同tensor-href路径端点距离全部≤7px，原q_weight圆心对应端点距离6px，162px悬空消除。这个口径只证明圆点与其引用路径某端相接，**不宣称所有方向已从JSON独立证明**；新圆点/IO语义仍需后续独立适配端点核查。
- 628实际SVG边标签可见行拼接与全文一致，50条件标签同时保留tensor名/shape，无旧12类维度截断；本轮实际计算类型未截断。已具名4项源码别名按原静态来源修正；没有重审模型。
- 暗主题flow/weight/cache的实际CSS箭头、marker、图例成对，分别对暗card静态对比6.55/8.03/8.75；框主字14.49、副字7.39保持。是静态配色证据，非真实浏览器字体/CSS视觉确认。

剩余一个同类遗漏：修正后的长源码名仍被 `_wrap` 的 `lines[:3]` 截断。实际候选SVG确认Prefill/Decode各5处（10次出现，不拆10项任务）：

- `moe_expert_gu`：完整 `F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx])`，实际框内停在 `self.gate_up_proj[expe`。
- `moe_expert_slice`：第二个 `expert_idx` 末尾被截。
- `i_topk`：末尾 `-1)` 被截。
- `attn_residual_mix` / `ffn_residual_mix`：后半段矩阵乘和hidden_states被截。

这不是仅title里有全文即可关闭的问题，用户要求框内用源码名。应仅补节点名称换行/高度的完整显示，保留已正确箭头标签、配色、连接、数据和粒度，不开启新布局目标或模型任务。

证据：`structure_reader/.work/coordination/hf_m1/codex_onnx_view_review02/` 的 `result.json`、`text_and_ports.json`（实际SVG文字）、`colour_pairs.json`、`raw_log_review.json`及两复跑原始命令日志。browser_visual=not_verified；native_excel=not_verified；user_reading=not_passed。
