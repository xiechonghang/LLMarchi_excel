/archify
继续当前会话，D02/#18 Codex复核未通过，只返修D02，不启动D03–D05。仓库 /Users/xiechonghang/Workspace/LLMarchi_excel。

先读 docs/tasks/deepseek-structure-reading-2026-11-01/D02-codex-review.md，其中四项阻塞均必须修复；该报告覆盖原D02-result.md中未经浏览器实证的结论。d02.md、父规格和AGENTS.md约束保持。

当前候选 0eb1ca6461639b112c52c4bd7df0494ce2b971425307b6fdcc8d1d9e6ecfe1b7；新冻结快照 /private/tmp/llmarchi-D02-rework-baseline-nll29jib/manifest.json，继承D02及D01所有保护清单；新构建写新候选，不能覆盖当前候选或旧证据。HEAD947ae78f120279403ec0a4c539e99d3d8869e1d6；dirty tree和两处既有删除保留。

修复重点：
1. 恢复所有相关主图边的全称符号shape，同时保留post/comb/collapsed/branch_output等身份。不能semantic_first后只留名字；元数据/tooltip/完整文字面板不能代替主图维度。适当分行/扩画布/留出标签空间，保持完整可读与归属明确，四组Decoder及mHC全查。
2. 四组mHC两视口标签碰撞：f-7-o-in-o-attn-hc-collapse与f-15-o-in-o-ffn-hc-collapse实测3.386×10px。D02-codex-verify/page/result.json里424中8失败。用真实browser bbox复核，不能改探针门槛或只依赖同一个字符宽度估算写测试。
3. 两张原生CSA Decoder失败：decoder-csa-hash、decoder-csa-topk，1440投影字约5.9426px<6且原生阅读器overflow；浏览器可用不是环境失败。保留真实拓扑和文字改善布局/原生阅读，重跑原生gate。不要将通用复合页预期长滚动的处理方式泛化为忽略原生可读性失败。原回执在D02-codex-verify/native/<slug>/receipts/。
4. mHC就近参数需hc_mult、hc_sinkhorn_iters、hc_eps及实际本图必要维度与来源，精简Q/KV/o_groups等无关项；Sinkhorn图示一次初始列归一+(hc_sinkhorn_iters-1)次行列归一循环（核对modeling:975–979），用前向循环容器/注释或细图，无向上回流、无20份节点复制。保持具名算法粒度；不能仅在源码面板藏循环。

保留已正确的mHC.forward与Decoder.forward职责、两段残差、四层导航和D01共享修复。新增有意义的显示与参数/循环语义回归，完整构建新候选，29图/所有图位回查；实际浏览器受限就记录待Codex补跑，不能声称视觉通过。全部证据绑定新候选和HTML字节。Excel和历史逐字节保护。

交付D02-rework-result.md及D02-rework-verify/（均在同任务目录），列出四问题逐项响应、正式入口、测试、原生门禁和截图/目检范围、保护核对和限制。既有Codex证据/协调JSON/handoff只读，不删除原报告。不关票、不commit/push、不恢复自动化、不操作旧T14。完成停待复核。
