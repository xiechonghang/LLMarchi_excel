/archify
继续当前会话，只实施 D01/#17 的全图库连线标签布局返修，完成后停止等待 Codex 复核。项目 /Users/xiechonghang/Workspace/LLMarchi_excel。

先读 AGENTS.md、领域文档、正式规格、d01.md，以及 docs/tasks/deepseek-structure-reading-2026-11-01/D01-final-codex-review.md。该复核结论是本轮范围依据。用户要求后续开发仍委托 dsh；D01 未验收，暂不实施 D02–D05。

最新基线候选 f1ea240e2c0c328b075863a963526a1e64bdb717e1584b50f3b6d36821279614；HEAD 947ae78f120279403ec0a4c539e99d3d8869e1d6。新冻结快照：/private/tmp/llmarchi-D01-label-baseline-3vkf8qpl/manifest.json。先核对并继承其中两份旧保护清单：22242 历史产物、此前候选、现候选、Excel 均保持逐字节不变；新构建使用新候选目录。接手前 dirty tree 和两处既有删除不动。

根因证据：最终入口两视口交互358项通过，但截图发现真实标签重叠。新增只读探针 D01-final-codex-verify/verify-final.mjs；label-audit/result.json 424项中26项失败，每视口13张图66对重叠。具体图清单及定位见复核报告；page-verified/*-full.png 为实际嵌入截图。上述路径均以 docs/tasks/deepseek-structure-reading-2026-11-01/ 为前缀。

修复共用边标签布局，覆盖全部29张唯一图/33图位，而非仅截图举例。为平行边、跨层长边、汇入边提供可用标签位置与间距，必要时调整路径/画布/层间距；标签靠近其对应边，关联明确。避免标签互相遮挡、遮住节点、越界或被嵌入裁切。不得删掉真实边/参数/来源、用省略号或隐藏文字、过度缩字解决，不得声称设备融合。保留既有阶段切换、符号化、端点权重和来源修复；不在本轮扩展其他模型或提前实现 D02–D04 语义。

增加有意义的平行/汇入/长标签布局回归验证，重新生成候选、Archify 原生校验及可运行的浏览器证据。若浏览器仍受环境限制，诚实记录，Codex 可补跑；不可仅凭静态校验声称视觉通过。可运行上述探针：node <verify-final.mjs绝对路径> <新候选/viewer/structure.html> <新的证据目录>（--geometry-only 可只查几何；完整验证仍需截图及目检）。探针只读，不修改其断言来掩盖失败；CSS隐藏的 fine 标签不属于可见文字错误。

交付 docs/tasks/deepseek-structure-reading-2026-11-01/D01-label-result.md 与 D01-label-verify/，报告新候选、修改文件、问题对应、测试、限制、保护核对。所有 Codex 验证证据、协调 JSON、handoff 文件只读，外部并行更新不动。不关 Issue、不 commit/push、不恢复自动化、不操作旧 T14 会话。完成即停待复核。
