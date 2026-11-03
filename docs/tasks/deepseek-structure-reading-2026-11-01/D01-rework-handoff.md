请继续 D01（#17）返修，不启动 D02–D05。先读仓库 AGENTS.md 和 docs/tasks/deepseek-structure-reading-2026-11-01/D01-codex-review.md，再读 #17、父规格 #16。Codex 复核结论为 changes_requested。

本轮修复：①页签切换/reveal 的全局选择器使其他章节图全部隐藏，隔离页签组并验证跨层往返；②Embedding、HyperHead投影及lm_head三个权重节点缺消费连线，按权重—消费者关系去重；③外部张量来源不能统一指向首个算子，应对应真实定义或消费位置；④公开12图243处主图省略号，完整文字表不能替代主图修复，按#17共同显示约束回查全图库。

不要提前实现D02–D04的语义细分。保留现候选、旧产物和Excel字节，保留此前dirty tree及两处删除。改生成器后构建新候选，验证实际交付HTML/JSON、页签及辅助输入、来源联查，补有意义的回归测试。原生/嵌入图两种桌面视口及逐张截图目检按规格执行；若浏览器环境仍拒绝启动，保留原始错误和未验收状态，不关闭sandbox、不绕过限制，不以静态证据冒充视觉通过。

报告写 D01-rework-result.md，证据写 D01-rework-verify/，均放 docs/tasks/deepseek-structure-reading-2026-11-01/。完成后停待Codex复核；不关票、不commit/push、不恢复自动化、不修改Codex协调文件。请先核对当前基线变化，所有D01-codex-*、execution.json、tickets.json为外部协调更新。
