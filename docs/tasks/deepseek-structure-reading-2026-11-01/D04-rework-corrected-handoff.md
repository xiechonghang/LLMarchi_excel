/archify
这是D04正确原会话“实施 D01 图形显示修复”的返修派单纠正。只D04/#20，不操作E01、不D05。

Codex此前把D04-rework-handoff误投到E01会话；用户选择手动停止E01，当前GUI无停止按钮。误派期间已修改两个主仓文件：render/archify_layers.py和tests/test_deepseek_reading_page.py，并触发构建。不要回滚/覆盖这些部分改动，不把它们当已验收；它们也不是其它独立功能的已冻结基线。

先完整读原D04-rework-handoff.md及D04-codex-review.md（四项技术返修仍全部有效），再读D04-codex-verify/misrouted-partial-changes.diff。新接手快照 /private/tmp/llmarchi-D04-corrected-baseline-4spkipj8/manifest.json，包含当前两个文件的部分修改与28870现有输出，继承原返修基线/private/tmp/llmarchi-D04-rework-baseline-a280g_p4/manifest.json。以原基线评估完整返修差异，以新快照防止覆盖当前工作；全部旧候选/Excel/证据保护。

在本正确D04会话核对并继续部分修改，先确认没有并发构建写相同目标；不要操作/停止E01会话或其进程。已有新产物未验收，不覆盖。若旧部分构建完整可只读检查，但最终交付必须匹配最终修复源码身份、全部四项复核要求与实际证据，不把半成品直接声明通过。

四项原要求摘要：真实moe_combine/index_add及token_idx回填；关系层单专家expert_down_out的正确形状和来源；原始scores gather/归一化/缩放与Hash/TopK差异主图可见；所有MoE边真实身份+完整shape；共享Clamp/局部参数及mHC就近来源恢复。保804/804显示已过成果与D01-D03，不重做其它内容。

仓库/模型/权限、dirty/既有删除/E01隔离、原四项测试与门禁要求不变。交付仍用 D04-rework-result.md、D04-rework-verify/，并诚实记录接手存在误派部分变更、哪些核验/修正/复用。Codex协调JSON/报告/旧证据/handoff只读。不Git、不关Issue、不改自动化、不E01/旧T14、不D05；完成停止待技术复核与用户视觉审查。
