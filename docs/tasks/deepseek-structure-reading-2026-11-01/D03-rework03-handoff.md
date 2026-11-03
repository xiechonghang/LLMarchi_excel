/archify
继续原会话，只收尾D03剩余一个输出扇出规则缺口，不重做已通过功能，不启动D04/D05。

先完整读 D03-rework02-codex-review.md。当前8136候选的真实输入推导、未知id拒绝、缓存累计投影/全称/来源已通过；新候选55图位644/644浏览器通过，52/52阅读测试通过，23013旧输出及Excel保持。此前两类问题已解决，不重做。

唯一P1：_stage_graph用图内consumed直接排除s-out，漏同tensor图内外双重消费者。独立审计stage-boundary-audit.json与final-boundary-evidence.json（D03-rework02-codex-verify内）确认：两组CSA Q图q_a_normed还要供图外i_q_proj；三组compressor的cache_compressed_c_post还要供图外kv_concat；当前五图段出口均丢。真实图内边保留，同时按所有block ops的真实外部消费者增加段边界输出/明确消费者；不要全量泄漏纯内部中间张量，不造新fact。保具名身份/shape/消费者来源与前向布局。

仅必要共享规则修复和一个有意义回归（内部+外部双消费，最终candidate/relations五图均有真实出口；纯内部tensor不冒充输出）。保持阶段输入推导、HCA/Attention/QO/缓存正确部分，保全部门槛/既有测试。新身份构建，旧8136及历史只读。

新快照 /private/tmp/llmarchi-D03-rework03-baseline-jp8h40gz/manifest.json，继承全部保护。项目 /Users/xiechonghang/Workspace/LLMarchi_excel，dirty/既有删除、Excel与E01隔离边界不变。Codex报告/协调JSON/旧证据/handoff只读。

交付D03-rework03-result.md、D03-rework03-verify/：规则修复及五图最终事实对照、新候选入口、必要测试/机器门禁、历史保护、精简审图入口。机器证据绑定新候选；浏览器受限一次失败即可交Codex补跑，不反复失败、不改sandbox。用户负责人眼视觉，不逐张长图。完成本票停止待复核，不关票、不commit/push、不自动化、不操作E01/旧T14、不启动D04/D05。
