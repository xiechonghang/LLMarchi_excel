/archify
继续原会话，D03返修技术复核尚未通过，只续做剩余两类技术问题，不重做已通过布局/入口，不启动D04/D05。

完整读取 docs/tasks/deepseek-structure-reading-2026-11-01/D03-rework-codex-review.md 及 D03-rework-codex-verify/stage-fact-audit.json。新快照 /private/tmp/llmarchi-D03-rework02-baseline-k3jd784o/manifest.json；继承先前全部保护，8cfa840daf2cefb28ad298670faff9ae45765da15ddebf3af8e280ae57d03613只读保护，新构建写新身份。仓库 /Users/xiechonghang/Workspace/LLMarchi_excel，当前模型/权限保持。

已确认必须保留：55图位真实浏览器644/644通过，HCA压缩器原生browser-check通过，48项阅读测试通过；真实Attention block_bias连接/HCA区别/Q/O入口已经修好。Excel与22801历史文件未变。不要再做无谓图框/导航重构或重复旧返修。

剩余P1：_stage_graph手工inputs/outputs和事实不一致。CSA不存在c_mask_extend op与block_bias tensor，Indexer不存在i_weights，却被空shape兜底画成标量。compressor和Indexer从窗口阶段向pool传的是chunk而非真实windows；CSA/Indexer缺overlap/entry_count输入，Indexer q缺cos/sin，scorer缺attn_normed，topk缺自身compressed状态。精确逐阶段差异在审计JSON。使用真实op_ids的inputs/outputs/生产消费关系推导聚合边界，不继续手写一套阶段边界；必要分组仍可保留，所有合法外部依赖/输出/状态消费者不可删。不存在id须报具名错误，真实[]shape才是标量。修复实际最终candidate/relations，不只bundle。补失败回归：未知op/tensor拒绝、CSA真实bias rank、Indexer scorer真实输入、windows形状、全部跨阶段/边界输入覆盖；保已恢复完整trace断言。不要为测试通过把差异expected排除。

剩余P2：cache_cumlen目前仅调用前，报告声称调用后并不成立；就阅读投影补调用后kvcachelen+seqlen及更新依赖，保旧Excel不变。缓存逐项表必须复用全称符号展开（目前仍B/Dh/W/Tbuf），来源从真实读写算子/已有映射追溯，不能读取不存在的tensor.source_refs后全“—”。保已有新KV/full输入输出与分图，不把逐项维度再藏JSON。

父规格/原D03-handoff及E01隔离、dirty/既有删除、历史证据/Excel保护仍有效。Codex报告、旧证据、协调JSON、handoff只读。本轮只必要源码/测试改动，复用确定性事实和正式构建；不要建立新通用平台/复制事实。交付 D03-rework02-result.md 与 D03-rework02-verify/：两类问题逐项结果、新候选/正式页、真实构建命令、必要测试/机器门禁、全源绑定、保护核对及精简用户审图入口。环境限制一次尝试即可记待Codex，不反复无效启动；布局保持通过门槛，修复新候选重新绑证据。

用户负责人眼视觉，Codex不逐张读长截图。完成后停待技术复核及用户视觉审查，不关Issues、不commit/push、不创建/恢复自动化、不操作E01/旧T14、不开始D04/D05。
