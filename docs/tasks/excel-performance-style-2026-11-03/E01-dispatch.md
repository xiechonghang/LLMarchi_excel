# E01 / #22 DSH 实施派单

本任务由用户在2026-11-03明确要求新建DSH会话派发。只实施E01，完成停待Codex复核；不启动E02/E03。

项目 `/Users/xiechonghang/Workspace/LLMarchi_excel`；先读AGENTS.md、docs/agents/domain.md、GLOSSARY.md与本目录plan.md。GitHub任务：https://github.com/xiechonghang/LLMarchi_excel/issues/22 。模型沿用DSH当前DeepSeek-V4.1-Flash 1017 High、工作区内修改模式。

先建立独立快照，建议写在 `outputs/excel-performance-style-2026-11-03/E01/workspace/`，只复制所需源码、测试、依赖元数据与输入，包含当前dirty/untracked模块，不递归复制outputs/.git/虚拟环境。记录逐文件来源与sha256，开发在快照中执行。不能把Git HEAD当成当前完整源码。原工作区只允许新增本任务的E01-result.md、E01-verify/及独立outputs子树；不能修改原src、tests、CLI/构建入口、其他任务文档、协调JSON或旧证据。所需源码修改留在快照中并生成清晰patch待集成。权限不足时如实说明，不扩大访问模式。

输入优先冻结既有同一候选的layer_structures.json/network_overview.json，候选可从 `structure_reader/outputs/deepseek_v4_flash_0731_hf/c47633db4bec814108f664efb4a0898f862c97f49eeaa860291868fa872b2486/` 查验；文件不存在则只读查找完整候选，不猜造数据。记录确切候选、数据与依赖源码哈希。只读复用hf_workbook.build_structures_workbook_spec及既有性能映射，禁止重建HTML。

读取根目录dpskperfexcel草稿.xlsx作为布局参考，数值/层型/结构均为示意。按plan.md实现DeepSeek Prefill/Decode新样表：集中参数、右侧硬件吞吐、按真实通路组织，主表12列，每行一个算子；batch=1，n仅表示head或1；element/ms；CUBE head/其他batch因子折算一次，VEC n*T*D；两吞吐独立；op(ms)=MAX两项；同op复合矩阵项SUM/repeat后入CUBE列。未知不可被MAX/SUM吞成0。层数与串行总计、条件与excluded/container按真实规则，不重复计费；保留必要辅助明细与可追溯信息，别把27列原表原样搬来。

已有工程使用openpyxl；遵守本会话可用技能与工具约束，在隔离快照复用现有导出实现完成布局，不替换全项目技术栈。不为本票修改模型事实、共享schema/求值器、extract/render或其他模型路径。白底黑字、紧凑、适量细边框，以草稿阅读方式为准。旧正式导出保持原行为，正式集成为E03。

交付同一独立run目录中的xlsx、源码patch、冻结清单、复现命令、测试/渲染证据和未覆盖限制。报告写回原项目 `docs/tasks/excel-performance-style-2026-11-03/E01-result.md`，证据写E01-verify/。涵盖不等吞吐联动、head/分组/复合项与repeat、条件0/未知、实际层数和汇总、旧导出行为；渲染并查看各工作表，原生Excel未测则明确。并行HTML引起的原目录外部变化只记录归因，不回滚或纳入自己的patch。

不访问或续写DSH旧HTML/T14会话，不改D01—D05状态，不关Issues、不commit/push、不恢复/创建自动化，不清理旧产物，不自动接下一票。完成本票停止等待Codex复核与用户阅读。
