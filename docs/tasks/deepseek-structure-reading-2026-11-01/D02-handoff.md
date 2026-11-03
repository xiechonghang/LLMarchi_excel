/archify
D01/#17 已经 Codex 独立复核通过，现按既定计划只派发 D02/#18：Decoder 与 mHC 完整分支数据流。继续在本会话实施，当前会话标题可能仍是“实施 D01 图形显示修复”，本条及D02-execution.json作为任务身份。不是继续D01返修，不做D03–D05、不推广其他模型。完成D02停止等待Codex复核。

仓库 /Users/xiechonghang/Workspace/LLMarchi_excel。先读AGENTS.md、领域文档、docs/tasks/deepseek-structure-reading-2026-11-01/d02.md及父规格 docs/requirements/deepseek_structure_reading_integration_spec_2026_11_01.md、D01-label-codex-review.md。GitHub #18正文与原生依赖关系为准。接手HEAD947ae78f120279403ec0a4c539e99d3d8869e1d6；最新已接受候选 2fa51e64bd77da2b9328cd2548322311e37ead087833750185225ba1df32c97d。新快照 /private/tmp/llmarchi-D02-baseline-v5dfmeiv/manifest.json，先核对并继承旧保护清单，dirty tree/既有删除/Excel/全部历史候选不覆盖不清理。协调文件外部更新只读。

本票从正式HTML进入Decoder：按四个真实层组保留实际结构区别，清楚展示attn_hc与ffn_hc各自 collapsed→Norm→Attention或MoE分支→branch_output→post展开；comb转置混合原始多流后Add。明确mHC.forward只返回post/comb/collapsed，真正分支调用与残差合并在Decoder；不能画成mHC内部调用Attention/FFN或虚构回调。数据流向前展开，无向上回流。mHC权重细图展示具名投影、拆分、pre/post门控、comb和Sinkhorn循环，形状/消费者/来源逐边核对。模块未展开须明示且可进入实际细图。参数用全称且图边就近。复用已存在结构事实/投影器/生成器，必要扩展，别再建一套手工图。

Attention/MoE内部本票沿用现有可进入内容，D03/D04分别改进；不要把既有Attention顺序布局误当最终关系图已验收。可复用子图而不引入假的Q→KV数据依赖。本票新增/变化的Decoder/mHC必须语义和分层完整。

必须保留D01两阶段符号、整网端点、页签独立、来源、权重依赖、全图库不省略文字、标签防重叠等修复。共享渲染变化须回查全图库：每条边标签贴近其所属边，节点/完整文字/路径/箭头外沿/标签包围盒不重叠或越界；mHC左侧裁切专项回归，平行/汇入/长标签都检查。原生图与最终嵌入两入口、1440×900及1920×1080、100%浏览器缩放、查看器缩放与长图滚动都需验证与截图目检。

复用只读探针 docs/tasks/deepseek-structure-reading-2026-11-01/D01-final-codex-verify/verify-final.mjs （新候选绝对路径、全新证据输出目录）；需要根据D02新增真实入口扩展新验收而不是删断言。29张原生图门禁以及项目专用最终入口探针是不同范围：Archify通用browser-check把长文档纵向滚动判overflow，D01报告已诚实记录，不许把该条写成通过。环境限制诚实记录，Codex可补跑，不关闭浏览器sandbox保护。

构建新候选，提供D02-result.md与D02-verify/：正式HTML入口、源码/配置与构建命令、需求对应/语义依据、全图库清单、回归测试、来源绑定浏览器证据、逐图目检清单、Excel和历史保护核对、限制。路径前缀 docs/tasks/deepseek-structure-reading-2026-11-01/。不改Codex协调JSON/既有验收证据/handoff；不关票、不commit/push、不恢复自动化、不访问或恢复旧T14会话。完成停待复核，不自行接下一票。
