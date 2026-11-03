/archify
D02/#18 已由 Codex 技术复核并由用户确认视觉无问题，已验收关闭。继续本会话只实施 D03/#19：Attention 总览、Q/KV/O 与 SWA/CSA/HCA 缓存。会话标题仍可能为“实施 D01 图形显示修复”，本条与 D03-execution.json 为当前任务身份。完成 D03 停待复核，不做 D04/D05，不回到 T14。

仓库 /Users/xiechonghang/Workspace/LLMarchi_excel；保持当前 DeepSeek-V4.1-Flash 1017 High、工作区内修改模式。先读 AGENTS.md、docs/agents/domain.md、GLOSSARY.md、d03.md、父规格 docs/requirements/deepseek_structure_reading_integration_spec_2026_11_01.md、D02-rework-codex-review.md。GitHub #19 正文与原生依赖为准。最新接受候选 c47633db4bec814108f664efb4a0898f862c97f49eeaa860291868fa872b2486，HEAD947ae78f120279403ec0a4c539e99d3d8869e1d6。

派发前快照 /private/tmp/llmarchi-D03-baseline-348utrl8/manifest.json：222源码/测试/配置，22641现有输出文件，2827本系列协调/证据文件；源码副本在该目录 files/。先核对继承保护，保留dirty tree和既有删除。现有候选/Excel/证据不覆盖，不清理；新构建写新候选。两项动态例外 .DS_Store/监控状态如有外部更新须如实归因，不能回滚或声称全无变化。

本票必须交付：
1. 保持 Q/KV 关联的 Attention 总览，按真实四层组展示 hidden_states、q_residual、Q、当前/可读KV、mask、sinks/scaling 的来源和消费者。Q 与 KV 可并行派生，不能造 Q→KV 顺序依赖或仅有末端箭头的断线。复合节点明示未展开并有真实有效入口。
2. 进入 Q/KV/O 具名 Linear、Norm、View/Transpose/轴布局、部分 RoPE、逆 RoPE、分组输出与完整全称符号 shape；辅助 cos/sin 局部接入，所有真实消费者与条件分支对照 HF 源码/config。来源通过现有结构事实/语义 bundle/投影器生成，不复制手工原型坐标或维护第二套事实。
3. SWA 无压缩器；CSA 保留 Ca/Cb 重叠、主压缩器与 Indexer 自身小压缩器/状态、mask选择；HCA 保持自身压缩与可见性规则。区分逻辑历史 kvcachelen、实际主KV/压缩条目/缓冲余量与调用前后状态。Indexer 正确接 q_residual/hidden_states 及自身压缩状态，不把外层压缩KV当唯一输入，不把TopK mask画成物理裁剪KV。
4. 真实层组均可从正式入口下钻/返回各对应通路、缓存、压缩与 Indexer；使用同一正式构建与产物清单扩展，不把原29张图数当上限。必要的局部参数与来源靠近本图，未交付D04内容保留明确说明。

保留 D01/D02 已验收功能：仅 Prefill/Decode、全称符号维度、seqlen 为本次输入、逻辑历史定义、两段 mHC 返回/Decoder 分支及 combᵀ残差、mHC局部参数与Sinkhorn循环、导航/来源联查、完整文字/共享标签布局。必要更改现有 DeepSeek extract/semantic/render/build 及对应测试；优先局部扩展，勿引入新框架或修改其他模型功能。若底层事实修正可能改变旧Excel，请先采用阅读投影保持旧导出；确需跨出本票边界则报告具体差异待协调，不能静默改表或复制伪事实。

并行 E01 Excel 改版正在另一会话，以独立 outputs/excel-performance-style-2026-11-03/E01/workspace 快照开发。禁止修改该输出树、docs/tasks/excel-performance-style-2026-11-03/、原工作区export公式/求值器/Excel布局，不操作E01会话。其外部新增文件不要回滚或纳入D03改动。本票读HTML管线，可以在现有build入口做必需图集成，不能把E01 patch自动集成。

验证：以源码/config→正式完整构建→离线入口为主边界，补真实语义/边依赖/shape/状态分支回归；复用现有测试与工具，不建新的通用验收平台、不反复重跑不相关整套。共享布局改变须机器检查所有受影响图，回归游离Q标签和缓存/Indexer负坐标裁切；完整文字、节点/路径/箭头/标签边界及关联、投影比例不能靠隐藏/删边/缩字通过。

浏览器与视觉分工按用户最新要求：机器检查原生与嵌入、两个视口1440×900/1920×1080、导航/缩放/长图滚动和实际bbox，保留最终候选截图；用户负责人眼视觉审查，Codex无需逐张读取所有长图。交付一份精简人工审查入口清单，重点指向新增/变化的Attention、Q/KV/O、三类缓存/Indexer。不要等待逐图目检才报告完成；未做的视觉项标“待用户审查”，不能伪称逐图目检通过。环境不支持Chrome时可由Codex补跑，不重复多次已知失败尝试，不关闭sandbox/绕过访问保护。

既有只读页面探针 docs/tasks/deepseek-structure-reading-2026-11-01/D01-final-codex-verify/verify-final.mjs 可复用，新图增加必要D03专项断言并写新证据；不能删改既有探针或降低门槛。原生图使用原Archify门禁，复合长文档通用browser-check滚动限制与单图字号失败严格分开。所有证据绑定新候选及HTML字节，不沿用c476截图冒充新版本。

交付 docs/tasks/deepseek-structure-reading-2026-11-01/D03-result.md 与 D03-verify/：入口/新候选、源码/config和实际构建命令、需求与源码事实映射、全图库清单、测试及必要浏览器结果、精简用户审图入口、历史/Excel保护结果、失败/跳过/限制。协调JSON、Codex报告、旧handoff/证据只读。本轮不关Issues、不commit/push、不恢复或新建自动化、不启动D04/D05，不访问旧T14。实施到可交付，完成停待Codex技术复核及用户视觉审查。
