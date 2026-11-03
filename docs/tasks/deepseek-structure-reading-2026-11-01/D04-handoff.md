/archify
D03/#19 已技术复核通过且用户确认“视觉没有问题”，已验收关闭。继续当前原会话，只实施 D04/#20：MoE关系、Hash/TopK与专家内部。标题可能仍为“实施 D01 图形显示修复”，本条与D04-execution.json作为任务身份。完成D04停止待Codex技术复核及用户视觉审查，不启动D05。

仓库 /Users/xiechonghang/Workspace/LLMarchi_excel。保持DeepSeek-V4.1-Flash 1017 High、工作区内修改模式。先读AGENTS.md、docs/agents/domain.md、GLOSSARY.md、d04.md、父规格docs/requirements/deepseek_structure_reading_integration_spec_2026_11_01.md、D03-acceptance.md及D03-rework03-codex-review.md。GitHub #20正文为准；D01/#17已关闭满足依赖，D03也已关闭。仅DeepSeek，不推广其它模型。

接受基线73034fda94f87647a270cee4e88035c8e1fee83c927dde6c3feb5f6019bc8ca7。新冻结/private/tmp/llmarchi-D04-baseline-_y3z11q3/manifest.json：222源码/测试/配置、27208现有输出文件、5144本系列协调/证据文件；源码副本files/。先核对继承所有历史保护，dirty tree/两处既有删除不清理。旧候选/Excel/证据只读，新构建新身份。外部动态.DS_Store/监控状态变化如实列出，不回滚或冒称全部不变。

## 本票交付

1. 正式第三层MoE只展示模块关系：gate/路由、激活专家、共享专家、输出加和。激活专家与共享专家都从真实hidden_states派生，不伪造gate→共享专家数据依赖。路由indices/weights分别传向dispatch/专家与加权，不能只把流程画成一串无语义箭头。关系层复合模块明示未展开并有实际有效下钻入口。
2. 第四层有Hash/TopK路由、单激活专家与共享专家的真实独立通路及返回。Hash层按tid2eid[input_ids]选择，保留input_ids/表来源；TopK用scores+e_score_correction_bias选择。两种都用原始scores gather计权，再归一化和routed_scaling_factor；不能用纠偏分数计权，不能给Hash造TopK选择。scores的实际激活函数对照config/源码，不猜造。
3. 展示token dispatch/hit expert/专家输入的真实依赖与符号规模。expert_token_count保持符号；只对实际命中专家循环，保留token_idx/top_k_pos及哨兵分支，不能假设所有专家或全token都参与单个激活专家。条件与循环次数/边界可读，不机械复制专家节点。
4. 激活专家按当前DeepseekV4Experts展示gate_up_proj[expert_idx]→chunk/split→gate的上界Clamp、up的双侧Clamp→实际激活×up→down_proj[expert_idx]；其输出乘对应路由权重后index_add_回填，随后View恢复batch/seqlen。共享专家DeepseekV4MLP使用独立gate_proj/up_proj/down_proj和全token输入，不当成只吃激活token的同一模板。最终routed+shared加和顺序正确；外层mHC残差仍属于Decoder，不搬进MoE。
5. 四真实层组正确进入自身Hash或TopK；每图只放实际必要全称参数与来源，包含专家维度、num_experts_per_tok、intermediate_size、routed_scaling_factor、swiglu_limit等适用项。不虚构Engram/latentMoE。各图实际张量身份、符号维度、所有消费者、权重/源码联查完整。新图与旧表粒度差异登记，Excel公式/计费/汇总不改。

## 复用与防回归

直接扩展既有结构facts、semantic bundle、确定性投影/生成器和正式构建，不写第二套手工张量inputs/outputs或原型坐标。若复用阶段图，以真实op成员推导边界；未知id具名报错，只有真实[]shape为标量。同一张量被图内与图外消费时，两类边都保留；纯内部tensor不伪造段出口。必要阅读投影标明是投影且给真实来源。

保留D01/D02/D03全部已接受内容：两阶段符号定义、四层导航/来源、整网端点、Decoder/mHC职责与Sinkhorn、Attention/QKV-O、SWA/CSA/HCA/Indexer、真实阶段边界与缓存累计投影。只必要DeepSeek事实/语义/阅读图/正式build集成及对应测试改动；不做通用重构/新平台。共享渲染影响时机器回查所有受影响图；特别检查Hash/TopK游离标签、MoE裁切、完整文字、节点/路由/箭头/标签外沿、所属线距离及实际投影字号，保持原6px门槛。不删边/隐藏标签/缩字或无限扩画布过关。

E01正在另一会话隔离进行layout_v2：禁止修改outputs/excel-performance-style-2026-11-03/、docs/tasks/excel-performance-style-2026-11-03/、原export公式/共享求值器/Excel布局，不访问E01会话，不自动集成其patch。外部新增文件只记录，不回滚、不归入D04。确需影响旧Excel或跨域事实时报告具体差异，不静默修改。

## 验证与交付

主边界是源码/config→完整正式构建→最终离线HTML。按真实Hash/TopK、路由计权与单专家/共享专家边界补有意义回归，检查最终candidate/relations及可达入口，不仅内存字段存在。正向依赖和同类缺陷一次全查，避免单图样例通过。复用已有检查，必要测试一次通过后不反复跑不相关整套。

原D03只读探针D03-rework03-verify/verify-final.mjs可复制到D04-verify，仅按真实manifest新增图位更新数量契约并补D04必要行为，不删旧断言、不降阈值、不修改历史探针。图数按真实交付，不硬守51/55。原生与嵌入是不同范围，最终两个视口1440×900/1920×1080、导航/缩放/滚动和实际bbox需机器检查并绑定新HTML字节。复合长页通用browser-check纵向overflow限制与单张图字号失败严格分开；原生图仍需真实门禁。dsh环境受限一次尝试即可记录待Codex补跑，不反复已知失败、不关闭sandbox。

视觉由用户审阅，Codex不逐张读取长截图。交精简新增/变化审图入口，不以未逐图人眼审查拖延交付，未做视觉项写待用户确认，不能冒称视觉通过。

交付 docs/tasks/deepseek-structure-reading-2026-11-01/D04-result.md 与 D04-verify/：新候选/正式入口、需求与源码映射、实际构建命令、全图库清单、测试与原生/嵌入证据、五类关键数据依赖/符号/条件对照、精简用户审图入口、Excel/历史保护及限制。既有Codex报告/证据、协调JSON、旧handoff只读。不关Issues、不commit/push、不创建/恢复自动化、不访问旧T14；只实施D04，完成停止待复核，不接D05。
