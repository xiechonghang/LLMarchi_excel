/archify
继续当前会话，只返修D04/#20，不启动D05。Codex技术未验收，先完整读 D04-codex-review.md，四项一次解决；D04-handoff/d04.md与父规格保持。

项目 /Users/xiechonghang/Workspace/LLMarchi_excel。当前f5b8f0da862daf292a2eb342906218d9204124d9f40ab4cba3663e85e4319fe4只读；新快照 /private/tmp/llmarchi-D04-rework-baseline-a280g_p4/manifest.json，继承27208历史输出及全部证据/Excel保护，dirty/既有删除/E01隔离保持。新构建新候选，不覆盖旧产物。

保留已过的71图位浏览器804/804、57阅读测试、正确模块分层/Hash-vs-TopK/专家切片/共享并行。不要重做D03阶段输入/扇出、布局或旧功能。

四项：
1. 汇合细图加入真实moe_combine（index_add），输入expert_down_out+expert_token_idx、零final/命中专家累加后产生routed_out，再View，再+shared。当前细图无index_add，关系图单专家却传已全局回填routed_out，语义错误。关系层传真实单专家加权输出及dispatch token_idx到汇合，专家权重只乘一次，保持num_expert_tokens符号与哨兵条件，不把专家模板当整网结果。检查四组最终candidate/relations与操作覆盖，不能只标题写index_add。
2. 路由主图/可达细图明确logits→真实score激活→选择→原始scores gather→除sum+1e-20→routed_scaling_factor；TopK纠偏只选择，Hash用tid2eid不走TopK。当前只有一个“路由权重/选择”框与“1个真实算子”副行，不满足内部说明。可用主图公式/注释保持具名算法粒度，不要求机械微拆；复用facts、投影标明与来源，不造第二套张量事实或改Excel。
3. MoE全部渲染路径保真实张量身份+完整全称shape；特别route indices/weights及dispatch token_idx/top_k_pos等同shape输入，当前最终图只剩shape无法辨认。不能用metadata/tooltip替代主图，不隐藏/删边/缩字，不降原6px门槛。
4. 共享专家显式gate上界/up双侧Clamp、实际激活×up/down，参数提供swiglu_limit，移除shared/route无关hc_eps等。mHC局部参数就近config/源码说明恢复，不把D02来源改成“来源随参数卡”但卡无实际范围。各图实际必要参数与来源清楚，保既有模块职责。

用现有结构facts/阶段生成器，补有意义回归：真实index_add及消费者、每专家输出shape/回填token_idx/不重复weights、所有MoE最终可见身份+shape、路由公式和共享Clamp/参数/mHC来源。正向与错误行为一起覆盖，不删expected/旧断言。共享渲染变动全受影响图机器检查，保已有804门槛。

交付D04-rework-result.md和D04-rework-verify/：四问题响应、新候选/正式入口、源码config依据及实际构建命令、必要测试/机器原生和嵌入证据、历史/Excel保护、精简用户审图入口与限制。证据绑定新HTML；dsh浏览器受限一次尝试可记待Codex，不反复无效启动，不关sandbox。

用户审视觉，Codex不逐张读长截图。Codex报告/协调JSON/旧证据/handoff只读；不改E01树、Excel/求值器、旧T14；不关Issue、不commit/push、不恢复/创建自动化、不D05。完成D04返修停待Codex技术复核和用户视觉确认。
