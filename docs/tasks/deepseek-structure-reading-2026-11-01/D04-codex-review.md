# D04 复核：显示通过，语义/展开尚需返修

日期：2026-11-03（Asia/Shanghai）。候选`f5b8f0da862daf292a2eb342906218d9204124d9f40ab4cba3663e85e4319fe4`；正式页SHA256 `40f00919ead17351033a60298aba555361e5bf31897f5b59271eacb51481bd43`。D04/#20未验收，不关闭、不启动D05。

## 已通过与保护

- 双视口1440×900、1920×1080动态遍历71图位，浏览器804/804通过，0失败/运行错误，178截图；保留原6px门槛与旧交互断言。证据`D04-codex-verify/page/result.json`。
- 5文件diff独立阅读，阅读页面57/57测试通过；563全套为dsh报告，未冒充Codex全跑。
- 27208既有输出无变化，Excel与D03候选逐字节一致，当前来源清单匹配；5144协调文件只tickets.json为Codex派发外部更新。证据`D04-codex-verify/protection.json`。
- 保留正确方向：关系与内部页签分层、Hash查表/TopK选择差异、激活专家与共享分支并行、专家切片真实权重边、实际token数符号。新增图使用既有阶段facts推导，未重演D03手写输入清单。

## 阻塞项（一次处理）

1. **P1 实际IndexAdd缺失，关系层单专家输出shape/消费者错误。** 四组内部图所含阶段中均无`moe_combine`；当前`moe-combine-*`仅`moe_routed_view`与`moe_merge`，输入已是routed_out。因此按expert_token_idx把expert_down_out累加到零初始化final的真正`index_add_`没有任何D04细图承担。关系图r-expert→r-combine却直接传`routed_out [batch_size×seqlen,hidden_size]`，把单专家的`expert_down_out [num_expert_tokens,hidden_size]`冒充为已全局回填结果，且没有dispatch→combine的token_idx。对照modeling:1041–1055：down输出乘对应weights→index_add_(0,token_idx,current)→View→+shared。须在实际汇合细图加入真实moe_combine及token_idx/专家加权输出、零初始化/命中专家循环累加；关系层保模块抽象但边要表达正确边界与依赖，不用标题“index_add”冒充已有算子。

2. **P2 路由内部仍未展示计权算法。** `moe-route-*`把全部投影/score激活/gather/归一化/缩放包在一个“Hash路由权重”或“TopK选择”节点，副行只写“1个真实算子”。公开candidate卡/主图未显示scores按indices gather、归一化分母与routed_scaling_factor的顺序，不能看清纠偏选择与原始scores计权的区别；底层facts中moe_router.op已有这些行为。对照源码TopK1076–1084、Hash1105–1115：logits→实际scoring_func→选indices→scores.gather→除sum+1e-20→乘routed_scaling_factor。Hash只有tid2eid选择，TopK纠偏只用于选择。复用已有事实提供可见前向细图或就近主图公式/注释，不强制机械微拆，不伪造张量/改Excel。明确复合部分仍未展开时应有实际展开入口，不能以源码面板代替本票路由说明。

3. **P2 同shape不同张量在主图无法辨认。** 关系图dispatch→expert的expert_token_idx和expert_topk_pos最终都只显示`num_expert_tokens`，route→dispatch也只剩shape；内部图大多数边同样只有shape。虽然内存edge.label设置了semantic，部分渲染路径仍用shape覆盖身份，导致indices/weights、token_idx/top_k_pos无法直接识别。要求所有MoE关系/路由/专家/汇合边在实际主图同时显示真实张量身份与完整全称shape，特别这些同shape多输入。保持布局/字号门槛，不靠JSON、tooltip或完整文字表代替。实际交付摘录在`D04-codex-verify/moe-delivery-evidence.json`。

4. **P2 共享专家Clamp/就近参数缺失，mHC参数来源说明回归。** `moe-shared-*`只有“gate/up独立投影”“激活/down_proj”两节点，隐藏了共享gate上界Clamp/up双侧Clamp；参数卡反而包含hc_eps，无本图必需swiglu_limit。路由卡也混入hc_eps和swiglu_limit。用真实shared_gate_up/shared_act_down行为呈现共享Clamp与激活乘法，精简各图参数，保Lim与相关来源；专家总数/实际命中数/单专家token数概念清楚。`build_hf_decoder.py`同时把已接受mHC的局部config/源码范围说明替换为“config/modeling来源随参数卡给出”，但卡items无实际源码范围，需恢复此前D02就近来源，不把MoE统一处理造成的回归带入。

## 验证与停止点

机器显示通过不等于语义通过，测试目前只检查若干节点/边存在，缺少所有本票必需算子的覆盖和单专家/汇合边界对应。返修补最终candidate/relations的moe_combine/token_idx/shape/weights只乘一次及主图身份/路由公式/共享Clamp回归；不排除expected、不降门槛。

用户负责人眼视觉，本轮不逐张读长截图。发现明确技术阻塞后未重建全部67图原生门禁，不声称67张独立浏览器通过；dsh静态67/67保持其证据身份。先修上述问题，再给用户最终精简入口；不让用户审有已知语义问题的版本。

仅D04返修，保D01–D03、已通过804项显示及历史/Excel/E01隔离；不Git、不改自动化、不旧T14、不D05。
