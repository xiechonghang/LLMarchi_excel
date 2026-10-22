# 反馈06复核02：修复01范围改善，完整转换仍changes_required

原Harness同App/1017 High修复01正常完成9b1bc6532fa545febb49407e3b59d62a43654e67f38e7e992635a3cda5bac088。Codex核144资产链接按viewer路径全存在、72模型无重名、963旧交付无漂移、三数据旧同、原36SVG逐字节嵌入相同；这些修复保留。fix01新增7个真实生产builder小例已覆盖CSA/Indexer/mask。证据fixed01_basics.json；其中旧left_svg_equal=false是review过滤错了真实phase-onnx marker，明确不采信，校正见left_svg_correction.json，不是产品缺陷。

继续小合成输入实际production builder复核，发现原图转ONNX其余明确遗漏，证据preliminary_runtime.json、additional_runtime.json（无HF、无真实权重、无候选修改）：

- attn_core：B1/T3/Nh2/Dh4，sink_softmax/Concat实际ORT拒绝，Inferred=1 Declared=3 Dimension=2。HF:730先expand sinks到[B,Nh,T,1]，当前只有reshape。
- attn_out：B1/T3/Nh2/Dh4/G2/Ro2，attn_inv_rope/Mul Incompatible dimensions；HF:865逆RoPE前后transpose(1,2)，当前缺失。
- attn_out独立grouped例：B1/T2/Nh2/Dh6/G3/Ro2，o_a_proj/MatMul Incompatible dimensions；HF:300-329先reshape(-1,G,in).transpose(0,1)，按G做bmm后transpose回。当前直接[B,T,G,in]@ [G,in,Ro]把T误当batch匹配G，不能由改输出shape遮错。
- moe_route：真实小图NonZero返回hit_experts[1,2]，HF:1044 nonzero是[2,1]，缺转置。ffn_flat作为其它模块消费的边界产物也未导出。
- moe_expert：真实小图routed_view实际[3,4,1]，宣称/JSON是[1,3,4]。Reshape([0,0,-1])复制了flat的BT/H，缺原B/T运行信息。还把expert_token_idx/topk_pos直接当外部输入，省略JSON的moe_select_tokens从expert_mask[expert_idx]作where生成两下标的依赖；输入已被选好不等于实现此选中逻辑。原JSON模块已经包含该选择步骤，不是要求扩全专家循环。
- source mapping：8份CSA/Indexer原/简文件的实际node前缀在mapping.groups里缺失。_window_common gmap同名src==target也执行del，把c_kv_gate_proj/c_store_window/c_pool等有效组删掉；报告mapping0null不能证明实际node均绑定。需要按实际ModelProto逐node查映射，不仅枚举mapping中现存项。metadata已共享不解决误删。

这仍是本轮实际ONNX转换正确性，不是新增模型/布局目标。保已修窗口和mask/链接/唯一名/左图/三数据；只同Harness有限补齐注意力布局、MoE边界/shape、实际映射，并复用新增小fixture核当前18个builder，避免只验证三类后就宣称全部通过。全尺寸真实权重/HF不执行，浏览器受限状态仍不代签。
