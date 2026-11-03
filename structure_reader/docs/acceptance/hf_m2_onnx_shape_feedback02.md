# M2阅读反馈续：ONNX维度数值实核

2026-10-26。用户回复上次澄清问题：“缩写和维度数值都需要检查”，并要求继续。此回复覆盖之前“仅待用户具体指出数值”的当前状态；完整名称612候选保持，但它只证明名称修改没有改变旧数学，不构成数值/shape语义验收。M2用户整体阅读未通过，M3未启，自动PAUSED。

本轮先对根config、固定HF实现和实际ONNX核静态参数/形状及关键轴变换。已读实际612文件发现：kv_weight仍为符号[head_dim,hidden_size]，尽管512/4096是确定配置；o_a已写[8192,4096]但o_b仍符号；position_ids固定轴'1'、mHC scale的'3'等以dim_param保存。这是静态已知信息未正确落入数值维，而非已发现512/64/32局部数值错误。需把已知配置维、纯静态组合和数字字面量写真正dim_value；batch/token/缓存/专家命中等运行时长度继续完整符号，不填猜测值。

除元数据外，核截图KV/RoPE的rank/尾64与前448/32频率/广播及main-vs-YaRN常数，反向RoPE、GroupedLinear组轴/权重；四组Q/KV/mHC/MoE/CSA/HCA/Indexer的关键参数shape按HF来源逐项对照。有限小尺寸生产builder配独立NumPy期望，检查实际值与形状，禁止仅checker自证、仅原简互比或新resolver自抄expected。未导入运行HF/真实权重，不能宣称全模型数值等价。

基线6126726e…；任务`.work/coordination/hf_m2_onnx_shapes01/task.md`及before/findings/11975旧候选保护。继续原Harness App1017 High，Codex派单验收；新内容身份、受影响检查与两真实构建后交阅读。原图/JSON/Excel先保持，若上游事实错误须具名另核范围；旧候选/冻结只读，需求/计划/progress由Codex作本次授权更新，未Git推送，不绕过浏览器/协议限制。
