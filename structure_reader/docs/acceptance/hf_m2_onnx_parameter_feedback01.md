# M2阅读反馈：实际ONNX维度名仍为缩写

2026-10-25。用户截图指出ONNX参数不合预期，延续先前“参数/维度不要B/H/Rq等缩写”的阅读诉求。当前基线570b4a27…，本轮尚未获得用户M2整体确认，M3不启动，自动检查仍PAUSED。

Codex实际读取四组ONNX：输入attn_normed仍[B,T,H]、KV weight仍[Dh,H]、position_ids仍['1',T]，推导中间形状也保留B/T/Dh和unk__N。之前完整维度名仅进入HTML/新Excel，未覆盖ONNX protobuf，因此该阅读遗漏成立。截图512=head_dim、64=qk_rope_head_dim、32=频率数量，与根config/partial RoPE对应；截图本身尚无某项数值错误证据。已异步请求用户指出是否另有具体数值/shape问题，当前按已确认的全名称要求推进。

仅修实际ONNX输入/输出/中间value_info的符号维度名称及必要文档/状态同步；复用当前完整名称映射，动态维不填猜测数字，不为好看重写数学或推导未知shape。Mul算子的标准输入端口A/B不是batch符号，保留ONNX标准schema。类型/算子属性/initializer/权重/真实已知数字保持，JSON/Excel/原HTML框图和源码联查保持。

新产品身份与原简ONNX、对应Netron链接需新候选，不覆写570b或历史。原Harness App1017 High实现，Codex仅派单验收；证据与修前/历史保护在.work/coordination/hf_m2_onnx_names01/。真实Netron GUI/浏览器阅读未代签，不绕过此前file/协议限制，不Git提交推送。
