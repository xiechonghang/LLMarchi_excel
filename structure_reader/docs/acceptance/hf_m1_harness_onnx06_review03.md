# 反馈06复核03：计算转换复核通过，尚有两项说明/映射收尾

候选3a7f214c282f4043b23a36e3b8b55a983a7cae7d5db88306b57a8d1e4a5c6c7d。Codex复跑受影响115项0skip通过；Harness基础tiny配置T=Nh=G=3，报告所谓“互不巧合”不采信，Codex额外改内存SMALL为B2/T3/Nh2/Dh6/G4/Ro2/Rk2复用5项实际builder/NumPy检查全通过，无候选/产品改动。实际Attention布局及MoE补齐保留。

实际资产核：72原/简ModelProto checker通过、无重复node；144链接按viewer真实目录存在；HTML id唯一与内链0断；原36 SVG逐字节完整嵌入，JSON/config/xlsx与032ea同；1267旧文件逐项无漂移。详codex_harness_onnx06_review03/{replay.py,result.json,artifacts.json,tests.stderr,unequal_axes.stderr}。

仍有限两项：
1 简化器插入Reshape_0/1共8次（CSA/Indexer×2阶段）没有JSON/source映射；原节点已全绑定。state已如实列sim_extra_nodes，但fix02报告“72模型0缺映射”过强，实际只有68份满足原检查。prefill/attn_csa的Reshape_0读取c_rope/cos_tile，Reshape_1读取c_rope/sin_tile；可依真实相邻producer/consumer的唯一c_rope来源诚实绑定“onnxsim插入，关联来源组”，不要假称源码原生节点。实际右图分组也应使用该已验证关联。
2 真实HTML没有“FP32”/“特化”字样，也没有明确告诉用户右框是语义分组；dtype=float32+opset文字不能替代其含义。MoE“单专家模板”和mask默认子集限制主要留在报告，应该给紧凑可见标签，避免误解为完整可运行量化模型或全专家循环。只加必要阅读说明，不增加满页技术说明。

计算图数值小例/源码静态映射、真实HF数值等价（未做）、浏览器（blocked/not_verified）、用户阅读（pending）分开。仅收尾映射和紧凑页面含义，不再重跑旧整套或改计算builder。
