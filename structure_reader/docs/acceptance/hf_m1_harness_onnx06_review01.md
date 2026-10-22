# 反馈06实际ONNX初版复核01：changes_required

候选5b9129483cf89f36e47aaee64caf52fb209253263c5138c410224fda89f61ee0，原Harness App/1017 High正常完成。实际生成36组原/简ONNX、对应HTML左右栏、工具隔离安装与原三数据保持的成果保留；不是最终通过。

Codex具名证据：.work/coordination/hf_m1/codex_harness_onnx06_review01/first_diagnostics.json 与 runtime_diagnostics.json。只使用实际builder的小尺寸合成参数，未执行HF/读取权重/改候选。第二诊断仅在内存把节点名改唯一以露出后续错误，不作为产品修复。

1. 页面在viewer/decoder.html，144个href写onnx/...，实际资产在候选根onnx/，144链接按页面目录解析全部不存在，应../onnx/...。原测试只数href字符串不能检出。
2. 72个原/简模型中64个含重复NodeProto.name，GB.node对有显式output节点不加序号。实际Tiny CSA的ORT加载报two nodes with same node name(c_kv_gate_proj/Transpose)；checker passed不足以证明工具可读取。组名重映射后也要全局唯一，并同步mapping而非丢源关联。
3. 仅内存唯一化节点名后实际Tiny CSA运行失败：c_window_view/Slice Ends must be a 1-D array。_window_common把usable/ueff/neff/nwin等标量直接作Slice starts/ends。需要实际生产builder小例，现RMSNorm测试手写另一图/if False未调用helper，不能覆盖产品helper；小RoPE和Scatter测试也未覆盖动态压缩/Indexer。
4. build_attn_indexer的c_block_bias取Indexer compressed长度，而JSON真实inputs是cache_compressed_c_post和indexer_topk；当前漏CSA依赖，最终Slice用n1(=N+1)没有移除哨兵列。HF modeling:690-706明确用CSA compressed_len，最后[..., :compressed_len]。ScatterElements更新值为0-D标量而不是与indices同shape，ONNX schema/runtime需真实核对。作为同一block mask转换缺陷收尾。
5. build_attn_mask用key_idx=Range(0,L)和绝对positions=P+Range(T)比较，P超过窗口后丢key位置偏移。仅节点名内存唯一化后，真实Tiny W4/P10/T1输出全-inf，正确保存窗口本次可读四key(绝对7,8,9,10)应全0；这不是2D padding新范围。只修当前causal+sliding已分析路径。
6. 新identity记录onnx/sim/ORT但未纳入实际派生常量所用numpy和序列化protobuf等锁定依赖；补最小实际依赖锁定/身份。报告源码mapping“全模块”须只指关联存在，不能暗称逐节点语义独立证明。check_n=0仍不数值等价。

仅同一反馈06转换与可用性修复，不扩旧走线/全模型/Excel/M2M3；保原SVG与正确Q/YaRN/外部权重/18模块两阶段及新schema成果。browser_visual=blocked/not_verified、user_reading=pending。新产品须新ID，所有已存候选/证据只读。
