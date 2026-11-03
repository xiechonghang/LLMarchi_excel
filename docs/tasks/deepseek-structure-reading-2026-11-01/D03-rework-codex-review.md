# D03 返修技术复核：显示通过，阶段事实仍需修复

日期：2026-11-03（Asia/Shanghai）。候选 `8cfa840daf2cefb28ad298670faff9ae45765da15ddebf3af8e280ae57d03613`；正式页 SHA256 `8966274700688fde8ffeec4aaa3977aac0adac94f3c88c4ecb115def370fb384`。D03/#19 尚未验收，D04/D05 不启动。

## 本轮已通过，后续必须保留

- 两视口1440×900、1920×1080动态遍历全部55图位，644/644项通过，0失败、0运行错误，146张截图。探针与D01原件仅33→55图位断言不同，其余断言/6px字号门槛未改。
- HCA压缩器原生 browser-check 独立复跑通过，Chrome available，diagnostics为空。此前过宽/过小字问题已解决；不是继续要求重做已通过布局。
- Attention最终candidate中compressor/Indexer→block_bias→mask_extend恢复；HCA真实类型、position_ids输入及None分支说明已改；Q/O有独立入口。完整trace expected排除已撤销。
- 独立阅读8文件差异，阅读页面48/48测试通过。554全套为dsh报告，不声称Codex复跑全套。
- 22641+160既有输出逐字节保持，Excel与上一候选逐字节相同；来源清单与当前源码匹配。见`D03-rework-codex-verify/protection.json`。

## 剩余阻塞

### P1 阶段图手工输入/输出清单与真实事实不一致，丢边并伪造标量

`render/archify_layers.py::_stage_graph`虽然为阶段收集op_ids，却不用其实际inputs/outputs推导边界；调用方手工维护另一套inputs/outputs。缺失op静默跳过，缺失tensor通过`_shape_of(None) or '标量'`伪装为标量。这不是合法标量状态，应报错。

已在同一实际结构事实与最终candidate确认：

- CSA compressor声明`c_mask_extend`，该op在CSA事实中不存在（真实mask扩展需用该层已有事实id）。声明输出`block_bias`亦不存在，真实为`c_block_bias`；交付`s-bias→s-mask`显示“标量”，把 `[batch_size,1,seqlen,num_compressed_entries]` 偏置错写成scalar。
- 两组Indexer引用不存在的`i_weights`，最终`n-s-in→n-i-score`显示“标量”。真实`i_scorer`输入是`i_q, cache_compressed_i_post, attn_normed, i_w_weight`；weights计算在scorer内部，不能伪造外部scalar。
- 三组compressor的窗口阶段包含`c_window_view`，其真实输出是`c_windows_kv/c_windows_gate`；图却跨阶段传`c_chunk_kv/c_chunk_gate`到pool，跳过窗口重排后轴语义，和`c_pool`真实消费者不符。
- 两组CSA窗口阶段遗漏`cache_overlap_c_kv/cache_overlap_c_gate`及`cache_entry_count_c`；两组Indexer窗口遗漏独立overlap与entry_count，pool也用chunk替代windows。
- Indexer `i_q_proj`实际需要`cos_q/sin_q`，阶段图丢失这两输入；`i_scorer`丢`attn_normed`、`i_topk`丢`cache_compressed_i_post`。这些是实际消费者，不能以阶段名/详情/缓存表代替连线。

独立逐阶段核对见`D03-rework-codex-verify/stage-fact-audit.json`。Q/O当前没有检测到同类missing id/input，保留。修复应复用既有事实推导阶段外部输入/输出与真实消费者，按图抽象合理聚合；不能继续手写第二套边界。对不存在id报具名错误，只有事实中真实shape=[]才可显示标量。阶段内部仍保留具名算子、条件、来源与可达细节；不能为通过布局删依赖。

验证必须检查最终candidate/relations逐项对应真实跨阶段/边界张量，包括上述失败例，不能只查op_ids非空或图中存在任意c_/i_节点。

### P2 缓存累计长度调用后状态与可见全称符号仍未补齐

新增cache.update新KV/full与分图保留，但真实事实仅有`cache_cumlen`调用前状态，没有对应调用后状态；最终cache-main写回边仍只有keys_post。页面只是“cumlen同步推进”一句，不能支持报告“调用前/后累计长度已展示”。请用可追溯读取投影表达调用后`kvcachelen+seqlen`及更新依赖，保持旧Excel不变。

新增`_cache_state_section`直接join底层shape，表格仍暴露`B/Dh/W/Tbuf`等旧别名，而表标题称“完整符号shape”。当前缓存逐项形状主要靠此表提供，须复用现有全称符号展开；来源列直接读取不存在的tensor.source_refs，实际为“—”，应从真实读写算子/已有来源映射追溯，不能造来源。其余逐项表格及独立状态分图方向保留。

## 范围与结论边界

本轮不逐张读长截图，按用户要求将人眼视觉留待用户。已通过机器显示门禁不等于语义通过。因以上确切语义阻塞，本轮未重建全部51张原生图，不宣称51/51原生浏览器复核通过；已有dsh静态51项为其结果。

只续做本D03返修的剩余技术问题，保留644项显示通过成果、block_bias/HCA/Q/O已正确部分、Excel与历史保护。修复后再交用户精简审图入口，不让用户重复审查已知语义错误版本。
