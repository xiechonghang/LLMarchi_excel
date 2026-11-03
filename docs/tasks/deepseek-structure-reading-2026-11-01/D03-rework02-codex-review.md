# D03 rework02：已修两类问题，剩余一个扇出边界缺口

2026-11-03（Asia/Shanghai）。候选 `8136e2132e06c2c84c8e97a6fde9900ac1e613121636fdd6543a95e7091d8531`，正式页 SHA256 `fa0f2dd3d0f534de75520bf754516e27df907e680cabc186c9e855066e213d81`。D03尚未验收，D04/D05未启动。

## 已通过，保留

- 阶段输入改为真实op inputs/outputs推导，未知op/tensor具名报错；所有四组Q/O、三组compressor、两组Indexer的真实跨阶段/外部输入独立核对无遗漏。之前chunk/windows、CSA偏置rank、Indexer RoPE/scorer/topk输入等问题已解决。
- 缓存调用后累计长度以明确阅读投影`kvcachelen+seqlen`出现；缓存表全称符号展开、真实读写来源补齐，未修改旧Excel事实。
- 新候选55图位双视口实际浏览器644/644通过，0失败/错误，146截图；不是沿用8cfa旧结果。阅读页面52/52测试独立复跑通过；558全套为dsh报告结果。
- 23013既有输出文件全部匹配，Excel与上一候选一致，来源清单与当前源码一致。详`D03-rework02-codex-verify/protection.json`。

## 唯一剩余阻塞：P1 同一tensor既供阶段内又供图外消费时，图外出口被删除

`_stage_graph`当前用全部阶段inputs的`consumed`集合决定输出：只要tensor在图内任意阶段消费，就`continue`跳过`s-out`。这样没有检查同一tensor还有图外真实消费者，丢失了扇出。这是本次输入推导已正确之后的输出边界缺口，不要求重做阶段分组、布局或其它已通过功能。

独立检查真实facts与最终candidate/relations确认五张图受影响：

- `q-csa-hash`、`q-csa-topk`：`q_a_normed`在Q图内被q_b_proj使用，同时供图外`i_q_proj`使用。最终Q图只输出q_rope，没有q_residual到Indexer的段出口。
- `compressor-csa-hash`、`compressor-hca-topk`、`compressor-csa-topk`：`cache_compressed_c_post`在图内构建bias，同时供图外`kv_concat`使用。最终压缩器只有状态写回/mask等出口，没有compressed KV交给Attention的段出口。

事实审计`D03-rework02-codex-verify/stage-boundary-audit.json`；交付原始输出边与relations摘录`final-boundary-evidence.json`。主Attention总览已有相应消费者，不代表独立通路图可丢真实出口。

修复规则：内部消费和外部消费分别判断。真实阶段输出如果被图外任意op消费，即使也有图内消费者，也必须连段输出/明确外部消费者边界；图内已有边照保。真实末端和状态写回继续保留。不要简单把全部内部tensor暴露成输出导致假边或宽图。由现有block ops真实消费者推导，保留tensor身份、shape、外部消费者与来源。

补一个共享规则回归：同一张量同时拥有内部与外部消费者，内部边与边界边均存在；实例覆盖上述2个Q图和3个压缩器图的最终candidate/relations。无外部消费者的纯内部tensor仍不泄漏为段输出。既有52测试和644显示门槛不降、不删断言。

## 结论边界

本轮未逐张人眼目检、未完成全部51张原生独立重建；发现此明确语义缺口后，只续做共享输出规则。用户视觉待最终语义正确候选，避免重复审图。保留所有已通过部分、Excel/历史/E01隔离；不关票、不Git、不改自动化。
