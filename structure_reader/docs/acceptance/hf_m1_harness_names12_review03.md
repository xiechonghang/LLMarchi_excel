# 反馈12最终复核：HTML完整参数与维度名称

2026-10-24。原Harness App / DeepSeek-V4.1-Flash 1017 High第19轮正常完成同任务语义收尾并空闲。当前候选`7b8711f2768cc6dbaa434edae57ff602b25453929039de3ab371b99f145c286d`。

[新版HTML](../../outputs/deepseek_v4_flash_0731_hf/7b8711f2768cc6dbaa434edae57ff602b25453929039de3ab371b99f145c286d/viewer/decoder.html)。仅`display_static_engineering=passed`，实际阅读待用户确认。

## 本轮可见成果

讲解、图中shape、算子流、参数区及张量索引改用完整名称，例如batch_size、sequence_length、hidden_size、q_lora_rank；参数中文含义与当前值保持。派生缓存长度区分CSA/Indexer及前后状态。参数区去掉冗余缩写列，内部id/锚点保持。真实源码与数学转置.T/^T、键K不冒充维度参数，源码快照原文保持；缓存讲解的前态增量明确为sequence_length。

长名标签按实际完整行数换行，初版48处缺尾已关闭。不是仅在title保全文，512个实际可见边标签拼接全文与title相同。原38段讲解、层次、纵向与真实依赖、类型、条件、源码/图联查及Netron入口保持。

## Codex核验

- 13项新增/受影响阅读、说明与资产检查通过，无skip。读取实际helper和接入/有限测试，实际页面无compressed_kv.sequence_length、comb.sequence_length、k^sequence_length等误改；合法维度仍全名。
- 36图、528条真实SVG边：依次按href/边类型/marker身份核对，进出端点定位实际矩形/圆点/状态字段/控制容器边界，配对与14de基线一致，0悬空。仅对本轮几何保持负责，不声称重新独立证明模型全部方向语义。
- 对512个原标签的shape及条件按独立列出的映射展开，全文多重集与新标签一致；布局避让可改变标签DOM顺序，不以顺序变化误判数据变化。结合逐SVG边身份/端点与可见全文检查，不声称新增了label到每条path的显式metadata绑定。
- 所有内部id/href与基线一致（本地Netron候选身份例外），无重复id或断链；36本地链接resolve和108下载检查通过，未实际触发协议/GUI。
- reading_document/effective_config/xlsx与144个ONNX目录文件逐字节保持；源码pre全文保持。启动器二进制SHA仍859728cbcdf9102837e2e235e106f4336776e68361201fea40390fcfb6b07054。
- 151交付、23实际工具源（含新增display_names.py）、7输入SHA当前匹配，canonical identity重算命中当前ID；两次实际新build原始argv/cwd/rc正确，同ID、第二次reused=true。6087旧候选保护复算无漂移，源/测试变化仅授权展示范围；需求/计划/progress由Codex更新。原始初版和两次收尾证据保留，不擦历史。

证据：`.work/coordination/hf_m1/codex_harness_names12_review03/{verify_final.py,final_result.json,shape_comparison.json,targeted_tests.stderr}`；Harness原始日志`logs/dpsk_harness_names12_fix02_*`，实际图/文字样板`names12/*_fix02*`。

`browser_visual=blocked/not_verified`、`native_excel=not_verified`、`user_reading=pending`。没有绕过既有file/协议限制，没有代签长标签的真实浏览器排版/窄屏观感。自动PAUSED、M1待整体确认、M2/M3未启。本轮未提交或推送GitHub，用户删除保持。
