# 用户阅读反馈12：参数与维度用完整名称

2026-10-24。用户认为B/H尚可猜测，但Rq等缩写使阅读困难，要求参数不要使用缩写。以14de8a08…为基线，在HTML的讲解、图示shape、算子流、参数及张量索引使用完整名称，例如batch_size、hidden_size、q_lora_rank，并保现有中文含义和数值。真实源码表达式与快照保原文，内部id/锚点不改；模型/Excel/ONNX数据不改。

继续原Harness App / DeepSeek-V4.1-Flash 1017 High实现，Codex派单验收。长名要完整可见、必要换行，保方向、层次与联查。自动PAUSED，M1待整体阅读确认，M2/M3未启，不自动提交或推送。

工程交付：7b8711f2768cc6dbaa434edae57ff602b25453929039de3ab371b99f145c286d已完成完整名称及长标签、派生缓存与转置/键语义收尾。13项受影响检查通过，模型/Excel/ONNX保持；仅静态工程通过，实际阅读待确认。详[最终复核及入口](hf_m1_harness_names12_review03.md)。
