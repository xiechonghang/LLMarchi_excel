# 用户阅读反馈09：Decoder 总览展开留白（2026-10-21）

> 反馈09工程交付（2026-10-21）：原Harness App1017完成2877a154…，仅两总览展开，Q/KV/CSA横距181→277，关键纵距77→129/58→110，画布2436×2167；不缩字/不隐藏依赖。Codex核每阶段30节点/44语义边/3容器与全文保持、34详图字节同，7受影响检查rc0；72ONNX/144目录文件/三数据/启动器保持，151交付/21源/7输入/两真实build同新ID第二次reused与2179旧保护已核。本轮总览间距静态工程passed，浏览器/用户阅读未代签；原本地协议点击阻断保持未绕过。自动PAUSED、M1待整体确认/M2M3未启。详[复核及入口](hf_m1_harness_spacing09_review01.md)。

用户提供实际总览截图，明确“这里明显太拥挤了，我希望图能展开一点”。图中Attention的Q/KV/CSA并列、mask/cache/Indexer支路、容器标题附近的长连线和标签拥挤。截图原件副本在codex_harness_spacing09_dispatch01/user_crowded_overview.png。以当前aa581b11本地Netron版为基线。

本轮仅展开Prefill/Decode两张Decoder总览：实际增加框之间横纵间距、容器标题留白，给分支/读写线与标签分配空间，允许画布变宽变高与正常滚动。不能只整体缩放或缩小字体、隐藏标签/依赖来假装宽松。保纵向主流、Q/KV/CSA真实并行、Attention包含CSA包含Indexer、cache读写和配置选择、源码与回链及原文字/配色。

仍由原DeepSeek Harness App / DeepSeek-V4.1-Flash 1017 High实施，Codex只派单/验收。只总览几何/必要局部渲染与受影响测试，不扩算法/通用图引擎/章节任务；其余34模块详图字节保持，72原简ONNX/三数据/Excel/启动器源字节保持。新候选本地Netron链接必须正确对应新内容身份。旧候选/冻结/证据只读；反馈08本地真实点击仍blocked/user_action_required，不能借本轮绕过已拒绝协议/浏览器通道。自动PAUSED，M1用户整体确认待，M2/M3不启。
