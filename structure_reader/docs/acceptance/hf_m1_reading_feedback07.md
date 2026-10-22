# 用户阅读反馈07：以 Netron 链接代替并排 ONNX 图（2026-10-21）

> 工程交付已完成：候选4a8fc8f1…，36模块实际Netron入口，原详图/ONNX字节保持；逐链接对应/7资产通过，1个真实Netron自动载入样例核实。本地HTML真实点击与M1整体用户阅读待确认。详[复核及入口](hf_m1_harness_netron07_review01.md)。

用户明确：“onnx的图不用放在旁边阅读，而是给一个链接，点进去自动使用netron打开这个onnx结果”。覆盖反馈06的并排阅读方式，不撤销真实 ONNX/onnxsim 产物。

基线788ee019821817dff2504938f5b51e02b59a2e714a957eff67dc4c3df9ccb975。原36详图/源码/参数/双向定位和三数据保持；72原/简ONNX及来源映射保留。本轮只把并排ONNX图改为对应模块的Netron打开链接，优先简化结果，明确Prefill/Decode与模块，CSA和Indexer分别可打开，保留原始结果下载。不能以Netron首页或下载链接冒称自动载入。

由原DeepSeek Harness App / DeepSeek-V4.1-Flash 1017 High会话实施，Codex仅派单/验收。官方browser.js支持identifier查询与hash模型URL、data URL请求；有限核网页版可从链接内携带的既有无权重ONNX数据自动载入，无需本地服务、读取file URL或系统协议。链接明示“Netron 网页（在线）”，不许声称离线/绝不传输；不将文件发布到公共存储，不引入本地服务、系统安装/协议或浏览器限制绕过。具体方案需核官方实际版本兼容，自动载入未实际核验则如实记录。

保持FP32无真实权重阅读特化、专家模板和mask子集的已有说明与数值限制；不重跑字节相同图的数值全套，不执行HF/加载权重，不动Excel。新内容身份和仅受影响检查，旧候选/证据只读。工程、真实Netron交互、用户阅读分开，不代表M1整体通过；自动PAUSED，M2/M3未启。
