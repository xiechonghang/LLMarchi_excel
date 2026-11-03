# D02 Codex 复核：退回返修

时间：2026-11-02T23:43:42.970732+08:00。候选 `0eb1ca6461639b112c52c4bd7df0494ce2b971425307b6fdcc8d1d9e6ecfe1b7`；正式页SHA256 `3a86694d5771737058a3a5b98ef6e959512e8c76c0cd21e43b761b573045d70c`。未验收，不关闭#18，D03–D05未派发。

## 已确认

核对 modeling_deepseek_v4.py:952–984、1169–1185，mHC.forward返回post/comb/collapsed，Decoder调用Norm→分支，再post展开与combᵀ混合原多流。新投影将残差算子从mHC内部拆出，方向正确。仍需修复以下可见交付缺陷，测试通过不能替代它们。

39项阅读页面测试独立复跑通过；dsh全套545项记录未冒充Codex重跑。独立哈希确认22242历史文件、248此前候选、f1ea的124和D01已接受2fa的124文件均未改变；Excel逐字节一致。证据D02-codex-verify/protection.json。

## 阻塞项

1. **P1 符号维度展示回归**。archify_layers.py中新增semantic_first及跨秩label_text选择，以edge.label替换edge.shape。四组Decoder的collapsed/post/comb/branch_output/原始多流/combᵀ等边只显示名字；mHC返回边也丢失主图形状。截图1440x900-fig-decoder-swa-hash-full.png及mhc图可见。元数据仍有shape不等于主图传递了符号维度。要求同时展示张量身份及完整全称shape，不能通过只放弹窗/源码/完整文字面板代替；调整布局容纳，而非去掉信息。
2. **P2 四张mHC图标签仍重叠**。原只读探针424项中416通过、8失败、0运行错误，102截图；四组×两视口各失败一次。f-7-o-in-o-attn-hc-collapse与f-15-o-in-o-ffn-hc-collapse的batch_size,seqlen,hc_mult,hidden_size标签，实际bbox重叠约3.386px×10px。离线估算0不能代替浏览器实测；不降低探针阈值，修复共用布局并全图回归。证据D02-codex-verify/page/result.json。
3. **P2 CSA两组原生图可读性未过**。独立重建29/29与交付字节一致，但finalize/browser-check仅27/29通过；decoder-csa-hash、decoder-csa-topk失败。Chrome实际可用，诊断是viewer/viewport-overflow和viewer/projected-text-readability（1440×900边文字约5.9426px<6），不是环境不可用。native_evidence.py摘要把非零浏览器结果计为environment_failures，不能照抄分类。详见native/<slug>/receipts/*.browser-check.json。注意区分复合阅读长文档预期滚动与单张原生图实际字号失败，不以改工具/隐藏标签/缩字绕过。
4. **P2 mHC就近参数与Sinkhorn循环未呈现到位**。mHC候选参数卡前12项仍是通用block前缀，包含num_attention_heads/q_lora_rank/o_groups/sliding_window等，却无hc_mult/hc_sinkhorn_iters等mHC参数；主图只写comb+Sinkhorn，未说明源码中一次初始列归一及(hc_sinkhorn_iters-1)次行/列归一循环。要求mHC局部参数与配置来源就近可见，循环边界/次数使用全称符号表达，避免向上回流；可保留具名算法粒度，不要求机械展开20份节点。源码面板存在for语句不等于图上已经展示了循环。AC08/AC13及D02权重细图验收仍未满足。

## 证据与目检边界

独立浏览器探针截图102张，仅实际查看两张代表完整图（Decoder swa-hash、mHC swa-hash）；发现阻塞后未宣称29张全部目检通过。原生29图重新构建/校验记录在D02-codex-verify/native-evidence.json；27通过2失败。复合structure.html通用browser-check仍将长文档滚动判失败，保留原始记录；不与上述两张原生图字号失败混为一谈。GUI核对当前会话最终回复、空输入框及无运行按钮（/private/tmp/D02-review-dsh-current.png）。

下一动作：仅D02返修，保留本轮正确的Decoder/mHC职责与全部已通过保护，不提前推进D03。
