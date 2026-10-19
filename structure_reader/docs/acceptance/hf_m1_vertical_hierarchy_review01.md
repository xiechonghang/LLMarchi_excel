# M1 纵向结构图/章节编号复核01（2026-10-20）

候选 c95fda9c9c76728ab6e4005327e0b758bed8ccfbf2535a5eb407a166420a805c，裁定 changes_required，仅以下两项收尾。

已核保留：产品只 render/diagram.py、decoder_page.py 和图测试；48图+44页面复跑 rc0，三数据与c97同字节，6manifest交付哈希/13工具源同源、98旧候选文件未变。复用独立SVG解析与JSON方向期望，对真实四向几何端点/marker留白定位，42模块/cache548方向0缺/多/未定位，op→op无上行。可见长名/类型/边名shape条件与实际CSS补证函数均0issue，标签块互叠0。Attention→CSA→Indexer为x.2→x.2.4→x.2.4.1，符合源码包含。

1. 实际SVG标签画布边界遗漏。按当前渲染器相同的有限文字度量，全44图有58标签占用超viewBox左界（同类问题，不拆任务）。具名：mod-prefill-mhc_attn 的 self.fn 权重标签中心 x=8，估算左界-70.25；attn_q 的 self.q_a_proj.weight 同样左界-70.25；cache-prefill-main 的产出本次读视图左界-15.25，Decode同类。全部文字虽然在XML中完整，不能据此宣称画布内完整可见。需要容纳最终实际标签/节点/路径的边距和画布，保留真实端点/细引线关联，不能缩名/shape或字体来掩盖。静态度量不是浏览器视觉代签。
2. 新章节编号未约束实际呈现顺序：两阶段先x.7缓存附图，内部x.7.2→x.7.3→x.7.1，然后x.1…x.6，与目录章节阅读顺序不一致。仅调整各阶段图块和缓存子图实际顺序，使章节按数值层级顺序显示；缓存标题应各自明确关联主KV x.2.3、CSA x.2.4、Indexer x.2.4.1，保原锚点和内容。

证据 .work/coordination/hf_m1/codex_vertical_hierarchy_review01/{spot_review.py,spot_result.json,test_commands.json,*stderr}。原GLM日志与报告保留只读。7当前产物未再纯build逐字节复算；独立最终展示未派。browser_visual=blocked/not_verified，user_reading=pending（c97清晰度已认可，当前新布局未读）；M1整体仍未最终确认。
