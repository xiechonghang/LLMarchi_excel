# M1 图示修复01复核：缓存映射关闭，剩多行边标签间距

2026-10-18，Codex。对应用户已认可算子粒度/Excel、只要求直观连接和缓存图的阅读反馈。本轮没有增加模型分析范围。

候选：`8ab8e14cbf695e93dc19cfce42844f751977c6a9fe00c34099505a9473586c2e`。结论：展示 `changes_required`，尚不派新展示独立验收。

已关闭：两阶段三个缓存族下游正确（主KV→kv_concat；CSA→kv_concat/c_block_bias；Indexer→i_scorer/i_topk）；kv_full为本次读视图、压缩条目兼本次读/持久保存，新输入及缓存算子activation路径已进入图。模块输入/权重引用独立反查无漏边、无假边、无重复边；不同tensor保留。全页marker/DOM ID无重名、2212本地链接与marker引用闭合。

Codex实际图23项、受影响HF页面44项原始rc0。修前五文件本次仅helper与图测试改变；当前纯构建7份字节与实际候选相同、12工具源/manifest匹配。reading_document.json与xlsx仍逐字节等于29d70；输入、旧候选保护复算未变。开发者两次实际构建日志保留，本轮因已发现布局缺陷不重复物化两次候选、重跑Excel/模型门禁。

剩一个布局缺陷：`render_svg`先把边标签换为最多两行，行间11px，却仍按标签首行固定13px排距。两行文字之间只剩2px基线距离，10px文字发生相交，不能以“首行坐标不同”视为不重叠。例如Prefill缓存CSA图，x=516：chunk_kv两行基线33.25/44.25，chunk_gate两行46.25/57.25；前者末行与后者首行仅2px。两阶段36对近距标签是同一个排版缺陷，不拆成36任务。

证据：`structure_reader/.work/coordination/hf_m1/codex_diagram_review02/result.json`、`facts.json`、`label_geometry.json`及两组原始命令日志。坐标核验属于静态SVG几何，真实浏览器字体/CSS/事件仍未核验，不能代签视觉或用户阅读通过。

仅把此多行标签间距问题回原GLM修复；保留缓存/节点/marker/去重成果、JSON与Excel。修后先核本缺陷与数据身份，再交DeepSeek按展示改动独立复核；不启动M2/M3，不清冗余，不扩图平台。
