# M1 图示收尾02复核：已知缺陷关闭，交展示范围独立验证

2026-10-18，Codex。当前候选 `32af1e923cca4f8c1d535786e4b612d4ccfdfe736c47aeb3671230f77f2d5733`。

本轮仅helper的render_svg及对应图测试变化，保留已核模型事实和Excel。Codex实际25图测试、44受影响页面测试原始rc0；两次实际build-hf均rc0、同固定ID/reused。当前纯构建7产物逐字节同实际候选，manifest及12工具源SHA吻合，JSON/xlsx与29d70相同，输入/旧候选保护未变。

复核实际44张SVG，首/末行完整占用核查最小块间基线距离13px，0标签越出画布；不是仅比较首行位置。前轮缓存真实下游、kv_full本次读视图/压缩缓存兼保存、新输入和缓存activation链保持。各模块按输入/权重反查无假边/漏边/重复，2212本地链接和marker闭合，全页ID无重名。对应用户只要求直观连接图的反馈，未改粒度/Excel、未清冗余、未进入M2/M3。

结论仅为已知修复范围认可，正式新展示独立验证待完成。证据 `hf_m1/codex_diagram_review03/{result.json,facts.json,label_geometry.json,layout_build_facts.json}` 与原始测试/build日志。

真实浏览器字体/CSS/滚动/点击仍未验证。GLM报告称旧label/节点矩形相交127处“先前已认可”并不准确：此前没有浏览器视觉认可。这些近似几何是否导致文字遮挡需要独立复核有限核实并给出具体例子/风险；不据此代签可读，也不自动展开通用布局平台。用户最终阅读确认仍待后续图示交付。

新展示身份冻结于 `hf_m1/diagram_gate/round01/freeze.json`，84文件SHA `ee8d5c0aff7133f04eb4e2d06dad3e0cd0a7a27ec4d45c6e335fb2405564e8ca`，保护42旧候选文件。旧M1工程gate/round01仍只对29d70身份成立，不覆写或移植旧通过。
