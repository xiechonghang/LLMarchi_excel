# M1 纵向结构图/层级编号最终静态展示复核（2026-10-20）

正式裁定：**display_static_engineering=passed**。仅本轮用户反馈03的纵向图与章节层级，保留此前已认可清晰度/粒度/Excel内容；不代表真实浏览器、原生Excel或M1整体用户阅读通过。

## 阅读入口与实际变化

[纵向结构图与章节编号HTML](../../outputs/deepseek_v4_flash_0731_hf/2c9a8b21887bc4e17829ee3108a7212d6eaa39ade01cb8e66fbdb29d4f9b40f6/viewer/decoder.html)。

- 主数据流自上而下，文字正常横向；原计算框源码名+类型、tensor/权重箭头名+shape、分支/残差/缓存条件保留。44图按最终内容四向范围设置viewBox，负原点属于正常画布容纳，初版58同类左边界外标签收尾已关闭。
- Prefill为第1章、Decode第2章；真实包含 Attention `x.2` → CSA `x.2.4` → Indexer `x.2.4.1`。目录、详细图标题、父章节链接与总览同锚点编号对应。每阶段先x.1…x.6再x.7缓存附图，main/csa/indexer依7.1/2/3排列，各直接关联x.2.3/x.2.4/x.2.4.1。
- 同候选tables/decoder_reading.xlsx及JSON/effective_config与c97/原认可版本逐字节相同；不清冗余，不重拆算子，不进入M2/M3。

## 本次核验

原可见Terminal22 DeepSeek正常完成独立复验并停止；Codex已读实际两脚本、8项有限正负例、原始argv/cwd/stdout/stderr/rc，复跑8项及固定候选检查均rc0。103文件freeze SHA55da87749388328dc7273f64c87f0ac42081f5728e5b084ec733ee8c9fd1a4c2、105旧候选保护开始结束与当前完整复算吻合（含存在性），两新检查SHA吻合。冻结产品/历史未改。

合并独立和Codex针对性回放：
- 42模块/cache548真实SVG端点→JSON期望，无缺失、多余、反向或未定位；168计算框→计算框实际路径全部源底→目标顶且向下。2导航总览各40边端点定位，聚合方向不计548。
- 全44图标签越界/互叠0，计算框遮挡0；774实际边/细引线路径控制点均在实际viewBox，132marker唯一闭合；2296页内链接无断链、无重复id。
- 源码长名/类型、628可见边标签、538有href名shape及50条件同块保持。两阶段52目录目标、42总览目标（含6缓存通道）编号逐目标一致，实际全部图标题按数值层级顺序；Decode错Indexer前缀、cache总览编号不一致两个有限内存反例被拒。
- GLM原始两真实build同2c9、第二次reused，Codex48图+44页面rc0与当前纯构建7份字节等候选、13工具源同源已在review02核实；本轮未重复构建或旧整套审计。

## 覆盖口径校正（保留原报告，不追改历史）

独立脚本的op主流条件只拒上行，名为horizontal的fixture实际是反向边例；不据此声称已单独拒绝所有横图。Codex补核当前168条源底→目标顶向下，旧c97横向对照为0/168，明确当前纵向事实。独立显式包含/cache关联/顺序主要核Prefill，nav编号统计36不含6cache；Codex直接补足两阶段完整52目录/42导航/关联及实际次序，未再派补证或改产品。独立端点边界容差6.5px，Codex四向精确锚点小于0.6px匹配548边；路径/细引线画布范围由Codex实际XML补核。

旧文字补证q_a_norm误用tensor键的空绑定不采信；真实q_a_normed=q_residual及源813的既有直接核实继续保留，名字helper/JSON字节未改，不扩大75/164全表语义独立覆盖。90无href为80导航+10下游，不宣称全90独立JSON shape绑定。

## 状态与限制

browser_visual=blocked/not_verified；native_excel=not_verified；user_reading=pending；M1整体最终阅读待用户明确确认。静态字宽、有限CSS声明与几何不等真实字体/CSS/硬件点击/滚动；未绕过file安全限制。用户之前认可的是c97清晰度，新纵向/编号效果待阅读。GLM/DeepSeek均完成，Qwen维持空闲；无新反馈不再派旧任务或重复门禁。自动检查实际PAUSED，保持暂停。

证据：`.work/coordination/hf_m1/vertical_hierarchy_gate/round01/` 和 `codex_vertical_hierarchy_gate_review01/`；前序初版/收尾证据只读保留。
