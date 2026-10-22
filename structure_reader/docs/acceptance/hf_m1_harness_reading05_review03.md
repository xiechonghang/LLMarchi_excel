# 反馈05最终静态复核03：Harness App / 1017

日期：2026-10-20（Asia/Shanghai）。开发由用户指定的安装版DeepSeek Harness App、DeepSeek-V4.1-Flash 1017 High完成；会话“DeepSeek Harness 结构图交接开发”（session-9a4c37d2-1504-47b6-84a5-6bc0fc8e8dcf）。Codex仅派单/需求映射/实际复核，未开发产品；原OpenCode三个角色保持空闲未恢复。

裁定仅 **display_static_engineering=passed**，对应当前候选 `032ea081ea9d5ac048972808edb6b1513c5a0b5ef088ad733cf4543044a12a6e`。这不是M1整体用户确认，browser_visual=blocked/not_verified、native_excel=not_verified、user_reading=pending；M2/M3未启。旧e007/c97及各候选/门禁结论不移植。

## 用户可见交付

- HTML：`outputs/deepseek_v4_flash_0731_hf/032ea081ea9d5ac048972808edb6b1513c5a0b5ef088ad733cf4543044a12a6e/viewer/decoder.html`。
- 总览：主流纵向、Q/共享KV/CSA真实并列展开，长依赖分道与成对同名来源/去向接口；两阶段共88条导航路径的实际曲线对计算/导航/接口矩形高密度采样穿框0。总览聚合方向不计入下方432模块依赖全量统计；采样不是连续曲线形式化证明或浏览器观察。
- 包含：两总览实际Attention边界包含CSA，CSA包含Indexer；两详细复合图也为CSA包含Indexer。均从实际SVG矩形读取包围关系，不以文字注记代替。
- 缓存：同一past_key_values.layers[layer_idx]:DeepseekV4CSACache对象的compressor/indexer键各6字段，存储框在对应父子容器内，字段实际端点表达读前/写后/本次视图与闭窗条件。12字段按对应键几何归属，表头空间≥42px；main累计名为cumulative_length，full明确为update局部返回视图，不冒充持久字段。独立缓存图专区已取消。
- 配置：复合图内三选项及控制/实例化线，当前CSA唯一生效；HCA无Indexer、SWA=None无压缩器，未展开变体没有假切换链接。
- 联查：每阶段参数区先于图/算子卡片；182源码面板2888行逐行与冻结HF吻合且所有locator指定行覆盖；150算子卡片→本阶段具体框且该框→对应卡片实际href逐项匹配，全部id/内链唯一。真实滚动/高亮/按钮点击未在浏览器核验。

## 实际复核与证据边界

1. 当前7交付字节与一次内存纯构建逐字节相同，13实际工具源与7实际输入SHA吻合；原始两真实build完整argv数组（Python -c/main入口）、cwd/env/stdout/stderr/returncode均核，rc0同ID、第二次reused。未重复CLI落盘构建旧候选。
2. JSON/effective_config/xlsx与e007/原认可版本逐字节相同；196旧候选保护逐项存在且SHA未变。接口与导出器未变，用户根CONTEXT.md及2026_10_17_workspace_structure_review.md删除保持，未Git提交或恢复删除。
3. Codex独立复用旧graph_expect，仅从不变JSON取得方向期望；以实际SVG路径端点→计算框边界/圆点/字段时态端口定位：34模块图432依赖missing/extra/unresolved/duplicate全0。没有导入产品renderer/命名helper来取得方向正确答案；身份纯构建与方向核验分开。
4. 有限内存反例：反向一条实际tensor路径、把实际缓存写端点偏移162px均被独立检查拒绝，未改任何候选。不是全边变异平台，也不宣称覆盖全部坏图。
5. 512实际可见边标签与各自title全文一致，条件分隔符｜由换行替代，未误当截断；75种计算框名称/类型可见全文与已认可c97签名一致（两阶段150框）。这保留已认可来源命名，不扩大成75/164全语义逐个独立证明。
6. 新store框的实际有限CSS声明按各dark媒体块作用域核：字/底浅4.82、暗10.95，线/marker对相应底浅6.22/暗7.68；固定浅底计算框原深字规则保持。仅静态声明配对，不是浏览器字体/CSS实测。早期css_pairs.json是错误媒体分段探针，已标accepted=false，不采信；正式用css_pairs_verified.json及final_replay.py递归媒体作用域解析。
7. 当前两个受影响图页测试复跑98项rc0；144个marker定义唯一、528处url引用闭合。测试数量不是需求进度。

独立/复核证据：`.work/coordination/hf_m1/codex_harness_reading05_review03/`（identity_and_visible、links_sources、directions、crossings、containers_and_choices、css_pairs_verified、final_geometry_and_css、build_logs）。方向/穿框参数化脚本在review01，最终有限回放final_replay.py在review03；受影响测试原始输出在codex_harness_reading05_checks/。
开发原始记录：`dpsk_harness_reading05_result01.md`、`dpsk_harness_reading05_fix_result01.md`、`dpsk_harness_reading05_fix_result02.md`与logs相应前缀。开发者自检从产品构建图取得部分事实，不冒称独立方向门禁；Codex实际端点→JSON补核后方作本裁定。最初headless SVG截图不作为browser证明，后续两收尾均未调用浏览器/服务/CDP。

## 留给实际阅读的限制

模块详图仍有18条路径的静态采样穿过其它框（绘制在框后），复合图约2655×2375需要横向滚动。此项按原反馈重点只修总览，不扩成全页新走线任务，留用户实际阅读反馈；不以“框内文字没遮挡”宣称整页所有走线已经完美。真实浏览器字体/点击/滚动、原生Excel、用户整体阅读均未代签。

当前授权HTML反馈05的实现和静态复核完成，Harness正常停止等待；只新用户反馈/实际新错误才继续同App/1017，不重复派旧单/门禁。自动检查仍PAUSED，不因开启Harness自行恢复定时任务。
