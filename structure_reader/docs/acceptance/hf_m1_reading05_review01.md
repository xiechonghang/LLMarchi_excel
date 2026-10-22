# 反馈05初版复核01：changes_required

候选02b084b298a05879a275f684cba489efc9f46198190c2e05cf1583c3cbdfbb67。仅审新7点实际diff/HTML，不重复旧整套。初版正常完成，原GLM空闲。

- 参数已前置；182源码面板有实际快照而非不可用提示，保留这两项方向，仍需定位行/版本与入口行为针对核验。
- 总览wide参数已定义但decoder_page调用render_svg(build_overview_graph(phase))未传wide，故报告“横向展开”没有接入实际产品；不是加参数存在就完成用户第1点。
- CSA/Indexer仍同级nav节点及同级h4，只有sub_txt包含说明，未实现原任务要求真实嵌套边界/父子子图。
- 缓存增加每组一个display carrier，泛化连ops[0]，并未按原JSON前后状态字段读写接线。旧缓存独立专区仍两处，与用户第4点及任务明确“不再独立专区”冲突。虚构泛化载体边不能替代真实字段/时态读写。
- 150回图onclick实际目标figop-*，但HTML无一个figop目标id（0/150），点击必无效；报告“图内新增锚点”不符合实际。
- 配置表如实SWA=None/HCA无Indexer方向保留，但不在Attention/压缩器图体现选择分支。表内x.2.4/x.2.4.1是不存在的旧章节引用，当前任务明确不恢复旧编号。
- 没有新增功能测试；仅旧43/44通过不说明7新功能完成。既有测试放过所有空title边太宽，应限定确实展示控制/包含边，并保真实缓存tensor边的方向/引用保障。

133旧候选当前逐项SHA不变；证据codex_reading05_review01/result.json。新02b的7文件纳入本轮protected_candidates.json，共140文件，只读。before_fix01.json保当前产品/测试SHA。暂不派DeepSeek，原GLM有限补足原7要求，不是新增任务；模型/Excel/旧证据不动。浏览器与用户阅读未核验。
