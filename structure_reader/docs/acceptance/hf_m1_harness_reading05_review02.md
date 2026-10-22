# Harness反馈05有限收尾复核02：三处主体关闭，两个同类命名/一处接口线收尾

92ad3e62f92222904720cb5c40283743a5967c9a7cd7fcb706b39870347fdec1正常完成空闲。同App/1017开发，Codex不开发产品。

原三处主要完成：core/mask具名穿框关闭；复合CSA/Indexer各有实际统一Cache存储框与键，12字段被正确包围；暗store底/字/线marker成对，浅4.82/暗10.95，透明frame暗描边也修正。保当前成果。独立实际几何解析432模块/cache依赖仍0缺/多/未定位/重复，不变JSON为期望，实际36SVG、512可见标签与182源码/150回链保持。实际两build完整argv.json含Python -c/main入口、cwd/env/stdout/stderr/rc，同92ad第二次reused。189旧保护逐项未变。

还需有限修正（不是重分析模型或新增全页路由）：

1 实际Cache属性名字存在同一个展示生成器错误：_CACHE_FIELD_SPECS main.cumlen显示self.cumlen，_field_node title又生成past_key_values.layers[layer_idx].cumlen。HF modeling_deepseek_v4.py:196真实为self.cumulative_length，无cumlen属性。main.full title生成past_key_values.layers[layer_idx].full，但HF:197-201的full是update局部返回变量，不是存储属性。visible full行已写“本次读视图，不持久”，此正确区别应保，并让title/owner诚实。内部展示键main.cumlen/main.full、原JSONcache_cumlen/kv_full与端点不改，只给真实源码名字/本次返回语义。
2 总览仍同一接口stub曲线在小范围穿过相邻chip（高密度采样可见，23点漏掉）：Prefill/Decode各一次，path M210,797 C210,831 461.31,818.5 461.31,784.5，穿过“同名接口（来源端）· attn_out_streams”矩形translate(223.45,760)、216×44。约t=.8处(435,800)在此chip内部（含圆角内侧），线被遮断。仅修这条去向端stub绕行拐弯，不扩大模块剩余穿线任务；原三条core/mask走线不回退。证据codex_harness_reading05_review01/crossings92ad.json。

注意本条报告中36图已由实际SVG核实，不采报告误写旧数；复核02只这两组，非静态最终通过，browser仍blocked/not_verified，用户阅读pending，M1未最终确认。
