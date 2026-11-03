# M2四组合数据阶段复核（非最终交付）

2026-10-24。原Harness App1017 High第22-24轮实际新增HCA/SWA/Hash数据与有限分支检查；第20/21轮仅交来源分析、未实施的断点保留，不当作完成。

Codex读真实源码及新增extract/m2_hca.py、m2_swa_hash.py、原HF提取分派/Hash公共分支和contracts。初版HCA独立values误述、条目RoPE错连query位置、KV归一权重/复合类型、mask来源/调用顺序、batch轴等已要求修正。现实际数据按0/2/3/4两阶段回放：SWA+Hash每阶段58算子，CSA+Hash76，HCA+TopK68，CSA+TopK75；数量仅范围记录。

独立关键期望来自原config前43项及pinned HF源码，不从新输出抄golden。检查实际引用/生产消费顺序、真实source文件/locator、HCA128及head_dim/无Indexer/条目B轴和entry_count位置依赖、主KV别名与先update后压缩、SWA无压缩/独立持久字段、Hash input_ids和tid2eid查表→scores权重链/无选择bias与TopK。默认层4文档与7b候选逐对象相同，7455旧候选复算无漂移。

证据`.work/coordination/hf_m1/codex_harness_m2_dispatch01/{source_expectations.json,check_data.py,data_review_result.json}`及Harness各data/fix原始logs和JSON小样。复核脚本初版两条探测误将“不是别名”的否定句及包含tid2eid的indices名称当缺陷，已只在本次新脚本修正判定，不改产品去迎合误报。

尚未验收M2：容器、HTML/两表Excel及ONNX/Netron待接入；HCA M含义/SWA不适用参数/CSAHash input_ids旧注记同类阅读遗漏在view01一并收尾。自动PAUSED，M3未启，不Git提交推送。
