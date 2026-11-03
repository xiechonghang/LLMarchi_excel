# 反馈12初版复核：完整名称尚需有限收尾

2026-10-24，候选ac8058e484ffe0053a19e5f6c7be3788da7d577cf519fc2fa6abf9e6bce8e54d。主要参数与shape已展开，原讲解/数据保持，方向保留待最终具名复核；本轮changes_required。

1. 实际可见SVG48个边标签被截断。`_wrap_edge_label`虽写`[:6]`，内部`_wrap(text,34)`仍默认max_lines=3。例如self.keys前态只显示`[batch_size,1,min(past_sequence_length,`，后续sliding_window/head_dim仅title存在。必须完整可见，并按真实行数安排必要标签空间。
2. 完整名称覆盖遗漏：参数区派生id如Nwin/Tbuf/Tbufp/Tcc/Tci仍直接显示，表达式列未展开；条件标签/图例仍“仅N_win>0”等，scope/cache等阅读说明仍B/T/P。真实源码原文允许保留，但阅读维度/条件/公式需要完整名。参数id/shape token/symbol有不同拼法（Nwin_i/N_win_i、Tbufp/Tbuf'），需要按实际清单正确映射前后状态，不能仅PARAM_NAMES中27项就声称全部参数。

证据：`.work/coordination/hf_m1/codex_harness_names12_review01/initial_findings.json`，从实际SVG tspan拼接与title全文对照，只读候选。保留现有正确全名映射与数据，有限修同一展示需求，不扩模型/布局任务。
