继续 D01（#17）剩余显示返修，保持在本会话执行，不启动 D02–D05。

先读 AGENTS.md、docs/tasks/deepseek-structure-reading-2026-11-01/D01-rework-codex-review.md、#17及父规格#16。你本轮候选1bc0ea5c的四项修复已经Codex确认：页签按组隔离、六个端点权重依赖正确、来源映射更正、29图省略号清零；保留这些成果。

新的实际浏览器证据：Codex通过正常授权在文件系统沙箱外运行Archify原生命令，Chrome自身sandbox始终开启，已能完成检查。全部29图测量28通过、network失败。证据为同目录D01-rework-codex-verify/browser-gallery/summary.json。network在1440×900下连线batch_size,seqlen,hidden_size只有5.602497px（低于6px），边id f-0-n-net-embed-n-net-mask-build；图SHA705df05864f25a5327bb03be57b8b95de487ded4bea0003f306d849b04ebe9e4。network-capture/visual/有四张原生截图，Codex已查看两张浅色截图，看到下半部主链偏右且左下大块留白、文字很细小。

本轮目标：
1. 在生成器中修复整网布局与连线文字的实际可读性，保留真实节点、连接、数据来源和全称参数，不删边、不用省略号、不裁剪或缩小字体凑门禁。允许合规可读的纵向滚动，不必强行压入900px；不要只改声明掩盖失败。修复后核对全部受影响图，尤其边文字，不能只测主节点字号。
2. 补最终viewer/structure.html的原生/嵌入章节与交互验证：1440×900和1920×1080，100%缩放，跨层返回、页签、参数/源码联查、长图滚动和查看器缩放。按规格对最终所有图保存对应截图并逐张查看，不能用28图自动通过冒充全图人工目检。单个Archify图的browser-check也不能代替最终离线章节入口验收。
3. 按实际环境使用Archify原生命令；不修改/关闭Chrome sandbox，不绕过访问控制。若你的执行环境仍无法启动Chrome，完成有依据的代码调整与新候选后如实提交，交Codex在已授权环境补跑，不能再次把全项目描述成浏览器不可用或自称验收通过。

新报告D01-display-result.md、新证据D01-display-verify/，均放docs/tasks/deepseek-structure-reading-2026-11-01/。生成新候选，保留此前所有候选和旧证据、Excel字节、dirty tree和两处既有删除；D01-rework-codex-*与协调JSON为Codex外部更新，不改。不得提前做D02–D04语义细分，不关票、不commit/push、不恢复自动化；完成后停止待Codex复核。

额外派发前快照（继承原22242项历史保护，另冻结两个D01候选与相关源码）：/private/tmp/llmarchi-D01-display-baseline-r0r_jicd/manifest.json
