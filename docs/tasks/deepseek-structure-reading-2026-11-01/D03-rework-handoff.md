/archify
继续当前会话，只返修D03/#19。Codex复核未通过，不启动D04/D05。

完整读取 docs/tasks/deepseek-structure-reading-2026-11-01/D03-codex-review.md（五项阻塞及独立证据），以实际最终candidate/relations/浏览器实测为准。原D03-handoff.md、d03.md与父规格仍有效。仓库 /Users/xiechonghang/Workspace/LLMarchi_excel，保持1017 High、工作区内修改。

返修前候选7c724c14554c404fc838785a956819f85a0f3d8039a353d9a9792f981274fe01；新快照 /private/tmp/llmarchi-D03-rework-baseline-x57e9jsx/manifest.json，继承D03 22641输出文件与历史保护，保留当前候选、Codex证据及dirty tree/既有删除。新构建写新候选，不覆盖旧候选/报告。E01隔离边界保持，不碰Excel公式/求值器及其独立输出/文档。

一次收尾五项：
1. block_bias是实际计算节点，不能当无来源aux输入；最终Attention三组实际交付必须保留compressor→bias、CSA的Indexer→bias、bias→mask_extend。恢复test_archify_preview完整trace断言，不能expected减掉丢边。测试须覆盖最终candidate/relations真实连接，不仅bundle字段。
2. HCA block_bias自身类型/位置输入/压缩率/因果规则及None分支正确，无CSA条件/TopK/Indexer；HCA细图纳入hca_block_bias并处理mask_extend真实边界，不能仅用c_*前缀遗漏。
3. 缓存调用前/后逐项身份/完整符号shape在实际主图或真实下钻可见，不能六项只显示第一项shape而其余藏JSON。补cache.update新KV、本次可读full、累计长度更新的真实来源/边界；cache/compressor/indexer就近参数各自精简且完整，Indexer专属参数与来源显示。
4. 布局修复8张新增图双视口字号，共16真实失败。HCA原生viewBox4616、1440字号2.333px，不能无限扩画布或靠缩字/隐藏删边。重排或合理分层，保完整文字与真实边，原生和嵌入均达原6px门槛。Codex现有证据Chrome available，不能笼统写环境失败；你受限环境一次检查不可用即可交Codex补跑，勿反复已知失败。
5. 保留关联Attention总览，补Q、O各自真实可达通路/返回，与KV并列但无伪造顺序数据依赖；标题不能冒充独立展开。

复用已正确的并行Q/KV、q_residual→Indexer、mask非物理裁剪、D01/D02所有验收功能与历史/Excel保护。不做通用重构/新门禁平台。新增必要最终公开行为回归，实际完整构建交付，所有受影响图机器检查。

原只读D01 probe会动态遍历全部图位，但有33硬编码断言。可复制成D03-verify内新版探针并只将数量契约改为manifest/实际42或修复后合法图数，新增Q/O会增加数量；保全部旧断言与阈值，旧探针不改。Codex旧探针本轮514项496通过/18失败中2项数量断言为历史工具假设，不是产品缺陷；16项字号失败不得忽略。实际源码/HTML/回执绑定新候选。

交付 D03-rework-result.md、D03-rework-verify/：五项逐项响应、新候选及正式入口、源码事实/实际构建命令、测试、原生及嵌入机器检查、保护核对、精简用户视觉入口与限制。用户负责人眼视觉，不要求你或Codex逐张读取长截图，不以未逐张目检拖延交付。视觉项如实写待用户审查；Codex补跑浏览器缺口。

Codex报告/证据/协调JSON/旧handoff只读。不关票、不commit/push、不恢复/创建自动化、不操作E01或旧T14、不启动D04/D05。完成本轮D03返修后停止等待复核。
