# E01 Codex复核：未通过，返修E01

复核日期：2026-11-03。固定范围为E01-source.patch的三个新增文件及冻结快照，不包含并行HTML工作区改动。规格依据plan.md/E01-dispatch.md。交付xlsx sha256：7abe374d3978b0d3cc0579787dafa054b2a84a03ba22d852937f70d7c135a48e。

## Spec轴

1. **P1：吞吐守卫未使用，负耗时被MAX掩成0。** 快照 `structure_reader/src/structure_reader/export/deepseek_perf_layout.py:929` 解包thru_guard，931行生成公式却只检查维度guard。实际Prefill J67除以J7但不检查ISNUMBER/>0。Codex用Artifact Tool对导入副本内存重算：J7=-4096时J67=-8192、K67=0、L67=0；J7="4096"数字文本被接受，J7=TRUE时J67/L67=33554432。与非法吞吐显示未知的约定冲突。修复时合并对应吞吐守卫，CUBE/VEC分别覆盖负数、零、空白、文本数字、普通文本和布尔；结果及依赖小计不得伪装健康0。条件明确不执行时仍保留约定的合法0。
2. **P2：batch并未固定为1。** 600—605行将普通C12值引用为B且guard=TRUE；CUBE保留B、VEC映射剔除B。内存重算：C12=2时q_a_proj成本从8192变16384，q_a_norm仍128。请在计算源将B固定1，或显式校验固定单元格恰为数值1并拒绝非法改动；不能静默生成不一致结果。补对应联动回归。
3. **P2：新表吞吐单位未统一。** 632—634行及实际Prefill/Decode K7/K8仍有MAC/ms、元素/ms，而新表约定均为element/ms。统一新表可见单位、必要辅助文案与报告，乘积口径不乘2；旧导出历史口径保持不变。

## Standards与交付轴

4. **P2：验证目录越出约定路径，报告定位不一致。** 报告称证据在 `docs/tasks/excel-performance-style-2026-11-03/E01-verify/`，该路径不存在；进一步核查，verify_workbook.py、verification.json、protected-paths.sha256、patch-integration.txt实际均交付于仓库根目录 `E01-verify/`。不是材料缺失，根目录复现脚本入口也存在；但该目录不属于派单允许新增范围。返修证据写入约定任务目录，所有报告、复现命令和清单统一使用准确路径；绑定新xlsx与patch哈希重新出具回执。原目录保留作为旧证据，不擅自删除。现有验证未覆盖上述非法吞吐与固定batch情形，必须补上行为回归。

## 已核查通过的部分

- Codex独立运行新布局11项测试通过，但现有测试未覆盖上述非法吞吐和固定batch情形。
- Codex用bundled Python从冻结JSON独立重建；xlsx与交付件字节相同。
- git apply --check通过，patch仅新增三文件。114项快照哈希一致；原目录与快照差异为报告列出的5个HTML相关文件。仅凭哈希不代签其具体修改者。
- 已查看Prefill顶部/算子区及辅助明细截图，主表为12列且维度与成本同排；当前截图不能替代上述联动验证。用户阅读体验未确认，原生Microsoft Excel未验证。
- 抽查head折入有效M、repeat及复合矩阵项SUM、op前数值检查、汇总COUNT完整性符合计划。未将同源自测等同完整正确性证明。

## 返修范围与停点

在现有E01隔离快照继续，只修以上四项；保留旧交付和报告，新文件用rework前缀或独立子目录，交付新xlsx/patch与E01-rework-result.md、E01-rework-verify/。旧源工作区、HTML、D01—D05、历史候选保护不变。完成停止待Codex复核，不启动E03、不关闭Issue、不提交推送、不恢复自动化。
