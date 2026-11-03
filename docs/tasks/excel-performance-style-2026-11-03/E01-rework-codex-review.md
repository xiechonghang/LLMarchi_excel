# E01返修独立复核：工程通过，待用户阅读

日期：2026-11-03（Asia/Shanghai）。复核对象为返修xlsx与返修patch，不覆盖首版未通过结论。新xlsx SHA-256：`6d376d94088a5f0b280526fb8e7a93be2c6cacbec218186c46aa9ffa7a745335`；patch：`d0b616c8c60498f2ca73ee8f3f8547130ca8397236a57e5b2718e26a0a69662b`。

## Spec轴：通过

- 两页564条成本公式均包含对应吞吐的数值、非布尔、正数守卫；普通CUBE、VEC、repeat和复合项均覆盖。条件0外层判断保持。
- B/batch_size计算来源固定为常量1，不引用可见C12；单元格仅显示“固定1”。
- 两页K7/K8均统一element/ms，成本乘积不额外乘2，维度映射与层数计费未改。
- Codex在Artifact Tool导入副本独立重算，CUBE吞吐=-4096/0/空白/数字文本/普通文本/TRUE均显示未知；改C12=2后q_a_proj=8192、q_a_norm=128，保持batch=1结果。
- Codex独立运行交付验证脚本，使用Codex bundled Python/Node调用现有LibreOffice Kit，得到 `passed=true`、`recalc=true`、`errors=0`。证据在[E01-rework-codex-verify/verification.json](E01-rework-codex-verify/verification.json)。包括两场景逐算子对照、独立吞吐联动、N_hit、条件0、非法seqlen、合法0、12种非法吞吐、batch改动和旧导出字节一致性。

## Standards与交付轴：通过

- 14项新布局测试由Codex独立执行通过。冻结JSON重建新xlsx得到相同交付哈希。
- git apply --check成功，patch仍只新增Excel模块/专用脚本/测试三文件，不修改HTML或主CLI。
- manifest的五项交付哈希、11项保护清单及五张截图哈希与实物一致；旧xlsx保持原哈希。证据路径已统一至任务目录，根目录旧证据保留。
- 已查看返修Decode顶部与算子区截图：element/ms与固定batch显示正确，参数、列标题、数值和分区可读。其余版面未改，沿用首版已查范围与返修截图绑定核验。
- 唯一非阻塞文档问题：rework/reproduce.md第一段cd进workspace后未返回根目录。Codex补充 `cd -`，未修改源码、xlsx、patch或DSH验证回执。

## 结论与边界

首版四项返修已解决，无剩余工程阻塞。E01独立样表可交用户阅读；本次独立复核完成E02的工程检查。用户尚未确认布局，不替用户验收使用体验。原生Microsoft Excel字体、打印与联动未实际验证；重算结果来自LibreOffice及Artifact Tool，不代签原生Excel行为。

E03正式集成尚未启动；源码patch未应用主工作区，旧导出与HTML路径保持。GitHub #22保留开放，不自动关闭、不提交推送、不恢复自动化。
