# D05 最终技术复核通过，待用户整体确认

日期：2026-11-03（Asia/Shanghai）。D05/#21最终交付工程复核通过。产品与D04 accepted完全同源同字节，用户整体确认仍待回复；不提前关闭D05/父规格/T14，不宣称DeepSeek整体已验收。

最终候选：`6ec5957a53a7ad82710eabf8d9bc0287b54789fe6499ef9319777102cfab5fc2`。
正式入口：`structure_reader/outputs/deepseek_v4_flash_0731_hf/6ec5957a53a7ad82710eabf8d9bc0287b54789fe6499ef9319777102cfab5fc2/viewer/structure.html`。
页面：2285219 bytes，SHA256 `09a5d7988a4aee1e956ec3872fa34c0170154009b251fa7eb6a4da057a34fc39`。

## 冻结与构建

Codex独立检查222/222基线源码/测试/配置未变，候选source_manifest的工具源、Archify产品源与模型/config输入指纹均匹配当前文件；Archify CLI指纹亦匹配已接受身份。29425/29425历史输出未变化。协调文件变化只有此前Codex派发更新的tickets.json。

再次实际运行正式build-hf --all-variants --archify-layers命令，returncode=0、reused=true、同一analysis_id，页面哈希不变。真实argv/cwd/PYTHONPATH/返回码在`D05-codex-verify/build-execution.json`，原始stdout/stderr另存。没有以已有build.log替代本次执行，也没有为新ID改产品源码。

## 证据复用逐项成立

不是只比正式页哈希：独立核对67张最终candidate JSON、HTML及manifest的view/relations/source指纹，逐张与D04 accepted原生delivery/browser回执绑定值比较；全部一致。71图位正式页与D04浏览器探针绑定页一致，178截图存在。故明确复用D04授权环境67/67原生全部门禁和804/804页面断言，不再运行同一浏览器整套。

D04的20张MoE最终事实审计，以及D01–D03已接受结构/语义结论，随未变源码/输入/候选关系文件继承。D05没有修改产品，未重复跑模型/图/Excel不相关测试。dsh本轮565/565为其报告测试结果，Codex没有冒称重新运行全套；阅读59项独立结果来自同身份D04复核。

复用与保护的实际绑定清单见`D05-codex-verify/binding-and-protection.json`，可复跑脚本`verify_freeze.py`。

## 交付文档补齐

原D05交付存在文档缺口，由Codex补齐而不改产品或重新派实现：

- 原native-summary在“复用D04”标题下误写dsh环境67次失败；已纠正为已接受的D04-rework-codex-verify 67/67 browser通过，原dsh回执/freeze原始记录保留且注明指针区别，不篡改证据。
- 新增`D05-codex-verify/final-inventory.json`和`.csv`：67图/71实际图位的层组、层号、类型、raw/normalized、阶段、路由、figure id与完整文件哈希。实际图位/路由来自同页面真实探针。
- 新增`graph-operator-map.json`、`graph-to-workbook.md`：逐图节点/阶段到同源旧表构建输入算子键、关系容器下钻、重复阅读视图不重复计费、外围/状态投影边界。用(group_id, operator_id)键，不臆造Excel行号，不改工作簿/公式。
- 最终阅读指南的入口改为可点击本地绝对路径，并将目标模型名称纠正为DeepSeek V4 Flash-0731，避免与dsh服务模型版本混淆。D05-result新增补充证据入口，原始冻结JSON保留。

AC01–AC23清单已交；AC01/20/21绑定与冻结、本次构建由Codex独立核验。AC03–18等按最终未变字节/依赖复用已接受技术结论。AC19按用户要求由用户审视觉，D01–D04已分阶段确认；D05整体review和AC23最终用户认可仍为待确认。

## 限制与下一步

本轮无新浏览器截图、不逐张读取长图；不把同字节复用称为新截图/新门禁。复合长页通用browser-check的纵向overflow限制原样保留，其与67张原生严格通过不是同一范围。

已在正确dsh会话“实施 D01 图形显示修复”只读确认D05完成、最终候选一致、空输入与无停止生成按钮。未操作E01/旧T14、不Git、不改自动化、不推广其它模型。

当前聊天已请求最终整体确认：同一已审页面，无需重新逐图阅读。收到确认后再完成D05阶段验收；父规格/T14的关闭不自动执行。
