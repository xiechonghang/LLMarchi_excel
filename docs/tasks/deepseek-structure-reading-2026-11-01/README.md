# DeepSeek结构阅读：已批准任务

当前状态（2026-11-03）：D01–D05全部技术验收通过，用户已确认本轮DeepSeek结构阅读整体交付完成。最终入口与证据见[D05最终验收](D05-acceptance.md)。正式候选`6ec5957a53a7ad82710eabf8d9bc0287b54789fe6499ef9319777102cfab5fc2`，67图/71图位；父规格#16与旧T14不自动关闭。

以下保留最初批准拆分与依赖记录；实际派发/验收状态以tickets.json及各票验收报告为准。父规格：[GitHub #16](https://github.com/xiechonghang/LLMarchi_excel/issues/16)。D01—D05为本轮编号，不占用旧T编号。

## GitHub入口

- [D01 / #17 正式入口与整网端点](https://github.com/xiechonghang/LLMarchi_excel/issues/17)：无阻塞。
- [D02 / #18 Decoder与mHC](https://github.com/xiechonghang/LLMarchi_excel/issues/18)：被#17阻塞。
- [D03 / #19 Attention与缓存](https://github.com/xiechonghang/LLMarchi_excel/issues/19)：被#17阻塞。
- [D04 / #20 MoE与专家内部](https://github.com/xiechonghang/LLMarchi_excel/issues/20)：被#17阻塞。
- [D05 / #21 全图库验收](https://github.com/xiechonghang/LLMarchi_excel/issues/21)：被#18、#19、#20阻塞。

五票均为#16的原生子Issue，标签为`ready-for-agent`；标签不代表已派发。依赖采用GitHub原生阻塞关系，状态清单见同目录tickets.json。

## 已批准拆分

1. [D01 正式阅读入口与整网简单端点](d01.md)：无依赖。构建可打开的Archify离线入口，只有Prefill/Decode与符号形状，交付整网、Embedding和输出端算子展开及四层导航契约。
2. [D02 Decoder与mHC完整分支](d02.md)：依赖D01。从正式入口读清两段mHC、分支计算和残差合并，并展开权重生成。
3. [D03 Attention与Q/KV/O、三类缓存](d03.md)：依赖D01。保持关联的Attention总览，交付各通路、SWA/CSA/HCA缓存及压缩、Indexer入口与真实数据流。
4. [D04 MoE关系与路由/专家细节](d04.md)：依赖D01。关系层展示路由、激活专家、共享专家；下一层区分Hash/TopK和专家内部。
5. [D05 全图库验收与正式交付](d05.md)：依赖D02、D03、D04。冻结最终实现，整套重建、全图语义/显示复核，交用户整体review。

## 依赖与执行

D01 → D02 / D03 / D04 → D05。后三个模块切片各自只依赖D01，不伪造模块间的阻塞关系。dsh实际可串行执行D01、D02、D03、D04、D05，串行安排不等于依赖。

每票是结构事实→图生成→页面入口→验证的完整切片，均能独立演示。显示问题每票负责，D05负责最终同一候选的全量收口，不允许把可读性全部推迟到最后。必要局部整理并入对应切片；不为未证实的大重构另开前置票。

## 全票共同验收约束

- 父规格AC01—AC23为事实依据；原型只提供展示方向，不是手工节点/坐标真值，也不限制图数。
- 每票交付新候选、可打开的离线入口、源码/config范围、测试、原生/嵌入显示证据和限制。共享显示逻辑变化后检查所有受影响图。
- 文字、输入来源、符号轴、真实边、标注距离、节点/路径/标签/箭头边界、可读比例均核对；不靠省略号、小字、删边或仅有字段骗过检查。
- 接手前登记Excel/历史候选/证据保护清单；交付逐字节比较，既有删除保持。新增图与旧表粒度差异登记，不改公式/计费/汇总。
- 正式门禁、实际浏览器和人工目检分开，绑定最终候选和HTML。环境不可用保留待验收，不绕过保护，不沿用旧候选回执。
- dsh负责实施，完成停待Codex复核；本轮只开发DeepSeek，其他模型仅做必要共享代码保护回归。不自行关票、提交推送、恢复自动化。

## 规格覆盖

AC01—AC07在D01落实主路径，D02—D04继承并扩展自身入口。AC08归D02；AC09—AC11归D03；AC12归D04。AC13—AC22贯穿每票，D05对冻结整套产物重验；AC23由D05收口。

## 既有任务边界

T14维持既有状态，四模型第三轮产物及DeepSeek历史候选受保护。本次是DeepSeek正式集成，不宣告T14通过。本轮仅发布五个子Issue和原生关系，未编辑或关闭父规格正文。实际dsh派发是后续独立步骤。
