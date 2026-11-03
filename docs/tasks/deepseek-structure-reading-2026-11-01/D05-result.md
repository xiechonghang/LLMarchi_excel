# D05 最终冻结、全图库一致证据与正式交付

日期：2026-11-03（Asia/Shanghai）。D01–D04 已技术复核并由用户确认视觉；本轮只做
D05/#21 最终冻结、全图库证据与正式交付。未关闭 D05/父规格/T14，未推广其它模型。

## 1. 最终候选与构建

| 项 | 值 |
| --- | --- |
| 候选 | `structure_reader/outputs/deepseek_v4_flash_0731_hf/6ec5957a53a7ad82710eabf8d9bc0287b54789fe6499ef9319777102cfab5fc2` |
| 正式入口 | `viewer/structure.html`（2,285,219 bytes，sha256 `09a5d7988a4aee1e956ec3872fa34c0170154009b251fa7eb6a4da057a34fc39`） |
| 图 | **67 张** Archify 图、**71 个**嵌入图位 |
| 源码 | D05 基线 222/222 未变；构建确定性复用 **同一候选 ID** |

实际构建：

```bash
PYTHONPATH=structure_reader/src \
  /Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -c "from structure_reader.cli import main; import sys; sys.exit(main(['build-hf','--model','deepseek_v4_flash_0731_hf','--all-variants','--archify-layers','--json']))"
```

argv/cwd/返回/日志在 `D05-verify/build.log`；源码、tool、archify product 指纹在
`D05-verify/freeze.json`。本轮未修改产品源码，不制造新 ID 或新页面。

## 2. 全图库

从最终 `data/archify/manifest.json` 枚举，完整 67 行清单见
[`D05-verify/graphs.md`](D05-verify/graphs.md)，机器清单见 `graphs.json`。

| level | 数量 | 说明 |
| --- | ---: | --- |
| network / embedding / head | 3 | 整网与两端点 |
| group / decoder | 8 | 四真实层组 + Decoder |
| mhc | 4 | 两段 mHC、combᵀ、局部参数 |
| attention / kv | 10 | 4 总览 + 1 raw + 4 KV + 1 raw |
| q / o | 8 | 独立 Q/O 通路 |
| cache-main / cache-compressor / cache-indexer | 9 | 三类缓存分图 |
| compressor / indexer | 5 | 压缩器与 Lightning Indexer |
| moe | 4 | 第三层模块关系 |
| moe-route / moe-expert / moe-shared / moe-combine | 16 | 第四层路由、单专家、共享、汇合 |

页内 71 个图位来自 67 张图 + group/Attention 等页签中的重复嵌入位；完整路由清单
和每图 HTML/relations 哈希见 `graphs.json`。

## 3. AC01–AC23

逐项状态见 [`D05-verify/ac-evidence.md`](D05-verify/ac-evidence.md)。摘要：

- **实测**：AC01 构建/来源、AC02 阶段/59 项阅读、565 项全套测试、AC21 历史保护、AC22 诚实边界。
- **同字节复用 D04 accepted**：AC03–AC18、AC20；最终页面 sha256 与 D04 accepted
  完全相同，D04 的 67/67 native、71 图位 804/804、178 截图按同候选同字节复用。
- **用户已确认/最终待整体**：AC19 的 D01–D04 已由用户逐阶段确认；D05 整体视觉
  review 仍待用户。
- **待复核/待用户**：AC23 整体交付需 Codex 技术复核 + 用户整体 review；不提前宣称
  DeepSeek 完成。

## 4. 语义/来源审计

沿用 D01–D04 已接受审计并核对最终同候选字节：

- D03：阶段输入由真实 op facts 推导，未知 id 拒绝；图内外双消费扇出、缓存累计投影、
  Attention/QKV-O、压缩器/Indexer 全部保留。
- D04：MoE 关系 7 节点；Hash `tid2eid`、TopK `e_score_correction_bias`；原始 scores
  gather/归一化/`routed_scaling_factor`；真实 `moe_combine/index_add`、token_idx 回填、
  weights 只乘一次、共享 Clamp 与全 token、局部参数/mHC 来源。
- 最终 candidate/relations 全部真实输入输出/消费者/权重/条件/循环边界；共享消费者与
  段出口不丢，纯内部张量不冒充输出；只有真实 `[]` 才为标量。
- 67 图标签重叠 0、SVG 文字重叠 0、省略号 0；71 图位短 iframe 0；
  6px 门槛沿用 D04 accepted 804/804。

## 5. 图表粒度与旧 Excel 映射

- 关系层（MoE/Attention）只画模块关系；内部层用真实 op 展开；合并向量项（如
  cache/压缩状态）在图中按阶段聚合，逐 tensor 身份/shape 在页面表/relations 中保留。
- 图只增加阅读层粒度，不改变既有 Excel 公式、计费、性能汇总或 layout_v2；
  最新两候选 `decoder_variants.xlsx` 逐字节一致。
- 旧表字段与图节点的对应/差异在 D01–D04 结果及本页图库中登记；不重复计费。

## 6. 正式阅读指南

见 [`D05-verify/reading-guide.md`](D05-verify/reading-guide.md)：

整网 → 四层组 → Decoder/mHC → Attention/QKV-O/缓存 → MoE 关系 → 路由/单专家/
共享/汇合 → 来源/参数；仅 Prefill/Decode，`seqlen` 为本次输入，`kvcachelen` 为逻辑历史。

## 7. 保护

- **29425/29425** 输出文件未变，硬变更 0；
- 6770 协调文件：6769/6770；`tickets.json` 为外部派发更新；
- 22242 继承保护：22240/22242，仅 `.DS_Store`、`monitor-state.json` 外部动态；
- D01–D04 已接受候选、两处删除、旧证据只读；Excel 最新两候选逐字节一致；
- E01 隔离树未触碰；当前 dirty 127（外部 E01 项/新增候选未跟踪，未回滚）；
- 源文件 222/222 未变。

## 8. 限制与停止点

1. D05 未新增浏览器截图；按同候选同页面字节复用 D04 的 804/804 和 67/67 native。
   本机 Chrome sandbox 受限，未重复无效启动。
2. 用户整体视觉 review 待完成；D05 未逐张目检、未宣称整体验收。
3. 父规格/T14 未关闭，不 commit/push，不恢复自动化，不启动其它模型。
4. 停待 Codex 技术复核与用户整体 review；确认前不对外称 DeepSeek 整体完成。

## Codex 交付补充（2026-11-03）

- 独立构建的真实argv/cwd/returncode见 `D05-codex-verify/build-execution.json`，构建stdout仍为同ID、reused=true，页面字节相同。
- 最终67图/71图位的类型、层号、形态、实际导航路由与完整文件指纹见 `D05-codex-verify/final-inventory.json` / `.csv`。
- 图节点到旧表构建输入算子键的逐图对应与粒度说明见 `D05-codex-verify/graph-operator-map.json`、`graph-to-workbook.md`。
- 原生摘要中误引用dsh受限环境次数的文字已纠正；复用的实际通过回执来自 `D04-rework-codex-verify`，绑定核验见 `D05-codex-verify/binding-and-protection.json`。旧原始回执与freeze.json保持可追溯，不把受限失败写成通过。
- 仅补交付文档/证据，不改产品源码、已接受候选或Excel；用户整体确认仍单独等待。
