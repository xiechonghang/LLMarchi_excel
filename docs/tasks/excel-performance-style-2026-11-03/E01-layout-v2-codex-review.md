# layout-v2复核：未通过，保留模板布局继续定点返修

日期：2026-11-03。对象：xlsx `2d511eddfa670fd3328410f01c35b3a931c4db7d9e7fbc18b07fd9787bd30d92`，patch `8cfbbef355182664c3633b0c39b5f74d7a00bd50eb02ba35f91df69bd323be41`。依据E01-layout-feedback02.md。用户布局尚未确认，不进入E03。

## Spec轴

### P1：直接参数引用绕过输入合法性检查

隔离源码deepseek_perf_layout_v2.py:578—593的_expr_formula把原始输入直接返回，640—656写入维度格，700—701的成本守卫仅检查数值类型。最终实物Prefill E69=$C$5、G69=D69*E69、J69对G/H/I仅作ISNUMBER，L69=MAX(J69,K69)。

Codex用Artifact Tool导入交付件副本并独立重算q_a_proj，记录：

- seqlen=8：cube/op=8192。
- seqlen=-1：cube=-1024，op=0。
- seqlen空白：T=0，cube/op=0。
- seqlen=1.5：cube/op=1536。
- seqlen=TRUE：cube/op=1024。
- seqlen=abc：未知。

这违反“复杂推导与检查提升参数层，必要输入缺失/非法不能当0”的要求。请在参数层建立经校验的输入/值或状态，让主表直接引用该校验结果；所有维度来源包括基础配置、作用域参数与复合项统一保留既有合法性语义，不恢复逐行长守卫。空白、负数、小数、布尔、数字文本、普通文本逐类验证；允许的真实0与条件不执行0分别保留。对seqlen/kvcachelen/head/分组等适用输入补行为回归，不能只测吞吐。非法依赖不得被MAX/SUM/直接引用伪装0；正常场景成本保持。

### P2：A列结构标题裁切，关键类别不可辨

查看绑定最终重算副本的prefill_main.png：A列宽17、区块行高20，长标题在列边界截断。尤其“路由专家切片（Hash）”与“路由专家切片（TopK）”都只显示“路由专家切片”，括号中的类别不可见；Attention Q/KV前置、mHC调用次数等也有截断。结构只标一次后，这些标题是唯一主表分区标识，批注不能代替可见区分。

请用明确短标签（如Hash专家切片/TopK专家切片）、适当列宽/换行与对应行高修正，不缩小字体或重加重复横幅。逐个核21个区块标题，显示完整结构名和需要的调用次数，保持A列每结构一次。

## Standards与交付轴

### 文档残留旧哈希

layout-v2/reproduce.md:18与E01-layout-v2-result.md:125仍写66c0…c421。实际xlsx、源码独立重建、manifest、verification与patch-integration均为2d511e…d92，截图绑定重算副本03456b…c0b，实物链没有版本分叉证据。返修后全文统一新版本哈希，复现命令补回根目录的cd；不覆盖旧记录来伪装过去通过。

## 已通过范围

- Codex独立运行8项layout-v2测试通过，但未覆盖非法维度输入。
- 两次独立冻结输入重建得到相同最终xlsx哈希；patch仅4个Excel相关新增文件，apply-check通过，内容与workspace一致；11项保护核对通过。
- 模板映射291实例=282计费+8excluded+1container，计费次数合计2979；mHC7行×86、QKV8行×43正确，冻结数据中签名包含参数/条件/guard等，未发现误并。
- 合法输入N_hit=6时，两场景CUBE/VEC/op总成本与前轮口径一致。主表101模板、维度短公式与提取共用结构的方向符合用户要求，返修保留此布局。
- 已实际查看prefill_main截图，来源与哈希绑定核验通过；未代签原生Excel行为或用户阅读确认。

## 交付

继续原DSH Excel会话与隔离workspace，仅修上述问题，保留现有layout-v2文件。新产物用layout-v2-rework/，报告E01-layout-v2-rework-result.md，证据E01-layout-v2-rework-verify/（本任务目录）。交新xlsx、累积patch、全部标题截图和合法/非法输入联动回执；同一最终文件统一构建/验证/截图/哈希。原工作区src/tests/HTML与旧候选不改，不接入E03、不关票、不提交推送、不恢复自动化。完成停待复核。
