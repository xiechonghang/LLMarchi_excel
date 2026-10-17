# 模型结构阅读实施计划v5：Transformers实现与双阶段阅读

更新：2026-10-17。状态：需求已获用户确认，本计划替换旧V1→V2→V3执行主线；已完成代码复用分析；M1主干、独立Excel和独立测试准备已在三个新建VS Code可见终端正式派发，当前处理中。

## 1. 当前唯一执行入口

先读[已确认需求](../structure_reader/docs/requirements_alignment_2026_10_17.md)、[代码复用核查](../structure_reader/docs/hf_reuse_audit_2026_10_17.md)，再按本文推进。最新实际进度只记在[progress顶部](../structure_reader/docs/progress.md)。

用户当前指令优先，其次是已确认需求和本计划。旧model_structure_plan_v3/v4、project_execution_handbook_v4及历史S/V/B/R编号用于查历史，不再补充新需求或决定下一派单。旧参考实现验收结论保留其原范围，不继承为HF实现通过。

输入约定：模型HF资料由scripts/download_hf_model.py无权重下载到models/，其中含config.json；模型实现源码使用用户从GitHub克隆到hflib/transformers的官方Transformers仓库。开发不重新下载或执行模型包参考程序。

当前目标是先交付DeepSeek HF decoder阅读样板：mHC、Attention/KVCache、MoE，prefill与decode各一套完整内容；离线HTML与两表Excel同时交付。后续补齐配置实际启用的变体、整网至logits和项目Skill。Qwen或其他模型不作这个闭环的前置。

## 2. 范围与不变量

- 只采用指定版本的Transformers模型源码及必要被调用定义，加对应HF config.json及其配置类解释；选择eager、普通推理、use_cache开启，不加载权重或执行模型参考程序。
- 首个样板建议选真实layer_idx=4：CSA Attention + TopK MoE + 两处mHC。其余变体清楚标未覆盖，允许分批，但最终不能遗漏配置下实际使用的结构。
- 页面先完整Prefill Decoder，再完整Decode Decoder，不以阶段/章节标签切换隐藏内容；阶段内部可有锚点、局部放大和详情，但主数据流必须直接可读。
- 算子粒度为矩阵乘、维度变换、mul/add等；熟悉算法可为具名算子。黑盒醒目标注“内部未分析”，由用户验收后决定是否继续展开，不能将整个Attention/MoE/mHC仅画外框就声称已交付其结构。
- 默认符号shape并标注配置值。层数、专家数、head数、组数等由参数体现，无“重复次数”列；同构模板去重，但保留真实顺序、参数身份及变体归属。
- KVCache讲清状态、shape、读写、当前位置和条件公式。区分本次可读视图、持久存储、容量和有效长度；解码历史作为输入前提，不用P>0证明缓存内容有效。
- 输出.xlsx仅Prefill、Decode两表，各自含参数符号区及引用这些参数的算子区；不能用CSV替代，不能新增第三张参数表。
- 正确性目标为静态结构一致。不进行训练、设备优化、生成采样或数值等价扩展；测试覆盖真实结构分支，不机械枚举大量长度。

## 3. 保留与迁移策略

1. 原样保留graph_ir、graph_document、graph_contracts及scenario_evaluator主体；确有新需求缺口才做小范围扩展，并回归受影响通用测试。
2. 复用注册查找、输入SHA、取证片段、原子写入、冲突拒绝和失败隔离机制；HF注册、有效配置、事实提取、层分组及模型检查另作显式适配。
3. 新HF注册键建议为deepseek_v4_flash_0731_hf，implementation_kind=transformers，analysis_path=eager。旧deepseek_v4_flash_0731保持历史参考实现身份，不静默改变同名命令含义或覆写其候选。
4. 新HF构建不读取reference ModelArgs、inference/config.json、generate.py或DSpark图；也不要求它们存在才能完成HF样板。
5. 七个DSpark提取器和dspark_contracts保留历史，不进入新默认构建、tool hash依赖和完成条件；旧B4补修及其正式V2门禁退出当前待办。
6. 旧参考图、变体和契约只提供主题与检查经验，HF事实从当前源码重建；模型专用可信目录和测试期望不能从新生成图自动复制得出。
7. 本次不删代码、不搬旧证据、不清候选、不整理整个历史测试树。新增HF功能按提交形成清楚边界，避免先做全仓大重构。

## 4. 新HF事实与最小接口

### 输入与有效配置

在既有注册基础上增加明确的HF来源类型。输出raw_config、effective_parameters和source_manifest；每个参数说明原字段、默认/别名/派生规则与来源。

必须处理本地DeepSeek配置类中的attribute_map、legacy compress_ratios→layer_types、num_hash_layers→mlp_layer_types、RoPE派生参数。43层主干与46项legacy数组区分，按HF解释截取；不存在源码调用依据的nextn/DSpark字段不自动转成计算结构。

来源清单从实际调用闭合：先固定modeling_deepseek_v4.py、configuration_deepseek_v4.py和根config，再按所选函数使用纳入cache_utils、masking_utils、modeling_rope_utils、activations及其他必要定义。只追踪影响当前结构/维度/缓存的定义，不扫描或执行整个Transformers框架。未解析的外部调用明确标记。

### 同一份阅读数据

以现有SemanticGraph为结构底座，增加薄阅读文档对象，提供：

- 模型/实现/配置/路径身份和覆盖范围。
- 可复用结构模板及其层分组/专家归属。
- prefill、decode各自的参数符号、算子、连接、缓存读写与必要条件。
- 算子到输入输出shape、权重shape、配置参数、来源和分析状态的引用。

HTML和Excel消费同一个对象。可用小型dataclass或明确字典契约实现，不建设通用工作流平台、Python解释器或抽象适配框架。最小对象形成后即可并行实现阅读页与Excel。

### 建议改动位置（开发任务已派发，尚未验收）

- 新增rules/deepseek_v4_hf.py、extract/deepseek_v4_hf.py、checks/deepseek_v4_hf_contracts.py。
- 小范围适配model_registry/inventory/evidence及构建调用点，显式区分旧reference与新HF。
- 新增render/decoder_page.py及必要模板资产；已有交互小函数确实可分离时才复用。
- 新增export/operator_workbook.py，从同一阅读数据生成xlsx。
- 构建编排归入build_pipeline，CLI只做参数与导航；不再复制两套HF图构建。

这些路径是实施建议；文件是否存在和功能是否支持以实际交付与验收为准。旧inspect/build仍是参考profile现有能力，新HF命令在实现后才加入用户指南。

## 5. 里程碑一：真实decoder阅读样板

### 先完成最小闭环

按下面顺序推进，在同一个样板里形成成果，不拆成大量独立门禁：

1. HF注册、必要源码清单和有效配置；能得到layer_idx=4的实际结构参数及来源，不再借用旧ModelArgs。
2. 保存一份真实decoder双阶段图与算子/参数阅读数据，立即运行冒烟检查；优先mHC→norm→Attention/KVCache→mHC残差、mHC→norm→MoE→mHC残差的整体闭合，再补模块内部。
3. 同批生成单文件HTML和Prefill/Decode两表xlsx，尽早给用户可读结果。不要等完整43层变体、Skill、全套旧门禁或新通用框架完成后才出样板。
4. 在当前源码临时输出根验证、修正实际样板，再提交里程碑核查与用户阅读验收。

### 样板必须表达

- mHC：流维度、权重生成、pre/post/comb及残差混合；comb方向依据HF实现。
- Attention：低秩Q、共享KV、部分RoPE、CSA压缩与Indexer、窗口/压缩KV读写、mask/block_bias、eager QK→scale/mask/sink→Softmax→AV，以及逆RoPE和分组输出投影。
- MoE：路由打分与TopK选择、权重归一、分派、合并gate/up投影及拆分、clamp/SiLU/mul、down投影、加权汇合、共享专家；同构专家只一份模板。
- 两阶段缓存：新输入、历史缓存、持久保存状态、当前Attention读到的视图、buffer/压缩条目/overlap及位置关系；未知运行时内容不填虚构值。

### 完成判据

- 用户能打开HTML，先完整读prefill再完整读decode，不需要切换章节才能看到另一部分。
- xlsx只有两表；每张自带参数符号区，所有算子行有名称、输入输出shape及必要权重shape，参数引用在本表可解析，无重复次数列，无256套专家副本。
- 图、页面与Excel的算子/张量/参数对应一致；实际文件已保存，不能以截图、README或开发摘要代替。
- 来源身份、配置映射、关键连接/轴、mHC方向、共享KV与缓存读写顺序核查通过；未分析项明确列出。
- 相同输入重复构建稳定复用；坏结构/来源失配/缺必需xlsx依赖只留诊断，不发布标为完成的候选。
- 当前里程碑的工程核查与用户阅读结论分别记录；用户提出展示修改时先修样板，再扩大覆盖。

## 6. 里程碑二：补齐当前配置实际启用的结构变体

在样板接口与阅读方式被确认后，补齐实际四组decoder模板：

- SWA + Hash：层0、1。
- CSA + Hash：层2。
- HCA + TopK：层3、5、…、41。
- CSA + TopK：层4、6、…、42，复用首批样板。

公共mHC/专家计算模板复用，三类Attention及其不同cache、两类路由保留真实差异。两个阶段内按结构分组连续显示；不复制每一层，不拼造不存在的组合。Hash路由的input_ids外部输入必须保留。

完成判据：43层恰好全部归属、无漏层或重复归属；目标配置实际使用的变体在HTML和两表Excel中均可定位；各自源码与cache规则已核对。只增加能区分新变体的必要负例/代表分支，不重新引入旧长度矩阵或DSpark链。

## 7. 里程碑三：整网概览、Skill与完整DeepSeek交付

- 加入粗粒度Embedding→mHC流展开→分组Decoder×N→HyperHead/Final Norm→LM Head→logits总览，按HF实际forward连接；不扩到采样或生成文本。
- 保留模块细节和双阶段完整阅读、结构去重与参数联查；整网只补导航和必要接口，不平铺43层全部算子。
- 工程命令稳定后编写项目Skill，只编排已实现的输入定位、分析、生成、核查和结果导航，不复制一套模型事实或偷偷调用其他模型服务。
- README、用户/开发指南、输出字段与实际命令同步。必要的额外JSON/内部报告可保存，但六类CSV、额外CLI子命令和第二模型不作首版强制条件。
- 对完整DeepSeek HF范围完成一次对应源码版本的独立核查和用户阅读验收；第二模型由后续需求决定。

## 8. 检查策略：正确性充分，场景数量受控

### 保留必要防线

- 通用图引用、方向、shape轴及参数来源；源码到图/表的对应关系。
- HF配置的关键解释分支、模块内真实连接、mHC方向、KV共享、mask与路由权重来源。
- 两阶段cache新建/读取/更新/返回顺序；无新压缩条目和有新条目等真正改变结构的条件，用符号规则及少量代表例核查。
- HTML含两个完整阶段、黑盒醒目标注、专家模板去重；xlsx打开/读取、两表与参数引用正确、与HTML同源。
- 源码失配、错误连接或维度、结果冲突、必需依赖缺失与失败隔离。

### 控制验证范围

通用基础改动回归受影响通用测试；HF模块改动运行对应HF测试。里程碑完成时跑一次当前有效范围的集成检查；发现具名缺陷后重测影响范围，避免每小改动全仓旧1538项加全部旧候选重建。

新的测试使用随源码保存的独立期望、小型配置/必要文本fixtures和探针，在tempfile输出根生成当次HTML/xlsx。不读取.work/coordination、按mtime选旧outputs或依赖已有候选。完整模型集成检查仍明确依赖只读HF输入；缺输入时如实报前提未满足，不下载权重、不伪造通过。

Node/浏览器依赖发现采用可配置路径或环境查找，缺失时标明未覆盖；跳过不能称页面行为通过。浏览器实际CSS/事件与用户阅读必须用真实候选核查；现有file安全限制不得用本地服务、换协议或浏览器/CDP绕过，可由用户打开真实离线文件后反馈。

xlsx为必需交付。本机工程解释器已能发现openpyxl，但实现阶段仍须真正生成并读取验证；其他环境缺依赖时明确诊断。避免工作簿创建/修改时间或ZIP时间导致同输入不稳定。

## 9. 协作、版本与接续

- Codex维护需求/计划、划定范围、审查实际成果并记录验收。当前任务只完成重规划，不自行接手产品实现。
- Claude承担HF事实提取、规则、检查与构建主干；按用户最新要求在VS Code为新阶段新增项目终端并启动claude新对话，不复用旧B4上下文。先核对项目，保留默认模型/账户与既有权限边界。
- 用户已允许pi并行，最新指定qwen3.8flash（pi模型ID qwen3.8-flash），覆盖先前glm5.3flash。按最新要求新增VS Code项目终端启动pi新对话；以docs/hf_reading_contract_v1.md冻结接口，先独立承担operator_workbook.py及自身测试，不修改Claude文件。先确认真实模型，不静默替换。
- opencode在新增VS Code项目终端启动新对话，保持huoshanai/deepseek-v4.1-flash；先做独立期望/测试准备（产品pending），再对新里程碑当前冻结版本发自包含验证任务；不重复旧阅读门禁或把旧参考V2通过当新HF通过。用户可先阅读明确标为候选的样板，工程独立结论随后补齐；二者不互相代签。
- 不后台/隐藏调用模型，不读取凭据、其他项目或权重，不安装下载或导入执行参考模型。依赖确有缺失时记录实际阻断再处理。
- 开发按工作包本地commit，提交注明范围。models/、hflib/继续Git忽略且只读；旧outputs/.work不自动入库或清理。
- 新正式验收结论与可复验指纹保存在Git跟踪的docs/acceptance下，必要测试fixtures在tests内；生成大产物仍在outputs，记录其内容ID与文件指纹。不要再将唯一正式结论只存入“可丢弃”的.work。
- 每包只需任务、公开结果、必要原始命令退出码；长期入口只有需求、本文、复用核查和progress最新状态。每里程碑形成一份简短验收记录，不增加逐微任务三件套。
- 现有automation处于PAUSED，保持暂停，不在重规划时恢复。后续如恢复监控，先更新为新HF里程碑及实际派单断点；不沿用旧B4/V3指令。

## 10. 当前下一步

### M1并行调度补充（2026-10-17，用户要求加强并行）

正式验收需要固定候选，但验证脚本和独立期望可以与开发同时推进。独立工作完成后应检查M1当前待办中是否还有不重叠的有效任务，不能笼统等主干全部完成；也不为了维持忙碌而新增M2/M3范围。

- Claude：继续HF M1修复01，拥有HF rules/extract/checks/build/CLI/HTML及自己的HF测试。修正真实数据流与shape是当前关键路径；不编辑Excel导出器或独立验证目录。
- pi（qwen3.8-flash）：已完成Excel模块与同名参数修复，新增并行任务是复合shape组成参数的本表联查，例如Nh×Dh、B×T、W-1；只改原export/operator_workbook.py与tests/hf测试/fixtures，冻结公开API和两表布局。不解释HF模型事实、不建立通用求值器、不等Claude数据改好。
- opencode（huoshanai/deepseek-v4.1-flash）：已完成独立源码准备，新增并行任务是将源码期望接到reading_document/HTML/xlsx产品检查；只改tests/hf_independent。使用手工小fixture自测，对显式固定的已拒绝首版候选做只读诊断，提前纠正自身CSA/HCA分支等准备错误。该工作不签新产品正式通过，不追随开发中的工作树反复构建。
- Codex：维护冻结接口和互斥文件范围、派单并确认收到、复核各自产物；有实质新缺陷时汇总交回对应开发者，避免重复派单。三方完成后核对当前文件指纹，统一构建一次稳定候选、两次复用，再执行对应候选的正式独立验证。

同一工作树允许不同文件并行，但最终身份必须覆盖实际使用的Excel导出器版本。若构建期间独立文件仍变化，不能声称同版稳定复用；各自公开实际版本，由Codex在收口时重新核对。任务明细和收到状态保存在.work/coordination/hf_m1，长期实际断点以progress顶部为准。

实施计划与复用分析已经形成。下一开发工作包是“HF decoder样板双阶段闭环”，从真实HF注册/有效配置和layer_idx=4开始，交付当次生成的HTML与两表xlsx；不是再补旧B4，也不是先迁移全部历史工程。

本文中所有新模块、命令扩展、样板页面、Excel和Skill均仍待实施；当前没有新HF产品通过结论。
