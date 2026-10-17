# 按新需求核查既有工程（2026-10-17）

本次核查对应用户已确认的[需求书](requirements_alignment_2026_10_17.md)，用于决定保留、适配及退出当前交付范围的资产。新的执行安排见[实施计划v5](../../hf_docs/project_execution_plan_v5.md)。本次没有修改产品源码、输入模型、第三方源码、旧候选或冻结证据，没有把旧验收结论改写为HF实现通过。

## 1. 结论

现有工程可以保留通用图表示、引用检查、受限表达式求值，以及内容寻址和失败隔离的实现经验；HF模型事实提取、配置解释、缓存描述、阅读页面及Excel需要适配或新增。

主要问题不是图太大，而是当前事实来源为模型仓库inference参考实现。即使把旧页面折叠得更简单，也不能满足“以Transformers实现与模型HF config.json为依据”的新需求。模块名称或部分参数相同，不能作为直接迁移连接和缓存规则的依据。

“退出当前交付范围”表示不进入新HF默认构建与完成条件，不表示源码错误或应立即删除。历史代码和候选先保留，后续按Git提交逐步隔离，禁止本轮批量搬迁或清理。

## 2. 核查依据与方法

- 基线Git提交：8c5c5ca。本次开始时，根目录CONTEXT.md和2026_10_17_workspace_structure_review.md已经处于用户工作区删除状态；不恢复、不纳入本次提交。
- 实际读取30个产品Python文件的定义与内部依赖，重点检查输入注册、配置加载、取证、构建、图校验、渲染、导出及页面测试。
- 只读检查本地Transformers的DeepSeek modeling/configuration及必要缓存入口；未导入torch、Transformers或模型程序，未加载权重、下载输入或修改第三方仓库。
- 本地Transformers HEAD：536ecc007387a50e77603bb5d92100e9b07514cc。相关DeepSeek目录、cache_utils.py、masking_utils.py、modeling_rope_utils.py、activations.py的Git只读状态核查无改动；不代表已经核对整个仓库或线上最新版本。
- modeling_deepseek_v4.py SHA256：d6eb834f9d10eca879ca5390f219259eb452d19e1cf72d827f4334cbe4682266。
- configuration_deepseek_v4.py SHA256：49c818404caff1b4e76dff0ae7f468512932a5c40cc41b6cbd6585da4dec3aa2。
- 模型根config.json SHA256：6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023。
- 上述身份用于本次分析定位，不是新的正式语义冻结或HF验收声明。HF适配必须纳入实际影响所选路径的必要源码依赖，不能仅沿用旧两文件HF对照基线。

## 3. 可以直接保留的核心

### graph_ir.py

保留NamedAxis、Tensor、Port、Node、Edge、AliasRelation、Guard、StateVersion/StateTransition和FoldGroup等表示能力。它们不需要因更换模型实现而整体重写。

新页面只投影用户需要的算子、参数、维度和缓存关系，不必把IR的全部状态细节直接铺到界面。复用能力不等于要求新适配器填满所有可选字段。跨阶段同名张量的语义用阶段上下文区别；展示参数值和形状公式的联查信息可放在薄阅读文档层，不为界面需要反向重写IR。

### graph_document.py

保留SemanticGraph及graph_to_json/graph_from_json的规范化序列化、稳定ID、排序和往返校验。它解决图数据保存，不负责解释HF模型事实。

### checks/graph_contracts.py

保留唯一ID、轴定义、端口引用、连接方向、边张量一致性、状态引用等通用检查。例如152行附近核对张量内嵌轴与named_axes完整定义。模型语义正确仍须新HF检查，不将通用检查通过等同于源码一致。

### scenario_evaluator.py

保留受限表达式、整数/布尔区分与三态求值。用来验证维度公式和真实条件；不再绑定固定12个长度场景作为产品交付前置。训练或设备路径不因求值器支持条件表达式就自动进入覆盖范围。

### 包入口薄封装

各__init__.py与scripts/read_model_structure.py的薄入口组织可保留，陈旧的S0说明随新入口落地更新；它们不是新增架构层的理由。

## 4. 适配之后可用的部分

### model_registry.py与configs/models/

可保留注册查找、路径解析、错误分类。现schema要求constructor中具备参考ModelArgs/入口信息（model_registry.py:102起）；新增HF注册不能伪填reference字段。

建议保留原注册表示历史参考实现，新增独立HF注册键deepseek_v4_flash_0731_hf，并显式声明implementation_kind/所选eager路径。只为两种已存在来源提供小范围分支，不搭建任意模型插件框架。新旧候选身份与构建依赖分开。

### inventory.py与旧S0输入机制

可复用文件读取、SHA、路径规范化、输入清单和失配诊断。constructor_state目前是ModelArgs默认值加inference/config.json，entrypoint_overrides来自generate.py；不能用作HF有效配置。

新输入使用根config.json、configuration_deepseek_v4.py及必要HF依赖；根据实际HF配置定义处理默认值、attribute_map、兼容字段和派生值，逐项记来源。不运行模型构造程序来探值，也不把原有8角色测试作为新HF注册必须满足的要求。

### rules/deepseek_v4_flash_0731.py

AxisDef及配置来源的组织经验有用，load_model_values本身不能直接复用。该函数201行起实际读取inference/config.json和ModelArgs，并要求DSpark字段。

新增HF规则模块，保留B/T/P等调用符号；H、head_dim、q_lora_rank、专家参数等来自HF有效配置。先核对配置类的legacy转换，不把raw JSON字段缺失直接解释为模型没有该结构。

### extract/evidence.py

SourceEvidence、EvidenceRegistry、片段/行号验证可复用。SOURCE_FILES、_BASELINE_ROLES与load_evidence_base固定参考源码角色，必须改为按新注册提供来源清单，或新增HF专用的薄适配入口。

保留精确源码依据，但用户默认看到的是结构、维度和简明解释；证据在需要时联查。不要为了固定行号复用旧model.py证据。

### layer_plan.py

保留“模板加实例归属”的思路，重新按HF有效layer_types和mlp_layer_types构建分组。现文件直接依赖旧提取器与DSpark检查（187、274行附近），不适合原样接入HF。

新规划只表达真实结构变体、层索引范围、共享模板和参数引用；不复制43张同构层图，也不强制保留40/41/42→DSpark读出。

### build_pipeline.py与cli.py

保留内容寻址、原子提交、既有结果一致性检查、冲突拒绝与_failed诊断。当前run_build（326行起）直接调用旧build_reference_graph；cli._build_documents（645行起）再次完成旧图、四变体和DSpark强制装配。因此“复用落盘机制”不等于“直接调用旧构建”。

把新的HF读取→配置解释→提取→检查→阅读数据→HTML/Excel生成安排到构建层。CLI保留参数解析与结果导航，不再为HF路径强制生成旧DSpark结果。新profile的身份覆盖真正影响HF产物的源码、模板、导出依赖和分析设定；旧DSpark源码变化不应无条件使HF候选变更。

### render/structure_page.py

可复用离线payload安全嵌入、源码详情、图交互与投影守恒的经验。现接口与前端深度绑定plan/scenarios/variants/dspark以及图、场景切换，不符合两阶段连续展示。

建议为HF样板新增范围受控的decoder页面入口，复用确实独立的小函数和样式；不把旧2556行模板整体搬入新文件。页面先完整Prefill Decoder，再完整Decode Decoder。每个阶段提供自己的总览、mHC、Attention/KVCache和MoE；跨变体同样连续分节阅读，不以标签切换藏掉内容。HTML仍是单文件离线交付，开发时可以分开HTML/JS/CSS再内嵌。

### export/semantic_tables.py

68行CSV投影方式可作为图表同源的范例，已有connections/tensor_axes数据可用于核对；目前没有逐算子Excel生成器，不能以CSV改扩展名交付。

新增.xlsx导出：只有Prefill和Decode两张表，每张都含参数符号区和算子区；算子行引用本表参数，不设重复次数列。专家只列模板内部算子，参数保留真实HF字段映射。HTML和Excel从同一个阅读数据对象派生；xlsx元数据与ZIP时间需稳定或有明确一致性策略，避免同输入重跑因时间戳产生冲突。

本次仅发现工程解释器可定位openpyxl，未实际创建或验证工作簿。xlsx现为必交，依赖缺失不能静默降级为“HTML加CSV已完成”。

### 旧主图、变体提取与模型语义检查

extract/deepseek_v4_reference.py、extract/deepseek_v4_variants.py和checks/deepseek_v4_contracts.py中的结构主题、错误分类及检查经验可供重新阅读HF源码时参考；实际节点、端口、边、参数名、轴、缓存槽位规则、证据和golden不能直接搬用。

HF需要新的extract/deepseek_v4_hf.py、rules/deepseek_v4_hf.py、checks/deepseek_v4_hf_contracts.py（建议路径，尚未实现）。只编写这个已选模型的显式适配，不先开发通用Python解释器或模型反编译器。

## 5. 退出本轮交付主线、只读保留的内容

### 旧DSpark专用链

七个提取文件：deepseek_v4_dspark.py、dspark_stage.py、dspark_endpoints.py、dspark_exit.py、dspark_markov.py、dspark_confidence.py、dspark_integration.py，共10217行；另有checks/dspark_contracts.py共2939行。

这13156行主要服务旧参考实现草稿stage、入口/出口、Markov、Confidence及装配检查，不是当前HF decoder样板的必要功能。本次读取的HF模型类列表与选定forward并没有这条参考DSpark链，不能因为配置存在nextn等字段就补画对应结构。

不继续把B4补修、旧完整V2门禁或DSpark候选冻结作为新样板的前置。保留历史成果；只有将来所选HF源码存在并实际使用相关分支、且符合用户范围时才重新排入。

### 长度场景矩阵与版本演示

旧12场景导航、按T3/4/5/127/128/129和多组P构建的大量展示及_decode_version_chain不再是新产品必交。必须保留真实缓存的顺序、长度公式、窗口/压缩条件；只选区分实际分支的少量用例，不机械复制旧全矩阵。

### 历史页面与门禁测试

test_page_topology_r1/r2a/r2b、test_page_js_execution中的mtime选择已有outputs、固定Node路径及.work探针依赖，不进入新HF交付测试套件。test_b4_fix01_page已改为当前源码临时构建，不能误列为仍依赖mtime；其测试方法有用，但断言对象仍是旧参考页面。

test_b4_fix01~05_defense、test_dspark_*及对应mutant fixtures保留旧profile回归；不要求为新HF页面改标题、改字段来维持旧页面布局断言。旧测试数不是新完成度目标。新的测试与探针必须随源码保存，在临时输出根运行，不读取历史coordination作为前提。

### 旧计划附加目标

六类CSV齐套、四个独立CLI子命令、Qwen作为首版强制前置、生成采样链和逐包重复交接，不继续约束新交付。Skill仍为最终入口，第二模型在DeepSeek新阅读闭环明确后按用户需要扩展。

过程证据、旧候选和冻结文件均保留；本次不做GC、压缩归档或删除。

## 6. HF实现必须重新核对的事实

### 配置和变体

configuration_deepseek_v4.py:236起把legacy compress_ratios转换成layer_types并截取前num_hidden_layers项；把num_hash_layers转换成mlp_layer_types；通过attribute_map把num_local_experts映射到n_routed_experts、intermediate_size映射到moe_intermediate_size。

本地根配置包含46个compress_ratios，但HF主干只消费43层。按该配置及配置类规则静态推导，实际四组为：

- sliding_attention + hash_moe：层0、1。
- compressed_sparse_attention + hash_moe：层2。
- heavily_compressed_attention + moe：层3、5、…、41，共20层。
- compressed_sparse_attention + moe：层4、6、…、42，共20层。

这是源码与JSON的静态推导，未实例化HF配置类。首个样板建议选真实layer_idx=4（CSA + TopK MoE + 两处mHC），其余三组进入随后覆盖；不能拼造源码未出现的SWA+TopK组合。

### Attention、MoE与mHC

- modeling_deepseek_v4.py:714 eager_attention_forward实际包含KV广播、QK矩阵乘、scale、mask加法、sink拼接、减max、Softmax、去sink、AV矩阵乘等。旧opaque sparse_attn节点不能代替这条已显式可读路径。
- :752 Attention包含低秩Q、共享KV、部分RoPE、缓存更新、压缩KV拼接、block_bias扩展与逆RoPE/分组输出投影。张量布局按实际view/transpose跟踪，不仅照抄注释。
- :1025 Experts使用合并gate_up权重及down权重；clamp、SiLU、mul、分派/加权index_add都需准确表达。一份专家模板足够，num_experts与num_experts_per_tok不混用，实际命中专家/每专家token数保持数据相关。
- :873 HyperConnection与:1137 Decoder展示两次mHC前后计算；comb转置后作用于残差的方向必须按HF实际代码核对，不能仅因shape相同判正确。

### KVCache依赖不止modeling.py

DeepseekV4HCACache/CSACache定义在modeling_deepseek_v4.py:156/247；DynamicCache和DynamicSlidingWindowLayer来自cache_utils.py，缓存类型通过映射与子类注册选择（cache_utils.py:36、1764、1808附近）。

HF压缩缓存使用buffer_kv/buffer_gate、compressed_kv、entry_count及CSA overlap等状态，不是直接照旧参考实现固定槽位图。DeepseekV4HCACache.update返回本次可读full，同时持久保存的是sliding_window-1尾部；“本次参与Attention的KV长度”和“调用后保存长度”要分开。masking_utils.py与modeling_rope_utils.py是因果/位置解释的必要来源，需要按实际调用纳入来源闭合。

根配置虽声明FP8 quantization_config，当前选择的是HF逻辑eager路径。应记录该配置及当前分析边界，不把旧参考实现量化kernel节点或设备后端自动画入逻辑算子图；若加载侧改写会影响所声明结构，明确未覆盖，而不宣称已分析量化执行路径。

## 7. 本次验证与局限

实际使用工程Python 3.12运行：

1. `PYTHONDONTWRITEBYTECODE=1 <工程Python> -m unittest discover -s structure_reader/tests -p 'test_graph*.py' -q`：716项，1.360秒，原始退出码0。
2. 同样运行`test_scenario_evaluator.py`：45项，0.001秒，原始退出码0。

共761项，只证明本次抽取出的通用基础仍可复用。不重复全量1538项、不重新构建旧参考候选、不冒称HF语义、Excel或浏览器已通过。未修改产品代码，因此本轮没有对尚不存在的新HF产品运行验收。

## 8. 接下来如何使用本结论

先让新HF路径真正脱离reference ModelArgs与DSpark强依赖，保存一份真实decoder样板阅读数据，然后尽快生成HTML和两表Excel交给用户阅读。不要先花多个工作包重命名所有旧模块、归档所有历史记录或建设通用适配框架。

保留、适配与退出范围的决定均服务于这个交付闭环；详细里程碑、边界、验收与分工以实施计划v5为准。
