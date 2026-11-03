# M2 ONNX完整维度名称与实际尺寸复核

2026-10-26；Codex派单/验收，产品由原DeepSeek Harness App、DeepSeek-V4.1-Flash 1017 High第42/43轮完成。用户要求“缩写和维度数值都需要检查”。本轮静态尺寸与有限数值工程检查 passed，M2用户阅读仍 pending，M3未启；自动保持 PAUSED。

当前入口：[四组合阅读页](../../outputs/deepseek_v4_flash_0731_hf/9a3f171878fd10983dea558f07cde0571a85a4e63f373e928885fbcd3d31be05/viewer/decoder_variants.html)。请从本页对应模块链接打开新的简化ONNX；已打开的旧Netron文件不会自动更新。

## 修正与实际依据

上版612已将符号展开，但KV等已知权重仍是符号维度，固定轴字符串1/3也未写为数值。当前使用本phase真实参数，有限白名单整数算术具体化纯静态表达式；batch_size、sequence_length、历史长度、窗口/命中数等仍为完整符号，不猜默认值；未知保留unresolved_dimension。state输出形状同步。

- Q投影权重[1024,4096]、后投影[32768,1024]；KV权重[512,4096]，KV输出[batch_size,1,sequence_length,512]；position_ids为[1,sequence_length]，1是真实整数轴。
- 输出分组投影[8192,4096]，按8组解释为[8,1024,4096]；后投影[4096,8192]。mHC fn[24,16384]、base[24]、scale[3]；router[256,4096]、Hash表[129280,6]，专家/共享专家及norm尺寸逐实际资产核对。
- CSA投影[1024,4096]，位置偏置[4,1024]；HCA投影[512,4096]，偏置[128,512]；Indexer投影[256,4096]，偏置[4,256]，Q投影[8192,1024]、头权重投影[64,4096]。原报告把HCA尺寸写成CSA的段落不采信，实际两种文件正确。
- 截图512是head_dim，旋转末64维，对应32个频率，前448维保持。SWA主RoPE theta10000；CSA/HCA压缩RoPE theta160000、YaRN factor16的根配置常数已对源码公式独立复算。标准Mul的第二输入B是ONNX端口名，不代表batch维度。

唯一产品文件改动为onnx_module/build.py；本轮两测试文件受影响。已修复apply_display_dims返回整数却写dim_param的TypeError，实际protobuf断言静态H/固定1写dim_value，动态B写完整dim_param。HF/config输入与数学builders未改。

独立期望读取原始config与只读官方HF，revision536ecc007387a50e77603bb5d92100e9b07514cc；未导入执行HF。关键源码为modeling_deepseek_v4.py:303–329分组投影、332–356交错末维RoPE、359–440 HCA、459起Indexer、586–618 CSA；配置与modeling_rope_utils的原公式为频率依据。Codex检查器未调用产品静态维resolver作为期望。

## 验证与身份

证据目录：`.work/coordination/hf_m2_onnx_shapes01/`，原task/fix01、原报告/原始日志均保留。

- Codex独立检查276原/简ModelProto及464个权重/持久buffer实例，四组合两阶段具名shape符合根配置与源码。原始模型节点、属性、连接与initializer与612逐字节同；简化模型中28个因具体尺寸产生常量折叠变化，不移植旧简化文件结论。checker、shape inference、simplify状态均通过。
- Codex复跑本轮7项：静态维/protobuf字段3项及数值/表达式4项。实际生产Ctx/GB的KV partial RoPE与inverse RoPE+GroupedLinear，以手写NumPy为期望，原图和简化图均在1e-5容差内。Prefill使用batch2/token3、头/组非等轴；Decode使用batch2/token1、position7，KV hidden_size7/head_dim4，输出分组头3/head_dim4。Harness报告Decode分组head_dim6是笔误，以实际测试的4为准。不是实际模型权重或HF整模型等价验证。
- JSON、Excel字节及142个SVG与612同；561份交付hash、28工具源、7输入及内容身份当前复算匹配。两真实子进程all-variants --onnx构建rc0，同9a3f身份，第一次reused=false/第二次true。runner/argv/cwd/env/stdout/stderr/returncode均已读。
- 当前138个Netron链接由既有实际binary逐一resolve-only通过，另旧7b链接兼容；launcher SHA b7b10037e451e45f56aace3cfad3982d12589d11ffcde2dbb0d3e068efbf94e1未改。复用旧检查器只将结果输出重定向本轮目录，旧证据未写。
- 11975旧资产及加入中间候选c4e9后的12537保护文件全部无漂移。旧freeze不修改；本轮需求/计划/progress授权更新另记，新身份单独冻结。

正式结论：onnx_shape_static_engineering=passed，finite_synthetic_numeric=passed；browser_visual=blocked/not_verified、Netron协议GUI=not_verified、native_excel=not_verified、HF_numeric_equivalence=not_claimed、user_reading=pending。根配置静态shape已核，不把7项小例扩大为全模型/全部动态分支数值证明。未提交或推送Git，用户删除文件保持删除，原App正常完成空闲，不重复旧任务。
