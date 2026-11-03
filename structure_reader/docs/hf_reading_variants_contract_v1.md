# HF配置实际变体容器v1（M2增量）

2026-10-24，由Codex协调M2公共边界。`hf_reading_contract_v1.md`原文及单层v1语义保持，已有默认layer4构建继续可用。本容器仅组织同一模型实际配置的单层模板，不复制第二套模型事实。

容器schema为`hf_decoder_variants.v1`，含原model身份对象与`variants`数组。每项含`id`、`label`、`layer_indices`（零起始真实归属层数组）、`representative_layer_index`、`document`（合法`hf_decoder_reading.v1`，含prefill/decode两阶段及真实该组合scope）。代表层必须属于该项层集合，各集合不重叠并恰好覆盖实际num_hidden_layers；组合以静态HF有效layer_types与mlp_layer_types共同确定。同组只权重位置不同的层共享模板，不能以层数冒充已分析算子数。

单层文档仍是参数/张量/算子唯一事实源。HTML按完整Prefill（各组）再完整Decode（各组）连续呈现，组名/归属/代表层明确，所有组间锚点唯一且参数/张量/算子/源码联查真实。公共模板可以引用复用，不拆新算子或制造重复事实。Excel仅Prefill/Decode两页，每页按组显示参数区/算子行；组内参数引用落本页对应组，不能错误跨组引用相同缩写。

T04增量（2026-10-28）：单层与分组 Excel 已收敛到 `export/operator_workbook.py:export_workbook` 一个实现；`export/hf_workbook.py` 只做模型适配（两页、四组、场景默认值），`render/m2_variants.py` 的重复 Workbook 流程删除。每页顶部增加可填写的性能配置（seqlen/kvcachelen/B/精度/位宽），派生参数写成引用本页输入或本组参数单元格的 Excel 公式；未知输入保持未知，不默认 0。每页四组仍是同一结构的场景投影，成本列在用户公式提供前统一写“待提供”。原单层 `export_operator_workbook(reading_document)` 签名与 v1 校验保持，作为薄兼容适配；分组不再调用私有函数自建工作簿。

原build-hf可加`--all-variants`选项，默认保持单层；无需新CLI子命令或新通用框架。M2内容身份覆盖所有真实影响源、输入、所选组与参数；一组缺事实/检查失败不得发布全覆盖成功。reading_document.json可保存此容器，必要的逐组v1/来源/ONNX清单作为明确关联交付；目录结构确定、内链正确。数学ONNX与单层图使用同一组文档，不从已有SVG猜结构。


T05增量（2026-10-28，GLM-5.3）：模型键 `glm_5_3_hf` 接入同一条 `build-hf` 流水线，但只提供分层入口（`--all-variants`）：`_run_variants` 按模型适配声明来源/输入身份/文档/检查/渲染 profile/工作簿适配，不新增第三套构建或导出流程。层组容器新增可选的 `indexer_type`（`full`/`shared`）：GLM-5.3 的三种真实组合为 full+dense、shared+sparse、full+sparse，恰好覆盖 78 层；DeepSeek 不写该键，输出字节不变。页面 profile（`render/hf_model_profile.py`）把层型短名、层组说明、层型选择表、模块分组、总览布局、模块讲解与源码命名表与通用装配分开；工作簿公共声明提炼到 `export/hf_workbook_common.py`，DeepSeek/GLM 各自只做薄适配，唯一 Workbook 创建/保存仍在 `export/operator_workbook.py`。GLM-5.3 未实现可选 ONNX 与单层长页：显式 `--onnx` 或缺少 `--all-variants` 时具名拒绝，不静默忽略、不虚构支持。来源漂移（config/源码 SHA）仍按指纹具名拒绝。

T06增量（2026-10-29，GLM-5.3-Flash）：模型键 `glm_5_3_flash_hf` 接入同一条 `build-hf` 流水线，同样只提供分层入口（`--all-variants`）。它是独立适配，**不预设与 GLM-5.3 相同**：`glm5_next` 是 45 层混合模型（34 层 KDA 线性注意力 + 11 层 MLA/DSA 稀疏注意力）、4 路 mHC 残差、NoPE（`qk_rope_head_dim=0`）、k-pool=4 压缩 DSA indexer、288 路由专家与 SwiGLU clamp。三组真实层型为 `kda-dense`(0,1,2)、`dsa-moe`(3,7,…,43)、`kda-moe`(其余 31 层)，恰好覆盖 45 层；容器仍用可选 `indexer_type`（KDA 为 `none`、DSA 为 `full`），不虚构 shared 模板。页面 profile/讲解/源码命名表/工作簿适配各自独立：KDA 的 delta-rule 扫描与固定 conv/recurrent 状态、MLA/DSA 的压缩 latent+packed 缓存与 k-pool 池/tail、mHC 的两处 post/comb 混合分别表达；没有网络级 4D causal mask/rope。工作簿两页三组，KDA 组只声明固定状态派生量（T_tok/S_rec/S_conv），DSA 组才声明 Lkv/Lidx/池/tail 派生量；成本仍“待提供”。GLM-5.3-Flash 未实现单层长页与可选 ONNX，显式 `--onnx` 或缺少 `--all-variants` 时具名拒绝。来源漂移（config/源码 SHA）仍按指纹具名拒绝。


T07增量（2026-10-29，Qwen3.8-Flash-Next）：模型键 `qwen_3_8_flash_next_hf` 接入同一条
`build-hf` 流水线，同样只提供分层入口（`--all-variants`）。它是独立适配，**不预设
Qwen3-Next/Qwen3.5/Qwen3-MoE 或 GLM/DeepSeek 的结构**：`qwen4_exp` 是 48 层
12×(3×Gated DeltaNet + 1×Qwen Sparse Attention)、4 路 Gated Residual（读门逐元素、
写门每支路标量）、N-gram PLE（layer 1，16 头×160 维）、512 选 10 softmax MoE +
共享专家 sigmoid 标量门。三组真实层型为 `gdn-moe`(35)、`gdn-ple-moe`(1, layer 1)、
`qsa-moe`(3,7,…,47, 12)，恰好覆盖 48 层；层组容器新增可选 `ple` 维度（DeepSeek/
GLM 不写该键，输出字节不变）。页面 profile/讲解/源码命名表/工作簿适配各自独立：
GDN 的短卷积+递推固定状态、QSA 的 partial RoPE+K/V 缓存+micro-block indexer+块
top-k/tail、Gated Residual 的读门/写门、PLE 的 n-gram/短卷积、softmax MoE 分别表达；
网络末端没有 Final RMSNorm。工作簿两页三组，GDN 组只声明固定状态派生量
（T_tok/S_rec/S_conv/S_ple_conv/L_ngram_ctx），QSA 组才声明 Lkv/Lidx/块预算/tail
派生量；成本仍“待提供”。Qwen3.8-Flash-Next 未实现单层长页与可选 ONNX，显式
`--onnx` 或缺少 `--all-variants` 时具名拒绝。来源漂移（config/源码 SHA）仍按指纹
具名拒绝。


T08增量（2026-10-29，HiDream-O1-Image）：模型键 `hidream_o1_image_hf` 接入同一条
`build-hf` 流水线，只提供分层入口（`--all-variants`），但**不投影 Prefill/Decode**：
它是单次生成网络，层文档 schema 为 `hidream_o1_layer_document.v1`（analysis=
single_forward、phases=["generation"]），不伪造两阶段。场景轴从 seqlen/kvcachelen
换成图像高/宽、prompt token 数、参考图 patch 数、条件图 token 数（`HIDREAM_BIND_KEYS`）；
`hf_scenario.evaluate_derivations/structure_document` 增加可选 `bind_keys` 与
`default_scenario`（既有文本模型不传时字节不变）。两个真实重复层组是**独立索引
空间**（`uit-decoder` 36 层 0..35；`vit-tower` 27 blocks 0..26，仅 K>0 执行），因此
`build_structures_container` 增加可选 `build_variant_fn(workspace, group, model_key)`
——不能只用 representative_layer_index 反查两个都从 0 开始的层组。页面 profile 只
替换展示文案（网络标题/说明、Block 标题、场景条/面板、header shape 说明），默认值
与既有页面逐字节一致。工作簿仍是唯一导出实现 `operator_workbook.export_workbook`，
HiDream 适配声明**单 sheet**（Generation）；整网摘要新增可选 `count_note`，避免把
36 层与 27 blocks 相加成 63 层（旧模型不传时字节不变）。结构事实：36 层 Qwen3-VL
风格 decoder（GQA 32/8、逐头 q/k RMSNorm、MRoPE 只作用 q/k、两趟注意力 + index_copy、
两个 RMSNorm 站点 + 标量残差、SwiGLU 12288、条件 DeepStack、无 KV cache/MoE/adaLN）；
27 个 ViT block（LayerNorm、合并 qkv 3456 + bias、2D rotary、非因果分图注意力、
GELU-tanh 4304、block 8/16/24 的 DeepStack tap）；终点 x_pred 不是 logits，无 lm_head/
VAE/独立文本编码器。显式 `--onnx` 或缺少 `--all-variants` 具名拒绝。
