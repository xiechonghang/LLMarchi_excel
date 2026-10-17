# HF decoder 样板独立期望与检查清单（layer_idx=4）

角色：opencode（测试/独立验证）。状态：**验证准备；产品 pending**。
本清单由只读 HF 输入独立推导，**不**读取任何生成图/产品常量，也**不**作为正式通过结论。
首个样板建议层为真实 `layer_idx=4`：`compressed_sparse_attention`（CSA）+ `moe`（TopK）+ 两处 mHC。

## 0. 输入身份（只读）

| 来源 | 路径 | SHA256 |
| --- | --- | --- |
| 模型根 config | `models/deepseek-ai/DeepSeek-V4-Flash-0731/config.json` | `6c8f3d2d…9a023` |
| HF modeling | `hflib/transformers/.../models/deepseek_v4/modeling_deepseek_v4.py` | `d6eb834f…682266` |
| HF configuration | `…/models/deepseek_v4/configuration_deepseek_v4.py` | `49c81840…43aa2` |
| cache_utils | `.../transformers/cache_utils.py` | `2bef552b…cb246` |
| masking_utils | `.../transformers/masking_utils.py` | `c1d0e1e6…1a8415` |
| modeling_rope_utils | `.../transformers/modeling_rope_utils.py` | `6a072707…abfac24` |
| activations | `.../transformers/activations.py` | `5b20c0a3…8468665` |

Transformers HEAD `536ecc007387a50e77603bb5d92100e9b07514cc`；模型包 revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`。
模型包 `inference/model.py` 为参考推理实现身份，**不作为 HF 事实来源**。
`DeepSeek-V4.1-Flash` 是 `model_type=deepseek_v41` 的另一模型，不用于本样板。

## 1. 有效配置（configuration_deepseek_v4.py）

| 检查项 | 独立期望 | 源码位置 |
| --- | --- | --- |
| legacy `compress_ratios`(46) → `layer_types`，截取前 43 | 见下 | configuration:236,254-255,262 |
| `num_hash_layers`(3) → `mlp_layer_types` | 0-2 `hash_moe`，其余 `moe` | configuration:264-268 |
| `attribute_map` | `num_local_experts`→`n_routed_experts`；`intermediate_size`→`moe_intermediate_size` | configuration:104-107 |
| `partial_rotary_factor`(legacy `qk_rope_head_dim`/`head_dim`) | 64/512 = 0.125；`qk_rope_head_dim`=64 | configuration:270-278 |
| rope 两标签 | `main`(`default`,θ=10000,partial)；`compress`(`yarn`,θ=160000,factor=16,attention_factor=1.0) | configuration:288-307 |
| `compress_rates` | CSA=4，HCA=128 | configuration:145,244-250 |

layer_types（由压缩比 0/4/128 推导）：层0、1=sliding；层2=CSA；层3=HCA；此后偶数层=CSA、奇数层=HCA。
四组归属：sliding+hash(0,1)2 层；CSA+hash(2)1 层；HCA+moe(3,5…41)20 层；CSA+moe(4,6…42)20 层；合计 43。
**layer 4 = compressed_sparse_attention + moe（TopK），rope 标签=compress。**
负例：若忽略 legacy `compress_ratios` 走 V4-Pro 默认分支，层4 会变成 HCA。

关键值：hidden 4096；heads 64；kv_heads 1（groups=64）；head_dim 512；q_lora 1024；o_groups 8；o_lora 1024；
moe_inter 2048；routed 256；top-k 6；shared 1；scoring `sqrtsoftplus`；routed_scaling 1.5；swiglu_limit 10.0；
sliding_window 128；hc_mult 4；hc_sinkhorn_iters 20；hc_eps 1e-6；index_n_heads 64；index_head_dim 128；index_topk 512。

## 2. mHC（modeling:873-1003,1137-1185）

- 流张量 `[B,S,N=4,D=4096]`；`fn`[(2+N)·N=24, N·D=16384]、`base`[24]、`scale`[3]。
- 权重：`pre=sigmoid(pre_w·pre_scale+pre_b)+hc_eps`；`post=2·sigmoid(…)`；`comb=softmax(comb_w·comb_scale+comb_b,dim=-1)+hc_eps`；`pre` 就地折叠出 `collapsed[B,S,D]`，返回 `post,comb,collapsed`。
- comb 经 Sinkhorn-Knopp 双随机化：先按列归一，再 `hc_sinkhorn_iters-1` 次行/列交替。
- **连接方向**：解码层用 `post.unsqueeze(-1)·block_out.unsqueeze(-2) + comb.T @ hidden_streams`，即 `sum_j comb[j,k]·residual[j,d]`；comb 非对称，方向不可省。两处 mHC：`attn_hc`（进 attention 前）与 `ffn_hc`（进 MoE 前）。
- 负例：2×2 之外的非对称双随机 comb 下 `comb.T@H ≠ comb@H`（测试用 3×3 夹具验证）。

## 3. Attention / KVCache（modeling:156-329,586-870; cache_utils:113-322,1764-1805; masking_utils:88-95,1468-1473; rope_utils:345+）

- 低秩 Q：`q_a_proj[1024,4096]→q_a_norm[1024]→q_b_proj[32768,1024]→view→transpose→q_b_norm→RoPE`。
- 共享 KV-MQA：`kv_proj[512,4096]`（单头），`num_key_value_groups=64`，`repeat_kv` 后与 Q 广播；K=V 同一张量，无 `v_proj`。
- 部分/交错 RoPE：仅作用尾 64 通道（`[nope448|rope64]`），`repeat_interleave(2)` + `rotate_half`(0::2/1::2)；输出再以 `-sin` 施加逆旋转。
- 分组输出投影：`o_a_proj`（8 组，组内 4096→1024，`[8192,4096]`）→ `flatten` → `o_b_proj[4096,8192]`。
- eager 路径：`QK→scale→+mask→拼 sink→减 max→softmax→去 sink→AV`；HF 明确禁用 flash/sdpa/flex，`_can_compile_fullgraph=False`。
- CSA cache：`DeepseekV4CSACache(DynamicSlidingWindowLayer)`，`compress_rate=4`；`buffer_kv/gate`、`compressed_kv`、`entry_count` 均按 `{"compressor","indexer"}` 分键；`overlap_kv/gate` 仅持久 `chunk[:,-1,:,:512]`（Ca）。
- **读视图 vs 持久存储**：`update` 返回 `full=cat([keys,new])` 供本次 Attention；持久 `keys=full[..., -127:]`（`sliding_window-1`），`values=keys`。decode 但 `seq_len==1` 时压缩器返回 `(compressed_kv, None)`，不产生 block_bias。
- 压缩：CSA 双序列 `[B,n_win,2m,512]`（Cb 当前窗 + 上一窗 Ca），HCA 单序列；窗口按 `compress_rate` 对齐，余量留 buffer。
- Indexer：`∑_h w·ReLU(q·K)`, top-k=min(512,compressed_len)；越界补偿 `-1` 哨兵；CSA `block_bias` 用 `scatter_` 置 0，多余列 `compressed_len` 切片去掉；HCA `block_bias` 用 `entry_idx >= (pos+1)//m` 掩 -inf。
- mask：三类 layer_type 映射同一 `create_sliding_window_causal_mask`，overlay 规则 `kv_idx > q_idx - sliding_window`；mask 尺寸取自 cache `get_mask_sizes`；压缩 KV 拼接后 mask 与 block_bias 拼接，**不得 0 填充**。
- 负例：持久 KV 长度是 `sliding_window-1=127` 而非 128；RoPE 作用于尾部而非头部；KV 投影输出 `head_dim` 而非 `2·head_dim`（限定在 Attention 类内核对）。

## 4. TopK MoE（modeling:1006-1134; activations:217-221）

- 层4 走 `DeepseekV4TopKRouter`（`mlp_layer_types[4]=="moe"`）。
- 路由：`logits=F.linear(flat,weight[256,4096])`→`scores=sqrt(softplus(logits))`→`topk(scores+bias,6,sorted=False)`→`weights=gather(scores)`→`weights/(sum+1e-20)`→`×routed_scaling_factor(1.5)`。bias 仅进入选择，不进入权重。
- 专家权重合并：`gate_up_proj[256,4096,4096]`、`down_proj[256,4096,2048]`；无独立 gate/up 参数。
- 计算：`chunk(2)→gate.clamp(max=10.0)、up.clamp(±10.0)→silu(gate)·up`→`down`→`×top_k_weights`→`index_add_` 加权汇合。
- 动态 token 维：`flat=hidden.view(-1,4096)`、`indices[T,6]`；`B,S` 保留符号，不填运行时命中专家/每专家 token 数。
- 共享专家：单个 `DeepseekV4MLP`，宽度经 attribute_map 取 `moe_intermediate_size=2048`；`routed + shared_experts(residual)`，residual 为进入 MoE 的（已 norm）张量。
- 负例：层4 非 hash_moe；`n_routed_experts(256) ≠ num_experts_per_tok(6)`；gate 单边 clamp、up 双边 clamp；缩放须在归一之后。

## 5. 两阶段与终点

- prefill/decode 各一套完整 decoder（mHC→norm→Attention/KV→mHC 残差；mHC→norm→MoE→mHC 残差）。
- 分析到 `lm_head` 产生 logits 为止；不含采样/生成循环/训练。
- 现场可读：prefill 的压缩窗口是否满、decode 的 `seq_len==1` 分支；未知运行时内容不填虚构值。

## 6. 测试与负例覆盖

- `test_hf_source_identity.py`：输入存在/哈希身份、models 与 hflib 分离、参考实现不当来源。
- `test_hf_effective_config.py`：43 层截取、层4 归属、四组计数、rope/alias/部分旋转等；含“忽略 legacy 会改层4”等负例。
- `test_hf_mhc.py`：shape、公式、Sinkhorn 次序、`comb.T` 方向 + 非对称夹具。
- `test_hf_csa_attention.py`：低秩 Q、单 KV 广播、部分/交错 RoPE、eager 算子序、分组投影 + 头/尾、多头负例。
- `test_hf_csa_cache.py`：缓存类/注册、full vs 127 持久、buffer/overlap/entry_count、decode 无 bias、indexer 哨兵、block_bias 规则、mask 来源。
- `test_hf_topk_moe.py`：路由序、bias 仅选择、合并 gate_up、clamp/silu/mul、加权 index_add、共享专家、动态 token 维 + 负例。
- `test_hf_decoder_layer.py`：两处 mHC 装配、eager 标志、logits 终点、mask 拼接非 0 填充。

运行（工程 Python，PYTHONDONTWRITEBYTECODE=1）：
`<eng-python> structure_reader/tests/hf_independent/run_checks.py`（98 项）。

## 7. 产品 pending（交付冻结后由正式验证任务执行）

- 真实 CLI build 的 `reading_document` 是否为 `layer_idx=4`、`attention_type`/`moe_type`/`analysis_path` 与上表一致。
- HTML 先完整 prefill 再完整 decode，无切页隐藏；黑盒中文“内部未分析”标注。
- xlsx 仅两表、参数区可联查、无重复次数列、与 HTML 同源同值。
- 上表各 shape/方向/缓存条件与产品读数逐项比对（含本目录负例注入）。
- 相同输入重复构建稳定、失败隔离、缺依赖诊断。

在正式验证任务下达前，本清单仅表示“期望与检查方法已就绪”，**不签通过、不重跑旧 B4、不重跑阅读门禁、不以生成输出当正确答案**。
