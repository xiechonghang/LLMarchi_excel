# T06 结果：GLM-5.3-Flash 语言主干的分层图与同源工作簿

状态：**第三轮复核的跨模型显示污染 P2 已修复，显示候选已发布并通过针对性验证，等待 Codex 最终复核**。
本票只做 T06（GitHub #7）：按官方 `zai-org/GLM-5.3-Flash` 与本地官方 transformers
`glm5_next` 实现，交付 text_config 语言主干的整网→层组→Block→算子分层 HTML、
两页三组同源工作簿、真实语义检查与证据。未 commit/push、未关闭 #7、未进入 T07、
未修改协调/自动化文件；未运行/实例化模型、未下载权重。

## 第一轮复核修复（4 项 P2）

Codex 2026-10-29 第一轮复核暂不验收；旧候选 `84229272…` 原样保留。本轮按
`T06-review/round-1-findings.md` 的源码证据原位修复：

| 编号 | 复核问题 | 源码证据 | 本轮修复 |
| --- | --- | --- | --- |
| P2-1 | mHC 系数写 20 | `modeling:276-281`：`(2+hc)*hc=24`，`split([4,4,16])` | 新增参数 `Mhc=(2+Hc)*Hc`（Hc=4→24）；fn/base/mixes 形状改为 `[Mhc,Hc×H]`/`[Mhc]`/`[B,seqlen,Mhc]`；图/表/讲解/契约/负例同步；Excel 活算 24 |
| P2-2 | `Ksel` 被写成全长统一值 | `get_visible_tokens:929-933`、`append_visible_tail:1043-1057` 按 query 可见长度 | 新增 `j`、`V_q=kvcachelen+j+1`、逐 query `K_sel=min(V_q,Ki+V_q%Kpool)`，另设 `K_sel_last=min(L_idx,...)` 仅表示最后 query；不再称统一值/一般最大值；Excel 填 j 可算 |
| P2-3 | DSA APE 与 valid_candidates→selected_valid→invalid 依赖缺失 | pool `:960-1008` 用 `index_kpool_compress_ape`；mask `:868-880`；selected_valid `:890-892`；query filter `:912-913` | pool 增加 `g_dsa_ix_ape` 权重且不再直接依赖 visible；新增 `pool_end`、`valid_candidates`、`selected_valid`、`query_mask` 具名张量/算子；mask 只消费 valid_candidates；invalid 消费 selected_valid；保留 query padding 末端过滤 |
| P2-4 | KDA 递推误差用未衰减状态且方向错 | `modeling:506-510`：`A=S⊙exp(g)`，`mem=A^T k`，`delta=beta(v-mem)`，`S=A+k delta^T` | 修正 note/讲解/数据/快照；补非零初态+非零 g 的纯 Python 代数测试（明确区分未衰减/错误方向）；recurrent 条件改为“已有状态且 seq_len==1”，P0/S1 走 chunk |

修复后候选：`3495213f6b949ee402772d7cad5e42ed3c0c673bbc71f33dcc525df5a6243851`。
旧候选 `84229272…`、首个正式候选 `171fd61f…` 与接手前候选均保留；本轮未删除旧候选。

Standards 的 2 项非阻塞 P3（`build_hf_decoder.py` 三个闭包与 GLM 适配重复、
`glm53flash_workbook.py` 的 `_parameter_rows/_network_section` 无调用 wrapper）
本轮未扩张重构，留待 T09/收尾；本轮只处理 4 项 P2 及其同源回归。

## 第二轮复核修复（query 输入有效域，1 项 P2）

Codex 第二轮确认 mHC 24、DSA 真实依赖、KDA 正确递推与首轮 4 项核心修复通过，
但发现新增 `j` 只有 `ISNUMBER` 守卫：填负数/小数/越界，或 seqlen 空白/非法/0 时，
`Vq/Ksel` 仍会出确定数值。本轮只补这一有限边界：

- 在 Block 参数上给 `j` 声明整数域 `{kind: integer, min: 0, max_symbol: seqlen}`；
  `export/operator_workbook.py` 的通用参数解析新增可选 `domain` 守卫，生成
  `IFERROR(AND(ISNUMBER(j),INT(j)=j,j>=0,j<seqlen,seqlen输入守卫),FALSE)`。
  该守卫只作用于声明了 domain 的参数行，其它模型/参数仍走原 `ISNUMBER` 路径。
- `Vq` 公式引用 `j` 与 `kvcachelen`，守卫自动包含 `j<seqlen` 与 seqlen 的
  `ISNUMBER/INT/>=0`；`Ksel` 引用 `V_q`，`V_q` 非数值时 `ISNUMBER(V_q)` 为假，
  因此 `Ksel` 同样未知。没有引入任意 eval、Excel 数据验证或只改文案。
- `Ksel_last` 仍只由 `L_idx/Ki/Kpool` 决定，与 `j` 无关；j 非法但 S/P 有效时
  `Ksel_last` 保持最后 query 的 2048。
- 现成工作簿同文件编辑重算：S=4/P=4092/B=2 时 j=0/1/2/3 → V_q=4093/4094/4095/4096、
  Ksel=2049/2050/2051/2048；j=-1/0.5/4/abc/空白 → V_q/Ksel 未知、Ksel_last 仍 2048；
  S=空白/abc/0 且 j=0 → V_q/Ksel 未知；恢复 S=4/j=3 → Ksel=2048；无 Excel 错误值。
- 静态场景仍不绑定 j（Ksel/V_q 未知），没有为图示引入 token 枚举；`Ksel_last`
  独立保留。

第二轮返修候选：`9879345bd3894f7faed6c1ad4d2fcb054132e5d455493f858896203d40acd71d`。
首轮候选 `3495213f…`、第一轮候选 `84229272…`、首个正式候选 `171fd61f…` 与接手前
候选均保留；仅清理本轮中间候选 `9d9167ff…`（domain 校验补齐前）并记录。

## 第三轮复核修复（跨模型显示污染，1 项 P2）

Codex 第三轮确认 query 有效域守卫、434 项测试、17 组现成 Excel native 编辑重算
均通过；独立重建 DeepSeek 时发现共享显示 token 泄漏：

- `render/display_names.py` 的全局 `j -> query_index` 把 DeepSeek mHC 中真实流索引
  `sum_j comb[j,k]·residual[j,d]` 改成 `query_index`，影响 4 个 DeepSeek 层组，
  改变了残差混合语义。本轮删除该全局改名；Flash 仍显示符号 `j`，参数行的
  “query 在本次 seqlen 中的局部索引”文字解释保留。没有新建渲染器、没有把其它
  模型局部变量当参数扩展。
- `render/diagram_source_labels.py` 的 `pad -> Pad` 会让 DeepSeek HCA 的
  `concat + pad` 变成 `concat + Pad`；本轮移除该新增映射，恢复原 fallback 大小写，
  避免旧 HTML 的非必要变化。
- 新增同进程回归：Flash → DeepSeek → GLM → Flash，断言 DeepSeek mHC 表达式仍为
  `sum_j comb[j,k]·residual[j,d]`、不含 `query_index`；GLM 不含 `query_index`；
  Flash 第二次构建与第一次逐字节相同。DeepSeek/GLM 的结构 JSON/工作簿与既有
  候选逐字节相同。
- 新候选工作簿 SHA256 仍为
  `f60e631ec3326480996f156b0dd065101e1810ddbf1f0e2b3d735d1789ed5c8f`，与
  `9879345b…` 完全相同；**复用**该候选/Codex 已完成的 17 组现成 Excel native
  编辑重算证据，不重复昂贵验证。新候选只做 manifest/data/HTML/工作簿结构、
  xlsx SHA 一致性与点击导航的针对性检查。

第三轮显示候选：`29f82a3e5481c971cdf36bb61f17ac60ef20036d86bdce33b49343bb09f41003`。
`9879345b…`、`3495213f…`、`84229272…`、`171fd61f…` 及接手前候选均保留。

## 接手确认

- 工作区/cwd：`/Users/xiechonghang/Workspace/LLMarchi_excel`。
- HEAD：`947ae78f120279403ec0a4c539e99d3d8869e1d6`（main，与派单一致）；本票未产生 commit。
- 接手快照：`/var/folders/cf/v8s2g_0n0_5c9x9brbcy1nw00000gn/T/llmarchi-T06-baseline-20261029-qq_zsdee/manifest.json`
  （450 项，2026-10-29T00:24:27+08:00 采集；同目录 `files/` 为内容副本）。改动/新增均对照该快照。
- 快照后由协调流程写入的 `docs/handoffs/…`、任务目录 README/execution/monitor/tickets/
  T06 派单文件只读，未编辑；`changed-files.json` 将其列为 external coordination。
- 快照前已存在的 HEAD 脏改动/删除（`D CONTEXT.md`、`D 2026_10_17_workspace_structure_review.md` 等）
  与全部接手前候选未动；未 reset/clean、未批量格式化。

## 官方来源与配置身份（先锁定，再开发）

| 项 | 值 | 证据 |
| --- | --- | --- |
| 模型仓库 | `zai-org/GLM-5.3-Flash` | HF API `/api/models/zai-org/GLM-5.3-Flash` |
| 仓库 revision | `46a050bec13002ad29ee795c16e61cf4fa3c4a9c`（2026-10-15） | 同上；`rules/glm53flash_hf.py:MODEL_REPOSITORY` |
| 根配置 | `models/zai-org/GLM-5.3-Flash/config.json`，69,416 B，SHA256 `bb8f01c4…23e9f` | 与官方 revision 的 raw config 逐字节相同（`T06-verify/official-config-verification.json`） |
| 模型卡 | `models/zai-org/GLM-5.3-Flash/README.md`，8,027 B，SHA256 `e6c5d414…87d02` | 与官方 raw README 逐字节相同 |
| model_type / architectures | `glm5_next` / `["Glm5NextForConditionalGeneration"]`；`text_config.model_type=glm5_next_text` | 根配置；`load_hf_model_values` 双条件校验 |
| 实现 | 本地官方 transformers 检出 `hflib/transformers` HEAD `536ecc007387a50e77603bb5d92100e9b07514cc`（5.19.0.dev0），模块 `src/transformers/models/glm5_next/` | 模型卡指向该模块；`@auto_docstring(checkpoint="zai-org/GLM-5.3-Flash")` |
| checkpoint 保存版本 | config `transformers_version="5.16.0"` | 记录在 `data/effective_config.json` |
| 已分析输入 SHA | 7 项（config + modeling/configuration + cache_utils/masking_utils/activations/configuration_utils） | `reports/source_manifest.json`，构建时逐项复算；漂移即具名拒绝 |
| 权重 | 未下载/未读取任何权重分片 | 只取上述两个小文件 |

已分析输入指纹（SHA256）：

```
config.json                 bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f
modeling_glm5_next.py       caa38805e6bb7f8772b7016f98897ca0408090fd00c314ea6c8e63c543b586cf
configuration_glm5_next.py  0a4df8f0238fd6e903fc94e7d89c0feee0bbea39bd8c92ccb42506e1b135d644
cache_utils.py              2bef552bc39e8a4dc7ddfcf25740e6a4a7c4b836d92f5e56629a8ebfcd9cb246
masking_utils.py            c1d0e1e61754db819382bb03acdd2ab38c1bed3e9e9cf010918d2bf2771a8415
activations.py              5b20c0a3625edc0001a98f09ce3c6b5baa1100e1d7ad8dee649e4d45c8468665
configuration_utils.py      6cde90e0992d0dcf0216ab773254c5dd243e04f3c896d6e3d88a2b8be0d46567
```

**身份辨伪**：Flash 是 `glm5_next`（45 层混合 KDA/MLA-DSA、4 路 mHC、NoPE、
k-pool indexer、288 专家），不是 GLM-5.3 的 `glm_moe_dsa`（78 层全 MLA/DSA、
普通残差、部分 RoPE），也不是 GLM-5 同族基座。本票没有把 GLM-5.3/DeepSeek 的
注意力、缓存、残差或层组结论套到 Flash 上；tests 对其它模型机制有拒绝断言。

**版本差异（明确，不混用）**：根配置记录 5.16.0，实现取本地 5.19.0.dev0。两版
`glm5_next` 有实质差异；本票以**本地检出实际代码**为唯一实现版本：

| 事实 | 5.16（checkpoint 保存版本） | 本地 5.19（本票实现口径） |
| --- | --- | --- |
| MLA/DSA 缓存 | `update(key_states, value_states)` 写展开后的 K/V | `update(k_pass, k_rot)` 写**压缩 latent**（keys 512、values 0 维 k_rot），用时 `kv_b_proj` 展开全部 Lkv（`f0d7783377`） |
| 层型旧名 | 配置使用 `deepseek_sparse_attention` | `PreTrainedConfig.__post_init__` 的 `remap_legacy_layer_types` 规范为 `indexed_attention`（`e2d83fdf9c`） |
| 专家 one-hot | `num_classes=E` | `num_classes=E+1`（多一个永远跳过的 no-expert 槽） |
| chunk KDA decay | 严格上三角未屏蔽（后续修复 NaN） | `masked_fill(strict_upper, -inf)`（`0b179b2df5`） |

本票只表达 5.19 的压缩 latent/NoPE/mHC 语义，不把 5.16 的展开 K/V 缓存结论混入；
若复核要求改用 5.16，需重新锁定该 tag 并重做缓存与注意力语义。

**配置声明但本地实现不读取的字段**（结构事实以源码为准，不按字段名推断）：
`mhc`、`mla_use_nope`、`indexer_rope_interleave`、`index_share_for_mtp_iteration`、
`topk_method`、`index_kpool_compress`。例如本地实现无条件构造 `attn_hc/ffn_hc`、
无条件做 k-pool 压缩、DSA 无 RoPE；配置字段只作为来源记录。

## 关键静态解释（每条有源码行号证据，见 `rules/glm53flash_hf.py`）

- 45 层（0..44）：34 层 `linear_attention`（KDA：0,1,2,4,5,6,…,44）+ 11 层
  `indexed_attention`（MLA/DSA：3,7,11,…,43）；`mlp_layer_types` 前 3 层 dense，
  其余 42 层 sparse。三组真实组合：`kda-dense`（0,1,2）、`dsa-moe`（11 层）、
  `kda-moe`（31 层）。
- **mHC**：`hc_mult=4`、`hc_sinkhorn_iters=20`、`hc_eps=1e-6`。网络入口把 embedding
  广播成 4 路相同流（modeling:1493）；每层 `attn_hc`/`ffn_hc` 两处：无权重
  RMSNorm → 线性投影出 `(2+Hc)×Hc=24` 个系数（pre 4 + post 4 + comb 16）→
  pre/post 门控、comb softmax+20 轮 Sinkhorn → `(pre*streams).sum(2)` 坍缩；
  fn/base/mixes 的系数维都是 `Mhc=(2+Hc)*Hc`，不是 20；子层输出按 post 展开、
  combᵀ 混合 residual 流后相加
  （:1335-1337/:1344-1346）。末端 `hc_head` 是 4 路**均值**（:332-336/:1509），
  不是加权 HyperHead，也不是普通单流残差。
- **KDA**：q/k/v 投影（64×128=8192）→ depthwise 因果短卷积（kernel=4；conv state
  `[B,24576,4]` 固定）→ forget gate `g=-5·sigmoid(exp(A_log)·g_raw)`、beta →
  kernel 内 L2Norm/缩放 128^-0.5 → delta-rule 扫描（状态 `[B,64,128,128]` 固定；
  `A=S_{t-1}⊙exp(g)`、`mem=A^T k`、`delta=beta(v-mem)`、`S=A+k delta^T`、
  `o=S^T q`；recurrent 仅在 cache 已有状态且 `seq_len==1` 时走，空缓存首次调用
  （含 P=0/S=1）与多 token 续接走 chunk_size=64 分块）→ gated RMSNorm → o_proj。
  KDA 状态不随 kvcachelen 增长。
- **MLA/DSA**：q 低秩 1536 + 分层 RMSNorm（eps=1e-5）；NoPE（`qk_rope_head_dim=0`，
  `head_dim=0`，`qk_head_dim=256`），K 只有 256 维 nope 段，无 RoPE 调用；KV 缓存
  压缩 latent `[B,1,L,512]`，values 槽为 0 维 k_rot；`kv_b_proj` 在缓存之后展开
  全部 Lkv 为 `[B,Nh,L,256+256]`，拆 k_nope/value；key 仍保留 cat(nope, 0 维 rope)
  的源码语义。
- **DSA indexer**：wq_b(q_residual)、wk(hidden)、带 weight+bias 的 LayerNorm(128,
  eps=1e-6)、gate_scores(128) 与 valid(1) 打包成 `[B,L,257]` packed 缓存；
  k-pool=4 把连续 4 个 token 用 softmax(gate+`index_kpool_compress_ape`) 压成一个
  候选池（pool 只依赖 packed valid，不直接依赖逐 query visible）；每 query 选
  `SelP=min(index_topk//kpool, floor(Lidx/kpool))` 个池（≤512），由 `pool_indices`
  求 `pool_end`，gather visible 并与 `pool_valid` 相与得 `valid_candidates`；
  mask 只消费 `valid_candidates`。top-k 后用 `selected_valid=valid_candidates.gather(
  selected)` 判断有效性，展开成原始 token；当前不完整池的 tail（≤3）直接追加，
  pad 到 `Kout=Ki+Kpool-1=2051`，最后按 query `attention_mask` 末端过滤（padding
  query 整行 -1）。
- **稀疏 mask**：`selected_counts.scatter_add_(-1, safe, valid)` → `counts.ne(0)
  .unsqueeze(1)` → `[B,1,seqlen,Lkv]`，**True=可见**（不是 True=屏蔽）；eager 路径
  再把 True→0、False→min。网络入口只有 `create_recurrent_attention_mask` 的 2D
  padding mask；DSA 的因果性由 indexer 的 `kv_pos ≤ q_pos` + 池/tail 选择保证，
  KDA 的因果性由短卷积 + 递推顺序保证。
- **NoPE 的 position_ids**：网络仍按 `arange(seqlen)+past_seen` 计算并转发
  position_ids，但源码不构造 rotary_emb；该张量不参与 Attention 数学。本票在整网
  总览保留这一步并如实注明，不把它当成位置编码事实。
- **MoE**：sigmoid 打分 + `e_score_correction_bias` + 288 选 8 + 归一(+1e-20) +
  ×2.5；专家 `one_hot(top_k,E+1)`，保留完整分配矩阵与 `(top_k_pos, token_idx)`
  两索引；专家/共享专家/稠密 MLP 的 SwiGLU 都先 `gate.clamp(max=10)`、
  `up.clamp(±10)` 再 `silu(gate)*up`；共享专家宽度 2048，与 routed 并行相加。
- **MTP/视觉边界**：`num_nextn_predict_layers=1`；本地实现
  `_keys_to_ignore_on_load_unexpected=[r"layers\\.45\\.", r"layers\\.\\d+\\.shared_head\\."]`
  （:1378）忽略 checkpoint 的 layer 45 与 shared_head，forward 只循环 0..44；
  vision 24 层与图像/视频 placeholder 替换不在本主干。

## 场景（同一参数化结构）

结构 shape 保留符号 `seqlen`/`kvcachelen`；页面/工作簿只绑定取值。DSA 组：

| 场景 | seqlen | kvcachelen | Lkv=Lidx | Npool | SelP | Ksel(j) | Ksel_last | Kout |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prefill-empty | 8 | 0 | 8 | 2 | 2 | j=0..7 → 1..8 | 8 | 2051 |
| decode-nonzero | 1 | 4096 | 4097 | 1024 | 512 | j=0 → 2049 | 2049 | 2051 |
| decode-long | 1 | 131072 | 131073 | 32768 | 512 | j=0 → 2049 | 2049 | 2051 |
| decode-multi | 4 | 4092 | 4096 | 1024 | 512 | 2049/2050/2051/2048 | 2048 | 2051 |
| prefill-symbolic | 符号 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 2051 |

**逐 query**：无 padding 时第 j 个 query 的可见长度 `V_q=kvcachelen+j+1`，
`K_sel(j)=min(V_q, Ki + (V_q % Kpool))`；它不是全长统一值，也不是一般最大值/
统一上界（S=4/P=4092 时依次为 2049/2050/2051/2048）。只有 j=seqlen-1 才等于
`K_sel_last=min(L_idx, Ki+(L_idx % Kpool))`。j 未给定、或 padding 使可见长度变化时
如实未知，不枚举具体 token、不默认 0。KDA 组不引用 kvcachelen：
`S_rec=NhL×DhL×DhL`=1,048,576、`S_conv=3×NhL×DhL×Kconv`=98,304，均固定；
`T_tok=B×seqlen`。

## 交付入口与真实产物

```bash
PY=/Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
cd /Users/xiechonghang/Workspace/LLMarchi_excel
PYTHONPATH=structure_reader/src "$PY" structure_reader/scripts/read_model_structure.py \
  build-hf --model glm_5_3_flash_hf --all-variants --json
```

- **第三轮显示最终候选**：`29f82a3e5481c971cdf36bb61f17ac60ef20036d86bdce33b49343bb09f41003`
- **真实 HTML**：[viewer/structure.html](../../../structure_reader/outputs/glm_5_3_flash_hf/29f82a3e5481c971cdf36bb61f17ac60ef20036d86bdce33b49343bb09f41003/viewer/structure.html)
  （1,872,018 B；SHA256 `713a9403…f8fcd0a`）
- **真实 Excel**：[tables/decoder_variants.xlsx](../../../structure_reader/outputs/glm_5_3_flash_hf/29f82a3e5481c971cdf36bb61f17ac60ef20036d86bdce33b49343bb09f41003/tables/decoder_variants.xlsx)
  （194,249 B；SHA256 `f60e631e…89ed5c8f`，与第二轮候选逐字节相同）
- 同候选：`data/layer_structures.json`（572,480 B；SHA256 `79610b97…e155535d`，与第二轮候选相同）、
  `data/network_overview.json`（44,051 B；`d2cb687c…94640f7`）、
  `data/effective_config.json`（114,909 B；`f585adc1…be3c5e`）、
  `reports/source_manifest.json`（10,782 B）、`run_manifest.json`。
- **保留的首个正式候选**：`171fd61f8080e7bce60178fdfa99a818156a3f085da25dcae0f203b6320dbc74/`。
- **保留的第二轮返修候选**：`9879345bd3894f7faed6c1ad4d2fcb054132e5d455493f858896203d40acd71d/`
  （query 有效域修复对象，工作簿与新候选逐字节相同；按要求不覆盖、不删除）。
- **保留的首轮返修候选**：`3495213f6b949ee402772d7cad5e42ed3c0c673bbc71f33dcc525df5a6243851/`
  （首轮 4 项 P2 修复对象，含 query 有效域缺口；按要求不覆盖、不删除）。
- **保留的第一轮候选**：`84229272e64c5a797195f3c7b8dd83487ef338e810b6db6bd652a14c0d4c936f/`
  （第一轮复核对象，含 4 项 P2 错误；按要求不覆盖、不删除）。
- 更早中间候选 `4d8c5d7a…`/`c8f35d9a…`/`423b82e0…` 已在上一轮精确清理；
  接手前 DeepSeek `c3fd4296…`、GLM-5.3 `abefb101…` 等候选未动。
- 重复执行同一命令：analysis_id 不变、`reused: true`，候选目录不增。

## 已实现

### 1. 分层数据（同一份参数化结构）

- `extract/glm53flash_hf.py`：三种真实层型组合各一份 Block 结构，shape 保留符号
  `seqlen`/`kvcachelen`；算子/张量数 52/90（kda-dense）、105/154（dsa-moe）、
  80/125（kda-moe）。
  - KDA：mHC attn/ffn 两处、input/post RMSNorm、q/k/v 投影、qkv 拼/转置、conv
    state 写、因果短卷积、qkv view/split、forget gate（f_a/f_b/dt/A_log/下界）、
    beta、L2Norm/缩放、delta-rule scan、recurrent state 写、gated RMSNorm、
    o_proj；dense MLP 或 TopK MoE + 共享专家；mHC expand/comb/add。
  - DSA：mHC 两处、input/post RMSNorm、q 低秩（q_a/q_a_norm/q_b/view，**无
    split/rope**）、kv_a 压缩投影/拆分/归一/单头 view、压缩 latent 缓存、kv_b 展开、
    kv view/split、k_rot 0 维广播、key cat、indexer 投影/packed 缓存/可见性/
    k-pool（含 APE）/pool_end/valid_candidates/池打分/池 top-k/selected_valid/
    展开/tail/pad/query padding 过滤、counts 稀疏 mask、eager QK/mask/softmax/
    AV/transpose/reshape/o_proj、TopK MoE + 共享专家、mHC expand/comb/add。
- `extract/glm53flash_network.py`：token→logits 整网（embedding、默认 position_ids、
  2D padding mask、4 路 mHC 展开、45 层真实顺序与三组归属、hc_head 均值、
  Final RMSNorm、logits 切片、lm_head）；KDA/DSA 两类缓存读前/写后；MTP/视觉边界显式。

### 2. 分层浏览页（复用同一渲染核心）

- `viewer/structure.html`：10 个层级 section（整网 + 3 层组 + 3 Block + 3 算子层）、
  50 个 SVG、49 段模块讲解、136 个层格；默认只显示当前层级，面包屑/链接可逐级进入
  与返回；场景按钮切换 5 个预设；源码快照、参数/张量/图/算子联查沿用 T03/T04/T05
  的锚点体系。三组分别显示真实层索引与次数（3/11/31，共 45）。
- 模型 profile（`render/hf_model_profile.py`）把层型短名、层组说明、层型选择表、
  模块分组、总览布局、模块讲解与源码命名表从通用页面装配中分开；DeepSeek/GLM-5.3
  默认 profile 保持既有输出。Flash 的图例只含自身机制（KDA conv/recurrent 状态、
  MLA 压缩 latent、DSA packed 缓存、mHC 4 路流、NoPE），没有 GLM-5.3 的
  shared indexer/部分 RoPE 或 DeepSeek 的 CSA/HCA/滑窗/Hash 图例。

### 3. 同源两页工作簿

- `tables/decoder_variants.xlsx`：`Prefill`/`Decode` 两页，每页 3 组、**237 个算子行**
  （52/105/80），来自与 HTML 同一份 `hf_layer_structure.v1`。
- 每页顶部可填写性能配置（seqlen 8/1、kvcachelen 0/4096、batch 1、精度、位宽、
  专家命中数未知）；派生参数 `Mhc/Lkv/Lidx/Npool/SelP/j/Vq/Ksel/Ksel_last/Kout/Ttot`
  与 KDA 的 `Srec/Sconv` 是引用本页输入/本组参数单元格的活公式（带 `IF`/`ISNUMBER`/
  `MOD`/`IFERROR` 守卫），并有实际输入/输出/权重 shape 公式列。`j` 是 query 索引
  占位（未给定保持未知）；其 domain 守卫为整数 `0<=j<seqlen`，且 seqlen 必须是
  正整数输入；只有 `V_q` 得到数值后 `Ksel` 才计算。填入合法 j 后 `V_q/Ksel` 随
  query 活算，`Ksel_last` 始终是最后 query 的值、不依赖 j。
- 数据相关量 `N_hit`/`N_e` 写“未知（原因）”，不写 null/0；成本列统一“待提供”，
  无猜测耗时、无成本 SUM；层组循环次数来自真实层索引，不复制 45 层/288 专家。
- 整网页尾列出网络外围 9 个算子、三组真实层索引/次数（3/11/31，共 45）、MTP 与
  视觉边界。

## 复用 / 重构 / 删除 / 新增

### 保留复用

- `build_hf_decoder.py` 的资料核验/身份/内容寻址发布/失败隔离；`cli._materialize`。
- `extract/hf_layer_structures.py` 的层组容器；`extract/hf_scenario.py` 的场景投影
  与三态求值；`scenario_evaluator.py` 白名单与 Excel 编译。
- `render/diagram.py` 的真实张量引用建图/SVG/锚点；`render/detail_render.py` 的算子卡/
  张量/参数联查；`render/hf_structure_page.py` 的层级/路由/场景条。
- `export/operator_workbook.py` 唯一 Workbook 创建/sheet/保存/稳定 ZIP；
  `export/hf_workbook_common.py` 的输入/参数/组/整网声明；T04 的活公式、实际 shape、
  未知/恢复与显式引用消歧。
- `checks/hf_contract_core.py` 的通用骨架/数据流/输入身份检查。

### 原位重构（迁移调用后删重复）

| 文件 | 改动 |
| --- | --- |
| `build_hf_decoder.py` | `_Adapter` 注册表新增 `glm_5_3_flash_hf`：共同编排、来源/工具 SHA、检查、渲染、导出仍只一处；Flash 单层/ONNX 显式拒绝。 |
| `render/hf_model_profile.py` | 新增 `GLM53FLASH_PAGE_PROFILE` 与 Flash 模块规则/布局/frames/图例/源码表；DeepSeek/GLM-5.3 profile 不改。 |
| `render/display_names.py` | 新增 Flash 参数 id→配置字段显示名（Hc/NhL/DhL/Kconv/Kpool/Ipack/Npool/SelP/Kout/Srec/Sconv/Swi/GateLB/Sit）；只增键，不改旧映射。 |
| `render/diagram_source_labels.py` | 只新增 Flash 计算类型 token（conv1d/scan/l2norm/rmsnorm-gated/sigmoid+scale/pool+softmax/compare+and 等）。首版曾重复定义 `mask_build` 覆盖 GLM-5.3 既有 “Compare + MaskedFill”，已改为独立 `compare+and` 并回归确认 GLM-5.3 字节只差候选 id。 |
| `cli.py` | `--model`/`--layer-index`/`--all-variants` 帮助列出 Flash 与真实层组，不再写死“四组”。 |
| `structure_reader/README.md`、`docs/hf_reading_variants_contract_v1.md` | 增加 Flash 入口、身份/边界与 T06 增量说明。 |

### 新增

| 文件 | 职责 |
| --- | --- |
| `rules/glm53flash_hf.py` | 官方身份/revision、7 项源码指纹、text/vision 边界、配置静态解释（层型重映射、NoPE、mHC、KDA、k-pool、MoE）、3 组归属、参数来源记录、场景。 |
| `extract/glm53flash_hf.py` | 三种层型组合的参数化 Block 结构（mHC/KDA/MLA+DSA/MoE/缓存/布局）。 |
| `extract/glm53flash_network.py` | 45 层 token→logits 整网总览与层序/组归属。 |
| `checks/glm53flash_hf_contracts.py` | Flash 语义检查：身份/层组/KDA/DSA/mHC/MoE/场景/输入身份；拒绝 GLM-5.3/DeepSeek 机制与 RoPE。 |
| `export/glm53flash_workbook.py` | Flash 薄适配：两页三组、派生映射、真实层归属校验、KDA/DSA 分组派生、成本待提供。 |
| `render/glm53flash_narrative.py` | Flash 模块/总览中文讲解（按 KDA/DSA、dense/MoE 分支）。 |
| `render/glm53flash_source_labels.py` | Flash 算子源码表达式/张量短名/行号。 |
| `tests/test_glm53flash_structure.py` | 62 项回归：身份/层组/KDA/DSA/mHC/MoE/场景/契约拒绝/产物/工作簿/profile/锚点；含首轮 4 项 P2、query 有效域、Flash→DeepSeek→GLM→Flash 显示不污染与 KDA 非零初态/非零 g 代数检查。 |
| `T06-verify/verify_glm53flash.py`、`final-independent-artifacts.json` | 落盘候选 + LibreOffice 重算 + 编辑链 + 锚点闭合独立验证。 |
| `T06-verify/click_navigation_check.py`、`navigation-click-check.json` | headless Chrome 真实 DOM 点击：DSA/KDA/KDA-MoE 组、模块、张量、参数、返回。 |
| `T06-verify/official-config-verification.json` | 官方 config/README 与固定 revision 逐字节核对。 |
| `T06-verify/changed-files.json`、`incremental.patch` | 相对接手快照的改动/新增证据（含外部协调文件与接手前改动分类）。 |
| `T06-verify/{network,dsa-block,kda-block,dsa-indexer,dsa-pool,kda-scan,moe-expert}.png` | 真实候选/模块图（headless Chrome 渲染）。 |

### 删除 / 退役

- 本票未删除产品文件；仅删除本票精确中间候选目录（见上）。DeepSeek 单层/四组
  ONNX、GLM-5.3 分层页与全部接手前候选保留。

## 实际验证证据

### 相关测试（本票环境）

| 测试 | 数量 | 结果 |
| --- | --- | --- |
| `tests/test_glm53flash_structure.py`（本票新增） | 47 | OK |
| `tests/test_glm53_structure.py` | 30 | OK |
| `tests/test_hf_decoder_reading.py` | 61 | OK |
| `tests/test_hf_layer_levels.py` | 41 | OK |
| `tests/test_hf_network_overview.py` | 21 | OK |
| `tests/test_operator_workbook_v2.py` | 24 | OK |
| `tests/hf/test_operator_workbook.py`（v1 兼容） | 57 | OK |
| `tests/test_hf_build_missing_xlsx.py` | 2 | OK |
| `tests/test_scenario_evaluator.py` | 45 | OK |
| `tests/test_m2_guard.py` | 2 | OK |
| `tests/test_page_topology_r2b.py` / `r2a` / `r1` | 30 / 27 / 17 | OK |
| `tests/test_page_js_execution.py` | 2 | OK |
| `tests/test_model_registry.py` | 15 | OK |

合计 **436 项通过**（62 项本票新增 + 374 项既有直接相关）。未跑的 `test_hf_decoder_diagram`
在 T05 接手时已有 6 项旧布局断言失败，本票未改该历史测试；ONNX 测试未计为通过
（本环境缺 onnx 依赖，且 Flash 显式拒绝 ONNX）。

### DeepSeek / GLM-5.3 不回退

第三轮修复全局 `j`/`pad` 显示泄漏后，用本票代码在内存中重新构建（不发布新候选），
与既有最终候选逐字节比较：

| 模型 | 产物 | 结果 |
| --- | --- | --- |
| DeepSeek（T04 `c3fd4296…`） | `data/layer_structures.json` / `data/network_overview.json` / `tables/decoder_variants.xlsx` | 完全相同 |
| DeepSeek | `viewer/structure.html` | 除 T05 已改的共享 CSS 中性注释与候选 id 外相同；README 仅候选 id 不同 |
| GLM-5.3（T05 `abefb101…`） | `data/layer_structures.json` / `data/network_overview.json` / `data/effective_config.json` / `tables/decoder_variants.xlsx` | 完全相同 |
| GLM-5.3 | `viewer/structure.html` | 仅候选 id 不同（183 个字符差异，来自多处候选 id 出现）；README 仅候选 id 不同 |

说明：本票新增共享类型 token 后，DeepSeek/GLM-5.3 的工具源码身份/analysis_id 会变化，
但结构数据与工作簿字节一致；原候选目录未重写、未删除。首版 `mask_build` 重复键导致
GLM-5.3 的 `mask_build` 图类型从 “Compare + MaskedFill” 变成 “Compare + And”，
已改为独立 `compare+and` 并回归确认；这是一个实际发现的跨模型回归，未留在最终代码。
第三轮又发现 `j -> query_index` 全局改名污染 DeepSeek 的 mHC 流索引，已删除该全局
token 并加入同进程顺序构建回归；现在 DeepSeek 恢复 `sum_j comb[j,k]·residual[j,d]`，
DeepSeek/GLM 仍只差共享 CSS 中性注释与候选 id。

### Flash 真实候选独立验证

**第三轮显示候选的针对性验证**：
[`round-3-display-check.json`](T06-verify/round-3-display-check.json)：`status=passed`，
对象是落盘候选 `29f82a3e…`；复算 7 个交付 SHA、7 项输入 SHA、26 项工具 SHA，
identity/analysis_id 一致；数据/HTML/工作簿结构检查通过；xlsx SHA256 =
`f60e631e…89ed5c8f`，与第二轮候选逐字节相同。**未重复** LibreOffice native
17 组重算，直接复用下条证据。

[`final-independent-artifacts.json`](T06-verify/final-independent-artifacts.json)：
`status=passed`，对象是第二轮返修落盘候选 `9879345b…`（不是内存 spec）；因其
工作簿与新候选 SHA256 完全相同，下列 native 重算/逐 query 编辑证据同样适用于
`29f82a3e…`：

- `run_manifest` 7 个交付 SHA（另 run_manifest 自身）、7 项输入 SHA、26 项工具 SHA 全部复算一致；identity
  规范化重算 SHA256 == analysis_id。
- 数据：三组 3/11/31 恰好覆盖 45 层；DSA 层 3,7,…,43；dense 0/1/2；Block 算子/
  张量 52/90、105/154、80/125；参数与 `config.json` 独立核对（H=4096、Nh=64、
  Dn/Dv/Dq=256、Rk=0、Rq=1536、Rkv=512、Ih=32/Id=128、Ipack=257、Ki=2048、
  Kpool=4、E=288/K=8、Es=2048、Is=12288、Scl=2.5、Swi=10、hc_mult=4、
  Mhc=24、Sinkhorn=20、eps 1e-5/1e-6、GateLB=-5）；KDA/DSA/mHC/MoE/缓存/掩码
  语义通过；未出现 RoPE/shared indexer/滑窗/CSA/HCA/Hash。
- 场景独立期望：Prefill(8,0) Lkv=8、Ksel_last=8、Ksel(j)=1..8；Decode(1,4096)
  Lkv=4097、Ksel_last=2049；长历史(131072) Lkv=131073、Ksel_last=2049；
  多 token(4,4092) Lkv=4096、Ksel(j)=2049/2050/2051/2048、Ksel_last=2048；
  符号场景 Lkv/Npool/Vq/Ksel/Ksel_last unknown。
- 契约负例：mHC=20、K_sel 回退为全长公式、pool 去 APE、删 valid_candidates、
  删 selected_valid、删 query_mask、KDA 旧错误公式均被具名拒绝；KDA 非零初态/
  非零 g 的纯 Python 代数检查区分“未衰减状态”和“方向错误”两种错误公式。
- HTML（第三轮候选）：1,872,018 B、1,560 个 id 无重复、3,830 个普通
  锚点全部有目标；三组层级链接、场景 id、源码快照齐备；页面散文无其它模型事实
  （源码快照中原样保留的官方注释不计为页面事实）。
- 工作簿结构：2 页 × 3 组 × **237 行**（52/105/80）；输入/派生公式/实际 shape/
  循环归属/成本待提供/数据相关未知；无成本 SUM；无 GLM-5.3 组 id。
- **LibreOffice native 重算**（同一现成文件，不重新导出）：
  - 原场景 DSA `Lkv/Lidx/Npool/SelP/Ksel_last/Kout`：Prefill `8/8/2/2/8/2051`，
    Decode `4097/4097/1024/512/2049/2051`；KDA `Mhc/Srec/Sconv` 恒为
    `24/1048576/98304`，`Ttok` Prefill 8、Decode 1；`Ksel`/`V_q` 未给 j 时保持未知。
  - 实际 shape：KDA q_proj 输出 `[1,8,8192]`、conv state `[1,24576,4]`、
    recurrent state `[1,64,128,128]`；DSA latent 缓存 `[1,1,8,512]`、
    packed 缓存 `[1,8,257]`、pool keys `[1,2,128]`、top-k `[1,8,2051]`、
    sparse mask `[1,1,8,8]`、expert mask `[289,8,8]`；Decode 分别
    `[1,1,4097,512]`、`[1,4097,257]`、`[1,1024,128]`、`[1,1,2051]`、
    `[1,1,1,4097]`、`[289,8,1]`。
  - 编辑链（同一文件，不重新导出）：seqlen=4/kvcachelen=4092/B=2 →
    `Lkv=4096, Npool=1024, SelP=512, Ksel_last=2048, Ttok=8`，KDA q_proj 输出
    `[2,4,8192]`、scan 状态 `[2,64,128,128]`、latent `[2,1,4096,512]`、mask
    `[2,1,4,4096]`、expert mask `[289,8,8]`；清空 seqlen → 未知（原因），不是 0；
    填 `abc` → 未知且无 `#VALUE`；恢复 8 → 恢复 `8/8/2/2/8`；清空 kvcachelen →
    `Lkv/Ksel_last` 未知、`Ttok` 仍 8；填 0 → `Lkv=8`；填 131072 且 seqlen=1 →
    `Lkv=131073, Ksel_last=2049`；全表扫描无 Excel 错误值。
  - **逐 query 编辑**：同一工作簿 S=4/P=4092/B=2 时 j=0/1/2/3 →
    `V_q=4093/4094/4095/4096`、`Ksel=2049/2050/2051/2048`；j=-1/0.5/4/abc/空白 →
    `V_q/Ksel` 未知、`Ksel_last` 仍 2048；S=空白/abc/0 且 j=0 → `V_q/Ksel` 未知；
    恢复 S=4/j=3 → `Ksel=2048`；无 Excel 错误值。证明 Ksel 逐 query 活算，
    且 query 有效域守卫真实生效、`Ksel_last` 独立。
- 重算后端为 LibreOffice native（提示缺 Calibri 字体，不影响数值）；**未做
  Microsoft Excel 原生 GUI 打开验证**。

### 页面渲染与真实点击证据（headless Chrome，非用户验收）

- [整网页](T06-verify/network.png)、[DSA Block](T06-verify/dsa-block.png)、
  [KDA-MoE Block](T06-verify/kda-block.png)、
  [DSA indexer 模块](T06-verify/dsa-indexer.png)、
  [DSA k-pool 模块](T06-verify/dsa-pool.png)、
  [KDA scan 模块](T06-verify/kda-scan.png)、
  [KDA MoE 专家模块](T06-verify/moe-expert.png)。
- 已用 URL hash 实际加载 `#/network`、`#/group/<gid>`、`#/group/<gid>/block`、
  `#mod-block-<gid>-<module>` 并截图；另用 headless Chrome 同源 iframe 做**真实
  DOM click**：整网 → dsa-moe 层组 → Block → 讲解“indexer 投影”→ 张量
  topk_indices → 返回 Block → 模块卡片 moe_expert → 参数 SelP → 返回整网；
  KDA 组 Block → 讲解 delta-rule 扫描 → 张量 recurrent_states → 参数 GateLB →
  讲解 mHC 模块 → 参数 Mhc；另点击 valid_candidates 张量与 Ksel_last 参数；
  KDA-MoE Block → 讲解专家模板 → 参数 E。20 步均记录 hash/可见层级，`ok=true`、
  无错误、无缺目标/重复 id（[navigation-click-check.json](T06-verify/navigation-click-check.json)）。
  宽 Block 图在页面内横向滚动（与 T05 同一 `.diagram-scroll` 交互），不是裁剪丢失。
- 未做人工鼠标/键盘/窄屏阅读，也未做真实用户阅读认可。

## 未验证 / 明确限制

- 用户性能公式与 XPU 能力仍未提供：无算子耗时/组小计/整网串行汇总；成本列“待提供”，
  不猜、不记 0、不做父子重复计数。
- GLM-5.3-Flash **未实现可选 ONNX**：显式 `--onnx` 具名拒绝（不虚构、不静默忽略）；
  DeepSeek ONNX 路径未改、未回退。
- GLM-5.3-Flash **只提供分层入口**（`--all-variants`）；单层长页未实现，缺该参数时具名拒绝。
- 只覆盖 text_config 语言主干：vision 24 层、图像/视频 placeholder 替换不在本票；
  MTP 层 45 不执行；FP8 量化执行/设备 kernel、torch.compile、generate 采样循环、
  loss/aux_loss 未分析；不做数值等价。
- 本配置 `indexer_types` 全为 full；shared indexer 分支（复用上一层 top-k）未激活、
  未分析；`index_kpool_always_select_tail=false` 的变体未分析（构建会具名拒绝）。
- 配置声明但本地实现不读取的字段已列在上文；结构事实以源码为准，不能按字段名推断。
- 未在原生 Microsoft Excel 中做 GUI 阅读验证；未做真实设备性能/Profiler/并行融合通信。
- 静态工程、LibreOffice 重算与 headless 渲染通过不代表用户阅读认可；需 Codex/用户复核。

## 保护与停止

- 未 reset/clean、未批量格式化、未恢复根两项删除、未改接手前未提交改动；相对 T06
  快照的 17 个产品改动/新增、14 项证据与 29 项外部协调文件（截至本轮采集，含 Codex T06-review 持续写入、
  docs/agents 平台文件与派单文件）见
  [changed-files.json](T06-verify/changed-files.json) 与
  [incremental.patch](T06-verify/incremental.patch)。
- 接手前 DeepSeek `c3fd4296…`、GLM-5.3 `abefb101…`、T04/T05 首轮候选及证据未动；
  本票首个正式候选 `171fd61f…`、第一轮候选 `84229272…`、首轮返修候选 `3495213f…`、
  第二轮返修候选 `9879345b…` 与第三轮显示候选 `29f82a3e…` 均保留；更早仅删除
  本票中间候选 `4d8c5d7a…`/`c8f35d9a…`/`423b82e0…`，第二轮仅清理中间候选
  `9d9167ff…`（domain 校验补齐前）；本轮未删除任何候选。
- 协调/自动化文件只读，未编辑；定时检查保持外部设置的 ACTIVE/45 分钟，本票未改。
- **完成即停止，等待 Codex 复核**：不 commit/push、不关闭 #7、不进入 T07。
