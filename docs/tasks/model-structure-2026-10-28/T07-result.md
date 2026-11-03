# T07 结果：Qwen3.8-Flash-Next 语言主干的分层图与同源工作簿

状态：**已交付并通过本票独立验证，停止等待 Codex 复核**。
本票只做 T07（GitHub #8）：按官方 `Qwen/Qwen3.8-Flash-Next` 与本地官方 transformers
`qwen4_exp` 实现，交付 text_config 语言主干的整网→层组→Block→算子分层 HTML、
两页三组同源工作簿、真实语义检查与证据。未 commit/push、未关闭 #8、未进入 T08、
未修改协调/自动化文件；未运行/实例化模型、未下载任何权重分片。

## 第一轮复核修复（5 项 P2）

Codex 2026-10-29 第一轮复核暂不验收；旧正式候选 `c0d4ed84…` 原样保留。本轮按
`T07-review/round-1-findings.md` 的源码证据原位修复，随后冻结结构、只生成一个最终
候选并做一次受影响验证：

| 编号 | 复核问题 | 源码证据 | 本轮修复 |
| --- | --- | --- | --- |
| P2-1 | 多出源码不存在的逐 token indexer K Norm | `modeling:703-708` 缓存 raw_keys；`modeling:735` 只在 micro-block mean 后 `k_layernorm` | 删除 `g_qsa_ix_k_norm` 及孤立 `g_qsa_ix_token_k_normed`；保留 `k_layernorm` 只作用于 pooled key；图/表/契约/负例同步 |
| P2-2 | QSA 用全长 Lidx 的 Npool/Bsel 当逐 query pool/topk shape | `modeling:720-750` 每个 batch/query 在可见索引内算 Nq=floor(Vq/4)、Bq=min(512,Nq) | 新增逐 query `Nq/Bq/Rtail` 与 `g_qsa_ix_query_guard`；block tokens/pooled keys/norm/RoPE/score/topk 全部改 `Nq/Bq`；Nq=0 时 pool/norm/RoPE/score/topk 跳过、tail 与输出 mask 仍执行；保留 Ksel(j)/Ksel_last 区分，eager QK 仍为全长 `[S,Lkv]` |
| P2-3 | 4D 残差/PLE 与 3D flat 直接 Add/Mul | `modeling:1241/1294/1471` 层间 hidden 为 `[B,S,Hc*H]`；`:1015-1018` 读取时 unflatten | 统一层间 flat 边界；补 `read_sigmoid/read_view/norm_view`、`expand_flat`、PLE `query_view`/`gated_value_flat`/`value_mask_flatten`；4D 只在读门/写门内部出现，Add/Mul 两侧同型 |
| P2-4 | True=可见的 bool selected mask 直接加到 float causal mask | `modeling:766-769` 先 where(True,0,min)；`:848-851` 才相加 | 拆成 `g_qsa_ix_mask_bool → g_qsa_ix_mask_float → g_qsa_mask_combine`；明确 eager float 分支；bool/SDPA 才是相与 |
| P2-5 | 全局 `Dq→query_width` 覆盖 GLM `Dq→qk_head_dim` | `display_names.py` 同键后写覆盖先写 | Qwen 改用独立 `Dqw/D_qw`；删除全局覆盖；同时移除会污染 GLM 的 `Nmtp` 映射；用已有三模型 HTML 归一化候选 id 后逐字节比较 |

修复后候选：`c921712221160320176fe73003fc02b78c6c53a0d9c8bf895908b0250bef8454`。
旧正式候选 `c0d4ed8408b5feda4e545c56016d02d0ecb819b6e561e6081e8feb44c5ac87e7`
及全部接手前候选保留；本轮未删除旧候选，仅清理本票中间候选 `9dedbe81…`。

## 接手确认

- 工作区/cwd：`/Users/xiechonghang/Workspace/LLMarchi_excel`。
- HEAD：`947ae78f120279403ec0a4c539e99d3d8869e1d6`（main，与派单一致）；本票未产生 commit。
- 接手快照：`/var/folders/cf/v8s2g_0n0_5c9x9brbcy1nw00000gn/T/llmarchi-T07-baseline-20261029-d90y8w3e/manifest.json`
  （500 项，2026-10-29T10:51:49+08:00 采集；同目录 `files/` 为内容副本）。改动/新增均对照该快照。
- 快照后由协调流程写入的 `docs/handoffs/…`、任务目录 README/execution/monitor/tickets/
  T07 派单文件只读，未编辑；`changed-files.json` 将其列为 external coordination。
- 快照前已存在的 HEAD 脏改动/删除（`D CONTEXT.md`、`D 2026_10_17_workspace_structure_review.md` 等）
  与全部接手前候选未动；未 reset/clean、未批量格式化。

## 官方来源与配置身份（先锁定，再开发）

| 项 | 值 | 证据 |
| --- | --- | --- |
| 模型仓库 | `Qwen/Qwen3.8-Flash-Next` | HF API `/api/models/Qwen/Qwen3.8-Flash-Next` |
| 仓库 revision | `de4b8e4d43b917e7706784d8bb445c9af86a3540`（2026-08-27） | 同上；`rules/qwen38flash_hf.py:MODEL_REPOSITORY` |
| 根配置 | `models/Qwen/Qwen3.8-Flash-Next/config.json`，4,745 B，SHA256 `889658f2…d2e74b` | 与官方 revision 的 raw config 逐字节相同（`T07-verify/official-config-verification.json`） |
| 模型卡 | `models/Qwen/Qwen3.8-Flash-Next/README.md`，65,155 B，SHA256 `35ca37cc…b225b2` | 与官方 raw README 逐字节相同 |
| model_type / architectures | `qwen4_exp` / `["Qwen4ExpForConditionalGeneration"]`；`text_config.model_type=qwen4_exp_text` | 根配置；`load_hf_model_values` 双条件校验 |
| 实现 | 本地官方 transformers 检出 `hflib/transformers` HEAD `536ecc007387a50e77603bb5d92100e9b07514cc`（5.19.0.dev0），模块 `src/transformers/models/qwen4_exp/` | 模型卡指向 transformers；`@auto_docstring(checkpoint="Qwen/Qwen4-Exp")` |
| checkpoint 保存版本 | config `transformers_version="5.8.0.dev0"` | 记录在 `data/effective_config.json` |
| 已分析输入 SHA | 8 项（config/README + modeling/configuration + cache_utils/masking_utils/activations/configuration_utils） | `reports/source_manifest.json`，构建时逐项复算；漂移即具名拒绝 |
| 权重 | 未下载/未读取任何权重分片（仓库 131 个 shard 全部未取） | 只取 config 与 README 两个小文件 |

已分析输入指纹（SHA256）：

```
config.json                  889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b
README.md                    35ca37ccc366f1ba478dab33841a2c0c18ce53fd62f291ca05341f7728b225b2
modeling_qwen4_exp.py        f94401c10c1bbafd814cc174ba6f40dde1f40b35715ce1ae827ccf8ee3cd81fa
configuration_qwen4_exp.py   ac0950ab34516d26fe4f6b28cffe0649d763449effe44e42b5913463806f7e0b
cache_utils.py               2bef552bc39e8a4dc7ddfcf25740e6a4a7c4b836d92f5e56629a8ebfcd9cb246
masking_utils.py             c1d0e1e61754db819382bb03acdd2ab38c1bed3e9e9cf010918d2bf2771a8415
activations.py               5b20c0a3625edc0001a98f09ce3c6b5baa1100e1d7ad8dee649e4d45c8468665
configuration_utils.py       6cde90e0992d0dcf0216ab773254c5dd243e04f3c896d6e3d88a2b8be0d46567
```

**身份辨伪**：这是 **Qwen3.8-Flash-Next（qwen4_exp）**，不是 Qwen3-Next、Qwen3.5、
Qwen3-MoE 或其它名称相近的 Qwen。模型卡明确这是 Qwen4 架构的实验预览；本票没有把
Qwen3-Next 的 GQA/线性注意力/Qwen3.5 的视觉结构套到本模型。模型卡给出的规模
（125B total / 6B activated、51B n-gram embedding、4B MTP）只作为来源记录，未读取
权重数值。

**版本差异（明确，不混用）**：根配置记录 `transformers_version="5.8.0.dev0"`，但
`qwen4_exp` 在本地 `v5.8.0` tag 中**不存在**：该模块首次提交 `fc5c5bde8e`
（2026-08-26，版本 5.16.0.dev0），首次随 `v5.17.0` 发布。本票以**本地官方检出
5.19.0.dev0 的实际代码**为唯一实现版本；该模块在首次提交后仍有语义提交
（例如 `full_attention → indexed_attention` 的层型重映射、partial mask、MRoPE
调整）。没有用任何“名称相近的 Qwen”实现替代，也没有把 5.8 的旧结论混入。

## 关键静态解释（每条有源码行号证据，见 `rules/qwen38flash_hf.py`）

- **48 层**：12×(3×Gated DeltaNet → MoE → 1×Qwen Sparse Attention → MoE)；
  `layer_types` 显式每 4 层一个 `full_attention`（3,7,11,…,47，共 12），其余 36 层
  `linear_attention`。`Qwen4ExpTextConfig.__post_init__` 把 `full_attention` 规范为
  `indexed_attention`（configuration:179-183）。全部 48 层都是 `SparseMoeBlock`，
  没有 dense MLP 层。
- **Gated DeltaNet**（modeling:456-616）：`in_proj_qkv` 2560→10240；depthwise 因果
  短卷积（kernel=4，conv state `[B,10240,4]`）；split q/k `[B,S,16,128]`、v
  `[B,S,48,128]`；q/k `repeat_interleave(3)` 到 48 头；`beta=sigmoid(b)`；
  `g=-exp(A_log)·softplus(a+dt_bias)`；kernel 内 L2Norm(q/k) + q/√128；gated
  delta-rule（chunk 64 / recurrent 单 token；状态 `[B,48,128,128]` 固定）；
  Gated RMSNorm（输出门 activation=sigmoid）；`o_proj` 6144→2560。
  状态大小固定，不随 kvcachelen 增长。
- **Qwen Sparse Attention（QSA）**（modeling:810-892）：`q_proj` 输出
  2×24×256=12288，按头 chunk 成 q 与 gate（各 24×256）；q/k 各 RMSNorm(256)；
  partial RoPE 只作用于前 `256×0.25=64` 维（后 192 维 nope）；KV 2 头，GQA
  `repeat_kv` 12 倍到 24 头；`o_proj` 6144→2560；输出乘 `sigmoid(gate)`。
- **QSA indexer**（modeling:664-770）：`index_qk_proj` 2560→(4+1)×128=640；
  4 个 query 头 + 1 个共享 key 头；q/k 各 RMSNorm(128)；q 与 pooled block key
  只对前 64 维做 RoPE；逐 token raw key 写入 indexer 缓存 `[B,L,128]`；每
  `compress_ratio=4` 个连续**可见** token 取均值压成一个 micro-block；block key
  在组首位置做 RoPE；分数 = `relu(q·k).sum(-1)/√128`（4 个 indexer 头求和，
  **不是 softmax/加权**）；每 query 选 `min(512, 完整块数)` 个块 → 展开成 token →
  追加不完整 tail（≤3）→ pad 到 `2048+4-1=2051`（-1）→ scatter 成
  `True=可见` 的 additional mask。
- **注意力掩码**：网络入口同时构建 `create_causal_mask`（4D float additive，
  QSA 用）与 `create_recurrent_attention_mask`（2D padding，GDN 用）
  （modeling:1446-1460）；QSA 把 selected mask 与 causal mask 相加（float）
  或相与（bool）（modeling:848-851）。selected mask 只收窄可见范围，不删除 K/V。
- **Gated Residual / hyper-connections**（modeling:994-1022）：4 路流先做分组
  RMSNorm（group_size=2560）；`input_mix_weight_down` 10240→320、SiLU(·/4)、
  `input_mix_weight_up` 320→10240、sigmoid 得**逐元素读门** `[B,S,4,H]`；
  四路加权后取 mean 得单流；`block_inject_weight` 10240→4 得**每支路标量写门**
  `2×sigmoid(·/4)`；子层输出按写门展开后加到**原 4 路流**。网络末端
  `hyper_connection_mixer` 用同一读门结构加权 mean 成单流；**没有 Final RMSNorm**，
  直接进 lm_head。
- **N-gram / PLE**（modeling:1071-1246）：`ple_layer_ids=[2]`（one-indexed）→
  layer_idx=1；`ngram_size=3`、`heads_per_ngram=8` → 16 个哈希头（bigram 8 +
  trigram 8），每头 160 维 → `ple_embed_dim=2560`。token×每层乘子后 XOR，按每头
  素数码表大小取模再加头偏移；EOS 切断 n-gram 上下文；上一 token 上下文 `[B,2]`
  （state_idx=2）+ 4-tap dilation-3 depthwise 短卷积（state 9 步，state_idx=1）；
  `key_proj` 2560→10240、`value_proj` 2560→2560，query 是当前 4 路流；
  gate=(key_normed·query_normed).sum(-1)/√H、abs/clamp/sqrt/sign、sigmoid(gate)·value；
  输出加回 4 路流。
- **MoE**（modeling:951-991）：router softmax 512 选 10，`norm_topk_prob=True`
  把 top-10 归一化；**没有 correction bias、没有 routed scaling factor、SwiGLU
  没有 clamp**；专家 `gate_up_proj [512,2×640,2560]`、`down_proj [512,2560,640]`；
  共享专家 2560→640→2560，另有 `shared_expert_gate` 2560→1 sigmoid 标量门；
  汇合为 routed + 门控共享专家。
- **缓存**：`linear_attention` → `LinearAttentionLayer`（conv/recurrent 状态，固定）；
  `indexed_attention` → `DynamicIndexedLayer`（K/V `[B,2,L,256]` + indexer keys
  `[B,L,128]`）。`number_of_conv_states=3`（PLE 开启），但只有 layer 1 的 PLE
  使用 state_idx=1/2。
- **MTP/视觉边界**：配置声明 `mtp_num_hidden_layers=1`、`mtp.layer_types=
  ["full_attention"]`，但本地 HF 实现没有 MTP 模块，
  `_keys_to_ignore_on_load_unexpected=[r"^mtp.*"]`（modeling:1313/1589），
  forward 只循环 0..47；MTP 不在本主干。视觉 27 层、图像/视频 placeholder 替换、
  视觉网格 M-RoPE 不在本票范围。

## 场景（同一参数化结构）

结构 shape 保留符号 `seqlen`/`kvcachelen`；页面/工作簿只绑定取值。QSA 组：

| 场景 | seqlen | kvcachelen | Lkv=Lidx | Npool | Bsel | Ksel(j) | Ksel_last | Kout |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prefill-empty | 8 | 0 | 8 | 2 | 2 | j=0..7 → 1..8 | 8 | 2051 |
| decode-nonzero | 1 | 4096 | 4097 | 1024 | 512 | j=0 → 2049 | 2049 | 2051 |
| decode-long | 1 | 131072 | 131073 | 32768 | 512 | j=0 → 2049 | 2049 | 2051 |
| decode-multi | 4 | 4092 | 4096 | 1024 | 512 | 2049/2050/2051/2048 | 2048 | 2051 |
| prefill-symbolic | 符号 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 2051 |

表中 `Npool/Bsel` 是**最后一个 query**（j=seqlen-1）的完整块/选中块数，只用于
`Ksel_last` 与场景摘要；逐 query 的实际 pool/norm/RoPE/score/topk shape 用
`N_q=floor(V_q/Kc)`、`B_q=min(512,N_q)`。例如 S8/P0 的 j=0..7 为
`N_q=0,0,0,1,1,1,1,2`、`B_q=0,0,0,1,1,1,1,2`；S4/P4092 的
`N_q=1023,1023,1023,1024`、`B_q=512`。

**逐 query**：无 padding 时第 j 个 query 的可见长度 `V_q=kvcachelen+j+1`，
`K_sel(j)=min(V_q, Ki + (V_q % Kc))`；它不是全长统一值，也不是一般最大值/统一上界
（S=4/P=4092 时依次为 2049/2050/2051/2048）。只有 j=seqlen-1 才等于
`K_sel_last=min(L_idx, Ki+(L_idx % Kc))`。j 未给定、或 padding 使可见长度变化时
如实未知，不枚举具体 token、不默认 0。

GDN 组不引用 kvcachelen：`S_rec=NvH×Dk×DvH=48×128×128=786,432`、
`S_conv=Dqkv×Kconv=10240×4=40,960`；PLE 组另有
`S_ple_conv=10240×9=92,160`、`L_ctx=2`；`T_tok=B×seqlen`。

## 交付入口与真实产物

```bash
PY=/Users/xiechonghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
cd /Users/xiechonghang/Workspace/LLMarchi_excel
PYTHONPATH=structure_reader/src "$PY" structure_reader/scripts/read_model_structure.py \
  build-hf --model qwen_3_8_flash_next_hf --all-variants --json
```

- **最终候选**：`c921712221160320176fe73003fc02b78c6c53a0d9c8bf895908b0250bef8454`
- **真实 HTML**：[viewer/structure.html](../../../structure_reader/outputs/qwen_3_8_flash_next_hf/c921712221160320176fe73003fc02b78c6c53a0d9c8bf895908b0250bef8454/viewer/structure.html)
  （2,101,011 B；SHA256 `12b14bc4…2017da`）
- **真实 Excel**：[tables/decoder_variants.xlsx](../../../structure_reader/outputs/qwen_3_8_flash_next_hf/c921712221160320176fe73003fc02b78c6c53a0d9c8bf895908b0250bef8454/tables/decoder_variants.xlsx)
  （208,947 B；SHA256 `65253558…cacfe7`）
- 同候选：`data/layer_structures.json`（627,192 B；SHA256 `345f450d…9e6611`）、
  `data/network_overview.json`（51,037 B；`c7ddf7a2…8992a9`）、
  `data/effective_config.json`（30,654 B；`1a6f8500…40bfa`）、
  `reports/source_manifest.json`（11,090 B；`984d0cc2…1ce868`）、`run_manifest.json`。
- **清理的本票中间候选**（仅本票、仅 Qwen 目录，已精确删除并记录）：
  `236b0aa3…`、`23002294…`、`08a98674…`、`71ff077f…`、`1f3c90e6…`、`78bf0a90…`、`5feb7915…`、`9dedbe81…`。
  接手前 DeepSeek `c3fd4296…`、GLM-5.3 `abefb101…`、GLM-5.3-Flash 全部候选
  （含 T06 最终 `29f82a3e…`）未动；T06 及更早候选均保留。
- 重复执行同一命令：analysis_id 不变、`reused: true`，候选目录不增。

## 已实现

### 1. 分层数据（同一份参数化结构）

- `extract/qwen38flash_hf.py`：三种真实层型组合各一份 Block 结构，shape 保留符号
  `seqlen`/`kvcachelen`；算子/张量数 73/107（gdn-moe）、100/146（gdn-ple-moe）、
  95/133（qsa-moe）。
  - GDN：attn/ffn 两处 Gated Residual；qkv 合并投影、z/beta/decay 投影、conv state、
    因果短卷积、q/k/v split、beta、decay、q/k repeat、L2Norm、q 缩放、delta-rule
    扫描、recurrent state 写回、Gated RMSNorm、o_proj；TopK MoE + 共享专家。
  - PLE：上下文 [B,2]、EOS shift、bigram/trigram 哈希、16 头查表、key/value/query
    投影、流级 gate、保号 sqrt、sigmoid 门控、分组 RMSNorm、dilation-3 短卷积
    （state 9）、加回 4 路流。
  - QSA：q/gate 合并投影、q/k RMSNorm、partial RoPE（64）、K/V 投影与缓存、
    indexer QK 投影、逐 token raw key 缓存（无逐 token K Norm）、逐 query N_q/B_q 的 micro-block 均值、block key RMSNorm/RoPE、relu(q·k) 求和打分、块 top-k、tail、pad 2051、bool→0/min selected mask、causal mask 合并、
    eager GQA QK/softmax/AV、输出 sigmoid 门、o_proj。
- `extract/qwen38flash_network.py`：token→logits 整网（embedding、PLE token id、
  4 行 position_ids、causal + recurrent 两套 mask、4 路 repeat、48 层真实顺序与
  三组归属、hyper_connection_mixer 读门加权 mean、logits 切片、lm_head）；
  **没有 Final RMSNorm**；MTP/视觉边界显式。

### 2. 分层浏览页（复用同一渲染核心）

- `viewer/structure.html`：10 个层级 section（整网 + 3 层组 + 3 Block + 3 算子层）、
  50 个 SVG、46 个模块折叠、1686 个唯一 id、普通锚点全部闭合；默认只显示当前层级，
  面包屑/链接逐级进入与返回；场景按钮切换 5 个预设；源码快照、参数/张量/图/算子
  联查沿用既有锚点体系。三组分别显示真实层索引与次数（35/1/12，共 48）。
- 模型 profile（`render/hf_model_profile.py`）把 Qwen 的层型短名、层组说明、层型
  选择表、模块分组、总览布局、模块讲解与源码命名表从通用页面装配中分开；
  DeepSeek/GLM-5.3/GLM-5.3-Flash 默认 profile 保持既有输出。图例只含 Qwen 自身
  机制（Gated DeltaNet、QSA micro-block indexer、4 路 Gated Residual、N-gram PLE），
  不含 KDA/mHC/Sinkhorn/CSA/HCA/滑窗/Hash 等其它模型机制。
- PLE 组 id 使用 `gdn-ple-moe`（不是 `gdn-moe-ple`），避免与 `gdn-moe` 形成
  锚点前缀歧义；这是实际点击验证发现并修复的导航问题。

### 3. 同源两页工作簿

- `tables/decoder_variants.xlsx`：`Prefill`/`Decode` 两页，每页 3 组、**268 个算子行**
  （73/100/95），来自与 HTML 同一份 `hf_layer_structure.v1`。
- 每页顶部可填写性能配置（seqlen 8/1、kvcachelen 0/4096、batch 1、精度、位宽、
  专家命中数未知）；派生参数 `Lkv/Lidx/Npool/Bsel/j/Vq/Ksel/Ksel_last/Kout/Ttok/
  Srec/Sconv/Sple/Lctx` 是引用本页输入/本组参数单元格的活公式（带 `IF`/`ISNUMBER`/
  `MOD`/`IFERROR` 守卫），并有实际输入/输出/权重 shape 公式列。`j` 是 query 索引
  占位（未给定保持未知）；其 domain 守卫为整数 `0<=j<seqlen`，且 seqlen 必须是
  正整数输入；只有 `V_q` 得到数值后 `Ksel` 才计算。填入合法 j 后 `V_q/Ksel` 随
  query 活算，`Ksel_last` 始终是最后 query 的值、不依赖 j。
- 数据相关量 `N_hit`/`N_e` 写“未知（原因）”，不写 null/0；成本列统一“待提供”，
  无猜测耗时、无成本 SUM；层组循环次数来自真实层索引，不复制 48 层/512 专家。
- 整网页尾列出网络外围 9 个算子、三组真实层索引/次数（35/1/12，共 48）、MTP 与
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
| `build_hf_decoder.py` | `_Adapter` 注册表新增 `qwen_3_8_flash_next_hf`：共同编排、来源/工具 SHA、检查、渲染、导出仍只一处；Qwen 单层/ONNX 显式拒绝（并补全拒绝文案）。 |
| `extract/hf_layer_structures.py` | 层组容器新增可选 `ple` 维度（Qwen 同一 attention_type 下由 PLE 区分层型）；DeepSeek/GLM 没有该键时不写，结构 JSON/工作簿字节不变。 |
| `export/operator_workbook.py` | 参数 id 同时作为可引用别名（Qwen 的 `Dqkv/D_qkv`、`Pfrac/partial_rotary_factor`、`Lpconv/L_ple_conv`）；只在 id≠symbol 时加入，冲突仍由 ambiguous 检查拒绝。既有三模型工作簿 SHA 逐字节不变。 |
| `render/hf_model_profile.py` | 新增 `QWEN38FLASH_PAGE_PROFILE` 与 Qwen 模块规则/布局/frames/图例/源码表；DeepSeek/GLM-5.3/Flash profile 不改。 |
| `render/display_names.py` | 新增 Qwen 参数 id→配置字段显示名（NkH/NvH/Dk/DvH/Dqkv/Dz/Rhc/Nrep/Pfrac/Drot/Iqh/Ikh/Btop/Kc/Ngm/Hng/Ngh/Dgh/Dple/Kple/Dild/Lpconv/Lctx/Vng/Vngpad/Eos/Sple/Bsel/Dq/Dkv 等）；只增键，不改旧映射。 |
| `cli.py` | `--model` 帮助列出 Qwen 模型键，不再只写三个模型。 |
| `structure_reader/README.md`、`docs/hf_reading_variants_contract_v1.md` | 增加 Qwen 入口、身份/边界与 T07 增量说明。 |

### 新增

| 文件 | 职责 |
| --- | --- |
| `rules/qwen38flash_hf.py` | 官方身份/revision、8 项源码指纹、text/vision 边界、配置静态解释（GDN/QSA/indexer/Gated Residual/PLE/MoE）、3 组归属、参数来源记录、场景。 |
| `extract/qwen38flash_hf.py` | 三种层型组合的参数化 Block 结构（GDN/PLE/QSA/MoE/缓存/布局）。 |
| `extract/qwen38flash_network.py` | 48 层 token→logits 整网总览与层序/组归属。 |
| `checks/qwen38flash_hf_contracts.py` | Qwen 语义检查：身份/层组/GDN/QSA/indexer/PLE/Gated Residual/MoE/场景/输入身份；拒绝 KDA/mHC/Sinkhorn/CSA/HCA/滑窗/Hash 与 sigmoid 路由。 |
| `export/qwen38flash_workbook.py` | Qwen 薄适配：两页三组、派生映射、真实层归属校验、GDN/QSA 分组派生、成本待提供。 |
| `render/qwen38flash_narrative.py` | Qwen 模块/总览中文讲解（按 GDN/QSA、PLE/MoE 分支）。 |
| `render/qwen38flash_source_labels.py` | Qwen 算子源码表达式/张量短名/行号。 |
| `tests/test_qwen38flash_structure.py` | 49 项回归：身份/层组/GDN/QSA/indexer/PLE/Gated Residual/MoE/场景/契约拒绝/产物/工作簿/profile/锚点/同进程显示不污染。 |
| `models/Qwen/Qwen3.8-Flash-Next/config.json`、`README.md` | 官方 revision 的小文件来源（4,745 B + 65,155 B），不含权重。 |

### 删除 / 退役

- 本票未删除产品文件；仅删除本票精确中间候选（见上）。DeepSeek 单层/四组 ONNX、
  GLM-5.3 分层页、GLM-5.3-Flash 分层页与全部接手前候选保留。

## 实际验证证据

### 相关测试（本票环境）

| 测试 | 数量 | 结果 |
| --- | --- | --- |
| `tests/test_qwen38flash_structure.py`（本票新增） | 49 | OK |
| `tests/test_glm53flash_structure.py` | 62 | OK |
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

合计 **485 项**（49 项本票新增 + 436 项既有直接相关；明细见
[`tests.json`](T07-verify/tests.json)）。P2 返修后重跑新增 49 项与受影响既有
231 项，全部通过；其余既有项未在冻结后重复昂贵全跑。未跑的
`test_hf_decoder_diagram` 在 T05 接手时已有 6 项旧布局断言失败，本票未改该历史测试；
ONNX 测试未计为通过（本环境缺 onnx 依赖，且 Qwen 显式拒绝 ONNX）。

### 最终候选独立验证

[`final-independent-artifacts.json`](T07-verify/final-independent-artifacts.json)：
`status=passed`，对象是落盘候选 `c9217122…`：

- `run_manifest` 7 个交付 SHA、8 项输入 SHA、26 项工具 SHA 全部复算一致；
  `analysis_id` 与候选目录一致。
- 数据：三组 35/1/12 恰好覆盖 48 层；QSA 层 3,7,…,47；PLE 层 0-based 1；
  Block 算子/张量 73/107、100/146、95/133；参数与 `config.json` 独立核对
  （H=2560、48 层、Nh=24/Nkv=2、Dh=256、rotary=64、GDN 16/48×128、conv=4、
  indexer 4/1×128、budget 2048/compress 4/topk 512、E=512/K=10/Es=640/Ess=640、
  hc_count=4/lowrank=320、PLE 16×160/conv 9/dilation 3、eps=1e-6、
  eos=248044）。
- 场景独立期望：Prefill(8,0) Lkv=8、Npool=2、Bsel=2、Ksel_last=8；Decode(1,4096)
  Lkv=4097、Npool=1024、Bsel=512、Ksel_last=2049；长历史(131072) Lkv=131073、
  Npool=32768；多 token(4,4092) Lkv=4096、Ksel(j)=2049/2050/2051/2048、
  Ksel_last=2048；符号场景 Lkv/Npool/Vq/Ksel/Ksel_last unknown。
- 契约负例：K_sel 回退为全长公式、删 selected mask、读门写成标量、router 改成
  sigmoid 均被具名拒绝；PLE 非 layer 1、QSA 带 PLE 也被拒绝。
- HTML：2,101,011 B、1686 个 id 无重复、普通锚点全部闭合、50 个 SVG、46 个模块；
  页面散文无 KDA/mHC/Sinkhorn/GLM/DeepSeek 等其它模型事实（源码快照中的官方注释
  不计为页面事实）。
- 工作簿结构：2 页 × 3 组 × **268 行**（73/100/95）；输入/派生公式/实际 shape/
  循环归属/成本待提供/数据相关未知；无成本 SUM；无其它模型组 id。

### 现成工作簿公式重算（同一文件，不重新导出）

[`formula-recalc-check.json`](T07-verify/formula-recalc-check.json)：`status=passed`，
对象是同一份落盘 `tables/decoder_variants.xlsx`（SHA256 `65253558…cacfe7`）的临时
副本，用独立 Excel 公式引擎 `formulas 1.3.4` 改单元格后重算；不是重新导出、不是
改 spec 再生成。24 例：

- 原场景：Prefill `Lkv/Lidx/Npool/Bsel/Ksel_last/Kout=8/8/2/2/8/2051`；
  Decode `4097/4097/1024/512/2049/2051`；GDN `Srec/Sconv=786432/40960`；
  PLE `Sple/Lctx=92160/2`；`Ttok` Prefill 8、Decode 1。
- 多 token：S=4/P=4092/B=2 → `Lkv=4096, Npool=1024, Bsel=512, Ksel_last=2048,
  Ttok=8`；逐 query j=0/1/2/3 → `V_q=4093/4094/4095/4096`、
  `N_q=1023/1023/1023/1024`、`B_q=512`、`Ksel=2049/2050/2051/2048`。
- **逐 query 零块分支**：S=8/P=0、j=0/1/2/3/6/7 →
  `N_q=0/0/0/1/1/2`、`B_q=0/0/0/1/1/2`、`R_tail=1/2/3/0/3/0`、
  `Ksel=1/2/3/4/7/8`；N_q=0 时 pool/norm/RoPE/score/topk 不执行，tail 与输出
  mask 仍按源码执行；`Ksel_last` 始终为 8。
- 边界：j=-1/0.5/4/abc/空白 → `V_q/N_q/B_q/R_tail/Ksel` 未知、`Ksel_last` 仍 2048；
  seqlen 空白/abc → `Lkv/Vq/Ksel/Ksel_last` 未知；seqlen=0 且 j=0 →
  `Lkv=4092, Ksel_last=2048`、`Vq/Ksel` 未知（j<seqlen 守卫）；
  长历史 131072 → `Lkv=131073, Npool=32768, Bsel=512, Ksel_last=2049`；
  恢复 S=4/P=4092/j=3 → `Ksel=2048`；全表扫描无 Excel 错误值。
- **未做 Microsoft Excel / LibreOffice 原生 GUI 打开验证**：本机未找到 LibreOffice
  可执行文件，改用独立公式引擎；`formulas` 的结果语义可核对，但不等于 Excel GUI
  验收。

### 页面渲染与真实点击证据（headless Chrome，非用户验收）

- [整网页](T07-verify/qwen-network.png)、[GDN Block](T07-verify/qwen-block-gdn-moe.png)、
  [QSA Block](T07-verify/qwen-block-qsa-moe.png)、
  [GDN+PLE Block](T07-verify/qwen-block-gdn-ple-moe.png)。
- 已用 URL hash 实际加载 `#/network`、`#/group/<gid>`、`#/group/<gid>/block`、
  `#/group/<gid>/ops/<module>` 并截图；另用 headless Chrome 同源 iframe 做**真实
  DOM click**：整网 → gdn-moe 层组 → Block → 模块 gdn_scan → 张量
  `g_gdn_state_pre` → 参数 NvH → 返回整网；qsa-moe Block → 模块 qsa_indexer →
  张量 `g_qsa_ix_mask` → 参数 Ksel_last → 返回 Block；gdn-ple-moe Block →
  模块 ple_ngram → 张量 `g_ple_ids` → 参数 Ngh。16 步均记录 hash/可见层级，
  `ok=true`、无错误、无缺目标/重复 id
  （[navigation-click-check.json](T07-verify/navigation-click-check.json)）。
  宽 Block 图在页面内横向滚动（与 T05/T06 同一 `.diagram-scroll` 交互），不是裁剪丢失。
- 未做人工鼠标/键盘/窄屏阅读，也未做真实用户阅读认可。

### 跨模型不回退

[`cross_model_regression.py`](T07-verify/cross_model_regression.py) /
[`cross-model-regression.json`](T07-verify/cross-model-regression.json)：用本票代码在
内存重新构建三模型（不发布新候选、不改旧候选），与既有最终候选逐字节比较：

| 模型 | 产物 | 结果 |
| --- | --- | --- |
| DeepSeek（T04 `c3fd4296…`） | `data/layer_structures.json` / `data/network_overview.json` / `tables/decoder_variants.xlsx` | 完全相同 |
| GLM-5.3（T05 `abefb101…`） | 同上 | 完全相同 |
| GLM-5.3-Flash（T06 `29f82a3e…`） | 同上 | 完全相同 |

说明：本票新增共享 `ple` 维度、`display_names` 键和参数 id 别名后，三模型的结构
JSON/网络/工作簿字节不变；`test_same_process_display_not_polluted` 还验证
Qwen→DeepSeek→GLM→Qwen 同进程构建后 DeepSeek 仍为
`sum_j comb[j,k]·residual[j,d]`、GLM/Qwen 不含 `query_index`、Qwen 第二次构建与
第一次逐字节相同。本票未重写旧候选目录，旧候选及其证据保留。

第一轮复核 P2-5 还做了**已有三模型 HTML 的实际逐字节比较**：
[`cross-model-display-check.py`](T07-verify/cross_model_display_check.py) /
[`cross-model-display-check.json`](T07-verify/cross-model-display-check.json)
用当前代码在内存重建 DeepSeek/GLM-5.3/GLM-5.3-Flash，仅归一化候选 id（以及
T05/T06 已验收的共享 CSS 中性注释）后与既有最终候选 HTML 逐字节比较，三模型均为
`true`；GLM/GLM-Flash HTML 仍显示 `qk_head_dim`，不含 Qwen 的 `query_width`。
Qwen 的 `Dq` 已改为 `Dqw/D_qw`，并移除会覆盖旧模型的 `Nmtp` 全局映射。

## 未验证 / 明确限制

- 用户性能公式与 XPU 能力仍未提供：无算子耗时/组小计/整网串行汇总；成本列“待提供”，
  不猜、不记 0、不做父子重复计数。
- Qwen3.8-Flash-Next **只提供分层入口**（`--all-variants`）；单层长页未实现，
  缺该参数时具名拒绝。未实现可选 ONNX，显式 `--onnx` 具名拒绝（不虚构、不静默忽略）。
- 只覆盖 text_config 语言主干：视觉 27 层、图像/视频 placeholder 替换、视觉网格
  M-RoPE 不在本票；MTP 配置声明 1 层但本地 HF 实现没有模块、加载时忽略、forward
  不执行；`generate`/采样循环、loss/aux_loss、`output_router_logits` 辅助输出、
  设备 kernel、bfloat16 数值、torch.compile/梯度检查点未分析；不做数值等价。
- PLE 的 n-gram 哈希具体命中头/token 是数据相关量，图上不猜；只固定 16 头×160 维、
  上下文 [B,2]、dilation-3 state 9 与码表构造规则。
- 版本口径：config 5.8.0.dev0 与本地 5.19.0.dev0 的差异已在上文说明；本票只表达
  本地实际代码。若复核要求改用某个历史 tag，需要重新锁定该 tag 并重做层型/缓存/
  RoPE 语义。
- 静态工程、公式引擎重算与 headless 渲染通过不代表用户阅读认可；需 Codex/用户复核。
- 可选 Archify：本票**未实际调用**。模型事实稳定后，L0 总览与局部机制图已由同一份
  结构 JSON 经既有 renderer 生成；没有为出图引入平行 viewer 或另一份模型结构。
  图形检查不代替语义/公式验收。

## 保护与停止

- 未 reset/clean、未批量格式化、未恢复根两项删除、未改接手前未提交改动；相对 T07
  快照的 18 项产品改动/新增、19 项证据与 14 项外部协调文件见
  [changed-files.json](T07-verify/changed-files.json) 与
  [incremental.patch](T07-verify/incremental.patch)；73 项 `models/`/`.claude/` 接手前
  输入单列为 `preexisting_input`，不归本票。
- 接手前 DeepSeek `c3fd4296…`、GLM-5.3 `abefb101…`、GLM-5.3-Flash 全部候选及
  T06 最终 `29f82a3e…` 未动；本票保留第一轮正式候选 `c0d4ed84…` 与最终候选 `c9217122…`，精确清理 8 个本票
  中间候选（见上）。
- 协调/自动化文件只读，未编辑；定时检查保持外部设置的 ACTIVE/45 分钟，本票未改。
- **完成即停止，等待 Codex 复核**：不 commit/push、不关闭 #8、不进入 T08。
