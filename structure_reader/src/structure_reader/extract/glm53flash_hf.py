"""GLM-5.3-Flash（zai-org/GLM-5.3-Flash, HF transformers ``glm5_next``）Block 结构提取。

从 ``rules/glm53flash_hf.py`` 解释的文本有效配置与本地官方 Transformers 检出源码，
构建**一份** 参数化 Block 结构（``hf_layer_structure.v1``），供分层浏览页与同源
工作簿使用；同一份结构再由 ``extract/hf_scenario.py`` 投影成既有
``hf_decoder_reading.v1`` 的 prefill/decode 场景。本模块不生成渲染文件、不执行
模型、不加载权重、不下载权重。

本配置只存在三种真实层型组合（不虚构第四种）：

- ``kda-dense``：KDA 线性注意力 + dense MLP（层 0/1/2，first_k_dense_replace=3）；
- ``dsa-moe``：MLA/DSA 稀疏注意力 + TopK MoE（层 3/7/11/…/43，共 11 层）；
- ``kda-moe``：KDA 线性注意力 + TopK MoE（其余 31 层）。

结构事实（行号按本地检出 536ecc00…，文件 modeling_glm5_next.py，下称 modeling；
配置规则见 configuration_glm5_next.py，下称 configuration）：

- DecoderLayer（modeling:1278-1348）**不是普通残差**：每层有 ``attn_hc`` 与
  ``ffn_hc`` 两个 mHC 模块，把 4 路流坍缩成单流（pre 加权和），子层输出再按
  ``post`` 展开、按 ``comb``（Sinkhorn 双随机矩阵）与输入流混合：
  ``post*out + comb^T @ residual_streams``。网络入口把 embedding 广播成 4 路
  （:1493），末端 ``hc_head`` 取 4 路均值（:332-336/:1509）。
- KDA（modeling:620-769）：q/k/v 投影 → depthwise 因果短卷积（kernel 4，conv state
  固定 4 步）→ forget gate ``-5*sigmoid(exp(A_log)*g)`` + beta → 分块/递推
  delta-rule 扫描（recurrent state ``[B,64,128,128]``）→ gated RMSNorm → o_proj。
  状态大小固定，不随 kvcachelen 增长；recurrent 仅在“已有状态且 seq_len==1”时走，
  空缓存首次 Prefill（含 P=0/S=1）与多 token 续接走 chunk 路径（chunk_size=64）。
- MLA/DSA（modeling:1100-1275）：q 低秩 1536 + NoPE（``qk_rope_head_dim=0``，K 只有
  256 维 nope 段）；KV 缓存写**压缩 latent**（keys 512、values 0 维 k_rot），用时
  ``kv_b_proj`` 展开全部 Lkv；indexer 独立 wq_b/wk/k_norm/weights_proj，并把
  k(128)+gate(128)+valid(1) 打包成 [B,L,257] packed 缓存。
- DSA indexer（modeling:772-1060）：k-pool=4 把连续 4 个 token 压成一个候选池
  （softmax 加权键）；每 query 按分数选 ``min(index_topk//kpool, 池数)`` 个池，展开
  成原始 token；当前不完整池的 tail 直接追加；再用 selected counts scatter 成
  稀疏可见 mask（True=可见）。
- 稀疏 mask（modeling:1237-1275）：``selected_counts.scatter_add_(-1, safe, valid)``
  → ``mask = counts.ne(0).unsqueeze(1)``（[B,1,S,Lkv]）；eager 路径再 where 成
  0/min。top-k 只能收窄可见范围。
- eager 注意力（modeling:1075-1097）：repeat_kv（Nkv=Nh → 恒等）→ QKᵀ×scaling
  （1/16）→ +mask → softmax(fp32) → AV → transpose(1,2).contiguous()。
- 稠密 MLP / MoE / 共享专家（modeling:85-206）：SwiGLU 先 clamp（gate≤10、up±10）；
  MoE sigmoid 打分 + correction bias + 288 选 8 + 归一(+1e-20) + ×2.5，专家
  ``one_hot(top_k,E+1)`` 后按 (top_k_pos, token_idx) gather/加权/index_add。
- 缓存（cache_utils:1067-1149/:331-380）：KDA 层 → LinearAttentionLayer
  （conv_states + recurrent_states，固定大小）；DSA 层 → DynamicIndexedLayer
  （主 KV 压缩 latent + indexer packed keys）。Lkv=Lidx=kvcachelen+seqlen。

场景（seqlen/kvcachelen）只绑定取值：结构 shape 保留符号，空缓存 Prefill 的
kvcachelen=0；Decode 场景 seqlen=1。数据相关量（命中专家数、每专家 token 数、
具体选中的池/token）保留未知，不默认 0。
"""

from __future__ import annotations

from pathlib import Path

from structure_reader.extract.hf_scenario import STRUCTURE_SCHEMA
from structure_reader.rules.glm53flash_hf import (
    GLM53FLASH_SCENARIOS,
    MODEL_REPOSITORY,
    SOURCE_FILES,
    HfModelValues,
    load_hf_model_values,
)

__all__ = [
    "ReadingDocError",
    "SAMPLE_LAYER_INDEX",
    "STRUCTURE_SCHEMA",
    "build_block_structure",
    "build_layer_documents",
    "build_layer_structure",
    "build_reading_document",
]

SAMPLE_LAYER_INDEX = 0  # 第一组 kda-dense（其余组由层组容器逐组构建）
SCHEMA = "hf_decoder_reading.v1"

_EXPR_LKV = "kvcachelen + seqlen"
_EXPR_LIDX = "kvcachelen + seqlen"
_EXPR_NPOOL = "L_idx // Kpool"
_EXPR_SELP = "min(Ki // Kpool, N_pool)"
_EXPR_VQ = "kvcachelen + j + 1"
_EXPR_KSEL = "min(V_q, Ki + (V_q % Kpool))"
_EXPR_KSEL_LAST = "min(L_idx, Ki + (L_idx % Kpool))"
_EXPR_KOUT = "Ki + Kpool - 1"
_EXPR_TTOK = "B * seqlen"

_SCOPE_NOTES = [
    "范围：单个 decoder Block（同一层组只保留一份模板；真实层索引、顺序与循环次数见整网层组）。",
    "实现路径：transformers eager（本地 glm5_next 实现；_supports_flash_attn=False、"
    "_supports_sdpa=True，modeling:1357-1362）。注意 eager/sdpa 的 DSA 稀疏 mask 是同一"
    "份 selected counts 语义；本图表达该 mask 与 eager 的 where 转换。",
    "静态结构一致为目标：不加载权重、不运行模型、不做数值等价；shape 保留符号 "
    "seqlen/kvcachelen，场景预设只绑定取值，不改变结构。",
    "同一权重/激活张量可被多个算子引用（共享引用非重复定义）；缓存调用前/调用后以两个"
    "张量分别表达，不混为原地更新。",
    "本模型是多模态仓库中的**语言主干**；视觉/图像输入路径不在本票范围，本结构不含图像编码器。",
    "版本口径：根配置记录 transformers_version=5.16.0，实现取本地官方检出 5.19.0.dev0。"
    "本票只表达本地代码的压缩 latent 缓存/NoPE/mHC 语义，不混用 5.16 的展开 K/V 缓存结论。",
]


class ReadingDocError(ValueError):
    """阅读数据构建失败（配置/范围/事实来源不一致），具名拒绝。"""


def _block_builder(values: HfModelValues, layer_index: int) -> "_BlockBuilder":
    attention_type = values.layer_type(layer_index)
    mlp_type = values.mlp_layer_type(layer_index)
    if attention_type == "linear_attention" and mlp_type == "dense":
        return _KdaBlockBuilder(values, layer_index, "dense")
    if attention_type == "indexed_attention" and mlp_type == "sparse":
        return _DsaBlockBuilder(values, layer_index)
    if attention_type == "linear_attention" and mlp_type == "sparse":
        return _KdaBlockBuilder(values, layer_index, "sparse")
    raise ReadingDocError(
        f"layer_index={layer_index} 是 {attention_type} + {mlp_type}；"
        "GLM-5.3-Flash 已核实的真实组合为 KDA+dense、DSA+sparse、KDA+sparse，"
        "不虚构第四种组合"
    )


def build_block_structure(values: HfModelValues, layer_index: int) -> dict:
    """构建**一份** Block 结构（参数/张量/算子/缓存注记/派生量），不含阶段复制。"""
    if not 0 <= layer_index < values.num_hidden_layers:
        raise ReadingDocError(f"layer_index {layer_index} 越界（0..{values.num_hidden_layers - 1}）")
    builder = _block_builder(values, layer_index)
    block = builder.build()
    block["layer_index"] = layer_index
    block["attention_type"] = values.layer_type(layer_index)
    block["moe_type"] = values.mlp_layer_type(layer_index)
    block["indexer_type"] = values.indexer_type(layer_index)
    block["kind"] = builder.kind
    return block


def build_layer_documents(
    workspace_root: str | Path,
    layer_index: int = SAMPLE_LAYER_INDEX,
    model_key: str = "glm_5_3_flash_hf",
) -> tuple[dict, dict]:
    """一次构建，两种同源表示：一份 Block 结构 + prefill/decode 场景投影。"""
    from structure_reader.extract.hf_scenario import (
        project_legacy_phase,
        structure_document,
    )

    values = load_hf_model_values(workspace_root, model_key=model_key)
    block = build_block_structure(values, layer_index)
    block["scope_notes"] = list(_SCOPE_NOTES)
    structure = structure_document(values, block, scenarios=GLM53FLASH_SCENARIOS)
    legacy = {
        "schema": SCHEMA,
        "model": {
            "model_key": values.model_key,
            "display_name": values.display_name,
            "implementation": values.implementation,
            "source_revision": values.source_revision,
            "config_path": values.config_path,
            "model_revision": values.model_revision,
        },
        "scope": {
            "layer_index": layer_index,
            "attention_type": block["attention_type"],
            "moe_type": block["moe_type"],
            "indexer_type": block["indexer_type"],
            "analysis_path": "eager",
            "notes": [
                f"样板范围：单个 decoder 层（layer_idx={layer_index}，"
                f"{'KDA 线性注意力' if block['kind'] == 'kda' else 'MLA/DSA 稀疏注意力'} + "
                f"{block['moe_type']}）；整网 Embedding/4 路 mHC/45 层/Final RMSNorm/"
                "lm_head 见整网总览。",
                "模型身份：zai-org/GLM-5.3-Flash revision "
                f"{MODEL_REPOSITORY['revision']}，model_type=glm5_next（text_config=glm5_next_text）；"
                f"配置保存的 transformers_version={values.checkpoint_transformers_version}，"
                "实现取本地官方检出 5.19.0.dev0。",
                "实现口径：只表达本地 5.19 的压缩 latent 缓存、NoPE、mHC 4 路流与 k-pool "
                "DSA；不与其它 GLM 变体的普通残差或 5.16 的展开 K/V 缓存结论混用。",
                "静态结构一致为目标：不加载权重、不运行模型、不做数值等价；"
                "seqlen/kvcachelen 保留符号，两阶段只是场景预设。",
                "视觉/图像输入路径不在本票范围；本结构只覆盖 token 输入到 logits 的语言主干。",
            ],
        },
        "phases": [
            project_legacy_phase(block, phase_id) for phase_id in ("prefill", "decode")
        ],
    }
    return structure, legacy


def build_layer_structure(
    workspace_root: str | Path,
    layer_index: int = SAMPLE_LAYER_INDEX,
    model_key: str = "glm_5_3_flash_hf",
) -> dict:
    """构建 ``hf_layer_structure.v1``：一份 Block 结构 + 场景预设/派生值。"""
    return build_layer_documents(workspace_root, layer_index, model_key)[0]


def build_reading_document(
    workspace_root: str | Path,
    layer_index: int = SAMPLE_LAYER_INDEX,
    model_key: str = "glm_5_3_flash_hf",
) -> dict:
    """既有冻结接口 ``hf_decoder_reading.v1``：同一份结构的 prefill/decode 投影。"""
    return build_layer_documents(workspace_root, layer_index, model_key)[1]


class _BlockBuilder:
    """单 Block 结构累积器：参数 → 边界/缓存张量 → 算子按真实阅读顺序追加。"""

    kind = "block"

    def __init__(self, values: HfModelValues, layer_idx: int):
        self.values = values
        self.layer_idx = layer_idx
        self.parameters: list[dict] = []
        self.tensors: list[dict] = []
        self.operators: list[dict] = []
        self.derivations: list[dict] = []
        self.cache_notes: list[str] = []
        self.coverage_notes: list[str] = []

    # ------------------------------------------------------------------
    # 记录辅助（与既有 HF 阅读文档同构）
    # ------------------------------------------------------------------

    def param(self, pid, symbol, meaning, value, expression, note, source_refs,
              unknown_reason: str = "", domain: dict | None = None) -> str:
        self.parameters.append({
            "id": pid, "symbol": symbol, "meaning": meaning, "value": value,
            "expression": expression, "note": note, "source_refs": source_refs,
            **({"unknown_reason": unknown_reason} if unknown_reason else {}),
            **({"domain": dict(domain)} if domain else {}),
        })
        return pid

    def tensor(self, tid, name, shape, parameter_refs, kind, note) -> str:
        self.tensors.append({
            "id": tid, "name": name, "shape": shape, "parameter_refs": parameter_refs,
            "kind": kind, "note": note,
        })
        return tid

    def weight(self, tid, name, shape, parameter_refs, note) -> str:
        return self.tensor(tid, name, shape, parameter_refs, "weight", note)

    def op(self, oid, module_path, name, op, inputs, outputs, weights,
           parameter_refs, source_refs, note, analysis_status="analyzed") -> str:
        self.operators.append({
            "id": oid, "module_path": module_path, "name": name, "op": op,
            "inputs": inputs, "outputs": outputs, "weights": weights,
            "parameter_refs": parameter_refs, "source_refs": source_refs,
            "analysis_status": analysis_status, "note": note,
        })
        return oid

    def derivation(self, did, label, expression, note) -> str:
        self.derivations.append({
            "id": did, "label": label, "expression": expression, "note": note,
        })
        return did

    def src(self, locator, note) -> dict:
        return {"path": SOURCE_FILES["modeling_glm5_next.py"],
                "locator": locator, "kind": "source", "note": note}

    def srcf(self, file_key, locator, note) -> dict:
        return {"path": SOURCE_FILES[file_key], "locator": locator,
                "kind": "source", "note": note}

    def cfg(self, locator, note) -> dict:
        return {"path": SOURCE_FILES["config.json"], "locator": locator,
                "kind": "config", "note": note}

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------

    def build(self) -> dict:
        raise NotImplementedError

    def _common_parameters(self) -> None:
        v = self.values
        cfg = self.cfg
        self.param("B", "B", "batch 批大小", None, None,
                   "调用输入维度，静态分析保留符号", [])
        self.param("seqlen", "seqlen", "本次输入的新 token 数（seqlen）", None, None,
                   "Prefill 为提示长度、Decode 为 1；结构保留符号，场景预设只绑定取值", [])
        self.param("H", "H", "hidden_size 隐层维度", v.hidden_size, None,
                   "config.hidden_size", [cfg("text_config.hidden_size", "文本子配置字段")])
        self.param("Hc", "Hc", "hc_mult mHC 残差流数", v.hc_mult, None,
                   "config.hc_mult=4；每层两处 mHC 把 4 路流坍缩/展开/混合，"
                   "网络末端取 4 路均值",
                   [cfg("text_config.hc_mult", "文本子配置字段"),
                    self.src("218-329", "Glm5NextTextHyperConnection"),
                    self.src("332-336", "Glm5NextTextHyperHead.mean(dim=2)")])
        self.param("Mhc", "Mhc", "mHC 单站点系数个数（(2+Hc)×Hc）", (2 + v.hc_mult) * v.hc_mult,
                   "Mhc = (2 + Hc) * Hc",
                   "pre Hc + post Hc + comb Hc×Hc；Hc=4 时为 4+4+16=24。"
                   "权重 fn/base 的系数维与 mixes 最后一维都用它；Excel 按 Hc 活算",
                   [cfg("text_config.hc_mult", "文本子配置字段"),
                    self.src("276-281", "self.fn = Parameter(torch.empty((2 + hc) * hc, hc*hidden_size))"),
                    self.src("305-310", "split([hc, hc, hc*hc])")])
        self.param("E", "E", "n_routed_experts 路由专家总数", v.n_routed_experts, None,
                   "config.n_routed_experts=288（标准名 num_local_experts 经 attribute_map）",
                   [cfg("text_config.n_routed_experts", "文本子配置字段"),
                    self.srcf("configuration_glm5_next.py", "98", "attribute_map")])
        self.param("K", "K", "num_experts_per_tok 每 token 选择数", v.num_experts_per_tok, None,
                   "config.num_experts_per_tok=8；与专家总数 288 不同参数",
                   [cfg("text_config.num_experts_per_tok", "文本子配置字段")])
        self.param("Es", "Es", "moe_intermediate_size 单专家中间维", v.moe_intermediate_size, None,
                   "config.moe_intermediate_size=2048；共享专家宽度 = Es×n_shared_experts",
                   [cfg("text_config.moe_intermediate_size", "文本子配置字段")])
        self.param("Is", "Is", "intermediate_size 稠密 MLP 中间维", v.intermediate_size, None,
                   "config.intermediate_size=12288；只用于 dense 层",
                   [cfg("text_config.intermediate_size", "文本子配置字段")])
        self.param("Scl", "Scl", "routed_scaling_factor 路由权重缩放", v.routed_scaling_factor,
                   None, "config.routed_scaling_factor=2.5；归一后乘",
                   [cfg("text_config.routed_scaling_factor", "文本子配置字段")])
        self.param("Ng", "Ng", "n_group 路由专家组数", v.n_group, None,
                   "config.n_group=1；源码仍执行组选择分支，单组掩码全通过",
                   [cfg("text_config.n_group", "文本子配置字段")])
        self.param("Ktg", "Ktg", "topk_group 选中的专家组数", v.topk_group, None,
                   "config.topk_group=1", [cfg("text_config.topk_group", "文本子配置字段")])
        self.param("Swi", "Swi", "swiglu_limit SwiGLU 截断上限", v.swiglu_limit, None,
                   "config.swiglu_limit=10.0；gate clamp(max=10)、up clamp(±10) 后才 SiLU(gate)*up",
                   [cfg("text_config.swiglu_limit", "文本子配置字段"),
                    self.src("95-103", "Glm5NextTextMLP：gate/up clamp"),
                    self.src("136-141", "Glm5NextTextExperts._apply_gate：gate/up clamp")])
        self.param("EpsN", "EpsN", "主 RMSNorm epsilon", v.rms_norm_eps, None,
                   "config.rms_norm_eps=1e-5；input/post/final RMSNorm、q_a/kv_a/o_norm、"
                   "mHC input_norm 都用它",
                   [cfg("text_config.rms_norm_eps", "文本子配置字段"),
                    self.src("1127", "q_a_layernorm = RMSNorm(..., eps=config.rms_norm_eps)"),
                    self.src("1135", "kv_a_layernorm = RMSNorm(..., eps=config.rms_norm_eps)"),
                    self.src("1293-1294", "input/post_attention_layernorm")])
        self.param("Sit", "Sit", "mHC Sinkhorn 迭代次数", v.hc_sinkhorn_iters, None,
                   "config.hc_sinkhorn_iters=20；comb 的行列归一迭代次数",
                   [cfg("text_config.hc_sinkhorn_iters", "文本子配置字段"),
                    self.src("321-324", "for _ in range(hc_sinkhorn_iters - 1): 行/列归一")])
        self.param("Ttok", "T_tok", "本调用展平 token 数（MoE 输入行数）", None, None,
                   "T_tok = B*seqlen；MoE 把 [B,seqlen,H] 展平成 [Ttok,H] 再路由",
                   [self.src("201-204", "hidden_states.view(-1, hidden_dim)")])

    def _build_derivations(self) -> None:
        premise = ("按固定配置 + 「空态起步、连续正常执行」前提推导；"
                   "不覆盖裁剪/回滚后的自定义缓存快照")
        self.derivation("T_tok", "MoE 路由的展平 token 数", _EXPR_TTOK,
                        f"B×seqlen；B 为可填性能配置。{premise}")

    def _mhc_site(self, site: str, streams_in: str, *, residual_note: str) -> None:
        """一个 mHC 站点：4 路流 → 权重(pre/post/comb) + 坍缩单流。"""
        v = self.values
        p, t = f"g_mhc_{site}", f"g_mhc_{site}"
        self.weight(f"{p}_fn_weight", f"self.{site}_hc.fn", ["Mhc", "Hc×H"],
                    ["Hc", "H", "Mhc"],
                    "mHC 权重投影：输出 (2+Hc)×Hc = 24 个系数（pre 4 + post 4 + comb 16）；"
                    "输入 4×H 维")
        self.weight(f"{p}_base", f"self.{site}_hc.base", ["Mhc"], ["Hc", "Mhc"],
                    "mHC 的 pre/post/comb 偏置（三段各 Hc/Hc/Hc²，共 24 项）")
        self.weight(f"{p}_scale", f"self.{site}_hc.scale", ["3"], ["Hc"],
                    "pre/post/comb 三个独立 scale")
        self.tensor(f"{p}_flat", f"hidden_streams.view(B,S,{v.hc_mult}*H).float()",
                    ["B", "seqlen", "Hc×H"], ["B", "seqlen", "Hc", "H"], "activation",
                    "4 路流展平后做**无权重** RMSNorm（eps=rms_norm_eps）；计算在 float32")
        self.op(f"{p}_in_norm", f"Glm5NextTextHyperConnection.input_norm",
                 "mHC 输入无权重 RMSNorm", "rmsnorm-unweighted",
                 [streams_in], [f"{p}_flat"], [],
                 ["Hc", "H", "EpsN"],
                 [self.src("296-301", "flattened = hidden_streams.view(B,S,-1).float(); "
                                      "input_norm(flattened)"),
                  self.src("209-215", "Glm5NextTextUnweightedRMSNorm")],
                 "只对 4 路流的拼接做 RMSNorm，不乘可学习 weight；这是 mHC 权重推断的输入")
        self.tensor(f"{p}_mixes", "mixes = F.linear(flat, self.fn.float())",
                    ["B", "seqlen", "Mhc"], ["B", "seqlen", "Hc", "Mhc"], "activation",
                    "(2+Hc)×Hc=24 个系数：pre Hc、post Hc、comb Hc×Hc")
        self.op(f"{p}_fn", f"Glm5NextTextHyperConnection.fn", "mHC 系数投影", "linear",
                 [f"{p}_flat"], [f"{p}_mixes"], [f"{p}_fn_weight"],
                 ["Hc", "H", "Mhc", "B", "seqlen"],
                 [self.src("303", "flattened = F.linear(flattened, self.fn.float())")],
                 "权重 [(2+Hc)*Hc, Hc*H]=[24,16384]；线性投影输出 pre/post/comb 三段")
        self.tensor(f"{p}_pre_logits", "pre_w", ["B", "seqlen", "Hc"],
                    ["B", "seqlen", "Hc"], "activation", "split([Hc,Hc,Hc*Hc]) 的第一段")
        self.tensor(f"{p}_post_logits", "post_w", ["B", "seqlen", "Hc"],
                    ["B", "seqlen", "Hc"], "activation", "split 的第二段")
        self.tensor(f"{p}_comb_logits", "comb_w", ["B", "seqlen", "Hc", "Hc"],
                    ["B", "seqlen", "Hc", "Hc"], "activation", "split 的第三段，view 成 Hc×Hc")
        self.op(f"{p}_split", "Glm5NextTextHyperConnection.forward", "mHC 系数拆分", "split",
                 [f"{p}_mixes"], [f"{p}_pre_logits", f"{p}_post_logits", f"{p}_comb_logits"],
                 [], ["Hc"],
                 [self.src("305-310", "split([hc, hc, hc*hc]); comb_w.view(...,hc,hc)")],
                 "pre/post 是向量系数，comb 是 Hc×Hc 矩阵")
        self.tensor(f"{p}_pre", "pre = sigmoid(pre_w*scale+base)+eps",
                    ["B", "seqlen", "Hc"], ["B", "seqlen", "Hc"], "activation",
                    "流坍缩权重；加 hc_eps=1e-6 保证正")
        self.tensor(f"{p}_post", "post = 2*sigmoid(post_w*scale+base)",
                    ["B", "seqlen", "Hc"], ["B", "seqlen", "Hc"], "activation",
                    "子层输出展开到 4 路流的系数（范围 (0,2)）")
        self.op(f"{p}_gate", "Glm5NextTextHyperConnection.forward", "mHC pre/post 门控",
                 "sigmoid+scale", [f"{p}_pre_logits", f"{p}_post_logits"],
                 [f"{p}_pre", f"{p}_post"], [f"{p}_base", f"{p}_scale"],
                 ["Hc"],
                 [self.src("312-314", "pre = sigmoid(pre_w*pre_scale+pre_b)+hc_eps; "
                                       "post = 2*sigmoid(post_w*post_scale+post_b)")],
                 "pre 用 sigmoid+eps；post 用 2×sigmoid；两者都只作用于流维")
        self.tensor(f"{p}_comb", "comb（Sinkhorn 双随机）",
                    ["B", "seqlen", "Hc", "Hc"], ["B", "seqlen", "Hc", "Hc"], "activation",
                    "softmax + eps 后做 20 轮行列归一；约束为双随机矩阵")
        self.op(f"{p}_sinkhorn", "Glm5NextTextHyperConnection.forward", "mHC comb Sinkhorn",
                 "sinkhorn", [f"{p}_comb_logits"], [f"{p}_comb"],
                 [f"{p}_base", f"{p}_scale"], ["Hc", "Sit"],
                 [self.src("315-324", "comb = softmax(...)+eps; 行列归一 hc_sinkhorn_iters 轮")],
                 "Sinkhorn 迭代次数 hc_sinkhorn_iters=20；comb 混合输入流与子层输出流")
        self.tensor(f"{p}_collapsed", f"collapsed = (pre.unsqueeze(-1)*streams).sum(2)",
                    ["B", "seqlen", "H"], ["B", "seqlen", "H"], "activation",
                    "坍缩后的单流，进入 RMSNorm 与 Attention/MLP")
        self.op(f"{p}_collapse", "Glm5NextTextHyperConnection.forward", "mHC 流坍缩",
                 "broadcast-mul+sum", [streams_in, f"{p}_pre"], [f"{p}_collapsed"], [],
                 ["Hc", "H", "B", "seqlen"],
                 [self.src("328", "(pre.unsqueeze(-1) * hidden_streams).sum(dim=2)")],
                 residual_note)

    def _mhc_apply(self, site: str, sublayer_out: str, residual_streams: str,
                   streams_out: str) -> None:
        """子层输出展开 post、comb 混合 residual 流，再相加成新的 4 路流。"""
        p = f"g_mhc_{site}"
        self.tensor(f"{p}_expanded", "post.unsqueeze(-1) * sublayer.unsqueeze(-2)",
                    ["B", "seqlen", "Hc", "H"], ["B", "seqlen", "Hc", "H"], "activation",
                    "子层单流输出按 post 广播到 4 路流")
        self.op(f"{p}_expand", "Glm5NextTextDecoderLayer.forward", "mHC 子层输出展开",
                 "broadcast-mul", [sublayer_out, f"{p}_post"], [f"{p}_expanded"], [],
                 ["Hc", "H", "B", "seqlen"],
                 [self.src("1335", "post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2)")],
                 "不是把单流复制 4 份：每路系数不同（post 的 Hc 个分量）")
        self.tensor(f"{p}_combined", "matmul(comb.transpose(-1,-2), residual)",
                    ["B", "seqlen", "Hc", "H"], ["B", "seqlen", "Hc", "H"], "activation",
                    "combᵀ 把 4 路输入流混合成 4 路残差")
        self.op(f"{p}_comb_mix", "Glm5NextTextDecoderLayer.forward", "mHC 残差流混合",
                 "matmul", [f"{p}_comb", residual_streams], [f"{p}_combined"], [],
                 ["Hc", "H", "B", "seqlen"],
                 [self.src("1335-1336", "torch.matmul(comb.to(dtype).transpose(-1,-2), residual)")],
                 "comb 是双随机矩阵；这是 4×4 的流间混合，不是标量残差")
        self.op(f"{p}_add", "Glm5NextTextDecoderLayer.forward", "mHC 流相加", "add",
                 [f"{p}_expanded", f"{p}_combined"], [streams_out], [],
                 ["Hc", "H", "B", "seqlen"],
                 [self.src("1335-1337", "hidden_states = post*out + matmul(comb^T, residual)")],
                 "两处 mHC 都没有普通单流残差相加；流间关系由 post/comb 决定")

    def _moe_ops(self, ffn_in: str) -> None:
        src = self.src
        self.tensor("g_moe_flat", "hidden_states.view(-1, hidden_dim)", ["Ttok", "H"],
                    ["Ttok", "H"], "activation", "MoE 先把 [B,seqlen,H] 展平（:203）")
        self.op("g_moe_flatten", "Glm5NextTextMoE.forward", "MoE 输入展平", "view",
                [ffn_in], ["g_moe_flat"], [], ["B", "seqlen", "Ttok", "H"],
                [src("199-204", "orig_shape=...; hidden_states.view(-1, hidden_dim)")],
                "展平后按 token 路由；residuals 保留原 [B,seqlen,H] 供共享专家")
        self.weight("g_moe_router_weight", "self.mlp.gate.weight", ["E", "H"], ["E", "H"],
                    "TopkRouter.weight（:150）；forward 强制 float32（:159）")
        self.weight("g_moe_router_bias", "self.mlp.gate.e_score_correction_bias", ["E"], ["E"],
                    "非学习 buffer；加到 sigmoid 分数上参与选择（:155/:161）")
        self.tensor("g_router_logits", "router_logits = F.linear(x.float(), W.float())",
                    ["Ttok", "E"], ["Ttok", "E"], "activation", "路由打分在 float32（:159）")
        self.op("g_moe_router", "Glm5NextTextTopkRouter.forward", "MoE 路由打分", "linear",
                ["g_moe_flat"], ["g_router_logits"], ["g_moe_router_weight"],
                ["Ttok", "E", "H"],
                [src("158-160", "router_logits = F.linear(hidden_states.float(), weight.float()); "
                                "scores = router_logits.sigmoid()")],
                "强制 float32；配置 moe_router_dtype=float32 与源码一致，但源码不读该字段")
        self.tensor("g_router_scores", "scores = router_logits.sigmoid()", ["Ttok", "E"],
                    ["Ttok", "E"], "activation", "sigmoid 打分（不是 softmax）（:160）")
        self.op("g_moe_sigmoid", "Glm5NextTextTopkRouter.forward", "Sigmoid 打分", "sigmoid",
                ["g_router_logits"], ["g_router_scores"], [], ["Ttok", "E"],
                [src("160", "scores = router_logits.sigmoid()")],
                "GLM-5.3-Flash 用 sigmoid；不是 sqrt(softplus) 变体")
        self.tensor("g_scores_for_choice", "scores_for_choice = scores + bias", ["Ttok", "E"],
                    ["Ttok", "E"], "activation", "加 e_score_correction_bias（:161）")
        self.op("g_moe_bias_add", "Glm5NextTextTopkRouter.forward", "路由修正偏置", "add",
                ["g_router_scores", "g_moe_router_bias"], ["g_scores_for_choice"], [],
                ["Ttok", "E"],
                [src("161", "scores_for_choice = scores + self.e_score_correction_bias")],
                "偏置只参与选择；最终权重仍取 sigmoid 原分数（:177）")
        self.tensor("g_group_scores", "group_scores（top2 求和）", ["Ttok", "Ng"],
                    ["Ttok", "Ng"], "activation",
                    "view(-1,n_group,E/n_group).topk(2).sum(-1)（:162-166）")
        self.op("g_moe_group_score", "Glm5NextTextTopkRouter.forward", "专家组分数",
                "view+topk+sum", ["g_scores_for_choice"], ["g_group_scores"], [],
                ["Ng", "E"],
                [src("162-166", "scores_for_choice.view(-1, num_group, num_experts//num_group)"
                               ".topk(2, dim=-1)[0].sum(dim=-1)")],
                "Ng=1 时只有一个组分数；源码仍执行该分支，如实表达、不省略也不夸大")
        self.tensor("g_group_idx", "group_idx = topk(group_scores, topk_group)",
                    ["Ttok", "Ktg"], ["Ttok", "Ktg"], "activation", "选中的专家组索引（:167）")
        self.op("g_moe_group_topk", "Glm5NextTextTopkRouter.forward", "选专家组", "topk",
                ["g_group_scores"], ["g_group_idx"], [], ["Ng", "Ktg"],
                [src("167", "group_idx = torch.topk(group_scores, k=self.topk_group, "
                            "dim=-1, sorted=False)[1]")],
                "Ktg=1、Ng=1：唯一组就是全部专家")
        self.tensor("g_group_mask", "group_mask（选中组=1）", ["Ttok", "Ng"],
                    ["Ttok", "Ng"], "activation",
                    "zeros_like 后 scatter_(1, group_idx, 1)（:168-169）")
        self.op("g_moe_group_scatter", "Glm5NextTextTopkRouter.forward", "组掩码 scatter",
                "zeros+scatter", ["g_group_idx"], ["g_group_mask"], [], ["Ttok", "Ng"],
                [src("168-169", "group_mask = torch.zeros_like(group_scores); "
                                "group_mask.scatter_(1, group_idx, 1)")],
                "Ng=1 时该掩码全 1")
        self.tensor("g_group_mask_expanded", "group_mask.unsqueeze(-1).expand(-1,Ng,E÷Ng)",
                    ["Ttok", "Ng", "E÷Ng"], ["Ttok", "Ng", "E"], "activation",
                    "group_mask.unsqueeze(-1).expand(-1, n_group, E//n_group)（:170-174）")
        self.op("g_moe_group_expand", "Glm5NextTextTopkRouter.forward", "组掩码 expand",
                "unsqueeze+expand", ["g_group_mask"], ["g_group_mask_expanded"], [],
                ["Ng", "E"],
                [src("170-174", "group_mask.unsqueeze(-1).expand(-1, self.num_group, "
                                "self.num_experts // self.num_group)")],
                "把每组一个 0/1 标志扩成该组内每个专家一个标志；中间仍是 3D")
        self.tensor("g_score_mask", "score_mask = ...reshape(-1, num_experts)",
                    ["Ttok", "E"], ["Ttok", "E"], "activation",
                    "expand 后 reshape(-1, num_experts)（:170-175）：最终与 scores_for_choice 同型")
        self.op("g_moe_group_reshape", "Glm5NextTextTopkRouter.forward", "组掩码 reshape",
                "reshape", ["g_group_mask_expanded"], ["g_score_mask"], [], ["Ttok", "E"],
                [src("170-175", ".reshape(-1, self.num_experts)")],
                "3D [Ttok,Ng,E/Ng] → 2D [Ttok,E]；下游 masked_fill 必须同型")
        self.tensor("g_masked_scores", "scores_for_choice.masked_fill(~score_mask, -inf)",
                    ["Ttok", "E"], ["Ttok", "E"], "activation",
                    "未被选中的组置 -inf（:175）")
        self.op("g_moe_group_masked_fill", "Glm5NextTextTopkRouter.forward", "组外专家屏蔽",
                "masked_fill", ["g_scores_for_choice", "g_score_mask"], ["g_masked_scores"], [],
                ["Ng", "E"],
                [src("175", "scores_for_choice = scores_for_choice.masked_fill("
                            "~score_mask.bool(), float('-inf'))")],
                "Ng=1 时掩码全 True，不改变任何分数（等价于直接在 288 个专家里 top-8）")
        self.tensor("g_topk_indices", "topk_indices = topk(..., k=K)", ["Ttok", "K"],
                    ["Ttok", "K"], "activation", "选中的专家索引（:176）")
        self.op("g_moe_topk", "Glm5NextTextTopkRouter.forward", "TopK 专家选择", "topk",
                ["g_masked_scores"], ["g_topk_indices"], [], ["K", "E"],
                [src("176", "topk_indices = torch.topk(scores_for_choice, k=self.top_k, "
                            "dim=-1, sorted=False)[1]")],
                "每 token 选 K=8 个专家；排序无关（sorted=False）")
        self.tensor("g_topk_weights_raw", "topk_weights = scores.gather(1, indices)",
                    ["Ttok", "K"], ["Ttok", "K"], "activation", "取 sigmoid 原分数（:177）")
        self.op("g_moe_weight_gather", "Glm5NextTextTopkRouter.forward", "收集路由权重",
                "gather", ["g_router_scores", "g_topk_indices"], ["g_topk_weights_raw"], [],
                ["Ttok", "K"],
                [src("177", "topk_weights = scores.gather(1, topk_indices)")],
                "用未加偏置的 scores；偏置不影响权重数值")
        self.tensor("g_topk_weights_norm", "topk_weights /= sum + 1e-20", ["Ttok", "K"],
                    ["Ttok", "K"], "activation", "归一化（:178-180）")
        self.op("g_moe_weight_norm", "Glm5NextTextTopkRouter.forward", "路由权重归一化",
                "normalize", ["g_topk_weights_raw"], ["g_topk_weights_norm"], [], ["K"],
                [src("178-180", "if self.norm_topk_prob: topk_weights /= "
                                "topk_weights.sum(-1,keepdim=True)+1e-20")],
                "norm_topk_prob=true；分母 +1e-20 防零")
        self.tensor("g_topk_weights", "topk_weights = norm * routed_scaling_factor",
                    ["Ttok", "K"], ["Ttok", "K"], "activation", "乘 2.5（:181）")
        self.op("g_moe_weight_scale", "Glm5NextTextTopkRouter.forward", "路由权重缩放", "scale",
                ["g_topk_weights_norm"], ["g_topk_weights"], [], ["Scl"],
                [src("181", "topk_weights = topk_weights * self.routed_scaling_factor")],
                "缩放只作用于 routed 专家；共享专家另有独立路径")

        # experts（单份模板，不复制 288 份）
        self.weight("g_moe_experts_gate_up", "self.mlp.experts.gate_up_proj",
                    ["E", "2Es", "H"], ["E", "Es", "H"],
                    "专家权重以 3D 张量存储：gate_up_proj [E,2Es,H]（:115）")
        self.weight("g_moe_experts_down", "self.mlp.experts.down_proj",
                    ["E", "H", "Es"], ["E", "H", "Es"],
                    "专家权重以 3D 张量存储：down_proj [E,H,Es]（:116）")
        self.tensor("g_expert_mask", "expert_mask = one_hot(top_k_index,E+1).permute(2,1,0)",
                    ["E+1", "K", "Ttok"], ["E", "K", "Ttok"], "activation",
                    "one_hot(top_k_index, num_experts+1) 得 [Ttok,K,E+1]，permute(2,1,0) 得 "
                    "[E+1,K,Ttok]（:123-124）；最后一维 E 是 one_hot 的 no-expert 槽，循环里跳过")
        self.op("g_moe_expert_mask", "Glm5NextTextExperts.forward", "专家 one-hot 掩码",
                "one_hot+permute", ["g_topk_indices"], ["g_expert_mask"], [],
                ["E", "K", "Ttok"],
                [src("123-124", "with torch.no_grad(): expert_mask = "
                                "F.one_hot(top_k_index, num_experts+1); "
                                "expert_mask = expert_mask.permute(2, 1, 0)")],
                "保留完整的 [E+1,K,Ttok] 分配矩阵；E+1 的最后一槽在循环中 continue 跳过")
        self.tensor("g_expert_hit", "expert_hit（命中的专家索引，数据相关）", ["N_hit"],
                    ["N_hit"], "activation",
                    "expert_mask.sum(dim=(-1,-2))>0 后 nonzero 得命中专家索引（:125-126）；"
                    "命中集合本身是数据相关量，不默认 0/E")
        self.op("g_moe_expert_hit", "Glm5NextTextExperts.forward", "专家命中集合", "sum+nonzero",
                ["g_expert_mask"], ["g_expert_hit"], [], ["E", "N_hit"],
                [src("125-126", "hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()")],
                "只用于决定单专家模板循环哪些专家；N_hit 未给定，保持未知")
        self.tensor("g_expert_token_idx", "token_idx（命中该专家的 token 行，数据相关）",
                    ["N_e"], ["N_e"], "activation",
                    "torch.where(expert_mask[expert_idx]) 的第二个返回（:130）")
        self.tensor("g_expert_topk_pos", "top_k_pos（该 token 的第几个 top-k 槽）",
                    ["N_e"], ["N_e"], "activation",
                    "torch.where(expert_mask[expert_idx]) 的第一个返回（:130）")
        self.op("g_moe_expert_gather_indices", "Glm5NextTextExperts.forward",
                "命中专家 → (top_k_pos, token_idx)", "where",
                ["g_expert_mask", "g_expert_hit"],
                ["g_expert_topk_pos", "g_expert_token_idx"], [], ["N_e", "K", "Ttok"],
                [src("127-130", "for expert_idx in hit: top_k_pos, token_idx = "
                                "torch.where(mask[expert_idx])")],
                "对每个命中专家（单份模板）从完整分配矩阵取真实的 (top_k_pos, token_idx)")
        self.tensor("g_token_states", "hidden_states[token_idx]", ["N_e", "H"],
                    ["N_e", "H"], "activation", "按 token_idx gather 命中该专家的 token 行（:131）")
        self.op("g_moe_select_tokens", "Glm5NextTextExperts.forward", "选择专家 token 行",
                "gather", ["g_moe_flat", "g_expert_token_idx"], ["g_token_states"], [],
                ["N_e", "H"],
                [src("131", "current = self._apply_gate(F.linear(hidden_states[token_idx], "
                              "self.gate_up_proj[expert_idx]))")],
                "只 gather token_idx 指定的行；N_e 是数据相关量，不默认 0")
        self.tensor("g_gate_up", "F.linear(current_state, gate_up_proj[e])", ["N_e", "2Es"],
                    ["N_e", "Es"], "activation", "单专家 gate∪up 合并投影（:131）")
        self.op("g_moe_expert_gate_up", "Glm5NextTextExperts.forward", "专家 gate_up 投影",
                "linear", ["g_token_states"], ["g_gate_up"], ["g_moe_experts_gate_up"],
                ["H", "Es", "N_e", "E"],
                [src("131", "F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx])")],
                "gate/up 在权重里打包成 [E,2Es,H]；按专家索引取切片，不复制 E 份")
        self.tensor("g_expert_gate", "gate", ["N_e", "Es"], ["N_e", "Es"], "activation",
                    "chunk 前段（:137）")
        self.tensor("g_expert_up", "up", ["N_e", "Es"], ["N_e", "Es"], "activation",
                    "chunk 后段（:137）")
        self.op("g_moe_expert_chunk", "Glm5NextTextExperts.forward", "专家 gate/up 拆分",
                "chunk", ["g_gate_up"], ["g_expert_gate", "g_expert_up"], [], ["Es"],
                [src("136-137", "gate, up = gate_up.chunk(2, dim=-1)")],
                "沿最后一维均分；2Es=4096 → 两个 2048")
        self.tensor("g_expert_gate_clamped", "gate.clamp(max=Swi)", ["N_e", "Es"],
                    ["N_e", "Es"], "activation", "gate 只截上限 10")
        self.tensor("g_expert_up_clamped", "up.clamp(-Swi,Swi)", ["N_e", "Es"],
                    ["N_e", "Es"], "activation", "up 双向截断 ±10")
        self.op("g_moe_expert_clamp", "Glm5NextTextExperts._apply_gate", "专家 SwiGLU 截断",
                "clamp", ["g_expert_gate", "g_expert_up"],
                ["g_expert_gate_clamped", "g_expert_up_clamped"], [], ["Swi", "Es"],
                [src("136-141", "gate.clamp(min=None, max=swiglu_limit); "
                                "up.clamp(min=-swiglu_limit, max=swiglu_limit)")],
                "SwiGLU 截断是本模型 eager 路径的真实算子；不能省略或套用无 clamp 的 MLP")
        self.tensor("g_expert_hidden", "silu(gate.clamp)*up.clamp", ["N_e", "Es"],
                    ["N_e", "Es"], "activation", "SiLU 门控（:141）")
        self.op("g_moe_expert_act", "Glm5NextTextExperts._apply_gate", "专家 SiLU 门控",
                "silu*mul", ["g_expert_gate_clamped", "g_expert_up_clamped"],
                ["g_expert_hidden"], [], ["Es"],
                [src("141", "return F.silu(gate) * up")],
                "clamp 后才做 silu(gate)*up")
        self.tensor("g_expert_out", "F.linear(current, down_proj[e])", ["N_e", "H"],
                    ["N_e", "H"], "activation", "单专家 down 投影（:132）")
        self.op("g_moe_expert_down", "Glm5NextTextExperts.forward", "专家 down 投影", "linear",
                ["g_expert_hidden"], ["g_expert_out"], ["g_moe_experts_down"],
                ["Es", "H", "N_e", "E"],
                [src("132", "current = F.linear(current, self.down_proj[expert_idx]) * "
                            "top_k_weights[token_idx, top_k_pos, None]")],
                "按专家索引取 [H,Es] 切片")
        self.tensor("g_expert_weighted", "current * top_k_weights[token_idx, top_k_pos]",
                    ["N_e", "H"], ["N_e", "H"], "activation",
                    "按 (token_idx, top_k_pos) 取该 token 对该专家的路由权重后缩放（:132）")
        self.op("g_moe_expert_weight", "Glm5NextTextExperts.forward", "专家输出加权", "scale",
                ["g_expert_out", "g_topk_weights", "g_expert_token_idx", "g_expert_topk_pos"],
                ["g_expert_weighted"], [], ["N_e", "K"],
                [src("132", "* top_k_weights[token_idx, top_k_pos, None]")],
                "权重索引同时用 token_idx 与 top_k_pos：只拿命中专家集合无法确定该乘哪个槽")
        self.tensor("g_routed_init", "final_hidden_states = zeros_like(hidden_states)",
                    ["Ttok", "H"], ["Ttok", "H"], "activation",
                    "index_add 的累加初值（:122）")
        self.op("g_moe_result_init", "Glm5NextTextExperts.forward", "专家输出累加初值",
                "zeros_like", ["g_moe_flat"], ["g_routed_init"], [], ["Ttok", "H"],
                [src("122", "final = torch.zeros_like(hidden_states)")],
                "按展平 token 形状建零张量；后续按 token_idx 累加")
        self.tensor("g_routed_flat", "final_hidden_states（index_add 后）", ["Ttok", "H"],
                    ["Ttok", "H"], "activation", "所有专家输出按 token 累加（:133）")
        self.op("g_moe_index_add", "Glm5NextTextExperts.forward", "专家输出累加",
                "index_add", ["g_routed_init", "g_expert_weighted", "g_expert_token_idx"],
                ["g_routed_flat"], [], ["Ttok", "H"],
                [src("133", "final.index_add_(0, token_idx, current.to(final.dtype))")],
                "按 token_idx 把每个专家的加权输出加回对应 token 行")
        self.tensor("g_routed", "hidden_states（view 回 [B,seqlen,H]）", ["B", "seqlen", "H"],
                    ["B", "seqlen", "H"], "activation", "routed 专家输出（:204）")
        self.op("g_moe_routed_view", "Glm5NextTextMoE.forward", "Routed 输出还原", "view",
                ["g_routed_flat"], ["g_routed"], [], ["B", "seqlen", "H"],
                [src("204", "self.experts(...).view(*orig_shape)")],
                "展平的 token 行还原成 [B,seqlen,H]")

        # shared experts（与 routed 并行）
        self.weight("g_moe_shared_gate_weight", "self.mlp.shared_experts.gate_proj.weight",
                    ["Es", "H"], ["Es", "H"], "共享专家 gate_proj（宽度 Es×n_shared_experts=2048）")
        self.weight("g_moe_shared_up_weight", "self.mlp.shared_experts.up_proj.weight",
                    ["Es", "H"], ["Es", "H"], "共享专家 up_proj")
        self.weight("g_moe_shared_down_weight", "self.mlp.shared_experts.down_proj.weight",
                    ["H", "Es"], ["H", "Es"], "共享专家 down_proj")
        self.tensor("g_shared_gate", "shared_experts.gate_proj(residuals)",
                    ["B", "seqlen", "Es"], ["B", "seqlen", "Es"], "activation",
                    "共享专家输入是 routed 之前的 residuals（:200/:205）")
        self.op("g_moe_shared_gate", "Glm5NextTextMoE.shared_experts", "共享专家 gate_proj",
                "linear", [ffn_in], ["g_shared_gate"], ["g_moe_shared_gate_weight"],
                ["H", "Es", "B", "seqlen"],
                [src("199-200", "residuals = hidden_states"),
                 src("205", "hidden_states = hidden_states + self.shared_experts(residuals)")],
                "共享专家宽度 Es×n_shared_experts=2048；与 routed 路径并行")
        self.tensor("g_shared_up", "shared_experts.up_proj(residuals)",
                    ["B", "seqlen", "Es"], ["B", "seqlen", "Es"], "activation",
                    "共享专家 up 投影（:98-99）")
        self.op("g_moe_shared_up", "Glm5NextTextMoE.shared_experts", "共享专家 up_proj",
                "linear", [ffn_in], ["g_shared_up"], ["g_moe_shared_up_weight"],
                ["H", "Es", "B", "seqlen"],
                [src("97-103", "Glm5NextTextMLP.forward：gate/up + clamp")],
                "与 gate 同一输入 residuals")
        self.tensor("g_shared_gate_clamped", "gate.clamp(max=Swi)", ["B", "seqlen", "Es"],
                    ["B", "seqlen", "Es"], "activation", "共享专家 gate 截上限 10")
        self.tensor("g_shared_up_clamped", "up.clamp(-Swi,Swi)", ["B", "seqlen", "Es"],
                    ["B", "seqlen", "Es"], "activation", "共享专家 up 双向截断 ±10")
        self.op("g_moe_shared_clamp", "Glm5NextTextMoE.shared_experts", "共享专家 SwiGLU 截断",
                "clamp", ["g_shared_gate", "g_shared_up"],
                ["g_shared_gate_clamped", "g_shared_up_clamped"], [], ["Swi", "Es"],
                [src("97-103", "gate = gate.clamp(min=None, max=swiglu_limit); "
                               "up = up.clamp(min=-swiglu_limit, max=swiglu_limit)")],
                "共享专家与专家/稠密 MLP 共用同一 clamp 规则")
        self.tensor("g_shared_act", "silu(gate.clamp)*up.clamp", ["B", "seqlen", "Es"],
                    ["B", "seqlen", "Es"], "activation", "SiLU 门控（:103）")
        self.op("g_moe_shared_act", "Glm5NextTextMoE.shared_experts", "共享专家 SiLU 门控",
                "silu*mul", ["g_shared_gate_clamped", "g_shared_up_clamped"],
                ["g_shared_act"], [], ["Es"],
                [src("103", "self.down_proj(self.act_fn(gate) * up)")],
                "与 dense/专家相同的 MLP 语义，只是宽度不同")
        self.tensor("g_shared_out", "shared_experts(residuals)", ["B", "seqlen", "H"],
                    ["B", "seqlen", "H"], "activation", "共享专家输出（:205）")
        self.op("g_moe_shared_down", "Glm5NextTextMoE.shared_experts", "共享专家 down_proj",
                "linear", ["g_shared_act"], ["g_shared_out"], ["g_moe_shared_down_weight"],
                ["Es", "H", "B", "seqlen"],
                [src("93", "self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)")],
                "回投影到 hidden_size")
        self.tensor("g_mlp_out", "mlp_out（本组 MLP/MoE 输出）", ["B", "seqlen", "H"],
                    ["B", "seqlen", "H"], "activation", "MoE 汇合（:205-206）")
        self.op("g_moe_merge", "Glm5NextTextMoE.forward", "Routed + 共享专家汇合", "add",
                ["g_routed", "g_shared_out"], ["g_mlp_out"], [], ["B", "seqlen", "H"],
                [src("205", "hidden_states = hidden_states + self.shared_experts(residuals)")],
                "共享专家是并行支路，不参与路由；两者直接相加")

    def _dense_ops(self, ffn_in: str) -> None:
        src = self.src
        self.weight("g_dense_gate_weight", "self.mlp.gate_proj.weight", ["Is", "H"],
                    ["Is", "H"], "dense MLP gate_proj（:91）")
        self.weight("g_dense_up_weight", "self.mlp.up_proj.weight", ["Is", "H"],
                    ["Is", "H"], "dense MLP up_proj（:92）")
        self.weight("g_dense_down_weight", "self.mlp.down_proj.weight", ["H", "Is"],
                    ["H", "Is"], "dense MLP down_proj（:93）")
        self.tensor("g_dense_gate", "self.mlp.gate_proj(x)", ["B", "seqlen", "Is"],
                    ["B", "seqlen", "Is"], "activation", "dense gate 投影（:98）")
        self.op("g_dense_gate", "Glm5NextTextMLP.gate_proj", "Dense gate_proj", "linear",
                [ffn_in], ["g_dense_gate"], ["g_dense_gate_weight"],
                ["H", "Is", "B", "seqlen"],
                [src("91", "self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)")],
                "dense 层用 intermediate_size=12288（不是专家宽度 2048）")
        self.tensor("g_dense_up", "self.mlp.up_proj(x)", ["B", "seqlen", "Is"],
                    ["B", "seqlen", "Is"], "activation", "dense up 投影（:99）")
        self.op("g_dense_up", "Glm5NextTextMLP.up_proj", "Dense up_proj", "linear",
                [ffn_in], ["g_dense_up"], ["g_dense_up_weight"],
                ["H", "Is", "B", "seqlen"],
                [src("92", "self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)")],
                "no bias")
        self.tensor("g_dense_gate_clamped", "gate.clamp(max=Swi)", ["B", "seqlen", "Is"],
                    ["B", "seqlen", "Is"], "activation", "gate 只截上限 10")
        self.tensor("g_dense_up_clamped", "up.clamp(-Swi,Swi)", ["B", "seqlen", "Is"],
                    ["B", "seqlen", "Is"], "activation", "up 双向截断 ±10")
        self.op("g_dense_clamp", "Glm5NextTextMLP.forward", "Dense SwiGLU 截断", "clamp",
                ["g_dense_gate", "g_dense_up"],
                ["g_dense_gate_clamped", "g_dense_up_clamped"], [], ["Swi", "Is"],
                [src("100-102", "gate = gate.clamp(min=None, max=self.swiglu_limit); "
                                "up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)")],
                "SwiGLU 截断是本模型 eager 路径的真实算子；不能省略或套用无 clamp 的 MLP")
        self.tensor("g_dense_act", "act_fn(gate.clamp) * up.clamp", ["B", "seqlen", "Is"],
                    ["B", "seqlen", "Is"], "activation", "SiLU(gate)×up（:103）")
        self.op("g_dense_act", "Glm5NextTextMLP.forward", "SiLU 门控相乘", "silu*mul",
                ["g_dense_gate_clamped", "g_dense_up_clamped"], ["g_dense_act"], [], ["Is"],
                [src("103", "return self.down_proj(self.act_fn(gate) * up)")],
                "clamp 后才做 silu(gate)*up")
        self.tensor("g_mlp_out", "mlp_out（本组 MLP/MoE 输出）", ["B", "seqlen", "H"],
                    ["B", "seqlen", "H"], "activation", "dense MLP 输出（:103）")
        self.op("g_dense_down", "Glm5NextTextMLP.down_proj", "Dense down_proj", "linear",
                ["g_dense_act"], ["g_mlp_out"], ["g_dense_down_weight"],
                ["Is", "H", "B", "seqlen"],
                [src("93", "self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)")],
                "回投影到 hidden_size")


class _KdaBlockBuilder(_BlockBuilder):
    """KDA（Kimi Delta Attention）+ dense/MoE 的真实单层结构。"""

    kind = "kda"

    def __init__(self, values: HfModelValues, layer_idx: int, mlp_type: str):
        super().__init__(values, layer_idx)
        self.mlp_type = mlp_type

    def build(self) -> dict:
        v = self.values
        title = (
            f"GLM-5.3-Flash Decoder Block（layer_idx={self.layer_idx}，KDA + "
            f"{self.mlp_type}；{v.num_hidden_layers} 层主干代表模板；同一结构由 "
            "seqlen/kvcachelen 场景参数化）"
        )
        self._build_parameters()
        self._build_derivations()
        self._build_boundaries_and_cache()
        self._build_flow()
        self._build_cache_notes()
        self._build_coverage_notes()
        return {
            "id": "block",
            "title": title,
            "parameters": self.parameters,
            "tensors": self.tensors,
            "operators": self.operators,
            "derivations": self.derivations,
            "cache_notes": self.cache_notes,
            "coverage_notes": self.coverage_notes,
        }

    def _build_parameters(self) -> None:
        v = self.values
        cfg = self.cfg
        self._common_parameters()
        self.param("NhL", "NhL", "linear_num_heads KDA 头数", v.linear_num_heads, None,
                   "config.linear_attn_config.num_heads=64；KDA 的 q/k/v 每个 8192 维",
                   [cfg("text_config.linear_attn_config", "文本子配置"),
                    self.src("630-632", "self.num_heads / self.head_dim / self.qkv_dim")])
        self.param("DhL", "DhL", "linear_head_dim KDA 头维", v.linear_head_dim, None,
                   "config.linear_attn_config.head_dim=128；qkv_dim=64×128",
                   [cfg("text_config.linear_attn_config", "文本子配置"),
                    self.src("630-632", "self.head_dim = config.linear_head_dim")])
        self.param("Kconv", "Kconv", "linear_conv_kernel_dim 短卷积核", v.linear_conv_kernel_dim,
                   None, "config.linear_attn_config.short_conv_kernel_size=4；depthwise causal conv",
                   [cfg("text_config.linear_attn_config", "文本子配置"),
                    self.src("634-651", "conv_dim=3*qkv_dim; Conv1d(kernel=4, groups=conv_dim)")])
        self.param("GateLB", "GateLB", "KDA forget gate 下界", v.linear_lower_bound, None,
                   "config.linear_attn_config.gate_lower_bound=-5.0；g=-5*sigmoid(exp(A_log)*g_raw)",
                   [cfg("text_config.linear_attn_config", "文本子配置"),
                    self.src("362-363", "return self.safe_gate_lower_bound * "
                                        "torch.sigmoid(decay_rate * g)")])
        self.param("Srec", "S_rec", "单层 recurrent state 元素数（每 batch）", None,
                   "S_rec = NhL × DhL × DhL",
                   "KDA 的 delta-rule 状态 [B,64,128,128]=1,048,576 个值；"
                   "大小固定，不随 kvcachelen 增长",
                   [self.src("492-496", "last_recurrent_state = zeros(B, num_heads, k_head_dim, v_head_dim)"),
                    self.srcf("cache_utils.py", "1067-1149", "LinearAttentionLayer.recurrent_states")])
        self.param("Sconv", "S_conv", "单层 conv state 元素数（每 batch）", None,
                   "S_conv = 3 × NhL × DhL × Kconv",
                   "conv state [B,3×8192,4]=98,304 个值；固定大小",
                   [self.src("643-651", "conv_dim=3*qkv_dim; kernel_size=4"),
                    self.srcf("cache_utils.py", "1099-1149", "update_conv_state/full_conv_states")])
        if self.mlp_type == "sparse":
            self.param("N_hit", "N_hit", "路由实际命中的不同专家数（数据相关）", None, None,
                       "数据相关量，未给定；不根据权重推断 MoE 实际路由，不默认 0 或 E",
                       [self.src("125-126", "one_hot(...).sum→nonzero：命中专家集合")],
                       unknown_reason="数据相关：路由命中集合未给定；不默认 0/E")
            self.param("N_e", "N_e", "单专家在本调用处理的 token 数（数据相关）", None, None,
                       "数据相关量，未给定；专家模板只有一份，不按 E 复制行",
                       [self.src("130", "torch.where(expert_mask[expert_idx])")],
                       unknown_reason="数据相关：每专家 token 数未给定；不默认 0")

    def _build_derivations(self) -> None:
        super()._build_derivations()
        premise = ("按固定配置 + 「空态起步、连续正常执行」前提推导；"
                   "KDA 状态大小固定，与 kvcachelen 无关")
        self.derivation("S_rec", "单层 recurrent state 元素数", "NhL * DhL * DhL",
                        f"64×128×128=1,048,576；每 batch 固定。{premise}")
        self.derivation("S_conv", "单层 conv state 元素数", "3 * NhL * DhL * Kconv",
                        f"3×64×128×4=98,304；每 batch 固定。{premise}")

    def _build_boundaries_and_cache(self) -> None:
        self.tensor(
            "g_hidden_streams_in", "hidden_states（4 路流层输入）", ["B", "seqlen", "Hc", "H"],
            ["B", "seqlen", "Hc", "H"], "boundary",
            "上一层输出或网络入口的 embedding 广播（modeling:1493）；DecoderLayer.forward "
            "的 hidden_states（:1312）")
        self.tensor(
            "g_attention_mask", "attention_mask（2D padding mask）", ["B", "seqlen"],
            ["B", "seqlen"], "boundary",
            "create_recurrent_attention_mask 的 [B,seqlen] padding mask；None 时网络入口补 "
            "ones（modeling:1471-1486）。不是 4D 因果 mask")
        self.tensor(
            "g_kda_conv_state_pre", "past_key_values.layers[layer_idx].conv_states（调用前）",
            ["B", "3×NhL×DhL", "Kconv"], ["B", "NhL", "DhL", "Kconv"], "cache",
            "LinearAttentionLayer 的 conv state；保存最近 Kconv 个 token 的 qkv 通道；"
            "首次调用按需左补零（cache_utils:1099-1149）")
        self.tensor(
            "g_kda_conv_state_post", "self.conv_states（调用后）",
            ["B", "3×NhL×DhL", "Kconv"], ["B", "NhL", "DhL", "Kconv"], "cache",
            "update_conv_state 把当前调用尾部写回同一静态地址；长度固定 Kconv")
        self.tensor(
            "g_kda_state_pre", "past_key_values.layers[layer_idx].recurrent_states（调用前）",
            ["B", "NhL", "DhL", "DhL"], ["B", "NhL", "DhL", "DhL"], "cache",
            "KDA delta-rule 的 [B,64,128,128] 状态；空缓存时由 kernel 以 zeros 起步")
        self.tensor(
            "g_kda_state_post", "self.recurrent_states（调用后）",
            ["B", "NhL", "DhL", "DhL"], ["B", "NhL", "DhL", "DhL"], "cache",
            "update_recurrent_state 写入 fp32 的新状态（modeling:761-762）；"
            "大小固定，不随历史长度增长")

    def _build_flow(self) -> None:
        v = self.values
        src, cfg = self.src, self.cfg
        # mHC 权重（每个站点独立实例）
        self._mhc_site("attn", "g_hidden_streams_in",
                       residual_note="坍缩后的单流进入 input_layernorm；mHC 的残差在 Attention "
                                     "之后由 post/comb 混合，不是普通相加")
        self.weight("g_input_norm_weight", "self.input_layernorm.weight", ["H"], ["H"],
                    "RMSNorm 可学习缩放（无 bias），eps=rms_norm_eps（:1293）")
        self.weight("g_post_norm_weight", "self.post_attention_layernorm.weight", ["H"], ["H"],
                    "RMSNorm 可学习缩放（无 bias），eps=rms_norm_eps（:1294）")
        self.tensor("g_attn_normed", "hidden_states = self.input_layernorm(collapsed)",
                    ["B", "seqlen", "H"], ["B", "seqlen", "H", "EpsN"], "activation",
                    "RMSNorm：fp32 方差归一后转回输入 dtype，再乘 weight（:64-79/:1315）")
        self.op("g_input_norm", "Glm5NextTextDecoderLayer.input_layernorm", "Input RMSNorm",
                "rmsnorm", ["g_mhc_attn_collapsed"], ["g_attn_normed"],
                ["g_input_norm_weight"], ["H", "EpsN"],
                [src("1315", "hidden_states = self.input_layernorm(hidden_states)"),
                 src("64-79", "Glm5NextTextRMSNorm.forward")],
                "层输入先归一化再进入 Attention；残差保留 mHC 的 4 路流，不是单流")

        # 0) padding mask 置零（KDA 专有）
        self.tensor("g_kda_masked_input", "apply_mask_to_padding_states(hidden, mask)",
                    ["B", "seqlen", "H"], ["B", "seqlen", "H"], "activation",
                    "padding 位置乘 0；全 1 mask 时是 no-op，但源码仍执行（:672）")
        self.op("g_kda_mask_apply", "Glm5NextTextLinearAttention.forward",
                "KDA padding 置零", "broadcast-mul",
                ["g_attn_normed", "g_attention_mask"], ["g_kda_masked_input"], [],
                ["B", "seqlen", "H"],
                [src("671-672", "hidden_states = apply_mask_to_padding_states(hidden_states, "
                                "attention_mask)"),
                 src("396-405", "apply_mask_to_padding_states")],
                "只处理 padding，不是因果 mask；KDA 的因果性来自短卷积+递推顺序")

        # 1) Q/K/V 投影 + 短卷积
        self.weight("g_kda_q_proj_weight", "self.self_attn.q_proj.weight", ["NhL×DhL", "H"],
                    ["NhL", "DhL", "H"], "nn.Linear(H, NhL*DhL, bias=False)（:639）")
        self.weight("g_kda_k_proj_weight", "self.self_attn.k_proj.weight", ["NhL×DhL", "H"],
                    ["NhL", "DhL", "H"], "nn.Linear(H, NhL*DhL, bias=False)（:640）")
        self.weight("g_kda_v_proj_weight", "self.self_attn.v_proj.weight", ["NhL×DhL", "H"],
                    ["NhL", "DhL", "H"], "nn.Linear(H, NhL*DhL, bias=False)（:641）")
        for name in ("q", "k", "v"):
            self.tensor(f"g_kda_{name}_flat", f"self.{name}_proj(hidden_states)",
                        ["B", "seqlen", "NhL×DhL"], ["B", "seqlen", "NhL", "DhL"],
                        "activation", f"{name} 投影输出 [B,S,8192]（:639-641）")
            self.op(f"g_kda_{name}_proj", f"Glm5NextTextLinearAttention.{name}_proj",
                    f"KDA {name.upper()} 投影", "linear",
                    ["g_kda_masked_input"], [f"g_kda_{name}_flat"],
                    [f"g_kda_{name}_proj_weight"], ["H", "NhL", "DhL", "B", "seqlen"],
                    [src("639-641", "self.q_proj/k_proj/v_proj = nn.Linear(H, qkv_dim, bias=False)"),
                     src("678-685", "torch.cat([q_proj, k_proj, v_proj], -1).transpose(1,2)")],
                    "三路投影后沿最后一维拼接；qkv_dim=8192")
        self.tensor("g_kda_qkv", "torch.cat([q,k,v], -1)", ["B", "seqlen", "3×NhL×DhL"],
                    ["B", "seqlen", "NhL", "DhL"], "activation",
                    "拼接后的 [B,S,24576]；随后 transpose 成 [B,24576,S] 供 Conv1d")
        self.op("g_kda_qkv_cat", "Glm5NextTextLinearAttention.forward", "KDA QKV 拼接", "concat",
                ["g_kda_q_flat", "g_kda_k_flat", "g_kda_v_flat"], ["g_kda_qkv"], [],
                ["NhL", "DhL", "B", "seqlen"],
                [src("678-685", "torch.cat([...], dim=-1).transpose(1, 2)")],
                "沿特征维拼接；order 固定 q/k/v")
        self.tensor("g_kda_qkv_t", "mixed_qkv.transpose(1,2)", ["B", "3×NhL×DhL", "seqlen"],
                    ["B", "NhL", "DhL", "seqlen"], "activation",
                    "Conv1d 需要 [B, channels, seq]；这一步是必要布局变换")
        self.op("g_kda_qkv_transpose", "Glm5NextTextLinearAttention.forward",
                "KDA QKV 转置", "transpose", ["g_kda_qkv"], ["g_kda_qkv_t"], [],
                ["B", "seqlen", "NhL", "DhL"],
                [src("678-685", ".transpose(1, 2)")],
                "把序列维移到最后一维；Conv1d 的 depthwise 通道对应 3×8192")
        self.weight("g_kda_conv_weight", "self.self_attn.conv1d.weight",
                    ["3×NhL×DhL", "1", "Kconv"], ["NhL", "DhL", "Kconv"],
                    "depthwise Conv1d weight [conv_dim,1,Kconv]（:644-651）；"
                    "bias=False")
        self.tensor("g_kda_conv_out", "causal_conv1d_fn / update(...)",
                    ["B", "3×NhL×DhL", "seqlen"], ["B", "NhL", "DhL", "seqlen"], "activation",
                    "因果短卷积输出（SiLU 激活），只取最后 seqlen 步（:694-719）")
        self.op("g_kda_conv_state_update", "LinearAttentionLayer.update_conv_state",
                "KDA conv state 写入", "cache_update",
                ["g_kda_conv_state_pre", "g_kda_qkv_t"], ["g_kda_conv_state_post"], [],
                ["B", "NhL", "DhL", "Kconv", "seqlen"],
                [src("703-708", "cache_params.update_conv_state(mixed_qkv, layer_idx, "
                                "conv_kernel_size=Kconv)"),
                 self.srcf("cache_utils.py", "1099-1149", "update_conv_state：首次左补零/续接 concat")],
                "源码把 update 返回的 full states 直接喂给卷积；本图把缓存写与卷积输入"
                "分开表达，避免把「前态+本次」误画成持久缓存")
        self.op("g_kda_conv1d", "Glm5NextTextLinearAttention.forward", "KDA 因果短卷积",
                "conv1d", ["g_kda_conv_state_pre", "g_kda_qkv_t"], ["g_kda_conv_out"],
                ["g_kda_conv_weight"], ["NhL", "DhL", "Kconv", "seqlen"],
                [src("694-719", "causal_conv1d_update（单 token）/ causal_conv1d_fn（prefill）"),
                 src("408-448", "depthwise Conv1d + SiLU；padding=Kconv-1 后 slice 最后 seqlen")],
                "depthwise、causal（只看左侧 Kconv-1 个历史 + 当前）；全 1 mask 时 padding 不")
        self.tensor("g_kda_qkv_seq", "mixed_qkv.transpose(1,2)（卷积后）",
                    ["B", "seqlen", "3×NhL×DhL"], ["B", "seqlen", "NhL", "DhL"], "activation",
                    "转回 [B,S,24576] 后 split q/k/v（:721-729）")
        self.op("g_kda_qkv_view", "Glm5NextTextLinearAttention.forward",
                "KDA 卷积输出布局", "transpose", ["g_kda_conv_out"], ["g_kda_qkv_seq"], [],
                ["B", "seqlen", "NhL", "DhL"],
                [src("721-725", "mixed_qkv.transpose(1, 2); torch.split([...], qkv_dim)")],
                "把通道维放回最后一维，随后按 8192 均分")
        self.tensor("g_kda_q", "query", ["B", "seqlen", "NhL", "DhL"],
                    ["B", "seqlen", "NhL", "DhL"], "activation", "q view（:727）")
        self.tensor("g_kda_k", "key", ["B", "seqlen", "NhL", "DhL"],
                    ["B", "seqlen", "NhL", "DhL"], "activation", "k view（:728）")
        self.tensor("g_kda_v", "value", ["B", "seqlen", "NhL", "DhL"],
                    ["B", "seqlen", "NhL", "DhL"], "activation", "v view（:729）")
        self.op("g_kda_qkv_split", "Glm5NextTextLinearAttention.forward", "KDA QKV 拆分",
                "split", ["g_kda_qkv_seq"], ["g_kda_q", "g_kda_k", "g_kda_v"], [],
                ["NhL", "DhL"],
                [src("721-729", "torch.split(..., [qkv_dim]*3, -1); view(hidden_shape)")],
                "q/k/v 都是 [B,S,64,128]")

        # 2) forget gate / beta
        self.weight("g_kda_f_a_weight", "self.self_attn.forget_gate.f_a_proj.weight",
                    ["DhL", "H"], ["DhL", "H"], "forget gate 低秩投影（:346）")
        self.weight("g_kda_f_b_weight", "self.self_attn.forget_gate.f_b_proj.weight",
                    ["NhL×DhL", "DhL"], ["NhL", "DhL", "DhL"], "forget gate 上投影（:347）")
        self.weight("g_kda_dt_bias", "self.self_attn.forget_gate.dt_bias", ["NhL×DhL"],
                    ["NhL", "DhL"], "forget gate 的 dt bias（:348）")
        self.weight("g_kda_a_log", "self.self_attn.forget_gate.A_log", ["NhL"], ["NhL"],
                    "每头 A_log；decay_rate=exp(A_log)（:349/:359）")
        self.tensor("g_kda_f_a", "f_a_proj(hidden)", ["B", "seqlen", "DhL"],
                    ["B", "seqlen", "DhL"], "activation", "低秩 forget gate（:356）")
        self.op("g_kda_f_a_proj", "Glm5NextTextForgetGate.f_a_proj", "KDA forget gate 低秩",
                "linear", ["g_kda_masked_input"], ["g_kda_f_a"], ["g_kda_f_a_weight"],
                ["H", "DhL", "B", "seqlen"],
                [src("346", "self.f_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)"),
                 src("356", "forget_gate = self.f_b_proj(self.f_a_proj(hidden_states))")],
                "先降到 128 维，再升到 8192")
        self.tensor("g_kda_f_b", "f_b_proj(f_a)", ["B", "seqlen", "NhL×DhL"],
                    ["B", "seqlen", "NhL", "DhL"], "activation", "forget gate logits（:356）")
        self.op("g_kda_f_b_proj", "Glm5NextTextForgetGate.f_b_proj", "KDA forget gate 上投影",
                "linear", ["g_kda_f_a"], ["g_kda_f_b"], ["g_kda_f_b_weight"],
                ["DhL", "NhL", "DhL", "B", "seqlen"],
                [src("347", "self.f_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)"),
                 src("356", "forget_gate = self.f_b_proj(self.f_a_proj(hidden_states))")],
                "输出 [B,S,8192]；随后加 dt_bias 并 view [B,S,64,128]")
        self.tensor("g_kda_g_raw", "forget_gate + dt_bias", ["B", "seqlen", "NhL×DhL"],
                    ["B", "seqlen", "NhL", "DhL"], "activation",
                    "float32 加法后 view（:357）")
        self.op("g_kda_dt_add", "Glm5NextTextForgetGate.forward", "KDA dt_bias 加法", "add",
                ["g_kda_f_b", "g_kda_dt_bias"], ["g_kda_g_raw"], [], ["NhL", "DhL"],
                [src("357", "g = (forget_gate.float() + self.dt_bias.float().view(1,1,-1))"
                            ".view(hidden_shape)")],
                "dt_bias 是每通道偏置；计算在 float32")
        self.tensor("g_kda_g", "g = -5*sigmoid(exp(A_log)*g_raw)", ["B", "seqlen", "NhL", "DhL"],
                    ["B", "seqlen", "NhL", "DhL"], "activation",
                    "每 token/头/通道的 log decay，范围 (-5,0)（:359-363）")
        self.op("g_kda_decay", "Glm5NextTextForgetGate.forward", "KDA forget gate 衰减",
                "sigmoid+scale", ["g_kda_g_raw"], ["g_kda_g"], ["g_kda_a_log"],
                ["NhL", "DhL", "GateLB"],
                [src("358-363", "decay_rate = torch.exp(A_log); "
                                "safe_gate_lower_bound * torch.sigmoid(decay_rate * g)")],
                "safe_gate_lower_bound=-5.0（linear_attn_config.gate_lower_bound）；"
                "输出 ≤0 才能作为 exp 的 log decay")
        self.weight("g_kda_b_proj_weight", "self.self_attn.b_proj.weight", ["NhL", "H"],
                    ["NhL", "H"], "beta 投影（:654）")
        self.tensor("g_kda_beta_logits", "self.b_proj(hidden)", ["B", "seqlen", "NhL"],
                    ["B", "seqlen", "NhL"], "activation", "beta logits（:733）")
        self.op("g_kda_b_proj", "Glm5NextTextLinearAttention.b_proj", "KDA beta 投影", "linear",
                ["g_kda_masked_input"], ["g_kda_beta_logits"], ["g_kda_b_proj_weight"],
                ["H", "NhL", "B", "seqlen"],
                [src("654", "self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)"),
                 src("733", "beta = torch.sigmoid(self.b_proj(hidden_states))")],
                "每头一个 beta；sigmoid 后是 delta-rule 的写入门控")
        self.tensor("g_kda_beta", "beta = sigmoid(b_proj(x))", ["B", "seqlen", "NhL"],
                    ["B", "seqlen", "NhL"], "activation", "范围 (0,1)（:733）")
        self.op("g_kda_beta", "Glm5NextTextLinearAttention.forward", "KDA beta sigmoid",
                "sigmoid", ["g_kda_beta_logits"], ["g_kda_beta"], [], ["NhL"],
                [src("733", "beta = torch.sigmoid(self.b_proj(hidden_states))")],
                "beta 控制新 value 写入状态的强度")

        # 3) l2norm / scale / scan
        self.tensor("g_kda_q_hat", "l2norm(query)", ["B", "seqlen", "NhL", "DhL"],
                    ["B", "seqlen", "NhL", "DhL"], "activation",
                    "kernel 内 use_qk_l2norm_in_kernel=True，在 fp32 上 L2 归一（:536-539）")
        self.op("g_kda_q_l2", "chunk_kimi_delta_attention / recurrent", "KDA Q L2Norm",
                "l2norm", ["g_kda_q"], ["g_kda_q_hat"], [], ["DhL"],
                [src("479-481", "if use_qk_l2norm_in_kernel: query = l2norm(query)"),
                 src("451-459", "l2norm：x/sqrt(sum(x^2)+1e-6)")],
                "L2 归一在 kernel 内、fp32 上执行；不是 RMSNorm")
        self.tensor("g_kda_k_hat", "l2norm(key)", ["B", "seqlen", "NhL", "DhL"],
                    ["B", "seqlen", "NhL", "DhL"], "activation", "L2 归一后的 key（:536-539）")
        self.op("g_kda_k_l2", "chunk_kimi_delta_attention / recurrent", "KDA K L2Norm",
                "l2norm", ["g_kda_k"], ["g_kda_k_hat"], [], ["DhL"],
                [src("479-481", "key = l2norm(key)"), src("451-459", "l2norm")],
                "q/k 都归一；缩放只作用于 q（:486-487）")
        self.tensor("g_kda_q_scaled", "q * (DhL**-0.5)", ["B", "seqlen", "NhL", "DhL"],
                    ["B", "seqlen", "NhL", "DhL"], "activation",
                    "query 乘 1/sqrt(128)（:486-487）")
        self.op("g_kda_q_scale", "recurrent_kimi_delta_attention / chunk", "KDA Q 缩放",
                "scale", ["g_kda_q_hat"], ["g_kda_q_scaled"], [], ["DhL"],
                [src("486-487", "scale = 1 / (query.shape[-1] ** 0.5); query = query * scale")],
                "缩放是 128^-0.5；不是 DSA 的 256^-0.5")
        self.tensor("g_kda_core_out", "core_attn_out（扫描输出）",
                    ["B", "seqlen", "NhL", "DhL"], ["B", "seqlen", "NhL", "DhL"], "activation",
                    "delta-rule 扫描输出；recurrent 仅在 cache 已有状态且 seq_len==1 时走，"
                    "空缓存首次调用与多 token 续接走 chunk 路径（chunk_size=64）")
        self.tensor("g_kda_state_new", "last_recurrent_state（本次扫描结果）",
                    ["B", "NhL", "DhL", "DhL"], ["B", "NhL", "DhL", "DhL"], "activation",
                    "本次扫描产生的新状态（fp32），随后写入 cache")
        self.op("g_kda_scan", "recurrent_kimi_delta_attention / chunk_kimi_delta_attention",
                "KDA delta-rule 扫描（递归/分块）", "scan",
                ["g_kda_q_scaled", "g_kda_k_hat", "g_kda_v", "g_kda_g", "g_kda_beta",
                 "g_kda_state_pre"],
                ["g_kda_core_out", "g_kda_state_new"], [], ["NhL", "DhL", "seqlen", "B"],
                [src("462-513", "recurrent_kimi_delta_attention：逐 token 状态更新"),
                 src("516-614", "chunk_kimi_delta_attention：chunk 64 的分块等价形式"),
                 src("735-759", "根据 use_precomputed_states and seq_len==1 选择 recurrent/chunk")],
                "真实语义（delta-rule）：先衰减 A = S_{t-1} ⊙ exp(g_t)；"
                "mem = A^T k_t；delta = beta_t ⊙ (v_t - mem)；"
                "S_t = A + k_t delta^T；o_t = S_t^T q_t。"
                "误差项必须用已衰减状态 A（不是未衰减 S_{t-1}），且方向是 A^T k（不是 "
                "S_{t-1} k）；输出是 S_t^T q。recurrent 仅在 cache 已有状态且 seq_len==1 时"
                "走；空缓存 P=0/S=1 没有 precomputed state，走 chunk。"
                "不是普通 Attention，也不能机械铺开 seqlen 个框；chunk 分块路径是同一"
                "递推的不同执行形式（chunk 含 64 长度 pad 与三角 mask）")
        self.op("g_kda_state_write", "LinearAttentionLayer.update_recurrent_state",
                "KDA recurrent state 写入", "cache_update",
                ["g_kda_state_pre", "g_kda_state_new"], ["g_kda_state_post"], [],
                ["B", "NhL", "DhL", "DhL"],
                [src("761-762", "cache_params.update_recurrent_state("
                                "last_recurrent_state.to(float32), layer_idx)"),
                 self.srcf("cache_utils.py", "1118-1130", "update_recurrent_state：copy_ 到静态状态")],
                "新状态写回同一 cache 对象；读前/写后分成两个张量表达")

        # 4) gated norm + o_proj
        self.weight("g_kda_g_a_weight", "self.self_attn.g_a_proj.weight", ["DhL", "H"],
                    ["DhL", "H"], "输出门低秩投影（:656）")
        self.weight("g_kda_g_b_weight", "self.self_attn.g_b_proj.weight",
                    ["NhL×DhL", "DhL"], ["NhL", "DhL", "DhL"], "输出门上投影（:657）")
        self.weight("g_kda_o_norm_weight", "self.self_attn.o_norm.weight", ["DhL"], ["DhL"],
                    "RMSNormGated 的 weight；eps=rms_norm_eps（:658）")
        self.weight("g_kda_o_proj_weight", "self.self_attn.o_proj.weight", ["H", "NhL×DhL"],
                    ["H", "NhL", "DhL"], "nn.Linear(NhL*DhL, H, bias=False)（:659）")
        self.tensor("g_kda_g_a", "g_a_proj(hidden)", ["B", "seqlen", "DhL"],
                    ["B", "seqlen", "DhL"], "activation", "输出门低秩（:765）")
        self.op("g_kda_g_a_proj", "Glm5NextTextLinearAttention.g_a_proj", "KDA 输出门低秩",
                "linear", ["g_kda_masked_input"], ["g_kda_g_a"], ["g_kda_g_a_weight"],
                ["H", "DhL", "B", "seqlen"],
                [src("656-657", "self.g_a_proj / self.g_b_proj"),
                 src("765", "gate = self.g_b_proj(self.g_a_proj(hidden_states))")],
                "输出门也是低秩 128→8192")
        self.tensor("g_kda_g_b", "g_b_proj(g_a)", ["B", "seqlen", "NhL×DhL"],
                    ["B", "seqlen", "NhL", "DhL"], "activation", "输出门 logits（:765）")
        self.op("g_kda_g_b_proj", "Glm5NextTextLinearAttention.g_b_proj", "KDA 输出门上投影",
                "linear", ["g_kda_g_a"], ["g_kda_g_b"], ["g_kda_g_b_weight"],
                ["DhL", "NhL", "DhL", "B", "seqlen"],
                [src("657", "self.g_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)"),
                 src("765", "gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(hidden_shape)")],
                "输出 [B,S,8192] 后 view [B,S,64,128]")
        self.tensor("g_kda_gate", "gate（view 后）", ["B", "seqlen", "NhL", "DhL"],
                    ["B", "seqlen", "NhL", "DhL"], "activation", "输出门 [B,S,64,128]")
        self.op("g_kda_gate_view", "Glm5NextTextLinearAttention.forward", "KDA 输出门布局",
                "view", ["g_kda_g_b"], ["g_kda_gate"], [], ["B", "seqlen", "NhL", "DhL"],
                [src("765", ".view(hidden_shape)")],
                "把 [B,S,8192] 拆成 [B,S,64,128]")
        self.tensor("g_kda_o_normed", "self.o_norm(core_out, gate)",
                    ["B", "seqlen", "NhL", "DhL"], ["B", "seqlen", "NhL", "DhL"], "activation",
                    "fp32 RMSNorm(head_dim) × sigmoid(gate)（:374-393/:766）")
        self.op("g_kda_o_norm", "Glm5NextTextLinearAttention.o_norm", "KDA gated RMSNorm",
                "rmsnorm-gated", ["g_kda_core_out", "g_kda_gate"], ["g_kda_o_normed"],
                ["g_kda_o_norm_weight"], ["NhL", "DhL", "EpsN"],
                [src("374-393", "RMSNormGated：fp32 方差归一 * weight * sigmoid(gate)"),
                 src("766", "output = self.o_norm(core_attn_out, gate)")],
                "先按 head_dim 做 RMSNorm，再乘 sigmoid(gate)；不是普通 RMSNorm")
        self.tensor("g_kda_o_flat", "output.reshape(B,S,-1)", ["B", "seqlen", "NhL×DhL"],
                    ["B", "seqlen", "NhL", "DhL"], "activation", "拼头（:766）")
        self.op("g_kda_o_reshape", "Glm5NextTextLinearAttention.forward", "KDA 拼头", "reshape",
                ["g_kda_o_normed"], ["g_kda_o_flat"], [], ["NhL", "DhL"],
                [src("766", ".reshape(batch_size, seq_len, -1)")],
                "把 64×128 拼成 8192")
        self.tensor("g_attn_out", "self.o_proj(output)", ["B", "seqlen", "H"],
                    ["B", "seqlen", "H"], "activation", "Attention 输出（:767）")
        self.op("g_kda_o_proj", "Glm5NextTextLinearAttention.o_proj", "KDA 输出投影 o_proj",
                "linear", ["g_kda_o_flat"], ["g_attn_out"], ["g_kda_o_proj_weight"],
                ["NhL", "DhL", "H", "B", "seqlen"],
                [src("659", "self.o_proj = nn.Linear(self.qkv_dim, self.hidden_size, bias=False)"),
                 src("767", "output = self.o_proj(output)")],
                "bias=False；输出回到 hidden_size=4096")

        # 5) mHC 残差（attn 段）+ post norm
        self._mhc_apply("attn", "g_attn_out", "g_hidden_streams_in",
                        "g_hidden_streams_mid")
        self.tensor("g_hidden_streams_mid", "hidden_states（attn 段之后，4 路流）",
                    ["B", "seqlen", "Hc", "H"], ["B", "seqlen", "Hc", "H"], "activation",
                    "post*attn_out + combᵀ@residual；仍是 4 路流")
        self._mhc_site("ffn", "g_hidden_streams_mid",
                       residual_note="坍缩后的单流进入 post_attention_layernorm；"
                                     "FFN 残差由 post/comb 混合")
        self.tensor("g_ffn_in", "self.post_attention_layernorm(collapsed)",
                    ["B", "seqlen", "H"], ["B", "seqlen", "H", "EpsN"], "activation",
                    "RMSNorm（eps=rms_norm_eps）（:1342）")
        self.op("g_post_norm", "Glm5NextTextDecoderLayer.post_attention_layernorm",
                "Post-Attention RMSNorm", "rmsnorm", ["g_mhc_ffn_collapsed"], ["g_ffn_in"],
                ["g_post_norm_weight"], ["H", "EpsN"],
                [src("1339-1342", "residual = hidden_states; post, comb, hidden_states = ffn_hc(...); "
                                  "hidden_states = self.post_attention_layernorm(hidden_states)")],
                "与 input_layernorm 同为 RMSNorm，但权重独立")

        # 6) MLP / MoE
        if self.mlp_type == "sparse":
            self._moe_ops("g_ffn_in")
        else:
            self._dense_ops("g_ffn_in")

        # 7) mHC 残差（FFN 段）
        self._mhc_apply("ffn", "g_mlp_out", "g_hidden_streams_mid",
                        "g_hidden_streams_out")
        self.tensor("g_hidden_streams_out", "hidden_states（层输出，4 路流）",
                    ["B", "seqlen", "Hc", "H"], ["B", "seqlen", "Hc", "H"], "boundary",
                    "第二次 mHC 混合后的层输出；写入 next hidden_states（:1344-1348）")

    def _build_cache_notes(self) -> None:
        self.cache_notes = [
            "层缓存类：config.layer_types 在该层为 linear_attention（KDA），dispatch 到 "
            "LinearAttentionLayer（cache_utils:1285-1294）。",
            "conv state：update_conv_state 保存最近 Kconv=4 个 token 的 qkv 通道，形状 "
            "[B,3×NhL×DhL,Kconv]=[B,24576,4]；首次调用按需左补零，之后与本次拼接并只保留"
            "最后 Kconv（cache_utils:1099-1149）。",
            "recurrent state：delta-rule 状态 [B,NhL,DhL,DhL]=[B,64,128,128]，"
            "update_recurrent_state 复制 fp32 新状态（modeling:761-762；cache_utils:1118-1130）。",
            "两套状态大小**固定**，不随 kvcachelen/seqlen 增长；本模型不会为 KDA 层额外"
            "保留随历史增长的 KV cache。kvcachelen 只影响同网的 DSA 层与本层在序列中的位置。",
            "KDA 的因果性不是靠 4D causal mask：padding mask 只置零 padding，因果结构由"
            "因果短卷积 + delta-rule 的递推顺序（及其分块等价形式）保证。",
            "recurrent 仅在 cache 已有状态且 seq_len==1 时走；空缓存首次调用（含 P=0/S=1）"
            "与多 token 续接走 chunk 路径（chunk_size=64，尾部 padding 与三角 mask）；"
            "两条路径是同一递推的不同执行形式。",
        ]

    def _build_coverage_notes(self) -> None:
        v = self.values
        kda = [i for i in range(v.num_hidden_layers) if v.layer_type(i) == "linear_attention"]
        dsa = [i for i in range(v.num_hidden_layers) if v.layer_type(i) == "indexed_attention"]
        self.coverage_notes = [
            "已表达：mHC 两处（attn/ffn）的 4 路流坍缩/pre/post/comb Sinkhorn/展开/混合；"
            "KDA 的 q/k/v 投影、depthwise 因果短卷积与 conv state、forget gate、beta、"
            "L2Norm/缩放、delta-rule 扫描（chunk/recurrent）、recurrent state 写回、"
            "gated RMSNorm、o_proj；" + ("稠密 MLP（含 SwiGLU clamp）。" if self.mlp_type == "dense"
                                          else "TopK MoE + 并行共享专家（含 SwiGLU clamp）。"),
            f"真实层归属：KDA 层 {len(kda)} 个；DSA 层 {len(dsa)} 个"
            f"（{dsa[:6]}…{dsa[-2:]}）；dense 层 0/1/2。本 Block 只代表 KDA+{self.mlp_type} 组模板。",
            "根配置 quantization_config 声明 FP8（e4m3, block 128×128）：本图只表达 HF 逻辑 "
            "eager 路径，不分析量化 kernel、FP8 数值或权重内容。",
            "未覆盖：MTP 层 45（modeling:1378 显式忽略 + forward 只循环 0..44）、视觉 24 层、"
            "generate/采样循环、loss/aux_loss、output_router_logits 辅助输出、"
            "torch.compile/梯度检查点、数值等价。",
            "数据相关量（N_hit、N_e、路由实际分布）保留未知；不根据权重推断 MoE 实际路由，"
            "不默认 0。专家只有一份模板，不按 288 个专家复制图。",
            "KDA 的 kernel 级分块实现（chunk 64 的三角 mask/递推重排）按语义操作与循环模板"
            "表达，不伪装成普通 Attention，也不铺开 seqlen 个递推步。",
        ]


class _DsaBlockBuilder(_BlockBuilder):
    """MLA/DSA 稀疏注意力 + TopK MoE 的真实单层结构（NoPE / k-pool indexer）。"""

    kind = "dsa"

    def __init__(self, values: HfModelValues, layer_idx: int):
        super().__init__(values, layer_idx)
        self.mlp_type = "sparse"

    def build(self) -> dict:
        v = self.values
        title = (
            f"GLM-5.3-Flash Decoder Block（layer_idx={self.layer_idx}，MLA/DSA + TopK MoE；"
            f"{v.num_hidden_layers} 层主干代表模板；同一结构由 seqlen/kvcachelen 场景参数化）"
        )
        self._build_parameters()
        self._build_derivations()
        self._build_boundaries_and_cache()
        self._build_flow()
        self._build_cache_notes()
        self._build_coverage_notes()
        return {
            "id": "block",
            "title": title,
            "parameters": self.parameters,
            "tensors": self.tensors,
            "operators": self.operators,
            "derivations": self.derivations,
            "cache_notes": self.cache_notes,
            "coverage_notes": self.coverage_notes,
        }

    def _build_parameters(self) -> None:
        v = self.values
        cfg = self.cfg
        self._common_parameters()
        self.param("Nh", "Nh", "num_attention_heads DSA query 头数", v.num_attention_heads, None,
                   "config.num_attention_heads=64；KV 头同为 64（num_key_value_groups=1）",
                   [cfg("text_config.num_attention_heads", "文本子配置字段"),
                    cfg("text_config.num_key_value_heads", "文本子配置字段")])
        self.param("Dn", "Dn", "qk_nope_head_dim 不旋转 q/k 头维", v.qk_nope_head_dim, None,
                   "config.qk_nope_head_dim=256；NoPE 没有 rope 段",
                   [cfg("text_config.qk_nope_head_dim", "文本子配置字段")])
        self.param("Dv", "Dv", "v_head_dim value 头维", v.v_head_dim, None,
                   "config.v_head_dim=256；kv_b_proj 输出 Nh×(Dn+Dv)",
                   [cfg("text_config.v_head_dim", "文本子配置字段"),
                    self.src("1136-1140", "kv_b_proj = Linear(kv_lora_rank, "
                                           "num_heads*(qk_nope+v_head))")])
        self.param("Rk", "Rk", "qk_rope_head_dim 旋转维（本模型为 0）", v.qk_rope_head_dim, None,
                   "config.qk_rope_head_dim=0 → NoPE；源码仍构造 0 维 k_rot 槽，"
                   "attention 不做 RoPE",
                   [cfg("text_config.qk_rope_head_dim", "文本子配置字段（0）"),
                    self.srcf("configuration_glm5_next.py", "199-202",
                              "self.head_dim = self.qk_rope_head_dim = 0"),
                    self.src("1189-1192", "kv_a_proj_with_mqa 输出 512+0；k_rot 0 维")])
        self.param("Dq", "Dq", "qk_head_dim 注意力头维（nope+rope）", v.qk_head_dim,
                   "Dq = Dn + Rk",
                   "派生 = 256+0 = 256；注意力缩放为 256^-0.5 = 1/16",
                   [cfg("text_config.qk_head_dim", "文本子配置字段"),
                    self.src("1122", "self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim")])
        self.param("Rq", "Rq", "q_lora_rank Q 低秩中间维", v.q_lora_rank, None,
                   "config.q_lora_rank=1536；q_a_layernorm 是 RMSNorm(1536, eps=1e-5)",
                   [cfg("text_config.q_lora_rank", "文本子配置字段"),
                    self.src("1126-1128", "q_a_proj → q_a_layernorm → q_b_proj")])
        self.param("Rkv", "Rkv", "kv_lora_rank 压缩 KV latent 维", v.kv_lora_rank, None,
                   "config.kv_lora_rank=512；缓存 keys 的最后一维是 512；"
                   "values 槽的 k_rot 是 0 维（NoPE）",
                   [cfg("text_config.kv_lora_rank", "文本子配置字段"),
                    self.src("1130-1135", "kv_a_proj_with_mqa(kv_lora_rank + qk_rope_head_dim=0)"),
                    self.src("1194-1198", "缓存写入压缩 latent")])
        self.param("Ih", "Ih", "index_n_heads indexer 打分头数", v.index_n_heads, None,
                   "config.index_n_heads=32",
                   [cfg("text_config.index_n_heads", "文本子配置字段"),
                    self.src("797-801", "wq_b/wk/weights_proj 形状")])
        self.param("Id", "Id", "index_head_dim indexer 头维", v.index_head_dim, None,
                   "config.index_head_dim=128；k_norm 是**带 bias 的 LayerNorm**（eps=1e-6）",
                   [cfg("text_config.index_head_dim", "文本子配置字段"),
                    self.src("799", "self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)")])
        self.param("Ipack", "I_pack", "indexer packed 状态宽度", 2 * v.index_head_dim + 1,
                   "I_pack = Id × 2 + 1",
                   "packed = k(128) + gate_scores(128) + valid(1) = 257；"
                   "indexer 缓存存的是这个打包张量，不只是键",
                   [self.src("836-849", "packed_states = cat([k, gate_scores, valid])；"
                                        "update_indexer(packed_states)")])
        self.param("Ki", "Ki", "index_topk 每 query 的原始 token 预算", v.index_topk, None,
                   "config.index_topk=2048；源码按池预算 select_k=min(Ki//Kpool, 池数)，"
                   "展开后最多 Ki 个 token，再追 tail",
                   [cfg("text_config.index_topk", "文本子配置字段"),
                    self.src("883", "select_k = min(self.index_topk // self.index_kpool, ...)")])
        self.param("Kpool", "Kpool", "index_kpool 每池压缩的 token 数", v.index_kpool, None,
                   "config.index_kpool=4；池内 4 个 token 压成 1 个候选键",
                   [cfg("text_config.index_kpool", "文本子配置字段"),
                    self.src("964-1003", "number_of_pools=ceil(seq_len/kpool)；池内 softmax 加权")])
        self.param("EpsA", "EpsA", "indexer LayerNorm epsilon", v.sub_norm_eps, None,
                   "源码字面量 1e-6（不是 config.rms_norm_eps=1e-5）；只用于 indexer k_norm",
                   [self.src("799", "nn.LayerNorm(self.head_dim, eps=1e-6)")])
        self.param("kvcachelen", "kvcachelen", "调用前逻辑历史 token 数（kvcachelen）", None,
                   "P = past_key_values.get_seq_length()（无 cache 时 0）",
                   "空缓存 Prefill 取 0；Decode 为调用前已处理历史；不是物理保留条目数或"
                   "缓存字节数。DSA 的两套缓存都线性保留全部历史；KDA 层状态大小固定",
                   [self.src("1466-1469", "past_seen = past_key_values.get_seq_length()"),
                    self.src("841-849", "kv_len / current_length 来自 cache")])
        self.param("Lkv", "L_kv", "本次可读主 KV（压缩 latent）长度", None,
                   "L_kv = kvcachelen + seqlen",
                   "update 后 keys 线性追加全部历史；每次调用对全部 Lkv 位置做 kv_b_proj 展开",
                   [self.src("1194-1198", "update 后 expand_kv(k_pass, k_rot)"),
                    self.srcf("cache_utils.py", "132-153", "DynamicLayer.update：concat")])
        self.param("Lidx", "L_idx", "本次可读 indexer packed 缓存长度", None,
                   "L_idx = kvcachelen + seqlen（与 Lkv 同式，但来自 indexer packed 缓存）",
                   "packed 缓存 [B,Lidx,257]；11 个 DSA 层各自实例化 indexer",
                   [self.src("841-849", "packed_states = update_indexer(...); kv_len = ..."),
                    self.srcf("cache_utils.py", "350-365", "DynamicIndexedLayer.update_indexer")])
        self.param("Npool", "N_pool", "候选池数（完整池）", None,
                   "N_pool = L_idx // Kpool",
                   "池从第一个有效 token 开始，每 Kpool 个 token 一池；"
                   "不完整尾池不进入候选池集合，而是作为 tail 追加",
                   [self.src("964-1008", "number_of_pools=ceil(seq_len/kpool); pool_valid=all；"
                                         "keep=pool_valid.any(0)")])
        self.param("SelP", "SelP", "每 query 选择的池数", None,
                   "SelP = min(Ki // Kpool, N_pool)",
                   "预算 2048/4=512 个池；Lidx 较小时全选",
                   [self.src("883-888", "select_k=min(index_topk//index_kpool, 池数); "
                                        "index_scores.topk(select_k)")])
        self.param("j", "j", "query 在本次 seqlen 中的局部索引", None,
                   "j = 0..seqlen-1（query 维；不枚举具体 token）",
                   "无 padding 时第 j 个 query 的可见长度 V_q=P+j+1；"
                   "不给定 j 时 V_q/K_sel 如实未知，不能把最后一个 query 的值冒充所有 query",
                   [self.src("915-933", "q_positions = current_length - q_length + arange(q_length)"),
                    self.src("1043-1057", "visible_count = token_visible.sum(-1)")],
                   unknown_reason="query 索引未给定：j 是 0..seqlen-1 的符号，不枚举具体 token",
                   domain={"kind": "integer", "min": 0, "max_symbol": "seqlen"})
        self.param("Vq", "V_q", "第 j 个 query 的可见历史 token 数", None,
                   "V_q = kvcachelen + j + 1",
                   "query 局部索引 j 对应可见长度；无 padding 时即 P+j+1。"
                   "Ksel 依赖 V_q，不依赖全长 L_idx",
                   [self.src("929-933", "kv_positions <= q_positions"),
                    self.src("1043-1057", "tail 用 visible_count=token_visible.sum(-1)")])
        self.param("Ksel", "K_sel", "第 j 个 query 的稀疏可见 key 数（池展开 + tail）", None,
                   "K_sel = min(V_q, Ki + (V_q % Kpool))",
                   "逐 query：池展开 4×SelP 个 token + 当前不完整池 tail（≤Kpool-1）；"
                   "无 padding 且 V_q≤Ki 时等于 V_q；超过 Ki 时最多 Ki+Kpool-1=2051。"
                   "不是全长统一值/一般最大值；具体选中哪些池仍由打分数据决定",
                   [self.src("894-911", "池展开 + append_visible_tail + pad 到 "
                                        "index_topk + index_kpool - 1")])
        self.param("Ksel_last", "K_sel_last", "最后一个 query（j=seqlen-1）的 Ksel", None,
                   "K_sel_last = min(L_idx, Ki + (L_idx % Kpool))",
                   "只有 j=seqlen-1 才退化为全长 Lidx；这是“最后 query”值，"
                   "不能称为每 query 值、一般最大值或统一上界（S=4/P=4092 时 Ksel=2049..2048）",
                   [self.src("929-933", "last query 的 q_position = P+seqlen-1"),
                    self.src("1043-1057", "visible_count 随 query 位置变化")])
        self.param("Kout", "K_out", "indexer top-k 输出宽度（含 -1 填充）", None,
                   "K_out = Ki + Kpool - 1",
                   "源码把 topk_indices pad 到 index_topk + index_kpool - 1 = 2051；"
                   "无效位置填 -1，mask 构建时忽略",
                   [self.src("902-911", "output_width = index_topk (+ kpool-1); F.pad(..., -1)")])
        self.param("N_hit", "N_hit", "路由实际命中的不同专家数（数据相关）", None, None,
                   "数据相关量，未给定；不根据权重推断 MoE 实际路由，不默认 0 或 E",
                   [self.src("125-126", "one_hot(...).sum→nonzero：命中专家集合")],
                   unknown_reason="数据相关：路由命中集合未给定；不默认 0/E")
        self.param("N_e", "N_e", "单专家在本调用处理的 token 数（数据相关）", None, None,
                   "数据相关量，未给定；专家模板只有一份，不按 E 复制行",
                   [self.src("130", "torch.where(expert_mask[expert_idx])")],
                   unknown_reason="数据相关：每专家 token 数未给定；不默认 0")

    def _build_derivations(self) -> None:
        premise = ("按固定配置 + 「空态起步、连续正常执行」前提推导；"
                   "不覆盖裁剪/回滚后的自定义缓存快照")
        self.derivation("L_kv", "本次可读主 KV（压缩 latent）长度", _EXPR_LKV,
                        f"kvcachelen 为调用前逻辑历史；latent 缓存无压缩条目。{premise}")
        self.derivation("L_idx", "本次可读 indexer packed 缓存长度", _EXPR_LIDX,
                        f"与主 KV 同式；11 个 DSA 层各自有 packed 缓存。{premise}")
        self.derivation("N_pool", "候选池数（完整池）", _EXPR_NPOOL,
                        f"池从第一个有效 token 开始，每 Kpool 个 token 一池。{premise}")
        self.derivation("SelP", "每 query 选择的池数", _EXPR_SELP,
                        f"预算 Ki//Kpool=512；Lidx 较小时全选。{premise}")
        self.derivation("V_q", "第 j 个 query 的可见历史 token 数", _EXPR_VQ,
                        f"query 局部索引 j 只在该 query 的因果可见范围内；{premise}")
        self.derivation("K_sel", "第 j 个 query 的稀疏可见 key 数（池展开 + tail）", _EXPR_KSEL,
                        f"逐 query：V_q≤Ki 时覆盖全部可见 token；超过 Ki 时最多 "
                        f"Ki+Kpool-1=2051。不是全长统一值。{premise}")
        self.derivation("K_sel_last", "最后一个 query（j=seqlen-1）的 Ksel", _EXPR_KSEL_LAST,
                        f"仅当 j=seqlen-1 时 V_q=L_idx；这是最后 query 值，"
                        f"不是所有 query 的统一值。{premise}")
        self.derivation("K_out", "indexer top-k 输出宽度（含 -1 填充）", _EXPR_KOUT,
                        f"源码 pad 到 Ki+Kpool-1=2051；无效位置 -1。{premise}")
        self.derivation("T_tok", "MoE 路由的展平 token 数", _EXPR_TTOK,
                        f"B×seqlen；B 为可填性能配置。{premise}")

    def _build_boundaries_and_cache(self) -> None:
        self.tensor(
            "g_hidden_streams_in", "hidden_states（4 路流层输入）", ["B", "seqlen", "Hc", "H"],
            ["B", "seqlen", "Hc", "H"], "boundary",
            "上一层输出或网络入口的 embedding 广播（modeling:1493）；DecoderLayer.forward "
            "的 hidden_states（:1312）")
        self.tensor(
            "g_attention_mask", "attention_mask（2D padding mask）", ["B", "seqlen"],
            ["B", "seqlen"], "boundary",
            "create_recurrent_attention_mask 的 [B,seqlen] padding mask；None 时网络入口补 "
            "ones（modeling:1471-1486）。DSA 的因果性由 indexer 的 get_visible_tokens 保证")
        self.tensor(
            "g_cache_kv_pre", "past_key_values.layers[layer_idx].keys（调用前，压缩 latent）",
            ["B", "1", "kvcachelen", "Rkv"], ["B", "kvcachelen", "Rkv"], "cache",
            "DynamicIndexedLayer.keys；存的是 kv_a_layernorm 后的 512 维 latent，"
            "不是展开后的 64 头 K（cache_utils:121-153 / modeling:1194-1198）")
        self.tensor(
            "g_cache_krot_pre", "past_key_values.layers[layer_idx].values（调用前，k_rot）",
            ["B", "1", "kvcachelen", "Rk"], ["B", "kvcachelen", "Rk"], "cache",
            "同一 update 的 values 槽存 k_rot；本模型 NoPE，Rk=0，槽是 0 维张量")
        self.tensor(
            "g_cache_kv_post", "self.keys（调用后，压缩 latent）",
            ["B", "1", "Lkv", "Rkv"], ["B", "Lkv", "Rkv"], "cache",
            "update(k_pass, k_rot) 追加本次 seqlen 个位置后的持久 keys（cache_utils:147-153）")
        self.tensor(
            "g_cache_krot_post", "self.values（调用后，k_rot）",
            ["B", "1", "Lkv", "Rk"], ["B", "Lkv", "Rk"], "cache",
            "追加后的 values；本模型为 0 维（NoPE），只保留与源码一致的槽位事实")
        self.tensor(
            "g_dsa_ix_cache_pre", "self.indexer_keys（调用前，packed）",
            ["B", "kvcachelen", "Ipack"], ["B", "kvcachelen", "Ipack"], "cache",
            "DynamicIndexedLayer.indexer_keys：k(128)+gate(128)+valid(1) 打包的 3D 缓存")
        self.tensor(
            "g_dsa_ix_cache_post", "self.indexer_keys（调用后，packed）",
            ["B", "Lidx", "Ipack"], ["B", "Lidx", "Ipack"], "cache",
            "update_indexer 以 dim=1 concat 后的全量 packed 状态")

    def _build_flow(self) -> None:
        v = self.values
        src, cfg = self.src, self.cfg
        self._mhc_site("attn", "g_hidden_streams_in",
                       residual_note="坍缩后的单流进入 input_layernorm；mHC 的残差在 Attention "
                                     "之后由 post/comb 混合，不是普通相加")
        self.weight("g_input_norm_weight", "self.input_layernorm.weight", ["H"], ["H"],
                    "RMSNorm 可学习缩放（无 bias），eps=rms_norm_eps（:1293）")
        self.weight("g_post_norm_weight", "self.post_attention_layernorm.weight", ["H"], ["H"],
                    "RMSNorm 可学习缩放（无 bias），eps=rms_norm_eps（:1294）")
        self.tensor("g_attn_normed", "hidden_states = self.input_layernorm(collapsed)",
                    ["B", "seqlen", "H"], ["B", "seqlen", "H", "EpsN"], "activation",
                    "RMSNorm：fp32 方差归一后转回输入 dtype，再乘 weight（:64-79/:1315）")
        self.op("g_input_norm", "Glm5NextTextDecoderLayer.input_layernorm", "Input RMSNorm",
                "rmsnorm", ["g_mhc_attn_collapsed"], ["g_attn_normed"],
                ["g_input_norm_weight"], ["H", "EpsN"],
                [src("1315", "hidden_states = self.input_layernorm(hidden_states)"),
                 src("64-79", "Glm5NextTextRMSNorm.forward")],
                "层输入先归一化再进入 Attention；残差保留 mHC 的 4 路流")

        # 1) Q 低秩（NoPE，无 split/rope）
        self.weight("g_dsa_q_a_weight", "self.self_attn.q_a_proj.weight", ["Rq", "H"],
                    ["Rq", "H"], "nn.Linear(H, q_lora_rank, bias=attention_bias=False)（:1126）")
        self.weight("g_dsa_q_a_norm_weight", "self.self_attn.q_a_layernorm.weight", ["Rq"],
                    ["Rq"], "q_a_layernorm：RMSNorm(1536)，eps=config.rms_norm_eps=1e-5（:1127）")
        self.weight("g_dsa_q_b_weight", "self.self_attn.q_b_proj.weight", ["Nh×Dq", "Rq"],
                    ["Nh", "Dq", "Rq"],
                    "nn.Linear(q_lora_rank, num_heads*qk_head_dim, bias=False)（:1128）")
        self.tensor("g_dsa_q_a", "self.q_a_proj(hidden_states)", ["B", "seqlen", "Rq"],
                    ["B", "seqlen", "Rq"], "activation", "Q 低秩投影（:1186）")
        self.op("g_dsa_q_a_proj", "Glm5NextTextAttention.q_a_proj", "Q 低秩投影 q_a_proj",
                "linear", ["g_attn_normed"], ["g_dsa_q_a"], ["g_dsa_q_a_weight"],
                ["H", "Rq", "B", "seqlen"],
                [src("1186", "q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))")],
                "bias=False；输出 [B,S,1536]")
        self.tensor("g_dsa_q_residual", "q_resid = self.q_a_layernorm(...)",
                    ["B", "seqlen", "Rq"], ["B", "seqlen", "Rq"], "activation",
                    "q_residual：进入 q_b_proj，也作为 indexer.wq_b 的输入（:1186/:1201）")
        self.op("g_dsa_q_a_norm", "Glm5NextTextAttention.q_a_layernorm", "Q 低秩 RMSNorm",
                "rmsnorm", ["g_dsa_q_a"], ["g_dsa_q_residual"], ["g_dsa_q_a_norm_weight"],
                ["Rq", "EpsN"],
                [src("1127", "self.q_a_layernorm = Glm5NextTextRMSNorm("
                             "self.q_lora_rank, eps=config.rms_norm_eps)")],
                "eps=1e-5（不是 indexer k_norm 的 1e-6）；同一 q_residual 被主注意力与 indexer 共享")
        self.tensor("g_dsa_q_flat", "self.q_b_proj(q_resid)", ["B", "seqlen", "Nh×Dq"],
                    ["B", "seqlen", "Nh", "Dq"], "activation", "q_b_proj 输出 [B,S,16384]")
        self.op("g_dsa_q_b_proj", "Glm5NextTextAttention.q_b_proj", "Q 上投影 q_b_proj",
                "linear", ["g_dsa_q_residual"], ["g_dsa_q_flat"], ["g_dsa_q_b_weight"],
                ["Rq", "Nh", "Dq", "B", "seqlen"],
                [src("1128", "self.q_b_proj = nn.Linear(q_lora_rank, num_heads*qk_head_dim, "
                            "bias=False)")],
                "输出最后一维 Nh×Dq=64×256")
        self.tensor("g_dsa_q", "query_states（view+transpose 后）", ["B", "Nh", "seqlen", "Dq"],
                    ["B", "Nh", "seqlen", "Dq"], "activation",
                    "view(B,S,Nh,Dq).transpose(1,2)；**没有 split/RoPE**（NoPE）")
        self.op("g_dsa_q_view", "Glm5NextTextAttention.forward", "Q reshape/transpose",
                "view+transpose", ["g_dsa_q_flat"], ["g_dsa_q"], [],
                ["B", "Nh", "seqlen", "Dq"],
                [src("1183-1187", "query_shape=(B,S,-1,qk_head_dim); "
                                  "q_b_proj(q_resid).view(query_shape).transpose(1,2)")],
                "NoPE：没有 q_pass/q_rot split，也没有 apply_rotary_pos_emb 调用")

        # 2) 压缩 KV latent（NoPE）
        self.weight("g_dsa_kv_a_weight", "self.self_attn.kv_a_proj_with_mqa.weight",
                    ["Rkv+Rk", "H"], ["Rkv", "Rk", "H"],
                    "nn.Linear(H, kv_lora_rank+qk_rope_head_dim=512+0)（:1130-1135）")
        self.weight("g_dsa_kv_a_norm_weight", "self.self_attn.kv_a_layernorm.weight", ["Rkv"],
                    ["Rkv"], "kv_a_layernorm：RMSNorm(512)，eps=config.rms_norm_eps=1e-5")
        self.tensor("g_dsa_kv_compressed", "self.kv_a_proj_with_mqa(hidden_states)",
                    ["B", "seqlen", "Rkv+Rk"], ["B", "seqlen", "Rkv", "Rk"], "activation",
                    "单头压缩 KV：512 维 latent + 0 维 k_rot（NoPE）")
        self.op("g_dsa_kv_a_proj", "Glm5NextTextAttention.kv_a_proj_with_mqa", "KV 压缩投影",
                "linear", ["g_attn_normed"], ["g_dsa_kv_compressed"], ["g_dsa_kv_a_weight"],
                ["H", "Rkv", "Rk", "B", "seqlen"],
                [src("1189-1192", "compressed_kv = self.kv_a_proj_with_mqa(hidden_states); "
                                  "kv_pass, k_rot = split([kv_lora_rank, qk_rope_head_dim])")],
                "输出 [B,S,512+0]；这份压缩投影就是 MLA 省 KV 缓存的来源")
        self.tensor("g_dsa_kv_pass", "kv_pass（未归一化 latent）", ["B", "seqlen", "Rkv"],
                    ["B", "seqlen", "Rkv"], "activation", "split 前段（512）")
        self.tensor("g_dsa_k_rot", "k_rot（0 维，NoPE）", ["B", "seqlen", "Rk"],
                    ["B", "seqlen", "Rk"], "activation",
                    "split 后段；qk_rope_head_dim=0，所以是 0 维张量")
        self.op("g_dsa_kv_a_split", "Glm5NextTextAttention.forward", "KV 拆分 latent/k_rot",
                "split", ["g_dsa_kv_compressed"], ["g_dsa_kv_pass", "g_dsa_k_rot"], [],
                ["Rkv", "Rk"],
                [src("1190", "kv_pass, k_rot = torch.split(compressed_kv, "
                             "[kv_lora_rank, qk_rope_head_dim], dim=-1)")],
                "NoPE：k_rot 宽度为 0；源码仍保留这个槽位")
        self.tensor("g_dsa_k_pass_normed", "k_pass = self.kv_a_layernorm(kv_pass)",
                    ["B", "seqlen", "Rkv"], ["B", "seqlen", "Rkv"], "activation",
                    "kv_a_layernorm 后的 latent；随后 view 成单头 4D（:1191）")
        self.op("g_dsa_kv_a_norm", "Glm5NextTextAttention.kv_a_layernorm", "KV latent RMSNorm",
                "rmsnorm", ["g_dsa_kv_pass"], ["g_dsa_k_pass_normed"],
                ["g_dsa_kv_a_norm_weight"], ["Rkv", "EpsN"],
                [src("1135", "self.kv_a_layernorm = Glm5NextTextRMSNorm("
                             "config.kv_lora_rank, eps=config.rms_norm_eps)"),
                 src("1191", "k_pass = self.kv_a_layernorm(kv_pass)")],
                "eps=1e-5；只有 latent 段归一化，k_rot 不归一化")
        self.tensor("g_dsa_k_pass_4d", "k_pass.view(B,1,seqlen,kv_lora_rank)",
                    ["B", "1", "seqlen", "Rkv"], ["B", "seqlen", "Rkv"], "activation",
                    "归一化后的 latent 再 view 成单头 4D，才进入 4D 缓存")
        self.op("g_dsa_k_pass_view", "Glm5NextTextAttention.forward", "KV latent 单头 view",
                "view", ["g_dsa_k_pass_normed"], ["g_dsa_k_pass_4d"], [],
                ["Rkv", "B", "seqlen"],
                [src("1191", "k_pass = self.kv_a_layernorm(kv_pass).view("
                             "batch_size, 1, seq_length, kv_lora_rank)")],
                "把 [B,S,512] 显式 view 成 [B,1,S,512]；缓存写入的是这个单头 4D 张量")
        self.tensor("g_dsa_k_rot_4d", "k_rot.view(B,1,seqlen,qk_rope_head_dim)",
                    ["B", "1", "seqlen", "Rk"], ["B", "seqlen", "Rk"], "activation",
                    "0 维 k_rot 的 4D 形态；缓存 values 槽与它保持一致")
        self.op("g_dsa_k_rot_view", "Glm5NextTextAttention.forward", "k_rot 单头 view",
                "view", ["g_dsa_k_rot"], ["g_dsa_k_rot_4d"], [], ["B", "seqlen", "Rk"],
                [src("1192", "k_rot = k_rot.view(batch_size, 1, seq_length, "
                             "self.qk_rope_head_dim)")],
                "NoPE：Rk=0，这个 4D 张量最后一维为 0")
        self.op("g_dsa_cache_kv_update", "DynamicIndexedLayer.update",
                "KV 缓存追加（压缩 latent）", "cache_update",
                ["g_cache_kv_pre", "g_cache_krot_pre", "g_dsa_k_pass_4d", "g_dsa_k_rot_4d"],
                ["g_cache_kv_post", "g_cache_krot_post"], [],
                ["Lkv", "Rkv", "Rk", "kvcachelen", "seqlen"],
                [src("1194-1198", "if past_key_values is not None: k_pass, k_rot = "
                                  "past_key_values.update(k_pass, k_rot, layer_idx)")],
                "缓存里是 512 维 latent + 0 维 k_rot（每 token 512 个值/层），"
                "**不是**展开后的 64 头 K/V；调用前/调用后分两个张量表达")
        self.weight("g_dsa_kv_b_weight", "self.self_attn.kv_b_proj.weight",
                    ["Nh×(Dn+Dv)", "Rkv"], ["Nh", "Dn", "Dv", "Rkv"],
                    "nn.Linear(512, num_heads*(256+256), bias=False)（:1136-1140）")
        self.tensor("g_dsa_kv_nope_flat", "self.kv_b_proj(k_pass)",
                    ["B", "1", "Lkv", "Nh×(Dn+Dv)"], ["B", "Lkv", "Nh", "Dn", "Dv"],
                    "activation", "对缓存返回的**全部 Lkv 个位置**做上投影")
        self.op("g_dsa_kv_b_proj", "Glm5NextTextAttention.kv_b_proj", "KV latent 展开 kv_b_proj",
                "linear", ["g_cache_kv_post"], ["g_dsa_kv_nope_flat"], ["g_dsa_kv_b_weight"],
                ["Rkv", "Nh", "Dn", "Dv", "Lkv"],
                [src("1155-1165", "expand_kv：kv_nope = kv_b_proj(kv_nope).view(key_shape)"
                                  ".transpose(1,2)")],
                "输出最后一维 Nh×(256+256)=64×512；随后 view+transpose 再拆 k_nope/v")
        self.tensor("g_dsa_kv_nope", "kv_nope（view+transpose 后）",
                    ["B", "Nh", "Lkv", "Dn+Dv"], ["B", "Nh", "Lkv", "Dn", "Dv"], "activation",
                    "view(B,Lkv,Nh,Dn+Dv).transpose(1,2)（:1162-1164）")
        self.op("g_dsa_kv_view", "Glm5NextTextAttention.expand_kv", "KV 展开布局",
                "view+transpose", ["g_dsa_kv_nope_flat"], ["g_dsa_kv_nope"], [],
                ["B", "Nh", "Lkv", "Dn", "Dv"],
                [src("1162-1164", "key_shape=(B,L,-1,qk_nope+v_head); "
                                  "kv_b_proj(...).view(key_shape).transpose(1,2)")],
                "把 Lkv 个位置的 512 维向量按 64 头重排")
        self.tensor("g_dsa_k_nope", "k_nope", ["B", "Nh", "Lkv", "Dn"],
                    ["B", "Nh", "Lkv", "Dn"], "activation", "split 前段（256）")
        self.tensor("g_dsa_v", "value_states", ["B", "Nh", "Lkv", "Dv"],
                    ["B", "Nh", "Lkv", "Dv"], "activation", "split 后段（256）")
        self.op("g_dsa_kv_split", "Glm5NextTextAttention.expand_kv", "拆分 k_nope/value",
                "split", ["g_dsa_kv_nope"], ["g_dsa_k_nope", "g_dsa_v"], [],
                ["Dn", "Dv"],
                [src("1165", "k_nope, value_states = torch.split(kv_nope, "
                             "[qk_nope_head_dim, v_head_dim], dim=-1)")],
                "k 头维 256、v 头维 256；NoPE 时 key 只有 nope 段")
        self.tensor("g_dsa_k_rot_expanded", "k_rot.expand(-1, n_heads, -1, -1)",
                    ["B", "Nh", "Lkv", "Rk"], ["B", "Nh", "Lkv", "Rk"], "activation",
                    "单头 k_rot 广播到 Nh 头（:1166）；NoPE 时是 0 维")
        self.op("g_dsa_k_rot_expand", "Glm5NextTextAttention.expand_kv", "k_rot 广播到各头",
                "expand", ["g_cache_krot_post"], ["g_dsa_k_rot_expanded"], [],
                ["Nh", "Rk"],
                [src("1166", "k_rot = k_rot.expand(-1, k_nope.shape[1], -1, -1)")],
                "NoPE：0 维广播后仍是 0 维；源码保留拼接步骤")
        self.tensor("g_dsa_key", "key_states = cat([k_nope, k_rot])",
                    ["B", "Nh", "Lkv", "Dq"], ["B", "Nh", "Lkv", "Dq"], "activation",
                    "key_states[..., :Dn]=k_nope; [..., Dn:]=k_rot（0 维）（:1167-1171）")
        self.op("g_dsa_key_cat", "Glm5NextTextAttention.expand_kv", "key 拼接 nope+rope",
                "concat", ["g_dsa_k_nope", "g_dsa_k_rot_expanded"], ["g_dsa_key"], [],
                ["Dn", "Rk", "Dq"],
                [src("1167-1171", "key_states = new_empty(..., qk_nope+qk_rope); "
                                  "copy_(k_nope); copy_(k_rot)")],
                "NoPE 时 Rk=0，key 实际等于 k_nope；不能因此省略源码的拼接语义")

        # 3) DSA indexer（k-pool）
        self.weight("g_dsa_ix_wq_b_weight", "self.self_attn.indexer.wq_b.weight",
                    ["Ih×Id", "Rq"], ["Ih", "Id", "Rq"],
                    "indexer.wq_b：Linear(q_lora_rank, Ih*Id, bias=False)（:797）")
        self.weight("g_dsa_ix_wk_weight", "self.self_attn.indexer.wk.weight", ["Id", "H"],
                    ["Id", "H"], "indexer.wk：Linear(H, head_dim, bias=False)（:798）")
        self.weight("g_dsa_ix_k_norm_weight", "self.self_attn.indexer.k_norm.weight", ["Id"],
                    ["Id"], "indexer.k_norm：LayerNorm(head_dim) 的 weight（**带 bias**）")
        self.weight("g_dsa_ix_k_norm_bias", "self.self_attn.indexer.k_norm.bias", ["Id"], ["Id"],
                    "indexer.k_norm 的 bias；这正是 LayerNorm 而非 RMSNorm 的证据（:799）")
        self.weight("g_dsa_ix_weights_proj_weight", "self.self_attn.indexer.weights_proj.weight",
                    ["Ih", "H"], ["Ih", "H"], "indexer.weights_proj：Linear(H, Ih)（:800）")
        self.weight("g_dsa_ix_gate_weight", "self.self_attn.indexer.index_kpool_compress_gate",
                    ["Id", "H"], ["Id", "H"],
                    "indexer.index_kpool_compress_gate：池内 softmax 的 per-token 打分投影（:807）")
        self.weight("g_dsa_ix_ape", "self.self_attn.indexer.index_kpool_compress_ape",
                    ["Kpool", "Id"], ["Kpool", "Id"],
                    "indexer.index_kpool_compress_ape：池内位置偏置（:806）")
        self.tensor("g_dsa_ix_q_flat", "self.indexer.wq_b(q_residual)",
                    ["B", "seqlen", "Ih×Id"], ["B", "seqlen", "Ih", "Id"], "activation",
                    "indexer 的独立 Q 投影（输入是 q_residual，不是 hidden_states）（:833）")
        self.op("g_dsa_ix_q_proj", "Glm5NextTextIndexer.wq_b", "Indexer Q 投影", "linear",
                ["g_dsa_q_residual"], ["g_dsa_ix_q_flat"], ["g_dsa_ix_wq_b_weight"],
                ["Rq", "Ih", "Id", "B", "seqlen"],
                [src("833", "q = self.wq_b(q_resid).view(hidden_shape)")],
                "indexer 有独立于主注意力的轻量投影（wq_b/wk/weights_proj）")
        self.tensor("g_dsa_ix_q", "q（indexer 打分头）", ["B", "seqlen", "Ih", "Id"],
                    ["B", "seqlen", "Ih", "Id"], "activation",
                    "view 后（:833）；保留 [B,S,H,D] 布局，不 transpose")
        self.op("g_dsa_ix_q_view", "Glm5NextTextIndexer.forward", "Indexer Q 布局", "view",
                ["g_dsa_ix_q_flat"], ["g_dsa_ix_q"], [], ["B", "seqlen", "Ih", "Id"],
                [src("833", "q = self.wq_b(q_resid).view(hidden_shape)")],
                "NoPE：indexer 也不做 RoPE（qk_rope_head_dim=0，源码无 apply_rotary）")
        self.tensor("g_dsa_ix_k_proj_out", "self.indexer.wk(hidden_states)",
                    ["B", "seqlen", "Id"], ["B", "seqlen", "Id"], "activation",
                    "indexer 的 K 投影：输入是 Attention 的 hidden_states（:834）")
        self.op("g_dsa_ix_k_proj", "Glm5NextTextIndexer.wk", "Indexer K 投影", "linear",
                ["g_attn_normed"], ["g_dsa_ix_k_proj_out"], ["g_dsa_ix_wk_weight"],
                ["H", "Id", "B", "seqlen"],
                [src("834", "k = self.k_norm(self.wk(hidden_states)).view(hidden_shape)"
                            ".squeeze(2)")],
                "单头、hidden_size→128；与主 KV 的 512 维 latent 完全独立")
        self.tensor("g_dsa_ix_k_normed", "self.indexer.k_norm(k)", ["B", "seqlen", "Id"],
                    ["B", "seqlen", "Id"], "activation",
                    "LayerNorm(128, eps=1e-6)：**带 weight 与 bias**（:799）")
        self.op("g_dsa_ix_k_norm", "Glm5NextTextIndexer.k_norm", "Indexer K LayerNorm",
                "layernorm", ["g_dsa_ix_k_proj_out"], ["g_dsa_ix_k_normed"],
                ["g_dsa_ix_k_norm_weight", "g_dsa_ix_k_norm_bias"], ["Id", "EpsA"],
                [src("799", "self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)"),
                 src("834", "k = self.k_norm(self.wk(hidden_states))")],
                "这是 LayerNorm 而非 RMSNorm（有 bias）；量化配置 modules_to_not_convert "
                "也单独列出 indexer.k_norm.bias")
        self.tensor("g_dsa_ix_gate_scores", "F.linear(hidden, gate_weight)",
                    ["B", "seqlen", "Id"], ["B", "seqlen", "Id"], "activation",
                    "每 token 的池内打分（:836）")
        self.op("g_dsa_ix_gate", "Glm5NextTextIndexer.forward", "Indexer 池内打分", "linear",
                ["g_attn_normed"], ["g_dsa_ix_gate_scores"], ["g_dsa_ix_gate_weight"],
                ["Id", "H", "B", "seqlen"],
                [src("836", "gate_scores = F.linear(hidden_states, "
                            "self.index_kpool_compress_gate)")],
                "gate_scores 不是注意力分数，而是池内 softmax 的权重 logits")
        self.tensor("g_dsa_ix_valid", "attention_mask.to(k.dtype)[..., None]",
                    ["B", "seqlen", "1"], ["B", "seqlen"], "activation",
                    "valid channel：padding/可见性标志（:837）")
        self.op("g_dsa_ix_valid", "Glm5NextTextIndexer.forward", "Indexer valid 通道",
                "cast+unsqueeze", ["g_attention_mask"], ["g_dsa_ix_valid"], [],
                ["B", "seqlen"],
                [src("837", "valid_channel = attention_mask.to(k.dtype)[..., None]")],
                "与 k/gate 一起打包，供后续池 valid 与因果可见性判断")
        self.tensor("g_dsa_ix_packed", "packed_states = cat([k, gate, valid])",
                    ["B", "seqlen", "Ipack"], ["B", "seqlen", "Ipack"], "activation",
                    "打包后进入 indexer 缓存（:839）")
        self.op("g_dsa_ix_pack", "Glm5NextTextIndexer.forward", "Indexer packed 打包",
                "concat", ["g_dsa_ix_k_normed", "g_dsa_ix_gate_scores", "g_dsa_ix_valid"],
                ["g_dsa_ix_packed"], [], ["Id", "Ipack", "B", "seqlen"],
                [src("839", "packed_states = torch.cat([k, gate_scores, valid_channel], dim=-1)")],
                "packed 宽度 257 = 128+128+1；缓存存的是全部三段，不只是键")
        self.op("g_dsa_ix_cache_update", "DynamicIndexedLayer.update_indexer",
                "Indexer packed 缓存追加", "cache_update",
                ["g_dsa_ix_cache_pre", "g_dsa_ix_packed"], ["g_dsa_ix_cache_post"], [],
                ["Lidx", "Ipack", "kvcachelen", "seqlen"],
                [src("841-846", "packed_states = past_key_values.update_indexer("
                                "packed_states, self.layer_idx); kv_len = ...")],
                "indexer 缓存是 3D [B,L,257]（无 value）；11 个 DSA 层各自有该缓存")
        self.tensor("g_dsa_ix_valid_keys", "packed_states[..., -1].bool()",
                    ["B", "Lidx"], ["B", "Lidx"], "activation",
                    "从 packed 缓存取回 valid 通道（:852）")
        self.op("g_dsa_ix_valid_keys", "Glm5NextTextIndexer.forward", "取回 valid 通道",
                "slice+cast", ["g_dsa_ix_cache_post"], ["g_dsa_ix_valid_keys"], [],
                ["B", "Lidx"],
                [src("852", "valid_keys = packed_states[..., -1].bool()")],
                "只取最后一维（valid），不是重新算 mask")
        self.tensor("g_dsa_ix_visible", "visible_tokens = causal & valid_keys",
                    ["B", "seqlen", "Lidx"], ["B", "seqlen", "Lidx"], "activation",
                    "每 query 对每个历史位置的因果+padding 可见性（:853-857/:915-933）")
        self.op("g_dsa_ix_visible", "Glm5NextTextIndexer.get_visible_tokens",
                "Indexer 因果可见性", "compare+and",
                ["g_dsa_ix_valid_keys", "g_attn_normed"], ["g_dsa_ix_visible"], [],
                ["B", "seqlen", "Lidx"],
                [src("851-857", "visible_tokens = self.get_visible_tokens(valid_keys, "
                                "q_length=seq_len, current_length=current_length)"),
                 src("929-933", "kv_positions <= q_positions & valid_keys")],
                "DSA 没有网络级 4D causal mask：因果性在 indexer 内由 kv_pos ≤ q_pos 构造；"
                "池只有在最后一个 token 可见时才可选")
        self.tensor("g_dsa_ix_pool_keys", "pool_keys（池内 softmax 加权键）",
                    ["B", "Npool", "Id"], ["B", "Npool", "Id"], "activation",
                    "每 Kpool 个 token 压成一个候选键（:997-1003）")
        self.tensor("g_dsa_ix_pool_indices", "pool_indices（原始 token 索引）",
                    ["B", "Npool", "Kpool"], ["B", "Npool", "Kpool"], "activation",
                    "池内 token 的原始位置；未满池的位置为 -1（:983-995）")
        self.tensor("g_dsa_ix_pool_valid", "pool_valid（完整池）", ["B", "Npool"],
                    ["B", "Npool"], "activation", "只有 Kpool 个 token 都有效的池才是候选（:994）")
        self.op("g_dsa_ix_pool", "Glm5NextTextIndexer.get_pooled_states", "DSA k-pool 压缩",
                "pool+softmax", ["g_dsa_ix_cache_post"],
                ["g_dsa_ix_pool_keys", "g_dsa_ix_pool_indices", "g_dsa_ix_pool_valid"],
                ["g_dsa_ix_ape"], ["Id", "Kpool", "Npool"],
                [src("935-1008", "number_of_pools=ceil(seq_len/kpool); "
                                 "logits = grouped_gate_scores + index_kpool_compress_ape; "
                                 "softmax 加权求和; pool_valid 要求整池有效")],
                "池只依赖 packed 状态里的 k/gate/valid 通道与 APE 位置偏置，"
                "不依赖逐 query 的 visible_tokens；不完整尾池不作为候选池")
        self.tensor("g_dsa_ix_pool_end", "pool_end = pool_indices[..., -1].clamp(0, kv_len-1)",
                    ["B", "Npool"], ["B", "Npool"], "activation",
                    "每个候选池最后一个 token 的全局位置；无效池 clamp 到 0 但仍会被 "
                    "pool_valid 拒绝（:868-869）")
        self.op("g_dsa_ix_pool_end", "Glm5NextTextIndexer.forward", "池末端位置", "slice+clamp",
                ["g_dsa_ix_pool_indices"], ["g_dsa_ix_pool_end"], [], ["B", "Npool"],
                [src("868-869", "pool_end = pool_indices[..., -1].clamp(0, kv_len - 1)")],
                "mask 需要 pool_indices 才能求 pool_end；不能用 pool_valid 或 visible 代替")
        self.tensor("g_dsa_ix_valid_candidates", "valid_candidates = pool_visible & pool_valid",
                    ["B", "seqlen", "Npool"], ["B", "seqlen", "Npool"], "mask",
                    "每个 query 可选的池：池末 token 对该 query 可见，且整池有效（:870-875）")
        self.op("g_dsa_ix_valid_candidates", "Glm5NextTextIndexer.forward", "池候选有效性",
                "gather+and", ["g_dsa_ix_visible", "g_dsa_ix_pool_end", "g_dsa_ix_pool_valid"],
                ["g_dsa_ix_valid_candidates"], [], ["B", "seqlen", "Npool"],
                [src("870-875", "pool_visible = visible_tokens.gather(-1, pool_end); "
                                "valid_candidates = pool_visible & pool_valid[:, None]")],
                "先按 pool_end gather 逐 query 可见性，再与 pool_valid 相与；"
                "这是 mask 的唯一有效性来源")
        self.tensor("g_dsa_ix_scores", "scores = q @ pool_keysᵀ * softmax_scale",
                    ["B", "seqlen", "Ih", "Npool"], ["B", "seqlen", "Ih", "Npool"],
                    "activation", "打分在 float32（:861）")
        self.op("g_dsa_ix_scores", "Glm5NextTextIndexer.forward", "Indexer 池打分",
                "matmul+scale", ["g_dsa_ix_q", "g_dsa_ix_pool_keys"], ["g_dsa_ix_scores"], [],
                ["Ih", "seqlen", "Npool", "Id"],
                [src("861", "scores = torch.matmul(q.float(), pool_keys.transpose(-1,-2)"
                           ".float().unsqueeze(1)) * self.softmax_scale")],
                "q [B,S,Ih,128] × pool_keysᵀ [B,1,128,Npool] → [B,S,Ih,Npool]；"
                "softmax_scale=128^-0.5")
        self.tensor("g_dsa_ix_relu", "scores = F.relu(scores)", ["B", "seqlen", "Ih", "Npool"],
                    ["B", "seqlen", "Ih", "Npool"], "activation", "ReLU 截断负分（:862）")
        self.op("g_dsa_ix_relu", "Glm5NextTextIndexer.forward", "Indexer ReLU", "relu",
                ["g_dsa_ix_scores"], ["g_dsa_ix_relu"], [], ["Ih", "Npool"],
                [src("862", "scores = F.relu(scores * self.softmax_scale)")],
                "只保留正相关；负分不贡献选择")
        self.tensor("g_dsa_ix_weights", "weights = weights_proj(hidden) * Ih^-0.5",
                    ["B", "seqlen", "Ih"], ["B", "seqlen", "Ih"], "activation",
                    "每头权重（float32，:865）")
        self.op("g_dsa_ix_weights", "Glm5NextTextIndexer.weights_proj", "Indexer 头权重",
                "linear+scale", ["g_attn_normed"], ["g_dsa_ix_weights"],
                ["g_dsa_ix_weights_proj_weight"], ["H", "Ih", "B", "seqlen"],
                [src("865", "weights = self.weights_proj(hidden_states.to(w.dtype)).float() * "
                            "(self.n_heads**-0.5)")],
                "与打分头一一对应；乘 Ih^-0.5")
        self.tensor("g_dsa_ix_index_scores", "index_scores（各头加权求和）",
                    ["B", "seqlen", "Npool"], ["B", "seqlen", "Npool"], "activation",
                    "weights.unsqueeze(-2) @ scores（:866）")
        self.op("g_dsa_ix_sum", "Glm5NextTextIndexer.forward", "Indexer 头加权求和",
                "matmul+sum", ["g_dsa_ix_weights", "g_dsa_ix_relu"],
                ["g_dsa_ix_index_scores"], [], ["Ih", "Npool"],
                [src("866", "index_scores = torch.matmul(weights.unsqueeze(-2), scores)"
                            ".squeeze(-2)")],
                "把每头的 [B,S,Npool] 打分成一个 [B,S,Npool] 选择分数")
        self.tensor("g_dsa_ix_masked_scores", "index_scores（无效池 -inf）",
                    ["B", "seqlen", "Npool"], ["B", "seqlen", "Npool"], "activation",
                    "pool 不可见/不完整的位置被掩成 -inf（:868-880）")
        self.op("g_dsa_ix_mask", "Glm5NextTextIndexer.forward", "Indexer 池掩码",
                "masked_fill", ["g_dsa_ix_index_scores", "g_dsa_ix_valid_candidates"],
                ["g_dsa_ix_masked_scores"], [], ["B", "seqlen", "Npool"],
                [src("876-880", "index_scores = index_scores.masked_fill("
                                "~valid_candidates, torch.finfo(dtype).min)")],
                "只消费 valid_candidates（由 pool_indices→pool_end→pool_visible→"
                "&pool_valid 得到）；不直接用 pool_valid/visible，也不完整尾池由 tail 处理")
        self.tensor("g_dsa_ix_selected", "selected = topk(index_scores, select_k)",
                    ["B", "seqlen", "SelP"], ["B", "seqlen", "SelP"], "activation",
                    "选中的池索引（:883-888）")
        self.op("g_dsa_ix_topk", "Glm5NextTextIndexer.forward", "Indexer 池 TopK", "topk",
                ["g_dsa_ix_masked_scores"], ["g_dsa_ix_selected"], [],
                ["Ki", "Kpool", "SelP", "Npool"],
                [src("883-888", "select_k = min(index_topk // index_kpool, P); "
                                "index_scores.topk(select_k, dim=-1).indices")],
                "输出是池索引而不是掩码；Lidx≤Ki 时全选")
        self.tensor("g_dsa_ix_selected_valid", "selected_valid = valid_candidates[selected]",
                    ["B", "seqlen", "SelP"], ["B", "seqlen", "SelP"], "mask",
                    "被选中池的有效性；无效池会在这里被标记，再用于把展开 token 填 -1")
        self.op("g_dsa_ix_selected_valid", "Glm5NextTextIndexer.forward",
                "选中池有效性", "gather",
                ["g_dsa_ix_valid_candidates", "g_dsa_ix_selected"],
                ["g_dsa_ix_selected_valid"], [], ["SelP", "B", "seqlen"],
                [src("890-892", "selected_valid = valid_candidates.gather(-1, selected)")],
                "selected_valid 来自 valid_candidates 的同位置 gather；"
                "不能只凭 selected 索引判断有效性")
        self.tensor("g_dsa_ix_selected_indices", "pool_indices[batch, selected]",
                    ["B", "seqlen", "SelP", "Kpool"], ["B", "seqlen", "SelP", "Kpool"],
                    "activation", "把选中的池展开成 4 个原始 token 索引（:891-896）")
        self.op("g_dsa_ix_select_pools", "Glm5NextTextIndexer.forward", "池展开为原始索引",
                "gather", ["g_dsa_ix_pool_indices", "g_dsa_ix_selected"],
                ["g_dsa_ix_selected_indices"], [], ["SelP", "Kpool"],
                [src("891-896", "selected_indices = pool_indices[batch_idx, selected]; "
                                "topk_indices = selected_indices.flatten(-2)")],
                "池内 4 个 token 一起选中；flatten 后是 4×SelP")
        self.tensor("g_dsa_ix_flat_indices", "selected_indices.flatten(-2)",
                    ["B", "seqlen", "SelP×Kpool"], ["B", "seqlen", "SelP", "Kpool"],
                    "activation", "展开后的原始索引（:895-896）")
        self.op("g_dsa_ix_flatten", "Glm5NextTextIndexer.forward", "池索引展平", "flatten",
                ["g_dsa_ix_selected_indices"], ["g_dsa_ix_flat_indices"], [],
                ["SelP", "Kpool"],
                [src("895-896", "[B,S,K,P] -> [B,S,K*P]")],
                "K=512、P=4 → 最多 2048 个 token 索引")
        self.tensor("g_dsa_ix_valid_indices", "无效池位置 -1", ["B", "seqlen", "SelP×Kpool"],
                    ["B", "seqlen", "SelP", "Kpool"], "activation",
                    "selected_valid=False 的位置填 -1（:897-900）")
        self.op("g_dsa_ix_invalid", "Glm5NextTextIndexer.forward", "无效池位置填 -1",
                "masked_fill", ["g_dsa_ix_flat_indices", "g_dsa_ix_selected_valid"],
                ["g_dsa_ix_valid_indices"], [], ["SelP", "Kpool"],
                [src("893-900", "topk_indices.masked_fill(~selected_valid[..., None]"
                                ".expand_as(selected_indices).flatten(-2), -1)")],
                "selected_valid 先 expand 到每个池的 Kpool 个 token 再 flatten；"
                "没有 selected_valid 就没有合法的无效位掩码")
        self.tensor("g_dsa_ix_tail_indices", "tail_indices（当前不完整池）",
                    ["B", "seqlen", "Kpool-1"], ["B", "seqlen", "Kpool"], "activation",
                    "visible_count % Kpool 个最后可见 token（:1010-1060）")
        self.op("g_dsa_ix_tail", "Glm5NextTextIndexer.append_visible_tail",
                "不完整尾池 tail", "tail_append",
                ["g_dsa_ix_visible", "g_dsa_ix_valid_keys"], ["g_dsa_ix_tail_indices"], [],
                ["Kpool", "seqlen", "Lidx"],
                [src("1010-1060", "tail_count = visible_count % kpool; tail_start = "
                                  "first_key + visible_count - tail_count; tail 最多 Kpool-1 个")],
                "tail 是当前 query 所在的不完整池；它不属于候选池，因此不会被池打分/"
                "重复选择")
        self.tensor("g_dsa_ix_with_tail", "cat([topk_indices, tail])",
                    ["B", "seqlen", "SelP×Kpool+Kpool-1"],
                    ["B", "seqlen", "SelP", "Kpool"], "activation",
                    "池展开索引 + tail（:904）")
        self.op("g_dsa_ix_concat_tail", "Glm5NextTextIndexer.append_visible_tail",
                "追加 tail", "concat", ["g_dsa_ix_valid_indices", "g_dsa_ix_tail_indices"],
                ["g_dsa_ix_with_tail"], [], ["SelP", "Kpool"],
                [src("904-905", "topk_indices = self.append_visible_tail(topk_indices, ...)")],
                "tail 只追加当前不完整池的可见 token，不改变已选池集合")
        self.tensor("g_dsa_ix_padded_indices", "topk_indices（pad 后，尚未按 query 过滤）",
                    ["B", "seqlen", "Kout"], ["B", "seqlen", "Kout"], "activation",
                    "pad 到 Ki+Kpool-1=2051（:907-911）；无效/越界位置 -1")
        self.op("g_dsa_ix_pad", "Glm5NextTextIndexer.forward", "top-k 输出 pad", "pad",
                ["g_dsa_ix_with_tail"], ["g_dsa_ix_padded_indices"], [], ["Kout", "Ki", "Kpool"],
                [src("902-911", "output_width = index_topk (+ kpool-1); "
                                "F.pad(..., value=-1); [..., :output_width]")],
                "输出宽度固定 2051；mask 构建时忽略 -1")
        self.tensor("g_dsa_topk_indices", "topk_indices（int32，-1 填充；query padding 已过滤）",
                    ["B", "seqlen", "Kout"], ["B", "seqlen", "Kout"], "activation",
                    "pad 后再执行 topk_indices.masked_fill(~attention_mask[..., None], -1) "
                    "（:912-913）；padding query 的整行为 -1，不参与 mask counts")
        self.op("g_dsa_ix_query_mask", "Glm5NextTextIndexer.forward", "query padding 末端过滤",
                "masked_fill", ["g_dsa_ix_padded_indices", "g_attention_mask"],
                ["g_dsa_topk_indices"], [], ["B", "seqlen"],
                [src("912-913", "topk_indices = topk_indices.masked_fill("
                                "~attention_mask[..., None], -1)")],
                "保留 query padding 语义：padding query 的整个 top-k 行置 -1，"
                "后续 counts/mask 不会把 padding 当成可见 key")

        # 4) 稀疏 mask（True=可见）与 eager 注意力
        self.tensor("g_dsa_mask_counts", "selected_counts（每 query 每个 key 的计数）",
                    ["B", "seqlen", "Lkv"], ["B", "seqlen", "Lkv"], "activation",
                    "zeros + scatter_add_(-1, safe_indices, valid)（:1255-1262）")
        self.op("g_dsa_mask_counts", "Glm5NextTextAttention.build_attention_mask_from_topk",
                "稀疏索引 → counts", "scatter+add",
                ["g_dsa_topk_indices"], ["g_dsa_mask_counts"], [],
                ["B", "seqlen", "Lkv", "Kout"],
                [src("1251-1262", "topk_valid = ge(0) & lt(kv_length); safe_indices = clamp; "
                                  "selected_counts.scatter_add_(-1, safe, valid)")],
                "与 ones+scatter(False) 的语义相反：这里是 counts=0/1 再 ne(0)")
        self.tensor("g_dsa_sparse_mask", "mask = selected_counts.ne(0).unsqueeze(1)",
                    ["B", "1", "seqlen", "Lkv"], ["B", "seqlen", "Lkv"], "mask",
                    "True=可见（被选中的 key）；不是 True=屏蔽")
        self.op("g_dsa_sparse_mask", "Glm5NextTextAttention.build_attention_mask_from_topk",
                "counts → bool 可见 mask", "ne+unsqueeze", ["g_dsa_mask_counts"],
                ["g_dsa_sparse_mask"], [], ["B", "seqlen", "Lkv"],
                [src("1264-1265", "mask = selected_counts.ne(0).unsqueeze(1)")],
                "True=可见；SDPA 直接消费 bool，eager 再转 0/min")
        self.tensor("g_dsa_eager_mask", "mask = where(mask, 0, min_dtype)",
                    ["B", "1", "seqlen", "Lkv"], ["B", "seqlen", "Lkv"], "mask",
                    "eager 路径把 True→0、False→min（:1271-1275）")
        self.op("g_dsa_eager_mask", "Glm5NextTextAttention.build_attention_mask_from_topk",
                "eager mask 转换", "where", ["g_dsa_sparse_mask"], ["g_dsa_eager_mask"], [],
                ["B", "seqlen", "Lkv"],
                [src("1271-1275", "min_dtype = finfo(dtype).min; mask = torch.where(mask, 0.0, "
                                  "min_dtype)")],
                "本图表达 eager 路径；SDPA 返回 bool mask，语义相同")
        self.tensor("g_dsa_attn_logits", "attn_weights = QKᵀ×scaling",
                    ["B", "Nh", "seqlen", "Lkv"], ["B", "Nh", "seqlen", "Lkv"], "activation",
                    "matmul(query, key.transpose(2,3))*scaling（:1088）")
        self.op("g_dsa_qk", "eager_attention_forward", "QKᵀ 缩放", "matmul+scale",
                ["g_dsa_q", "g_dsa_key"], ["g_dsa_attn_logits"], [],
                ["Nh", "seqlen", "Lkv", "Dq"],
                [src("1088", "attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * "
                             "scaling"),
                 src("1147", "self.scaling = self.qk_head_dim ** (-0.5)")],
                "Nkv=Nh → repeat_kv 恒等（:1085-1087）；scaling=256^-0.5=1/16，NoPE 无 mscale")
        self.tensor("g_dsa_attn_masked", "attn_weights + attention_mask",
                    ["B", "Nh", "seqlen", "Lkv"], ["B", "Nh", "seqlen", "Lkv"], "activation",
                    "加 0/min 稀疏 mask（:1089-1090）")
        self.op("g_dsa_mask_add", "eager_attention_forward", "加稀疏可见 mask", "mask_add",
                ["g_dsa_attn_logits", "g_dsa_eager_mask"], ["g_dsa_attn_masked"], [],
                ["B", "Nh", "seqlen", "Lkv"],
                [src("1089-1090", "if attention_mask is not None: attn_weights = "
                                  "attn_weights + attention_mask")],
                "mask 与 logits 同形 [B,Nh,S,Lkv]（mask 的 1 广播到 Nh）")
        self.tensor("g_dsa_attn_probs", "attn_weights = softmax(..., dim=-1, dtype=float32)",
                    ["B", "Nh", "seqlen", "Lkv"], ["B", "Nh", "seqlen", "Lkv"], "activation",
                    "softmax 在 fp32 上计算后转回 query dtype（:1092）")
        self.op("g_dsa_softmax", "eager_attention_forward", "Softmax", "softmax",
                ["g_dsa_attn_masked"], ["g_dsa_attn_probs"], [], ["Nh", "seqlen", "Lkv"],
                [src("1092", "nn.functional.softmax(attn_weights, dim=-1, "
                             "dtype=torch.float32).to(query.dtype)")],
                "对 Lkv 个位置（含未被选中的 min 值）归一化；被 mask 的位置概率为 0")
        self.tensor("g_dsa_attn_head_out", "attn_output（AV 后，仍是 B,Nh,seqlen,Dv）",
                    ["B", "Nh", "seqlen", "Dv"], ["B", "Nh", "seqlen", "Dv"], "activation",
                    "AV 乘积（:1094）；头维是 v_head_dim=256，尚未 transpose")
        self.op("g_dsa_av", "eager_attention_forward", "AV 加权求和", "matmul",
                ["g_dsa_attn_probs", "g_dsa_v"], ["g_dsa_attn_head_out"], [],
                ["Nh", "seqlen", "Dv", "Lkv"],
                [src("1094", "attn_output = torch.matmul(attn_weights, value_states)")],
                "AV 输出布局仍是 [B,Nh,seqlen,Dv]")
        self.tensor("g_dsa_attn_heads_seq", "attn_output.transpose(1,2).contiguous()",
                    ["B", "seqlen", "Nh", "Dv"], ["B", "seqlen", "Nh", "Dv"], "activation",
                    "eager 在 AV 后显式 transpose(1,2).contiguous()（:1095）")
        self.op("g_dsa_attn_transpose", "eager_attention_forward", "AV 输出转置+连续化",
                "transpose+contiguous", ["g_dsa_attn_head_out"], ["g_dsa_attn_heads_seq"], [],
                ["B", "seqlen", "Nh", "Dv"],
                [src("1095", "attn_output = attn_output.transpose(1, 2).contiguous()")],
                "把头维移到 seqlen 之后，下一步才能按 seqlen reshape")
        self.tensor("g_dsa_attn_flat", "attn_output.reshape(B,seqlen,Nh*Dv)",
                    ["B", "seqlen", "Nh×Dv"], ["B", "seqlen", "Nh", "Dv"], "activation",
                    "在 transpose+contiguous 之后拼头（:1233）")
        self.op("g_dsa_attn_reshape", "Glm5NextTextAttention.forward", "注意力输出拼头",
                "reshape", ["g_dsa_attn_heads_seq"], ["g_dsa_attn_flat"], [],
                ["Nh", "Dv"],
                [src("1233", "attn_output = attn_output.reshape(batch_size, seq_length, -1)"
                             ".contiguous()")],
                "输入已是 [B,seqlen,Nh,Dv]；reshape 只把 Nh×Dv 并成一维=16384")
        self.weight("g_dsa_o_proj_weight", "self.self_attn.o_proj.weight", ["H", "Nh×Dv"],
                    ["H", "Nh", "Dv"],
                    "nn.Linear(num_heads*v_head_dim, hidden_size, bias=False)（:1142-1146）")
        self.tensor("g_attn_out", "self.o_proj(attn_output)", ["B", "seqlen", "H"],
                    ["B", "seqlen", "H"], "activation", "输出投影（:1234）")
        self.op("g_dsa_o_proj", "Glm5NextTextAttention.o_proj", "输出投影 o_proj", "linear",
                ["g_dsa_attn_flat"], ["g_attn_out"], ["g_dsa_o_proj_weight"],
                ["Nh", "Dv", "H", "B", "seqlen"],
                [src("1142-1146", "self.o_proj = nn.Linear(num_heads*v_head_dim, "
                                  "hidden_size, bias=attention_bias)"),
                 src("1233-1234", "attn_output.reshape(...); attn_output = self.o_proj(attn_output)")],
                "bias=False；输出回到 hidden_size=4096")

        # 5) mHC 残差（attn 段）+ post norm
        self._mhc_apply("attn", "g_attn_out", "g_hidden_streams_in",
                        "g_hidden_streams_mid")
        self.tensor("g_hidden_streams_mid", "hidden_states（attn 段之后，4 路流）",
                    ["B", "seqlen", "Hc", "H"], ["B", "seqlen", "Hc", "H"], "activation",
                    "post*attn_out + combᵀ@residual；仍是 4 路流")
        self._mhc_site("ffn", "g_hidden_streams_mid",
                       residual_note="坍缩后的单流进入 post_attention_layernorm；"
                                     "FFN 残差由 post/comb 混合")
        self.tensor("g_ffn_in", "self.post_attention_layernorm(collapsed)",
                    ["B", "seqlen", "H"], ["B", "seqlen", "H", "EpsN"], "activation",
                    "RMSNorm（eps=rms_norm_eps）（:1342）")
        self.op("g_post_norm", "Glm5NextTextDecoderLayer.post_attention_layernorm",
                "Post-Attention RMSNorm", "rmsnorm", ["g_mhc_ffn_collapsed"], ["g_ffn_in"],
                ["g_post_norm_weight"], ["H", "EpsN"],
                [src("1339-1342", "residual = hidden_states; post, comb, hidden_states = "
                                  "ffn_hc(...); hidden_states = self.post_attention_layernorm(...)")],
                "与 input_layernorm 同为 RMSNorm，但权重独立")

        # 6) MoE
        self._moe_ops("g_ffn_in")

        # 7) mHC 残差（FFN 段）
        self._mhc_apply("ffn", "g_mlp_out", "g_hidden_streams_mid",
                        "g_hidden_streams_out")
        self.tensor("g_hidden_streams_out", "hidden_states（层输出，4 路流）",
                    ["B", "seqlen", "Hc", "H"], ["B", "seqlen", "Hc", "H"], "boundary",
                    "第二次 mHC 混合后的层输出；写入 next hidden_states（:1344-1348）")

    def _build_cache_notes(self) -> None:
        self.cache_notes = [
            "层缓存类：config.layer_types 规范后为 indexed_attention（DSA），dispatch 到 "
            "DynamicIndexedLayer（cache_utils:1285-1294）。",
            "主 KV 缓存：update(k_pass, k_rot) 写入**压缩 latent** —— keys=[B,1,Lkv,512]、"
            "values=[B,1,Lkv,0]（NoPE）（modeling:1194-1198；cache_utils:132-153）。"
            "expand_kv/kv_b_proj 在每次调用对全部 Lkv 个位置展开（:1155-1172），"
            "缓存本身不存展开后的 64 头 K/V。",
            "indexer packed 缓存：[B,Lidx,257]（k 128 + gate 128 + valid 1），"
            "由 update_indexer 线性追加（modeling:841-849；cache_utils:350-365）；"
            "11 个 DSA 层各自实例化 indexer 与自己的 packed 缓存。",
            "Lkv = Lidx = kvcachelen + seqlen：主 KV 与 packed 缓存都线性保留全部历史。"
            "kvcachelen 是调用前逻辑历史 token 数（空缓存 Prefill 取 0），"
            "不是字节数或物理保留条目数。",
            "稀疏可见集合：每 query 选 SelP 个池，展开 4×SelP 个 token，再追当前不完整池的 "
            "tail（≤3）；K_sel = min(Lidx, Ki + (Lidx % Kpool))，不是固定 2048。"
            "具体选中哪些池由打分数据决定，图上不猜具体 token。",
            "没有 KDA 的 conv/recurrent state；DSA 层没有固定的 O(1) 状态，"
            "两套缓存随历史增长。",
        ]

    def _build_coverage_notes(self) -> None:
        v = self.values
        kda = [i for i in range(v.num_hidden_layers) if v.layer_type(i) == "linear_attention"]
        dsa = [i for i in range(v.num_hidden_layers) if v.layer_type(i) == "indexed_attention"]
        self.coverage_notes = [
            "已表达：mHC 两处（attn/ffn）的 4 路流坍缩/pre/post/comb Sinkhorn/展开/混合；"
            "MLA 的 Q 低秩、NoPE（无 RoPE 调用）、压缩 latent 缓存、kv_b 展开、"
            "DSA indexer 的 k-pool 压缩/池 top-k/tail/packed 缓存、"
            "selected counts 稀疏 mask、eager 注意力、TopK MoE + 并行共享专家（含 SwiGLU clamp）。",
            f"真实层归属：DSA 层 {len(dsa)} 个（{dsa}）；KDA 层 {len(kda)} 个；"
            "dense 层 0/1/2。本 Block 代表 DSA+MoE 组模板。",
            "根配置 quantization_config 声明 FP8（e4m3, block 128×128）：本图只表达 HF 逻辑 "
            "eager 路径，不分析量化 kernel、FP8 数值或权重内容。",
            "未覆盖：MTP 层 45（modeling:1378 显式忽略 + forward 只循环 0..44）、视觉 24 层、"
            "generate/采样循环、loss/aux_loss、output_router_logits 辅助输出、"
            "torch.compile/梯度检查点、数值等价。",
            "数据相关量（N_hit、N_e、具体选中的池/token、路由实际分布）保留未知；"
            "不根据权重推断 MoE 实际路由，不默认 0。专家只有一份模板，不按 288 个专家复制图。",
            "NoPE：源码没有 RoPE 调用，也没有 4D 因果 mask；DSA 的因果性由 indexer 的 "
            "kv_pos ≤ q_pos + 池/tail 选择保证。本模型没有部分 RoPE；注意力头维全部是 nope。",
        ]
