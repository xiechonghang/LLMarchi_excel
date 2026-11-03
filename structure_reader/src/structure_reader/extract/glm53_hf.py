"""GLM-5.3（zai-org/GLM-5.3, HF transformers ``glm_moe_dsa``）单 Block 结构提取。

从 ``rules/glm53_hf.py`` 解释的有效配置与本地官方 Transformers 检出源码，构建
**一份** 参数化 Block 结构（``hf_layer_structure.v1``），供分层浏览页与同源工作簿
使用；同一份结构再由 ``extract/hf_scenario.py`` 投影成既有
``hf_decoder_reading.v1`` 的 prefill/decode 场景。本模块不生成渲染文件、不执行
模型、不加载权重、不下载权重。

本配置只存在三种真实层型组合（不虚构第四种）：

- ``full-dense``：full DSA indexer + dense MLP（层 0/1/2，first_k_dense_replace=3）；
- ``full-moe``：full DSA indexer + TopK MoE（每 4 层一个 full，其余 shared）；
- ``shared-moe``：shared DSA indexer + TopK MoE（不实例化 indexer，复用上一
  full 层的 top-k）。

结构事实（行号按本地检出 536ecc00…，文件 modeling_glm_moe_dsa.py，下称 modeling；
配置规则见 configuration_glm_moe_dsa.py，下称 configuration）：

- DecoderLayer（modeling :586-616）：``residual = hidden_states`` →
  ``input_layernorm`` → Attention → ``residual + attn`` → ``post_attention_layernorm``
  → MLP/MoE → 第二次 ``residual + mlp``。两处都是**普通残差相加**：本模型没有
  mHC/HyperConnection/多流/Sinkhorn，也没有滑动窗口或压缩注意力。
- MLA Attention（modeling :370-451）：q_a_proj→q_a_layernorm→q_b_proj→
  view(B,S,Nh,Dq).transpose(1,2)→split(qk_nope=192, qk_rope=64)→**交错**部分 RoPE
  →cat；kv_a_proj_with_mqa→split(512,64)→kv_a_layernorm(512)→k_rot view(B,1,S,64)
  →交错 RoPE→``past_key_values.update(k_pass, k_rot, layer_idx)``：缓存里存的是
  **压缩 latent**（keys=512、values=64），不是展开后的 K/V；使用前
  ``kv_b_proj`` 对全部 Lkv 个位置展开（:402/:351-368），再 cat 出 key=cat(k_nope,
  k_rot)（:365-368）。o_proj 前 reshape 为 [B,S,Nh*Dv]。
- DSA indexer（modeling :153-242）：wq_b(q_residual)→view(B,S,Ih,Id)→split(64,64)
  →交错 RoPE；wk(hidden_states)→k_norm(LayerNorm 128, **带 bias**)→交错 RoPE→
  ``update_indexer`` 追加 [B,Lidx,Id] 并返回全量；打分
  ``relu(q·kᵀ*Id^-0.5)`` → ``weights_proj(hidden)*Ih^-0.5`` 加权求和 → 加
  attention_mask[:,0] → ``topk(min(index_topk, Lidx))``。shared 层不执行这些算子，
  消费上一层 full indexer 的 ``prev_topk_indices``（:404-417）。
- 稀疏 mask（modeling :419-430）：``new_ones([B,S,Lkv]).scatter(-1, topk, False)``
  → 与因果 mask 合并（bool 用 ``& ~``、float 用 ``masked_fill(min)``）；eager 路径
  仍对全长度 Lkv 做 QKᵀ，只是被 mask 掉未选中的 key。
- eager 注意力（modeling :257-279/:434-447）：repeat_kv（Nkv=Nh → 恒等）→
  QKᵀ×scaling（1/16）→+mask→softmax(fp32)→AV→transpose(1,2)。
- 稠密 MLP（modeling :454-467）：gate/up（Is=12288）→SiLU(gate)*up→down。
- MoE（modeling :470-572）：flatten→router ``sigmoid(linear(float32))``→
  +e_score_correction_bias→group_scores（n_group=1，源码仍执行组选择但单组掩码
  全通过）→topk 8→gather 权重→归一(+1e-20)→×2.5；专家为单份模板
  （gate_up [E,2Es,H]、down [E,H,Es]，Es=2048），``one_hot(K,E+1)`` 命中的专家
  逐个 einsum；共享专家（宽度 Es×n_shared_experts=2048）与 routed 并行相加。
- 缓存（cache_utils :113-172/:331-365）：层型 ``indexed_attention`` →
  ``DynamicIndexedLayer``；主 KV ``DynamicLayer.update`` 以 concat 追加
  （keys=[B,1,Lkv,512]、values=[B,1,Lkv,64]）；indexer 键缓存 [B,Lidx,128]
  只在 full 层存在。Lkv=Lidx=kvcachelen+seqlen，无窗口截断、无压缩条目。

场景（seqlen/kvcachelen）只绑定取值：结构 shape 保留符号，空缓存 Prefill 的
kvcachelen=0；Decode 场景 seqlen=1。数据相关量（命中专家数、每专家 token 数）
保留未知，不默认 0。
"""

from __future__ import annotations

from pathlib import Path

from structure_reader.extract.hf_scenario import STRUCTURE_SCHEMA
from structure_reader.rules.glm53_hf import (
    GLM53_SCENARIOS,
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

SAMPLE_LAYER_INDEX = 0  # 第一组 full-dense（其余组由层组容器逐组构建）
SCHEMA = "hf_decoder_reading.v1"

# 场景派生式（白名单：整数/变量/+ - * // %/min/max）。参数 expression 与
# derivations 共用同一份字符串，避免同一条公式写两遍而漂移。
_EXPR_LKV = "kvcachelen + seqlen"
_EXPR_LIDX = "kvcachelen + seqlen"
_EXPR_LMASK = "kvcachelen + seqlen"
_EXPR_KSEL = "min(Ki, L_idx)"
_EXPR_TTOK = "B * seqlen"

_SCOPE_NOTES = [
    "范围：单个 decoder Block（同一层组只保留一份模板；真实层索引、顺序与循环次数见整网层组）。",
    "实现路径：transformers eager（本地 glm_moe_dsa 实现；_supports_flash_attn=False、"
    "_supports_sdpa=True，modeling:626-628）。sparse_indices 交给 flash_mla 的核路径不在本图；"
    "本图表达 eager/sdpa 的 mask 语义。",
    "静态结构一致为目标：不加载权重、不运行模型、不做数值等价；shape 保留符号 "
    "seqlen/kvcachelen，场景预设只绑定取值，不改变结构。",
    "同一权重/激活张量可被多个算子引用（共享引用非重复定义）；缓存调用前/调用后以两个"
    "张量分别表达，不混为原地更新。",
    "本模型是多模态仓库中的**语言主干**；视觉/图像输入路径不在本票范围，本结构不含图像编码器。",
]


class ReadingDocError(ValueError):
    """阅读数据构建失败（配置/范围/事实来源不一致），具名拒绝。"""


def _block_builder(values: HfModelValues, layer_index: int) -> "Glm53BlockBuilder":
    """按真实层型组合选择 Block 结构构建器（三种组合共用一套基类）。"""
    indexer_type = values.indexer_type(layer_index)
    mlp_type = values.mlp_layer_type(layer_index)
    if indexer_type == "full" and mlp_type == "dense":
        return Glm53BlockBuilder(values, layer_index)
    if indexer_type == "full" and mlp_type == "sparse":
        return Glm53BlockBuilder(values, layer_index)
    if indexer_type == "shared" and mlp_type == "sparse":
        return Glm53BlockBuilder(values, layer_index)
    raise ReadingDocError(
        f"layer_index={layer_index} 是 {indexer_type} + {mlp_type}；"
        "GLM-5.3 已核实的真实组合为 full+dense、full+sparse、shared+sparse，"
        "不虚构第四种组合"
    )


def build_block_structure(values: HfModelValues, layer_index: int) -> dict:
    """构建**一份** Block 结构（参数/张量/算子/缓存注记/派生量），不含阶段复制。"""
    if not 0 <= layer_index < values.num_hidden_layers:
        raise ReadingDocError(
            f"layer_index {layer_index} 越界（0..{values.num_hidden_layers - 1}）"
        )
    builder = _block_builder(values, layer_index)
    block = builder.build()
    block["layer_index"] = layer_index
    block["attention_type"] = values.layer_type(layer_index)
    block["moe_type"] = values.mlp_layer_type(layer_index)
    block["indexer_type"] = values.indexer_type(layer_index)
    return block


def build_layer_documents(
    workspace_root: str | Path,
    layer_index: int = SAMPLE_LAYER_INDEX,
    model_key: str = "glm_5_3_hf",
) -> tuple[dict, dict]:
    """一次构建，两种同源表示：一份 Block 结构 + prefill/decode 场景投影。"""
    from structure_reader.extract.hf_scenario import (
        project_legacy_phase,
        structure_document,
    )

    values = load_hf_model_values(workspace_root, model_key=model_key)
    block = build_block_structure(values, layer_index)
    block["scope_notes"] = list(_SCOPE_NOTES)
    structure = structure_document(values, block, scenarios=GLM53_SCENARIOS)
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
                f"{block['indexer_type']} DSA indexer + {block['moe_type']}）；"
                "整网 Embedding/78 层展开/Final RMSNorm/LM head 见整网总览。",
                "模型身份：zai-org/GLM-5.3 revision "
                f"{MODEL_REPOSITORY['revision']}，model_type=glm_moe_dsa；"
                f"配置保存时的 transformers_version="
                f"{values.checkpoint_transformers_version}，实现取本地官方检出。",
                "实现路径：transformers eager；根配置声明 FP8 量化（quantization_config），"
                "本样板只表达 HF 逻辑 eager 路径，不分析量化执行与设备 kernel。",
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
    model_key: str = "glm_5_3_hf",
) -> dict:
    """构建 ``hf_layer_structure.v1``：一份 Block 结构 + 场景预设/派生值。"""
    return build_layer_documents(workspace_root, layer_index, model_key)[0]


def build_reading_document(
    workspace_root: str | Path,
    layer_index: int = SAMPLE_LAYER_INDEX,
    model_key: str = "glm_5_3_hf",
) -> dict:
    """既有冻结接口 ``hf_decoder_reading.v1``：同一份结构的 prefill/decode 投影。"""
    return build_layer_documents(workspace_root, layer_index, model_key)[1]


class Glm53BlockBuilder:
    """单 Block 结构累积器：参数 → 边界/缓存张量 → 算子按真实阅读顺序追加。"""

    def __init__(self, values: HfModelValues, layer_idx: int):
        self.values = values
        self.layer_idx = layer_idx
        self.indexer_type = values.indexer_type(layer_idx)
        self.mlp_type = values.mlp_layer_type(layer_idx)
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
              unknown_reason: str = "") -> str:
        self.parameters.append({
            "id": pid, "symbol": symbol, "meaning": meaning, "value": value,
            "expression": expression, "note": note, "source_refs": source_refs,
            # 静态未知（数据相关量）：工作簿写“未知（原因）”，不写 null/0。
            **({"unknown_reason": unknown_reason} if unknown_reason else {}),
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
        return {"path": SOURCE_FILES["modeling_glm_moe_dsa.py"],
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
        v = self.values
        combo = f"{self.indexer_type} DSA indexer + {self.mlp_type}"
        title = (
            f"GLM-5.3 Decoder Block（layer_idx={self.layer_idx}，{combo}；"
            f"{v.num_hidden_layers} 层主干代表模板；同一结构由 seqlen/kvcachelen 场景参数化）"
        )
        self._build_parameters()
        self._build_derivations()
        self._build_boundary_and_cache()
        self._decoder_layer_flow()
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

    # ------------------------------------------------------------------
    # 参数区
    # ------------------------------------------------------------------

    def _build_parameters(self) -> None:
        v = self.values
        cfg, src = self.cfg, self.src
        self.param("B", "B", "batch 批大小", None, None,
                   "调用输入维度，静态分析保留符号", [])
        self.param("seqlen", "seqlen", "本次输入的新 token 数（seqlen）", None, None,
                   "Prefill 为提示长度、Decode 为 1；结构保留符号，场景预设只绑定取值", [])
        self.param("kvcachelen", "kvcachelen", "调用前逻辑历史 token 数（kvcachelen）", None,
                   "P = past_key_values.get_seq_length()（无 cache 时 0）",
                   "空缓存 Prefill 取 0；Decode 为调用前已处理历史；"
                   "不是物理保留条目数或缓存字节数。本模型**没有**滑动窗口/压缩条目，"
                   "主 KV 与 indexer 缓存都线性保留全部历史",
                   [src("691-694", "past_seen_tokens = past_key_values.get_seq_length()")])
        self.param("H", "H", "hidden_size 隐层维度", v.hidden_size, None,
                   "config.hidden_size", [cfg("hidden_size", "根配置字段")])
        self.param("Nh", "Nh", "num_attention_heads query 头数", v.num_attention_heads, None,
                   "config.num_attention_heads",
                   [cfg("num_attention_heads", "根配置字段")])
        self.param("Nkv", "Nkv", "num_key_value_heads KV 头数", v.num_key_value_heads, None,
                   "config.num_key_value_heads；=Nh → repeat_kv 为恒等（不复制 KV 头）",
                   [cfg("num_key_value_heads", "根配置字段"),
                    src("245-254", "repeat_kv：n_rep==1 直接返回")])
        self.param("Dq", "Dq", "qk_head_dim 注意力头维（nope+rope）", v.qk_head_dim,
                   "Dq = Dn + Rk",
                   "派生 = 192+64；注意力缩放为 Dq^-0.5 = 1/16",
                   [cfg("qk_head_dim", "根配置字段（与派生一致）"),
                    self.srcf("configuration_glm_moe_dsa.py", "149",
                              "self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim")])
        self.param("Dn", "Dn", "qk_nope_head_dim 不旋转的 q/k 头维", v.qk_nope_head_dim, None,
                   "config.qk_nope_head_dim；q/k 的 nope 段与 value 段维数不同（192 vs 256）",
                   [cfg("qk_nope_head_dim", "根配置字段")])
        self.param("Rk", "Rk", "qk_rope_head_dim 旋转维（=运行期 head_dim）", v.qk_rope_head_dim,
                   "Rk = config.qk_rope_head_dim（运行期 head_dim）",
                   "只旋转 q/k 的末 64 维（32 对，交错）；根配置 head_dim=192 在 "
                   "__post_init__ 被覆盖为 64，不参与 RoPE",
                   [cfg("qk_rope_head_dim", "根配置字段"),
                    self.srcf("configuration_glm_moe_dsa.py", "150-152",
                              "self.head_dim = self.qk_rope_head_dim"),
                    src("142-149", "cos/sin 取前一半，按 0::2/1::2 交错")])
        self.param("Dv", "Dv", "v_head_dim value 头维", v.v_head_dim, None,
                   "config.v_head_dim；kv_b_proj 输出 Nh×(Dn+Dv) 后拆成 k_nope/v",
                   [cfg("v_head_dim", "根配置字段"),
                    src("358-361", "key_shape=(...); split([qk_nope, v_head])")])
        self.param("Rq", "Rq", "q_lora_rank Q 低秩中间维", v.q_lora_rank, None,
                   "config.q_lora_rank；q_a_layernorm 的 RMSNorm 维（eps=1e-6）",
                   [cfg("q_lora_rank", "根配置字段"),
                    src("324-326", "q_a_proj → q_a_layernorm → q_b_proj")])
        self.param("Rkv", "Rkv", "kv_lora_rank 压缩 KV latent 维", v.kv_lora_rank, None,
                   "config.kv_lora_rank；缓存 k_pass 的最后一维（512），k_rot 另存 64",
                   [cfg("kv_lora_rank", "根配置字段"),
                    src("328-332", "kv_a_proj_with_mqa 输出 kv_lora_rank + qk_rope_head_dim"),
                    src("396-398", "缓存写入压缩 latent")])
        self.param("Ih", "Ih", "index_n_heads indexer 打分头数", v.index_n_heads, None,
                   "config.index_n_heads；full 层才有 indexer 模块",
                   [cfg("index_n_heads", "根配置字段"),
                    src("178-181", "wq_b/wk/weights_proj 形状")])
        self.param("Id", "Id", "index_head_dim indexer 头维", v.index_head_dim, None,
                   "config.index_head_dim；k_norm 是**带 bias 的 LayerNorm**（不是 RMSNorm）",
                   [cfg("index_head_dim", "根配置字段"),
                    src("180", "k_norm = nn.LayerNorm(head_dim, eps=1e-6)")])
        self.param("Ki", "Ki", "index_topk 每 query 选择的 key 数上限", v.index_topk, None,
                   "config.index_topk；实际 topk 数 = min(Ki, Lidx)",
                   [cfg("index_topk", "根配置字段"),
                    src("241-242", "topk = min(self.index_topk, index_scores.shape[-1])")])
        self.param("E", "E", "n_routed_experts 路由专家总数", v.n_routed_experts, None,
                   "config.n_routed_experts（标准名 num_local_experts 经 attribute_map 映射）",
                   [cfg("n_routed_experts", "根配置字段"),
                    self.srcf("configuration_glm_moe_dsa.py", "90", "attribute_map"),
                    src("519-521", "gate_up_proj [E,2Es,H] / down_proj [E,H,Es]")])
        self.param("K", "K", "num_experts_per_tok 每 token 选择数", v.num_experts_per_tok, None,
                   "config.num_experts_per_tok；与专家总数 E 不同参数",
                   [cfg("num_experts_per_tok", "根配置字段"),
                    src("502", "topk(scores_for_choice, k=self.top_k)")])
        self.param("Es", "Es", "moe_intermediate_size 单专家中间维", v.moe_intermediate_size, None,
                   "config.moe_intermediate_size；共享专家宽度 = Es×n_shared_experts",
                   [cfg("moe_intermediate_size", "根配置字段"),
                    src("561-563", "shared_experts = GlmMoeDsaMLP(moe_intermediate_size*n_shared_experts)")])
        self.param("Is", "Is", "intermediate_size 稠密 MLP 中间维", v.intermediate_size, None,
                   "config.intermediate_size；只用于 dense 层（0..first_k_dense_replace-1）",
                   [cfg("intermediate_size", "根配置字段"),
                    src("455-463", "GlmMoeDsaMLP 用 config.intermediate_size")])
        self.param("Scl", "Scl", "routed_scaling_factor 路由权重缩放", v.routed_scaling_factor,
                   None, "config.routed_scaling_factor；归一后乘 2.5",
                   [cfg("routed_scaling_factor", "根配置字段"),
                    src("507", "topk_weights = topk_weights * routed_scaling_factor")])
        self.param("Ng", "Ng", "n_group 路由专家组数", v.n_group, None,
                   "config.n_group；=1 时源码仍执行组选择但单组掩码全通过，不改变选择集合",
                   [cfg("n_group", "根配置字段"),
                    src("488-501", "group_scores → group_idx → score_mask")])
        self.param("Ktg", "Ktg", "topk_group 选中的专家组数", v.topk_group, None,
                   "config.topk_group；与 Ng=1 一致",
                   [cfg("topk_group", "根配置字段"),
                    src("493", "torch.topk(group_scores, k=self.topk_group)")])
        self.param("EpsN", "EpsN", "rms_norm_eps 主 RMSNorm epsilon", v.rms_norm_eps, None,
                   "config.rms_norm_eps；input/post_attention/final RMSNorm",
                   [cfg("rms_norm_eps", "根配置字段"),
                    src("583-584", "input_layernorm/post_attention_layernorm(config.rms_norm_eps)")])
        self.param("EpsA", "EpsA", "子层 RMSNorm/LayerNorm epsilon", 1e-6, None,
                   "源码默认值，不是根配置字段：q_a_layernorm/kv_a_layernorm 用 "
                   "GlmMoeDsaRMSNorm 默认 eps，indexer k_norm 用 LayerNorm(eps=1e-6)",
                   [src("325", "GlmMoeDsaRMSNorm(self.q_lora_rank)（默认 eps=1e-6）"),
                    src("333", "GlmMoeDsaRMSNorm(config.kv_lora_rank)"),
                    src("180", "nn.LayerNorm(self.head_dim, eps=1e-6)")])
        self.param("ThetaM", "ThetaM", "rope_theta RoPE 底数", v.rope_theta, None,
                   "config.rope_parameters.rope_theta=8e6（default 型，无 YaRN mscale）",
                   [cfg("rope_parameters", "根配置字段"),
                    src("84-101", "compute_default_rope_parameters")])
        self.param("Vocab", "Vocab", "vocab_size 词表大小", v.vocab_size, None,
                   "config.vocab_size；embedding/lm_head（tie_word_embeddings=false）",
                   [cfg("vocab_size", "根配置字段")])
        self.param("Lkv", "L_kv", "本次可读主 KV 长度（压缩 latent 长度）", None,
                   "L_kv = kvcachelen + seqlen（无滑动窗口/压缩截断）",
                   "update 后 keys/values 线性追加全部历史；kv_b_proj 对全部 Lkv 个位置展开",
                   [src("396-402", "update 后 expand_kv(k_pass, k_rot)"),
                    self.srcf("cache_utils.py", "132-153", "DynamicLayer.update：concat")])
        self.param("Lidx", "L_idx", "本次可读 indexer 键缓存长度", None,
                   "L_idx = kvcachelen + seqlen（与 Lkv 同式，但来自 indexer 键缓存）",
                   "只有 full indexer 层有该缓存；shared 层复用上一 full 层的 top-k",
                   [src("225-226", "k = past_key_values.update_indexer(k, self.layer_idx)"),
                    self.srcf("cache_utils.py", "350-365", "DynamicIndexedLayer.update_indexer")])
        self.param("Lmask", "L_mask", "模型级因果 mask 的 KV 长度", None,
                   "L_mask = kvcachelen + seqlen（allow_is_causal_skip=False）",
                   "网络入口构建一次 float mask；层内再按 top-k 改成稀疏 mask",
                   [src("696-706", "create_causal_mask(..., allow_is_causal_skip=False)"),
                    self.srcf("cache_utils.py", "165-172", "get_mask_sizes：kv_length=已缓存+query")])
        self.param("Ksel", "K_sel", "每 query 实际选择的 key 数", None,
                   "K_sel = min(Ki, L_idx)",
                   "Lidx 不超过 Ki 时 top-k 返回全部位置（稀疏 mask 不删任何位置）；"
                   "超过 Ki 时只保留 Ki 个",
                   [src("241-242", "topk = min(index_topk, index_scores.shape[-1])")])
        self.param("Ttok", "T_tok", "本调用展平 token 数（MoE 输入行数）", None,
                   "T_tok = B * seqlen",
                   "MoE 把 [B,seqlen,H] 展平成 [T_tok,H] 再路由",
                   [src("484-485", "hidden_states.view(-1, hidden_dim)")])
        self.param("N_hit", "N_hit", "路由实际命中的不同专家数（数据相关）", None, None,
                   "数据相关量，未给定；不根据权重推断 MoE 实际路由，不默认 0 或 E",
                   [src("532-535", "one_hot(...).sum→nonzero：命中专家集合")],
                   unknown_reason="数据相关：路由命中集合未给定；不默认 0/E")
        self.param("N_e", "N_e", "单专家在本调用处理的 token 数（数据相关）", None, None,
                   "数据相关量，未给定；专家模板只有一份，不按 E 复制行",
                   [src("540-542", "top_k_pos, token_idx = torch.where(expert_mask[expert_idx])")],
                   unknown_reason="数据相关：每专家 token 数未给定；不默认 0")

    def _build_derivations(self) -> None:
        premise = ("按固定配置 + 「空态起步、连续正常执行」前提推导；"
                   "不覆盖裁剪/回滚后的自定义缓存快照")
        self.derivation("L_kv", "本次可读主 KV 长度（压缩 latent）", _EXPR_LKV,
                        f"kvcachelen 为调用前逻辑历史；本模型无滑动窗口/压缩条目。{premise}")
        self.derivation("L_idx", "本次可读 indexer 键缓存长度", _EXPR_LIDX,
                        f"与主 KV 同式（两套缓存都线性保留）；仅 full indexer 层存在。{premise}")
        self.derivation("L_mask", "模型级因果 mask 的 KV 长度", _EXPR_LMASK,
                        f"get_mask_sizes 的 kv_length=已缓存+query；层内稀疏 mask 复用该长度。{premise}")
        self.derivation("K_sel", "每 query 实际选择 key 数", _EXPR_KSEL,
                        "topk(min(index_topk, Lidx))：不超过 Ki 时全选，超过时截到 Ki。"
                        f"{premise}")
        self.derivation("T_tok", "MoE 路由的展平 token 数", _EXPR_TTOK,
                        f"B×seqlen；B 为可填性能配置。{premise}")

    # ------------------------------------------------------------------
    # 边界与缓存载体
    # ------------------------------------------------------------------

    def _build_boundary_and_cache(self) -> None:
        v = self.values
        self.tensor(
            "g_hidden_in", "hidden_states（层输入）", ["B", "seqlen", "H"],
            ["B", "seqlen", "H"], "boundary",
            "上一层输出或 Embedding 输出；DecoderLayer.forward 的 hidden_states（modeling:597）")
        self.tensor(
            "g_position_ids", "position_ids（默认 arange+past）", ["1", "seqlen"],
            ["seqlen", "kvcachelen"], "boundary",
            "网络入口默认 arange(seqlen)+past_seen 后 unsqueeze(0)（modeling:691-694）；"
            "调用方可传 [B,seqlen]（本图按默认路径）")
        self.tensor(
            "g_attention_mask", "attention_mask（因果 float mask）", ["B", "1", "seqlen", "Lmask"],
            ["B", "seqlen", "Lmask"], "boundary",
            "create_causal_mask(..., allow_is_causal_skip=False) 的 float 4D mask"
            "（modeling:696-706）；indexer 取 [:,0,:,:]，层内再按 top-k 改成稀疏 mask")
        self.tensor("g_cos", "cos（interleaved RoPE）", ["1", "seqlen", "Rk"], ["seqlen", "Rk"],
                    "boundary", "网络级 rotary_emb 预计算（modeling:708-709），传给每层；"
                    "apply 时只取前 Rk÷2 个角")
        self.tensor("g_sin", "sin（interleaved RoPE）", ["1", "seqlen", "Rk"], ["seqlen", "Rk"],
                    "boundary", "同一 rotary_emb 调用的 sin；主注意力与 indexer 共用")
        self.tensor(
            "g_cache_k_pass_pre", "past_key_values.layers[layer_idx].keys（调用前，压缩 latent）",
            ["B", "1", "kvcachelen", "Rkv"], ["B", "kvcachelen", "Rkv"], "cache",
            "DynamicLayer.lazy_initialization 的 keys；存的是 kv_a_layernorm 后的 k_pass，"
            "不是展开后的 K（cache_utils:121-130 / modeling:396-398）")
        self.tensor(
            "g_cache_k_rot_pre", "past_key_values.layers[layer_idx].values（调用前，k_rot）",
            ["B", "1", "kvcachelen", "Rk"], ["B", "kvcachelen", "Rk"], "cache",
            "同一 update 的 values 槽存 k_rot（64 维）；名字是 values 但语义是 RoPE 后的 k_rot")
        self.tensor(
            "g_cache_k_pass_post", "self.keys（调用后，压缩 latent）",
            ["B", "1", "Lkv", "Rkv"], ["B", "Lkv", "Rkv"], "cache",
            "update(k_pass, k_rot) 追加本次 seqlen 个位置后的持久 keys（cache_utils:147-153）")
        self.tensor(
            "g_cache_k_rot_post", "self.values（调用后，k_rot）",
            ["B", "1", "Lkv", "Rk"], ["B", "Lkv", "Rk"], "cache",
            "追加后的持久 values（k_rot）；与 keys 一起构成 MLA 压缩缓存")
        if self.indexer_type == "full":
            self.tensor(
                "g_cache_indexer_keys_pre", "self.indexer_keys（调用前）",
                ["B", "kvcachelen", "Id"], ["B", "kvcachelen", "Id"], "cache",
                "DynamicIndexedLayer 的 indexer 键缓存（3D、单头，cache_utils:331-365）")
            self.tensor(
                "g_cache_indexer_keys_post", "self.indexer_keys（调用后）",
                ["B", "Lidx", "Id"], ["B", "Lidx", "Id"], "cache",
                "update_indexer 以 dim=1 concat 后的全量 indexer 键")
        else:
            self.tensor(
                "g_prev_topk_indices", "prev_topk_indices（上一 full indexer 层）",
                ["B", "seqlen", "Ksel"], ["B", "seqlen", "Ksel"], "boundary",
                "shared 层不实例化 indexer（modeling:348-349），直接复用上一层 full "
                "indexer 的 int32 top-k 索引（:404-417）；不是持久缓存")

    # ------------------------------------------------------------------
    # DecoderLayer 主流程
    # ------------------------------------------------------------------

    def _decoder_layer_flow(self) -> None:
        v = self.values
        src, cfg = self.src, self.cfg
        # 权重张量（按出现顺序声明；同组共享引用）
        self.weight("g_input_norm_weight", "self.input_layernorm.weight", ["H"], ["H"],
                    "RMSNorm 可学习缩放（无 bias），eps=rms_norm_eps（modeling:583）")
        self.weight("g_q_a_proj_weight", "self.q_a_proj.weight", ["Rq", "H"], ["Rq", "H"],
                    "nn.Linear(hidden_size, q_lora_rank, bias=attention_bias=False)（:324）")
        self.weight("g_q_a_norm_weight", "self.q_a_layernorm.weight", ["Rq"], ["Rq"],
                    "q_a_layernorm：RMSNorm(q_lora_rank)，默认 eps=1e-6（:325）")
        self.weight("g_q_b_proj_weight", "self.q_b_proj.weight", ["Nh×Dq", "Rq"],
                    ["Nh", "Dq", "Rq"],
                    "nn.Linear(q_lora_rank, num_heads*qk_head_dim, bias=False)（:326）")
        self.weight("g_kv_a_proj_weight", "self.kv_a_proj_with_mqa.weight",
                    ["Rkv+Rk", "H"], ["Rkv", "Rk", "H"],
                    "nn.Linear(hidden_size, kv_lora_rank+qk_rope_head_dim)（:328-332）")
        self.weight("g_kv_a_norm_weight", "self.kv_a_layernorm.weight", ["Rkv"], ["Rkv"],
                    "kv_a_layernorm：RMSNorm(kv_lora_rank)，默认 eps=1e-6（:333）")
        self.weight("g_kv_b_proj_weight", "self.kv_b_proj.weight",
                    ["Nh×(Dn+Dv)", "Rkv"], ["Nh", "Dn", "Dv", "Rkv"],
                    "nn.Linear(kv_lora_rank, num_heads*(qk_nope+v_head), bias=False)（:334-338）")
        self.weight("g_o_proj_weight", "self.o_proj.weight", ["H", "Nh×Dv"],
                    ["H", "Nh", "Dv"],
                    "nn.Linear(num_heads*v_head_dim, hidden_size, bias=False)（:340-344）")
        if self.indexer_type == "full":
            self.weight("g_ix_wq_b_weight", "self.indexer.wq_b.weight", ["Ih×Id", "Rq"],
                        ["Ih", "Id", "Rq"],
                        "indexer.wq_b：Linear(q_lora_rank, n_heads*head_dim, bias=False)（:178）")
            self.weight("g_ix_wk_weight", "self.indexer.wk.weight", ["Id", "H"],
                        ["Id", "H"],
                        "indexer.wk：Linear(hidden_size, head_dim, bias=False)（:179）")
            self.weight("g_ix_k_norm_weight", "self.indexer.k_norm.weight", ["Id"], ["Id"],
                        "indexer.k_norm：LayerNorm(head_dim) 的 weight（**带 bias**）")
            self.weight("g_ix_k_norm_bias", "self.indexer.k_norm.bias", ["Id"], ["Id"],
                        "indexer.k_norm 的 bias；这正是 LayerNorm 而非 RMSNorm 的证据（:180）")
            self.weight("g_ix_weights_proj_weight", "self.indexer.weights_proj.weight",
                        ["Ih", "H"], ["Ih", "H"],
                        "indexer.weights_proj：Linear(hidden_size, n_heads, bias=False)（:181）")
        if self.mlp_type == "sparse":
            self.weight("g_moe_router_weight", "self.mlp.gate.weight", ["E", "H"], ["E", "H"],
                        "GlmMoeDsaTopkRouter.weight（:476）；forward 强制 float32（:485）")
            self.weight("g_moe_router_bias", "self.mlp.gate.e_score_correction_bias", ["E"], ["E"],
                        "非学习 buffer，float32（:481/_keep_in_fp32_modules_strict:637）；"
                        "加到 sigmoid 分数上参与选择（:487）")
            self.weight("g_moe_experts_gate_up", "self.mlp.experts.gate_up_proj",
                        ["E", "2Es", "H"], ["E", "Es", "H"],
                        "专家权重以 3D 张量存储：gate_up_proj [E,2Es,H]（:520）")
            self.weight("g_moe_experts_down", "self.mlp.experts.down_proj",
                        ["E", "H", "Es"], ["E", "H", "Es"],
                        "专家权重以 3D 张量存储：down_proj [E,H,Es]（:521）")
            self.weight("g_moe_shared_gate_weight", "self.mlp.shared_experts.gate_proj.weight",
                        ["Es", "H"], ["Es", "H"],
                        "共享专家 gate_proj（intermediate_size=Es×n_shared_experts）（:561-563）")
            self.weight("g_moe_shared_up_weight", "self.mlp.shared_experts.up_proj.weight",
                        ["Es", "H"], ["Es", "H"], "共享专家 up_proj（:461）")
            self.weight("g_moe_shared_down_weight", "self.mlp.shared_experts.down_proj.weight",
                        ["H", "Es"], ["H", "Es"], "共享专家 down_proj（:462）")
        else:
            self.weight("g_dense_gate_weight", "self.mlp.gate_proj.weight", ["Is", "H"],
                        ["Is", "H"], "dense MLP gate_proj（:460）")
            self.weight("g_dense_up_weight", "self.mlp.up_proj.weight", ["Is", "H"],
                        ["Is", "H"], "dense MLP up_proj（:461）")
            self.weight("g_dense_down_weight", "self.mlp.down_proj.weight", ["H", "Is"],
                        ["H", "Is"], "dense MLP down_proj（:462）")
        self.weight("g_post_norm_weight", "self.post_attention_layernorm.weight", ["H"], ["H"],
                    "RMSNorm 可学习缩放（无 bias），eps=rms_norm_eps（modeling:584）")

        # 1) input_layernorm
        self.tensor("g_attn_normed", "hidden_states = self.input_layernorm(hidden_states)",
                    ["B", "seqlen", "H"], ["B", "seqlen", "H", "EpsN"], "activation",
                    "RMSNorm：fp32 方差归一后转回输入 dtype，再乘 weight（:56-61/:598）")
        self.op("g_input_norm", "GlmMoeDsaDecoderLayer.input_layernorm", "Input RMSNorm",
                "rmsnorm", ["g_hidden_in"], ["g_attn_normed"], ["g_input_norm_weight"],
                ["H", "EpsN"],
                [src("597-598", "residual = hidden_states; hidden_states = self.input_layernorm(hidden_states)"),
                 src("56-61", "fp32 方差归一 → weight 乘 → to(input_dtype)")],
                "层输入先归一化再进入 Attention；残差保留归一化前的 hidden_states")

        # 2) Q 支路（低秩 + 部分交错 RoPE）
        self.tensor("g_q_a", "self.q_a_proj(hidden_states)", ["B", "seqlen", "Rq"],
                    ["B", "seqlen", "Rq"], "activation", "Q 低秩投影输出（:383）")
        self.op("g_q_a_proj", "GlmMoeDsaAttention.q_a_proj", "Q 低秩投影 q_a_proj", "linear",
                ["g_attn_normed"], ["g_q_a"], ["g_q_a_proj_weight"],
                ["H", "Rq", "seqlen", "B"],
                [src("383", "q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))"),
                 src("324", "self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=attention_bias)")],
                "bias=config.attention_bias=False；输出 [B,seqlen,q_lora_rank]")
        self.tensor("g_q_residual", "q_resid = self.q_a_layernorm(...)", ["B", "seqlen", "Rq"],
                    ["B", "seqlen", "Rq"], "activation",
                    "q_residual：既进入 q_b_proj（主注意力），也作为 indexer.wq_b 的输入（:406-408）")
        self.op("g_q_a_norm", "GlmMoeDsaAttention.q_a_layernorm", "Q 低秩 RMSNorm", "rmsnorm",
                ["g_q_a"], ["g_q_residual"], ["g_q_a_norm_weight"], ["Rq", "EpsA"],
                [src("325", "self.q_a_layernorm = GlmMoeDsaRMSNorm(self.q_lora_rank)"),
                 src("56-61", "GlmMoeDsaRMSNorm.forward")],
                "eps 为源码默认 1e-6（不是根配置 rms_norm_eps=1e-5）；同一 q_residual 被"
                "主注意力与 indexer 共享引用")
        self.tensor("g_q_flat", "self.q_b_proj(q_resid)", ["B", "seqlen", "Nh×Dq"],
                    ["B", "seqlen", "Nh", "Dq"], "activation",
                    "q_b_proj 输出 [B,seqlen,Nh*Dq]（:384）")
        self.op("g_q_b_proj", "GlmMoeDsaAttention.q_b_proj", "Q 上投影 q_b_proj", "linear",
                ["g_q_residual"], ["g_q_flat"], ["g_q_b_proj_weight"],
                ["Rq", "Nh", "Dq", "seqlen", "B"],
                [src("384", "q_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)")],
                "输出最后一维为 Nh×Dq=64×256")
        self.tensor("g_q", "q_states（view+transpose 后）", ["B", "Nh", "seqlen", "Dq"],
                    ["B", "Nh", "seqlen", "Dq"], "activation",
                    "view(B,seqlen,Nh,Dq).transpose(1,2)（:384）")
        self.op("g_q_view", "GlmMoeDsaAttention.forward", "Q reshape/transpose", "view+transpose",
                ["g_q_flat"], ["g_q"], [], ["B", "Nh", "seqlen", "Dq"],
                [src("381-384", "query_shape=(B,S,-1,qk_head_dim); view(...).transpose(1,2)")],
                "只改布局不改数值；后续按头做 RoPE 与 QKᵀ")
        self.tensor("g_q_pass", "q_pass（nope 段）", ["B", "Nh", "seqlen", "Dn"],
                    ["B", "Nh", "seqlen", "Dn"], "activation",
                    "torch.split(q_states, [qk_nope_head_dim, qk_rope_head_dim]) 的前段（:385）")
        self.tensor("g_q_rot", "q_rot（待旋转段）", ["B", "Nh", "seqlen", "Rk"],
                    ["B", "Nh", "seqlen", "Rk"], "activation", "split 的后段（64 维）（:385）")
        self.op("g_q_split", "GlmMoeDsaAttention.forward", "Q 拆分 nope/rope", "split",
                ["g_q"], ["g_q_pass", "g_q_rot"], [], ["Dn", "Rk"],
                [src("385", "q_pass, q_rot = torch.split(q_states, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)")],
                "只有 rope 段参与旋转；nope 段原样进入最终 q")
        self.tensor("g_q_rot_emb", "q_rot（RoPE 后）", ["B", "Nh", "seqlen", "Rk"],
                    ["B", "Nh", "seqlen", "Rk"], "activation",
                    "交错 RoPE：q1*cos-q2*sin / q2*cos+q1*sin（:145-149）")
        self.op("g_q_rope", "apply_rotary_pos_emb_interleave", "Q 交错 RoPE", "rope-interleaved",
                ["g_q_rot", "g_cos", "g_sin"], ["g_q_rot_emb"], [],
                ["Rk", "seqlen", "ThetaM"],
                [src("392-394", "q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)"),
                 src("141-149", "cos/sin 取前一半，按 (0::2,1::2) 成对旋转")],
                "GLM 用交错布局（不是 DeepSeek 的 half-split rotate_half）；"
                "cos/sin 来自网络级 rotary_emb，theta=8e6、32 对频率")
        self.tensor("g_q_full", "query_states = cat((q_pass, q_rot), -1)", ["B", "Nh", "seqlen", "Dq"],
                    ["B", "Nh", "seqlen", "Dq"], "activation",
                    "nope+rope 拼回完整 q（:400）")
        self.op("g_q_cat", "GlmMoeDsaAttention.forward", "Q 拼回完整头维", "concat",
                ["g_q_pass", "g_q_rot_emb"], ["g_q_full"], [], ["Dq", "Dn", "Rk"],
                [src("400", "query_states = torch.cat((q_pass, q_rot), dim=-1)")],
                "nope 在前、rope 在后；顺序与 key 的拼接一致")

        # 3) 共享 KV 支路（压缩 latent + cache）
        self.tensor("g_kv_compressed", "compressed_kv = self.kv_a_proj_with_mqa(hidden_states)",
                    ["B", "seqlen", "Rkv+Rk"], ["B", "seqlen", "Rkv", "Rk"], "activation",
                    "单头压缩 KV：前 512 维 latent + 后 64 维 k_rot（:387-388）")
        self.op("g_kv_a_proj", "GlmMoeDsaAttention.kv_a_proj_with_mqa", "KV 压缩投影",
                "linear", ["g_attn_normed"], ["g_kv_compressed"], ["g_kv_a_proj_weight"],
                ["H", "Rkv", "Rk", "seqlen", "B"],
                [src("387-388", "compressed_kv = self.kv_a_proj_with_mqa(hidden_states); "
                                "kv_pass, k_rot = split([kv_lora_rank, qk_rope_head_dim])")],
                "输出 [B,seqlen,512+64]；这份压缩投影就是 MLA 省 KV 缓存的来源")
        self.tensor("g_kv_pass", "kv_pass（未归一化 latent）", ["B", "seqlen", "Rkv"],
                    ["B", "seqlen", "Rkv"], "activation", "split 前段（512）（:388）")
        self.tensor("g_k_rot", "k_rot（view 后）", ["B", "1", "seqlen", "Rk"],
                    ["B", "seqlen", "Rk"], "activation", "split 后段 view 成单头 4D（:392）")
        self.op("g_kv_a_split", "GlmMoeDsaAttention.forward", "KV 拆分 latent/k_rot", "split",
                ["g_kv_compressed"], ["g_kv_pass", "g_k_rot"], [], ["Rkv", "Rk"],
                [src("388", "kv_pass, k_rot = torch.split(compressed_kv, [...])"),
                 src("392", "k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)")],
                "k_rot 以单头 4D 形态参与 RoPE 与缓存")
        self.tensor("g_k_pass_normed", "k_pass = self.kv_a_layernorm(kv_pass)",
                    ["B", "seqlen", "Rkv"], ["B", "seqlen", "Rkv"], "activation",
                    "kv_a_layernorm 后的 latent（仍是 3D [B,seqlen,512]）；随后单独 view 成单头 4D（:390）")
        self.op("g_kv_a_norm", "GlmMoeDsaAttention.kv_a_layernorm", "KV latent RMSNorm",
                "rmsnorm", ["g_kv_pass"], ["g_k_pass_normed"], ["g_kv_a_norm_weight"],
                ["Rkv", "EpsA"],
                [src("333", "self.kv_a_layernorm = GlmMoeDsaRMSNorm(config.kv_lora_rank)"),
                 src("390", "k_pass = self.kv_a_layernorm(kv_pass)")],
                "eps=1e-6；只有 latent 段归一化，k_rot 不归一化")
        self.tensor("g_k_pass_4d", "k_pass.view(B,1,seqlen,kv_lora_rank)",
                    ["B", "1", "seqlen", "Rkv"], ["B", "seqlen", "Rkv"], "activation",
                    "归一化后的 latent 再 view 成单头 4D，才进入 4D 缓存（:390-398）；"
                    "不能把 3D 张量直接当 4D 缓存输入")
        self.op("g_k_pass_view", "GlmMoeDsaAttention.forward", "KV latent 单头 view",
                "view", ["g_k_pass_normed"], ["g_k_pass_4d"], [], ["Rkv", "seqlen", "B"],
                [src("390", "k_pass = self.kv_a_layernorm(kv_pass).view(batch_size, 1, seq_length, kv_lora_rank)")],
                "把 [B,seqlen,512] 显式 view 成 [B,1,seqlen,512]；缓存写入的是这个单头 4D 张量")
        self.tensor("g_k_rot_emb", "k_rot（RoPE 后）", ["B", "1", "seqlen", "Rk"],
                    ["B", "seqlen", "Rk"], "activation", "与 q_rot 同一交错 RoPE（:394）")
        self.op("g_kv_rope", "apply_rotary_pos_emb_interleave", "K 交错 RoPE",
                "rope-interleaved", ["g_k_rot", "g_cos", "g_sin"], ["g_k_rot_emb"], [],
                ["Rk", "seqlen", "ThetaM"],
                [src("392-394", "k_rot view 后与 q_rot 一起 apply_rotary_pos_emb_interleave")],
                "k_rot 是唯一进入缓存的旋转键分量")
        self.op("g_cache_update", "DynamicIndexedLayer.update", "KV 缓存追加（压缩 latent）",
                "cache_update",
                ["g_cache_k_pass_pre", "g_cache_k_rot_pre", "g_k_pass_4d", "g_k_rot_emb"],
                ["g_cache_k_pass_post", "g_cache_k_rot_post"], [],
                ["Lkv", "Rkv", "Rk", "kvcachelen", "seqlen"],
                [src("396-398", "k_pass, k_rot = past_key_values.update(k_pass, k_rot, self.layer_idx)"),
                 self.srcf("cache_utils.py", "147-153", "self.keys/values = torch.cat([..., new], dim=-2)")],
                "缓存里是 512 维 latent + 64 维 k_rot（每 token 576 个值/层），"
                "**不是**展开后的 64 头 K/V；调用前/调用后分两个张量表达，不混为原地更新")
        self.tensor("g_kv_nope_flat", "self.kv_b_proj(k_pass)", ["B", "1", "Lkv", "Nh×(Dn+Dv)"],
                    ["B", "Lkv", "Nh", "Dn", "Dv"], "activation",
                    "对缓存返回的**全部 Lkv 个位置**做上投影（:402→:351-360）；"
                    "每次调用都重建整段 K/V，是 eager 路径的真实计算量")
        self.op("g_kv_b_proj", "GlmMoeDsaAttention.kv_b_proj", "KV latent 展开 kv_b_proj",
                "linear", ["g_cache_k_pass_post"], ["g_kv_nope_flat"], ["g_kv_b_proj_weight"],
                ["Rkv", "Nh", "Dn", "Dv", "Lkv"],
                [src("402", "key_states, value_states = self.expand_kv(k_pass, k_rot)"),
                 src("351-361", "kv_nope = self.kv_b_proj(kv_nope).view(key_shape).transpose(1,2); "
                                "split([qk_nope, v_head])")],
                "输出最后一维 Nh×(192+256)=64×448；随后 view+transpose 再拆 k_nope/v")
        self.tensor("g_kv_nope", "kv_nope（view+transpose 后）", ["B", "Nh", "Lkv", "Dn+Dv"],
                    ["B", "Nh", "Lkv", "Dn", "Dv"], "activation",
                    "view(B,Lkv,Nh,Dn+Dv).transpose(1,2)（:358-360）")
        self.op("g_kv_view", "GlmMoeDsaAttention.expand_kv", "KV 展开布局", "view+transpose",
                ["g_kv_nope_flat"], ["g_kv_nope"], [], ["B", "Nh", "Lkv", "Dn", "Dv"],
                [src("357-360", "key_shape=(batch,seq,-1,qk_nope+v_head); view(key_shape).transpose(1,2)")],
                "把 Lkv 个位置的 448 维向量按 64 头重排")
        self.tensor("g_k_nope", "k_nope", ["B", "Nh", "Lkv", "Dn"],
                    ["B", "Nh", "Lkv", "Dn"], "activation", "split 前段（192）（:361）")
        self.tensor("g_v", "value_states", ["B", "Nh", "Lkv", "Dv"],
                    ["B", "Nh", "Lkv", "Dv"], "activation", "split 后段（256）（:361）")
        self.op("g_kv_split", "GlmMoeDsaAttention.expand_kv", "拆分 k_nope/value", "split",
                ["g_kv_nope"], ["g_k_nope", "g_v"], [], ["Dn", "Dv"],
                [src("361", "k_nope, value_states = torch.split(kv_nope, [qk_nope_head_dim, v_head_dim], dim=-1)")],
                "k 头维 192、v 头维 256，两者不相等")
        self.tensor("g_k_rot_expanded", "k_rot.expand(-1, n_heads, -1, -1)",
                    ["B", "Nh", "Lkv", "Rk"], ["B", "Nh", "Lkv", "Rk"], "activation",
                    "单头 k_rot 广播到 Nh 头（:362）；不复制底层存储")
        self.op("g_k_rot_expand", "GlmMoeDsaAttention.expand_kv", "k_rot 广播到各头",
                "expand", ["g_cache_k_rot_post"], ["g_k_rot_expanded"], [], ["Nh", "Rk"],
                [src("362", "k_rot = k_rot.expand(-1, k_nope.shape[1], -1, -1)")],
                "单头共享的旋转键在注意力前广播成 Nh 份")
        self.tensor("g_key", "key_states = cat([k_nope, k_rot])", ["B", "Nh", "Lkv", "Dq"],
                    ["B", "Nh", "Lkv", "Dq"], "activation",
                    "key_states[..., :Dn]=k_nope; [..., Dn:]=k_rot（:365-368）")
        self.op("g_key_cat", "GlmMoeDsaAttention.expand_kv", "key 拼接 nope+rope", "concat",
                ["g_k_nope", "g_k_rot_expanded"], ["g_key"], [], ["Dn", "Rk", "Dq"],
                [src("365-368", "key_states = new_empty(..., qk_nope+qk_rope); copy_(k_nope); copy_(k_rot)")],
                "最终 K 的 256 维 = 192 维 nope + 64 维 rope；与 Q 的布局一致")

        # 4) DSA indexer（仅 full 层）
        if self.indexer_type == "full":
            self._indexer_ops()
        # 5) 稀疏 mask（full 用本层 top-k，shared 用上一层 top-k）
        topk_tid = "g_ix_topk_indices" if self.indexer_type == "full" else "g_prev_topk_indices"
        self.tensor("g_index_mask", "index_mask（True=屏蔽；top-k 选中位置为 False）",
                    ["B", "1", "seqlen", "Lidx"], ["B", "seqlen", "Lidx"], "mask",
                    "new_ones([B,seqlen,Lkv]) 先全 True，再把 top-k 选中位置 scatter 成 False"
                    "（:421-425）→ unsqueeze(1) 得 [B,1,seqlen,Lkv]；"
                    "True 表示该 key 要被 masked_fill 屏蔽，False（被选中）保留因果 mask 的可见性")
        self.op("g_index_mask_scatter", "GlmMoeDsaAttention.forward",
                "top-k 索引 → 稀疏掩码（True=屏蔽）",
                "scatter+unsqueeze", [topk_tid], ["g_index_mask"], [],
                ["B", "seqlen", "Ksel", "Lidx"],
                [src("421-425", "topk_indices.new_ones((B,S,Lkv),bool)"
                                ".scatter(-1, topk_indices.long(), False).unsqueeze(1)")],
                ("本层 full indexer 输出的 top-k（int32）散进全 True 的布尔矩阵，"
                 "选中位置变 False（True=屏蔽）；"
                 if self.indexer_type == "full" else
                 "shared 层复用上一 full 层的 top-k（int32）散进全 True 的布尔矩阵，"
                 "选中位置变 False（True=屏蔽）；")
                + "unsqueeze(1) 后是 4D [B,1,seqlen,Lidx]；每行选中数 = K_sel（=min(Ki,Lidx)）")
        self.tensor("g_sparse_mask", "attention_mask（稀疏化后；True 位置=min）",
                    ["B", "1", "seqlen", "Lidx"], ["B", "seqlen", "Lidx"], "mask",
                    "float mask 走 masked_fill(index_mask, min)（:426-430）："
                    "index_mask 为 True（未被 top-k 选中）的位置置 min；"
                    "False（选中）的位置保留网络级因果 mask 的值")
        self.op("g_sparse_mask_apply", "GlmMoeDsaAttention.forward", "因果 mask + 稀疏掩码",
                "masked_fill", ["g_attention_mask", "g_index_mask"], ["g_sparse_mask"], [],
                ["B", "seqlen", "Lidx", "Lmask"],
                [src("426-430", "attention_mask.masked_fill(index_mask, torch.finfo(dtype).min)"),
                 src("427-429", "bool 分支：attention_mask & ~index_mask")],
                "index_mask 为 True 的位置被置 min（softmax 后为 0）；False（被 top-k 选中）"
                "保留因果可见性。bool 分支是 & ~index_mask，与 float 的 masked_fill(True) 等价；"
                "top-k 只能进一步收窄、不能放大因果范围")

        # 6) 注意力核心
        self.tensor("g_attn_logits", "attn_weights = QKᵀ×scaling", ["B", "Nh", "seqlen", "Lkv"],
                    ["B", "Nh", "seqlen", "Lkv"], "activation",
                    "matmul(query, key.transpose(2,3))*scaling（:270/:434-447）；scaling=1/16")
        self.op("g_qk", "eager_attention_forward", "QKᵀ 缩放", "matmul+scale",
                ["g_q_full", "g_key"], ["g_attn_logits"], [], ["Nh", "seqlen", "Lkv", "Dq"],
                [src("437-447", "attention_interface(...) → eager_attention_forward"),
                 src("270", "attn_weights = torch.matmul(query, key_states.transpose(2,3)) * scaling"),
                 src("346", "self.scaling = yarn_apply_mscale(rope_parameters, qk_head_dim**-0.5)")],
                "Nkv=Nh → repeat_kv 恒等（:267-268）；scaling=qk_head_dim^-0.5=1/16，"
                "rope_type=default 不再乘 YaRN mscale")
        self.tensor("g_attn_masked", "attn_weights + attention_mask", ["B", "Nh", "seqlen", "Lkv"],
                    ["B", "Nh", "seqlen", "Lkv"], "activation",
                    "加稀疏因果 mask（:271-272）")
        self.op("g_mask_add", "eager_attention_forward", "加稀疏 mask", "mask_add",
                ["g_attn_logits", "g_sparse_mask"], ["g_attn_masked"], [],
                ["B", "Nh", "seqlen", "Lkv"],
                [src("271-272", "if attention_mask is not None: attn_weights = attn_weights + attention_mask")],
                "mask 与 logits 同形 [B,Nh,seqlen,Lkv]（mask 的 1 广播到 Nh）")
        self.tensor("g_attn_probs", "attn_weights = softmax(..., dim=-1, dtype=float32)",
                    ["B", "Nh", "seqlen", "Lkv"], ["B", "Nh", "seqlen", "Lkv"], "activation",
                    "softmax 在 fp32 上计算后转回 query dtype（:274）")
        self.op("g_softmax", "eager_attention_forward", "Softmax", "softmax",
                ["g_attn_masked"], ["g_attn_probs"], [], ["Nh", "seqlen", "Lkv"],
                [src("274", "softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)")],
                "对 Lkv 个位置（含未被 top-k 选中的 min 值）归一化；被 mask 的位置概率为 0")
        self.tensor("g_attn_head_out", "attn_output（AV 后，仍是 B,Nh,seqlen,Dv）",
                    ["B", "Nh", "seqlen", "Dv"], ["B", "Nh", "seqlen", "Dv"], "activation",
                    "AV 乘积（:276）；头维是 v_head_dim=256，尚未做 transpose")
        self.op("g_av", "eager_attention_forward", "AV 加权求和", "matmul",
                ["g_attn_probs", "g_v"], ["g_attn_head_out"], [], ["Nh", "seqlen", "Dv", "Lkv"],
                [src("276", "attn_output = torch.matmul(attn_weights, value_states)")],
                "AV 输出布局仍是 [B,Nh,seqlen,Dv]；transpose/contiguous 是单独一步（:277）")
        self.tensor("g_attn_heads_seq", "attn_output.transpose(1,2).contiguous()",
                    ["B", "seqlen", "Nh", "Dv"], ["B", "seqlen", "Nh", "Dv"], "activation",
                    "eager 在 AV 后显式 transpose(1,2).contiguous()（:277）："
                    "把头维移到 seqlen 之后，再把内存布局连续化")
        self.op("g_attn_transpose", "eager_attention_forward", "AV 输出转置+连续化",
                "transpose+contiguous", ["g_attn_head_out"], ["g_attn_heads_seq"], [],
                ["B", "seqlen", "Nh", "Dv"],
                [src("277", "attn_output = attn_output.transpose(1, 2).contiguous()")],
                "把 [B,Nh,seqlen,Dv] 变成 [B,seqlen,Nh,Dv] 并 contiguous；"
                "缺少这一步会把头维/序列维混在一起，不能只靠最终 reshape 掩盖")
        self.tensor("g_attn_flat", "attn_output.reshape(B,seqlen,Nh*Dv)", ["B", "seqlen", "Nh×Dv"],
                    ["B", "seqlen", "Nh", "Dv"], "activation",
                    "在 transpose+contiguous 之后拼头（:449）")
        self.op("g_attn_reshape", "GlmMoeDsaAttention.forward", "注意力输出拼头", "reshape",
                ["g_attn_heads_seq"], ["g_attn_flat"], [], ["Nh", "Dv"],
                [src("449", "attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()")],
                "输入已是 [B,seqlen,Nh,Dv]；reshape 只把 Nh×Dv 并成一维=16384")
        self.tensor("g_attn_out", "self.o_proj(attn_output)", ["B", "seqlen", "H"],
                    ["B", "seqlen", "H"], "activation", "输出投影（:450）")
        self.op("g_o_proj", "GlmMoeDsaAttention.o_proj", "输出投影 o_proj", "linear",
                ["g_attn_flat"], ["g_attn_out"], ["g_o_proj_weight"],
                ["Nh", "Dv", "H", "seqlen", "B"],
                [src("450", "attn_output = self.o_proj(attn_output)"),
                 src("340-344", "self.o_proj = nn.Linear(num_heads*v_head_dim, hidden_size, bias=attention_bias)")],
                "bias=False；输出回到 hidden_size=6144")

        # 7) 第一次残差 + post_attention_layernorm
        self.tensor("g_hidden_mid", "hidden_states = residual + attn_output", ["B", "seqlen", "H"],
                    ["B", "seqlen", "H"], "activation",
                    "第一次普通残差相加（:610）；没有 mHC/多流/门控")
        self.op("g_res_attn", "GlmMoeDsaDecoderLayer.forward", "Attention 残差相加",
                "add", ["g_hidden_in", "g_attn_out"], ["g_hidden_mid"], [],
                ["B", "seqlen", "H"],
                [src("597", "residual = hidden_states"),
                 src("610", "hidden_states = residual + hidden_states")],
                "残差用归一化前的层输入；与 DeepSeek 的 mHC post/comb 混合不同，这里是标准加法")
        self.tensor("g_ffn_in", "hidden_states = self.post_attention_layernorm(...)",
                    ["B", "seqlen", "H"], ["B", "seqlen", "H", "EpsN"], "activation",
                    "RMSNorm（eps=rms_norm_eps）（:613）")
        self.op("g_post_norm", "GlmMoeDsaDecoderLayer.post_attention_layernorm", "Post-Attention RMSNorm",
                "rmsnorm", ["g_hidden_mid"], ["g_ffn_in"], ["g_post_norm_weight"],
                ["H", "EpsN"],
                [src("612-613", "residual = hidden_states; hidden_states = self.post_attention_layernorm(hidden_states)")],
                "与 input_layernorm 同为 RMSNorm，但权重独立")

        # 8) MLP / MoE
        if self.mlp_type == "sparse":
            self._moe_ops()
        else:
            self._dense_ops()

        # 9) 第二次残差
        self.tensor("g_hidden_out", "hidden_states（层输出）", ["B", "seqlen", "H"],
                    ["B", "seqlen", "H"], "boundary",
                    "第二次普通残差相加后的层输出（:615）；写入 next hidden_states")
        self.op("g_res_ffn", "GlmMoeDsaDecoderLayer.forward", "FFN/MoE 残差相加", "add",
                ["g_hidden_mid", "g_mlp_out"], ["g_hidden_out"], [],
                ["B", "seqlen", "H"],
                [src("612", "residual = hidden_states"),
                 src("614-615", "hidden_states = self.mlp(hidden_states); hidden_states = residual + hidden_states")],
                "两处残差都是普通加法；层内没有额外的流混合/归一化")

    def _indexer_ops(self) -> None:
        src = self.src
        self.tensor("g_ix_q_flat", "self.indexer.wq_b(q_residual)", ["B", "seqlen", "Ih×Id"],
                    ["B", "seqlen", "Ih", "Id"], "activation",
                    "indexer 的独立 Q 投影（输入是 q_residual，不是 hidden_states）（:213-214）")
        self.op("g_ix_q_proj", "GlmMoeDsaIndexer.wq_b", "Indexer Q 投影", "linear",
                ["g_q_residual"], ["g_ix_q_flat"], ["g_ix_wq_b_weight"],
                ["Rq", "Ih", "Id", "seqlen", "B"],
                [src("213-214", "q = self.wq_b(q_resid); q = q.view(B,S,self.n_heads,self.head_dim)")],
                "indexer 有独立于主注意力的轻量投影（wq_b/wk/weights_proj）")
        self.tensor("g_ix_q", "q（indexer 打分头）", ["B", "seqlen", "Ih", "Id"],
                    ["B", "seqlen", "Ih", "Id"], "activation", "view 后（:214）")
        self.op("g_ix_q_view", "GlmMoeDsaIndexer.forward", "Indexer Q 布局", "view",
                ["g_ix_q_flat"], ["g_ix_q"], [], ["B", "seqlen", "Ih", "Id"],
                [src("214", "q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)")],
                "注意这里不做 transpose：indexer 打分保留 [B,S,H,D] 布局")
        self.tensor("g_ix_q_rot", "q_rot（indexer）", ["B", "seqlen", "Ih", "Rk"],
                    ["B", "seqlen", "Ih", "Rk"], "activation", "split 前段（64）")
        self.tensor("g_ix_q_pass", "q_pass（indexer）", ["B", "seqlen", "Ih", "Id-Rk"],
                    ["B", "seqlen", "Ih", "Id", "Rk"], "activation", "split 后段（128-64=64）")
        self.op("g_ix_q_split", "GlmMoeDsaIndexer.forward", "Indexer Q 拆分", "split",
                ["g_ix_q"], ["g_ix_q_rot", "g_ix_q_pass"], [], ["Rk", "Id"],
                [src("215", "q_rot, q_pass = torch.split(q, [qk_rope_head_dim, head_dim-qk_rope_head_dim], -1)")],
                "indexer 头维 128 中前 64 维旋转、后 64 维直通")
        self.tensor("g_ix_k_proj", "self.indexer.wk(hidden_states)", ["B", "seqlen", "Id"],
                    ["B", "seqlen", "Id"], "activation",
                    "indexer 的 K 投影：输入是 Attention 的 hidden_states（input_layernorm 输出）（:217）")
        self.op("g_ix_k_proj", "GlmMoeDsaIndexer.wk", "Indexer K 投影", "linear",
                ["g_attn_normed"], ["g_ix_k_proj"], ["g_ix_wk_weight"],
                ["H", "Id", "seqlen", "B"],
                [src("217", "k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)")],
                "单头、hidden_size→128；与主 KV 的 512 维 latent 完全独立")
        self.tensor("g_ix_k_normed", "self.indexer.k_norm(...)", ["B", "seqlen", "Id"],
                    ["B", "seqlen", "Id"], "activation",
                    "LayerNorm(128, eps=1e-6)：**带 weight 与 bias**（:180）")
        self.op("g_ix_k_norm", "GlmMoeDsaIndexer.k_norm", "Indexer K LayerNorm", "layernorm",
                ["g_ix_k_proj"], ["g_ix_k_normed"], ["g_ix_k_norm_weight", "g_ix_k_norm_bias"],
                ["Id", "EpsA"],
                [src("180", "self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)"),
                 src("217", "k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)")],
                "这是 LayerNorm 而非 RMSNorm（有 bias）；量化配置 modules_to_not_convert 里也"
                "单独列出 indexer.k_norm.bias，可交叉印证")
        self.tensor("g_ix_k", "k（unsqueeze 后）", ["B", "seqlen", "1", "Id"],
                    ["B", "seqlen", "Id"], "activation", "unsqueeze(2) 成伪单头（:217）")
        self.op("g_ix_k_view", "GlmMoeDsaIndexer.forward", "Indexer K 单头布局", "unsqueeze",
                ["g_ix_k_normed"], ["g_ix_k"], [], ["B", "seqlen", "Id"],
                [src("217", "k = ...unsqueeze(2)  # [B,S,1,D]")],
                "伪头维用于与 q 的 [B,S,H,D] 广播打分")
        self.tensor("g_ix_k_rot", "k_rot（indexer）", ["B", "seqlen", "1", "Rk"],
                    ["B", "seqlen", "Rk"], "activation", "split 前段")
        self.tensor("g_ix_k_pass", "k_pass（indexer）", ["B", "seqlen", "1", "Id-Rk"],
                    ["B", "seqlen", "Id", "Rk"], "activation", "split 后段")
        self.op("g_ix_k_split", "GlmMoeDsaIndexer.forward", "Indexer K 拆分", "split",
                ["g_ix_k"], ["g_ix_k_rot", "g_ix_k_pass"], [], ["Rk", "Id"],
                [src("218", "k_rot, k_pass = torch.split(k, [qk_rope_head_dim, head_dim-qk_rope_head_dim], -1)")],
                "与 q 相同的 64/64 拆分")
        self.tensor("g_ix_q_rot_emb", "q_rot（indexer RoPE 后）", ["B", "seqlen", "Ih", "Rk"],
                    ["B", "seqlen", "Ih", "Rk"], "activation", "交错 RoPE（unsqueeze_dim=2）")
        self.tensor("g_ix_k_rot_emb", "k_rot（indexer RoPE 后）", ["B", "seqlen", "1", "Rk"],
                    ["B", "seqlen", "Rk"], "activation", "同上")
        self.op("g_ix_rope", "GlmMoeDsaIndexer.forward", "Indexer 交错 RoPE", "rope-interleaved",
                ["g_ix_q_rot", "g_ix_k_rot", "g_cos", "g_sin"],
                ["g_ix_q_rot_emb", "g_ix_k_rot_emb"], [], ["Rk", "seqlen", "ThetaM"],
                [src("220-222", "q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin, unsqueeze_dim=2)"),
                 src("141-149", "交错成对旋转")],
                "indexer 也复用同一 cos/sin（与主注意力相同 theta/频率）")
        self.tensor("g_ix_q_cat", "q（拼回 128）", ["B", "seqlen", "Ih", "Id"],
                    ["B", "seqlen", "Ih", "Id"], "activation", "cat(q_rot,q_pass)（:222）")
        self.op("g_ix_q_cat", "GlmMoeDsaIndexer.forward", "Indexer Q 拼回", "concat",
                ["g_ix_q_rot_emb", "g_ix_q_pass"], ["g_ix_q_cat"], [], ["Id", "Rk"],
                [src("222", "q = torch.cat([q_rot, q_pass], dim=-1)")],
                "只有 rope 段被旋转")
        self.tensor("g_ix_k_cat", "k（拼回 128）", ["B", "seqlen", "Id"],
                    ["B", "seqlen", "Id"], "activation", "cat 后 squeeze(2)（:223）")
        self.op("g_ix_k_cat", "GlmMoeDsaIndexer.forward", "Indexer K 拼回", "concat",
                ["g_ix_k_rot_emb", "g_ix_k_pass"], ["g_ix_k_cat"], [], ["Id", "Rk"],
                [src("223", "k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)")],
                "压回 [B,S,128] 后进入 indexer 缓存")
        self.op("g_ix_cache_update", "DynamicIndexedLayer.update_indexer",
                "Indexer 键缓存追加", "cache_update",
                ["g_cache_indexer_keys_pre", "g_ix_k_cat"],
                ["g_cache_indexer_keys_post"], [], ["Lidx", "Id", "kvcachelen", "seqlen"],
                [src("225-226", "k = past_key_values.update_indexer(k, self.layer_idx)"),
                 self.srcf("cache_utils.py", "350-365",
                           "self.indexer_keys = torch.cat([..., new], dim=1); return full")],
                "indexer 缓存只有键（无 value），3D 单头 [B,Lidx,128]；shared 层没有该缓存")
        self.tensor("g_ix_scores", "scores = q.float() @ k.float()ᵀ * softmax_scale",
                    ["B", "seqlen", "Ih", "Lidx"], ["B", "seqlen", "Ih", "Lidx"], "activation",
                    "打分在 float32（:228）")
        self.op("g_ix_scores", "GlmMoeDsaIndexer.forward", "Indexer 打分", "matmul+scale",
                ["g_ix_q_cat", "g_cache_indexer_keys_post"], ["g_ix_scores"], [],
                ["Ih", "seqlen", "Lidx", "Id"],
                [src("228", "scores = torch.matmul(q.float(), k.transpose(-1,-2).float().unsqueeze(1)) * self.softmax_scale")],
                "q [B,S,Ih,128] × kᵀ [B,1,128,Lidx] → [B,S,Ih,Lidx]；"
                "self.softmax_scale = head_dim^-0.5 = 128^-0.5（:182）")
        self.tensor("g_ix_relu", "scores = F.relu(scores)", ["B", "seqlen", "Ih", "Lidx"],
                    ["B", "seqlen", "Ih", "Lidx"], "activation", "ReLU 截断负分（:229）")
        self.op("g_ix_relu", "GlmMoeDsaIndexer.forward", "Indexer ReLU", "relu",
                ["g_ix_scores"], ["g_ix_relu"], [], ["Ih", "Lidx"],
                [src("229", "scores = F.relu(scores)")],
                "只保留正相关；负分不贡献选择")
        self.tensor("g_ix_weights", "weights = self.weights_proj(hidden) * n_heads^-0.5",
                    ["B", "seqlen", "Ih"], ["B", "seqlen", "Ih"], "activation",
                    "每头权重（float32，:232）")
        self.op("g_ix_weights", "GlmMoeDsaIndexer.weights_proj", "Indexer 头权重", "linear+scale",
                ["g_attn_normed"], ["g_ix_weights"], ["g_ix_weights_proj_weight"],
                ["H", "Ih", "seqlen", "B"],
                [src("232", "weights = self.weights_proj(hidden_states.to(w.dtype)).float() * (self.n_heads**-0.5)")],
                "与打分头一一对应；乘 Ih^-0.5")
        self.tensor("g_ix_index_scores", "index_scores（各头加权求和）", ["B", "seqlen", "Lidx"],
                    ["B", "seqlen", "Lidx"], "activation",
                    "weights.unsqueeze(-2) @ scores（:233）")
        self.op("g_ix_sum", "GlmMoeDsaIndexer.forward", "Indexer 头加权求和", "matmul+sum",
                ["g_ix_weights", "g_ix_relu"], ["g_ix_index_scores"], [], ["Ih", "Lidx"],
                [src("231-233", "index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)")],
                "把每头的 [B,S,Lidx] 打分成一个 [B,S,Lidx] 选择分数")
        self.tensor("g_ix_masked_scores", "index_scores + mask", ["B", "seqlen", "Lidx"],
                    ["B", "seqlen", "Lidx"], "activation",
                    "因果/填充位置被掩成 -inf（:236-239）")
        self.op("g_ix_mask", "GlmMoeDsaIndexer.forward", "Indexer 因果掩码", "masked_fill",
                ["g_ix_index_scores", "g_attention_mask"], ["g_ix_masked_scores"], [],
                ["B", "seqlen", "Lidx"],
                [src("235-239", "if attention_mask.dtype == bool: masked_fill(~mask,-inf) else index_scores + mask"),
                 src("410", "self.indexer(..., attention_mask[:, 0, :, :], ...)")],
                "indexer 只取 mask 的第 0 个 batch/head 切片；保证 top-k 只在因果可见范围内选")
        self.tensor("g_ix_topk_indices", "topk_indices（int32）", ["B", "seqlen", "Ksel"],
                    ["B", "seqlen", "Ksel"], "activation",
                    "topk(min(Ki,Lidx)).indices（:241-242）")
        self.op("g_ix_topk", "GlmMoeDsaIndexer.forward", "Indexer TopK", "topk",
                ["g_ix_masked_scores"], ["g_ix_topk_indices"], [], ["Ki", "Ksel", "Lidx"],
                [src("241-242", "topk = min(self.index_topk, index_scores.shape[-1]); "
                                "index_scores.topk(topk, dim=-1).indices.to(torch.int32)")],
                "输出是索引而不是掩码：eager/sdpa 再把索引散成 mask；"
                "Lidx≤Ki 时等于全选（掩码不删任何位置）")

    def _dense_ops(self) -> None:
        src = self.src
        self.tensor("g_dense_gate", "self.mlp.gate_proj(x)", ["B", "seqlen", "Is"],
                    ["B", "seqlen", "Is"], "activation", "dense gate 投影（:466）")
        self.op("g_dense_gate", "GlmMoeDsaMLP.gate_proj", "Dense gate_proj", "linear",
                ["g_ffn_in"], ["g_dense_gate"], ["g_dense_gate_weight"],
                ["H", "Is", "seqlen", "B"],
                [src("460", "self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)"),
                 src("466", "self.act_fn(self.gate_proj(x)) * self.up_proj(x)")],
                "dense 层用 intermediate_size=12288（不是专家宽度 2048）")
        self.tensor("g_dense_up", "self.mlp.up_proj(x)", ["B", "seqlen", "Is"],
                    ["B", "seqlen", "Is"], "activation", "dense up 投影（:466）")
        self.op("g_dense_up", "GlmMoeDsaMLP.up_proj", "Dense up_proj", "linear",
                ["g_ffn_in"], ["g_dense_up"], ["g_dense_up_weight"],
                ["H", "Is", "seqlen", "B"],
                [src("461", "self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)")],
                "no bias")
        self.tensor("g_dense_act", "act_fn(gate) * up", ["B", "seqlen", "Is"],
                    ["B", "seqlen", "Is"], "activation", "SiLU(gate)×up（:466）")
        self.op("g_dense_act", "GlmMoeDsaMLP.forward", "SiLU 门控相乘", "silu*mul",
                ["g_dense_gate", "g_dense_up"], ["g_dense_act"], [], ["Is"],
                [src("466", "down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))"),
                 src("463", "self.act_fn = ACT2FN[config.hidden_act]（silu）")],
                "本模型没有 DeepSeek 的 swiglu_limit clamp；hidden_act=silu 直接作用")
        self.tensor("g_mlp_out", "mlp_out（本组 MLP/MoE 输出）", ["B", "seqlen", "H"],
                    ["B", "seqlen", "H"], "activation", "dense MLP 输出（:466）")
        self.op("g_dense_down", "GlmMoeDsaMLP.down_proj", "Dense down_proj", "linear",
                ["g_dense_act"], ["g_mlp_out"], ["g_dense_down_weight"],
                ["Is", "H", "seqlen", "B"],
                [src("462", "self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)")],
                "回投影到 hidden_size")

    def _moe_ops(self) -> None:
        src = self.src
        # router
        self.tensor("g_moe_flat", "hidden_states.view(-1, hidden_dim)", ["Ttok", "H"],
                    ["Ttok", "H"], "activation", "MoE 先把 [B,seqlen,H] 展平（:569）")
        self.op("g_moe_flatten", "GlmMoeDsaMoE.forward", "MoE 输入展平", "view",
                ["g_ffn_in"], ["g_moe_flat"], [], ["B", "seqlen", "Ttok", "H"],
                [src("569", "hidden_states = hidden_states.view(-1, hidden_states.shape[-1])")],
                "展平后按 token 路由；residuals 保留原 [B,seqlen,H] 供共享专家")
        self.tensor("g_router_logits", "router_logits = F.linear(x.float(), W.float())",
                    ["Ttok", "E"], ["Ttok", "E"], "activation",
                    "路由打分在 float32（:485）")
        self.op("g_moe_router", "GlmMoeDsaTopkRouter.forward", "MoE 路由打分", "linear",
                ["g_moe_flat"], ["g_router_logits"], ["g_moe_router_weight"],
                ["Ttok", "E", "H"],
                [src("485", "router_logits = F.linear(hidden_states.type(float32), self.weight.type(float32))"),
                 src("476", "self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))")],
                "强制 float32：根配置 moe_router_dtype=float32 与源码一致，但源码不读该字段")
        self.tensor("g_router_scores", "scores = router_logits.sigmoid()", ["Ttok", "E"],
                    ["Ttok", "E"], "activation", "sigmoid 打分（不是 softmax）（:486）")
        self.op("g_moe_sigmoid", "GlmMoeDsaTopkRouter.forward", "Sigmoid 打分", "sigmoid",
                ["g_router_logits"], ["g_router_scores"], [], ["Ttok", "E"],
                [src("486", "scores = router_logits.sigmoid()")],
                "GLM-5.3 用 sigmoid；不是 DeepSeek 的 sqrt(softplus)")
        self.tensor("g_scores_for_choice", "scores_for_choice = scores + bias", ["Ttok", "E"],
                    ["Ttok", "E"], "activation", "加 e_score_correction_bias（:487）")
        self.op("g_moe_bias_add", "GlmMoeDsaTopkRouter.forward", "路由修正偏置", "add",
                ["g_router_scores", "g_moe_router_bias"], ["g_scores_for_choice"], [],
                ["Ttok", "E"],
                [src("487", "scores_for_choice = scores + self.e_score_correction_bias")],
                "偏置只参与选择；最终权重仍取 sigmoid 原分数（:503）")
        self.tensor("g_group_scores", "group_scores（top2 求和）", ["Ttok", "Ng"],
                    ["Ttok", "Ng"], "activation",
                    "view(-1,n_group,E/n_group).topk(2).sum(-1)（:488-492）")
        self.op("g_moe_group_score", "GlmMoeDsaTopkRouter.forward", "专家组分数", "view+topk+sum",
                ["g_scores_for_choice"], ["g_group_scores"], [], ["Ng", "E"],
                [src("488-492", "scores_for_choice.view(-1, num_group, num_experts//num_group)"
                                ".topk(2, dim=-1)[0].sum(dim=-1)")],
                "Ng=1 时只有一个组分数；源码仍执行该分支，本图如实表达、不省略也不夸大")
        self.tensor("g_group_idx", "group_idx = topk(group_scores, topk_group)", ["Ttok", "Ktg"],
                    ["Ttok", "Ktg"], "activation", "选中的专家组的索引（:493）")
        self.op("g_moe_group_topk", "GlmMoeDsaTopkRouter.forward", "选专家组", "topk",
                ["g_group_scores"], ["g_group_idx"], [], ["Ng", "Ktg"],
                [src("493", "group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]")],
                "Ktg=1、Ng=1：唯一组就是全部专家")
        self.tensor("g_group_mask", "group_mask（选中组=1）", ["Ttok", "Ng"],
                    ["Ttok", "Ng"], "activation",
                    "zeros_like(group_scores) 后 scatter_(1, group_idx, 1)（:494-495）")
        self.op("g_moe_group_scatter", "GlmMoeDsaTopkRouter.forward", "组掩码 scatter",
                "zeros+scatter", ["g_group_idx"], ["g_group_mask"], [], ["Ttok", "Ng"],
                [src("494-495", "group_mask = torch.zeros_like(group_scores); "
                                "group_mask.scatter_(1, group_idx, 1)")],
                "每个 token 只在选中的组位置置 1；Ng=1 时该掩码全 1")
        self.tensor("g_group_mask_expanded", "group_mask.unsqueeze(-1).expand(-1,Ng,E÷Ng)",
                    ["Ttok", "Ng", "E÷Ng"], ["Ttok", "Ng", "E"], "activation",
                    "group_mask.unsqueeze(-1).expand(-1, n_group, E//n_group)（:496-499）")
        self.op("g_moe_group_expand", "GlmMoeDsaTopkRouter.forward", "组掩码 expand",
                "unsqueeze+expand", ["g_group_mask"], ["g_group_mask_expanded"], [],
                ["Ng", "E"],
                [src("496-499", "group_mask.unsqueeze(-1).expand(-1, self.num_group, "
                                "self.num_experts // self.num_group)")],
                "把每组一个 0/1 标志扩成该组内每个专家一个标志；中间仍是 3D")
        self.tensor("g_score_mask", "score_mask = ...reshape(-1, num_experts)", ["Ttok", "E"],
                    ["Ttok", "E"], "activation",
                    "expand 后 reshape(-1, num_experts)（:500）：最终与 scores_for_choice 同型")
        self.op("g_moe_group_reshape", "GlmMoeDsaTopkRouter.forward", "组掩码 reshape",
                "reshape", ["g_group_mask_expanded"], ["g_score_mask"], [], ["Ttok", "E"],
                [src("499-500", ".reshape(-1, self.num_experts)")],
                "3D [Ttok,Ng,E/Ng] → 2D [Ttok,E]；下游 masked_fill 必须是同型 [Ttok,E]")
        self.tensor("g_masked_scores", "scores_for_choice.masked_fill(~score_mask, -inf)",
                    ["Ttok", "E"], ["Ttok", "E"], "activation",
                    "未被选中的组置 -inf（:501）")
        self.op("g_moe_group_masked_fill", "GlmMoeDsaTopkRouter.forward", "组外专家屏蔽",
                "masked_fill", ["g_scores_for_choice", "g_score_mask"], ["g_masked_scores"], [],
                ["Ng", "E"],
                [src("501", "scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float('-inf'))")],
                "Ng=1 时掩码全 True，不改变任何分数（等价于直接在 256 个专家里 top-8）")
        self.tensor("g_topk_indices", "topk_indices = topk(..., k=K)", ["Ttok", "K"],
                    ["Ttok", "K"], "activation", "选中的专家索引（:502）")
        self.op("g_moe_topk", "GlmMoeDsaTopkRouter.forward", "TopK 专家选择", "topk",
                ["g_masked_scores"], ["g_topk_indices"], [], ["K", "E"],
                [src("502", "topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]")],
                "每 token 选 K=8 个专家；排序无关（sorted=False）")
        self.tensor("g_topk_weights_raw", "topk_weights = scores.gather(1, indices)", ["Ttok", "K"],
                    ["Ttok", "K"], "activation", "取 sigmoid 原分数（:503）")
        self.op("g_moe_weight_gather", "GlmMoeDsaTopkRouter.forward", "收集路由权重", "gather",
                ["g_router_scores", "g_topk_indices"], ["g_topk_weights_raw"], [], ["Ttok", "K"],
                [src("503", "topk_weights = scores.gather(1, topk_indices)")],
                "用未加偏置的 scores；偏置不影响权重数值")
        self.tensor("g_topk_weights_norm", "topk_weights /= sum + 1e-20", ["Ttok", "K"],
                    ["Ttok", "K"], "activation", "归一化（:504-506）")
        self.op("g_moe_weight_norm", "GlmMoeDsaTopkRouter.forward", "路由权重归一化", "normalize",
                ["g_topk_weights_raw"], ["g_topk_weights_norm"], [], ["K"],
                [src("504-506", "if self.norm_topk_prob: topk_weights /= topk_weights.sum(-1,keepdim=True)+1e-20")],
                "norm_topk_prob=true；分母 +1e-20 防零")
        self.tensor("g_topk_weights", "topk_weights = norm * routed_scaling_factor", ["Ttok", "K"],
                    ["Ttok", "K"], "activation", "乘 2.5（:507）")
        self.op("g_moe_weight_scale", "GlmMoeDsaTopkRouter.forward", "路由权重缩放", "scale",
                ["g_topk_weights_norm"], ["g_topk_weights"], [], ["Scl"],
                [src("507", "topk_weights = topk_weights * self.routed_scaling_factor")],
                "缩放只作用于 routed 专家；共享专家另有独立路径")

        # experts（单份模板，不复制 E 份）
        self.tensor("g_expert_mask", "expert_mask = one_hot(top_k_index,E+1).permute(2,1,0)",
                    ["E+1", "K", "Ttok"], ["E", "K", "Ttok"], "activation",
                    "one_hot(top_k_index, num_experts+1) 得 [Ttok,K,E+1]，permute(2,1,0) 得 "
                    "[E+1,K,Ttok]（:531-533）；最后一维 E 是 one_hot 的 no-expert 槽，循环里跳过")
        self.op("g_moe_expert_mask", "GlmMoeDsaExperts.forward", "专家 one-hot 掩码",
                "one_hot+permute", ["g_topk_indices"], ["g_expert_mask"], [],
                ["E", "K", "Ttok"],
                [src("531-533", "with torch.no_grad(): expert_mask = "
                                "torch.nn.functional.one_hot(top_k_index, num_experts+1); "
                                "expert_mask = expert_mask.permute(2, 1, 0)")],
                "保留完整的 [E+1,K,Ttok] 分配矩阵：token/top-k 位置与专家的真实对应关系，"
                "不是只留命中集合；E+1 的最后一槽在循环中 continue 跳过")
        self.tensor("g_expert_hit", "expert_hit（命中的专家索引，数据相关）", ["N_hit"],
                    ["N_hit"], "activation",
                    "expert_mask.sum(dim=(-1,-2))>0 后 nonzero 得命中专家索引（:534-535）；"
                    "命中集合本身是数据相关量，不默认 0/E")
        self.op("g_moe_expert_hit", "GlmMoeDsaExperts.forward", "专家命中集合", "sum+nonzero",
                ["g_expert_mask"], ["g_expert_hit"], [], ["E", "N_hit"],
                [src("534-535", "expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()")],
                "只用于决定单专家模板循环哪些专家；N_hit 未给定，保持未知")
        self.tensor("g_expert_token_idx", "token_idx（命中该专家的 token 行，数据相关）",
                    ["N_e"], ["N_e"], "activation",
                    "torch.where(expert_mask[expert_idx]) 的第二个返回（:540）；"
                    "同一 token 可在多个专家里出现；N_e 数据相关，不猜具体值")
        self.tensor("g_expert_topk_pos", "top_k_pos（该 token 的第几个 top-k 槽）",
                    ["N_e"], ["N_e"], "activation",
                    "torch.where(expert_mask[expert_idx]) 的第一个返回（:540）；"
                    "与 token_idx 一一对应，后续用它索引 topk_weights")
        self.op("g_moe_expert_gather_indices", "GlmMoeDsaExperts.forward",
                "命中专家 → (top_k_pos, token_idx)", "where",
                ["g_expert_mask", "g_expert_hit"],
                ["g_expert_topk_pos", "g_expert_token_idx"], [], ["N_e", "K", "Ttok"],
                [src("536-541", "for expert_idx in expert_hit: "
                                "top_k_pos, token_idx = torch.where(expert_mask[expert_idx])")],
                "对每个命中专家（单份模板）从完整分配矩阵取真实的 (top_k_pos, token_idx)；"
                "两个索引是 gather/权重/index_add 的共同依赖，交换两 token 的专家集合会改变它们")
        self.tensor("g_token_states", "hidden_states[token_idx]", ["N_e", "H"],
                    ["N_e", "H"], "activation",
                    "按 token_idx gather 命中该专家的 token 行（:541）")
        self.op("g_moe_select_tokens", "GlmMoeDsaExperts.forward", "选择专家 token 行", "gather",
                ["g_moe_flat", "g_expert_token_idx"], ["g_token_states"], [], ["N_e", "H"],
                [src("541", "current_state = hidden_states[token_idx]")],
                "只 gather token_idx 指定的行；N_e 是数据相关量，不默认 0")
        self.tensor("g_gate_up", "F.linear(current_state, gate_up_proj[e])", ["N_e", "2Es"],
                    ["N_e", "Es"], "activation", "单专家 gate∪up 合并投影（:542）")
        self.op("g_moe_expert_gate_up", "GlmMoeDsaExperts.forward", "专家 gate_up 投影",
                "linear", ["g_token_states"], ["g_gate_up"], ["g_moe_experts_gate_up"],
                ["H", "Es", "N_e", "E"],
                [src("542", "gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)")],
                "gate/up 在权重里打包成 [E,2Es,H]；按专家索引取切片，不复制 E 份")
        self.tensor("g_expert_gate", "gate", ["N_e", "Es"], ["N_e", "Es"], "activation",
                    "chunk 前段（:542）")
        self.tensor("g_expert_up", "up", ["N_e", "Es"], ["N_e", "Es"], "activation",
                    "chunk 后段（:542）")
        self.op("g_moe_expert_chunk", "GlmMoeDsaExperts.forward", "专家 gate/up 拆分", "chunk",
                ["g_gate_up"], ["g_expert_gate", "g_expert_up"], [], ["Es"],
                [src("542", ".chunk(2, dim=-1)")],
                "沿最后一维均分；2Es=4096 → 两个 2048")
        self.tensor("g_expert_hidden", "act_fn(gate) * up", ["N_e", "Es"], ["N_e", "Es"],
                    "activation", "SiLU 门控（:543）")
        self.op("g_moe_expert_act", "GlmMoeDsaExperts.forward", "专家 SiLU 门控", "silu*mul",
                ["g_expert_gate", "g_expert_up"], ["g_expert_hidden"], [], ["Es"],
                [src("543", "current_hidden_states = self.act_fn(gate) * up")],
                "本模型无 swiglu_limit clamp")
        self.tensor("g_expert_out", "F.linear(current, down_proj[e])", ["N_e", "H"],
                    ["N_e", "H"], "activation", "单专家 down 投影（:544）")
        self.op("g_moe_expert_down", "GlmMoeDsaExperts.forward", "专家 down 投影", "linear",
                ["g_expert_hidden"], ["g_expert_out"], ["g_moe_experts_down"],
                ["Es", "H", "N_e", "E"],
                [src("544", "current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])")],
                "按专家索引取 [H,Es] 切片")
        self.tensor("g_expert_weighted", "current * top_k_weights[token_idx, top_k_pos]",
                    ["N_e", "H"], ["N_e", "H"], "activation",
                    "按 (token_idx, top_k_pos) 取该 token 对该专家的路由权重后缩放（:545）")
        self.op("g_moe_expert_weight", "GlmMoeDsaExperts.forward", "专家输出加权", "scale",
                ["g_expert_out", "g_topk_weights", "g_expert_token_idx", "g_expert_topk_pos"],
                ["g_expert_weighted"], [], ["N_e", "K"],
                [src("545", "current_hidden_states = current_hidden_states * "
                            "top_k_weights[token_idx, top_k_pos, None]")],
                "权重索引同时用 token_idx 与 top_k_pos：只拿命中专家集合无法确定该乘哪个槽")
        self.tensor("g_routed_init", "final_hidden_states = zeros_like(hidden_states)",
                    ["Ttok", "H"], ["Ttok", "H"], "activation",
                    "index_add 的累加初值（:530）；每个 token 行从 0 开始累加")
        self.op("g_moe_result_init", "GlmMoeDsaExperts.forward", "专家输出累加初值",
                "zeros_like", ["g_moe_flat"], ["g_routed_init"], [], ["Ttok", "H"],
                [src("530", "final_hidden_states = torch.zeros_like(hidden_states)")],
                "按展平 token 形状建零张量；后续按 token_idx 累加")
        self.tensor("g_routed_flat", "final_hidden_states（index_add 后）", ["Ttok", "H"],
                    ["Ttok", "H"], "activation", "所有专家输出按 token 累加（:546）")
        self.op("g_moe_index_add", "GlmMoeDsaExperts.forward", "专家输出累加", "index_add",
                ["g_routed_init", "g_expert_weighted", "g_expert_token_idx"],
                ["g_routed_flat"], [], ["Ttok", "H"],
                [src("546", "final_hidden_states.index_add_(0, token_idx, current_hidden_states)")],
                "按 token_idx 把每个专家的加权输出加回对应 token 行；"
                "token_idx 与权重中的 token_idx 来自同一 where 结果")
        self.tensor("g_routed", "hidden_states（view 回 [B,seqlen,H]）", ["B", "seqlen", "H"],
                    ["B", "seqlen", "H"], "activation", "routed 专家输出（:570）")
        self.op("g_moe_routed_view", "GlmMoeDsaMoE.forward", "Routed 输出还原", "view",
                ["g_routed_flat"], ["g_routed"], [], ["B", "seqlen", "H"],
                [src("570", "hidden_states = self.experts(...).view(*orig_shape)")],
                "展平的 token 行还原成 [B,seqlen,H]")
        # shared experts（与 routed 并行）
        self.tensor("g_shared_gate", "shared_experts.gate_proj(residuals)", ["B", "seqlen", "Es"],
                    ["B", "seqlen", "Es"], "activation",
                    "共享专家输入是 routed 之前的 residuals（:566/:571）")
        self.op("g_moe_shared_gate", "GlmMoeDsaMoE.shared_experts", "共享专家 gate_proj", "linear",
                ["g_ffn_in"], ["g_shared_gate"], ["g_moe_shared_gate_weight"],
                ["H", "Es", "seqlen", "B"],
                [src("566", "residuals = hidden_states"),
                 src("571", "hidden_states = hidden_states + self.shared_experts(residuals)")],
                "共享专家宽度 Es×n_shared_experts=2048；与 routed 路径并行（不是串行）")
        self.tensor("g_shared_up", "shared_experts.up_proj(residuals)", ["B", "seqlen", "Es"],
                    ["B", "seqlen", "Es"], "activation", "共享专家 up 投影（:466）")
        self.op("g_moe_shared_up", "GlmMoeDsaMoE.shared_experts", "共享专家 up_proj", "linear",
                ["g_ffn_in"], ["g_shared_up"], ["g_moe_shared_up_weight"],
                ["H", "Es", "seqlen", "B"],
                [src("461", "self.up_proj = nn.Linear(...)")],
                "与 gate 同一输入 residuals")
        self.tensor("g_shared_act", "act_fn(gate) * up（共享专家）", ["B", "seqlen", "Es"],
                    ["B", "seqlen", "Es"], "activation", "SiLU 门控（:466）")
        self.op("g_moe_shared_act", "GlmMoeDsaMoE.shared_experts", "共享专家 SiLU 门控",
                "silu*mul", ["g_shared_gate", "g_shared_up"], ["g_shared_act"], [], ["Es"],
                [src("466", "self.act_fn(self.gate_proj(x)) * self.up_proj(x)")],
                "与 dense/专家相同的 MLP 语义，只是宽度不同")
        self.tensor("g_shared_out", "shared_experts(residuals)", ["B", "seqlen", "H"],
                    ["B", "seqlen", "H"], "activation", "共享专家输出（:571）")
        self.op("g_moe_shared_down", "GlmMoeDsaMoE.shared_experts", "共享专家 down_proj",
                "linear", ["g_shared_act"], ["g_shared_out"], ["g_moe_shared_down_weight"],
                ["Es", "H", "seqlen", "B"],
                [src("462", "self.down_proj = nn.Linear(...)")],
                "回投影到 hidden_size")
        self.tensor("g_mlp_out", "mlp_out（本组 MLP/MoE 输出）", ["B", "seqlen", "H"],
                    ["B", "seqlen", "H"], "activation", "MoE 汇合（:571）")
        self.op("g_moe_merge", "GlmMoeDsaMoE.forward", "Routed + 共享专家汇合", "add",
                ["g_routed", "g_shared_out"], ["g_mlp_out"], [], ["B", "seqlen", "H"],
                [src("572", "return hidden_states")],
                "共享专家是并行支路，不参与路由；两者直接相加")

    # ------------------------------------------------------------------
    # 缓存 / 覆盖注记
    # ------------------------------------------------------------------

    def _build_cache_notes(self) -> None:
        v = self.values
        self.cache_notes = [
            "层缓存类：config.layer_types 全为 indexed_attention（configuration:158-160），"
            "dispatch 到 DynamicIndexedLayer（cache_utils:1290）。本模型没有滑动窗口、"
            "压缩条目或 mHC 状态。",
            "主 KV：update(k_pass, k_rot) 写入的是压缩 latent —— keys=[B,1,Lkv,512]、"
            "values=[B,1,Lkv,64]（modeling:396-398；cache_utils:132-153）。"
            "expand_kv/kv_b_proj 在**每次调用**对全部 Lkv 个位置展开（:351-368/:402），"
            "缓存本身不存展开后的 64 头 K/V。",
            "indexer 键缓存：只有 full indexer 层有 [B,Lidx,128] 的单头键缓存"
            "（modeling:225-226；cache_utils:331-365）；shared 层不实例化 indexer"
            "（:348-349），复用上一层 full 层的 prev_topk_indices，不写缓存。",
            "Lkv = Lidx = kvcachelen + seqlen：两套缓存都线性保留全部历史。"
            "kvcachelen 是调用前逻辑历史 token 数（空缓存 Prefill 取 0），"
            "不是字节数或物理保留条目数。",
            "top-k 选择只影响注意力可见集合：K_sel=min(index_topk, Lidx)。"
            "Lidx≤index_topk 时全选，稀疏 mask 不删任何位置；"
            "超过时每 query 只保留 2048 个 key（eager 路径仍计算全长度 QKᵀ 再屏蔽）。",
        ]

    def _build_coverage_notes(self) -> None:
        v = self.values
        full_layers = [i for i in range(v.num_hidden_layers) if v.indexer_type(i) == "full"]
        shared_layers = [i for i in range(v.num_hidden_layers) if v.indexer_type(i) == "shared"]
        dense_layers = [i for i in range(v.num_hidden_layers) if v.mlp_layer_type(i) == "dense"]
        indexer_note = (
            f"本层是 {self.indexer_type} indexer："
            + ("实例化 wq_b/wk/k_norm/weights_proj 并写 indexer 键缓存。"
               if self.indexer_type == "full" else
               "不实例化 indexer；直接消费上一层 full 层的 top-k（prev_topk_indices）。")
        )
        self.coverage_notes = [
            f"已表达：DecoderLayer 两处普通残差、input/post RMSNorm、MLA（q_a/q_b、"
            f"kv_a latent、交错部分 RoPE、压缩 latent 缓存、kv_b 展开、o_proj）、"
            f"DSA 稀疏掩码与 eager 注意力、{'稠密 MLP' if self.mlp_type == 'dense' else 'TopK MoE + 共享专家'}。",
            indexer_note,
            f"真实层归属：dense 层 {dense_layers}；full indexer 层 {len(full_layers)} 个"
            f"（{full_layers[:6]}…{full_layers[-2:]}）；shared 层 {len(shared_layers)} 个。"
            "组的完整层索引见整网/层组层；本 Block 只代表本组模板。",
            "根配置 quantization_config 声明 FP8（e4m3, block 128×128）：本图只表达 HF "
            "逻辑 eager 路径，不分析量化 kernel、FP8 数值或权重内容。该配置的 "
            "modules_to_not_convert 只把 full 层（0/1/2/6/10/…/74）与 MTP 层 78 的 "
            "indexer/q_a/kv_a 等列为不转换，可交叉印证 shared 层没有 indexer 参数。",
            "网络级因果 mask 由 create_causal_mask(..., allow_is_causal_skip=False) 构建"
            "（modeling:696-706）；本图不分析 2D padding mask、batch packing 或 "
            "flash_mla 的 indices 核路径（_supports_flash_attn=False，modeling:626）。",
            "未覆盖：MTP 层 78（modeling:638 显式忽略 + forward 只循环 0..77）、"
            "generate/采样循环、loss/aux_loss、output_router_logits 辅助输出、"
            "torch.compile/梯度检查点、数值等价。",
            "数据相关量（N_hit、N_e、路由实际分布）保留未知；不根据权重推断 MoE 实际路由，"
            "不默认 0。专家只有一份模板，不按 256 个专家复制图。",
            "视觉/图像输入路径不在本票范围（模型卡 pipeline_tag=text-generation；"
            "本模块只解释语言主干）。",
        ]
