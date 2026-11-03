"""GLM-5.3 整网总览数据提取（token → logits，静态只读，不实例化/不运行模型）。

从 ``rules/glm53_hf.py`` 解释的有效配置与本地官方 Transformers 检出源码，构建
``hf_network_overview.v1``：一次网络调用的外围（embedding / 默认位置序列 / 因果
mask / 交错 RoPE）→ 78 层真实 ModuleList 循环（三组层型归属见 groups/
layer_order）→ Final RMSNorm → logits 位置切片 → lm_head → logits，以及到
Block/算子详图的往返目标。

表达事实（行号按本地检出 536ecc00…，文件 modeling_glm_moe_dsa.py，下称 modeling）：
- ``GlmMoeDsaForCausalLM.forward``（:749-804）：调用 ``self.model(...)``（:778-786），
  取 ``outputs.last_hidden_state``（:788），按 ``logits_to_keep`` 取
  ``slice(-k, None)``（默认 0 → slice(0, None)），再
  ``self.lm_head(hidden_states[:, slice_indices, :])``（:790-791）；
  ``lm_head`` 是 ``nn.Linear(hidden_size, vocab_size, bias=False)``（:742）。
- ``GlmMoeDsaModel.__init__``（:654-667）：``embed_tokens``（:658）、78 个
  ``GlmMoeDsaDecoderLayer``（:659-661）、Final RMSNorm（:662）、
  ``rotary_emb``（:663）。``_keys_to_ignore_on_load_unexpected =
  [r"model\\.layers\\.78.*"]``（:638）：checkpoint 的 layer 78 是 MTP 头，
  本地实现**不构造也不执行**它；forward 只循环 ``layers[:num_hidden_layers]``
  （:712）。
- ``GlmMoeDsaModel.forward``（:672-728）：embed（:685-686）→ use_cache 且无 cache
  时新建 DynamicCache（:688-689）→ ``position_ids = arange(seqlen)+past_seen``
  后 unsqueeze(0)（:691-694）→ ``create_causal_mask(..., allow_is_causal_skip=False)``
  并放在 ``{"indexed_attention": ...}``（:696-706）→
  ``position_embeddings = self.rotary_emb(hidden_states, position_ids)``（:708-709）
  → 逐层调用（:712-722），层间传递 ``topk_indices``（full→shared 的 top-k 复用）
  → ``hidden_states = self.norm(hidden_states)``（:724）。
- 交错 RoPE（:67-111/:114-150）：``head_dim`` 在 __post_init__ 被指向
  ``qk_rope_head_dim=64``；cos/sin 尾维 64（32 对），按 (0::2,1::2) 成对旋转。
- 缓存：所有层 ``layer_types="indexed_attention"`` → ``DynamicIndexedLayer``
  （cache_utils:1290）；主 KV 缓存压缩 latent（512+64），full indexer 层另有
  [B,L,128] 键缓存；逐层细节在 Block 详图，本图只表达网络级读前/写后状态。

三组真实层组与代表层来自 ``rules/glm53_hf.py:LAYER_GROUP_SPECS``；层归属由
``layer_group_assignments`` 按 ``indexer_types``/``mlp_layer_types`` 逐层推导并
校验三组恰好覆盖全部 78 层。共享模板不代表共享权重：同组层各有独立参数，
``tie_word_embeddings=false`` 时 ``embed_tokens`` 与 ``lm_head`` 也是两份独立权重。

视觉/图像输入路径不在本票范围：GLM-5.3 仓库 ``pipeline_tag=text-generation``，
本模块只解释语言主干。本模块只读源码/配置，不导入 torch/transformers、不实例化
或运行模型、不下载权重，不生成 HTML/Excel，不做数值等价声明。
"""

from __future__ import annotations

from pathlib import Path

from structure_reader.rules.glm53_hf import (
    MODEL_REPOSITORY,
    SOURCE_FILES,
    HfModelValues,
    layer_group_assignments,
    load_hf_model_values,
)

__all__ = [
    "NETWORK_SCHEMA",
    "DETAIL_PAGE",
    "NetworkDocError",
    "build_network_overview",
    "logits_slice_length",
]

NETWORK_SCHEMA = "hf_network_overview.v1"
DETAIL_PAGE = "structure.html"
PATH_ID = "network"


class NetworkDocError(ValueError):
    """整网总览数据构建失败（配置/范围/事实来源不一致），具名拒绝。"""


def logits_slice_length(seq_len: int, logits_to_keep: int) -> int:
    """int 型 ``logits_to_keep`` 下 lm_head 实际消费的位置数（modeling :790）。"""
    if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len < 0:
        raise ValueError(f"seq_len 必须是非负整数: {seq_len!r}")
    if not isinstance(logits_to_keep, int) or isinstance(logits_to_keep, bool):
        raise ValueError(f"logits_to_keep 只分析 int 路径: {logits_to_keep!r}")
    if logits_to_keep == 0:
        return seq_len
    if logits_to_keep > 0:
        return min(seq_len, logits_to_keep)
    raise ValueError(
        f"logits_to_keep={logits_to_keep}（负整数，slice 正起点）不在本票已分析范围"
    )


def build_network_overview(
    workspace_root: str | Path,
    model_key: str = "glm_5_3_hf",
) -> dict:
    """构建 ``hf_network_overview.v1``（单次网络调用的 token→logits 主干）。"""
    workspace_root = Path(workspace_root)
    values = load_hf_model_values(workspace_root, model_key=model_key)
    try:
        groups = layer_group_assignments(values)
    except ValueError as exc:
        raise NetworkDocError(f"三组层归属校验失败: {exc}") from exc
    builder = _NetworkBuilder(values, groups)
    path = builder.build()
    layer_order = _layer_order(values, groups)
    full_layers = [i for i in range(values.num_hidden_layers)
                   if values.indexer_type(i) == "full"]
    shared_layers = [i for i in range(values.num_hidden_layers)
                     if values.indexer_type(i) == "shared"]
    dense_layers = [i for i in range(values.num_hidden_layers)
                    if values.mlp_layer_type(i) == "dense"]
    return {
        "schema": NETWORK_SCHEMA,
        "model": {
            "model_key": values.model_key,
            "display_name": values.display_name,
            "implementation": values.implementation,
            "source_revision": values.source_revision,
            "config_path": values.config_path,
            "model_revision": values.model_revision,
        },
        "scope": {
            "entry_class": "GlmMoeDsaForCausalLM",
            "main_module_class": "GlmMoeDsaModel",
            "decoder_layer_class": "GlmMoeDsaDecoderLayer",
            "analysis_path": "eager",
            "layer_count": values.num_hidden_layers,
            "hidden_size": values.hidden_size,
            "vocab_size": values.vocab_size,
            "hidden_act": values.hidden_act,
            "dense_layers": dense_layers,
            "indexer_full_layers": full_layers,
            "indexer_shared_layers": shared_layers,
            "topk_layers": len(full_layers),
            "shared_topk_layers": len(shared_layers),
            "main_path": [op["id"] for op in path["operators"]],
            "notes": [
                "范围：单次网络调用（一次 forward）从 input_ids 到 logits；不含 generate "
                "采样循环、外部控制循环或多次调用耗时。",
                f"表达方式：整网外围与终点按源码逐句静态分析；循环体 "
                f"{values.num_hidden_layers} 层不铺开，三组层型的层内结构由代表层 "
                f"Block/算子详图（同候选的 {DETAIL_PAGE}）提供，本总览给出真实层归属、"
                "顺序与往返入口。",
                "本模型是标准单流残差：层内两处普通残差相加；没有 mHC/HyperConnection/"
                "多流/Sinkhorn，也没有滑动窗口或压缩条目。不要把这些 DeepSeek 机制套到"
                "GLM-5.3 上。",
                "注意力是 MLA + DSA 稀疏选择：所有层都是 indexed_attention，full 层运行"
                "自己的 indexer 并写 [B,L,128] 键缓存，shared 层复用上一层 full 层的 top-k；"
                "组详图分别给出两类层型。",
                "MTP：checkpoint 的 model.layers.78.* 是训练期多 token 预测头，本地实现"
                "显式忽略（modeling:638）且 forward 只循环 0..77 —— 不在本主干。",
                "tie_word_embeddings=false（根配置）：embed_tokens 与 lm_head 是两份独立权重。",
                "视觉/图像输入路径不在本票范围：模型卡 pipeline_tag=text-generation，"
                "本图只覆盖 token 输入到 logits 的语言主干。",
                "本图不加载权重、不运行模型、不做数值等价；seqlen/kvcachelen 等调用维度保留"
                "符号，空缓存 Prefill 取 kvcachelen=0；场景预设只绑定取值，不改变整网结构。",
            ],
        },
        "path": path,
        "groups": groups,
        "layer_order": layer_order,
        "detail": {
            "href": DETAIL_PAGE,
            "groups": [
                {
                    "id": group["id"],
                    "label": group["label"],
                    "count": group["count"],
                    "layer_indices": group["layer_indices"],
                    "representative_layer_index": group["representative_layer_index"],
                    "indexer_type": group["indexer_type"],
                    "anchors": _detail_anchors(group["id"]),
                }
                for group in groups
            ],
        },
    }


def _detail_anchors(group_id: str) -> dict:
    base = f"/group/{group_id}"
    return {
        "group": base,
        "block": f"{base}/block",
        "operators": f"{base}/ops",
        "tensors": f"{base}/ops",
    }


def _layer_order(values: HfModelValues, groups: list[dict]) -> list[dict]:
    """真实层顺序：ModuleList 索引 0..N-1，逐层标注层型/MLP 类型/所属组。"""
    group_of = {
        index: group["id"] for group in groups for index in group["layer_indices"]
    }
    order = []
    for index in range(values.num_hidden_layers):
        group_id = group_of.get(index)
        if group_id is None:  # 归属校验已保证覆盖；此处不静默补位
            raise NetworkDocError(f"层 {index} 未归属任何真实层组（覆盖校验与实际不一致）")
        order.append({
            "index": index,
            "attention_type": values.layer_type(index),
            "indexer_type": values.indexer_type(index),
            "moe_type": values.mlp_layer_type(index),
            "group_id": group_id,
        })
    return order


class _NetworkBuilder:
    """网络级算子/张量/参数累积器（按真实阅读顺序追加）。"""

    def __init__(self, values: HfModelValues, groups: list[dict]):
        self.values = values
        self.groups = groups
        self.parameters: list[dict] = []
        self.tensors: list[dict] = []
        self.operators: list[dict] = []
        self.cache_notes: list[str] = []
        self.coverage_notes: list[str] = []

    # ------------------------------------------------------------------
    # 记录辅助
    # ------------------------------------------------------------------

    def param(self, pid, symbol, meaning, value, expression, note, source_refs) -> str:
        self.parameters.append({
            "id": pid, "symbol": symbol, "meaning": meaning, "value": value,
            "expression": expression, "note": note, "source_refs": source_refs,
        })
        return pid

    def tensor(self, tid, name, shape, parameter_refs, kind, note) -> str:
        self.tensors.append({
            "id": tid, "name": name, "shape": shape, "parameter_refs": parameter_refs,
            "kind": kind, "note": note,
        })
        return tid

    def op(self, oid, module_path, name, op, inputs, outputs, weights,
           parameter_refs, source_refs, note, analysis_status="analyzed") -> str:
        self.operators.append({
            "id": oid, "module_path": module_path, "name": name, "op": op,
            "inputs": inputs, "outputs": outputs, "weights": weights,
            "parameter_refs": parameter_refs, "source_refs": source_refs,
            "analysis_status": analysis_status, "note": note,
        })
        return oid

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
    # 主路径
    # ------------------------------------------------------------------

    def build(self) -> dict:
        self._build_parameters()
        self._build_boundaries()
        self._build_forward_path()
        self._build_cache_notes()
        self._build_coverage_notes()
        return {
            "id": PATH_ID,
            "title": (
                f"GLM-5.3 整网主干（{self.values.num_hidden_layers} 层，token → logits；"
                "单次 forward）"
            ),
            "parameters": self.parameters,
            "tensors": self.tensors,
            "operators": self.operators,
            "cache_notes": self.cache_notes,
            "coverage_notes": self.coverage_notes,
        }

    def _build_parameters(self) -> None:
        v = self.values
        cfg, src = self.cfg, self.src
        self.param("B", "B", "batch 批大小", None, None,
                   "调用输入维度，静态分析保留符号", [])
        self.param("seqlen", "seqlen", "本次调用新输入 token 数（seqlen）", None,
                   "T = inputs_embeds.shape[1]",
                   "整网模板对 Prefill/Decode 同构：Prefill 为提示长度、Decode 为 1；"
                   "结构保留符号，场景参数只绑定取值",
                   [src("691-694", "torch.arange(inputs_embeds.shape[1])")])
        self.param("kvcachelen", "kvcachelen", "调用前逻辑历史 token 数（kvcachelen）", None,
                   "P = past_key_values.get_seq_length()（无 cache 时 0）",
                   "空缓存 Prefill 取 0；Decode 为调用前已处理历史；不是物理保留条目数"
                   "或字节数。本模型无滑动窗口/压缩条目",
                   [src("692", "past_seen_tokens = past_key_values.get_seq_length()")])
        self.param("H", "H", "hidden_size 隐层维度", v.hidden_size, None,
                   "config.hidden_size", [cfg("hidden_size", "根配置字段")])
        self.param("Vocab", "Vocab", "vocab_size 词表大小", v.vocab_size, None,
                   "config.vocab_size", [cfg("vocab_size", "根配置字段")])
        self.param("Nlay", "N_layer", "num_hidden_layers 主干层数（循环次数）",
                   v.num_hidden_layers, None,
                   "config.num_hidden_layers=78（0..77）；layer_order 逐层列真实顺序。"
                   "checkpoint 的 layer 78 是 MTP，被显式忽略（modeling:638）",
                   [cfg("num_hidden_layers", "根配置字段"),
                    src("659-661", "nn.ModuleList([GlmMoeDsaDecoderLayer(config, i) for i in range(...)])"),
                    src("638", "_keys_to_ignore_on_load_unexpected = [r\"model\\.layers\\.78.*\"]")])
        self.param("Rk", "Rk", "qk_rope_head_dim 旋转维（=运行期 head_dim）",
                   v.qk_rope_head_dim, None,
                   "config.qk_rope_head_dim=64；根配置 head_dim=192 被 __post_init__ 覆盖，"
                   "RoPE cos/sin 尾维为 64（32 对）",
                   [cfg("qk_rope_head_dim", "根配置字段"),
                    cfg("head_dim", "根配置保存值（被覆盖）"),
                    self.srcf("configuration_glm_moe_dsa.py", "150-152",
                              "self.head_dim = self.qk_rope_head_dim"),
                    src("142-149", "apply 时只取前一半角度")])
        self.param("ThetaM", "ThetaM", "rope_theta RoPE 底数", v.rope_theta, None,
                   "config.rope_parameters.rope_theta=8e6（default 型，无 YaRN mscale）",
                   [cfg("rope_parameters", "根配置字段"),
                    src("84-101", "compute_default_rope_parameters")])
        self.param("Lmask", "L_mask", "模型级因果 mask 的 KV 长度（数据相关）", None,
                   "L_mask = kvcachelen + seqlen（allow_is_causal_skip=False）",
                   "create_causal_mask 的 kv_length = 已缓存 + query（cache_utils:165-172）；"
                   "mask 在网络入口构建一次并传给全部层",
                   [src("696-706", "create_causal_mask(..., allow_is_causal_skip=False)"),
                    self.srcf("cache_utils.py", "165-172", "get_mask_sizes：kv_length")])
        self.param("Lk", "Lk", "logits_to_keep 保留末位 token 数", 0,
                   "logits_to_keep 默认 0",
                   "0 → slice(0, None) 保留全部 seqlen 个位置；k>0 → slice(-k, None) 保留末 "
                   "min(seqlen, k) 个位置（越界不扩大）；负整数与 tensor 索引不在已分析范围",
                   [src("758", "logits_to_keep: int | torch.Tensor = 0"),
                    src("790", "slice(-logits_to_keep, None) if isinstance(int)")])
        self.param("Ssel", "S_sel", "logits 序列位置数（切片后）", None,
                   "S_sel = seqlen（Lk=0）或 min(seqlen, Lk)（Lk>0）",
                   f"lm_head 只对切片后的 hidden_states 计算，输出 [B,S_sel,Vocab]；"
                   f"例：seqlen=1、Lk=5 时 S_sel={logits_slice_length(1, 5)}（负起点裁剪为 0）",
                   [src("788-791", "hidden_states[:, slice_indices, :] → self.lm_head(...)")])
        self.param("EpsN", "EpsN", "rms_norm_eps Final RMSNorm epsilon", v.rms_norm_eps, None,
                   "config.rms_norm_eps", [cfg("rms_norm_eps", "根配置字段"),
                                           src("724", "hidden_states = self.norm(hidden_states)")])
        self.param("Nmtp", "N_mtp", "num_nextn_predict_layers MTP 头数（不在主干）",
                   v.num_nextn_predict_layers, None,
                   "checkpoint 的 layer 78 是 MTP（含 hnorm/enorm/eh_proj/shared_head）；"
                   "本地实现忽略其权重且 forward 不执行 —— 本图明确不含 MTP",
                   [cfg("num_nextn_predict_layers", "根配置字段"),
                    src("638", "_keys_to_ignore_on_load_unexpected"),
                    src("712", "layers[:num_hidden_layers] 循环")])

    def _build_boundaries(self) -> None:
        self.tensor(
            "input_ids", "input_ids", ["B", "seqlen"], ["B", "seqlen"], "boundary",
            "调用入口 token id；与 inputs_embeds 恰一（modeling:682-686）。"
            "本模型没有 hash 路由，MoE 路由不消费 input_ids")
        self.tensor(
            "attention_mask", "attention_mask（调用方 2D mask，可选）", ["B", "kvcachelen+seqlen"],
            ["kvcachelen", "seqlen", "B"], "boundary",
            "调用方提供的可见性/padding 掩码；None 时只按因果性构建（modeling:691-706）。"
            "若已是 dict（generate 预建 per-layer mask）则直接取用")
        self.tensor(
            "past_key_values", "past_key_values（DynamicCache，调用前）", ["Nlay"], ["Nlay"],
            "cache",
            "use_cache 且未传入时新建 DynamicCache（:688-689）；按 layer_types="
            "indexed_attention 分发 DynamicIndexedLayer（cache_utils:1290）；逐层字段见组详图")
        self.tensor(
            "inputs_embeds", "inputs_embeds", ["B", "seqlen", "H"], ["B", "seqlen", "H"],
            "activation",
            "embed_tokens 输出（:685-686）；调用方也可直接传 inputs_embeds 跳过 embedding")
        self.tensor(
            "position_ids", "position_ids", ["1", "seqlen"], ["seqlen", "kvcachelen"],
            "activation",
            "默认 arange(seqlen)+past_seen 后 unsqueeze(0)（:691-694）；供 rotary_emb 与 "
            "mask 消费；调用方可传入自定义 [B,seqlen] 位置（本图不含该可选组合）")
        self.tensor(
            "causal_mask", "causal_mask", ["B", "1", "seqlen", "Lmask"],
            ["B", "seqlen", "Lmask"], "mask",
            "create_causal_mask(..., allow_is_causal_skip=False) 生成的 4D float mask"
            "（:696-706）；放字典键 'indexed_attention' 传给全部层")
        self.tensor(
            "hidden_after_layers", "hidden_states（全部层之后）", ["B", "seqlen", "H"],
            ["B", "seqlen", "H"], "activation",
            "循环最后一次赋值的 hidden_states（:712-722）；仍是单流，未做任何收拢")
        self.tensor(
            "past_key_values_post", "past_key_values（各层 update 后，同一对象）",
            ["Nlay"], ["Nlay"], "cache",
            "各层 update/update_indexer 原地写入同一 DynamicCache（:718-719），"
            "model 与 ForCausalLM 一并返回（:725-727/:800）；逐层字段见组详图")

    def _build_forward_path(self) -> None:
        v = self.values
        src, cfg = self.src, self.cfg

        # 1) embedding
        self.tensor(
            "embed_tokens_weight", "self.embed_tokens.weight", ["Vocab", "H"],
            ["Vocab", "H"], "weight",
            "nn.Embedding(vocab_size, hidden_size, padding_idx)（:658）；"
            "tie_word_embeddings=false → 与 lm_head.weight 独立")
        self.op(
            "net_embed", "GlmMoeDsaModel.embed_tokens", "Token Embedding（查表）",
            "embedding+gather", ["input_ids"], ["inputs_embeds"],
            ["embed_tokens_weight"], ["B", "seqlen", "H", "Vocab"],
            [src("658", "self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx)"),
             src("685-686", "if inputs_embeds is None: inputs_embeds = self.embed_tokens(input_ids)"),
             src("682-684", "input_ids 与 inputs_embeds 恰一（XOR 检查）")],
            "token id → [B,seqlen,H] 嵌入；padding_idx=config.pad_token_id=154820")

        # 2) 默认 position_ids
        self.op(
            "net_position_ids", "GlmMoeDsaModel.forward", "默认位置序列",
            "arange+add+unsqueeze", ["past_key_values"], ["position_ids"], [],
            ["seqlen", "kvcachelen"],
            [src("691-694", "past_seen_tokens = get_seq_length(); position_ids = "
                            "arange(T) + past_seen; unsqueeze(0)")],
            "仅在调用方未给 position_ids 时执行；kvcachelen 来自 cache 的 "
            "get_seq_length()；输出 [1,seqlen] 供 rotary 与 mask 使用")

        # 3) 因果 mask
        self.op(
            "net_mask_build", "GlmMoeDsaModel.forward", "因果 mask 构建",
            "mask_build", ["inputs_embeds", "attention_mask", "past_key_values", "position_ids"],
            ["causal_mask"], [], ["B", "seqlen", "Lmask"],
            [src("696-706", "causal_mask_mapping = {'indexed_attention': create_causal_mask(...)}"),
             src("704", "allow_is_causal_skip=False（强制构建，保证 indexer 因果性）"),
             self.srcf("masking_utils.py", "850-", "create_causal_mask"),
             self.srcf("cache_utils.py", "165-172", "get_mask_sizes：kv_length=已缓存+query")],
            "网络入口构建一次，循环内传同一 mask（:715）；层内再用 top-k 索引把它改成"
            "稀疏 mask（DSA，组详图）")

        # 4) 交错 RoPE 频率
        for key in ("cos", "sin"):
            self.tensor(
                f"{key}_rope", f"position_embeddings {key}", ["1", "seqlen", "Rk"],
                ["Rk", "seqlen"], "activation",
                f"self.rotary_emb(hidden_states, position_ids) 的 {key}（:708-709）；"
                "尾维 64=32 对频率，按交错布局使用")
        self.op(
            "net_rope", "GlmMoeDsaModel.rotary_emb", "交错 RoPE 频率", "cos/sin",
            ["inputs_embeds", "position_ids"], ["cos_rope", "sin_rope"], [],
            ["Rk", "seqlen", "ThetaM"],
            [src("708-709", "position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)"),
             src("103-111", "freqs = position_ids[...,None]*inv_freq; cat((freqs,freqs)); cos/sin"),
             src("84-101", "compute_default_rope_parameters：dim=config.head_dim=64；theta=8e6"),
             cfg("rope_parameters", "rope_type=default；无 YaRN mscale")],
            "只有一组 RoPE 频率（不像多组压缩 RoPE 的模型）；主注意力与 indexer 共用，"
            "apply 时按 (0::2,1::2) 交错旋转")

        # 5) 层循环
        self.op(
            "net_layer_loop", "GlmMoeDsaModel.layers",
            f"{v.num_hidden_layers} 层循环（三组真实层型，顺序见 layer_order）",
            "loop",
            ["inputs_embeds", "cos_rope", "sin_rope", "position_ids", "causal_mask",
             "past_key_values"],
            ["hidden_after_layers", "past_key_values_post"], [], ["Nlay", "B", "seqlen"],
            [src("659-661", "self.layers = nn.ModuleList([GlmMoeDsaDecoderLayer(config, i) "
                            "for i in range(num_hidden_layers)])"),
             src("712-722", "for i, decoder_layer in enumerate(self.layers[:num_hidden_layers]): "
                            "hidden_states, topk_indices = decoder_layer(..., prev_topk_indices=topk_indices)"),
             src("575-616", "GlmMoeDsaDecoderLayer.forward（循环体；MLA+DSA+MoE，见组详图）")],
            f"同一模板按 ModuleList 顺序重复 {v.num_hidden_layers} 次，不铺开 78 层："
            "组内结构由代表层详图提供；topk_indices 在层间单向传递（full 层产出、shared 层复用）；"
            "cache 由各层 update/update_indexer 原地写回")

        # 6) Final RMSNorm
        self.tensor(
            "final_norm_weight", "self.norm.weight", ["H"], ["H"], "weight",
            "Final RMSNorm 可学习缩放（:662/:53）")
        self.tensor(
            "final_hidden", "hidden_states（Final RMSNorm 后）", ["B", "seqlen", "H"],
            ["H", "EpsN"], "activation",
            "model 的 last_hidden_state（:724-727）；进入 lm_head 前不再有其它网络级变换")
        self.op(
            "net_final_norm", "GlmMoeDsaModel.norm", "Final RMSNorm", "rmsnorm",
            ["hidden_after_layers", "final_norm_weight"], ["final_hidden"], [],
            ["H", "EpsN"],
            [src("724", "hidden_states = self.norm(hidden_states)"),
             src("662", "self.norm = GlmMoeDsaRMSNorm(hidden_size, eps=rms_norm_eps)"),
             src("56-61", "fp32 方差归一 → weight 乘 → to(input_dtype)")],
            "本模型没有 HyperHead/mHC 收拢：循环输出直接进 RMSNorm")

        # 7) logits 切片 + lm_head
        self.tensor(
            "lm_input", "hidden_states[:, slice_indices, :]（切片后）", ["B", "Ssel", "H"],
            ["B", "Ssel", "H"], "activation",
            "只保留需要的序列位置（:790）；Lk=0 时保留全部 seqlen 位；"
            "Lk>0 时长度为 min(seqlen, Lk)")
        self.op(
            "net_logits_slice", "GlmMoeDsaForCausalLM.forward", "logits 位置切片",
            "slice", ["final_hidden"], ["lm_input"], [], ["Lk", "Ssel"],
            [src("788", "hidden_states = outputs.last_hidden_state"),
             src("790", "slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int)")],
            "切片发生在 lm_head 之前（只对保留位置做投影）")
        self.tensor(
            "lm_head_weight", "self.lm_head.weight", ["Vocab", "H"], ["Vocab", "H"], "weight",
            "nn.Linear(hidden_size, vocab_size, bias=False)（:742）；"
            "tie_word_embeddings=false → 与 embed_tokens.weight 不共享")
        self.tensor(
            "logits", "logits", ["B", "Ssel", "Vocab"], ["B", "Ssel", "Vocab"], "boundary",
            "网络终点（:791/:798-804）：未归一化分数；Lk>0 时序列轴为 min(seqlen, Lk)；"
            "不在此图做 softmax/采样")
        self.op(
            "net_lm_head", "GlmMoeDsaForCausalLM.lm_head", "LM Head 投影（得到 logits）",
            "linear", ["lm_input"], ["logits"], ["lm_head_weight"], ["Vocab", "H", "Ssel"],
            [src("742", "self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)"),
             src("790-791", "logits = self.lm_head(hidden_states[:, slice_indices, :])"),
             src("797-804", "MoeCausalLMOutputWithPast(logits=logits, past_key_values=..., ...)")],
            "网络终点：hidden → vocab 的线性投影；loss/aux_loss/router_logits 属训练或"
            "可选辅助输出，不在本主干")

    def _build_cache_notes(self) -> None:
        self.cache_notes = [
            "past_key_values 在网络入口按需新建 DynamicCache（modeling:688-689）；所有层"
            "layer_types=indexed_attention → DynamicIndexedLayer（cache_utils:1290）。",
            "主 KV 缓存存的是压缩 latent：keys=[B,1,L,512]、values=[B,1,L,64]"
            "（modeling:396-398）；展开成 64 头 K/V 发生在每层调用内（kv_b_proj，"
            "不再是缓存内容）。",
            "只有 full indexer 层额外有 indexer 键缓存 [B,L,128]（modeling:225-226/"
            "cache_utils:350-365）；shared 层没有 indexer，也没有 indexer 缓存。",
            "P = past_key_values.get_seq_length()（:692）：决定默认 position_ids 的起点与"
            "模型级 mask 尺寸；空缓存 Prefill 时 kvcachelen=0。",
            "本模型没有滑动窗口、压缩条目或 mHC 状态；Lkv=Lidx=kvcachelen+seqlen，"
            "线性保留全部历史（逐层见组详图）。",
        ]

    def _build_coverage_notes(self) -> None:
        v = self.values
        full_layers = [i for i in range(v.num_hidden_layers)
                       if v.indexer_type(i) == "full"]
        shared_layers = [i for i in range(v.num_hidden_layers)
                         if v.indexer_type(i) == "shared"]
        self.coverage_notes = [
            "已表达（与 modeling 行号逐条对应）：ForCausalLM.forward → Model.forward → "
            "embed_tokens、默认 position_ids、因果 mask、交错 RoPE 频率、"
            f"{v.num_hidden_layers} 层循环（真实层序与三组归属）、Final RMSNorm、"
            "logits 位置切片、lm_head → logits、past_key_values 读取与写回。",
            "层内结构不在此图重复铺开：MLA（q_a/q_b、压缩 latent 缓存、kv_b 展开、"
            "交错部分 RoPE）、DSA indexer 与稀疏 mask、稠密 MLP / TopK MoE + 共享专家"
            "由同候选分层浏览页的层组/Block/算子层提供（structure.html）。",
            f"full indexer 层 {len(full_layers)} 个（{full_layers}）；"
            f"shared 层 {len(shared_layers)} 个（前 6 个 {shared_layers[:6]}…）；"
            "dense MLP 层 0/1/2。shared 层的 top-k 复用上一 full 层，不是持久缓存。",
            "同构模板复用不代表共享权重：同组每层是独立 GlmMoeDsaDecoderLayer 实例；"
            "本配置 tie_word_embeddings=false，embed_tokens 与 lm_head 权重也独立。",
            "MTP：model.layers.78.* 是训练期多 token 预测头（hnorm/enorm/eh_proj/"
            "shared_head），本地实现显式忽略（modeling:638）且 forward 只循环 0..77 —— "
            "不在本主干，未分析。",
            "logits_to_keep 只分析非负 int 路径：Lk=0 保留全部 seqlen 位，Lk>0 为 "
            "min(seqlen, Lk)（slice(-Lk,None) 负起点越界裁剪）；负整数与 tensor 型索引未分析。",
            "未覆盖（明确列出，不宣称已分析）：generate/采样循环与多次调用累计、"
            "loss/aux_loss 训练路径、output_router_logits 辅助输出、量化执行"
            "（FP8 e4m3 与设备 kernel）、flash_mla indices 核路径、torch.compile/"
            "梯度检查点、数值等价与权重内容。",
            "视觉/图像输入路径不在本票范围：模型卡 pipeline_tag=text-generation，"
            "本图只覆盖 token 输入到 logits 的语言主干。",
            "数据相关量（路由命中专家数、每专家 token 数、命中分布等）保留符号与派生式；"
            "不填运行时假设值，不根据权重推断 MoE 实际路由。",
        ]
