"""GLM-5.3-Flash 整网总览数据提取（token → logits，静态只读，不实例化/不运行模型）。

从 ``rules/glm53flash_hf.py`` 解释的文本有效配置与本地官方 Transformers 检出源码，
构建 ``hf_network_overview.v1``：一次网络调用的外围（文本 embedding / 默认
position_ids / 2D padding mask / 4 路 mHC 展开）→ 45 层真实 ModuleList 循环
（三组层型归属见 groups/layer_order）→ mHC 均值收拢 → Final RMSNorm →
logits 位置切片 → lm_head → logits，以及到 Block/算子详图的往返目标。

表达事实（行号按本地检出 536ecc00…，文件 modeling_glm5_next.py，下称 modeling）：

- 语言主干入口是 ``Glm5NextTextModel``（``model.language_model``，modeling:1889）；
  ``Glm5NextForConditionalGeneration`` 在 ``self.model`` 之后用独立 ``lm_head``
  得到 logits（:2145/:2251-2253）。本图只走 text_config 路径：视觉 ``self.visual``
  与图像/视频 placeholder masked_scatter（:2014-2041）不在范围。
- ``Glm5NextTextModel.forward``（:1447-1510）：embed（:1463-1464）→ use_cache 且无
  cache 时新建 DynamicCache（:1460-1461）→ 默认 position_ids=arange(seqlen)+past
  （:1466-1469）→ ``create_recurrent_attention_mask`` 得 2D padding mask，None 时
  补 ones（:1471-1486）→ ``causal_mask_mapping={"indexed_attention": mask,
  "linear_attention": mask}``（:1488-1491）→ embedding 广播成 ``hc_mult=4`` 路流
  （:1493）→ 逐层调用（:1495-1507）→ ``self.norm(self.hc_head(hidden_states))``
  （:1509）。
- **NoPE**：DSA 层调用 ``self_attn(..., position_embeddings=None)``（:1499-1503），
  源码没有 rotary_emb；position_ids 被计算并传入，但不参与 Attention 数学。
- mHC：网络入口把单流 embedding 广播成 4 路（:1493）；每层两处 HyperConnection
  坍缩/混合（Block 详图）；末端 ``hc_head`` 是 4 路**均值**（:332-336），随后
  Final RMSNorm（:1509）。不是普通单流残差。
- 缓存：KDA 层 ``linear_attention`` → LinearAttentionLayer（conv/recurrent 状态，
  固定大小）；DSA 层 ``indexed_attention`` → DynamicIndexedLayer（压缩 latent +
  indexer packed 缓存）。P=past_key_values.get_seq_length()（:1467）。
- MTP：checkpoint 的 model.layers.45.* 与 shared_head.* 是训练期多 token 预测头，
  本地实现显式忽略（:1378）且 forward 只循环 0..44（:1495-1496）——不在本主干。

三组真实层组与代表层来自 ``rules/glm53flash_hf.py:LAYER_GROUP_SPECS``；层归属由
``layer_group_assignments`` 按 ``layer_types``/``mlp_layer_types`` 逐层推导并校验
三组恰好覆盖全部 45 层。共享模板不代表共享权重：同组层各有独立参数，
``tie_word_embeddings=false`` 时 ``embed_tokens`` 与 ``lm_head`` 也是两份独立权重。

视觉/图像输入路径不在本票范围：模型仓库 ``pipeline_tag=image-text-to-text``，
本模块只解释 text_config 的语言主干。本模块只读源码/配置，不导入
torch/transformers、不实例化或运行模型、不下载权重，不生成 HTML/Excel，不做数值
等价声明。
"""

from __future__ import annotations

from pathlib import Path

from structure_reader.rules.glm53flash_hf import (
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
    """int 型 ``logits_to_keep`` 下 lm_head 实际消费的位置数（modeling :2252-2253）。"""
    if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len < 0:
        raise ValueError(f"seq_len 必须是非负整数: {seq_len!r}")
    if not isinstance(logits_to_keep, int) or isinstance(logits_to_keep, bool):
        raise ValueError(f"logits_to_keep 只分析 int 路径: {logits_to_keep!r}")
    if logits_to_keep == 0:
        return seq_len
    if logits_to_keep > 0:
        return min(seq_len, logits_to_keep)
    raise ValueError(f"logits_to_keep={logits_to_keep}（负整数，slice 正起点）不在本票已分析范围")


def build_network_overview(
    workspace_root: str | Path,
    model_key: str = "glm_5_3_flash_hf",
) -> dict:
    """构建 ``hf_network_overview.v1``（单次网络调用的 token→logits 语言主干）。"""
    workspace_root = Path(workspace_root)
    values = load_hf_model_values(workspace_root, model_key=model_key)
    try:
        groups = layer_group_assignments(values)
    except ValueError as exc:
        raise NetworkDocError(f"三组层归属校验失败: {exc}") from exc
    builder = _NetworkBuilder(values, groups)
    path = builder.build()
    layer_order = _layer_order(values, groups)
    kda_layers = [i for i in range(values.num_hidden_layers)
                  if values.layer_type(i) == "linear_attention"]
    dsa_layers = [i for i in range(values.num_hidden_layers)
                  if values.layer_type(i) == "indexed_attention"]
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
            "entry_class": "Glm5NextForConditionalGeneration",
            "main_module_class": "Glm5NextTextModel",
            "decoder_layer_class": "Glm5NextTextDecoderLayer",
            "analysis_path": "eager",
            "layer_count": values.num_hidden_layers,
            "hidden_size": values.hidden_size,
            "vocab_size": values.vocab_size,
            "hidden_act": values.hidden_act,
            "dense_layers": dense_layers,
            "kda_layers": kda_layers,
            "dsa_layers": dsa_layers,
            "indexer_full_layers": dsa_layers,
            "indexer_shared_layers": [],
            "topk_layers": len(dsa_layers),
            "shared_topk_layers": 0,
            "hc_mult": values.hc_mult,
            "main_path": [op["id"] for op in path["operators"]],
            "notes": [
                "范围：单次网络调用（一次 forward）从 input_ids 到 logits；不含 generate "
                "采样循环、外部控制循环或多次调用耗时。",
                f"表达方式：整网外围与终点按源码逐句静态分析；循环体 "
                f"{values.num_hidden_layers} 层不铺开，三组层型的层内结构由代表层 "
                f"Block/算子详图（同候选的 {DETAIL_PAGE}）提供，本总览给出真实层归属、"
                "顺序与往返入口。",
                "本模型是混合注意力：34 层 KDA 线性注意力（固定 conv/recurrent 状态）"
                "+ 11 层 MLA/DSA 稀疏注意力（压缩 latent + indexer packed 缓存）；"
                "不是 45 层同一种 MLA，也不是 78 层 GLM-5.3。",
                "本模型是 4 路 mHC 残差：网络入口把 embedding 广播成 4 路，层内两处 "
                "post/comb 混合，末端 hc_head 取均值再 Final RMSNorm；不是普通单流残差。",
                "NoPE：qk_rope_head_dim=0，DSA 层调用 position_embeddings=None，源码没有 "
                "RoPE；position_ids 被计算/转发但不参与 Attention 数学。",
                "MTP：checkpoint 的 model.layers.45.* 与 shared_head.* 是训练期多 token "
                "预测头，本地实现显式忽略（modeling:1378）且 forward 只循环 0..44 —— "
                "不在本主干。",
                "tie_word_embeddings=false（根配置）：embed_tokens 与 lm_head 是两份独立权重。",
                "视觉/图像输入路径不在本票范围：模型卡 pipeline_tag=image-text-to-text，"
                "本图只覆盖 text_config 的 token 输入到 logits 语言主干。",
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
    group_of = {index: group["id"] for group in groups for index in group["layer_indices"]}
    order = []
    for index in range(values.num_hidden_layers):
        group_id = group_of.get(index)
        if group_id is None:
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
        return {"path": SOURCE_FILES["modeling_glm5_next.py"],
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
                f"GLM-5.3-Flash 整网主干（{self.values.num_hidden_layers} 层混合 KDA/DSA + "
                "4 路 mHC，token → logits；单次 forward）"
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
                   [src("1466-1469", "torch.arange(inputs_embeds.shape[1]) + past_seen")])
        self.param("kvcachelen", "kvcachelen", "调用前逻辑历史 token 数（kvcachelen）", None,
                   "P = past_key_values.get_seq_length()（无 cache 时 0）",
                   "空缓存 Prefill 取 0；Decode 为调用前已处理历史；不是物理保留条目数"
                   "或字节数。KDA 状态固定大小，DSA 缓存线性增长",
                   [src("1467", "past_seen = past_key_values.get_seq_length()")])
        self.param("H", "H", "hidden_size 隐层维度", v.hidden_size, None,
                   "config.hidden_size=4096", [cfg("text_config.hidden_size", "文本子配置字段")])
        self.param("Hc", "Hc", "hc_mult mHC 残差流数", v.hc_mult, None,
                   "config.hc_mult=4；网络入口广播成 4 路，末端均值收拢",
                   [cfg("text_config.hc_mult", "文本子配置字段"),
                    src("1493", "inputs_embeds.unsqueeze(2).expand(-1,-1,hc_mult,-1)"),
                    src("332-336", "hc_head.mean(dim=2)")])
        self.param("Vocab", "Vocab", "vocab_size 词表大小", v.vocab_size, None,
                   "config.vocab_size=154880", [cfg("text_config.vocab_size", "文本子配置字段")])
        self.param("Nlay", "N_layer", "num_hidden_layers 主干层数（循环次数）",
                   v.num_hidden_layers, None,
                   "config.num_hidden_layers=45（0..44）；layer_order 逐层列真实顺序。"
                   "checkpoint 的 layer 45 是 MTP，被显式忽略（modeling:1378）",
                   [cfg("text_config.num_hidden_layers", "文本子配置字段"),
                    src("1434-1436", "nn.ModuleList([Glm5NextTextDecoderLayer(config, i) ...])"),
                    src("1378", "_keys_to_ignore_on_load_unexpected = [r\"layers\\.45\\.\", ...]")])
        self.param("Lk", "Lk", "logits_to_keep 保留末位 token 数", 0,
                   "logits_to_keep 默认 0",
                   "0 → slice(0, None) 保留全部 seqlen 个位置；k>0 → slice(-k, None) 保留末 "
                   "min(seqlen, k) 个位置（越界不扩大）；负整数与 tensor 索引不在已分析范围",
                   [src("2195", "logits_to_keep: int | torch.Tensor = 0"),
                    src("2252", "slice(-logits_to_keep, None) if isinstance(int)")])
        self.param("Ssel", "S_sel", "logits 序列位置数（切片后）", None,
                   "S_sel = seqlen（Lk=0）或 min(seqlen, Lk）（Lk>0）",
                   "lm_head 只对切片后的 hidden_states 计算，输出 [B,S_sel,Vocab]；"
                   f"例：seqlen=1、Lk=5 时 S_sel={logits_slice_length(1, 5)}（负起点裁剪为 0）",
                   [src("2251-2253", "hidden_states[:, slice_indices, :] → self.lm_head(...)")])
        self.param("Lkv", "L_kv", "本次可读 DSA 缓存长度（latent/packed）", None,
                   "L_kv = kvcachelen + seqlen",
                   "DSA 层的主 KV latent 与 indexer packed 缓存都线性保留全部历史；"
                   "KDA 层没有这个长度（状态固定）",
                   [src("1495-1507", "逐层调用；DSA 层 cache read/write"),
                    self.srcf("cache_utils.py", "165-172", "get_mask_sizes：kv_length=已缓存+query")])
        self.param("Nmtp", "N_mtp", "num_nextn_predict_layers MTP 头数（不在主干）",
                   v.num_nextn_predict_layers, None,
                   "checkpoint 的 layer 45 与 shared_head 是 MTP；本地实现忽略其权重且 "
                   "forward 不执行 —— 本图明确不含 MTP",
                   [cfg("text_config.num_nextn_predict_layers", "文本子配置字段"),
                    src("1378", "_keys_to_ignore_on_load_unexpected"),
                    src("1495", "layers[:num_hidden_layers] 循环")])
        self.param("EpsN", "EpsN", "rms_norm_eps Final RMSNorm epsilon", v.rms_norm_eps, None,
                   "config.rms_norm_eps=1e-5",
                   [cfg("text_config.rms_norm_eps", "文本子配置字段"),
                    src("1509", "hidden_states = self.norm(self.hc_head(hidden_states))")])

    def _build_boundaries(self) -> None:
        self.tensor(
            "input_ids", "input_ids", ["B", "seqlen"], ["B", "seqlen"], "boundary",
            "文本 token id；与 inputs_embeds 恰一（modeling:1457-1458）。"
            "本模型没有 hash 路由，MoE 路由不消费 input_ids")
        self.tensor(
            "attention_mask", "attention_mask（调用方 2D mask，可选）",
            ["B", "kvcachelen+seqlen"], ["kvcachelen", "seqlen", "B"], "boundary",
            "调用方提供的 2D 可见性/padding 掩码；None 时 create_recurrent_attention_mask "
            "返回 None，网络入口再补 ones（:1471-1486）。不是 4D 因果 mask")
        self.tensor(
            "past_key_values", "past_key_values（DynamicCache，调用前）", ["Nlay"], ["Nlay"],
            "cache",
            "use_cache 且未传入时新建 DynamicCache（:1460-1461）；按 layer_types 分发 "
            "LinearAttentionLayer / DynamicIndexedLayer；逐层字段见组详图")
        self.tensor(
            "inputs_embeds", "inputs_embeds", ["B", "seqlen", "H"], ["B", "seqlen", "H"],
            "activation",
            "embed_tokens 输出（:1463-1464）；调用方也可直接传 inputs_embeds 跳过 embedding")
        self.tensor(
            "position_ids", "position_ids（默认 arange+past）", ["1", "seqlen"],
            ["seqlen", "kvcachelen"], "activation",
            "默认 arange(seqlen)+past_seen 后 unsqueeze(0)（:1466-1469）；NoPE 下被转发但"
            "不参与 Attention 数学")
        self.tensor(
            "padding_mask", "attention_mask（2D padding mask）", ["B", "seqlen"],
            ["B", "seqlen"], "mask",
            "create_recurrent_attention_mask 的 [B,seqlen] mask；None 时补 ones"
            "（:1471-1486）；KDA 用它置零 padding，DSA indexer 用它的 valid 通道")
        self.tensor(
            "hidden_streams", "hidden_states（4 路 mHC 流）", ["B", "seqlen", "Hc", "H"],
            ["B", "seqlen", "Hc", "H"], "activation",
            "inputs_embeds.unsqueeze(2).expand(-1,-1,hc_mult,-1).contiguous()"
            "（:1493）：4 路初始流是同一 embedding 的广播，不是独立投影")
        self.tensor(
            "hidden_after_layers", "hidden_states（全部层之后，4 路流）",
            ["B", "seqlen", "Hc", "H"], ["B", "seqlen", "Hc", "H"], "activation",
            "循环最后一次赋值的 hidden_states（:1495-1507）；仍是 4 路流，未做收拢")
        self.tensor(
            "past_key_values_post", "past_key_values（各层 update 后，同一对象）",
            ["Nlay"], ["Nlay"], "cache",
            "KDA 层 update_conv_state/update_recurrent_state、DSA 层 update/update_indexer "
            "原地写入同一 DynamicCache；逐层字段见组详图")
        self.tensor(
            "hidden_mean", "hc_head(hidden_streams)", ["B", "seqlen", "H"],
            ["B", "seqlen", "H"], "activation",
            "4 路流按均值收拢（:332-336/:1509）；不是加权 HyperHead")
        self.tensor(
            "final_hidden", "hidden_states（Final RMSNorm 后）", ["B", "seqlen", "H"],
            ["H", "EpsN"], "activation",
            "model 的 last_hidden_state（:1509-1510）；进入 lm_head 前不再有其它网络级变换")

    def _build_forward_path(self) -> None:
        v = self.values
        src, cfg = self.src, self.cfg

        # 1) embedding
        self.tensor(
            "embed_tokens_weight", "self.embed_tokens.weight", ["Vocab", "H"], ["Vocab", "H"],
            "weight",
            "nn.Embedding(vocab_size, hidden_size, padding_idx)（:1433）；"
            "tie_word_embeddings=false → 与 lm_head.weight 独立")
        self.op(
            "net_embed", "Glm5NextTextModel.embed_tokens", "Token Embedding（查表）",
            "embedding+gather", ["input_ids"], ["inputs_embeds"],
            ["embed_tokens_weight"], ["B", "seqlen", "H", "Vocab"],
            [src("1433", "self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx)"),
             src("1463-1464", "if inputs_embeds is None: inputs_embeds = self.embed_tokens(input_ids)")],
            "token id → [B,seqlen,H] 嵌入；padding_idx=config.pad_token_id")

        # 2) 默认 position_ids（NoPE 下被转发但不参与数学）
        self.op(
            "net_position_ids", "Glm5NextTextModel.forward", "默认位置序列",
            "arange+add+unsqueeze", ["past_key_values"], ["position_ids"], [],
            ["seqlen", "kvcachelen"],
            [src("1466-1469", "past_seen = get_seq_length(); position_ids = "
                              "torch.arange(T) + past_seen; unsqueeze(0)")],
            "NoPE：源码不构造 rotary_emb，position_ids 只被转发；本图保留这一步，"
            "避免把位置编码当成已分析事实")

        # 3) 2D padding mask
        self.op(
            "net_mask_build", "Glm5NextTextModel.forward", "2D padding mask 构建",
            "mask_build", ["inputs_embeds", "attention_mask", "past_key_values"],
            ["padding_mask"], [], ["B", "seqlen"],
            [src("1471-1486", "create_recurrent_attention_mask(...)；None 时 ones"),
             src("2358", "create_masks_for_generate 也返回同样的两个键"),
             self.srcf("masking_utils.py", "1434-1470", "create_recurrent_attention_mask")],
            "构建一次 2D padding mask 并同时传给 indexed_attention/linear_attention；"
            "DSA 的因果性在 indexer 内构造，KDA 的因果性由短卷积+递推顺序保证")

        # 4) mHC 展开
        self.op(
            "net_hc_expand", "Glm5NextTextModel.forward", "4 路 mHC 流展开",
            "unsqueeze+expand+contiguous", ["inputs_embeds"], ["hidden_streams"], [],
            ["B", "seqlen", "Hc", "H"],
            [src("1493", "inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1)"
                        ".contiguous()")],
            "把单流 embedding 广播成 4 路相同流；不是线性投影，也不是把 4 路当独立 token")

        # 5) 层循环
        self.op(
            "net_layer_loop", "Glm5NextTextModel.layers",
            f"{v.num_hidden_layers} 层循环（三组真实层型，顺序见 layer_order）",
            "loop",
            ["hidden_streams", "padding_mask", "position_ids", "past_key_values"],
            ["hidden_after_layers", "past_key_values_post"], [],
            ["Nlay", "B", "seqlen", "Hc"],
            [src("1434-1436", "self.layers = nn.ModuleList([Glm5NextTextDecoderLayer(config, i) "
                                "for i in range(num_hidden_layers)])"),
             src("1495-1507", "for i, decoder_layer in enumerate(self.layers[:num_hidden_layers]): "
                              "hidden_states, topk_indices = decoder_layer(...)"),
             src("1278-1348", "Glm5NextTextDecoderLayer.forward（循环体；KDA 或 MLA/DSA + MoE，"
                              "见组详图）")],
            f"同一 ModuleList 顺序重复 {v.num_hidden_layers} 次，不铺开 45 层：34 层 KDA + "
            "11 层 DSA，组内结构由代表层详图提供；KDA 层写固定 conv/recurrent 状态，"
            "DSA 层写压缩 latent + indexer packed 缓存；DSA 的 indexer 全部是 full，"
            "没有 shared 层间 top-k 传递")

        # 6) hc_head 均值 + Final RMSNorm
        self.op(
            "net_hc_head", "Glm5NextTextHyperHead.forward", "mHC 4 路流均值收拢",
            "mean", ["hidden_after_layers"], ["hidden_mean"], [], ["Hc", "H", "B", "seqlen"],
            [src("332-336", "return hidden_streams.mean(dim=2)"),
             src("1509", "hidden_states = self.norm(self.hc_head(hidden_states))")],
            "4 路流按均值收拢成单流；不是加权 HyperHead，也不是普通残差")
        self.tensor(
            "final_norm_weight", "self.norm.weight", ["H"], ["H"], "weight",
            "Final RMSNorm 可学习缩放（:1437/:64-79）")
        self.op(
            "net_final_norm", "Glm5NextTextModel.norm", "Final RMSNorm", "rmsnorm",
            ["hidden_mean", "final_norm_weight"], ["final_hidden"], [], ["H", "EpsN"],
            [src("1509", "hidden_states = self.norm(self.hc_head(hidden_states))"),
             src("1437", "self.norm = Glm5NextTextRMSNorm(hidden_size, eps=rms_norm_eps)"),
             src("64-79", "fp32 方差归一 → weight 乘 → to(input_dtype)")],
            "均值收拢后做 Final RMSNorm；本模型是 4 路均值，不是加权 HyperHead")

        # 7) logits 切片 + lm_head
        self.tensor(
            "lm_input", "hidden_states[:, slice_indices, :]（切片后）", ["B", "Ssel", "H"],
            ["B", "Ssel", "H"], "activation",
            "只保留需要的序列位置（:2251-2252）；Lk=0 时保留全部 seqlen 位；"
            "Lk>0 时长度为 min(seqlen, Lk)")
        self.op(
            "net_logits_slice", "Glm5NextForConditionalGeneration.forward", "logits 位置切片",
            "slice", ["final_hidden"], ["lm_input"], [], ["Lk", "Ssel"],
            [src("2251-2252", "hidden_states = outputs.last_hidden_state; "
                              "slice_indices = slice(-logits_to_keep, None) if int")],
            "切片发生在 lm_head 之前（只对保留位置做投影）")
        self.tensor(
            "lm_head_weight", "self.lm_head.weight", ["Vocab", "H"], ["Vocab", "H"], "weight",
            "nn.Linear(hidden_size, vocab_size, bias=False)（:2145）；"
            "tie_word_embeddings=false → 与 embed_tokens.weight 不共享")
        self.tensor(
            "logits", "logits", ["B", "Ssel", "Vocab"], ["B", "Ssel", "Vocab"], "boundary",
            "网络终点（:2251-2254）：未归一化分数；Lk>0 时序列轴为 min(seqlen, Lk)；"
            "不在此图做 softmax/采样")
        self.op(
            "net_lm_head", "Glm5NextForConditionalGeneration.lm_head",
            "LM Head 投影（得到 logits）", "linear", ["lm_input"], ["logits"],
            ["lm_head_weight"], ["Vocab", "H", "Ssel"],
            [src("2145", "self.lm_head = nn.Linear(config.text_config.hidden_size, "
                        "config.text_config.vocab_size, bias=False)"),
             src("2251-2253", "logits = self.lm_head(hidden_states[:, slice_indices, :])"),
             src("2272-2280", "MoeCausalLMOutputWithPast(logits=logits, ...)")],
            "网络终点：hidden → vocab 的线性投影；loss/aux_loss/router_logits 属训练或"
            "可选辅助输出，不在本主干")

    def _build_cache_notes(self) -> None:
        self.cache_notes = [
            "past_key_values 在网络入口按需新建 DynamicCache（modeling:1460-1461）；"
            "KDA 层 linear_attention → LinearAttentionLayer（cache_utils:1285-1294），"
            "DSA 层 indexed_attention → DynamicIndexedLayer。",
            "KDA 层缓存 conv state [B,3×NhL×DhL,Kconv]=[B,24576,4] 与 recurrent state "
            "[B,64,128,128]；两者大小固定，不随历史增长（cache_utils:1067-1149）。",
            "DSA 层缓存压缩 latent：keys=[B,1,L,512]、values=[B,1,L,0]（NoPE）；"
            "另有 indexer packed 缓存 [B,L,257]（k+gate+valid）（modeling:1194-1198/"
            "841-849；cache_utils:331-380）。",
            "P = past_key_values.get_seq_length()（:1467）：决定默认 position_ids 的起点与"
            "DSA 缓存长度；空缓存 Prefill 时 kvcachelen=0。",
            "本模型两类层的时间/空间规模不同：KDA 固定 O(1) 状态，DSA 随历史线性增长；"
            "不能用一个 P+S 或滑动窗口公式套全部层。",
        ]

    def _build_coverage_notes(self) -> None:
        v = self.values
        kda = [i for i in range(v.num_hidden_layers) if v.layer_type(i) == "linear_attention"]
        dsa = [i for i in range(v.num_hidden_layers) if v.layer_type(i) == "indexed_attention"]
        self.coverage_notes = [
            "已表达（与 modeling 行号逐条对应）：ForConditionalGeneration.lm_head → "
            "TextModel.forward → embed_tokens、默认 position_ids、2D padding mask、"
            "4 路 mHC 展开、45 层循环（真实层序与三组归属）、hc_head 均值、Final RMSNorm、"
            "logits 位置切片、lm_head → logits、past_key_values 读取与写回。",
            "层内结构不在此图重复铺开：KDA（q/k/v、短卷积、forget gate、delta-rule 扫描、"
            "gated norm）与 MLA/DSA（q 低秩、压缩 latent 缓存、kv_b 展开、k-pool indexer、"
            "稀疏 mask）以及稠密 MLP / TopK MoE + 共享专家由同候选分层浏览页的层组/Block/"
            "算子层提供（structure.html）。",
            f"KDA 层 {len(kda)} 个（{kda[:6]}…{kda[-2:]}）；DSA 层 {len(dsa)} 个（{dsa}）；"
            "dense MLP 层 0/1/2。DSA 的 indexer_types 全为 full，没有 shared 层。",
            "同构模板复用不代表共享权重：同组每层是独立 Glm5NextTextDecoderLayer 实例；"
            "本配置 tie_word_embeddings=false，embed_tokens 与 lm_head 权重也独立。",
            "MTP：model.layers.45.* 与 shared_head.* 是训练期多 token 预测头，本地实现"
            "显式忽略（modeling:1378）且 forward 只循环 0..44 —— 不在本主干，未分析。",
            "NoPE：源码没有 rotary_emb；position_ids 被计算/转发但不参与 Attention 数学。"
            "DSA 的因果性由 indexer 的 kv_pos ≤ q_pos + 池/tail 选择保证，没有网络级 4D "
            "因果 mask；KDA 的因果性由短卷积+递推顺序保证。",
            "未覆盖（明确列出，不宣称已分析）：generate/采样循环与多次调用累计、"
            "loss/aux_loss 训练路径、output_router_logits 辅助输出、量化执行"
            "（FP8 e4m3 与设备 kernel）、视觉 24 层、torch.compile/梯度检查点、"
            "数值等价与权重内容。",
            "数据相关量（路由命中专家数、每专家 token 数、具体选中的池/token 等）保留符号；"
            "不填运行时假设值，不根据权重推断 MoE 实际路由。",
        ]
