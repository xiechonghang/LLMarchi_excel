"""Qwen3.8-Flash-Next 整网总览数据提取（token → logits，静态只读，不实例化/不运行模型）。

从 ``rules/qwen38flash_hf.py`` 解释的文本有效配置与本地官方 Transformers 检出源码，
构建 ``hf_network_overview.v1``：一次网络调用的外围（文本 embedding / 4 行
position_ids + mRoPE 重组 / causal + recurrent 两套 mask / PLE token id /
4 路 Gated Residual 展开）→ 48 层真实 ModuleList 循环（三组层型归属见
groups/layer_order）→ ``hyper_connection_mixer`` 加权 mean 收拢 → logits 位置切片
→ lm_head → logits，以及到 Block/算子详图的往返目标。

表达事实（行号按本地检出 536ecc00…，文件 modeling_qwen4_exp.py，下称 modeling）：

- 语言主干入口是 ``Qwen4ExpTextModel``（``model.language_model``，modeling:2017）；
  ``Qwen4ExpForConditionalGeneration`` 在 ``self.model`` 之后用独立 ``lm_head``
  得到 logits（:2401-2404/:2514-2518）。本图只走 text-only 路径：视觉
  ``self.visual``、图像/视频 placeholder masked_scatter 与视觉网格 M-RoPE 不在范围。
- ``Qwen4ExpTextModel.forward``（:1391-1489）：embed（:1410-1411）→ PLE token id
  （:1413-1416）→ use_cache 且无 cache 时新建 DynamicCache（:1418-1419）→ 默认
  position_ids=arange(seqlen)+past 扩展成 4 行（:1421-1436）→ 两套 mask
  （:1446-1460）→ PLE 的 padding 位置替换成 eos（:1462-1467）→ hidden 复制 4 路
  （:1471）→ 逐层调用（:1473-1482）→ ``hyper_connection_mixer``（:1484）。
- **没有 Final RMSNorm**：``Qwen4ExpTextModel`` 只有 embed_tokens / layers /
  rotary_emb / hyper_connection_mixer；末端直接返回 mixer 的单流 hidden，随后
  lm_head。这是与 DeepSeek/GLM/GLM-Flash 的实质差异，不能补一个不存在的 norm。
- Gated Residual mixer：与层内读门同构（分组 RMSNorm → 10240→320 → SiLU(·/4) →
  320→10240 → sigmoid → 4 路加权 mean），不是普通均值/求和（:994-1022）。
- position_ids：默认 ``arange(seqlen)+past_seen``，view(1,1,-1).expand(4,B,-1)；
  row0 给 text_position_ids/causal mask，rows1..3 给 mRoPE。text-only 时三行相同，
  ``recomposition_frequencies`` 的 T/H/W 段取到相同值，等价于同一组 1D RoPE 的
  前 64 维；本图保留 4 行/重组步骤并如实注明，不把视觉网格混入。
- mask：``create_causal_mask``（4D additive，QSA 用）+ ``create_recurrent_attention_mask``
  （2D padding，GDN 用），一次构建后按 layer_types 分派（:1446-1460）。
- 缓存：GDN 层 linear_attention → LinearAttentionLayer（conv/recurrent 固定状态）；
  QSA 层 indexed_attention → DynamicIndexedLayer（K/V [B,2,L,256] + indexer keys
  [B,L,128]）。P=past_key_values.get_seq_length()（:1423）：混合 cache 取第一个
  attention 层（layer 3）的长度。
- MTP：``mtp_num_hidden_layers=1`` 只是配置声明；本地 HF 实现没有 MTP 模块，
  ``_keys_to_ignore_on_load_unexpected=[r"^mtp.*"]``（modeling:1313/1589），
  forward 只循环 0..47 —— 不在本主干。

三组真实层组与代表层来自 ``rules/qwen38flash_hf.py:LAYER_GROUP_SPECS``；层归属由
``layer_group_assignments`` 按 ``layer_types``/PLE 声明逐层推导并校验三组恰好覆盖
全部 48 层。共享模板不代表共享权重：同组层各有独立参数，
``tie_word_embeddings=false`` 时 ``embed_tokens`` 与 ``lm_head`` 也是两份独立权重。

视觉/图像输入路径不在本票范围：模型仓库 ``pipeline_tag=image-text-to-text``，
本模块只解释 text_config 的语言主干。本模块只读源码/配置，不导入
torch/transformers、不实例化或运行模型、不下载权重，不生成 HTML/Excel，不做数值
等价声明。
"""

from __future__ import annotations

from pathlib import Path

from structure_reader.rules.qwen38flash_hf import (
    MODEL_REPOSITORY,
    SOURCE_FILES,
    HfModelValues,
    gdn_layer_indices,
    layer_group_assignments,
    load_hf_model_values,
    ple_layer_indices,
    qsa_layer_indices,
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
    """int 型 ``logits_to_keep`` 下 lm_head 实际消费的位置数（modeling:2517-2518）。"""
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
    model_key: str = "qwen_3_8_flash_next_hf",
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
    gdn_layers = gdn_layer_indices(values)
    qsa_layers = qsa_layer_indices(values)
    ple_layers = ple_layer_indices(values)
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
            "entry_class": "Qwen4ExpForConditionalGeneration",
            "main_module_class": "Qwen4ExpTextModel",
            "decoder_layer_class": "Qwen4ExpTextDecoderLayer",
            "analysis_path": "eager",
            "layer_count": values.num_hidden_layers,
            "hidden_size": values.hidden_size,
            "vocab_size": values.vocab_size,
            "hidden_act": values.hidden_act,
            "gdn_layers": gdn_layers,
            "qsa_layers": qsa_layers,
            "ple_layers": ple_layers,
            "topk_layers": list(range(values.num_hidden_layers)),
            "hc_count": values.hc_count,
            "hc_lowrank": values.hc_lowrank,
            "main_path": [op["id"] for op in path["operators"]],
            "notes": [
                "范围：单次网络调用（一次 forward）从 input_ids 到 logits；不含 generate "
                "采样循环、外部控制循环或多次调用耗时。",
                f"表达方式：整网外围与终点按源码逐句静态分析；循环体 "
                f"{values.num_hidden_layers} 层不铺开，三组层型的层内结构由代表层 "
                f"Block/算子详图（同候选的 {DETAIL_PAGE}）提供，本总览给出真实层归属、"
                "顺序与往返入口。",
                "本模型是 12×(3×Gated DeltaNet + 1×QSA)：36 层线性注意力（固定 "
                "conv/recurrent 状态）+ 12 层 QSA 稀疏注意力（K/V + indexer key 随历史"
                "线性增长）；不是 48 层同一种 GQA，也不是 Qwen3-Next 的混合结构。",
                "本模型是 4 路 Gated Residual（hyper-connections）：网络入口 repeat 成 "
                "4 路，层内两站点读门/写门混合，末端 hyper_connection_mixer 用同一读门"
                "加权 mean 收拢；不是普通单流残差，也不是 mHC/Sinkhorn。",
                "没有 Final RMSNorm：TextModel 末端只有 hyper_connection_mixer，"
                "随后直接 lm_head；不虚构一个不存在的 norm。",
                "position_ids 是 4 行（row0 text + rows1..3 mRoPE），text-only 时三行"
                "相同，mRoPE 重组退化为同一组 1D RoPE 的 64 维；视觉网格 M-RoPE 不在"
                "本票范围。",
                "MTP：配置声明 mtp_num_hidden_layers=1，但本地 HF 实现没有 MTP 模块，"
                "加载时忽略 mtp.*，forward 只循环 0..47 —— 不在本主干。",
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
                    "ple": group["ple"],
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
    """真实层顺序：ModuleList 索引 0..N-1，逐层标注层型/PLE/所属组。"""
    group_of = {index: group["id"] for group in groups for index in group["layer_indices"]}
    order = []
    for index in range(values.num_hidden_layers):
        group_id = group_of.get(index)
        if group_id is None:
            raise NetworkDocError(f"层 {index} 未归属任何真实层组（覆盖校验与实际不一致）")
        order.append({
            "index": index,
            "attention_type": values.layer_type(index),
            "ple": values.is_ple_layer(index),
            "moe_type": "sparse",
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

    def src(self, locator, note) -> dict:
        return {"path": SOURCE_FILES["modeling_qwen4_exp.py"],
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
                f"Qwen3.8-Flash-Next 整网主干（{self.values.num_hidden_layers} 层 "
                "Gated DeltaNet/QSA + 4 路 Gated Residual，token → logits；单次 forward）"
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
                   [src("1421-1425", "torch.arange(inputs_embeds.shape[1]) + past_seen_tokens")])
        self.param("kvcachelen", "kvcachelen", "调用前逻辑历史 token 数（kvcachelen）", None,
                   "P = past_key_values.get_seq_length()（无 cache 时 0）",
                   "空缓存 Prefill 取 0；Decode 为调用前已处理历史；不是物理保留条目数"
                   "或字节数。GDN 状态固定大小，QSA 缓存线性增长",
                   [src("1423", "past_seen_tokens = past_key_values.get_seq_length()")])
        self.param("H", "H", "hidden_size 隐层维度", v.hidden_size, None,
                   "config.hidden_size=2560", [cfg("text_config.hidden_size", "文本子配置字段")])
        self.param("Hc", "Hc", "hc_count Gated Residual 流数", v.hc_count, None,
                   "config.hc_count=4；网络入口 repeat 成 4 路，末端 mixer 加权 mean",
                   [cfg("text_config.hc_count", "文本子配置字段"),
                    src("1471", "hidden_states.repeat(1, 1, hc_count)"),
                    src("1484", "hidden_states = self.hyper_connection_mixer(hidden_states)")])
        self.param("Rhc", "R_hc", "hc_lowrank 读门低秩维", v.hc_lowrank, None,
                   "config.hc_lowrank=320；末端 mixer 也是 10240→320→10240",
                   [cfg("text_config.hc_lowrank", "文本子配置字段"),
                    src("1384", "self.hyper_connection_mixer = Qwen4ExpTextGatedResidual(config, use_combine=False)")])
        self.param("Vocab", "Vocab", "vocab_size 词表大小", v.vocab_size, None,
                   "config.vocab_size=248320", [cfg("text_config.vocab_size", "文本子配置字段")])
        self.param("Nlay", "N_layer", "num_hidden_layers 主干层数（循环次数）",
                   v.num_hidden_layers, None,
                   "config.num_hidden_layers=48（0..47）；layer_order 逐层列真实顺序。"
                   "配置声明 MTP 1 层，但本地实现没有 MTP 模块，forward 只循环 0..47",
                   [cfg("text_config.num_hidden_layers", "文本子配置字段"),
                    src("1380-1382", "nn.ModuleList([Qwen4ExpTextDecoderLayer(config, i) ...])"),
                    src("1313", "_keys_to_ignore_on_load_unexpected = [r\"^mtp.*\"]")])
        self.param("Lk", "Lk", "logits_to_keep 保留末位 token 数", 0,
                   "logits_to_keep 默认 0",
                   "0 → slice(0, None) 保留全部 seqlen 个位置；k>0 → slice(-k, None) 保留末 "
                   "min(seqlen, k) 个位置（越界不扩大）；负整数与 tensor 索引不在已分析范围",
                   [src("2516", "logits_to_keep: int | torch.Tensor = 0"),
                    src("2517", "slice(-logits_to_keep, None) if isinstance(int)")])
        self.param("Ssel", "S_sel", "logits 序列位置数（切片后）", None,
                   "S_sel = seqlen（Lk=0）或 min(seqlen, Lk）（Lk>0）",
                   "lm_head 只对切片后的 hidden 计算，输出 [B,S_sel,Vocab]",
                   [src("2517-2518", "hidden_states[:, slice_indices, :] → self.lm_head(...)")])
        self.param("Lkv", "L_kv", "本次可读 QSA K/V 缓存长度", None,
                   "L_kv = kvcachelen + seqlen",
                   "QSA 层的 K/V 与 indexer key 缓存都线性保留全部历史；"
                   "GDN 层没有这个长度（状态固定）",
                   [src("870-871", "QSA 层 past_key_values.update(...)"),
                    self.srcf("cache_utils.py", "132-153", "DynamicLayer.update：concat")])
        self.param("Lidx", "L_idx", "本次可读 QSA indexer key 缓存长度", None,
                   "L_idx = kvcachelen + seqlen",
                   "indexer key [B,L,128] 与主 K/V 同式；12 个 QSA 层各自实例化 indexer",
                   [src("707-708", "raw_keys = past_key_values.update_indexer(raw_keys, layer_idx)")])
        self.param("Drot", "D_rot", "RoPE 旋转维（partial）", v.rotary_dim,
                   "D_rot = head_dim * partial_rotary_factor",
                   "256×0.25=64；QSA 与 indexer 都只转前 64 维",
                   [cfg("text_config.rope_parameters.partial_rotary_factor", "文本子配置字段"),
                    src("107-115", "dim = int(head_dim * partial_rotary_factor)")])
        self.param("Npos", "N_pos", "max_position_embeddings", v.max_position_embeddings,
                   None, "config.max_position_embeddings=262144",
                   [cfg("text_config.max_position_embeddings", "文本子配置字段")])
        self.param("Nmtp", "N_mtp", "MTP 声明层数（不在主干）", v.mtp_num_hidden_layers,
                   None,
                   "配置 mtp_num_hidden_layers=1、mtp.layer_types=['full_attention']；"
                   "本地 HF 实现没有 MTP 模块，加载时忽略 mtp.* —— 本图明确不含 MTP",
                   [cfg("text_config.mtp", "文本子配置字段"),
                    src("1313", "_keys_to_ignore_on_load_unexpected"),
                    src("1473", "layers[:num_hidden_layers] 循环")])
        self.param("EpsN", "EpsN", "RMSNorm epsilon（含 mixer 分组 norm）", v.rms_norm_eps,
                   None, "config.rms_norm_eps=1e-6",
                   [cfg("text_config.rms_norm_eps", "文本子配置字段"),
                    src("994-1022", "Gated Residual hc_norm 也用它")])
        self.param("Eos", "Eos", "PLE n-gram 上下文切断 token", v.eos_token_id, None,
                   "config.eos_token_id 首项；PLE 的 padding 位置也替换成它",
                   [cfg("text_config.eos_token_id", "文本子配置字段"),
                    src("1464-1467", "ple_input_ids = torch.where(conv_mask.bool(), "
                                    "ple_input_ids, eos_token_id)")])

    def _build_boundaries(self) -> None:
        v = self.values
        self.tensor(
            "input_ids", "input_ids", ["B", "seqlen"], ["B", "seqlen"], "boundary",
            "文本 token id；与 inputs_embeds 恰一（modeling:1407-1408）。"
            "PLE 层还直接消费该 id")
        self.tensor(
            "attention_mask", "attention_mask（调用方 2D mask，可选）",
            ["B", "kvcachelen+seqlen"], ["B", "kvcachelen", "seqlen"], "boundary",
            "调用方提供的 2D 可见性/padding 掩码；None 时 create_causal_mask / "
            "create_recurrent_attention_mask 按全 1 构建（:1446-1466）。不是 4D mask 本身")
        self.tensor(
            "past_key_values", "past_key_values（DynamicCache，调用前）", ["Nlay"], ["Nlay"],
            "cache",
            "use_cache 且未传入时新建 DynamicCache（:1418-1419）；按 layer_types 分发 "
            "LinearAttentionLayer / DynamicIndexedLayer；逐层字段见组详图")
        self.tensor(
            "inputs_embeds", "inputs_embeds", ["B", "seqlen", "H"], ["B", "seqlen", "H"],
            "activation",
            "embed_tokens 输出（:1410-1411）；调用方也可直接传 inputs_embeds 跳过 embedding")
        self.tensor(
            "position_ids", "position_ids（默认 4 行：text + mRoPE grid）",
            ["4", "B", "seqlen"], ["seqlen", "kvcachelen"], "activation",
            "默认 arange(seqlen)+past_seen 后 view(1,1,-1).expand(4,B,-1)（:1421-1425）；"
            "row0 给 text_position_ids/causal mask，rows1..3 给 mRoPE。text-only 时三行相同")
        self.tensor(
            "causal_mask", "causal_mask（4D additive，QSA 用）",
            ["B", "1", "seqlen", "Lkv"], ["B", "seqlen", "Lkv"], "mask",
            "create_causal_mask(allow_is_causal_skip=False)：0=可见、min=屏蔽；"
            "一次构建后传给所有 indexed_attention 层（:1446-1460）")
        self.tensor(
            "conv_mask", "conv_mask（2D padding，GDN 用）", ["B", "seqlen"],
            ["B", "seqlen"], "mask",
            "create_recurrent_attention_mask 的 [B,seqlen] mask；None 时源码补 ones；"
            "只置零 padding，不提供因果性（:1458-1462）")
        self.tensor(
            "hidden_streams", "hidden_states（4 路 Gated Residual 流，flat）",
            ["B", "seqlen", "Hc×H"], ["B", "seqlen", "Hc", "H"], "activation",
            "inputs_embeds.repeat(1,1,hc_count) 得 [B,S,Hc*H]；层间实际边界是 flat，"
            "读门内部才 unflatten（:1015-1018）。repeat 是沿最后一维复制 4 份，不交错")
        self.tensor(
            "hidden_after_layers", "hidden_states（全部层之后，4 路流 flat）",
            ["B", "seqlen", "Hc×H"], ["B", "seqlen", "Hc", "H"], "activation",
            "循环最后一次赋值的 hidden_states（:1473-1482）；仍是 flat 4 路流，未做收拢")
        self.tensor(
            "past_key_values_post", "past_key_values（各层 update 后，同一对象）",
            ["Nlay"], ["Nlay"], "cache",
            "GDN 层 update_conv_state/update_recurrent_state、QSA 层 update/update_indexer "
            "原地写入同一 DynamicCache；逐层字段见组详图")

    def _build_forward_path(self) -> None:
        v = self.values
        src, cfg = self.src, self.cfg

        # 1) embedding
        self.tensor(
            "embed_tokens_weight", "self.embed_tokens.weight", ["Vocab", "H"], ["Vocab", "H"],
            "weight",
            "nn.Embedding(vocab_size, hidden_size, padding_idx)（:1379）；"
            "tie_word_embeddings=false → 与 lm_head.weight 独立")
        self.op(
            "net_embed", "Qwen4ExpTextModel.embed_tokens", "Token Embedding（查表）",
            "embedding+gather", ["input_ids"], ["inputs_embeds"],
            ["embed_tokens_weight"], ["B", "seqlen", "H", "Vocab"],
            [src("1379", "self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx)"),
             src("1410-1411", "if inputs_embeds is None: inputs_embeds = self.embed_tokens(input_ids)")],
            "token id → [B,seqlen,H] 嵌入；padding_idx=config.pad_token_id（本配置 null）")

        # 3) 默认 position_ids（4 行）
        self.op(
            "net_position_ids", "Qwen4ExpTextModel.forward", "默认位置序列（4 行）",
            "arange+add+expand", ["past_key_values"], ["position_ids"], [],
            ["seqlen", "kvcachelen"],
            [src("1421-1425", "position_ids = arange(T) + past_seen; "
                              "view(1,1,-1).expand(4, B, -1)"),
             src("1429-1436", "text_position_ids=row0; position_ids=rows1..3")],
            "text-only 时 4 行相同；row0 给 causal mask，rows1..3 给 mRoPE 重组；"
            "视觉网格位置不在本票范围")

        # 4) 两套 mask
        self.op(
            "net_mask_build", "Qwen4ExpTextModel.forward", "causal + recurrent mask 构建",
            "mask_build", ["inputs_embeds", "attention_mask", "past_key_values"],
            ["causal_mask", "conv_mask"], [], ["B", "seqlen", "Lkv"],
            [src("1446-1460", "causal_mask_mapping = {'indexed_attention': "
                              "create_causal_mask(...), 'linear_attention': "
                              "create_recurrent_attention_mask(...)}"),
             self.srcf("masking_utils.py", "850-...", "create_causal_mask：4D additive"),
             self.srcf("masking_utils.py", "1434-1470", "create_recurrent_attention_mask：2D padding")],
            "两套 mask 一次构建；QSA 用 4D causal mask 再叠加 indexer selected mask，"
            "GDN 用 2D padding mask（因果性由卷积+递推保证）")

        # 2) PLE token id
        self.tensor(
            "ple_input_ids", "ple_input_ids（PLE 用 token id）", ["B", "seqlen"],
            ["B", "seqlen"], "activation",
            "上层传入 input_ids；PLE 开启时即使 inputs_embeds 路径也要还原 id"
            "（:1413-1416）。conv_mask=false 的位置替换为 eos（:1464-1467）")
        self.op(
            "net_ple_input_ids", "Qwen4ExpModel.forward / Qwen4ExpTextModel.forward",
            "PLE token id 选择与 padding 替换", "where",
            ["input_ids", "conv_mask"], ["ple_input_ids"], [], ["B", "seqlen", "Eos"],
            [src("1413-1416", "if self.config.ple_layer_ids and ple_input_ids is None: "
                              "ple_input_ids = input_ids ..."),
             src("1464-1467", "ple_input_ids = torch.where(conv_mask.bool(), ple_input_ids, eos_token_id)")],
            "PLE 需要原始 token id 做 n-gram；padding 位置用 eos 切断上下文")

        # 5) 4 路展开
        self.op(
            "net_hidden_repeat", "Qwen4ExpTextModel.forward", "4 路 Gated Residual 展开",
            "repeat", ["inputs_embeds"], ["hidden_streams"], [],
            ["B", "seqlen", "Hc", "H"],
            [src("1471", "hidden_states = hidden_states.repeat(1, 1, self.config.hc_count)")],
            "沿最后一维复制 4 份得 [B,S,Hc*H]（等价于 4 路相同流）；不是线性投影，"
            "也不是 4 个独立 token；层间保持 flat 边界")

        # 6) 层循环
        self.op(
            "net_layer_loop", "Qwen4ExpTextModel.layers",
            f"{v.num_hidden_layers} 层循环（三组真实层型，顺序见 layer_order）",
            "loop",
            ["hidden_streams", "causal_mask", "conv_mask", "position_ids",
             "ple_input_ids", "past_key_values"],
            ["hidden_after_layers", "past_key_values_post"], [],
            ["Nlay", "B", "seqlen", "Hc"],
            [src("1380-1382", "self.layers = nn.ModuleList([Qwen4ExpTextDecoderLayer(config, i) "
                              "for i in range(num_hidden_layers)])"),
             src("1473-1482", "for layer_idx, decoder_layer in enumerate(self.layers[:num_hidden_layers]): "
                              "hidden_states = decoder_layer(...)"),
             src("1249-1301", "Qwen4ExpTextDecoderLayer.forward（循环体；GDN/QSA + MoE，见组详图）")],
            f"同一 ModuleList 顺序重复 {v.num_hidden_layers} 次，不铺开 48 层：36 层 GDN + "
            "12 层 QSA（含 layer 1 的 PLE）；组内结构由代表层详图提供；GDN 层写固定 "
            "conv/recurrent 状态，QSA 层写 K/V + indexer key 缓存；PLE 层另有 2 个固定状态")

        # 7) 末端 mixer（Gated Residual，use_combine=False）
        self.weight("net_mixer_norm_weight", "self.hyper_connection_mixer.hc_norm.weight",
                    ["Hc×H"], ["Hc", "H"],
                    "mixer 分组 RMSNorm 的 weight（每 2560 维流独立归一）")
        self.weight("net_mixer_down_weight",
                    "self.hyper_connection_mixer.input_mix_weight_down.weight",
                    ["Rhc", "Hc×H"], ["Rhc", "Hc", "H"], "mixer 读门低秩 10240→320")
        self.weight("net_mixer_up_weight",
                    "self.hyper_connection_mixer.input_mix_weight_up.weight",
                    ["Hc×H", "Rhc"], ["Hc", "H", "Rhc"], "mixer 读门上投影 320→10240")
        self.tensor("hidden_mixed", "mixed_input = (read*normed).mean(-2)",
                    ["B", "seqlen", "H"], ["B", "seqlen", "H"], "activation",
                    "4 路流经分组 RMSNorm、低秩读门、sigmoid 加权 mean 收拢成单流；"
                    "这是 TextModel 的最后一步，**没有 Final RMSNorm**")
        self.op("net_mixer_mix", "Qwen4ExpTextGatedResidual.forward",
                "末端 Gated Residual 加权 mean 收拢", "rmsnorm+lowrank+sigmoid+mean",
                ["hidden_after_layers"], ["hidden_mixed"],
                ["net_mixer_norm_weight", "net_mixer_down_weight", "net_mixer_up_weight"],
                ["Hc", "H", "Rhc", "B", "seqlen"],
                [src("1012", "hyper_input_normed = self.hc_norm(hyper_input)"),
                 src("1013", "F.silu(self.input_mix_weight_down(hyper_input_normed) / self.hc_count)"),
                 src("1014-1015", "sigmoid(self.input_mix_weight_up(...)); unflatten(-1,(Hc,H))"),
                 src("1016-1018", "(input_mix_weight * normed.unflatten(...)).mean(dim=-2)"),
                 src("1484", "hidden_states = self.hyper_connection_mixer(hidden_states)")],
                "读门是逐元素 [B,S,4,H]、数据相关；四路加权后取 mean。"
                "不是普通 average pooling，也不是 Final RMSNorm；收拢后直接进 lm_head")
        self.tensor(
            "lm_input", "hidden_states[:, slice_indices, :]（切片后）", ["B", "Ssel", "H"],
            ["B", "Ssel", "H"], "activation",
            "只保留需要的序列位置（:2517）；Lk=0 时保留全部 seqlen 位；"
            "Lk>0 时长度为 min(seqlen, Lk)")
        self.op(
            "net_logits_slice", "Qwen4ExpForConditionalGeneration.forward", "logits 位置切片",
            "slice", ["hidden_mixed"], ["lm_input"], [], ["Lk", "Ssel"],
            [src("2516-2517", "hidden_states = outputs[0]; "
                              "slice_indices = slice(-logits_to_keep, None) if int")],
            "切片发生在 lm_head 之前；本模型没有 final norm，切片前不再有网络级变换")
        self.tensor(
            "lm_head_weight", "self.lm_head.weight", ["Vocab", "H"], ["Vocab", "H"], "weight",
            "nn.Linear(hidden_size, vocab_size, bias=False)（:2404）；"
            "tie_word_embeddings=false → 与 embed_tokens.weight 不共享")
        self.tensor(
            "logits", "logits", ["B", "Ssel", "Vocab"], ["B", "Ssel", "Vocab"], "boundary",
            "网络终点（:2517-2518）：未归一化分数；Lk>0 时序列轴为 min(seqlen, Lk)；"
            "不在此图做 softmax/采样")
        self.op(
            "net_lm_head", "Qwen4ExpForConditionalGeneration.lm_head",
            "LM Head 投影（得到 logits）", "linear", ["lm_input"], ["logits"],
            ["lm_head_weight"], ["Vocab", "H", "Ssel"],
            [src("2404", "self.lm_head = nn.Linear(config.text_config.hidden_size, "
                        "config.text_config.vocab_size, bias=False)"),
             src("2517-2518", "logits = self.lm_head(hidden_states[:, slice_indices, :])")],
            "网络终点：hidden → vocab 的线性投影；loss/aux_loss/router_logits 属训练或"
            "可选辅助输出，不在本主干")

    def _build_cache_notes(self) -> None:
        self.cache_notes = [
            "past_key_values 在网络入口按需新建 DynamicCache（modeling:1418-1419）；"
            "GDN 层 linear_attention → LinearAttentionLayer（cache_utils:1285-1294），"
            "QSA 层 indexed_attention → DynamicIndexedLayer。",
            "GDN 层缓存 conv state [B,10240,4] 与 recurrent state [B,48,128,128]；"
            "两者大小固定，不随历史增长（cache_utils:1067-1149）。",
            "QSA 层缓存 K/V [B,2,L,256] 与 indexer key [B,L,128]；随历史线性增长"
            "（modeling:707-708/870-871；cache_utils:331-407）。",
            "PLE 层（layer 1）额外使用短卷积 state [B,10240,9]（state_idx=1）与 n-gram "
            "上下文 [B,2]（state_idx=2）；number_of_conv_states=3，但只有 layer 1 使用 "
            "state_idx=1/2（configuration:169-171；modeling:1126-1144/1211-1214）。",
            "P = past_key_values.get_seq_length()（:1423）：决定默认 position_ids 的起点与"
            "QSA 缓存长度；混合 cache 取第一个 attention 层（layer 3）的 keys 长度；"
            "空缓存 Prefill 时 kvcachelen=0。",
            "本模型两类层的时间/空间规模不同：GDN 固定 O(1) 状态，QSA 随历史线性增长；"
            "不能用一个 P+S 或滑动窗口公式套全部层。",
        ]

    def _build_coverage_notes(self) -> None:
        v = self.values
        gdn = gdn_layer_indices(v)
        qsa = qsa_layer_indices(v)
        ple = ple_layer_indices(v)
        self.coverage_notes = [
            "已表达（与 modeling 行号逐条对应）：ForConditionalGeneration.lm_head → "
            "Qwen4ExpModel.text-only 路径 → Qwen4ExpTextModel.forward → embed_tokens、"
            "PLE token id、默认 4 行 position_ids、causal + recurrent 两套 mask、"
            "4 路 Gated Residual 展开、48 层循环（真实层序与三组归属）、"
            "hyper_connection_mixer 读门加权 mean、logits 位置切片、lm_head → logits、"
            "past_key_values 读取与写回。",
            "层内结构不在此图重复铺开：GDN（q/k/v 合并投影、短卷积、beta/decay、"
            "delta-rule 扫描、Gated RMSNorm）、QSA（q/gate、partial RoPE、2 头 KV、"
            "micro-block indexer、selected mask、eager attention）以及 N-gram PLE 与 "
            "TopK MoE + 共享专家由同候选分层浏览页的层组/Block/算子层提供（structure.html）。",
            f"GDN 层 {len(gdn)} 个（{gdn[:6]}…{gdn[-2:]}）；QSA 层 {len(qsa)} 个"
            f"（{qsa}）；PLE 层 {ple}（0-based，第 2 层）；全部 48 层都是 TopK MoE，"
            "没有 dense MLP 层。",
            "同构模板复用不代表共享权重：同组每层是独立 Qwen4ExpTextDecoderLayer 实例；"
            "本配置 tie_word_embeddings=false，embed_tokens 与 lm_head 权重也独立。",
            "MTP：配置声明 mtp_num_hidden_layers=1、full_attention，但本地 HF 实现没有 "
            "MTP 模块，加载时忽略 mtp.*，forward 只循环 0..47 —— 不在本主干，未分析。",
            "position_ids：text-only 时没有 mm_token_type_ids/image_grid_thw，"
            "compute_3d_position_ids 返回 None，TextModel 用 arange+past 默认值并扩成 "
            "4 行；mRoPE 重组仍执行，但三行相同，等价于 1D RoPE 的前 64 维。"
            "视觉网格 M-RoPE 不在本票范围。",
            "未覆盖（明确列出，不宣称已分析）：generate/采样循环与多次调用累计、"
            "loss/aux_loss 训练路径、output_router_logits 辅助输出、设备 kernel/dtype "
            "数值、视觉 27 层、torch.compile/梯度检查点、数值等价与权重内容。",
            "数据相关量（路由命中专家数、每专家 token 数、具体选中的块/token 等）保留符号；"
            "不填运行时假设值，不根据权重推断 MoE 实际路由。",
        ]
