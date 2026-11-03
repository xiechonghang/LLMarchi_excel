"""DeepSeek-V4-Flash 整网总览数据提取（token → logits，静态只读）。

从 rules/deepseek_v4_hf.py 解释的有效配置与本地官方 Transformers 检出源码，
构建 ``hf_network_overview.v1``：一次网络调用的外围（embedding / 位置 / mask /
mHC 流扩展）、真实层循环（43 层按 ModuleList 顺序，四类层组归属见 groups/
layer_order）、循环后 HyperHead 收拢、Final RMSNorm、lm_head 前切片与 logits，
以及到既有 Block/算子详图的往返目标。

表达事实（modeling_deepseek_v4.py 行号按本地检出 536ecc00…，下称 modeling）：
- ``DeepseekV4ForCausalLM.forward``（:1461-1536）：调用 ``self.model(...)``（:1497-1506）
  取 ``outputs.last_hidden_state``（:1508），按 ``logits_to_keep`` 取
  ``slice(-k, None)``（默认 0 → slice(0, None) 全位置，:1510），再
  ``self.lm_head(hidden_states[:, slice_indices, :])``（:1511）；``lm_head`` 为
  ``nn.Linear(hidden_size, vocab_size, bias=False)``（:1451）。
- ``DeepseekV4Model.forward``（:1318-1369）：``embed_tokens``（:1302/:1333）、
  ``position_ids = arange(seqlen) + past_seen``（:1334-1337）、
  ``create_sliding_window_causal_mask``（:1344-1350，attention_mask 为 dict 时取
  首个预建 mask，:1341-1342）、``inputs_embeds.unsqueeze(2).expand(..., hc_mult, -1)``
  把残差扩展为 hc_mult 路流（:1351）、``position_embeddings`` 的 main/compress 两组
  cos/sin（:1352-1355）、``for layer in self.layers``（:1357-1366，module list :1304-1306）、
  ``self.norm(self.hc_head(hidden_states))``（:1368）。
- ``DeepseekV4HyperHead.forward``（:999-1003）：``input_norm(x.flatten(2).float())``
  → ``F.linear(flat, self.hc_fn.float())`` → ``sigmoid(mixes*hc_scale+hc_base)+eps``
  → ``(pre.unsqueeze(-1) * x).sum(dim=2)``；收拢 hc_mult 路流为单流。
- ``DeepseekV4RMSNorm``（:45-63）：fp32 方差归一 + 可学习 weight（fp32 乘）后转回。
- 层循环体内每个 ``DeepseekV4DecoderLayer``（:1137-1186）的双 mHC 残差、Attention
  （含 cache/压缩/Indexer）与 MoE 事实不在本图重复铺开：四类层组的层内结构由
  ``extract/deepseek_v4_hf.py``（经 ``m2_hca.py``/``m2_swa_hash.py``）构建的既有
  Block/算子详图提供，本总览只连到该详图并给出真实层归属与顺序。

四类层组与代表层来自 ``rules/deepseek_v4_hf.py:LAYER_GROUP_SPECS``；层归属由
``layer_group_assignments`` 按 ``layer_types``/``mlp_layer_types`` 逐层推导并校验
四组恰好覆盖全部主干层。共享模板不代表共享权重：同组层各有独立参数，
``tie_word_embeddings=false``（本配置）时 ``embed_tokens`` 与 ``lm_head`` 也是两份
独立权重（modeling 的 ``_tied_weights_keys`` 仅在 tie 开启时生效，:1443）。

本模块只读源码/配置，不导入 torch/transformers、不实例化或运行模型、不下载权重；
不生成 HTML/Excel，不做数值等价声明。
"""

from __future__ import annotations

from pathlib import Path

from structure_reader.rules.deepseek_v4_hf import (
    SOURCE_FILES,
    HfConfigError,
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
# 分层浏览页（整网→层组→Block→算子；与整网总览同候选、同 viewer/ 目录）。
DETAIL_PAGE = "structure.html"
PATH_ID = "network"


class NetworkDocError(ValueError):
    """整网总览数据构建失败（配置/范围/事实来源不一致），具名拒绝。"""


def logits_slice_length(seq_len: int, logits_to_keep: int) -> int:
    """int 型 ``logits_to_keep`` 下 lm_head 实际消费的位置数（源码 :1510-1511）。

    官方源码：``slice_indices = slice(-logits_to_keep, None)``，
    ``logits = self.lm_head(hidden_states[:, slice_indices, :])``。
    - ``Lk=0``：``slice(0, None)`` 保留全部 seqlen 个位置；
    - ``Lk>0``：``slice(-Lk, None)`` 的负起点越界时被裁剪到 0，长度为
      ``min(seqlen, Lk)``（例：seqlen=1、Lk=5 仍只输出 1 个位置，不是 5）；
    - 负整数（正起点）与 tensor 型 ``logits_to_keep`` 不在本票已分析范围，明确拒绝。
    """
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
    model_key: str = "deepseek_v4_flash_0731_hf",
) -> dict:
    """构建 ``hf_network_overview.v1``（单次网络调用的 token→logits 主干）。"""
    workspace_root = Path(workspace_root)
    values = load_hf_model_values(workspace_root, model_key=model_key)
    try:
        groups = layer_group_assignments(values)
    except ValueError as exc:
        raise NetworkDocError(f"四类层组归属校验失败: {exc}") from exc
    builder = _NetworkBuilder(values, groups)
    path = builder.build()
    layer_order = _layer_order(values, groups)
    return {
        "schema": NETWORK_SCHEMA,
        "model": {
            "model_key": values.model_key,
            "display_name": values.display_name,
            "implementation": values.implementation,
            "source_revision": values.source_revision,
            "config_path": values.config_path,
        },
        "scope": {
            "entry_class": "DeepseekV4ForCausalLM",
            "main_module_class": "DeepseekV4Model",
            "decoder_layer_class": "DeepseekV4DecoderLayer",
            "analysis_path": "eager",
            "layer_count": values.num_hidden_layers,
            "hidden_size": values.hidden_size,
            "vocab_size": values.vocab_size,
            "hc_mult": values.hc_mult,
            "main_path": [op["id"] for op in path["operators"]],
            "notes": [
                "范围：单次网络调用（一次 forward）从 input_ids 到 logits；不含 generate 采样循环、"
                "外部控制循环或多次调用耗时。",
                f"表达方式：整网外围与终点按源码逐句静态分析；循环体 {values.num_hidden_layers} 层不铺开，"
                "四类层组的层内结构由既有代表层 Block/算子详图（同候选的"
                f"{DETAIL_PAGE}）提供，本总览给出真实层归属、顺序与往返入口。",
                "mHC 是所选实现的真实残差机制（不是普通残差）：网络入口把单流扩展为 "
                "hc_mult 路流（modeling:1351），层内双 mHC 混合（:1169-1186），"
                "循环后再由 HyperHead 收拢（:999-1003/:1368）；本图不替换为标准残差模板。",
                "本图不加载权重、不运行模型、不做数值等价；seqlen/kvcachelen 等调用维度保留符号，"
                "空缓存 Prefill 取 kvcachelen=0；场景预设只绑定取值，不改变整网结构。",
                "tie_word_embeddings=false（根配置）：embed_tokens 与 lm_head 是两份独立权重；"
                "modeling:1443 的 _tied_weights_keys 仅在 tie 开启时生效，本配置未启用。",
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
                    "anchors": _detail_anchors(group["id"]),
                }
                for index, group in enumerate(groups)
            ],
        },
    }


def _detail_anchors(group_id: str) -> dict:
    """分层浏览页内该组的真实路由（同一份结构，不再按 prefill/decode 各建一套锚点）。"""
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
        if group_id is None:  # layer_group_assignments 已保证覆盖；此处不静默补位
            raise NetworkDocError(f"层 {index} 未归属任何真实层组（覆盖校验与实际不一致）")
        order.append({
            "index": index,
            "attention_type": values.layer_type(index),
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
    # 记录辅助（与 extract/deepseek_v4_hf.py 的阅读文档同构）
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
        return {
            "path": SOURCE_FILES["modeling_deepseek_v4.py"],
            "locator": locator, "kind": "source", "note": note,
        }

    def srcf(self, file_key, locator, note) -> dict:
        return {"path": SOURCE_FILES[file_key], "locator": locator, "kind": "source",
                "note": note}

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
                f"DeepSeek-V4-Flash-0731 整网主干（{self.values.num_hidden_layers} 层，"
                f"token → logits；单次 forward）"
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
                   "本票保留符号，统一场景参数属后续任务",
                   [src("1336", "torch.arange(inputs_embeds.shape[1])")])
        self.param("kvcachelen", "kvcachelen", "调用前逻辑历史 token 数（kvcachelen）", None,
                   "P = past_key_values.get_seq_length()（无 cache 时 0）",
                   "空缓存 Prefill 取 0；Decode 为调用前已处理历史；"
                   "不是物理保留条目数或字节数",
                   [src("1335", "past_seen = past_key_values.get_seq_length()")])
        self.param("H", "H", "hidden_size 隐层维度", v.hidden_size, None,
                   "config.hidden_size", [cfg("hidden_size", "根配置字段")])
        self.param("Vocab", "Vocab", "vocab_size 词表大小", v.vocab_size, None,
                   "config.vocab_size", [cfg("vocab_size", "根配置字段")])
        self.param("Nlay", "N_layer", "num_hidden_layers 主干层数（循环次数）", v.num_hidden_layers,
                   None, "config.num_hidden_layers；layer_order 逐层列真实顺序",
                   [cfg("num_hidden_layers", "根配置字段"),
                    src("1304-1306", "nn.ModuleList([DecoderLayer(config, i) for i in range(...)])")])
        self.param("Nhc", "Nhc", "hc_mult mHC 流数", v.hc_mult, None,
                   "config.hc_mult；网络入口把单流扩展为 Nhc 路流，循环后 HyperHead 收拢",
                   [cfg("hc_mult", "根配置字段"),
                    src("1351", "inputs_embeds.unsqueeze(2).expand(-1,-1,hc_mult,-1)"),
                    src("995-997", "HyperHead hc_fn/hc_base/hc_scale 形状")])
        self.param("W", "W", "sliding_window 滑动窗口", v.sliding_window, None,
                   "config.sliding_window；模型级 mask 的滑动窗口边界",
                   [cfg("sliding_window", "根配置字段"),
                    self.srcf("masking_utils.py", "1078-1162",
                              "create_sliding_window_causal_mask")])
        self.param("Lmask", "L_mask", "模型级 mask 的 KV 长度（数据相关）", None,
                   "L_mask = min(kvcachelen, W-1) + seqlen（is_full 分支在 min(cumulative,W-1) 下等价）",
                   "get_mask_sizes 依据首个 sliding 层的 cumulative_length 计算；"
                   "mask 在网络入口构建一次并传给全部层",
                   [self.srcf("cache_utils.py", "276-286", "Cache.get_mask_sizes"),
                    src("1344-1350", "create_sliding_window_causal_mask(...)")])
        self.param("Rk", "Rk", "qk_rope_head_dim 部分旋转维", v.qk_rope_head_dim,
                   "Rk = int(Dh × 64/512)",
                   "rotary_emb 的 cos/sin 尾维为 Rk÷2；main/compress 两组使用不同 "
                   "inv_freq（压缩路径为 YaRN）",
                   [cfg("qk_rope_head_dim", "legacy 字段"),
                    src("142-153", "DeepseekV4RotaryEmbedding.forward(layer_type)")])
        self.param("Lk", "Lk", "logits_to_keep 保留末位 token 数", 0,
                   "logits_to_keep 默认 0",
                   "0 → slice(0, None) 保留全部 seqlen 个位置；k>0 → slice(-k, None) 保留末 "
                   "min(seqlen, k) 个位置（越界不扩大）；负整数与 tensor 索引不在本题已分析范围",
                   [src("1471", "logits_to_keep: int | torch.Tensor = 0"),
                    src("1510", "slice(-logits_to_keep, None) if isinstance(int)")])
        self.param("Ssel", "S_sel", "logits 序列位置数（切片后）", None,
                   "S_sel = seqlen（Lk=0）或 min(seqlen, Lk)（Lk>0）",
                   f"lm_head 只对切片后的 hidden_states 计算，输出 [B,S_sel,Vocab]；"
                   f"例：seqlen=1、Lk=5 时 S_sel={logits_slice_length(1, 5)}（负起点裁剪为 0），不是 5；"
                   "Lk<0 与 tensor 型索引未分析，不声明其长度",
                   [src("1510-1511", "hidden_states[:, slice_indices, :] → self.lm_head(...)")])
        self.param("EpsN", "EpsN", "rms_norm_eps RMSNorm epsilon", v.rms_norm_eps, None,
                   "config.rms_norm_eps", [cfg("rms_norm_eps", "根配置字段"),
                                           src("57-58", "variance+eps → rsqrt")])
        self.param("Eps", "Eps", "hc_eps HyperHead sigmoid 权重下限", v.hc_eps, None,
                   "config.hc_eps", [cfg("hc_eps", "根配置字段"),
                                     src("1002", "pre = sigmoid(...) + self.eps")])

    def _build_boundaries(self) -> None:
        v = self.values
        self.tensor(
            "input_ids", "input_ids", ["B", "seqlen"], ["B", "seqlen"], "boundary",
            "调用入口 token id；与 inputs_embeds 恰一（modeling:1328-1329）；"
            "Hash MoE 层（" + ", ".join(str(i) for i in self._hash_layers()) +
            "）的 tid2eid 路由查表直接消费它（:1363→:1132）",
        )
        self.tensor(
            "attention_mask", "attention_mask", ["B", "P+T"], ["kvcachelen", "seqlen", "B"],
            "boundary",
            "调用方提供的可见性/padding 掩码；None 时全可见；若为 dict（generate 预建 "
            "per-layer mask）则取首个值（modeling:1341-1342）",
        )
        self.tensor(
            "past_key_values", "past_key_values（DynamicCache，调用前）", ["Nlay"], ["Nlay"],
            "cache",
            "use_cache 且未传入时新建 DynamicCache（:1330-1331）；按层型分发 per-layer cache"
            "（cache_utils.py:1808-1896），逐层字段见组详图；本图只表达网络级读前状态",
        )
        self.tensor(
            "inputs_embeds", "inputs_embeds", ["B", "seqlen", "H"], ["B", "seqlen", "H"],
            "activation",
            "embed_tokens 输出（:1333）；调用方也可直接传 inputs_embeds 跳过 embedding"
            "（与 input_ids 二选一，本图按 input_ids 路径表达）",
        )
        self.tensor(
            "position_ids", "position_ids", ["1", "seqlen"], ["seqlen", "kvcachelen"], "activation",
            "默认路径 arange(seqlen)+past_seen 后 unsqueeze(0)（:1334-1337）；供两组 rotary 与 "
            "mask 消费；调用方可传入自定义 [B,seqlen] 位置（本图不含该可选组合）",
        )
        self.tensor(
            "causal_mask", "causal_mask", ["B", "1", "seqlen", "Lmask"], ["B", "seqlen", "Lmask"],
            "mask",
            "create_sliding_window_causal_mask 生成的 4D mask（:1344-1350）："
            "首个 sliding 层尺寸 + 滑动窗口边界；与层内压缩条目扩展分开表达",
        )
        self.tensor(
            "hidden_streams", "hidden_states（mHC 流扩展后）", ["B", "seqlen", "Nhc", "H"],
            ["Nhc", "H"], "activation",
            f"inputs_embeds 在 dim=2 扩展为 Nhc 路残差流（:1351）；{self.values.num_hidden_layers} 层循环的输入",
        )
        self.tensor(
            "streams_after_layers", "hidden_states（全部层之后）", ["B", "seqlen", "Nhc", "H"],
            ["Nhc", "H"], "activation",
            "循环最后一次赋值的 hidden_states（:1358-1366）；仍为 Nhc 路流，未收拢",
        )
        self.tensor(
            "past_key_values_post", "past_key_values（各层 update 后，同一对象）",
            ["Nlay"], ["Nlay"], "cache",
            "各层 update 原地写入同一 DynamicCache（:1364），model 与 ForCausalLM 一并返回"
            "（:1369/:1532）；逐层字段见组详图",
        )

    def _build_forward_path(self) -> None:
        v = self.values
        src = self.src
        cfg = self.cfg

        # 1) embedding
        self.tensor(
            "embed_tokens_weight", "self.embed_tokens.weight", ["Vocab", "H"],
            ["Vocab", "H"], "weight",
            "nn.Embedding(vocab_size, hidden_size, padding_idx)（:1303）；"
            "tie_word_embeddings=false → 与 lm_head.weight 独立",
        )
        self.op(
            "net_embed", "DeepseekV4Model.embed_tokens", "Token Embedding（查表）",
            "embedding+gather", ["input_ids"], ["inputs_embeds"],
            ["embed_tokens_weight"], ["B", "seqlen", "H", "Vocab"],
            [src("1302-1303", "self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx)"),
             src("1332-1333", "if inputs_embeds is None: inputs_embeds = self.embed_tokens(input_ids)"),
             src("1328-1329", "input_ids 与 inputs_embeds 恰一（XOR 检查）")],
            "token id → [B,seqlen,H] 嵌入；padding_idx=config.pad_token_id 参与 embedding 构造"
            "（不参与前向计算结构）",
        )

        # 2) 默认 position_ids
        self.op(
            "net_position_ids", "DeepseekV4Model.forward", "默认位置序列",
            "arange+add+unsqueeze", ["past_key_values"], ["position_ids"], [],
            ["seqlen", "kvcachelen"],
            [src("1334-1337", "past_seen = get_seq_length(); position_ids = arange(T) + past_seen; unsqueeze(0)")],
            "仅在调用方未给 position_ids 时执行：kvcachelen 来自 cache 的 get_seq_length()；"
            "输出 [1,seqlen] 供两组 rotary 与 mask 使用",
        )

        # 3) 模型级 mask
        self.op(
            "net_mask_build", "DeepseekV4Model.forward", "滑动窗口因果 mask 构建",
            "mask_build", ["inputs_embeds", "attention_mask", "past_key_values", "position_ids"],
            ["causal_mask"], [], ["B", "seqlen", "Lmask", "W"],
            [src("1341-1342", "attention_mask 为 dict 时取 next(iter(...))"),
             src("1344-1350", "causal_mask = create_sliding_window_causal_mask(config, inputs_embeds, ...)"),
             self.srcf("masking_utils.py", "1078-1162", "create_sliding_window_causal_mask"),
             self.srcf("cache_utils.py", "276-286", "get_mask_sizes：is_full → W-1+query")],
            "网络入口构建一次，循环内传同一 mask（:1362）；层内还会把压缩条目拼接到 mask 尾部"
            "（Attention :837-841，属组详图）",
        )

        # 4) 两组 rotary（main / compress）
        for key, layer_type in (("main", "main"), ("compress", "compress")):
            self.tensor(
                f"cos_{key}", f'position_embeddings["{key}"] cos', ["1", "seqlen", "Rk÷2"],
                ["Rk", "seqlen"], "activation",
                f"self.rotary_emb(inputs_embeds, position_ids, layer_type='{layer_type}') 的 cos（:1353-1354）",
            )
            self.tensor(
                f"sin_{key}", f'position_embeddings["{key}"] sin', ["1", "seqlen", "Rk÷2"],
                ["Rk", "seqlen"], "activation",
                f"同一调用的 sin（:1353-1354）；main/compress 使用不同 inv_freq",
            )
            theta_ref = (cfg("rope_theta", "main RoPE base")
                         if key == "main" else cfg("compress_rope_theta", "compress RoPE base"))
            self.op(
                f"net_rope_{key}", "DeepseekV4Model.rotary_emb",
                f"RoPE 频率（{layer_type}）", "cos/sin",
                ["inputs_embeds", "position_ids"], [f"cos_{key}", f"sin_{key}"], [],
                ["Rk", "seqlen"],
                [src("1352-1355", "position_embeddings = {'main': rotary_emb(..., 'main'), "
                                   "'compress': rotary_emb(..., 'compress')}"),
                 src("142-153", "RotaryEmbedding.forward：按 layer_type 取 inv_freq → cos/sin"),
                 theta_ref],
                "两组频率以字典 position_embeddings 传给每层：Attention 用 main，"
                "压缩器/Indexer 用 compress；本图直接连四个分量，不虚构 dict 张量",
            )

        # 5) mHC 流扩展
        self.op(
            "net_hc_expand", "DeepseekV4Model.forward", "残差扩展为 mHC 多流",
            "unsqueeze+expand+contiguous",
            ["inputs_embeds"], ["hidden_streams"], [], ["Nhc", "H"],
            [src("1351", "hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, hc_mult, -1).contiguous()")],
            "普通 Transformer 的残差是单流；此处单流被复制为 Nhc 路并行流，"
            "层内由两个 HyperConnection 混合（组详图），循环后由 HyperHead 收拢",
        )

        # 6) 层循环
        self.op(
            "net_layer_loop", "DeepseekV4Model.layers",
            f"{v.num_hidden_layers} 层循环（四类真实层组，顺序见 layer_order）",
            "loop",
            ["hidden_streams", "cos_main", "sin_main", "cos_compress", "sin_compress",
             "position_ids", "causal_mask", "input_ids", "past_key_values"],
            ["streams_after_layers", "past_key_values_post"], [], ["Nlay", "Nhc"],
            [src("1304-1306", "self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(num_hidden_layers)])"),
             src("1357-1366", "for layer in self.layers: hidden_states = layer(hidden_states, position_embeddings=..., "
                              "position_ids=..., attention_mask=causal_mask, input_ids=input_ids, "
                              "past_key_values=past_key_values)"),
             src("1161-1186", "DeepseekV4DecoderLayer.forward（循环体；双 mHC + Attention + MoE，见组详图）")],
            f"同一模板按 ModuleList 顺序重复 N 次，不铺开 {self.values.num_hidden_layers} 层：组内结构由代表层详图提供；"
            "input_ids 仅供 hash_moe 层查表（其余层不消费）；cache 由各层 update 原地写回；"
            "残差流在循环内保持 Nhc 路（层内混合见组详图）",
        )

        # 7) HyperHead：收拢 mHC 流
        self.tensor(
            "hc_head_normed", "flat = self.input_norm(x.flatten(2).float())",
            ["B", "seqlen", "Nhc×H"], ["Nhc", "H", "EpsN"], "activation",
            "HyperHead 先把 Nhc 路流展平到特征维再 UnweightedRMSNorm（fp32，:1000/:993）",
        )
        self.op(
            "net_hc_head_norm", "DeepseekV4HyperHead", "HyperHead：流展平+归一",
            "flatten+float+rmsnorm-unweighted",
            ["streams_after_layers"], ["hc_head_normed"], [], ["Nhc", "H", "EpsN"],
            [src("1000", "flat = self.input_norm(x.flatten(2).float())"),
             src("993", "self.input_norm = DeepseekV4UnweightedRMSNorm(eps=rms_norm_eps)"),
             src("71-72", "x * rsqrt(mean(x²)+eps)（无可学习权重）")],
            "收拢前的流归一；无权重 RMSNorm",
        )
        self.tensor(
            "hc_head_fn", "self.hc_fn", ["Nhc", "Nhc×H"], ["Nhc", "H"], "weight",
            "HyperHead 权重 (Nhc, Nhc*H)（:995）",
        )
        self.tensor(
            "hc_head_mixes", "mixes = F.linear(flat, self.hc_fn.float())", ["B", "seqlen", "Nhc"],
            ["Nhc", "H"], "activation",
            "每路流的收拢门控原始分数（:1001）",
        )
        self.op(
            "net_hc_head_project", "DeepseekV4HyperHead", "HyperHead：收拢门控投影",
            "linear", ["hc_head_normed"], ["hc_head_mixes"], ["hc_head_fn"], ["Nhc", "H"],
            [src("995", "self.hc_fn = nn.Parameter(empty(hc_mult, hc_mult * hidden_size))"),
             src("1001", "mixes = F.linear(flat, self.hc_fn.float())")],
            "输出 Nhc 个门控分数（每路流一个）",
        )
        self.tensor(
            "hc_head_base", "self.hc_base", ["Nhc"], ["Nhc"], "weight",
            "门控偏置（:996）",
        )
        self.tensor(
            "hc_head_scale", "self.hc_scale", ["1"], ["Nhc"], "weight",
            "门控缩放（:997）",
        )
        self.tensor(
            "hc_head_pre", "pre = sigmoid(mixes*hc_scale+hc_base)+eps", ["B", "seqlen", "Nhc"],
            ["Nhc", "Eps"], "activation",
            "收拢权重（:1002）；+eps 保证非零",
        )
        self.op(
            "net_hc_head_pre", "DeepseekV4HyperHead", "HyperHead：收拢权重",
            "affine+sigmoid+add", ["hc_head_mixes", "hc_head_base", "hc_head_scale"],
            ["hc_head_pre"], [], ["Nhc", "Eps"],
            [src("1002", "pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps")],
            "sigmoid 后加 hc_eps；fp32 计算",
        )
        self.tensor(
            "hc_head_collapsed", "hidden_states（HyperHead 收拢后）", ["B", "seqlen", "H"],
            ["Nhc", "H"], "activation",
            "Nhc 路流按 pre 加权求和为单流（:1003）",
        )
        self.op(
            "net_hc_head_collapse", "DeepseekV4HyperHead", "HyperHead：流收拢",
            "broadcast-mul+sum", ["hc_head_pre", "streams_after_layers"],
            ["hc_head_collapsed"], [], ["Nhc", "H"],
            [src("1003", "return (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)")],
            "mHC 流回到单流；这是 mHC 在整网出口处的第二次“混合”，不是普通残差相加",
        )

        # 8) Final RMSNorm
        self.tensor(
            "final_norm_weight", "self.norm.weight", ["H"], ["H"], "weight",
            "Final RMSNorm 可学习缩放（:1307/:51），fp32 保持",
        )
        self.tensor(
            "final_hidden", "hidden_states（Final RMSNorm 后）", ["B", "seqlen", "H"],
            ["H", "EpsN"], "activation",
            "model 的 last_hidden_state（:1368-1369）；进入 lm_head 前不再有其它网络级变换",
        )
        self.op(
            "net_final_norm", "DeepseekV4Model.norm", "Final RMSNorm", "rmsnorm",
            ["hc_head_collapsed", "final_norm_weight"], ["final_hidden"], [],
            ["H", "EpsN"],
            [src("1368", "hidden_states = self.norm(self.hc_head(hidden_states))"),
             src("1307", "self.norm = DeepseekV4RMSNorm(hidden_size, eps=rms_norm_eps)"),
             src("54-60", "fp32 方差归一 → weight(fp32) 乘 → to(input_dtype)")],
            "注意执行顺序：先 HyperHead 收拢、后共享 Final RMSNorm（不是先 norm 再收拢）",
        )

        # 9) logits 切片 + lm_head
        self.tensor(
            "lm_input", "hidden_states[:, slice_indices, :]（切片后）", ["B", "Ssel", "H"],
            ["B", "Ssel", "H"], "activation",
            "只保留需要的序列位置（:1510）；Lk=0 时 slice(0, None) 保留全部 seqlen 位；"
            "Lk>0 时 slice(-Lk, None) 长度为 min(seqlen, Lk)（越界不扩大）",
        )
        self.op(
            "net_logits_slice", "DeepseekV4ForCausalLM.forward", "logits 位置切片",
            "slice", ["final_hidden"], ["lm_input"], [], ["Lk", "Ssel"],
            [src("1508", "hidden_states = outputs.last_hidden_state"),
             src("1471", "logits_to_keep: int | torch.Tensor = 0"),
             src("1510", "slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep")],
            "切片发生在 lm_head 之前（只对保留位置做投影）；S_sel = seqlen（Lk=0）或 "
            "min(seqlen, Lk)（Lk>0，负起点越界裁剪）；负整数/tensor 索引不在本题分析范围",
        )
        self.tensor(
            "lm_head_weight", "self.lm_head.weight", ["Vocab", "H"], ["Vocab", "H"],
            "weight",
            "nn.Linear(hidden_size, vocab_size, bias=False)（:1451）；"
            "tie_word_embeddings=false → 与 embed_tokens.weight 不共享",
        )
        self.tensor(
            "logits", "logits", ["B", "Ssel", "Vocab"], ["B", "Ssel", "Vocab"], "boundary",
            "网络终点（:1511/:1531）：未归一化分数；Lk>0 时序列轴为 min(seqlen, Lk)；"
            "不在此图做 softmax/采样",
        )
        self.op(
            "net_lm_head", "DeepseekV4ForCausalLM.lm_head", "LM Head 投影（得到 logits）",
            "linear", ["lm_input"], ["logits"], ["lm_head_weight"], ["Vocab", "H", "Ssel"],
            [src("1451", "self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)"),
             src("1511", "logits = self.lm_head(hidden_states[:, slice_indices, :])"),
             src("1528-1536", "MoeCausalLMOutputWithPast(logits=logits, past_key_values=..., ...)")],
            "网络终点：hidden → vocab 的线性投影；loss/aux_loss 属训练/可选输出，不在本主干",
        )

    def _hash_layers(self) -> list[int]:
        return [i for i in range(self.values.num_hidden_layers)
                if self.values.mlp_layer_type(i) == "hash_moe"]

    def _build_cache_notes(self) -> None:
        self.cache_notes = [
            "past_key_values 在网络入口按需新建 DynamicCache（modeling:1330-1331），"
            "按层型分发 per-layer cache（cache_utils.py:1808-1896）；本图只表达网络级"
            "读前/写后状态，逐层字段（主 KV、压缩条目、Indexer、overlap）见四组代表层详图。",
            "循环内每层 update 原地写入同一 DynamicCache（:1364），"
            "model 以 past_key_values 返回（:1369），ForCausalLM 原样透传（:1532）。",
            "P = past_key_values.get_seq_length()（:1335）：决定默认 position_ids 的起点与"
            "模型级 mask 尺寸；空缓存 Prefill 时 kvcachelen=0。",
            "网络级 cache 状态不改变层内结构：滑动窗口保留长度、压缩条目与 Indexer 条目数"
            "均为调用历史相关的数据相关量，已在组详图中按符号表达。",
        ]

    def _build_coverage_notes(self) -> None:
        hash_layers = ", ".join(str(i) for i in self._hash_layers())
        self.coverage_notes = [
            "已表达（与 modeling 行号逐条对应）：ForCausalLM.forward → Model.forward → "
            "embed_tokens、默认 position_ids、滑动窗口因果 mask、main/compress 两组 RoPE 频率、"
            f"mHC 流扩展、{self.values.num_hidden_layers} 层循环（真实层序与四组归属）、"
            "HyperHead 收拢、Final RMSNorm、logits 位置切片、lm_head → logits、"
            "past_key_values 读取与写回。",
            "层内结构不在此图重复铺开：四类层组的双 mHC 残差、Attention（主 KV/压缩器/"
            "Indexer/部分 RoPE/逆 RoPE/分组输出投影）与 MoE（TopK/Hash 路由、专家模板、"
            "共享专家）由同候选分层浏览页的层组/Block/算子层提供（structure.html），"
            "本页给出真实层归属、顺序与进入/返回路径。",
            "同构模板复用不代表共享权重：同组每层是独立 DeepseekV4DecoderLayer 实例；"
            "本配置 tie_word_embeddings=false，embed_tokens 与 lm_head 权重也独立。",
            f"Hash MoE 层（{hash_layers}）的路由查表消费网络入口的 input_ids（:1363→:1132）；"
            "其余层的输入不含 input_ids。",
            "logits_to_keep 只分析非负 int 路径：Lk=0 保留全部 seqlen 位，Lk>0 为 "
            "min(seqlen, Lk)（slice(-Lk,None) 负起点越界裁剪）；负整数与 tensor 型索引未分析。",
            "未覆盖（明确列出，不宣称已分析）：generate/采样循环与多次调用累计、"
            "loss/aux_loss 训练路径、output_router_logits 辅助输出、量化执行"
            "（FP8/FP4 与设备 kernel）、torch.compile/梯度检查点、数值等价与权重内容。",
            "数据相关量（L_mask、各层缓存条目数、命中专家数等）保留符号与派生式；"
            "不填运行时假设值，不把物理保留长度当成逻辑历史长度。",
        ]
