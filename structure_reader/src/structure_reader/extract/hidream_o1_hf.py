"""HiDream-O1-Image 单次生成 forward 的参数化 Block 结构（静态分析，不运行模型）。

结构事实全部来自官方实现（GitHub ``HiDream-ai/HiDream-O1-Image`` commit
``2c2d29ff…``）的 ``models/qwen3_vl_transformers.py``（下称 modeling）、
``models/pipeline.py``（下称 pipeline）、``models/utils.py``（下称 utils）。
本模块只构造**结构数据**（参数/张量/算子/条件/派生），不建图、不渲染、不做 I/O。

两个真实重复层组，各自一份 Block 模板：

- ``uit-decoder``（36 层，layer 0..35）：``Qwen3VLTextDecoderLayer`` ——
  RMSNorm → q/k/v 投影（q 4096、k/v 1024）→ 逐头 q_norm/k_norm → MRoPE →
  两趟注意力（AR 子集 causal + 全序列 non-causal + index_copy）→ o_proj →
  残差 → RMSNorm → SwiGLU（gate/up 12288、down）→ 残差 → 条件 DeepStack 注入。
  层内**没有**时间步调制/adaLN、没有 MoE、没有 KV cache 读写。
- ``vit-tower``（27 blocks，block 0..26，仅参考图/编辑任务执行）：
  ``Qwen3VLVisionBlock`` —— LayerNorm → qkv 3456 → 2D rotary → 非因果注意力
  （按图切分 cu_seqlens）→ proj → 残差 → LayerNorm → GELU-tanh MLP 4304 → 残差；
  第 8/16/24 个 block 后另接 DeepStack merger。

shape 保留符号：``Himg``/``Wimg``（图像高宽）、``Ptxt``（prompt token 数）、
``Sref``（参考图 patch 数）、``Svis``（VLM 条件图 patch 数）等由分析场景绑定取值；
未知量（如未给参考图尺寸）保持未知，不默认 0。
"""
from __future__ import annotations

from pathlib import Path

from structure_reader.extract.hf_scenario import structure_document
from structure_reader.rules.hidream_o1_hf import (
    HIDREAM_BIND_KEYS,
    HIDREAM_SCENARIOS,
    HfModelValues,
    decoder_layer_indices,
    layer_group_assignments,
    load_hf_model_values,
    vision_block_indices,
)

__all__ = [
    "MODELING",
    "PIPELINE",
    "UTILS",
    "build_block_structure",
    "build_layer_structure",
    "build_layer_documents",
    "group_by_id",
]

MODELING = "models/HiDream/HiDream-O1-Image/qwen3_vl_transformers.py"
PIPELINE = "models/HiDream/HiDream-O1-Image/pipeline.py"
UTILS = "models/HiDream/HiDream-O1-Image/utils.py"

DOCUMENT_SCHEMA = "hidream_o1_layer_document.v1"


def _src(locator: str, note: str, path: str = MODELING) -> dict:
    return {"path": path, "locator": locator, "kind": "source", "note": note}


def _rule(locator: str, note: str, path: str = MODELING) -> dict:
    return {"path": path, "locator": locator, "kind": "derived", "note": note}


def _param(pid: str, symbol: str, meaning: str, *, value=None, expression=None,
           note: str = "", source_refs=None, unknown_reason: str = "",
           domain: dict | None = None, consistency: str | None = None) -> dict:
    row = {
        "id": pid, "symbol": symbol, "meaning": meaning, "value": value,
        "expression": expression, "note": note,
        "source_refs": list(source_refs or []),
    }
    if unknown_reason:
        row["unknown_reason"] = unknown_reason
    if domain:
        row["domain"] = dict(domain)
    if consistency:
        row["consistency"] = consistency
    return row


def _tensor(tid: str, name: str, shape: list, kind: str, note: str,
            parameter_refs: list) -> dict:
    return {"id": tid, "name": name, "shape": shape, "kind": kind,
            "note": note, "parameter_refs": list(parameter_refs)}


def _op(oid: str, module_path: str, name: str, op: str, inputs: list, outputs: list,
        *, weights=(), parameter_refs=(), source_refs, note: str,
        analysis_status: str = "analyzed") -> dict:
    return {
        "id": oid, "module_path": module_path, "name": name, "op": op,
        "inputs": list(inputs), "outputs": list(outputs), "weights": list(weights),
        "parameter_refs": list(parameter_refs), "source_refs": list(source_refs),
        "analysis_status": analysis_status, "note": note,
    }


# ---------------------------------------------------------------------------
# 共用参数（两个 Block 都需要的输入规模与常量）
# ---------------------------------------------------------------------------


def _io_parameters() -> list[dict]:
    """可填写输入规模参数（结构里保留符号；取值由场景/工作簿绑定）。"""
    return [
        _param("B", "B", "batch 批大小", note="调用输入维度；工作簿里可填写，结构保留符号 B",
               domain={"kind": "integer", "min": 1}),
        _param("Himg", "Himg", "目标图像高度（生效网络尺寸，必须按 32 对齐）",
               note=("填的是实际生效网络尺寸：必须能被 patch 边长整除（Himg % Ppix == 0），"
                     "否则视为未通过约束而不是静默 floor；原始请求尺寸的吸附在外部"),
               domain={"kind": "integer", "min": 32},
               consistency="Himg % Ppix == 0"),
        _param("Wimg", "Wimg", "目标图像宽度（生效网络尺寸，必须按 32 对齐）",
               note="与 Himg 同属可填写输入；patch 数 = (Himg/32)×(Wimg/32)，必须整除",
               domain={"kind": "integer", "min": 32},
               consistency="Wimg % Ppix == 0"),
        _param("Ptxt", "Ptxt", "文本 prompt token 数（含 chat template/boi/1×tms）",
               note=("pipeline 用 chat template + boi + 1 个 tms token 组文本序列"
                     "（pipeline:31-78）；长度由 tokenizer 决定，属可填写输入；至少 1 个 token"),
               domain={"kind": "integer", "min": 1}),
        _param("Kref", "Kref", "参考图/条件图数量 Kref", value=None,
               note="0=t2i（视觉塔与参考 patch 不执行）；>0 时视觉塔、参考图 patch 与条件图 token 必须同步 >0",
               domain={"kind": "integer", "min": 0}),
        _param("Sref", "Sref", "参考图像素 patch token 总数（数据相关）", value=None,
               note=("参考图先 resize 到 32 对齐再 patch 化（pipeline:214-230）；"
                     "Kref=0 时必须为 0，Kref>0 时必须 >0"),
               unknown_reason="参考图尺寸未给定（Kref>0 时由实际参考图决定）",
               domain={"kind": "integer", "min": 0},
               consistency=("((Kref > 0) or (Sref == 0)) and "
                            "((Kref == 0) or (Sref > 0))")),
        _param("Svis", "Svis", "VLM 条件图 patch token 总数（merge 前，数据相关）", value=None,
               note=("条件图 resize 到 384 级后用 16×16 patch 编码（pipeline:236-243）；"
                     "Kref=0 时必须为 0，Kref>0 时必须 >0 且能被 merge 单元整除（Svis % 4 = 0）"),
               unknown_reason="条件图尺寸未给定（Kref>0 时由实际参考图决定）",
               domain={"kind": "integer", "min": 0},
               consistency=("((Kref > 0) or (Svis == 0)) and "
                            "((Kref == 0) or (Svis > 0)) and "
                            "(Svis % (Mv * Mv) == 0)")),
    ]


def _derived_lengths() -> tuple[list[dict], list[dict]]:
    """长度派生参数 + 对应推导式（结构保留符号；场景/Excel 代入取值）。"""
    params = [
        _param("Spix", "S_pix", "目标图像 patch 数（=生成 token 数的一部分）", expression="S_pix = (Himg // Ppix) * (Wimg // Ppix)",
               note="每个 32×32×3 patch 对应一个生成 token（pipeline:317）",
               source_refs=[_src("313-317", "noise rearrange 成 [B,(H/32)*(W/32),C*32*32]", PIPELINE)]),
        _param("Sgen", "S_gen", "生成 token 数（vinputs 列数 = 目标 patch + 参考图 patch）",
               expression="S_gen = S_pix + Sref",
               note="参考图 patch 与目标 patch 共用同一个 x_embedder（pipeline:388）",
               source_refs=[_src("354-357/388", "vinputs=cat([z, ref_patches]) 与 vinput_mask", PIPELINE)]),
        _param("Stot", "S_tot", "主干序列长度 = 文本 token + 生成 token",
               expression="S_tot = Ptxt + S_gen",
               note="_forward_generation 把 x_embedder(vinputs) 拼在文本嵌入之后（modeling:1530）",
               source_refs=[_src("1525-1532", "inputs_embeds = cat([text_embeds, vinputs_embedded])")]),
        _param("Nar", "N_ar", "AR 文本行数（tms token 归生成侧）",
               expression="N_ar = Ptxt - 1",
               note=("token_types 把最后一个文本 token（tms）标成 >0，因此 AR 行数 = Ptxt-1"
                     "（pipeline:65-71）；flash 第 1 趟只在这 Nar 行上做 causal"),
               source_refs=[_src("65-71", "token_types[bgn:...]=1; token_types[txt-1:txt]=3", PIPELINE)]),
        _param("Ngt", "N_gt", "生成侧行数（tms + 生成 token）", expression="N_gt = S_gen + 1",
               note="这些行在 flash 第 2 趟里对全部 token 双向；eager 路径把它们的 mask 行置 0",
               source_refs=[_src("294-302", "vinput_mask = (token_types==1)；tms=3→bin=1", PIPELINE)]),
    ]
    derivations = [
        {"id": "S_pix", "expression": "(Himg // Ppix) * (Wimg // Ppix)",
         "label": "目标图像 patch 数", "note": "像素 patch 边长 32（pipeline:18）"},
        {"id": "S_gen", "expression": "S_pix + Sref",
         "label": "生成 token 数（vinputs 列数）", "note": "目标 patch + 参考图 patch"},
        {"id": "S_tot", "expression": "Ptxt + S_gen",
         "label": "主干序列长度", "note": "文本 token + 生成 token"},
        {"id": "N_ar", "expression": "Ptxt - 1",
         "label": "AR 文本行数", "note": "tms token 计入生成侧（token_types=3）"},
        {"id": "N_gt", "expression": "S_gen + 1",
         "label": "生成侧行数（含 tms）", "note": "双向注意力行数"},
        {"id": "N_vis", "expression": "Svis // (Mv * Mv)",
         "label": "DeepStack 特征行数", "note": "条件图 merge 后 token 数（2×2 merge）"},
    ]
    return params, derivations


def _norm_parameters() -> list[dict]:
    return [
        _param("H", "H", "hidden_size 隐层维度", value=4096,
               source_refs=[_src("1035-1045", "hidden_size = config.text_config.hidden_size")]),
        _param("Nh", "Nh", "num_attention_heads Q 头数", value=32,
               source_refs=[_src("418-419", "q_proj = Linear(hidden, num_heads*head_dim)")]),
        _param("Nkv", "Nkv", "num_key_value_heads KV 头数", value=8,
               source_refs=[_src("421-426", "k_proj/v_proj = Linear(hidden, num_kv*head_dim)")]),
        _param("Dh", "Dh", "head_dim 头维", value=128,
               source_refs=[_src("412-414", "head_dim = config.head_dim；scaling = head_dim**-0.5")]),
        _param("Hq", "H_q", "Q 拼接宽度（Nh×Dh）", value=4096, expression="H_q = Nh * Dh",
               note="o_proj 的输入宽度（modeling:427-429）",
               source_refs=[_src("427-429", "o_proj = Linear(num_heads*head_dim, hidden)")]),
        _param("Hkv", "H_kv", "K/V 拼接宽度（Nkv×Dh）", value=1024, expression="H_kv = Nkv * Dh",
               note="GQA：32 Q 头共享 8 个 KV 头（repeat 4）",
               source_refs=[_src("421-426", "k_proj/v_proj 输出 num_kv*head_dim")]),
        _param("Fi", "Fi", "SwiGLU 中间维（intermediate_size）", value=12288,
               source_refs=[_src("480-493", "gate_proj/up_proj: hidden→intermediate")]),
        _param("EpsN", "EpsN", "RMSNorm epsilon", value=1e-6,
               source_refs=[_src("358-372", "Qwen3VLTextRMSNorm(variance + eps)")]),
        _param("Ndh", "Ndh", "DeepStack 特征层数（deepstack_visual_indexes 长度）", value=3,
               note="3 个特征分别注入 decoder 的前 3 层（layer 0..2）",
               source_refs=[_src("913-919", "layer_idx in range(len(deepstack_visual_embeds))")]),
        _param("Nlay", "Nlay", "decoder 层数", value=36,
               source_refs=[_src("797-799", "self.layers = ModuleList(range(num_hidden_layers))")]),
        _param("Ppix", "Ppix", "像素 patch 边长（32×32 像素 = 1 token）", value=32,
               source_refs=[_src("1039", "self.patch_size = 32", PIPELINE),
                            _src("18", "PATCH_SIZE = 32", PIPELINE)]),
    ]


# ---------------------------------------------------------------------------
# 统一 decoder Block（36 层）
# ---------------------------------------------------------------------------


def _decoder_block(values: HfModelValues) -> dict:
    input_params = _io_parameters()
    derived_params, derivations = _derived_lengths()
    constants = _norm_parameters()
    params = [input_params[0], input_params[1], input_params[2], input_params[3],
              input_params[4],  # Kref：与 Sref/Svis 的一致性约束共用
              input_params[5],  # Sref
              input_params[6],  # Svis：与 vision/network 共用的输入 + merge 整除约束
              _param("Mv", "Mv", "视觉塔 spatial_merge_size", value=2,
                     source_refs=[_src("595-597", "spatial_merge_unit = merge * merge")]),
              _param("Nvis", "Nvis", "DeepStack 特征行数（=条件图 merge 后 token 数）",
                     value=None, expression="Nvis = Svis // (Mv * Mv)",
                     note="DeepStack 特征按 merge 后 token 行对齐 visual_pos_masks；t2i 时该算子不执行",
                     source_refs=[_src("938-945", "hidden_states[visual_pos_masks] += visual_embeds")]),
              derived_params[0], derived_params[1], derived_params[2],
              derived_params[3], derived_params[4],
              constants[0], constants[1], constants[2], constants[3],
              constants[4], constants[5], constants[6], constants[7],
              constants[8], constants[9], constants[10]]

    tensors = [
        _tensor("d_hidden_in", "hidden_states（层输入）", ["B", "Stot", "H"],
                "boundary",
                "上一层输出或网络入口的 x（文本嵌入 + patch 嵌入拼接，modeling:1530）；"
                "层间是标量残差流，不是多路流", ["B", "Stot", "H"]),
        _tensor("d_pos_cos", "cos（MRoPE，层间共享）", ["B", "Stot", "Dh"], "boundary",
                "位置嵌入在层循环外计算一次（modeling:1284-1289），层内只做旋转",
                ["B", "Stot", "Dh"]),
        _tensor("d_pos_sin", "sin（MRoPE，层间共享）", ["B", "Stot", "Dh"], "boundary",
                "同上；MRoPE 为 interleaved [24,20,20]，三轴位置来自 utils:246-349",
                ["B", "Stot", "Dh"]),
        _tensor("d_ar_index", "idx_ar（AR 行索引，层外只算一次）", ["Nar"], "boundary",
                "flash 路径按 token_types 取 AR 行（modeling:1291-1293）；每层共用同一索引",
                ["Nar"]),
        _tensor("d_visual_mask", "visual_pos_masks（条件图占位位置）", ["B", "Stot"], "boundary",
                "仅参考图任务存在：True=条件图 placeholder 位置（modeling:1505-1508）；"
                "DeepStack 只在这些行上相加", ["B", "Stot"]),
        _tensor("d_ds_embeds", "deepstack_visual_embeds[layer_idx]", ["Nvis", "H"], "boundary",
                "3 个 DeepStack 特征之一（[N_vis,H]），与 visual_pos_masks 的 True 行一一对应；"
                "t2i 时不存在", ["Nvis", "H"]),
        _tensor("d_normed", "input_layernorm 输出", ["B", "Stot", "H"], "activation",
                "RMSNorm（逐 token，按 H 归一）", ["B", "Stot", "H"]),
        _tensor("d_q_lin", "q_proj 输出", ["B", "Stot", "Hq"], "activation",
                "Linear(4096→4096)，无 bias（attention_bias=false）", ["B", "Stot", "Hq"]),
        _tensor("d_q_head", "q.view(B,S,Nh,Dh)", ["B", "Stot", "Nh", "Dh"], "activation",
                "按头拆分（modeling:448）", ["B", "Stot", "Nh", "Dh"]),
        _tensor("d_q_normed", "q_norm", ["B", "Stot", "Nh", "Dh"], "activation",
                "逐头 RMSNorm(Dh)，权重 [Dh]（modeling:430/448）", ["B", "Stot", "Nh", "Dh"]),
        _tensor("d_q", "q（rope 后）", ["B", "Nh", "Stot", "Dh"], "activation",
                "transpose(1,2) 后做 MRoPE（modeling:448-453）", ["B", "Nh", "Stot", "Dh"]),
        _tensor("d_k_lin", "k_proj 输出", ["B", "Stot", "Hkv"], "activation",
                "Linear(4096→1024)，无 bias", ["B", "Stot", "Hkv"]),
        _tensor("d_k_head", "k.view(B,S,Nkv,Dh)", ["B", "Stot", "Nkv", "Dh"], "activation",
                "按 KV 头拆分", ["B", "Stot", "Nkv", "Dh"]),
        _tensor("d_k_normed", "k_norm", ["B", "Stot", "Nkv", "Dh"], "activation",
                "逐头 RMSNorm(Dh)", ["B", "Stot", "Nkv", "Dh"]),
        _tensor("d_k", "k（rope 后）", ["B", "Nkv", "Stot", "Dh"], "activation",
                "MRoPE 后的 key", ["B", "Nkv", "Stot", "Dh"]),
        _tensor("d_v_lin", "v_proj 输出", ["B", "Stot", "Hkv"], "activation",
                "Linear(4096→1024)，无 bias；v 不做 RoPE", ["B", "Stot", "Hkv"]),
        _tensor("d_v", "v（按头）", ["B", "Nkv", "Stot", "Dh"], "activation",
                "view + transpose，无 norm（modeling:450）", ["B", "Nkv", "Stot", "Dh"]),
        _tensor("d_q_ar", "q[idx_ar]", ["B", "Nh", "Nar", "Dh"], "activation",
                "只在 AR 行上切（flash 第 1 趟）", ["B", "Nh", "Nar", "Dh"]),
        _tensor("d_k_ar", "k[idx_ar]", ["B", "Nkv", "Nar", "Dh"], "activation",
                "只切 AR 行", ["B", "Nkv", "Nar", "Dh"]),
        _tensor("d_v_ar", "v[idx_ar]", ["B", "Nkv", "Nar", "Dh"], "activation",
                "只切 AR 行", ["B", "Nkv", "Nar", "Dh"]),
        _tensor("d_score_ar", "attn_weights（AR 子集，causal）", ["B", "Nh", "Nar", "Nar"],
                "activation", "QK^T×Dh^-0.5，causal=True；flash 内核一次算完", ["B", "Nh", "Nar", "Nar"]),
        _tensor("d_prob_ar", "softmax（AR 子集）", ["B", "Nh", "Nar", "Nar"], "activation",
                "softmax 在 df32 上做（eager:175）", ["B", "Nh", "Nar", "Nar"]),
        _tensor("d_ctx_ar", "out_ar", ["B", "Nar", "Nh", "Dh"], "activation",
                "AR 行注意力输出", ["B", "Nar", "Nh", "Dh"]),
        _tensor("d_score_full", "attn_weights（全序列，non-causal）", ["B", "Nh", "Stot", "Stot"],
                "activation", "第 2 趟：全序列双向 QK^T", ["B", "Nh", "Stot", "Stot"]),
        _tensor("d_prob_full", "softmax（全序列）", ["B", "Nh", "Stot", "Stot"], "activation",
                "双向 softmax（causal=False）", ["B", "Nh", "Stot", "Stot"]),
        _tensor("d_ctx_full", "out_full", ["B", "Stot", "Nh", "Dh"], "activation",
                "全序列注意力输出（AR 行稍后被覆盖）", ["B", "Stot", "Nh", "Dh"]),
        _tensor("d_ctx", "attn_output（AR 行 = 第 1 趟）", ["B", "Stot", "Nh", "Dh"],
                "activation", "out_full[:, idx_ar] = out_ar（index_copy，modeling:1343-1345）",
                ["B", "Stot", "Nh", "Dh"]),
        _tensor("d_attn_flat", "attn_output.reshape(B,S,Hq)", ["B", "Stot", "Hq"],
                "activation", "拼头（modeling:1348）", ["B", "Stot", "Hq"]),
        _tensor("d_attn", "o_proj 输出", ["B", "Stot", "H"], "activation",
                "Linear(4096→4096)，无 bias", ["B", "Stot", "H"]),
        _tensor("d_h1", "残差后的 hidden_states", ["B", "Stot", "H"], "activation",
                "residual + attn_output（普通标量残差，modeling:532）", ["B", "Stot", "H"]),
        _tensor("d_h1n", "post_attention_layernorm 输出", ["B", "Stot", "H"], "activation",
                "RMSNorm（modeling:536）", ["B", "Stot", "H"]),
        _tensor("d_gate", "gate_proj 输出", ["B", "Stot", "Fi"], "activation",
                "Linear(4096→12288)，无 bias", ["B", "Stot", "Fi"]),
        _tensor("d_up", "up_proj 输出", ["B", "Stot", "Fi"], "activation",
                "Linear(4096→12288)，无 bias", ["B", "Stot", "Fi"]),
        _tensor("d_gate_act", "SiLU(gate)", ["B", "Stot", "Fi"], "activation",
                "ACT2FN['silu']（config hidden_act=silu）", ["B", "Stot", "Fi"]),
        _tensor("d_ffn_act", "SiLU(gate)*up", ["B", "Stot", "Fi"], "activation",
                "SwiGLU 的逐元素乘（modeling:492）", ["B", "Stot", "Fi"]),
        _tensor("d_ffn", "down_proj 输出", ["B", "Stot", "H"], "activation",
                "Linear(12288→4096)，无 bias", ["B", "Stot", "H"]),
        _tensor("d_h2", "FFN 残差后的 hidden_states", ["B", "Stot", "H"], "activation",
                "residual + mlp_output（modeling:538）", ["B", "Stot", "H"]),
        _tensor("d_hidden_out", "层输出（可能已注入 DeepStack）", ["B", "Stot", "H"], "activation",
                "t2i 时等于 d_h2；参考图任务的 layer 0..2 在条件图行上再加 DeepStack 特征",
                ["B", "Stot", "H"]),
        # 权重
        _tensor("d_w_norm1", "input_layernorm.weight", ["H"], "weight",
                "RMSNorm 权重（无 bias）", ["H"]),
        _tensor("d_w_q", "q_proj.weight", ["Hq", "H"], "weight",
                "Linear(4096→4096)，attention_bias=false → 无 bias", ["Hq", "H"]),
        _tensor("d_w_qn", "q_norm.weight", ["Dh"], "weight", "逐头 RMSNorm 权重", ["Dh"]),
        _tensor("d_w_k", "k_proj.weight", ["Hkv", "H"], "weight", "Linear(4096→1024)，无 bias",
                ["Hkv", "H"]),
        _tensor("d_w_kn", "k_norm.weight", ["Dh"], "weight", "逐头 RMSNorm 权重", ["Dh"]),
        _tensor("d_w_v", "v_proj.weight", ["Hkv", "H"], "weight", "Linear(4096→1024)，无 bias",
                ["Hkv", "H"]),
        _tensor("d_w_o", "o_proj.weight", ["H", "Hq"], "weight", "Linear(4096→4096)，无 bias",
                ["H", "Hq"]),
        _tensor("d_w_norm2", "post_attention_layernorm.weight", ["H"], "weight",
                "RMSNorm 权重", ["H"]),
        _tensor("d_w_gate", "gate_proj.weight", ["Fi", "H"], "weight",
                "SwiGLU gate（无 bias）", ["Fi", "H"]),
        _tensor("d_w_up", "up_proj.weight", ["Fi", "H"], "weight", "SwiGLU up（无 bias）",
                ["Fi", "H"]),
        _tensor("d_w_down", "down_proj.weight", ["H", "Fi"], "weight", "SwiGLU down（无 bias）",
                ["H", "Fi"]),
    ]

    ops = [
        _op("d_in_norm", "Qwen3VLTextDecoderLayer.input_layernorm", "RMSNorm(hidden_states)",
            "rmsnorm", ["d_hidden_in"], ["d_normed"], weights=["d_w_norm1"],
            parameter_refs=["H", "EpsN"],
            source_refs=[_src("504/519-520", "self.input_layernorm(hidden_states)")],
            note="逐 token RMSNorm(H)，eps=1e-6；不是 LayerNorm，也没有时间步调制"),
        _op("d_q_proj", "Qwen3VLTextAttention.q_proj", "self.q_proj(hidden_states)", "linear",
            ["d_normed"], ["d_q_lin"], weights=["d_w_q"], parameter_refs=["H", "Hq"],
            source_refs=[_src("448", "query_states = self.q_norm(self.q_proj(...).view(...))")],
            note="Linear(4096→4096)，无 bias"),
        _op("d_q_view", "Qwen3VLTextAttention.q_proj", "q.view(B,S,Nh,Dh)", "view",
            ["d_q_lin"], ["d_q_head"], parameter_refs=["B", "Stot", "Nh", "Dh"],
            source_refs=[_src("446-448", "hidden_shape = (*input_shape, -1, head_dim)")],
            note="按头拆成 4D；head_dim=128"),
        _op("d_q_norm", "Qwen3VLTextAttention.q_norm", "self.q_norm(q)", "rmsnorm",
            ["d_q_head"], ["d_q_normed"], weights=["d_w_qn"], parameter_refs=["Dh", "EpsN"],
            source_refs=[_src("430/448", "Qwen3VLTextRMSNorm(head_dim)")],
            note="逐头 RMSNorm(Dh)（Qwen3-VL 的 QK-norm）；对每个头独立归一"),
        _op("d_q_rope", "Qwen3VLTextAttention.forward",
            "apply_rotary_pos_emb(q, cos, sin)", "rope",
            ["d_q_normed", "d_pos_cos", "d_pos_sin"], ["d_q"],
            parameter_refs=["B", "Nh", "Stot", "Dh"],
            source_refs=[_src("452-453", "query_states, key_states = apply_rotary_pos_emb(...)"),
                         _src("378-402", "rotate_half 拼接 (q*cos + rotate_half(q)*sin)")],
            note="MRoPE（interleaved 24/20/20）作用在整 128 维；位置来自三层 position_ids"),
        _op("d_k_proj", "Qwen3VLTextAttention.k_proj", "self.k_proj(hidden_states)", "linear",
            ["d_normed"], ["d_k_lin"], weights=["d_w_k"], parameter_refs=["H", "Hkv"],
            source_refs=[_src("449", "key_states = self.k_norm(self.k_proj(...))")],
            note="Linear(4096→1024)，无 bias；GQA 8 个 KV 头"),
        _op("d_k_view", "Qwen3VLTextAttention.k_proj", "k.view(B,S,Nkv,Dh)", "view",
            ["d_k_lin"], ["d_k_head"], parameter_refs=["B", "Stot", "Nkv", "Dh"],
            source_refs=[_src("446-449", "同一 hidden_shape 规则")],
            note="按 KV 头拆成 4D"),
        _op("d_k_norm", "Qwen3VLTextAttention.k_norm", "self.k_norm(k)", "rmsnorm",
            ["d_k_head"], ["d_k_normed"], weights=["d_w_kn"], parameter_refs=["Dh", "EpsN"],
            source_refs=[_src("431/449", "Qwen3VLTextRMSNorm(head_dim)")],
            note="逐头 RMSNorm(Dh)；与 q_norm 各自独立权重"),
        _op("d_k_rope", "Qwen3VLTextAttention.forward",
            "apply_rotary_pos_emb(k, cos, sin)", "rope",
            ["d_k_normed", "d_pos_cos", "d_pos_sin"], ["d_k"],
            parameter_refs=["B", "Nkv", "Stot", "Dh"],
            source_refs=[_src("452-453", "同一个 position_embeddings 施加到 q 和 k")],
            note="key 侧也做 MRoPE；V 支路不做"),
        _op("d_v_proj", "Qwen3VLTextAttention.v_proj", "self.v_proj(hidden_states)", "linear",
            ["d_normed"], ["d_v_lin"], weights=["d_w_v"], parameter_refs=["H", "Hkv"],
            source_refs=[_src("450", "value_states = self.v_proj(hidden_states).view(...)")],
            note="Linear(4096→1024)，无 bias"),
        _op("d_v_view", "Qwen3VLTextAttention.v_proj", "v.view(B,S,Nkv,Dh).transpose(1,2)",
            "transpose", ["d_v_lin"], ["d_v"], parameter_refs=["B", "Nkv", "Stot", "Dh"],
            source_refs=[_src("450", "value_states.view(hidden_shape).transpose(1, 2)")],
            note="只做 view/transpose；V 不做 norm、不做 RoPE"),
        _op("d_gather_ar", "Qwen3VLModel._run_decoder_flash",
            "q[:, idx_ar] / k[:, idx_ar] / v[:, idx_ar]", "gather",
            ["d_q", "d_k", "d_v", "d_ar_index"], ["d_q_ar", "d_k_ar", "d_v_ar"],
            parameter_refs=["B", "Nh", "Nkv", "Nar", "Dh"],
            source_refs=[_src("1331-1337", "q_ar = q[:, idx_ar]；flash_attn_func(causal=True)")],
            note="条件：use_flash_attn=True（pipeline 固定 True）。第 1 趟只在 AR 行上算"),
        _op("d_qk_ar", "flash_attn_func（第 1 趟）",
            "matmul(q_ar, k_ar^T) * Dh^-0.5", "matmul-scaled",
            ["d_q_ar", "d_k_ar"], ["d_score_ar"], parameter_refs=["B", "Nh", "Nkv", "Nar", "Dh"],
            source_refs=[_src("1336", "flash_attn_func(q_ar, k_ar, v_ar, softmax_scale=Dh^-0.5, causal=True)")],
            note="causal=True：AR 行只看 AR 行（文本优先布局使 AR 行左侧没有生成 token）"),
        _op("d_softmax_ar", "flash_attn_func（第 1 趟）", "softmax(dim=-1)", "softmax",
            ["d_score_ar"], ["d_prob_ar"], parameter_refs=["B", "Nh", "Nar"],
            source_refs=[_src("1336", "flash attention 内核内部 softmax（语义等价于 :175）")],
            note="具名保留 softmax；flash 内核内部实现不改变语义"),
        _op("d_av_ar", "flash_attn_func（第 1 趟）", "matmul(softmax, v_ar)", "matmul",
            ["d_prob_ar", "d_v_ar"], ["d_ctx_ar"], parameter_refs=["B", "Nar", "Nh", "Dh"],
            source_refs=[_src("1336", "第 1 趟返回 out_ar")],
            note="只产出 AR 行的注意力输出"),
        _op("d_qk_full", "flash_attn_func（第 2 趟）",
            "matmul(q, k^T) * Dh^-0.5", "matmul-scaled",
            ["d_q", "d_k"], ["d_score_full"], parameter_refs=["B", "Nh", "Nkv", "Stot", "Dh"],
            source_refs=[_src("1339-1341", "flash_attn_func(q, k, v, causal=False)")],
            note="causal=False：所有行对全部 token 双向；O(S_tot²) 的规模就在这里"),
        _op("d_softmax_full", "flash_attn_func（第 2 趟）", "softmax(dim=-1)", "softmax",
            ["d_score_full"], ["d_prob_full"], parameter_refs=["B", "Nh", "Stot"],
            source_refs=[_src("1339-1341", "第 2 趟内核内部 softmax")],
            note="双向 softmax，没有 mask 加法"),
        _op("d_av_full", "flash_attn_func（第 2 趟）", "matmul(softmax, v)", "matmul",
            ["d_prob_full", "d_v"], ["d_ctx_full"], parameter_refs=["B", "Stot", "Nh", "Dh"],
            source_refs=[_src("1339-1341", "第 2 趟返回 out_full")],
            note="全序列输出；AR 行稍后被第 1 趟结果覆盖"),
        _op("d_merge_ar", "Qwen3VLModel._run_decoder_flash",
            "out_full[:, idx_ar] = out_ar", "index-copy",
            ["d_ctx_full", "d_ctx_ar", "d_ar_index"], ["d_ctx"],
            parameter_refs=["B", "Stot", "Nh", "Nar", "Dh"],
            source_refs=[_src("1343-1345", "out_full = out_full.clone(); out_full[:, idx_ar] = out_ar")],
            note=("条件：use_flash_attn=True。这一步把两趟语义合成一个结果：AR 行=因果、"
                  "生成行=双向。use_flash_attn=False 时同一语义由 4D additive mask 一次算出"
                  "（modeling:1577-1589），不改变连接与 shape")),
        _op("d_head_merge", "Qwen3VLTextAttention.forward",
            "attn_output.reshape(B,S,Hq)", "reshape",
            ["d_ctx"], ["d_attn_flat"], parameter_refs=["B", "Stot", "Hq"],
            source_refs=[_src("1348", "attn_output.reshape(*input_shape, -1)")],
            note="把 32 个头拼回 4096 维"),
        _op("d_o_proj", "Qwen3VLTextAttention.o_proj", "self.o_proj(attn_output)", "linear",
            ["d_attn_flat"], ["d_attn"], weights=["d_w_o"], parameter_refs=["H", "Hq"],
            source_refs=[_src("1349", "attn_output = attn.o_proj(attn_output)")],
            note="Linear(4096→4096)，无 bias；没有输出门/缩放"),
        _op("d_res_attn", "Qwen3VLTextDecoderLayer",
            "residual + attn_output", "add",
            ["d_hidden_in", "d_attn"], ["d_h1"], parameter_refs=["B", "Stot", "H"],
            source_refs=[_src("519-532", "residual = hidden_states; hidden_states = residual + hidden_states")],
            note="普通标量残差相加；没有多路流、没有读门/写门"),
        _op("d_post_norm", "Qwen3VLTextDecoderLayer.post_attention_layernorm",
            "RMSNorm(hidden_states)", "rmsnorm",
            ["d_h1"], ["d_h1n"], weights=["d_w_norm2"], parameter_refs=["H", "EpsN"],
            source_refs=[_src("536", "hidden_states = self.post_attention_layernorm(hidden_states)")],
            note="第二个 RMSNorm 站点"),
        _op("d_gate_proj", "Qwen3VLTextMLP.gate_proj", "self.gate_proj(x)", "linear",
            ["d_h1n"], ["d_gate"], weights=["d_w_gate"], parameter_refs=["H", "Fi"],
            source_refs=[_src("486/491-492", "gate_proj = Linear(hidden, intermediate)")],
            note="Linear(4096→12288)，无 bias"),
        _op("d_up_proj", "Qwen3VLTextMLP.up_proj", "self.up_proj(x)", "linear",
            ["d_h1n"], ["d_up"], weights=["d_w_up"], parameter_refs=["H", "Fi"],
            source_refs=[_src("487/492", "up_proj = Linear(hidden, intermediate)")],
            note="与 gate 并行；两条支路同形"),
        _op("d_silu", "Qwen3VLTextMLP.act_fn", "ACT2FN[hidden_act](gate)", "silu",
            ["d_gate"], ["d_gate_act"], parameter_refs=["B", "Stot", "Fi"],
            source_refs=[_src("489/492", "act_fn = ACT2FN[config.hidden_act]（silu）")],
            note="SiLU 只作用于 gate 支路"),
        _op("d_swiglu_mul", "Qwen3VLTextMLP", "SiLU(gate) * up", "mul",
            ["d_gate_act", "d_up"], ["d_ffn_act"], parameter_refs=["B", "Stot", "Fi"],
            source_refs=[_src("492", "down_proj(act_fn(gate_proj(x)) * up_proj(x))")],
            note="逐元素乘；没有 clamp"),
        _op("d_down_proj", "Qwen3VLTextMLP.down_proj", "self.down_proj(x)", "linear",
            ["d_ffn_act"], ["d_ffn"], weights=["d_w_down"], parameter_refs=["H", "Fi"],
            source_refs=[_src("488/492", "down_proj = Linear(intermediate, hidden)")],
            note="Linear(12288→4096)，无 bias"),
        _op("d_res_ffn", "Qwen3VLTextDecoderLayer",
            "residual + mlp_output", "add",
            ["d_h1", "d_ffn"], ["d_h2"], parameter_refs=["B", "Stot", "H"],
            source_refs=[_src("535-538", "residual = hidden_states; hidden_states = residual + hidden_states")],
            note="第二个残差站点；层输出在此完整"),
        _op("d_deepstack_add", "Qwen3VLTextModel._deepstack_process",
            "hidden_states[visual_pos_masks] += visual_embeds", "scatter-add",
            ["d_h2", "d_visual_mask", "d_ds_embeds"], ["d_hidden_out"],
            parameter_refs=["B", "Stot", "H", "Nvis", "Ndh"],
            source_refs=[_src("938-945", "local_this = hidden_states[visual_pos_masks] + visual_embeds"),
                         _src("913-919", "仅 layer_idx < len(deepstack_visual_embeds) 时执行")],
            note=("条件：参考图/编辑任务（Kref>0）且 layer_idx ∈ {0,1,2}；t2i 或 layer_idx≥3 时 "
                  "d_h2 原样作为输出（等价恒等）。只改条件图占位行，不改其它 token")),
    ]

    return {
        "id": "block-uit-decoder",
        "kind": "uit_decoder",
        "title": ("统一 Transformer Block：RMSNorm → q/k/v（GQA 32/8，逐头 QK-norm）→ "
                  "MRoPE → 两趟注意力（AR causal + 全序列双向）→ o_proj → 残差 → "
                  "RMSNorm → SwiGLU → 残差 → 条件 DeepStack 注入"),
        "layer_index": 0,
        "attention_type": "hybrid_causal_full",
        "moe_type": "dense_swiglu",
        "parameters": params,
        "tensors": tensors,
        "operators": ops,
        "derivations": derivations,
        "cache_notes": [
            "单次 forward 不使用 past_key_values：_forward_generation 调 language_model 时 "
            "use_cache=False（modeling:1603），_run_decoder_flash 直接循环 text_model.layers"
            "（modeling:1368-1378）。",
            "因此层内没有 KV cache 的读取/写回，也没有 cache_position；跨 50 步累积的 z 属于"
            "外部采样状态，不属于网络缓存。",
            "DeepStack 特征由视觉塔在本层之前算好（同一次 forward 内），以 visual_pos_masks "
            "为条件加到层输出；它们不是持久状态。",
            "位置嵌入 cos/sin 在层循环外算一次（modeling:1284-1289），层间共享，不写入任何缓存。",
        ],
        "coverage_notes": [
            "已表达：input_layernorm；q/k/v 投影（4096/1024/1024，无 bias）；q/k 逐头 RMSNorm(128)；"
            "MRoPE 施加到 q/k；GQA 8 KV 头；flash 两趟注意力（AR 子集 causal + 全序列 non-causal + "
            "index_copy）；o_proj；两个残差站点；post_attention_layernorm；SwiGLU（gate/up 12288、"
            "SiLU、逐元素乘、down）；条件 DeepStack 注入。",
            f"32 个 Q 头 / 8 个 KV 头：repeat_kv 由 flash 内核按 GQA 处理（eager 路径见 "
            "modeling:167-168 的 repeat_kv）；结构上 K/V 仍是 8 头，不复制成 32 份权重。",
            "use_flash_attn=False 时同一语义用 4D additive mask 实现（生成行 0、其余上三角 -inf，"
            "modeling:1577-1589）并走 Qwen3VLTextModel.forward；本结构不另建第二套层结构。",
            "未覆盖：时间步如何进入（在 network 层以 tms token 位置替换表达，不在 Block 内）；"
            "DeepStack 特征如何产生（在 vision 层组与 network 层）；bfloat16/FP32 数值差异；"
            "flash-attn 版本差异；torch.compile/梯度检查点；loss/反向；采样调度。",
            "数据相关量（哪些 token 是条件图占位、AR 行集合）保留符号/索引；不根据权重或 prompt "
            "内容推断具体 token。",
        ],
        "scope_notes": [
            "本组是 36 个同构 Qwen3VLTextDecoderLayer 的**一份模板**，真实层索引 0..35；"
            "模板复用不代表共享权重（每层独立实例）。",
            "层内没有 MoE、没有缓存压缩器、没有线性注意力：这是单一 dense 主干，不能套用 "
            "其它文本模型的层型或状态。",
            "时间步不以 adaLN/调制 MLP 进入层内：整网层把 t_emb 写进 tms token 的嵌入，"
            "层内没有额外的时间步通路。",
            "t2i 与参考图任务共用这份层结构；差别只在 DeepStack 注入这一条件算子是否执行。",
        ],
    }


# ---------------------------------------------------------------------------
# 视觉塔 Block（27 blocks，条件执行）
# ---------------------------------------------------------------------------


def _vision_block(values: HfModelValues) -> dict:
    params = [
        _param("Kref", "Kref", "参考图/条件图数量 Kref", value=None,
               note="0=t2i（视觉塔不执行）；>0 时视觉塔与分段注意力执行",
               domain={"kind": "integer", "min": 0}),
        _param("Svis", "Svis", "VLM 条件图 patch token 总数（merge 前，数据相关）", value=None,
               note=("条件图（384 级）按 16×16 patch 编码；Kref=0 时必须为 0，Kref>0 时必须 >0 "
                     "且能被 merge 单元整除"),
               unknown_reason="条件图尺寸未给定（Kref>0 时由实际条件图决定）",
               domain={"kind": "integer", "min": 0},
               consistency=("((Kref > 0) or (Svis == 0)) and "
                            "((Kref == 0) or (Svis > 0)) and "
                            "(Svis % (Mv * Mv) == 0)")),
        _param("S_i", "S_i", "单张条件图的 patch token 数（逐段长度，等长时填同一值）",
               value=None,
               note=("官方按 cu_seqlens 把 q/k/v 逐图切段（modeling:240-259/238-247），每段长度 "
                     "可能不同；本表用 S_i 表示等长假设下的每图长度，并要求 Kref×S_i = Svis。"
                     "不等长时无法用一个 S_i 表示，分段/总注意力规模如实未知"),
               unknown_reason="每图长度分布未给定（不能假设各图等长）",
               domain={"kind": "integer", "min": 0},
               consistency=("((Kref > 0) or (S_i == 0)) and "
                            "((Kref == 0) or (S_i > 0)) and "
                            "(S_i % (Mv * Mv) == 0) and "
                            "(Kref * S_i == Svis)")),
        _param("Mv", "Mv", "视觉塔 spatial_merge_size", value=2,
               source_refs=[_src("595-597", "spatial_merge_unit = merge * merge")]),
        _param("Svism", "S_vis_m", "merge 后条件 token 数（写回文本序列的 placeholder 数）",
               value=None, expression="S_vis_m = Svis // (Mv * Mv)",
               note="spatial_merge_size=2：每 4 个 patch token 合成 1 个（modeling:118-121）",
               source_refs=[_src("108-121", "PatchMerger.hidden_size = hidden * merge²")]),
        _param("Sattn", "S_attn", "逐图注意力 score 元素总数（等长时）", value=None,
               expression="S_attn = Nhv * Kref * S_i * S_i",
               note=("每图独立 [Nhv,S_i,S_i] 后再求和：Σ_i Nhv·S_i²。只有 Kref×S_i=Svis（等长）时"
                     "才能由总长推出；不等长时必须逐图长度，本表不伪造"),
               source_refs=[_src("240-259", "按 cu_seqlens 逐段 attention 后 concat")]),
        _param("Hv", "Hv", "视觉塔 hidden_size", value=1152,
               source_refs=[_src("593-613", "Qwen3VLVisionModel(config)")]),
        _param("Nhv", "Nhv", "视觉塔注意力头数", value=16,
               source_refs=[_src("186-192", "num_heads = config.num_heads；qkv = Linear(dim, dim*3)")]),
        _param("Dhv", "D_hv", "视觉塔 head_dim（Hv//Nhv）", value=72, expression="D_hv = Hv // Nhv",
               note="2D rotary 作用在 head_dim 的前一半（modeling:606-607）",
               source_refs=[_src("606-607", "rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)")]),
        _param("Hv3", "H_v3", "qkv 合并投影宽度（3×Hv）", value=3456, expression="H_v3 = 3 * Hv",
               source_refs=[_src("190", "self.qkv = Linear(dim, dim * 3, bias=True)")]),
        _param("Fv", "Fv", "视觉塔 MLP 中间维", value=4304,
               source_refs=[_src("61-71", "VisionMLP：Linear(hidden, intermediate)")]),
        _param("EpsV", "EpsV", "视觉塔 LayerNorm epsilon", value=1e-6,
               source_refs=[_src("269-270", "nn.LayerNorm(hidden_size, eps=1e-6)")]),
        _param("Pv", "Pv", "视觉 patch 边长（像素）", value=16,
               source_refs=[_src("83", "Conv3d(kernel=[temporal,patch,patch], stride=同)")]),
        _param("Tp", "Tp", "视觉 temporal patch", value=2,
               source_refs=[_src("82-83", "kernel_size=[temporal_patch_size, patch, patch]")]),
        _param("Npm", "Npm", "可学习位置嵌入数（48×48 网格）", value=2304,
               source_refs=[_src("603-604", "nn.Embedding(num_position_embeddings, hidden)")]),
        _param("Nv", "Nv", "视觉塔 block 数", value=27,
               source_refs=[_src("609", "blocks = ModuleList(range(config.depth))")]),
        _param("Ndh", "Ndh", "DeepStack 特征数", value=3,
               note="第 8/16/24 个 block 后各接一个 merger（deepstack=1-based block 序号）",
               source_refs=[_src("615-624", "deepstack_merger_list")]),
    ]
    derivations = [
        {"id": "S_vis_m", "expression": "Svis // (Mv * Mv)",
         "label": "merge 后条件 token 数", "note": "2×2 merge 单元；每个单元 4 个 patch token",
         "guard": "Svis % (Mv * Mv) == 0"},
        {"id": "S_attn_elems", "expression": "Nhv * Kref * S_i * S_i",
         "label": "逐图注意力 score 元素总数（等长时）",
         "note": ("每图独立 [Nhv,S_i,S_i] 后求和；仅在 Kref×S_i = Svis（等长）时成立。"
                  "不等长/分布未知时不给伪造总数"),
         "guard": "(Kref * S_i == Svis) and (S_i % (Mv * Mv) == 0)"},
        {"id": "H_v3", "expression": "3 * Hv", "label": "qkv 宽度", "note": "q/k/v 共享一次 Linear"},
        {"id": "D_hv", "expression": "Hv // Nhv", "label": "视觉塔 head_dim", "note": "72"},
    ]

    tensors = [
        _tensor("v_hidden_in", "hidden_states（视觉塔 block 输入）", ["Svis", "Hv"], "boundary",
                "patch_embed + 位置嵌入之后的状态；patch 顺序按 merge 单元重排", ["Svis", "Hv"]),
        _tensor("v_cu_seqlens", "cu_seqlens（每图边界，一维）", ["Kref+1"], "boundary",
                "grid_thw[:,1]*grid_thw[:,2] 的 cumsum 后在左端 pad 0（modeling:752-760）："
                "一维 [Kref+1]，前缀含 0；差分得到 Kref 段", ["Kref"]),
        _tensor("v_pos_emb", "position_embeddings（cos/sin）", ["Svis", "Dhv"], "boundary",
                "2D rotary（行/列各 Dhv/2 维）在塔入口按 grid 算一次（modeling:744-750）",
                ["Svis", "Dhv"]),
        _tensor("v_normed1", "norm1 输出", ["Svis", "Hv"], "activation",
                "LayerNorm(1152)，不是 RMSNorm", ["Svis", "Hv"]),
        _tensor("v_qkv", "self.qkv(hidden_states)", ["Svis", "Hv3"], "activation",
                "Linear(1152→3456)，bias=True", ["Svis", "Hv3"]),
        _tensor("v_q", "q", ["Nhv", "Svis", "Dhv"], "activation",
                "reshape(S,3,Nhv,Dhv).permute/unbind 后的 query", ["Nhv", "Svis", "Dhv"]),
        _tensor("v_k", "k", ["Nhv", "Svis", "Dhv"], "activation", "同上", ["Nhv", "Svis", "Dhv"]),
        _tensor("v_v", "v", ["Nhv", "Svis", "Dhv"], "activation", "同上；v 不做 rotary",
                ["Nhv", "Svis", "Dhv"]),
        _tensor("v_q_r", "q（rotary 后）", ["Nhv", "Svis", "Dhv"], "activation",
                "apply_rotary_pos_emb_vision（q/k 各自 float 计算后回原 dtype）",
                ["Nhv", "Svis", "Dhv"]),
        _tensor("v_k_r", "k（rotary 后）", ["Nhv", "Svis", "Dhv"], "activation",
                "同 q；cos/sin 只覆盖 head_dim 前一半", ["Nhv", "Svis", "Dhv"]),
        _tensor("v_q_seg", "q 的第 i 段", ["Nhv", "S_i", "Dhv"], "activation",
                "按 cu_seqlens 差分切出第 i 张图的 S_i 个 token；循环 Kref 段", ["Nhv", "S_i", "Dhv"]),
        _tensor("v_k_seg", "k 的第 i 段", ["Nhv", "S_i", "Dhv"], "activation",
                "同 q_seg；段与段之间不互相注意", ["Nhv", "S_i", "Dhv"]),
        _tensor("v_v_seg", "v 的第 i 段", ["Nhv", "S_i", "Dhv"], "activation",
                "同 q_seg", ["Nhv", "S_i", "Dhv"]),
        _tensor("v_scores", "attn_weights（第 i 段）", ["Nhv", "S_i", "S_i"], "activation",
                "段内非因果 QK^T×Dhv^-0.5；总元素 Σ_i S_i²（不是 (Σ_i S_i)²）",
                ["Nhv", "S_i"]),
        _tensor("v_probs", "softmax（第 i 段）", ["Nhv", "S_i", "S_i"], "activation",
                "段内独立 softmax", ["Nhv", "S_i"]),
        _tensor("v_ctx_seg", "AV 输出（第 i 段）", ["S_i", "Nhv", "Dhv"], "activation",
                "段内 AV 后转置回 [S,H,D]", ["S_i", "Nhv", "Dhv"]),
        _tensor("v_ctx", "逐段结果拼接", ["Svis", "Nhv", "Dhv"], "activation",
                "Kref 段按原顺序 concat 回 [Svis,H,D]（modeling:245-259）", ["Svis", "Nhv", "Dhv"]),
        _tensor("v_ctx_flat", "attn_output.reshape(S,-1)", ["Svis", "Hv"], "activation",
                "拼头", ["Svis", "Hv"]),
        _tensor("v_attn", "self.proj(attn_output)", ["Svis", "Hv"], "activation",
                "Linear(1152→1152)，bias=True", ["Svis", "Hv"]),
        _tensor("v_h1", "残差后的 hidden_states", ["Svis", "Hv"], "activation",
                "hidden_states + attn(norm1(hidden_states))（modeling:282-288）", ["Svis", "Hv"]),
        _tensor("v_normed2", "norm2 输出", ["Svis", "Hv"], "activation", "LayerNorm(1152)",
                ["Svis", "Hv"]),
        _tensor("v_fc1", "linear_fc1 输出", ["Svis", "Fv"], "activation",
                "Linear(1152→4304)，bias=True", ["Svis", "Fv"]),
        _tensor("v_act", "GELU(tanh) 输出", ["Svis", "Fv"], "activation",
                "hidden_act=gelu_pytorch_tanh", ["Svis", "Fv"]),
        _tensor("v_fc2", "linear_fc2 输出", ["Svis", "Hv"], "activation",
                "Linear(4304→1152)，bias=True", ["Svis", "Hv"]),
        _tensor("v_out", "block 输出", ["Svis", "Hv"], "activation",
                "hidden_states + mlp(norm2(hidden_states))（modeling:289）", ["Svis", "Hv"]),
        _tensor("v_tap", "DeepStack tap（条件读取）", ["Svis", "Hv"], "activation",
                "block 8/16/24 之后的 hidden_states 也送入对应 merger（modeling:770-774）",
                ["Svis", "Hv"]),
        _tensor("v_w_qkv", "qkv.weight", ["Hv3", "Hv"], "weight", "Linear bias=True → 另有 b_qkv [Hv3]",
                ["Hv3", "Hv"]),
        _tensor("v_b_qkv", "qkv.bias", ["Hv3"], "weight", "bias=True", ["Hv3"]),
        _tensor("v_w_proj", "proj.weight", ["Hv", "Hv"], "weight", "Linear bias=True", ["Hv", "Hv"]),
        _tensor("v_b_proj", "proj.bias", ["Hv"], "weight", "bias=True", ["Hv"]),
        _tensor("v_w_fc1", "mlp.linear_fc1.weight", ["Fv", "Hv"], "weight", "bias=True",
                ["Fv", "Hv"]),
        _tensor("v_w_fc2", "mlp.linear_fc2.weight", ["Hv", "Fv"], "weight", "bias=True",
                ["Hv", "Fv"]),
        _tensor("v_w_norm1", "norm1.weight", ["Hv"], "weight", "LayerNorm 权重（另有 bias [Hv]）",
                ["Hv"]),
        _tensor("v_w_norm2", "norm2.weight", ["Hv"], "weight", "LayerNorm 权重（另有 bias [Hv]）",
                ["Hv"]),
    ]

    ops = [
        _op("v_norm1", "Qwen3VLVisionBlock.norm1", "self.norm1(hidden_states)", "layernorm",
            ["v_hidden_in"], ["v_normed1"], weights=["v_w_norm1"], parameter_refs=["Svis", "Hv", "EpsV"],
            source_refs=[_src("269/282-284", "norm1 = nn.LayerNorm(hidden_size, eps=1e-6)")],
            note="LayerNorm（均值+方差），与文本主干的 RMSNorm 不同"),
        _op("v_qkv_proj", "Qwen3VLVisionAttention.qkv", "self.qkv(hidden_states)", "linear",
            ["v_normed1"], ["v_qkv"], weights=["v_w_qkv", "v_b_qkv"],
            parameter_refs=["Svis", "Hv", "Hv3"],
            source_refs=[_src("190/206-207", "qkv = Linear(dim, dim*3, bias=True)")],
            note="q/k/v 合并一次投影（bias=True），输出 3456 维"),
        _op("v_qkv_split", "Qwen3VLVisionAttention.forward",
            "reshape(S,3,Nhv,Dhv).permute(1,0,2,3).unbind(0)", "view-split",
            ["v_qkv"], ["v_q", "v_k", "v_v"], parameter_refs=["Svis", "Nhv", "Dhv"],
            source_refs=[_src("206-208", "reshape(seq_length, 3, num_heads, -1).permute(1,0,2,3).unbind(0)")],
            note="一次 view+permute 拆成 q/k/v 三块，不是三个独立投影"),
        _op("v_q_rope", "apply_rotary_pos_emb_vision", "apply_rotary_pos_emb_vision(q, cos, sin)",
            "rope", ["v_q", "v_pos_emb"], ["v_q_r"], parameter_refs=["Nhv", "Svis", "Dhv"],
            source_refs=[_src("210", "query_states, key_states = apply_rotary_pos_emb_vision(...)"),
                         _src("131-142", "float 计算 cos/sin 后回原 dtype")],
            note="2D rotary（行/列各 36 维）；作用在 head_dim 的整 72 维上"),
        _op("v_k_rope", "apply_rotary_pos_emb_vision", "apply_rotary_pos_emb_vision(k, cos, sin)",
            "rope", ["v_k", "v_pos_emb"], ["v_k_r"], parameter_refs=["Nhv", "Svis", "Dhv"],
            source_refs=[_src("210", "同一个 position_embeddings 施加到 q 和 k")],
            note="v 不做 rotary；cos/sin 由 grid 的行列坐标构造（modeling:628-665）"),
        _op("v_seg_split", "Qwen3VLVisionAttention.forward",
            "torch.split(q, lengths, dim=2) / 按 cu_seqlens 逐段切分", "segment-split",
            ["v_q_r", "v_k_r", "v_v", "v_cu_seqlens"], ["v_q_seg", "v_k_seg", "v_v_seg"],
            parameter_refs=["Kref", "Svis", "S_i", "Nhv", "Dhv"],
            source_refs=[_src("239-259", "lengths = cu_seqlens[1:] - cu_seqlens[:-1]；splits = torch.split(...)"),
                         _src("240-243", "按每段长度切 q/k/v，逐段独立 attention")],
            note=("按 cu_seqlens 的差分长度切段：循环边界 Kref 段（Kref），第 i 段长度 S_i；"
                  "段内 token 数不要求相等。本结构不把不同图拼成一个全连接矩阵")),
        _op("v_qk", "Qwen3VLVisionAttention.forward",
            "matmul(q_i, k_i^T) * Dhv^-0.5", "matmul-scaled",
            ["v_q_seg", "v_k_seg"], ["v_scores"], parameter_refs=["Nhv", "S_i", "Dhv"],
            source_refs=[_src("170", "scaling = head_dim**-0.5"),
                         _src("245-259", "逐段 attn_outputs.append(attention_interface(q,k,v))")],
            note=("逐段非因果 QK：第 i 段 [Nhv,S_i,S_i]；总元素 Σ_i Nhv·S_i²，"
                  "不是 Nhv·(Σ_i S_i)²。等长且 Kref×S_i=Svis 时总规模才可由输入推出")),
        _op("v_softmax", "eager_attention_forward", "softmax(dim=-1)", "softmax",
            ["v_scores"], ["v_probs"], parameter_refs=["Nhv", "S_i"],
            source_refs=[_src("175", "nn.functional.softmax(attn_weights, dim=-1, dtype=fp32)")],
            note="逐段 softmax；没有 mask 加法（非因果、无 padding）"),
        _op("v_av", "eager_attention_forward", "matmul(softmax_i, v_i)", "matmul",
            ["v_probs", "v_v_seg"], ["v_ctx_seg"], parameter_refs=["Nhv", "S_i", "Dhv"],
            source_refs=[_src("177-178", "matmul(attn_weights, value_states).transpose(1, 2)")],
            note="逐段 AV，输出 [S_i,H,D]；flash 路径同样按段分块"),
        _op("v_seg_concat", "Qwen3VLVisionAttention.forward",
            "torch.cat(attn_outputs, dim=1)", "segment-concat",
            ["v_ctx_seg"], ["v_ctx"], parameter_refs=["Kref", "Svis", "S_i", "Nhv", "Dhv"],
            source_refs=[_src("259", "attn_output = torch.cat(attn_outputs, dim=1)")],
            note="Kref 段按原顺序拼回总长 Svis；段间没有信息交换"),
        _op("v_head_merge", "Qwen3VLVisionAttention.forward", "reshape(seq_length, -1)", "reshape",
            ["v_ctx"], ["v_ctx_flat"], parameter_refs=["Svis", "Hv"],
            source_refs=[_src("261", "attn_output.reshape(seq_length, -1)")],
            note="16 个头拼回 1152 维"),
        _op("v_proj", "Qwen3VLVisionAttention.proj", "self.proj(attn_output)", "linear",
            ["v_ctx_flat"], ["v_attn"], weights=["v_w_proj", "v_b_proj"],
            parameter_refs=["Svis", "Hv"],
            source_refs=[_src("191/262", "proj = Linear(dim, dim)")],
            note="Linear(1152→1152)，bias=True"),
        _op("v_res1", "Qwen3VLVisionBlock", "hidden_states + attn(...)", "add",
            ["v_hidden_in", "v_attn"], ["v_h1"], parameter_refs=["Svis", "Hv"],
            source_refs=[_src("282-288", "hidden_states = hidden_states + self.attn(...)")],
            note="普通标量残差"),
        _op("v_norm2", "Qwen3VLVisionBlock.norm2", "self.norm2(hidden_states)", "layernorm",
            ["v_h1"], ["v_normed2"], weights=["v_w_norm2"], parameter_refs=["Svis", "Hv", "EpsV"],
            source_refs=[_src("270/289", "norm2 = nn.LayerNorm(hidden_size, eps=1e-6)")],
            note="MLP 前的 LayerNorm"),
        _op("v_fc1", "Qwen3VLVisionMLP.linear_fc1", "self.linear_fc1(hidden_state)", "linear",
            ["v_normed2"], ["v_fc1"], weights=["v_w_fc1"], parameter_refs=["Svis", "Hv", "Fv"],
            source_refs=[_src("66/71", "linear_fc1 = Linear(hidden, intermediate, bias=True)")],
            note="Linear(1152→4304)，bias=True"),
        _op("v_gelu", "Qwen3VLVisionMLP.act_fn", "ACT2FN[hidden_act](x)", "gelu-tanh",
            ["v_fc1"], ["v_act"], parameter_refs=["Svis", "Fv"],
            source_refs=[_src("68/71", "act_fn = ACT2FN[config.hidden_act]（gelu_pytorch_tanh）")],
            note="GELU（tanh 近似），不是 SiLU/SwiGLU"),
        _op("v_fc2", "Qwen3VLVisionMLP.linear_fc2", "self.linear_fc2(act)", "linear",
            ["v_act"], ["v_fc2"], weights=["v_w_fc2"], parameter_refs=["Svis", "Hv", "Fv"],
            source_refs=[_src("67/71", "linear_fc2 = Linear(intermediate, hidden, bias=True)")],
            note="Linear(4304→1152)，bias=True"),
        _op("v_res2", "Qwen3VLVisionBlock", "hidden_states + mlp(...)", "add",
            ["v_h1", "v_fc2"], ["v_out"], parameter_refs=["Svis", "Hv"],
            source_refs=[_src("289", "hidden_states = hidden_states + self.mlp(self.norm2(...))")],
            note="第二个残差站点；block 输出"),
        _op("v_deepstack_tap", "Qwen3VLVisionModel.forward",
            "if layer_num in deepstack_visual_indexes: merger(hidden_states)", "identity-read",
            ["v_out"], ["v_tap"], parameter_refs=["Svis", "Hv", "Ndh"],
            source_refs=[_src("770-774", "if layer_num in self.deepstack_visual_indexes: ...")],
            note=("条件：block 序号 ∈ {8,16,24}（0-based）。v_out 本身继续流向下一 block；"
                  "tap 只是同一状态被 merger 再读一次，不改变主线")),
    ]

    return {
        "id": "block-vit-tower",
        "kind": "vit_tower",
        "title": ("视觉塔 Block：LayerNorm → qkv（3456，bias）→ 2D rotary → 按 cu_seqlens "
                  "逐图分段（QK/softmax/AV，局部 [Nhv,S_i,S_i]）→ 拼接 → proj → 残差 → "
                  "LayerNorm → GELU-tanh MLP（4304）→ 残差；block 8/16/24 后另接 DeepStack merger"),
        "layer_index": 0,
        "attention_type": "bidirectional_vision",
        "moe_type": "dense_gelu_mlp",
        "parameters": params,
        "tensors": tensors,
        "operators": ops,
        "derivations": derivations,
        "cache_notes": [
            "视觉塔没有 KV cache：27 个 block 在一次 forward 内顺序执行，attention "
            "输出即用即弃（modeling:762-778）。",
            "逐图分段注意力没有跨调用状态：cu_seqlens 与每段长度来自本次 grid_thw，"
            "不写入缓存。",
            "DeepStack merger 的输出（3 个 [N_vis,Hv→H] 特征）在同一次 forward 内被 decoder "
            "前 3 层使用；它们不是跨调用状态。",
            "位置嵌入（可学习插值 + 2D rotary）在塔入口算一次（modeling:741-750），"
            "层间共享。",
        ],
        "coverage_notes": [
            "已表达：LayerNorm(q) → qkv 合并投影（bias）→ view/permute 拆分 → q/k 2D rotary → "
            "按 cu_seqlens 逐图切段（q/k/v 第 i 段 [Nhv,S_i,Dhv]）→ 段内非因果 QK/softmax/AV → "
            "逐段结果 concat → 拼头 → proj → 残差 → LayerNorm → fc1/GELU-tanh/fc2 → 残差；"
            "block 8/16/24 的 DeepStack tap。",
            "注意力规模按每段计算：score 张量是 [Nhv,S_i,S_i]，总元素 Σ_i Nhv·S_i²。"
            "只有每图等长（Kref×S_i=Svis）时才由输入推出总规模；不等长时保留 S_i 符号/未知，"
            "不把不同图当成一个全连接矩阵。",
            "每张条件图的 token 数 S_i 是数据相关量；缺分布/非法/与总长不一致时依赖量未知，"
            "恢复合法值后重算。",
            "未覆盖：flash-attention 版本的视觉注意力（配置 _attn_implementation 决定，语义同 eager）、"
            "bfloat16 数值、训练期 dummy 视觉前向（inference 的 no_grad 路径不执行）。",
            "视觉塔只在 Kref>0（参考图/编辑/IP）时执行；t2i 路径没有 pixel_values，"
            "本组整体不执行（不是把 block 结构删掉）。",
        ],
        "scope_notes": [
            "本组是 27 个同构 Qwen3VLVisionBlock 的**一份模板**，block 索引 0..26；"
            "与 decoder 的 36 层是两个独立索引空间，不能把 ViT block 当成 decoder 的第 36..62 层。",
            "执行条件：参考图/编辑/IP 任务（Kref>0）。t2i 时视觉塔不执行。",
            "每层独立实例；模板复用不代表共享权重。",
        ],
    }


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------


def group_by_id(values: HfModelValues, group_id: str) -> dict:
    for group in layer_group_assignments(values):
        if group["id"] == group_id:
            return group
    raise ValueError(f"HiDream-O1-Image 没有层组 {group_id!r}")


def build_block_structure(values: HfModelValues, group_id: str) -> dict:
    if group_id == "uit-decoder":
        return _decoder_block(values)
    if group_id == "vit-tower":
        return _vision_block(values)
    raise ValueError(f"未知 HiDream-O1-Image 层组 {group_id!r}")


def build_layer_structure(workspace_root: str | Path, group_id: str,
                          model_key: str = "hidream_o1_image_hf") -> dict:
    values = load_hf_model_values(workspace_root, model_key=model_key)
    block = build_block_structure(values, group_id)
    return structure_document(values, block, scenarios=HIDREAM_SCENARIOS,
                              bind_keys=HIDREAM_BIND_KEYS)


def _layer_document(structure: dict, block: dict, group: dict) -> dict:
    """单次 forward 的层文档：不投影 Prefill/Decode 两阶段，只有一个 generation 单元。"""
    return {
        "schema": DOCUMENT_SCHEMA,
        "model": structure["model"],
        "scope": {
            "analysis": "single_forward",
            "phases": ["generation"],
            "group_id": group["id"],
            "layer_index": block.get("layer_index"),
            "attention_type": block.get("attention_type"),
            "moe_type": block.get("moe_type"),
            "execution_condition": (
                "总是执行" if group["id"] == "uit-decoder"
                else "仅参考图/编辑/IP 任务（Kref>0）"),
            "notes": block.get("scope_notes", []),
        },
        "block": block,
        "scenario_derivations": structure["scenario_derivations"],
    }


def build_layer_documents(workspace_root: str | Path, layer_index: int | None = None,
                          model_key: str = "hidream_o1_image_hf",
                          group_id: str | None = None):
    """按层组构建（structure, document）；同组只建一份模板。

    ``group_id`` 优先；只给 ``layer_index`` 时按真实层索引反查层组（两个层组
    各自从 0 开始，交由调用方显式给 group_id，不在这里猜）。
    """
    values = load_hf_model_values(workspace_root, model_key=model_key)
    if group_id is None:
        candidates = [g for g in layer_group_assignments(values)
                      if layer_index in g["layer_indices"]]
        if len(candidates) != 1:
            raise ValueError(
                "HiDream-O1-Image 的 layer_index 同时属于多个层组（decoder 与 ViT 索引空间"
                "各自从 0 开始），必须显式给出 group_id")
        group = candidates[0]
    else:
        group = group_by_id(values, group_id)
    block = build_block_structure(values, group["id"])
    structure = structure_document(values, block, scenarios=HIDREAM_SCENARIOS,
                                   bind_keys=HIDREAM_BIND_KEYS)
    return structure, _layer_document(structure, block, group)
