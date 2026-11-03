"""Qwen3.8-Flash-Next（Qwen/Qwen3.8-Flash-Next, HF transformers ``qwen4_exp``）
语言主干 Block 结构提取。

从 ``rules/qwen38flash_hf.py`` 解释的文本有效配置与本地官方 Transformers 检出源码，
构建**一份** 参数化 Block 结构（``hf_layer_structure.v1``），供分层浏览页与同源
工作簿使用；同一份结构再由 ``extract/hf_scenario.py`` 投影成既有
``hf_decoder_reading.v1`` 的 prefill/decode 场景。本模块不生成渲染文件、不执行
模型、不加载权重、不下载权重。

本配置只存在三种真实层型组合（不虚构第四种，也没有 dense MLP 层）：

- ``gdn-moe``：Gated DeltaNet 线性注意力 + TopK MoE（35 层）；
- ``gdn-ple-moe``：Gated DeltaNet + N-gram PLE + TopK MoE（layer_idx=1，第 2 层）；
- ``qsa-moe``：Qwen Sparse Attention（QSA） + TopK MoE（3,7,…,47，共 12 层）。

结构事实（行号按本地检出 536ecc00…，文件 modeling_qwen4_exp.py，下称 modeling；
配置规则见 configuration_qwen4_exp.py，下称 configuration）：

- DecoderLayer（modeling:1249-1301）**不是普通残差**：每层有
  ``attn_hyper_connection`` 与 ``mlp_hyper_connection`` 两个 Gated Residual 站点。
  站点把 4 路流分组 RMSNorm 后，经 10240→320→10240 + sigmoid 得到**逐元素读门**
  [B,S,4,H]，四路加权 mean 成单流进入子层；``block_inject_weight`` 10240→4 得到
  **每支路标量写门** 2×sigmoid(·/4)；子层输出按写门展开后加到**原 4 路流**。
  网络入口 ``hidden_states.repeat(1,1,hc_count)`` 把 embedding 复制成 4 路
  （modeling:1471）；末端 ``hyper_connection_mixer`` 用同一读门结构加权 mean 成
  单流（modeling:1484），**没有 Final RMSNorm**。
- Gated DeltaNet（modeling:456-616）：``in_proj_qkv`` 2560→10240 → depthwise 因果
  短卷积（kernel 4，conv state [B,10240,4]）→ split q/k [B,S,16,128]、v
  [B,S,48,128] → q/k repeat_interleave(3) 到 48 头 → beta=sigmoid、
  g=-exp(A_log)*softplus(a+dt_bias) → L2Norm(q/k) + q/√128 → gated delta-rule
  （chunk 64 / recurrent 单 token；状态 [B,48,128,128] 固定）→ Gated RMSNorm
  （activation=sigmoid）→ out_proj 6144→2560。状态不随 kvcachelen 增长。
- QSA（modeling:810-892）：q_proj 12288 → 按 head chunk 成 q 与 gate（24×256 each）；
  q/k RMSNorm(256)；partial RoPE 只转前 64 维；KV 2 头（GQA repeat 12）；
  o_proj 6144→2560；输出乘 sigmoid(gate)。
- QSA indexer（modeling:664-770）：``index_qk_proj`` 2560→640；4 个 indexer query
  头 + 1 个共享 key 头；q/k RMSNorm(128)；每 4 个连续可见 token 取均值压成
  micro-block，block key 在组首位置做前 64 维 RoPE；分数 =
  relu(q·k).sum(-1)/√128（4 头求和，不是 softmax/加权）；每 query 选
  min(512, 完整块数) 块 + 不完整 tail（≤3），pad 到 2051（-1），scatter 成
  True=可见的 additional mask，再与 causal mask 相加/相与。
- 注意力掩码：网络入口同时构建 ``create_causal_mask``（4D float additive，
  QSA 用）与 ``create_recurrent_attention_mask``（2D padding，GDN 用）
  （modeling:1446-1460）。QSA 的 selected mask 与 causal mask 相加（modeling:848-851）。
- N-gram / PLE（modeling:1071-1246）：layer_idx=1 的 ``ple`` 子模块先于
  ``attn_hyper_connection``：上一 token 上下文 [B,2]（state_idx=2）+
  dilation-3 depthwise 短卷积（state 9 步，state_idx=1）；16 个哈希头
  （bigram 8 + trigram 8）×160 维；key 2560→10240、value 2560→2560、
  query 为当前 4 路流；gate=(key_normed*query_normed).sum(-1)/√H、
  abs().clamp_min(1e-6).sqrt()*sign、sigmoid(gate)*value；输出加回 4 路流。
- MoE（modeling:951-991）：router softmax 512 选 10，top-10 归一化；没有
  correction bias、没有 routed scaling factor；专家 gate_up [512,2×640,2560]、
  down [512,2560,640]；共享专家 2560→640→2560，另有 sigmoid 标量门；SwiGLU 无 clamp。
- 缓存（cache_utils:1067-1149/331-407）：GDN 层 → LinearAttentionLayer
  （conv/recurrent 状态，固定）；QSA 层 → DynamicIndexedLayer（K/V [B,2,L,256] +
  indexer keys [B,L,128]）。``number_of_conv_states=3`` 因 PLE 开启，但只有
  layer 1 使用 state_idx=1/2。

场景（seqlen/kvcachelen）只绑定取值：结构 shape 保留符号，空缓存 Prefill 的
kvcachelen=0；Decode 场景 seqlen=1。数据相关量（命中专家数、每专家 token 数、
具体选中的块/token）保留未知，不默认 0。
"""

from __future__ import annotations

import math
from pathlib import Path

from structure_reader.extract.hf_scenario import STRUCTURE_SCHEMA
from structure_reader.rules.qwen38flash_hf import (
    MODEL_REPOSITORY,
    QWEN38FLASH_SCENARIOS,
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

SAMPLE_LAYER_INDEX = 0  # 第一组 gdn-moe（其余组由层组容器逐组构建）
SCHEMA = "hf_decoder_reading.v1"

_EXPR_LKV = "kvcachelen + seqlen"
_EXPR_LIDX = "kvcachelen + seqlen"
_EXPR_NPOOL = "L_idx // Kc"
_EXPR_BSEL = "min(Btop, N_pool)"
_EXPR_VQ = "kvcachelen + j + 1"
_EXPR_NQ = "V_q // Kc"
_EXPR_BQ = "min(Btop, N_q)"
_EXPR_RTAIL = "V_q % Kc"
_EXPR_KSEL = "min(V_q, Ki + (V_q % Kc))"
_EXPR_KSEL_LAST = "min(L_idx, Ki + (L_idx % Kc))"
_EXPR_KOUT = "Ki + Kc - 1"
_EXPR_TTOK = "B * seqlen"

_SCOPE_NOTES = [
    "范围：单个 decoder Block（同一层组只保留一份模板；真实层索引、顺序与循环次数见整网层组）。",
    "实现路径：transformers eager（本地 qwen4_exp 实现；_supports_flash_attn=False、"
    "_supports_sdpa=True，modeling:1311-1312）。QSA 的 indexer mask 与 eager/sdpa 共用"
    "同一份 selected token 语义；本图表达 eager 的 additive mask 合并。",
    "静态结构一致为目标：不加载权重、不运行模型、不做数值等价；shape 保留符号 "
    "seqlen/kvcachelen，场景预设只绑定取值，不改变结构。",
    "同一权重/激活张量可被多个算子引用（共享引用非重复定义）；缓存调用前/调用后以两个"
    "张量分别表达，不混为原地更新。",
    "本模型是多模态仓库中的**语言主干**；视觉/图像输入路径不在本票范围，本结构不含图像编码器。",
    "版本口径：根配置记录 transformers_version=5.8.0.dev0，但 qwen4_exp 在 v5.8.0 中"
    "不存在；本票以本地官方检出 5.19.0.dev0 的实际代码为唯一实现版本（见 rules 说明）。",
]


class ReadingDocError(ValueError):
    """阅读数据构建失败（配置/范围/事实来源不一致），具名拒绝。"""


def _block_builder(values: HfModelValues, layer_index: int) -> "_BlockBuilder":
    attention_type = values.layer_type(layer_index)
    if attention_type == "linear_attention":
        return _GdnBlockBuilder(values, layer_index, ple=values.is_ple_layer(layer_index))
    if attention_type == "indexed_attention":
        if values.is_ple_layer(layer_index):
            raise ReadingDocError("PLE 不能出现在 QSA 层（配置类 validate_architecture）")
        return _QsaBlockBuilder(values, layer_index)
    raise ReadingDocError(
        f"layer_index={layer_index} 是 {attention_type}；"
        "Qwen3.8-Flash-Next 已核实的层型只有 linear_attention（GDN）与 "
        "indexed_attention（QSA），不虚构其它层型"
    )


def build_block_structure(values: HfModelValues, layer_index: int) -> dict:
    """构建**一份** Block 结构（参数/张量/算子/缓存注记/派生量），不含阶段复制。"""
    if not 0 <= layer_index < values.num_hidden_layers:
        raise ReadingDocError(f"layer_index {layer_index} 越界（0..{values.num_hidden_layers - 1}）")
    builder = _block_builder(values, layer_index)
    block = builder.build()
    block["layer_index"] = layer_index
    block["attention_type"] = values.layer_type(layer_index)
    block["moe_type"] = "sparse"
    block["ple"] = values.is_ple_layer(layer_index)
    block["kind"] = builder.kind
    return block


def build_layer_documents(
    workspace_root: str | Path,
    layer_index: int = SAMPLE_LAYER_INDEX,
    model_key: str = "qwen_3_8_flash_next_hf",
) -> tuple[dict, dict]:
    """一次构建，两种同源表示：一份 Block 结构 + prefill/decode 场景投影。"""
    from structure_reader.extract.hf_scenario import project_legacy_phase, structure_document

    values = load_hf_model_values(workspace_root, model_key=model_key)
    block = build_block_structure(values, layer_index)
    block["scope_notes"] = list(_SCOPE_NOTES)
    structure = structure_document(values, block, scenarios=QWEN38FLASH_SCENARIOS)
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
            "ple": block["ple"],
            "analysis_path": "eager",
            "notes": [
                f"样板范围：单个 decoder 层（layer_idx={layer_index}，"
                f"{'Gated DeltaNet 线性注意力' if block['kind'] == 'gdn' else 'Qwen Sparse Attention'} "
                f"+ TopK MoE{' + N-gram PLE' if block['ple'] else ''}）；整网 "
                "Embedding/4 路 Gated Residual/48 层/hyper_connection_mixer/lm_head "
                "见整网总览。",
                "模型身份：Qwen/Qwen3.8-Flash-Next revision "
                f"{MODEL_REPOSITORY['revision']}，model_type=qwen4_exp"
                "（text_config=qwen4_exp_text）；"
                f"配置保存的 transformers_version={values.checkpoint_transformers_version}，"
                "实现取本地官方检出 5.19.0.dev0（qwen4_exp 首次随 v5.17.0 发布）。",
                "实现口径：只表达本地 5.19 的 Gated DeltaNet/QSA micro-block/4 路 "
                "Gated Residual/PLE 语义；不与 Qwen3-Next/Qwen3.5/Qwen3-MoE 或其它 Qwen "
                "版本混用注意力/缓存/残差结论。",
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
    model_key: str = "qwen_3_8_flash_next_hf",
) -> dict:
    """构建 ``hf_layer_structure.v1``：一份 Block 结构 + 场景预设/派生值。"""
    return build_layer_documents(workspace_root, layer_index, model_key)[0]


def build_reading_document(
    workspace_root: str | Path,
    layer_index: int = SAMPLE_LAYER_INDEX,
    model_key: str = "qwen_3_8_flash_next_hf",
) -> dict:
    """既有冻结接口 ``hf_decoder_reading.v1``：同一份结构的 prefill/decode 投影。"""
    return build_layer_documents(workspace_root, layer_index, model_key)[1]


def _nth_prime_after(start: int, count: int) -> int:
    """与 modeling:1051-1068 同义：从 start 之后数 count 个素数。"""
    def is_prime(value: int) -> bool:
        if value < 2:
            return False
        if value % 2 == 0:
            return value == 2
        for divisor in range(3, math.isqrt(value) + 1, 2):
            if value % divisor == 0:
                return False
        return True

    prime = start
    for _ in range(count):
        prime += 1
        while not is_prime(prime):
            prime += 1
    return prime


def _ngram_vocab_layout(values: HfModelValues) -> tuple[list[int], list[int], int]:
    """16 个哈希头的素数码表大小/偏移与 pad 后总词表（静态复算，不读权重）。"""
    sizes: list[int] = []
    offsets: list[int] = []
    total = 0
    for head_idx in range(values.ngram_heads):
        size = _nth_prime_after(values.ngram_vocab_size_base - 1, head_idx + 1)
        sizes.append(size)
        offsets.append(total)
        total += size
    divisor = values.ngram_vocab_divisor
    padded = math.ceil(total / divisor) * divisor
    return sizes, offsets, padded


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
        return {"path": SOURCE_FILES["modeling_qwen4_exp.py"],
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
        src = self.src
        self.param("B", "B", "batch 批大小", None, None,
                   "调用输入维度，静态分析保留符号", [])
        self.param("seqlen", "seqlen", "本次输入的新 token 数（seqlen）", None, None,
                   "Prefill 为提示长度、Decode 为 1；结构保留符号，场景预设只绑定取值", [])
        self.param("H", "H", "hidden_size 隐层维度", v.hidden_size, None,
                   "config.hidden_size=2560", [cfg("text_config.hidden_size", "文本子配置字段")])
        self.param("Hc", "Hc", "hc_count Gated Residual 流数", v.hc_count, None,
                   "config.hc_count=4；网络入口 repeat 成 4 路，层内两站点读门/写门混合，"
                   "末端 mixer 加权 mean",
                   [cfg("text_config.hc_count", "文本子配置字段"),
                    src("994-1022", "Qwen4ExpTextGatedResidual"),
                    src("1471", "hidden_states.repeat(1, 1, hc_count)")])
        self.param("Rhc", "R_hc", "hc_lowrank 读门低秩维", v.hc_lowrank, None,
                   "config.hc_lowrank=320；读门 10240→320→10240，不是多系数投影",
                   [cfg("text_config.hc_lowrank", "文本子配置字段"),
                    src("1001-1002", "input_mix_weight_down/up = Linear(10240,320) / Linear(320,10240)")])
        self.param("E", "E", "num_experts 路由专家总数", v.num_experts, None,
                   "config.num_experts=512",
                   [cfg("text_config.num_experts", "文本子配置字段")])
        self.param("K", "K", "num_experts_per_tok 每 token 选择数", v.num_experts_per_tok, None,
                   "config.num_experts_per_tok=10；与专家总数 512 不同参数",
                   [cfg("text_config.num_experts_per_tok", "文本子配置字段")])
        self.param("Es", "Es", "moe_intermediate_size 单专家中间维", v.moe_intermediate_size, None,
                   "config.moe_intermediate_size=640",
                   [cfg("text_config.moe_intermediate_size", "文本子配置字段")])
        self.param("Ess", "Es_shared", "shared_expert_intermediate_size 共享专家中间维",
                   v.shared_expert_intermediate_size, None,
                   "config.shared_expert_intermediate_size=640；与 routed 专家同宽但独立权重",
                   [cfg("text_config.shared_expert_intermediate_size", "文本子配置字段")])
        self.param("EpsN", "EpsN", "RMSNorm epsilon", v.rms_norm_eps, None,
                   "config.rms_norm_eps=1e-6；所有 Qwen4ExpTextRMSNorm / Gated RMSNorm 共用",
                   [cfg("text_config.rms_norm_eps", "文本子配置字段"),
                    src("143-167", "Qwen4ExpTextRMSNorm"),
                    src("171-187", "Qwen4ExpTextRMSNormGated")])
        self.param("Ttok", "T_tok", "本调用展平 token 数（MoE 输入行数）", None, None,
                   "T_tok = B*seqlen；MoE 把 [B,seqlen,H] 展平成 [Ttok,H] 再路由",
                   [src("982", "hidden_states.view(-1, hidden_dim)")])
        self.param("N_hit", "N_hit", "路由实际命中的不同专家数（数据相关）", None, None,
                   "数据相关量，未给定；不根据权重推断 MoE 实际路由，不默认 0 或 E",
                   [src("932-934", "one_hot(...).sum→nonzero：命中专家集合")],
                   unknown_reason="数据相关：路由命中集合未给定；不默认 0/E")
        self.param("N_e", "N_e", "单专家在本调用处理的 token 数（数据相关）", None, None,
                   "数据相关量，未给定；专家模板只有一份，不按 E 复制行",
                   [src("940", "torch.where(expert_mask[expert_idx])")],
                   unknown_reason="数据相关：每专家 token 数未给定；不默认 0")

    def _build_derivations(self) -> None:
        premise = ("按固定配置 + 「空态起步、连续正常执行」前提推导；"
                   "不覆盖裁剪/回滚后的自定义缓存快照")
        self.derivation("T_tok", "MoE 路由的展平 token 数", _EXPR_TTOK,
                        f"B×seqlen；B 为可填性能配置。{premise}")

    # ------------------------------------------------------------------
    # Gated Residual（hyper-connections）
    # ------------------------------------------------------------------

    def _gated_residual_site(self, site: str, streams_in: str) -> None:
        """一个 Gated Residual 读门站点：4 路流 → 逐元素读门 + 标量写门 + 单流。"""
        v = self.values
        p = f"g_hc_{site}"
        src = self.src
        self.weight(f"{p}_norm_weight", f"self.{site}_hyper_connection.hc_norm.weight",
                    ["Hc×H"], ["Hc", "H"],
                    "分组 RMSNorm（group_size=H）的 weight；每个 2560 维流独立归一，"
                    "再乘 1+weight（Qwen4ExpTextRMSNorm 的 weight 初始为 0）")
        self.weight(f"{p}_down_weight",
                    f"self.{site}_hyper_connection.input_mix_weight_down.weight",
                    ["Rhc", "Hc×H"], ["Rhc", "Hc", "H"],
                    "读门低秩下投影：10240→320，bias=False")
        self.weight(f"{p}_up_weight",
                    f"self.{site}_hyper_connection.input_mix_weight_up.weight",
                    ["Hc×H", "Rhc"], ["Hc", "H", "Rhc"],
                    "读门上投影：320→10240，bias=False")
        self.weight(f"{p}_inject_weight",
                    f"self.{site}_hyper_connection.block_inject_weight",
                    ["Hc", "Hc×H"], ["Hc", "Hc", "H"],
                    "写门投影：10240→4，每个分支一个标量 logit，bias=False")
        self.tensor(f"{p}_normed", "hyper_input_normed = self.hc_norm(hyper_input)",
                    ["B", "seqlen", "Hc×H"], ["B", "seqlen", "Hc", "H"], "activation",
                    "分组 RMSNorm：每个 2560 维流独立做 fp32 方差归一，再乘 1+weight")
        self.op(f"{p}_norm", f"Qwen4ExpTextGatedResidual.hc_norm",
                "Gated Residual 分组 RMSNorm", "rmsnorm-grouped",
                [streams_in], [f"{p}_normed"], [f"{p}_norm_weight"],
                ["Hc", "H", "EpsN"],
                [src("1012", "hyper_input_normed = self.hc_norm(hyper_input)"),
                 src("152-156", "group_size 时 reshape 到 (-1, group_size) 做 RMSNorm")],
                "每个残差流独立归一（group_size=2560），不是把 4 路拼成一个 10240 维向量归一")
        self.tensor(f"{p}_down", "self.input_mix_weight_down(normed)",
                    ["B", "seqlen", "Rhc"], ["B", "seqlen", "Rhc"], "activation",
                    "读门低秩 10240→320")
        self.op(f"{p}_down", f"Qwen4ExpTextGatedResidual.input_mix_weight_down",
                "读门低秩下投影", "linear", [f"{p}_normed"], [f"{p}_down"],
                [f"{p}_down_weight"], ["Hc", "H", "Rhc", "B", "seqlen"],
                [src("1013", "F.silu(self.input_mix_weight_down(hyper_input_normed) / self.hc_count)")],
                "先除 hc_count=4 再过 SiLU；这一步是读门的低秩压缩")
        self.tensor(f"{p}_down_scaled", "silu(down / Hc)", ["B", "seqlen", "Rhc"],
                    ["B", "seqlen", "Rhc"], "activation",
                    "下投影结果除以 4 后 SiLU")
        self.op(f"{p}_down_act", f"Qwen4ExpTextGatedResidual.input_mix_weight_down",
                "读门低秩激活", "silu+scale", [f"{p}_down"], [f"{p}_down_scaled"],
                [], ["Rhc", "Hc"],
                [src("1013", "F.silu(self.input_mix_weight_down(hyper_input_normed) / self.hc_count)")],
                "除以 hc_count 是真实缩放，不是显示修正")
        self.tensor(f"{p}_read_logits", "self.input_mix_weight_up(...)",
                    ["B", "seqlen", "Hc×H"], ["B", "seqlen", "Hc", "H"], "activation",
                    "读门 logits（10240 维），随后 sigmoid")
        self.op(f"{p}_up", f"Qwen4ExpTextGatedResidual.input_mix_weight_up",
                "读门上投影", "linear", [f"{p}_down_scaled"], [f"{p}_read_logits"],
                [f"{p}_up_weight"], ["Rhc", "Hc", "H", "B", "seqlen"],
                [src("1014", "input_mix_weight = torch.sigmoid(self.input_mix_weight_up(...))")],
                "上投影回 10240 维；bias=False")
        self.tensor(f"{p}_read_flat", "input_mix_weight（sigmoid 后 flat）",
                    ["B", "seqlen", "Hc×H"], ["B", "seqlen", "Hc", "H"], "activation",
                    "读门是逐元素 [B,S,4,H]，但源码先得到 flat 10240 维再 unflatten")
        self.op(f"{p}_read_sigmoid", f"Qwen4ExpTextGatedResidual.forward",
                "读门 sigmoid", "sigmoid",
                [f"{p}_read_logits"], [f"{p}_read_flat"], [],
                ["Hc", "H", "B", "seqlen"],
                [src("1014", "input_mix_weight = torch.sigmoid(self.input_mix_weight_up(...))")],
                "sigmoid 在 flat 10240 维上逐元素作用")
        self.tensor(f"{p}_read", "input_mix_weight.unflatten(-1,(Hc,H))",
                    ["B", "seqlen", "Hc", "H"], ["B", "seqlen", "Hc", "H"], "activation",
                    "unflatten 成 [B,S,4,H]；每个流、每个 hidden 维都有独立门")
        self.op(f"{p}_read_view", f"Qwen4ExpTextGatedResidual.forward",
                "读门 unflatten", "unflatten",
                [f"{p}_read_flat"], [f"{p}_read"], [], ["Hc", "H", "B", "seqlen"],
                [src("1015", "input_mix_weight.unflatten(-1, (self.hc_count, self.hidden_size))")],
                "flat 读门按 (4,2560) unflatten，随后与 normed 的 4D 视图相乘")
        self.tensor(f"{p}_normed_4d", "hyper_input_normed.unflatten(-1,(Hc,H))",
                    ["B", "seqlen", "Hc", "H"], ["B", "seqlen", "Hc", "H"], "activation",
                    "normed 的 4D 视图；源码在 read 相乘时显式 unflatten")
        self.op(f"{p}_norm_view", f"Qwen4ExpTextGatedResidual.forward",
                "normed unflatten", "unflatten",
                [f"{p}_normed"], [f"{p}_normed_4d"], [], ["Hc", "H", "B", "seqlen"],
                [src("1016", "hyper_input_normed.unflatten(-1, (hc_count, hidden_size))")],
                "层间边界是 flat [B,S,Hc*H]；4D 只在读门相乘时出现")
        self.tensor(f"{p}_mixed", "mixed_input = (read * normed_4d).mean(dim=-2)",
                    ["B", "seqlen", "H"], ["B", "seqlen", "H"], "activation",
                    "4 路流按读门加权后取 mean，得到进入子层的单流")
        self.op(f"{p}_mix", f"Qwen4ExpTextGatedResidual.forward",
                "读门加权 mean 坍缩", "broadcast-mul+mean",
                [f"{p}_normed_4d", f"{p}_read"], [f"{p}_mixed"], [],
                ["Hc", "H", "B", "seqlen"],
                [src("1016-1018", "(input_mix_weight * hyper_input_normed.unflatten(...)).mean(dim=-2)")],
                "mean 而不是 sum：4 路流是残差副本，加权平均回单流")
        self.tensor(f"{p}_inject_logits", "self.block_inject_weight(normed)",
                    ["B", "seqlen", "Hc"], ["B", "seqlen", "Hc"], "activation",
                    "写门 logits：每个分支一个标量")
        self.op(f"{p}_inject_linear", f"Qwen4ExpTextGatedResidual.block_inject_weight",
                "写门投影", "linear", [f"{p}_normed"], [f"{p}_inject_logits"],
                [f"{p}_inject_weight"], ["Hc", "H", "B", "seqlen"],
                [src("1021", "injection_weights = 2 * torch.sigmoid(self.block_inject_weight(hyper_input_normed) / self.hc_count)")],
                "10240→4；每个分支一个标量，与读门的逐元素门不同")
        self.tensor(f"{p}_inject", "injection_weights = 2*sigmoid(logits/Hc)",
                    ["B", "seqlen", "Hc"], ["B", "seqlen", "Hc"], "activation",
                    "写门范围 (0,2)，控制子层输出写回每个残差流的强度")
        self.op(f"{p}_inject", f"Qwen4ExpTextGatedResidual.forward",
                "写门 sigmoid+scale", "sigmoid+scale",
                [f"{p}_inject_logits"], [f"{p}_inject"], [], ["Hc"],
                [src("1021", "2 * torch.sigmoid(self.block_inject_weight(...) / self.hc_count)")],
                "除以 hc_count 后乘 2；这是每分支标量写门")

    def _gated_residual_apply(self, site: str, sublayer_out: str,
                              residual_streams: str, streams_out: str) -> None:
        p = f"g_hc_{site}"
        src = self.src
        self.tensor(f"{p}_expanded", "out.unsqueeze(-2) * inject.unsqueeze(-1)",
                    ["B", "seqlen", "Hc", "H"], ["B", "seqlen", "Hc", "H"], "activation",
                    "子层单流输出按写门广播到 4 路流")
        self.op(f"{p}_expand", "Qwen4ExpTextDecoderLayer.forward",
                "子层输出按写门展开", "broadcast-mul",
                [sublayer_out, f"{p}_inject"], [f"{p}_expanded"], [],
                ["Hc", "H", "B", "seqlen"],
                [src("1293-1294", "injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)")],
                "每路写门是标量；不是把单流原样复制 4 份")
        self.tensor(f"{p}_expanded_flat", "injection.flatten(-2)",
                    ["B", "seqlen", "Hc×H"], ["B", "seqlen", "Hc", "H"], "activation",
                    "4D 写门结果显式 flatten 回层间 flat 10240 维；不能 4D 直接加 3D")
        self.op(f"{p}_expand_flat", "Qwen4ExpTextDecoderLayer.forward",
                "写门结果 flatten", "flatten",
                [f"{p}_expanded"], [f"{p}_expanded_flat"], [], ["Hc", "H", "B", "seqlen"],
                [src("1294", "injection.flatten(-2)")],
                "与 hyper_input 的 [B,S,Hc*H] 同型后再 add")
        self.tensor(streams_out, "hyper_input + injection.flatten(-2)",
                    ["B", "seqlen", "Hc×H"], ["B", "seqlen", "Hc", "H"], "activation",
                    "原 flat 4 路流 + flatten 后的写门输出；残差不是标量相加")
        self.op(f"{p}_add", "Qwen4ExpTextDecoderLayer.forward", "Gated Residual 相加",
                "add", [residual_streams, f"{p}_expanded_flat"], [streams_out], [],
                ["Hc", "H", "B", "seqlen"],
                [src("1294", "hidden_states = hyper_input + injection.flatten(-2)")],
                "残差底座是 attn_hc/ffn_hc 返回的 flat hyper_input（[B,S,Hc*H]），不是单流")

    # ------------------------------------------------------------------
    # MoE（Qwen 的 softmax 路由 + 无 clamp SwiGLU）
    # ------------------------------------------------------------------

    def _moe_ops(self, ffn_in: str) -> None:
        src = self.src
        self.tensor("g_moe_flat", "hidden_states.view(-1, hidden_dim)", ["Ttok", "H"],
                    ["Ttok", "H"], "activation", "MoE 先把 [B,seqlen,H] 展平（:982）")
        self.op("g_moe_flatten", "Qwen4ExpTextSparseMoeBlock.forward", "MoE 输入展平",
                "view", [ffn_in], ["g_moe_flat"], [], ["B", "seqlen", "Ttok", "H"],
                [src("981-982", "batch_size, sequence_length, hidden_dim = shape; "
                                "hidden_states.view(-1, hidden_dim)")],
                "展平后按 token 路由；共享专家也用同一展平输入")
        self.weight("g_moe_router_weight", "self.mlp.gate.weight", ["E", "H"], ["E", "H"],
                    "TopKRouter.weight；forward 用 F.linear，无 bias（:962）")
        self.tensor("g_router_logits", "router_logits = F.linear(x, W)",
                    ["Ttok", "E"], ["Ttok", "E"], "activation",
                    "路由打分；随后 softmax（不是 sigmoid）")
        self.op("g_moe_router", "Qwen4ExpTextTopKRouter.forward", "MoE 路由打分", "linear",
                ["g_moe_flat"], ["g_router_logits"], ["g_moe_router_weight"],
                ["Ttok", "E", "H"],
                [src("960-963", "router_logits = F.linear(hidden_states, self.weight); "
                                "router_probs = softmax(router_logits, dtype=float, dim=-1)")],
                "512 个专家的 softmax 概率；没有 correction bias")
        self.tensor("g_router_probs", "router_probs = softmax(router_logits)",
                    ["Ttok", "E"], ["Ttok", "E"], "activation",
                    "fp32 softmax（dtype=torch.float）")
        self.op("g_moe_softmax", "Qwen4ExpTextTopKRouter.forward", "Softmax 路由概率",
                "softmax", ["g_router_logits"], ["g_router_probs"], [], ["Ttok", "E"],
                [src("963", "router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)")],
                "softmax 而不是 sigmoid；这是与 sigmoid 路由 MoE 的实质差异")
        self.tensor("g_topk_values", "router_top_value", ["Ttok", "K"], ["Ttok", "K"],
                    "activation", "top-k 概率值；norm_topk_prob=true 时再归一")
        self.tensor("g_topk_indices", "router_indices", ["Ttok", "K"], ["Ttok", "K"],
                    "activation", "选中的专家索引")
        self.op("g_moe_topk", "Qwen4ExpTextTopKRouter.forward", "TopK 专家选择", "topk",
                ["g_router_probs"], ["g_topk_values", "g_topk_indices"], [],
                ["Ttok", "K", "E"],
                [src("964", "router_top_value, router_indices = torch.topk(router_probs, top_k, dim=-1)")],
                "每 token 从 512 个专家中选 10 个")
        self.tensor("g_topk_weights", "top_k_weights（top-10 归一化后）",
                    ["Ttok", "K"], ["Ttok", "K"], "activation",
                    "norm_topk_prob=true：除以 top-10 之和")
        self.op("g_moe_weight_norm", "Qwen4ExpTextTopKRouter.forward", "路由权重归一化",
                "normalize", ["g_topk_values"], ["g_topk_weights"], [], ["K"],
                [src("965-967", "if self.norm_topk_prob: router_top_value /= "
                                "router_top_value.sum(dim=-1, keepdim=True)")],
                "归一后 top-10 权重和为 1；没有 routed scaling factor")
        self.weight("g_moe_experts_gate_up", "self.mlp.experts.gate_up_proj",
                    ["E", "2Es", "H"], ["E", "Es", "H"],
                    "专家权重 3D 张量：[E,2×640,2560]（:920）")
        self.weight("g_moe_experts_down", "self.mlp.experts.down_proj",
                    ["E", "H", "Es"], ["E", "H", "Es"],
                    "专家权重 3D 张量：[E,2560,640]（:921）")
        self.tensor("g_expert_mask", "one_hot(top_k_index,E+1).permute(2,1,0)",
                    ["E+1", "K", "Ttok"], ["E", "K", "Ttok"], "activation",
                    "one_hot(...E+1) 得 [Ttok,K,E+1]，permute 成 [E+1,K,Ttok]（:932-933）；"
                    "最后一槽在循环里跳过")
        self.op("g_moe_expert_mask", "Qwen4ExpTextExperts.forward", "专家 one-hot 掩码",
                "one_hot+permute", ["g_topk_indices"], ["g_expert_mask"], [],
                ["E", "K", "Ttok"],
                [src("931-933", "expert_mask = F.one_hot(top_k_index, num_classes=num_experts+1)"
                                ".permute(2, 1, 0)")],
                "保留完整 [E+1,K,Ttok] 分配矩阵；不展开 512 个专家图")
        self.tensor("g_expert_hit", "expert_hit（命中的专家索引，数据相关）", ["N_hit"],
                    ["N_hit"], "activation",
                    "expert_mask.sum(-1,-2)>0 后 nonzero（:934）")
        self.op("g_moe_expert_hit", "Qwen4ExpTextExperts.forward", "专家命中集合",
                "sum+nonzero", ["g_expert_mask"], ["g_expert_hit"], [], ["E", "N_hit"],
                [src("934", "expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()")],
                "只用于决定单专家模板循环哪些专家；N_hit 未给定，保持未知")
        self.tensor("g_expert_topk_pos", "top_k_pos（该 token 的第几个 top-k 槽）",
                    ["N_e"], ["N_e"], "activation", "torch.where 的第一个返回（:940）")
        self.tensor("g_expert_token_idx", "token_idx（命中该专家的 token 行）",
                    ["N_e"], ["N_e"], "activation", "torch.where 的第二个返回（:940）")
        self.op("g_moe_expert_gather_indices", "Qwen4ExpTextExperts.forward",
                "命中专家 → (top_k_pos, token_idx)", "where",
                ["g_expert_mask", "g_expert_hit"],
                ["g_expert_topk_pos", "g_expert_token_idx"], [], ["N_e", "K", "Ttok"],
                [src("936-940", "for expert_idx in expert_hit: top_k_pos, token_idx = "
                                "torch.where(expert_mask[expert_idx])")],
                "从完整分配矩阵取真实的 (top_k_pos, token_idx)")
        self.tensor("g_token_states", "hidden_states[token_idx]", ["N_e", "H"],
                    ["N_e", "H"], "activation", "按 token_idx gather（:941）")
        self.op("g_moe_select_tokens", "Qwen4ExpTextExperts.forward", "选择专家 token 行",
                "gather", ["g_moe_flat", "g_expert_token_idx"], ["g_token_states"], [],
                ["N_e", "H"],
                [src("941", "current_state = hidden_states[token_idx]")],
                "只 gather token_idx 指定的行；N_e 是数据相关量")
        self.tensor("g_gate_up", "F.linear(current_state, gate_up_proj[e])",
                    ["N_e", "2Es"], ["N_e", "Es"], "activation",
                    "单专家 gate∪up 合并投影（:942）")
        self.op("g_moe_expert_gate_up", "Qwen4ExpTextExperts.forward", "专家 gate_up 投影",
                "linear", ["g_token_states"], ["g_gate_up"], ["g_moe_experts_gate_up"],
                ["H", "Es", "N_e", "E"],
                [src("942", "gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)")],
                "gate/up 在权重里打包成 [E,2Es,H]；按专家索引取切片")
        self.tensor("g_expert_gate", "gate", ["N_e", "Es"], ["N_e", "Es"], "activation",
                    "chunk 前段（:942）")
        self.tensor("g_expert_up", "up", ["N_e", "Es"], ["N_e", "Es"], "activation",
                    "chunk 后段（:942）")
        self.op("g_moe_expert_chunk", "Qwen4ExpTextExperts.forward", "专家 gate/up 拆分",
                "chunk", ["g_gate_up"], ["g_expert_gate", "g_expert_up"], [], ["Es"],
                [src("942", "gate, up = ... .chunk(2, dim=-1)")],
                "沿最后一维均分；2×640→两个 640")
        self.tensor("g_expert_hidden", "silu(gate) * up", ["N_e", "Es"],
                    ["N_e", "Es"], "activation", "SwiGLU 门控；**没有 clamp**")
        self.op("g_moe_expert_act", "Qwen4ExpTextExperts.forward", "专家 SiLU 门控",
                "silu*mul", ["g_expert_gate", "g_expert_up"], ["g_expert_hidden"], [],
                ["Es"],
                [src("943", "current_hidden_states = self.act_fn(gate) * up")],
                "与带 clamp 的 SwiGLU 变体不同：Qwen 源码没有 clamp")
        self.tensor("g_expert_out", "F.linear(act, down_proj[e])", ["N_e", "H"],
                    ["N_e", "H"], "activation", "单专家 down 投影（:944）")
        self.op("g_moe_expert_down", "Qwen4ExpTextExperts.forward", "专家 down 投影",
                "linear", ["g_expert_hidden"], ["g_expert_out"], ["g_moe_experts_down"],
                ["Es", "H", "N_e", "E"],
                [src("944", "current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])")],
                "按专家索引取 [H,Es] 切片")
        self.tensor("g_expert_weighted", "out * top_k_weights[token_idx, top_k_pos]",
                    ["N_e", "H"], ["N_e", "H"], "activation",
                    "按 (token_idx, top_k_pos) 取路由权重后缩放（:945）")
        self.op("g_moe_expert_weight", "Qwen4ExpTextExperts.forward", "专家输出加权",
                "scale", ["g_expert_out", "g_topk_weights", "g_expert_token_idx",
                          "g_expert_topk_pos"], ["g_expert_weighted"], [], ["N_e", "K"],
                [src("945", "current_hidden_states * top_k_weights[token_idx, top_k_pos, None]")],
                "权重索引同时用 token_idx 与 top_k_pos")
        self.tensor("g_routed_init", "final_hidden_states = zeros_like(hidden_states)",
                    ["Ttok", "H"], ["Ttok", "H"], "activation", "index_add 初值（:930）")
        self.op("g_moe_result_init", "Qwen4ExpTextExperts.forward", "专家输出累加初值",
                "zeros_like", ["g_moe_flat"], ["g_routed_init"], [], ["Ttok", "H"],
                [src("930", "final_hidden_states = torch.zeros_like(hidden_states)")],
                "按展平 token 形状建零张量")
        self.tensor("g_routed_flat", "final_hidden_states（index_add 后）",
                    ["Ttok", "H"], ["Ttok", "H"], "activation",
                    "所有专家输出按 token 累加（:946）")
        self.op("g_moe_index_add", "Qwen4ExpTextExperts.forward", "专家输出累加",
                "index_add", ["g_routed_init", "g_expert_weighted", "g_expert_token_idx"],
                ["g_routed_flat"], [], ["Ttok", "H"],
                [src("946", "final_hidden_states.index_add_(0, token_idx, "
                           "current_hidden_states.to(final_hidden_states.dtype))")],
                "按 token_idx 把每个专家的加权输出加回对应 token 行")

        # 共享专家（并行支路，带标量门）
        self.weight("g_moe_shared_gate_weight", "self.mlp.shared_expert.gate_proj.weight",
                    ["Ess", "H"], ["Ess", "H"], "共享专家 gate_proj 2560→640")
        self.weight("g_moe_shared_up_weight", "self.mlp.shared_expert.up_proj.weight",
                    ["Ess", "H"], ["Ess", "H"], "共享专家 up_proj 2560→640")
        self.weight("g_moe_shared_down_weight", "self.mlp.shared_expert.down_proj.weight",
                    ["H", "Ess"], ["H", "Ess"], "共享专家 down_proj 640→2560")
        self.weight("g_moe_shared_scalar_gate", "self.mlp.shared_expert_gate.weight",
                    ["1", "H"], ["H"], "共享专家标量门 2560→1（:978）")
        self.tensor("g_shared_gate", "shared_expert.gate_proj(x)", ["Ttok", "Ess"],
                    ["Ttok", "Ess"], "activation", "共享专家 gate（:983）")
        self.op("g_moe_shared_gate", "Qwen4ExpTextMLP.gate_proj", "共享专家 gate_proj",
                "linear", ["g_moe_flat"], ["g_shared_gate"], ["g_moe_shared_gate_weight"],
                ["H", "Ess", "Ttok"],
                [src("982-983", "shared_expert_output = self.shared_expert(hidden_states_reshaped); "
                                "MLP.forward: down_proj(act_fn(gate_proj(x)) * up_proj(x))")],
                "共享专家与 routed 专家并行，输入同一展平 hidden")
        self.tensor("g_shared_up", "shared_expert.up_proj(x)", ["Ttok", "Ess"],
                    ["Ttok", "Ess"], "activation", "共享专家 up（:983）")
        self.op("g_moe_shared_up", "Qwen4ExpTextMLP.up_proj", "共享专家 up_proj",
                "linear", ["g_moe_flat"], ["g_shared_up"], ["g_moe_shared_up_weight"],
                ["H", "Ess", "Ttok"],
                [src("907", "self.act_fn(self.gate_proj(x)) * self.up_proj(x)")],
                "无 clamp")
        self.tensor("g_shared_act", "silu(gate) * up", ["Ttok", "Ess"],
                    ["Ttok", "Ess"], "activation", "共享专家 SwiGLU（:907）")
        self.op("g_moe_shared_act", "Qwen4ExpTextMLP.forward", "共享专家 SiLU 门控",
                "silu*mul", ["g_shared_gate", "g_shared_up"], ["g_shared_act"], [],
                ["Ess"],
                [src("907", "down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))")],
                "与 routed 专家同样的无 clamp SwiGLU，但宽度是 shared_expert_intermediate_size")
        self.tensor("g_shared_out", "shared_expert(x)", ["Ttok", "H"],
                    ["Ttok", "H"], "activation", "共享专家 down 输出（:907）")
        self.op("g_moe_shared_down", "Qwen4ExpTextMLP.down_proj", "共享专家 down_proj",
                "linear", ["g_shared_act"], ["g_shared_out"], ["g_moe_shared_down_weight"],
                ["Ess", "H", "Ttok"],
                [src("903", "self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)")],
                "回投影到 hidden_size")
        self.tensor("g_shared_gate_scalar", "sigmoid(shared_expert_gate(x))", ["Ttok", "1"],
                    ["Ttok"], "activation",
                    "共享专家的标量门（每 token 一个），不是路由权重")
        self.op("g_moe_shared_gate_scalar", "Qwen4ExpTextSparseMoeBlock.forward",
                "共享专家标量门", "linear+sigmoid",
                ["g_moe_flat"], ["g_shared_gate_scalar"], ["g_moe_shared_scalar_gate"],
                ["H", "Ttok"],
                [src("987", "shared_expert_output = F.sigmoid(self.shared_expert_gate("
                            "hidden_states_reshaped)) * shared_expert_output")],
                "sigmoid 标量门乘共享专家输出；与 routed 路由无关")
        self.tensor("g_shared_gated", "sigmoid(gate) * shared_expert(x)", ["Ttok", "H"],
                    ["Ttok", "H"], "activation", "门控后的共享专家输出")
        self.op("g_moe_shared_gated", "Qwen4ExpTextSparseMoeBlock.forward",
                "共享专家门控", "broadcast-mul",
                ["g_shared_gate_scalar", "g_shared_out"], ["g_shared_gated"], [],
                ["Ttok", "H"],
                [src("987", "F.sigmoid(...) * shared_expert_output")],
                "标量门广播到 hidden 维")
        self.tensor("g_mlp_flat", "expert_output + shared_expert_output", ["Ttok", "H"],
                    ["Ttok", "H"], "activation", "routed 与共享专家汇合（:989）")
        self.op("g_moe_merge", "Qwen4ExpTextSparseMoeBlock.forward",
                "Routed + 共享专家汇合", "add",
                ["g_routed_flat", "g_shared_gated"], ["g_mlp_flat"], [], ["Ttok", "H"],
                [src("989", "expert_output = expert_output + shared_expert_output")],
                "共享专家是并行支路，不参与 top-k 路由；两者直接相加（仍在展平的 "
                "[Ttok,H] 上，随后才 reshape）")
        self.tensor("g_mlp_out", "mlp_out（本组 MoE 输出）", ["B", "seqlen", "H"],
                    ["B", "seqlen", "H"], "activation", "reshape 回 [B,seqlen,H]（:990-991）")
        self.op("g_moe_merge_view", "Qwen4ExpTextSparseMoeBlock.forward",
                "MoE 输出还原", "view", ["g_mlp_flat"], ["g_mlp_out"], [],
                ["B", "seqlen", "H"],
                [src("990-991", "expert_output.reshape(batch_size, sequence_length, hidden_dim)")],
                "与 g_routed_view 不同：这里包含共享专家相加后的最终 MoE 输出")


class _GdnBlockBuilder(_BlockBuilder):
    """Gated DeltaNet（+ 可选 N-gram PLE）+ TopK MoE 的真实单层结构。"""

    kind = "gdn"

    def __init__(self, values: HfModelValues, layer_idx: int, *, ple: bool):
        super().__init__(values, layer_idx)
        self.ple = ple

    def build(self) -> dict:
        v = self.values
        title = (
            f"Qwen3.8-Flash-Next Decoder Block（layer_idx={self.layer_idx}，"
            f"Gated DeltaNet{' + N-gram PLE' if self.ple else ''} + TopK MoE；"
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
        src = self.src
        self._common_parameters()
        self.param("NkH", "NkH", "linear_num_key_heads GDN QK 头数", v.linear_num_key_heads, None,
                   "config.linear_num_key_heads=16；q/k 投影各 16×128=2048",
                   [cfg("text_config.linear_num_key_heads", "文本子配置字段"),
                    src("460-465", "self.num_k_heads / self.key_dim")])
        self.param("NvH", "NvH", "linear_num_value_heads GDN V 头数", v.linear_num_value_heads, None,
                   "config.linear_num_value_heads=48；v 投影 48×128=6144",
                   [cfg("text_config.linear_num_value_heads", "文本子配置字段"),
                    src("460-465", "self.num_v_heads / self.value_dim")])
        self.param("Dk", "Dk", "linear_key_head_dim GDN key 头维", v.linear_key_head_dim, None,
                   "config.linear_key_head_dim=128；q/k head_dim",
                   [cfg("text_config.linear_key_head_dim", "文本子配置字段"),
                    src("462", "self.head_k_dim = config.linear_key_head_dim")])
        self.param("DvH", "DvH", "linear_value_head_dim GDN value 头维", v.linear_value_head_dim,
                   None, "config.linear_value_head_dim=128；v head_dim",
                   [cfg("text_config.linear_value_head_dim", "文本子配置字段"),
                    src("463", "self.head_v_dim = config.linear_value_head_dim")])
        self.param("Kconv", "Kconv", "linear_conv_kernel_dim GDN 短卷积核", v.linear_conv_kernel_dim,
                   None, "config.linear_conv_kernel_dim=4；depthwise causal conv",
                   [cfg("text_config.linear_conv_kernel_dim", "文本子配置字段"),
                    src("467-481", "Conv1d(kernel=4, groups=conv_dim, padding=3)")])
        self.param("Gmul", "Gmul", "q/k → v 头的 repeat 倍数", v.linear_num_value_heads //
                   v.linear_num_key_heads, "Gmul = NvH // NkH",
                   "48//16=3；q/k 在 delta-rule 前 repeat_interleave 到 48 头",
                   [src("573-575", "if num_v_heads // num_k_heads > 1: repeat_interleave")])
        self.param("Dqkv", "D_qkv", "GDN qkv 合并投影宽度（conv 通道）",
                   (2 * v.linear_num_key_heads * v.linear_key_head_dim
                    + v.linear_num_value_heads * v.linear_value_head_dim),
                   "D_qkv = 2 * NkH * Dk + NvH * DvH",
                   "2×16×128 + 48×128 = 10240；conv1d 的 depthwise 通道数",
                   [src("473", "self.conv_dim = self.key_dim * 2 + self.value_dim")])
        self.param("Dz", "D_z", "GDN 输出门 z 宽度",
                   v.linear_num_value_heads * v.linear_value_head_dim,
                   "D_z = NvH * DvH",
                   "48×128=6144；in_proj_z 与 Gated RMSNorm 的 gate",
                   [src("498", "self.in_proj_z = nn.Linear(hidden_size, value_dim)")])
        self.param("Srec", "S_rec", "单层 recurrent state 元素数（每 batch）", None,
                   "S_rec = NvH * Dk * Dv",
                   "48×128×128=786,432；大小固定，不随 kvcachelen 增长",
                   [src("426-427", "recurrent_state_shape = (B, num_v_heads, k_head_dim, v_head_dim)")])
        self.param("Sconv", "S_conv", "单层 conv state 元素数（每 batch）", None,
                   "S_conv = D_qkv * Kconv",
                   "10240×4=40,960；保存最近 4 个 token 的 qkv 通道",
                   [src("539-541", "update_conv_state(mixed_qkv, layer_idx, conv_kernel_size=4)"),
                    self.srcf("cache_utils.py", "1099-1149", "update_conv_state")])
        if self.ple:
            self.param("Ngm", "Ngm", "ngram_size 最大 token n-gram 阶数", v.ngram_size, None,
                       "config.ngram_size=3；PLE 覆盖 bigram(2) + trigram(3)",
                       [cfg("text_config.ngram_size", "文本子配置字段")])
            self.param("Hng", "Hng", "heads_per_ngram 每个 n-gram 阶的哈希头数",
                       v.heads_per_ngram, None, "config.heads_per_ngram=8",
                       [cfg("text_config.heads_per_ngram", "文本子配置字段")])
            self.param("Ngh", "N_ngram_heads", "PLE 的哈希头总数", v.ngram_heads,
                       "N_gh = (Ngm - 1) * Hng",
                       "(3-1)×8=16（8 bigram + 8 trigram）；每头独立素数码表",
                       [src("1077-1078", "self.ngram_heads = (ngram_size - 1) * heads_per_ngram")])
            self.param("Dgh", "D_ngram_head", "每个哈希头的 embedding 维度", v.ngram_head_dim,
                       "D_gh = Dple // Ngh",
                       "2560//16=160；每个头查表得到 160 维，拼成 2560",
                       [src("1082", "head_dim_per_ngram = embedding_dim // self.ngram_heads")])
            self.param("Dple", "Dple", "ple_embed_dim PLE embedding 总宽", v.ple_embed_dim,
                       None, "config.ple_embed_dim=2560",
                       [cfg("text_config.ple_embed_dim", "文本子配置字段")])
            self.param("Kple", "Kple", "ple_conv_kernel_size PLE 短卷积核", v.ple_conv_kernel_size,
                       None, "config.ple_conv_kernel_size=4",
                       [cfg("text_config.ple_conv_kernel_size", "文本子配置字段")])
            self.param("Dild", "D_dilate", "PLE 短卷积 dilation", v.ple_conv_dilation,
                       "D_dilate = Ngm",
                       "dilation=ngram_size=3；有效感受野 (4-1)*3+1=10",
                       [src("1191-1192", "conv_dilation = config.ngram_size")])
            self.param("Lpconv", "L_ple_conv", "PLE 短卷积 state 长度", v.ple_short_conv_state_len,
                       "L_ple_conv = (Kple - 1) * D_dilate",
                       "(4-1)×3=9；保存 9 个历史 token 的 10240 通道",
                       [src("1192", "self.short_conv_state_len = (conv_kernel_size - 1) * conv_dilation")])
            self.param("Sple", "S_ple_conv", "PLE 短卷积 state 元素数（每 batch）",
                       v.ple_short_conv_state_len
                       * (2 * v.linear_num_key_heads * v.linear_key_head_dim
                          + v.linear_num_value_heads * v.linear_value_head_dim),
                       "S_ple_conv = Dqkv * L_ple_conv",
                       "10240×9=92,160；仅 layer 1 的 PLE 短卷积使用；固定大小",
                       [self.src("1191-1192", "short_conv_state_len=(4-1)*3=9"),
                        self.srcf("cache_utils.py", "1099-1149", "update_conv_state")])
            self.param("Lctx", "L_ctx", "PLE n-gram 上下文长度", v.ngram_context_len,
                       "L_ctx = Ngm - 1",
                       "3-1=2；上一 token 上下文缓存 [B,2]，初始 eos",
                       [src("1122-1144", "previous_context [B,context_len]；update_conv_state(state_idx=2)")])
            self.param("Vng", "Vng", "ngram_vocab_size_base 每头基础词表下界",
                       v.ngram_vocab_size_base, None,
                       "config.ngram_vocab_size_base=20,000,000；每头取其后第 i 个素数",
                       [cfg("text_config.ngram_vocab_size_base", "文本子配置字段")])
            sizes, _offsets, padded = _ngram_vocab_layout(v)
            self.param("Vngpad", "Vng_pad", "PLE n-gram 拼接后的 pad 词表大小", padded,
                       "pad_to_divisor(sum(每头素数码表), 128)",
                       f"16 个头词表大小之和按 {v.ngram_vocab_divisor} 向上取整；"
                       f"静态复算为 {padded:,}；每头大小 {sizes[0]:,}..{sizes[-1]:,}",
                       [cfg("text_config.make_ngram_vocab_size_divisible_by", "文本子配置字段"),
                        src("1087-1104", "head_vocab_sizes/head_offsets/padded_vocab_size")])
            self.param("Eos", "Eos", "PLE n-gram 上下文切断 token", v.eos_token_id, None,
                       "config.eos_token_id 的首项；_shift_right_ignore_eos 在 EOS 后重置上下文",
                       [cfg("text_config.eos_token_id", "文本子配置字段"),
                        src("1106-1120", "_shift_right_ignore_eos")])

    def _build_derivations(self) -> None:
        super()._build_derivations()
        premise = ("按固定配置 + 「空态起步、连续正常执行」前提推导；"
                   "GDN 状态大小固定，与 kvcachelen 无关")
        self.derivation("S_rec", "单层 recurrent state 元素数", "NvH * Dk * DvH",
                        f"48×128×128=786,432；每 batch 固定。{premise}")
        self.derivation("S_conv", "单层 conv state 元素数", "Dqkv * Kconv",
                        f"10240×4=40,960；每 batch 固定。{premise}")
        if self.ple:
            self.derivation("S_ple_conv", "PLE 短卷积 state 元素数", "Dqkv * Lpconv",
                            f"10240×9=92,160；仅 layer 1 的 PLE 使用。{premise}")
            self.derivation("L_ngram_ctx", "PLE n-gram 上下文长度", "Ngm - 1",
                            f"3-1=2；上一 token 上下文固定 2 个 id。{premise}")

    def _build_boundaries_and_cache(self) -> None:
        v = self.values
        self.tensor(
            "g_hidden_streams_in", "hidden_states（4 路流层输入，flat）",
            ["B", "seqlen", "Hc×H"], ["B", "seqlen", "Hc", "H"], "boundary",
            "上一层输出或网络入口的 4 路 repeat（modeling:1471）；层间实际边界是 "
            "[B,S,Hc*H] flat，读门/写门内部才 unflatten 成 4D（:1015-1018）")
        self.tensor(
            "g_conv_mask", "conv_mask（2D padding mask）", ["B", "seqlen"],
            ["B", "seqlen"], "boundary",
            "create_recurrent_attention_mask 的 [B,seqlen] mask；None 时网络入口补 ones"
            "（modeling:1462-1467）。不是 4D 因果 mask")
        self.tensor(
            "g_gdn_conv_state_pre", "past_key_values.layers[layer_idx].conv_states[0]（调用前）",
            ["B", "Dqkv", "Kconv"], ["B", "Dqkv", "Kconv"], "cache",
            "LinearAttentionLayer 的 GDN conv state；保存最近 Kconv=4 个 token 的 qkv 通道；"
            "首次调用按需左补零（cache_utils:1099-1149）")
        self.tensor(
            "g_gdn_conv_state_post", "self.conv_states[0]（调用后）",
            ["B", "Dqkv", "Kconv"], ["B", "Dqkv", "Kconv"], "cache",
            "update_conv_state 把当前调用尾部写回同一静态地址；长度固定 Kconv")
        self.tensor(
            "g_gdn_state_pre", "past_key_values.layers[layer_idx].recurrent_states[0]（调用前）",
            ["B", "NvH", "Dk", "DvH"], ["B", "NvH", "Dk", "DvH"], "cache",
            "GDN delta-rule 的 [B,48,128,128] 状态；空缓存时以 zeros 起步")
        self.tensor(
            "g_gdn_state_post", "self.recurrent_states[0]（调用后）",
            ["B", "NvH", "Dk", "DvH"], ["B", "NvH", "Dk", "DvH"], "cache",
            "update_recurrent_state 写入 fp32 新状态（modeling:605-607）；"
            "大小固定，不随历史长度增长")
        if self.ple:
            self.tensor(
                "g_ple_input_ids", "ple_input_ids（token id）", ["B", "seqlen"],
                ["B", "seqlen"], "boundary",
                "PLE 需要原始 token id；conv_mask=false 的位置被替换为 eos"
                "（modeling:1464-1467）")
            self.tensor(
                "g_ple_context_pre", "past_key_values.layers[layer_idx].conv_states[2]（调用前）",
                ["B", "Lctx"], ["B", "Lctx"], "cache",
                "PLE n-gram 的上一 token 上下文 [B,2]；初始值 eos（modeling:1126-1129）；"
                "state_idx=2")
            self.tensor(
                "g_ple_context_post", "self.conv_states[2]（调用后）",
                ["B", "Lctx"], ["B", "Lctx"], "cache",
                "update_conv_state(state_idx=2, conv_kernel_size=2) 把当前 ids 尾部写回；"
                "长度固定 2")
            self.tensor(
                "g_ple_conv_state_pre", "past_key_values.layers[layer_idx].conv_states[1]（调用前）",
                ["B", "Dqkv", "Lpconv"], ["B", "Dqkv", "Lpconv"], "cache",
                "PLE 短卷积 state [B,10240,9]；state_idx=1；仅 layer 1 使用")
            self.tensor(
                "g_ple_conv_state_post", "self.conv_states[1]（调用后）",
                ["B", "Dqkv", "Lpconv"], ["B", "Dqkv", "Lpconv"], "cache",
                "update_conv_state(state_idx=1, conv_kernel_size=9) 写回；长度固定 9")

    def _build_flow(self) -> None:
        v = self.values
        src, cfg = self.src, self.cfg
        stream_source = "g_hidden_streams_in"
        if self.ple:
            self._ple_ops("g_hidden_streams_in")
            stream_source = "g_hidden_streams_ple"

        # Gated Residual（attn 站点）
        self._gated_residual_site("attn", stream_source)
        mixed = "g_hc_attn_mixed"

        # GDN
        self.weight("g_gdn_in_qkv_weight", "self.linear_attn.in_proj_qkv.weight",
                    ["Dqkv", "H"], ["Dqkv", "H"],
                    "nn.Linear(2560, 10240, bias=False)（:497）")
        self.weight("g_gdn_in_z_weight", "self.linear_attn.in_proj_z.weight",
                    ["Dz", "H"], ["Dz", "H"],
                    "nn.Linear(2560, 6144, bias=False)（:498）")
        self.weight("g_gdn_in_b_weight", "self.linear_attn.in_proj_b.weight",
                    ["NvH", "H"], ["NvH", "H"],
                    "nn.Linear(2560, 48, bias=False)（:499）")
        self.weight("g_gdn_in_a_weight", "self.linear_attn.in_proj_a.weight",
                    ["NvH", "H"], ["NvH", "H"],
                    "nn.Linear(2560, 48, bias=False)（:500）")
        self.weight("g_gdn_conv_weight", "self.linear_attn.conv1d.weight",
                    ["Dqkv", "1", "Kconv"], ["Dqkv", "Kconv"],
                    "depthwise Conv1d [10240,1,4]；bias=False（:474-481）")
        self.weight("g_gdn_dt_bias", "self.linear_attn.dt_bias", ["NvH"], ["NvH"],
                    "时间步 bias（:485）；初始化为 ones")
        self.weight("g_gdn_a_log", "self.linear_attn.A_log", ["NvH"], ["NvH"],
                    "每头 A_log；decay_rate=exp(A_log)（:488-489）")
        self.weight("g_gdn_norm_weight", "self.linear_attn.norm.weight", ["DvH"], ["DvH"],
                    "Gated RMSNorm 的 weight（head_v_dim=128）（:490-492）")
        self.weight("g_gdn_out_proj_weight", "self.linear_attn.out_proj.weight",
                    ["H", "Dz"], ["H", "Dz"],
                    "nn.Linear(6144, 2560, bias=False)（:493）")
        self.tensor("g_gdn_masked", "apply_mask_to_padding_states(mixed, conv_mask)",
                    ["B", "seqlen", "H"], ["B", "seqlen", "H"], "activation",
                    "padding 位置乘 0；全 1 mask 时是 no-op，但源码仍执行（:510）")
        self.op("g_gdn_mask", "Qwen4ExpTextGatedDeltaNet.forward",
                "GDN padding 置零", "broadcast-mul",
                [mixed, "g_conv_mask"], ["g_gdn_masked"], [], ["B", "seqlen", "H"],
                [src("510", "hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)")],
                "只处理 padding，不是因果 mask；GDN 因果性由短卷积 + 递推顺序保证")
        self.tensor("g_gdn_qkv_flat", "self.in_proj_qkv(masked)",
                    ["B", "seqlen", "Dqkv"], ["B", "seqlen", "Dqkv"], "activation",
                    "q/k/v 合并投影输出 [B,S,10240]")
        self.op("g_gdn_qkv_proj", "Qwen4ExpTextGatedDeltaNet.in_proj_qkv",
                "GDN QKV 合并投影", "linear",
                ["g_gdn_masked"], ["g_gdn_qkv_flat"], ["g_gdn_in_qkv_weight"],
                ["H", "Dqkv", "B", "seqlen"],
                [src("518-519", "mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)")],
                "一次线性投影出 q/k/v 三段；顺序 q(2048)、k(2048)、v(6144)")
        self.tensor("g_gdn_qkv_t", "mixed_qkv.transpose(1,2)",
                    ["B", "Dqkv", "seqlen"], ["B", "Dqkv", "seqlen"], "activation",
                    "Conv1d 需要 [B,channels,seq]；这一步是必要布局变换")
        self.op("g_gdn_qkv_transpose", "Qwen4ExpTextGatedDeltaNet.forward",
                "GDN QKV 转置", "transpose", ["g_gdn_qkv_flat"], ["g_gdn_qkv_t"], [],
                ["B", "seqlen", "Dqkv"],
                [src("519", "mixed_qkv.transpose(1, 2)")],
                "把序列维移到最后一维；depthwise 通道对应 10240")
        self.tensor("g_gdn_z_flat", "self.in_proj_z(masked)",
                    ["B", "seqlen", "Dz"], ["B", "seqlen", "Dz"], "activation",
                    "输出门 z 的投影 [B,S,6144]")
        self.op("g_gdn_z_proj", "Qwen4ExpTextGatedDeltaNet.in_proj_z",
                "GDN 输出门 z 投影", "linear",
                ["g_gdn_masked"], ["g_gdn_z_flat"], ["g_gdn_in_z_weight"],
                ["H", "Dz", "B", "seqlen"],
                [src("521-522", "z = self.in_proj_z(hidden_states); "
                                "z = z.reshape(batch_size, seq_len, -1, head_v_dim)")],
                "z 是 Gated RMSNorm 的 gate，不是 s 或 RoPE")
        self.tensor("g_gdn_z", "z（view 后）", ["B", "seqlen", "NvH", "DvH"],
                    ["B", "seqlen", "NvH", "DvH"], "activation",
                    "z view 成 [B,S,48,128]（:522）")
        self.op("g_gdn_z_view", "Qwen4ExpTextGatedDeltaNet.forward",
                "GDN z 布局", "view", ["g_gdn_z_flat"], ["g_gdn_z"], [],
                ["B", "seqlen", "NvH", "DvH"],
                [src("522", "z.reshape(batch_size, seq_len, -1, self.head_v_dim)")],
                "把 6144 拆成 48×128")
        self.tensor("g_gdn_b", "b = self.in_proj_b(masked)", ["B", "seqlen", "NvH"],
                    ["B", "seqlen", "NvH"], "activation", "beta logits（:524）")
        self.op("g_gdn_b_proj", "Qwen4ExpTextGatedDeltaNet.in_proj_b",
                "GDN beta 投影", "linear", ["g_gdn_masked"], ["g_gdn_b"],
                ["g_gdn_in_b_weight"], ["H", "NvH", "B", "seqlen"],
                [src("524", "b = self.in_proj_b(hidden_states)")],
                "每头一个 beta logit；sigmoid 后是 delta-rule 的写入门控")
        self.tensor("g_gdn_a", "a = self.in_proj_a(masked)", ["B", "seqlen", "NvH"],
                    ["B", "seqlen", "NvH"], "activation", "decay logits（:525）")
        self.op("g_gdn_a_proj", "Qwen4ExpTextGatedDeltaNet.in_proj_a",
                "GDN decay 投影", "linear", ["g_gdn_masked"], ["g_gdn_a"],
                ["g_gdn_in_a_weight"], ["H", "NvH", "B", "seqlen"],
                [src("525", "a = self.in_proj_a(hidden_states)")],
                "每头一个 decay logit；softplus 后乘 exp(A_log)")
        self.op("g_gdn_conv_state_update", "LinearAttentionLayer.update_conv_state",
                "GDN conv state 写入", "cache_update",
                ["g_gdn_conv_state_pre", "g_gdn_qkv_t"], ["g_gdn_conv_state_post"], [],
                ["B", "Dqkv", "Kconv", "seqlen"],
                [src("537-541", "cache_params.update_conv_state(mixed_qkv, layer_idx, "
                                "conv_kernel_size=Kconv)"),
                 self.srcf("cache_utils.py", "1099-1149", "update_conv_state：首次左补零/续接 concat")],
                "源码把 update 返回的 full states 直接喂给卷积；本图把缓存写与卷积输入"
                "分开表达，避免把「前态+本次」误画成持久缓存")
        self.tensor("g_gdn_conv_out", "causal_conv1d_fn / causal_conv1d_update",
                    ["B", "Dqkv", "seqlen"], ["B", "Dqkv", "seqlen"], "activation",
                    "因果短卷积 + SiLU，只取最后 seqlen 步（:543-553）")
        self.op("g_gdn_conv", "Qwen4ExpTextGatedDeltaNet.forward", "GDN 因果短卷积",
                "conv1d", ["g_gdn_conv_state_pre", "g_gdn_qkv_t"], ["g_gdn_conv_out"],
                ["g_gdn_conv_weight"], ["Dqkv", "Kconv", "seqlen"],
                [src("527-553", "causal_conv1d_update（单 token）/ causal_conv1d_fn（prefill）"),
                 src("202-242", "depthwise Conv1d + SiLU；padding=Kconv-1 后 slice 最后 seqlen")],
                "depthwise、causal（只看左侧 Kconv-1 个历史 + 当前）；"
                "conv state 长度固定")
        self.tensor("g_gdn_conv_seq", "mixed_qkv.transpose(1,2)（卷积后）",
                    ["B", "seqlen", "Dqkv"], ["B", "seqlen", "Dqkv"], "activation",
                    "转回 [B,S,10240] 后 split q/k/v（:555-568）")
        self.op("g_gdn_conv_transpose", "Qwen4ExpTextGatedDeltaNet.forward",
                "GDN 卷积输出布局", "transpose", ["g_gdn_conv_out"], ["g_gdn_conv_seq"], [],
                ["B", "seqlen", "Dqkv"],
                [src("555", "mixed_qkv = mixed_qkv.transpose(1, 2)")],
                "把通道维放回最后一维，随后按 q/k/v 拆分")
        self.tensor("g_gdn_q", "query（split 后）", ["B", "seqlen", "NkH", "Dk"],
                    ["B", "seqlen", "NkH", "Dk"], "activation", "q [B,S,16,128]（:566）")
        self.tensor("g_gdn_k", "key（split 后）", ["B", "seqlen", "NkH", "Dk"],
                    ["B", "seqlen", "NkH", "Dk"], "activation", "k [B,S,16,128]（:567）")
        self.tensor("g_gdn_v", "value（split 后）", ["B", "seqlen", "NvH", "DvH"],
                    ["B", "seqlen", "NvH", "DvH"], "activation", "v [B,S,48,128]（:568）")
        self.op("g_gdn_split", "Qwen4ExpTextGatedDeltaNet.forward", "GDN QKV 拆分",
                "split", ["g_gdn_conv_seq"], ["g_gdn_q", "g_gdn_k", "g_gdn_v"], [],
                ["NkH", "Dk", "NvH", "DvH"],
                [src("556-568", "torch.split(mixed_qkv, [key_dim, key_dim, value_dim], -1); "
                                "view(B,S,-1,head_dim)")],
                "q/k 是 16 头、v 是 48 头；两者头数不同")
        self.tensor("g_gdn_beta", "beta = sigmoid(b)", ["B", "seqlen", "NvH"],
                    ["B", "seqlen", "NvH"], "activation", "范围 (0,1)（:570）")
        self.op("g_gdn_beta", "Qwen4ExpTextGatedDeltaNet.forward", "GDN beta sigmoid",
                "sigmoid", ["g_gdn_b"], ["g_gdn_beta"], [], ["NvH"],
                [src("570", "beta = b.sigmoid()")],
                "beta 控制新 value 写入状态的强度")
        self.tensor("g_gdn_g", "g = -exp(A_log)*softplus(a+dt_bias)",
                    ["B", "seqlen", "NvH"], ["B", "seqlen", "NvH"], "activation",
                    "每 token/头的 log decay，≤0；softplus 后乘 exp(A_log)（:571-572）")
        self.op("g_gdn_decay", "Qwen4ExpTextGatedDeltaNet.forward", "GDN decay 计算",
                "softplus+scale", ["g_gdn_a", "g_gdn_dt_bias", "g_gdn_a_log"],
                ["g_gdn_g"], [], ["NvH"],
                [src("571-572", "g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)")],
                "g ≤ 0 才能作为 exp 的 log decay；A_log/dt_bias 是每头参数")
        self.tensor("g_gdn_q_rep", "query.repeat_interleave(Gmul, dim=2)",
                    ["B", "seqlen", "NvH", "Dk"], ["B", "seqlen", "NvH", "Dk"],
                    "activation", "16→48 头（:573-575）")
        self.tensor("g_gdn_k_rep", "key.repeat_interleave(Gmul, dim=2)",
                    ["B", "seqlen", "NvH", "Dk"], ["B", "seqlen", "NvH", "Dk"],
                    "activation", "16→48 头（:573-575）")
        self.op("g_gdn_q_repeat", "Qwen4ExpTextGatedDeltaNet.forward",
                "GDN Q 头扩展", "repeat_interleave",
                ["g_gdn_q"], ["g_gdn_q_rep"], [], ["NkH", "NvH", "Gmul"],
                [src("573-575", "query = query.repeat_interleave(num_v_heads // num_k_heads, dim=2)")],
                "每个 q 头复制 3 份对应 3 个 v 头；不是权重共享")
        self.op("g_gdn_k_repeat", "Qwen4ExpTextGatedDeltaNet.forward",
                "GDN K 头扩展", "repeat_interleave",
                ["g_gdn_k"], ["g_gdn_k_rep"], [], ["NkH", "NvH", "Gmul"],
                [src("573-575", "key = key.repeat_interleave(num_v_heads // num_k_heads, dim=2)")],
                "与 q 同样的 16→48 头映射")
        self.tensor("g_gdn_q_hat", "l2norm(query)", ["B", "seqlen", "NvH", "Dk"],
                    ["B", "seqlen", "NvH", "Dk"], "activation",
                    "kernel 内 use_qk_l2norm_in_kernel=True，在 fp32 上 L2 归一（:296-298）")
        self.op("g_gdn_q_l2", "torch_chunk_gated_delta_rule / recurrent", "GDN Q L2Norm",
                "l2norm", ["g_gdn_q_rep"], ["g_gdn_q_hat"], [], ["Dk"],
                [src("296-298", "if use_qk_l2norm_in_kernel: query = l2norm(query, dim=-1, eps=1e-6)"),
                 src("247-250", "l2norm：x/sqrt(sum(x^2)+1e-6)")],
                "L2 归一在 kernel 内、fp32 上执行；不是 RMSNorm")
        self.tensor("g_gdn_k_hat", "l2norm(key)", ["B", "seqlen", "NvH", "Dk"],
                    ["B", "seqlen", "NvH", "Dk"], "activation", "L2 归一后的 key（:296-298）")
        self.op("g_gdn_k_l2", "torch_chunk_gated_delta_rule / recurrent", "GDN K L2Norm",
                "l2norm", ["g_gdn_k_rep"], ["g_gdn_k_hat"], [], ["Dk"],
                [src("296-298", "key = l2norm(key)"), src("247-250", "l2norm")],
                "q/k 都归一；缩放只作用于 q")
        self.tensor("g_gdn_q_scaled", "q * Dk**-0.5", ["B", "seqlen", "NvH", "Dk"],
                    ["B", "seqlen", "NvH", "Dk"], "activation",
                    "query 乘 1/sqrt(128)（:299-301）")
        self.op("g_gdn_q_scale", "torch_chunk_gated_delta_rule / recurrent", "GDN Q 缩放",
                "scale", ["g_gdn_q_hat"], ["g_gdn_q_scaled"], [], ["Dk"],
                [src("299-301", "scaling = query.shape[-1] ** -0.5; query = query * scaling")],
                "缩放是 128^-0.5；不是 QSA 的 256^-0.5")
        self.tensor("g_gdn_core_out", "core_attn_out（扫描输出）",
                    ["B", "seqlen", "NvH", "DvH"], ["B", "seqlen", "NvH", "DvH"],
                    "activation",
                    "delta-rule 扫描输出；recurrent 仅在 cache 已有状态且 seq_len==1 时走，"
                    "空缓存首次调用与多 token 续接走 chunk 路径（chunk_size=64）")
        self.tensor("g_gdn_state_new", "last_recurrent_state（本次扫描结果）",
                    ["B", "NvH", "Dk", "DvH"], ["B", "NvH", "Dk", "DvH"], "activation",
                    "本次扫描产生的新状态（fp32），随后写入 cache")
        self.op("g_gdn_scan", "torch_recurrent_gated_delta_rule / torch_chunk_gated_delta_rule",
                "GDN gated delta-rule 扫描（递归/分块）", "scan",
                ["g_gdn_q_scaled", "g_gdn_k_hat", "g_gdn_v", "g_gdn_g", "g_gdn_beta",
                 "g_gdn_state_pre"],
                ["g_gdn_core_out", "g_gdn_state_new"], [], ["NvH", "Dk", "DvH", "seqlen", "B"],
                [src("390-450", "torch_recurrent_gated_delta_rule：逐 token 状态更新"),
                 src("253-387", "torch_chunk_gated_delta_rule：chunk 64 的分块等价形式"),
                 src("578-603", "根据 use_precomputed_states and seq_len==1 选择 recurrent/chunk")],
                "真实语义（gated delta rule）：先衰减 A = S_{t-1} ⊙ exp(g_t)；"
                "mem = A^T k_t；delta = beta_t ⊙ (v_t - mem)；"
                "S_t = A + k_t delta^T；o_t = S_t^T q_t。"
                "误差项用已衰减状态 A（不是未衰减 S_{t-1}），方向是 A^T k。"
                "recurrent 仅在 cache 已有状态且 seq_len==1 时走；空缓存 P=0/S=1 没有 "
                "precomputed state，走 chunk。"
                "不是普通 Attention，也不能机械铺开 seqlen 个框；chunk 分块是同一递推的"
                "不同执行形式（chunk 含 64 长度 pad 与三角 mask）")
        self.op("g_gdn_state_write", "LinearAttentionLayer.update_recurrent_state",
                "GDN recurrent state 写入", "cache_update",
                ["g_gdn_state_pre", "g_gdn_state_new"], ["g_gdn_state_post"], [],
                ["B", "NvH", "Dk", "DvH"],
                [src("605-607", "cache_params.update_recurrent_state(last_recurrent_state, layer_idx)"),
                 self.srcf("cache_utils.py", "1141-1160", "update_recurrent_state：copy_ 到静态状态")],
                "新状态写回同一 cache 对象；读前/写后分成两个张量表达")
        self.tensor("g_gdn_o_normed", "self.norm(core_out, z)",
                    ["B", "seqlen", "NvH", "DvH"], ["B", "seqlen", "NvH", "DvH"],
                    "activation", "Gated RMSNorm：fp32 方差归一 × weight × sigmoid(z)（:171-187/:609-612）")
        self.op("g_gdn_o_norm", "Qwen4ExpTextRMSNormGated.forward", "GDN Gated RMSNorm",
                "rmsnorm-gated", ["g_gdn_core_out", "g_gdn_z"], ["g_gdn_o_normed"],
                ["g_gdn_norm_weight"], ["NvH", "DvH", "EpsN"],
                [src("171-187", "RMSNormGated：fp32 方差归一 * weight * activation(gate)"),
                 src("490-492", "activation=config.output_gate_type（sigmoid）"),
                 src("609-612", "core_attn_out = self.norm(core_attn_out.reshape(-1,Hv), "
                                "z.reshape(-1,Hv))")],
                "先按 head_dim 做 RMSNorm，再乘 sigmoid(z)；不是普通 RMSNorm")
        self.tensor("g_gdn_o_flat", "output.reshape(B,S,-1)", ["B", "seqlen", "Dz"],
                    ["B", "seqlen", "Dz"], "activation", "拼头（:610-613）")
        self.op("g_gdn_o_reshape", "Qwen4ExpTextGatedDeltaNet.forward", "GDN 拼头",
                "reshape", ["g_gdn_o_normed"], ["g_gdn_o_flat"], [], ["NvH", "DvH"],
                [src("613", "core_attn_out.reshape(batch_size, seq_len, -1)")],
                "把 48×128 拼成 6144")
        self.tensor("g_attn_out", "self.out_proj(core_attn_out)", ["B", "seqlen", "H"],
                    ["B", "seqlen", "H"], "activation", "Attention 输出（:615）")
        self.op("g_gdn_out_proj", "Qwen4ExpTextGatedDeltaNet.out_proj", "GDN o_proj",
                "linear", ["g_gdn_o_flat"], ["g_attn_out"], ["g_gdn_out_proj_weight"],
                ["Dz", "H", "B", "seqlen"],
                [src("493", "self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)"),
                 src("615", "output = self.out_proj(core_attn_out)")],
                "bias=False；输出回到 hidden_size=2560")

        # Gated Residual（attn 段）
        self._gated_residual_apply("attn", "g_attn_out", stream_source,
                                   "g_hidden_streams_mid")
        # Gated Residual（ffn 站点）
        self._gated_residual_site("ffn", "g_hidden_streams_mid")
        self._moe_ops("g_hc_ffn_mixed")
        self._gated_residual_apply("ffn", "g_mlp_out", "g_hidden_streams_mid",
                                   "g_hidden_streams_out")

    # ------------------------------------------------------------------
    # PLE
    # ------------------------------------------------------------------

    def _ple_ops(self, streams_in: str) -> None:
        v = self.values
        src = self.src
        self.op("g_ple_context_update", "LinearAttentionLayer.update_conv_state",
                "PLE n-gram 上下文写入", "cache_update",
                ["g_ple_context_pre", "g_ple_input_ids"], ["g_ple_context_post"], [],
                ["B", "Lctx", "seqlen"],
                [src("1130-1144", "input_ids_to_cache=...; update_conv_state(input_ids_to_cache, "
                                  "layer_idx, state_idx=2, conv_kernel_size=context_len)")],
                "上下文缓存保存最近 2 个 token id；初始按 eos 左填；与 GDN conv state 分开")
        self.tensor("g_ple_history", "torch.cat([previous_context, input_ids], -1)",
                    ["B", "seqlen+Lctx"], ["B", "seqlen", "Lctx"], "activation",
                    "上一上下文 + 当前 ids；ngram 在完整历史上 shift")
        self.op("g_ple_history", "Qwen4ExpTextNGramEmbedding.forward",
                "PLE token 历史拼接", "concat",
                ["g_ple_context_pre", "g_ple_input_ids"], ["g_ple_history"], [],
                ["B", "seqlen", "Lctx"],
                [src("1146-1147", "token_history = torch.cat([previous_context, input_ids], dim=-1)")],
                "当前 ids 的 n-gram 需要前 2 个 token；不把历史当成本次 token")
        self.tensor("g_ple_shift1", "_shift_right_ignore_eos(history, 1)",
                    ["B", "seqlen+Lctx"], ["B", "seqlen", "Lctx"], "activation",
                    "右移 1 位；EOS 后段重置为 eos（bigram 的第二元）")
        self.op("g_ple_shift1", "Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos",
                "PLE 右移 1（bigram）", "shift_right+eos",
                ["g_ple_history"], ["g_ple_shift1"], [], ["B", "seqlen", "Lctx", "Eos"],
                [src("1106-1120", "_shift_right_ignore_eos(shift=1)"),
                 src("1148", "shifted_tokens = [... for shift in range(ngram_size)]")],
                "EOS 切断 n-gram：跨 EOS 的位置填 eos，不串到上一段")
        self.tensor("g_ple_shift2", "_shift_right_ignore_eos(history, 2)",
                    ["B", "seqlen+Lctx"], ["B", "seqlen", "Lctx"], "activation",
                    "右移 2 位；EOS 后段重置为 eos（trigram 的第三元）")
        self.op("g_ple_shift2", "Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos",
                "PLE 右移 2（trigram）", "shift_right+eos",
                ["g_ple_history"], ["g_ple_shift2"], [], ["B", "seqlen", "Lctx", "Eos"],
                [src("1106-1120", "_shift_right_ignore_eos(shift=2)")],
                "shift=0 直接返回 token_history（源码早退），所以 bigram/trigram 的第一元"
                "复用 g_ple_history，不额外建算子")
        self.weight("g_ple_layer_multipliers", "self.ple_embedding.layer_multipliers",
                    ["Ngm"], ["Ngm"],
                    "每层 3 个 splitmix64 派生的乘子（:1039-1048/:1097-1099）；"
                    "bigram/trigram 的 token×乘子后 XOR")
        self.tensor("g_ple_bigram_ids", "bigram mixed_ids % head_vocab + offset",
                    ["B", "seqlen+Lctx", "Hng"], ["B", "seqlen", "Lctx", "Hng"], "activation",
                    "8 个 bigram 哈希头；每个头独立素数码表与偏移")
        self.op("g_ple_bigram", "Qwen4ExpTextNGramEmbedding.forward",
                "PLE bigram 哈希", "xor+mod+offset",
                ["g_ple_history", "g_ple_shift1", "g_ple_layer_multipliers"],
                ["g_ple_bigram_ids"], [], ["Hng", "Vng"],
                [src("1151-1163", "for ngram in range(2, ngram_size+1): mixed_ids = "
                                  "shifted[0]*m0 XOR shifted[pos]*mpos; "
                                  "ngram_ids = mixed_ids % head_vocab_sizes + offsets")],
                "2-gram：当前 token 与前一 token 的哈希；8 个头共享乘子但码表不同")
        self.tensor("g_ple_trigram_ids", "trigram mixed_ids % head_vocab + offset",
                    ["B", "seqlen+Lctx", "Hng"], ["B", "seqlen", "Lctx", "Hng"], "activation",
                    "8 个 trigram 哈希头")
        self.op("g_ple_trigram", "Qwen4ExpTextNGramEmbedding.forward",
                "PLE trigram 哈希", "xor+mod+offset",
                ["g_ple_history", "g_ple_shift1", "g_ple_shift2",
                 "g_ple_layer_multipliers"],
                ["g_ple_trigram_ids"], [], ["Hng", "Vng"],
                [src("1151-1163", "for ngram in 2..3: XOR shifted[0..ngram-1] 的乘子")],
                "3-gram：当前 + 前 1 + 前 2；与 bigram 是同一头集合的两阶")
        self.tensor("g_ple_ids_all", "torch.cat(blocks, -1)",
                    ["B", "seqlen+Lctx", "Ngh"], ["B", "seqlen", "Lctx", "Ngh"], "activation",
                    "16 个哈希 id（8 bigram + 8 trigram）")
        self.op("g_ple_concat_ids", "Qwen4ExpTextNGramEmbedding.forward",
                "PLE 哈希 id 拼接", "concat",
                ["g_ple_bigram_ids", "g_ple_trigram_ids"], ["g_ple_ids_all"], [],
                ["Ngh", "Hng"],
                [src("1165", "ngram_ids = torch.cat(blocks, dim=-1)")],
                "固定顺序 bigram 8 个、trigram 8 个")
        self.tensor("g_ple_ids", "ngram_ids[:, -seqlen:]",
                    ["B", "seqlen", "Ngh"], ["B", "seqlen", "Ngh"], "activation",
                    "只取本次 seqlen 位置的 id；历史 2 位用于 shift")
        self.op("g_ple_slice", "Qwen4ExpTextNGramEmbedding.forward",
                "PLE 只取当前 token", "slice",
                ["g_ple_ids_all"], ["g_ple_ids"], [], ["B", "seqlen", "Ngh"],
                [src("1165", "ngram_ids = torch.cat(blocks, dim=-1)[:, -input_ids.shape[1]:]")],
                "历史 2 位只参与哈希，不产生自己的 PLE 输出")
        self.weight("g_ple_embedding_weight", "self.ple_embedding.ngram_embedding.weight",
                    ["Vngpad", "Dgh"], ["Vngpad", "Dgh"],
                    "每个哈希头的 160 维 embedding 拼在一个 pad 词表里（:1104）")
        self.tensor("g_ple_emb_heads", "self.ngram_embedding(ngram_ids)",
                    ["B", "seqlen", "Ngh", "Dgh"], ["B", "seqlen", "Ngh", "Dgh"],
                    "activation", "按 16 个哈希 id 查表；每头 160 维")
        self.op("g_ple_gather", "Qwen4ExpTextNGramEmbedding.forward",
                "PLE n-gram 查表", "embedding+gather",
                ["g_ple_ids"], ["g_ple_emb_heads"], ["g_ple_embedding_weight"],
                ["B", "seqlen", "Ngh", "Dgh", "Vngpad"],
                [src("1168-1171", "self.ngram_embedding(ngram_ids).flatten(-2)")],
                "id 已含头偏移，所以一次查表得到 16×160；flatten 成 2560")
        self.tensor("g_ple_emb", "embeddings.flatten(-2)", ["B", "seqlen", "Dple"],
                    ["B", "seqlen", "Dple"], "activation", "PLE embedding [B,S,2560]")
        self.op("g_ple_flatten", "Qwen4ExpTextNGramEmbedding.forward",
                "PLE embedding 拼宽", "flatten",
                ["g_ple_emb_heads"], ["g_ple_emb"], [], ["Dple", "Ngh", "Dgh"],
                [src("1171", ".flatten(-2)")],
                "16×160=2560；不是把 16 个头的 embedding 相加")
        self.weight("g_ple_key_weight", "self.ple.key_proj.weight", ["Dqkv", "Dple"],
                    ["Dqkv", "Dple"], "key_proj 2560→10240，bias=False（:1193）")
        self.weight("g_ple_value_weight", "self.ple.value_proj.weight", ["H", "Dple"],
                    ["H", "Dple"], "value_proj 2560→2560，bias=False（:1194）")
        self.weight("g_ple_norm_key_weight", "self.ple.norm_key.weight", ["Hc×H"],
                    ["Hc", "H"], "分组 RMSNorm key 的 weight（:1195）")
        self.weight("g_ple_norm_query_weight", "self.ple.norm_query.weight", ["Hc×H"],
                    ["Hc", "H"], "分组 RMSNorm query 的 weight（:1196）")
        self.weight("g_ple_norm_conv_weight", "self.ple.norm_conv.weight", ["Hc×H"],
                    ["Hc", "H"], "分组 RMSNorm conv 的 weight（:1197）")
        self.weight("g_ple_conv_weight", "self.ple.conv1d.weight",
                    ["Dqkv", "1", "Kple"], ["Dqkv", "Kple"],
                    "depthwise Conv1d [10240,1,4]，dilation=3，bias=False（:1198-1205）")
        self.tensor("g_ple_key_flat", "self.key_proj(embeddings)",
                    ["B", "seqlen", "Dqkv"], ["B", "seqlen", "Dqkv"], "activation",
                    "key 投影 2560→10240")
        self.op("g_ple_key_proj", "Qwen4ExpTextPLELayer.key_proj", "PLE key 投影",
                "linear", ["g_ple_emb"], ["g_ple_key_flat"], ["g_ple_key_weight"],
                ["Dple", "Dqkv", "B", "seqlen"],
                [src("1193", "self.key_proj = nn.Linear(ple_embed_dim, hc_hidden_size, bias=False)"),
                 src("1234", "key_normed = self.norm_key(self.key_proj(embeddings))")],
                "key 是每路残差流的 gate 键，不是 attention key")
        self.tensor("g_ple_key_normed_flat", "self.norm_key(key_flat)",
                    ["B", "seqlen", "Dqkv"], ["B", "seqlen", "Dqkv"], "activation",
                    "分组 RMSNorm（每组 2560）")
        self.op("g_ple_norm_key", "Qwen4ExpTextPLELayer.norm_key", "PLE key RMSNorm",
                "rmsnorm-grouped", ["g_ple_key_flat"], ["g_ple_key_normed_flat"],
                ["g_ple_norm_key_weight"], ["Hc", "H", "EpsN"],
                [src("1234", "self.norm_key(self.key_proj(embeddings))"),
                 src("152-156", "group_size 分组 RMSNorm")],
                "4 个流各自归一；key 的最后一维随后与 query 对齐")
        self.tensor("g_ple_key_normed", "key_normed.unflatten(-1,(Hc,H))",
                    ["B", "seqlen", "Hc", "H"], ["B", "seqlen", "Hc", "H"], "activation",
                    "unflatten 成 4 路流的 key")
        self.op("g_ple_key_view", "Qwen4ExpTextPLELayer.forward", "PLE key 分组成型",
                "view", ["g_ple_key_normed_flat"], ["g_ple_key_normed"], [],
                ["Hc", "H"], [src("1234", "key_normed = ... .unflatten(-1, (hc_count, hidden_size))")],
                "把 10240 拆成 4×2560")
        self.tensor("g_ple_value", "self.value_proj(embeddings)",
                    ["B", "seqlen", "H"], ["B", "seqlen", "H"], "activation",
                    "value 投影 2560→2560（共享给 4 路流）")
        self.op("g_ple_value_proj", "Qwen4ExpTextPLELayer.value_proj", "PLE value 投影",
                "linear", ["g_ple_emb"], ["g_ple_value"], ["g_ple_value_weight"],
                ["Dple", "H", "B", "seqlen"],
                [src("1194", "self.value_proj = nn.Linear(ple_embed_dim, hidden_size, bias=False)"),
                 src("1235", "value = self.value_proj(embeddings)")],
                "value 是单流 2560，随后广播到 4 路流")
        self.tensor("g_ple_query_normed_flat", "self.norm_query(streams_flat)",
                    ["B", "seqlen", "Dqkv"], ["B", "seqlen", "Hc", "H"], "activation",
                    "层间流是 flat [B,S,Hc*H]；query 先对 flat 做分组 RMSNorm")
        self.op("g_ple_norm_query", "Qwen4ExpTextPLELayer.norm_query", "PLE query RMSNorm",
                "rmsnorm-grouped", [streams_in], ["g_ple_query_normed_flat"],
                ["g_ple_norm_query_weight"], ["Hc", "H", "EpsN"],
                [src("1236", "query_normed = self.norm_query(hidden_states).unflatten(-1, (hc_count, hidden_size))")],
                "query 就是当前 flat 4 路残差流；norm 先 flat，随后 unflatten")
        self.tensor("g_ple_query_normed", "query_normed.unflatten(-1,(Hc,H))",
                    ["B", "seqlen", "Hc", "H"], ["B", "seqlen", "Hc", "H"], "activation",
                    "4D query 视图，用于与 key 的 4D 视图点积")
        self.op("g_ple_query_view", "Qwen4ExpTextPLELayer.forward",
                "PLE query unflatten", "unflatten",
                ["g_ple_query_normed_flat"], ["g_ple_query_normed"], [],
                ["Hc", "H"], [src("1236", ".unflatten(-1, (hc_count, hidden_size))")],
                "层间保持 flat；4D 只在 gate 点积时出现")
        self.tensor("g_ple_gate", "gate = (key_normed*query_normed).sum(-1)/sqrt(H)",
                    ["B", "seqlen", "Hc", "1"], ["B", "seqlen", "Hc"], "activation",
                    "每流一个标量门（点积 / √2560）")
        self.op("g_ple_gate", "Qwen4ExpTextPLELayer.forward", "PLE 流级 gate",
                "dot+scale", ["g_ple_key_normed", "g_ple_query_normed"],
                ["g_ple_gate"], [], ["Hc", "H", "B", "seqlen"],
                [src("1237", "gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(hidden_size)")],
                "每路流一个标量；不是 attention score，也不跨 token 归一")
        self.tensor("g_ple_gate_signed", "gate.abs().clamp_min(1e-6).sqrt()*sign(gate)",
                    ["B", "seqlen", "Hc", "1"], ["B", "seqlen", "Hc"], "activation",
                    "保号平方根，避免 0 附近梯度问题")
        self.op("g_ple_gate_sign", "Qwen4ExpTextPLELayer.forward", "PLE gate 保号缩放",
                "abs+sqrt+sign", ["g_ple_gate"], ["g_ple_gate_signed"], [], ["Hc"],
                [src("1238", "gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()")],
                "不是 clamp 到正数：保留符号，再 sqrt")
        self.tensor("g_ple_gated_value", "sigmoid(gate)*value.unsqueeze(-2)",
                    ["B", "seqlen", "Hc", "H"], ["B", "seqlen", "Hc", "H"], "activation",
                    "value 广播到 4 路流，每流乘自己的 sigmoid 门")
        self.op("g_ple_gated_value", "Qwen4ExpTextPLELayer.forward",
                "PLE value 门控", "broadcast-mul+sigmoid",
                ["g_ple_gate_signed", "g_ple_value"], ["g_ple_gated_value"], [],
                ["Hc", "H", "B", "seqlen"],
                [src("1239", "gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)")],
                "value 是共享单流，门是每流标量")
        self.tensor("g_ple_gated_value_flat", "gated_value.flatten(-2)",
                    ["B", "seqlen", "Dqkv"], ["B", "seqlen", "Hc", "H"], "activation",
                    "4D 门控值显式 flatten 回 10240，供分组 RMSNorm")
        self.op("g_ple_gated_value_flat", "Qwen4ExpTextPLELayer.forward",
                "PLE gated value flatten", "flatten",
                ["g_ple_gated_value"], ["g_ple_gated_value_flat"], [],
                ["Hc", "H", "B", "seqlen"],
                [src("1240", "gated_value.flatten(-2)")],
                "不能把 4D 值直接送进按最后一维分组的 conv norm")
        self.tensor("g_ple_value_normed", "self.norm_conv(gated_value_flat)",
                    ["B", "seqlen", "Dqkv"], ["B", "seqlen", "Hc", "H"], "activation",
                    "分组 RMSNorm（每组 2560），用于短卷积输入")
        self.op("g_ple_norm_conv", "Qwen4ExpTextPLELayer.norm_conv", "PLE conv RMSNorm",
                "rmsnorm-grouped", ["g_ple_gated_value_flat"], ["g_ple_value_normed"],
                ["g_ple_norm_conv_weight"], ["Hc", "H", "EpsN"],
                [src("1240-1241", "gated_value_normed = self.norm_conv(gated_value.flatten(-2))")],
                "先 flatten 10240 再分组归一；卷积前不乘 mask 之外的额外门")
        self.tensor("g_ple_gated_value_masked", "apply_mask(gated_value)",
                    ["B", "seqlen", "Hc", "H"], ["B", "seqlen", "Hc", "H"], "activation",
                    "padding 位置置零（conv_mask；:1242-1244）")
        self.tensor("g_ple_value_normed_masked", "apply_mask(gated_value_normed)",
                    ["B", "seqlen", "Dqkv"], ["B", "seqlen", "Dqkv"], "activation",
                    "卷积输入同样按 padding mask 置零")
        self.op("g_ple_mask", "Qwen4ExpTextPLELayer.forward", "PLE padding 置零",
                "broadcast-mul",
                ["g_ple_gated_value", "g_ple_value_normed", "g_conv_mask"],
                ["g_ple_gated_value_masked", "g_ple_value_normed_masked"], [],
                ["B", "seqlen", "Hc", "H"],
                [src("1242-1244", "gated_value = apply_mask_to_padding_states(...); "
                                  "gated_value_normed = apply_mask_to_padding_states(...)")],
                "两个张量都置零；全 1 mask 时 no-op")
        self.op("g_ple_conv_state_update", "LinearAttentionLayer.update_conv_state",
                "PLE 短卷积 state 写入", "cache_update",
                ["g_ple_conv_state_pre", "g_ple_value_normed_masked"],
                ["g_ple_conv_state_post"], [], ["B", "Dqkv", "Lpconv", "seqlen"],
                [src("1211-1214", "past_key_values.update_conv_state(hidden_states, layer_idx, "
                                  "state_idx=1, conv_kernel_size=9)"),
                 self.srcf("cache_utils.py", "1099-1149", "update_conv_state")],
                "PLE 的短卷积 state 与 GDN conv state 分开（state_idx=1）")
        self.tensor("g_ple_conv_out", "silu(self.conv1d(padded_normed))",
                    ["B", "seqlen", "Dqkv"], ["B", "seqlen", "Dqkv"], "activation",
                    "dilation=3、kernel=4 的 depthwise 因果卷积，输出取最后 seqlen 步")
        self.op("g_ple_conv", "Qwen4ExpTextPLELayer._short_conv", "PLE 膨胀短卷积",
                "conv1d", ["g_ple_conv_state_pre", "g_ple_value_normed_masked"],
                ["g_ple_conv_out"], ["g_ple_conv_weight"], ["Dqkv", "Kple", "Dild", "seqlen"],
                [src("1207-1224", "_short_conv：pad 9、conv1d(dilation=3)、silu、transpose")],
                "depthwise、dilation=3：每输出位置看 4 个相距 3 的历史 token；"
                "有效感受野 10")
        self.tensor("g_ple_out", "gated_value + short_conv(normed)",
                    ["B", "seqlen", "Dqkv"], ["B", "seqlen", "Dqkv"], "activation",
                    "PLE 输出 [B,S,10240]；包含 skip 的 gated value")
        self.tensor("g_ple_gated_value_masked_flat", "apply_mask(gated_value).flatten(-2)",
                    ["B", "seqlen", "Dqkv"], ["B", "seqlen", "Hc", "H"], "activation",
                    "4D 门控值先 flatten，才能与 flat 卷积输出相加")
        self.op("g_ple_value_mask_flatten", "Qwen4ExpTextPLELayer.forward",
                "PLE 门控值 flatten（加回前）", "flatten",
                ["g_ple_gated_value_masked"], ["g_ple_gated_value_masked_flat"], [],
                ["Hc", "H", "B", "seqlen"],
                [src("1245", "gated_value + self._short_conv(...)")],
                "源码在加回前把 gated_value 保持为 flat；本图显式 flatten")
        self.op("g_ple_add", "Qwen4ExpTextPLELayer.forward", "PLE 短卷积残差相加",
                "add", ["g_ple_gated_value_masked_flat", "g_ple_conv_out"],
                ["g_ple_out"], [], ["Dqkv", "B", "seqlen", "Hc"],
                [src("1245", "output = gated_value + self._short_conv(gated_value_normed, past_key_values)")],
                "gated_value 与短卷积输出都是 flat [B,S,10240]；不能 4D+3D 直接加")
        self.tensor("g_hidden_streams_ple", "hidden_states + ple_output（flat）",
                    ["B", "seqlen", "Hc×H"], ["B", "seqlen", "Hc", "H"], "activation",
                    "PLE 输出 [B,S,10240] 加回 flat 4 路残差流（:1274-1277）")
        self.op("g_ple_stream_add", "Qwen4ExpTextDecoderLayer.forward",
                "PLE 输出加回残差流", "add",
                [streams_in, "g_ple_out"], ["g_hidden_streams_ple"], [],
                ["B", "seqlen", "Hc", "H"],
                [src("1274-1277", "if self.ple is not None: hidden_states = hidden_states + "
                                  "self.ple(hidden_states, ple_input_ids, past_key_values, conv_mask=conv_mask)")],
                "PLE 是层输入的前置加性支路，不是替换 hidden；加法后仍进入 attn_hc")

    def _build_cache_notes(self) -> None:
        self.cache_notes = [
            "层缓存类：config.layer_types 在该层为 linear_attention（GDN），dispatch 到 "
            "LinearAttentionLayer（cache_utils:1285-1294）。",
            "GDN conv state：update_conv_state 保存最近 Kconv=4 个 token 的 qkv 通道，"
            "形状 [B,10240,4]；首次调用按需左补零，之后与本次拼接并只保留最后 4"
            "（cache_utils:1099-1149）。",
            "GDN recurrent state：delta-rule 状态 [B,48,128,128]，update_recurrent_state "
            "复制 fp32 新状态（modeling:605-607；cache_utils:1141-1160）。",
            "两套 GDN 状态大小**固定**，不随 kvcachelen/seqlen 增长；本模型不会为 GDN 层"
            "额外保留随历史增长的 KV cache。kvcachelen 只影响同网的 QSA 层与本层在序列中的位置。",
            "GDN 的因果性不是靠 4D causal mask：padding mask 只置零 padding，因果结构由"
            "因果短卷积 + delta-rule 的递推顺序（及其分块等价形式）保证。",
            "recurrent 仅在 cache 已有状态且 seq_len==1 时走；空缓存首次调用（含 P=0/S=1）"
            "与多 token 续接走 chunk 路径（chunk_size=64，尾部 padding 与三角 mask）；"
            "两条路径是同一递推的不同执行形式。",
        ]
        if self.ple:
            self.cache_notes.extend([
                "PLE 层（layer 1）额外使用两个固定大小状态：state_idx=1 的短卷积 state "
                "[B,10240,9]，state_idx=2 的 n-gram 上下文 [B,2]（modeling:1126-1144/"
                "1211-1214）。",
                "number_of_conv_states=3 是配置级常量，所有 linear_attention 层都会分配 3 个"
                "槽；但只有 layer 1 的 PLE 使用 state_idx=1/2，其余 GDN 层只用 state_idx=0。",
                "PLE 的 n-gram 上下文按 EOS 切断：跨 EOS 的 shift 结果填 eos，不把上一段"
                "文本拼进本段 n-gram（modeling:1106-1120）。",
            ])

    def _build_coverage_notes(self) -> None:
        v = self.values
        gdn = [i for i in range(v.num_hidden_layers) if v.layer_type(i) == "linear_attention"]
        qsa = [i for i in range(v.num_hidden_layers) if v.layer_type(i) == "indexed_attention"]
        self.coverage_notes = [
            "已表达：Gated Residual 两站点（attn/ffn）的分组 RMSNorm、10240→320→10240 "
            "逐元素读门、四路加权 mean、4 路写门展开相加；"
            + ("N-gram PLE（上下文/哈希/查表/短卷积/加回 4 路流）；" if self.ple else "")
            + "GDN 的 q/k/v 合并投影、depthwise 因果短卷积与 conv state、beta、decay、"
            "q/k repeat/L2Norm/缩放、gated delta-rule 扫描（chunk/recurrent）、"
            "recurrent state 写回、Gated RMSNorm、o_proj；TopK MoE + 并行共享专家。",
            f"真实层归属：GDN 层 {len(gdn)} 个；QSA 层 {len(qsa)} 个"
            f"（{qsa}）；PLE 层 {v.ple_layer_ids}（one-indexed）→ 0-based "
            f"{[i - 1 for i in v.ple_layer_ids]}。本 Block 只代表 GDN"
            f"{'+PLE' if self.ple else ''}+MoE 组模板。",
            "根配置声明 bfloat16 与 FP8 无（本模型没有 quantization_config）：本图只表达 "
            "HF 逻辑 eager 路径，不分析设备 kernel、dtype 数值或权重内容。",
            "未覆盖：MTP 1 层（配置声明 full_attention，但本地 HF 实现没有 MTP 模块，"
            "加载时忽略 mtp.*）、视觉 27 层、generate/采样循环、loss/aux_loss、"
            "output_router_logits 辅助输出、torch.compile/梯度检查点、数值等价。",
            "数据相关量（N_hit、N_e、路由实际分布、选中的 n-gram 头）保留未知；"
            "不根据权重推断 MoE 实际路由，不默认 0。专家只有一份模板，不按 512 个专家复制图。",
            "GDN 的 kernel 级分块实现（chunk 64 的三角 mask/递推重排）按语义操作与循环模板"
            "表达，不伪装成普通 Attention，也不铺开 seqlen 个递推步。",
        ]


class _QsaBlockBuilder(_BlockBuilder):
    """Qwen Sparse Attention（QSA，micro-block indexer）+ TopK MoE 的真实单层结构。"""

    kind = "qsa"

    def __init__(self, values: HfModelValues, layer_idx: int):
        super().__init__(values, layer_idx)
        self.ple = False

    def build(self) -> dict:
        v = self.values
        title = (
            f"Qwen3.8-Flash-Next Decoder Block（layer_idx={self.layer_idx}，"
            f"Qwen Sparse Attention（QSA） + TopK MoE；{v.num_hidden_layers} 层主干"
            "代表模板；同一结构由 seqlen/kvcachelen 场景参数化）"
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
        src = self.src
        self._common_parameters()
        self.param("Nh", "Nh", "num_attention_heads QSA query 头数", v.num_attention_heads, None,
                   "config.num_attention_heads=24；q_proj 输出 2×24×256=12288（q + gate）",
                   [cfg("text_config.num_attention_heads", "文本子配置字段"),
                    src("817-824", "self.num_key_value_groups / q_proj 2*Nh*Dh")])
        self.param("Nkv", "Nkv", "num_key_value_heads QSA KV 头数", v.num_key_value_heads, None,
                   "config.num_key_value_heads=2；repeat_kv 复制 12 份到 24 头",
                   [cfg("text_config.num_key_value_heads", "文本子配置字段"),
                    src("818", "num_key_value_groups = Nh // Nkv")])
        self.param("Dh", "Dh", "head_dim QSA 头维", v.head_dim, None,
                   "config.head_dim=256；q/k/v 最后一维；缩放 256^-0.5=1/16",
                   [cfg("text_config.head_dim", "文本子配置字段"),
                    src("817-819", "self.head_dim / self.scaling")])
        self.param("Drot", "D_rot", "RoPE 旋转维（partial）", v.rotary_dim,
                   "D_rot = Dh * Pfrac",
                   "256×0.25=64；只转 q/k 的前 64 维，后 192 维是 nope",
                   [cfg("text_config.rope_parameters.partial_rotary_factor", "文本子配置字段"),
                    src("108-115", "dim = int(head_dim * partial_rotary_factor)"),
                    src("649-659", "apply_rotary_pos_emb 只处理前 rotary_dim 维，其余 cat 回")])
        self.param("Pfrac", "partial_rotary_factor", "partial_rotary_factor", v.partial_rotary_factor,
                   None, "config.rope_parameters.partial_rotary_factor=0.25",
                   [cfg("text_config.rope_parameters.partial_rotary_factor", "文本子配置字段")])
        self.param("Theta", "Theta", "rope_theta", v.rope_theta, None,
                   "config.rope_parameters.rope_theta=10,000,000",
                   [cfg("text_config.rope_parameters.rope_theta", "文本子配置字段")])
        self.param("Nrep", "Nrep", "repeat_kv 的复制倍数", v.num_attention_heads //
                   v.num_key_value_heads, "Nrep = Nh // Nkv",
                   "24//2=12；K/V 从 2 头复制到 24 头（GQA）",
                   [src("818", "self.num_key_value_groups = config.num_attention_heads // "
                               "config.num_key_value_heads"),
                    src("773-782", "repeat_kv")])
        self.param("Dqw", "D_qw", "QSA q 拼接宽度",
                   v.num_attention_heads * v.head_dim, "D_qw = Nh * Dh",
                   "24×256=6144；o_proj 输入宽度，也是 gate 的宽度",
                   [src("831-833", "o_proj = Linear(Nh*head_dim, hidden_size)")])
        self.param("Dkv", "D_kv", "QSA K/V 拼接宽度",
                   v.num_key_value_heads * v.head_dim, "D_kv = Nkv * Dh",
                   "2×256=512；k_proj/v_proj 输出宽度",
                   [src("825-830", "k_proj/v_proj = Linear(H, Nkv*head_dim)")])
        self.param("Iqh", "Iqh", "indexer_n_heads QSA indexer query 头数", v.indexer_n_heads,
                   None, "config.indexer_n_heads=4",
                   [cfg("text_config.indexer_n_heads", "文本子配置字段")])
        self.param("Ikh", "Ikh", "indexer_kv_heads QSA indexer key 头数", v.indexer_kv_heads,
                   None, "config.indexer_kv_heads=1（源码 validate_architecture 强制）",
                   [cfg("text_config.indexer_kv_heads", "文本子配置字段"),
                    src("220-221", "if self.indexer_kv_heads != 1: raise")])
        self.param("Id", "Id", "indexer_head_dim indexer 头维", v.indexer_head_dim, None,
                   "config.indexer_head_dim=128；q/k 各 RMSNorm(128)",
                   [cfg("text_config.indexer_head_dim", "文本子配置字段")])
        self.param("Ki", "Ki", "indexer_budget 每 query 的原始 token 预算", v.indexer_budget,
                   None, "config.indexer_budget=2048；按 4 token/块展开",
                   [cfg("text_config.indexer_budget", "文本子配置字段"),
                    src("673-675", "self.token_budget / self.block_topk = budget // compress_ratio")])
        self.param("Kc", "Kc", "indexer_compress_ratio 每块压缩的 token 数",
                   v.indexer_compress_ratio, None,
                   "config.indexer_compress_ratio=4；4 个连续可见 token 均值压成 1 块",
                   [cfg("text_config.indexer_compress_ratio", "文本子配置字段"),
                    src("674-675", "self.compress_ratio / self.block_topk")])
        self.param("Btop", "Btop", "QSA 每 query 最多选择的块数", v.indexer_block_topk,
                   "Btop = Ki // Kc",
                   "2048//4=512；源码 topk(min(block_topk, num_complete_blocks))",
                   [src("675", "self.block_topk = self.token_budget // self.compress_ratio"),
                    src("748", "scores.topk(min(self.block_topk, num_complete_blocks), dim=0)")])
        self.param("kvcachelen", "kvcachelen", "调用前逻辑历史 token 数（kvcachelen）", None,
                   "P = past_key_values.get_seq_length()（无 cache 时 0）",
                   "空缓存 Prefill 取 0；Decode 为调用前已处理历史；不是物理保留条目数或"
                   "缓存字节数。QSA 的 K/V 与 indexer key 都线性保留全部历史",
                   [src("1423", "past_seen_tokens = past_key_values.get_seq_length()"),
                    src("870-871", "past_key_values.update(key_states, value_states, layer_idx)")])
        self.param("Lkv", "L_kv", "本次可读主 KV 长度", None, "L_kv = kvcachelen + seqlen",
                   "update 后 keys/values 线性追加全部历史；attention 在全部 Lkv 上做"
                   "masked softmax（稀疏 mask 只收窄可见范围）",
                   [src("870-871", "past_key_values.update(key_states, value_states, layer_idx)"),
                    self.srcf("cache_utils.py", "132-153", "DynamicLayer.update：concat")])
        self.param("Lidx", "L_idx", "本次可读 indexer key 缓存长度", None,
                   "L_idx = kvcachelen + seqlen（与 Lkv 同式，但来自 indexer key 缓存）",
                   "indexer key [B,Lidx,128]；每个 QSA 层独立实例化 indexer",
                   [src("707-708", "raw_keys = past_key_values.update_indexer(raw_keys, layer_idx)"),
                    self.srcf("cache_utils.py", "350-363", "DynamicIndexedLayer.update_indexer")])
        self.param("Npool", "N_pool", "最后一个 query（j=seqlen-1）的完整块数", None,
                   "N_pool = L_idx // Kc",
                   "仅用于 Ksel_last/场景摘要，**不是所有 query 的 pool/norm/RoPE/score/topk "
                   "实际 shape**；逐 query 的完整块数见 N_q。不完整尾块作为 tail 追加",
                   [src("720-750", "per query: num_complete_blocks = visible_len // compress_ratio")])
        self.param("Bsel", "B_sel", "最后一个 query（j=seqlen-1）选中的块数", None,
                   "B_sel = min(Btop, N_pool)",
                   "仅用于 Ksel_last/场景摘要，不是所有 query 的实际 top-k 宽度；"
                   "逐 query 的选中块数见 B_q",
                   [src("748", "scores.topk(min(self.block_topk, num_complete_blocks))")])
        self.param("Nq", "N_q", "第 j 个 query 的完整 micro-block 数", None,
                   "N_q = V_q // Kc",
                   "源码在每个 batch/query 的可见索引内计算；S8/P0 时依次为 0,0,0,1,1,1,1,2；"
                   "N_q=0 时 pool/norm/RoPE/score/topk 不执行，tail 仍追加",
                   [src("720-733", "for query_idx: local_visible_indices = nonzero(...); "
                                   "num_complete_blocks = visible_len // compress_ratio")])
        self.param("Bq", "B_q", "第 j 个 query 实际选择的块数", None,
                   "B_q = min(Btop, N_q)",
                   "per query；S4/P4092 时 N_q=1023,1023,1023,1024，B_q 均为 512"
                   "（min(Btop=512, N_q) 按 block 预算截断）；不是全长 Npool/Bsel",
                   [src("748", "scores.topk(min(self.block_topk, num_complete_blocks), dim=0)")])
        self.param("Rtail", "R_tail", "第 j 个 query 的不完整 tail token 数", None,
                   "R_tail = V_q % Kc",
                   "0..Kc-1；N_q=0 时全部可见 token 都走 tail；tail 直接追加，不参与块打分",
                   [src("753-754", "tail = local_visible_indices[num_complete_blocks*compress_ratio:]")])
        self.param("j", "j", "query 在本次 seqlen 中的局部索引", None,
                   "j = 0..seqlen-1（query 维；不枚举具体 token）",
                   "无 padding 时第 j 个 query 的因果可见长度 V_q=P+j+1；"
                   "不给定 j 时 V_q/K_sel 如实未知，不能把最后一个 query 的值冒充所有 query",
                   [src("720-755", "for query_idx in range(seq_length): visible_token_indices"
                                    "[batch_idx,0,query_idx]"),
                    src("929-933", "get_visible_tokens 的 q_positions/kv_positions")],
                   unknown_reason="query 索引未给定：j 是 0..seqlen-1 的符号，不枚举具体 token",
                   domain={"kind": "integer", "min": 0, "max_symbol": "seqlen"})
        self.param("Vq", "V_q", "第 j 个 query 的因果可见 token 数", None,
                   "V_q = kvcachelen + j + 1",
                   "无 padding 时即 P+j+1；Ksel 依赖 V_q，不依赖全长 L_idx",
                   [src("720-725", "local_visible_indices = nonzero(visible_token_indices"
                                    "[batch_idx,0,query_idx])")])
        self.param("Ksel", "K_sel", "第 j 个 query 的稀疏可见 key 数（块展开 + tail）",
                   None, "K_sel = min(V_q, Ki + (V_q % Kc))",
                   "逐 query：块展开 Kc×Bsel 个 token + 当前不完整 tail（≤Kc-1）；"
                   "无 padding 且 V_q≤Ki 时等于 V_q；超过 Ki 时最多 Ki+Kc-1=2051。"
                   "不是全长统一值；具体选中哪些块仍由打分数据决定",
                   [src("727-755", "block_token_indices 展开 + tail + selected_token_indices 填充")])
        self.param("Ksel_last", "K_sel_last", "最后一个 query（j=seqlen-1）的 Ksel", None,
                   "K_sel_last = min(L_idx, Ki + (L_idx % Kc))",
                   "只有 j=seqlen-1 才退化为全长 Lidx；这是“最后 query”值，"
                   "不能称为每 query 值、一般最大值或统一上界",
                   [src("720-725", "最后一个 query 的可见索引覆盖全部历史（无 padding）")])
        self.param("Kout", "K_out", "indexer 选择索引的输出宽度（含 -1 填充）", None,
                   "K_out = Ki + Kc - 1",
                   "selected_token_indices 固定宽度 token_budget + compress_ratio - 1 = 2051；"
                   "无效位置填 -1，scatter 时落到最后一列并丢弃",
                   [src("714-719", "torch.full((B,S,self.token_budget + self.compress_ratio - 1), -1)"),
                    src("757-764", "scatter -1 到 kv_length 后 [..., :kv_length]")])

    def _build_derivations(self) -> None:
        super()._build_derivations()
        premise = ("按固定配置 + 「空态起步、连续正常执行」前提推导；"
                   "不覆盖裁剪/回滚后的自定义缓存快照")
        self.derivation("L_kv", "本次可读主 KV 长度", _EXPR_LKV,
                        f"kvcachelen 为调用前逻辑历史；K/V 缓存无压缩条目。{premise}")
        self.derivation("L_idx", "本次可读 indexer key 缓存长度", _EXPR_LIDX,
                        f"与主 KV 同式；12 个 QSA 层各自有 indexer key 缓存。{premise}")
        self.derivation("N_pool", "最后一个 query（j=seqlen-1）的完整块数", _EXPR_NPOOL,
                        f"仅用于 Ksel_last/场景摘要；不是逐 query 的 pool/norm/score shape。{premise}")
        self.derivation("B_sel", "最后一个 query（j=seqlen-1）选中的块数", _EXPR_BSEL,
                        f"仅用于 Ksel_last/场景摘要；不是逐 query 的 top-k 宽度。{premise}")
        self.derivation("V_q", "第 j 个 query 的因果可见 token 数", _EXPR_VQ,
                        f"query 局部索引 j 只在该 query 的因果可见范围内；{premise}")
        self.derivation("N_q", "第 j 个 query 的完整 micro-block 数", _EXPR_NQ,
                        f"源码在每个 batch/query 的可见索引内计算；N_q=0 时跳过 pool/"
                        f"norm/RoPE/score/topk。{premise}")
        self.derivation("B_q", "第 j 个 query 实际选择的块数", _EXPR_BQ,
                        f"B_q=min(Btop,N_q)；逐 query 的 top-k 宽度。{premise}")
        self.derivation("R_tail", "第 j 个 query 的不完整 tail token 数", _EXPR_RTAIL,
                        f"R_tail=V_q%Kc（0..3）；tail 直接追加，不参与块打分。{premise}")
        self.derivation("K_sel", "第 j 个 query 的稀疏可见 key 数（块展开 + tail）",
                        _EXPR_KSEL,
                        f"逐 query：V_q≤Ki 时覆盖全部可见 token；超过 Ki 时最多 "
                        f"Ki+Kc-1=2051。不是全长统一值。{premise}")
        self.derivation("K_sel_last", "最后一个 query（j=seqlen-1）的 Ksel",
                        _EXPR_KSEL_LAST,
                        f"仅当 j=seqlen-1 时 V_q=L_idx；这是最后 query 值，"
                        f"不是所有 query 的统一值。{premise}")
        self.derivation("K_out", "indexer 选择索引输出宽度（含 -1 填充）", _EXPR_KOUT,
                        f"源码固定宽度 Ki+Kc-1=2051；无效位置 -1。{premise}")

    def _build_boundaries_and_cache(self) -> None:
        v = self.values
        self.tensor(
            "g_hidden_streams_in", "hidden_states（4 路流层输入，flat）",
            ["B", "seqlen", "Hc×H"], ["B", "seqlen", "Hc", "H"], "boundary",
            "上一层输出或网络入口的 4 路 repeat（modeling:1471）；层间实际边界是 "
            "[B,S,Hc*H] flat，读门/写门内部才 unflatten 成 4D（:1015-1018）")
        self.tensor(
            "g_causal_mask", "attention_mask（4D additive causal mask，调用方/网络构建）",
            ["B", "1", "seqlen", "Lkv"], ["B", "seqlen", "Lkv"], "boundary",
            "create_causal_mask 的输出：0=可见、min=屏蔽（eager float）；"
            "allow_is_causal_skip=False 强制生成（modeling:1446-1460）")
        self.tensor(
            "g_position_cos", "position_embeddings[0]（当前 seqlen 的 cos）",
            ["B", "seqlen", "Drot"], ["B", "seqlen", "Drot"], "boundary",
            "mRoPE 重组后只保留当前 seqlen；text-only 时三行 grid 相同，"
            "等价于 1D RoPE 的前 64 维（modeling:76-140/853-854）")
        self.tensor(
            "g_position_sin", "position_embeddings[1]（当前 seqlen 的 sin）",
            ["B", "seqlen", "Drot"], ["B", "seqlen", "Drot"], "boundary",
            "与 cos 同源；partial RoPE 只作用于前 Drot=64 维")
        self.tensor(
            "g_qsa_k_pre", "past_key_values.layers[layer_idx].keys（调用前）",
            ["B", "Nkv", "kvcachelen", "Dh"], ["B", "Nkv", "kvcachelen", "Dh"], "cache",
            "DynamicIndexedLayer.keys；QSA 的 K 缓存 [B,2,L,256]（cache_utils:113-153）")
        self.tensor(
            "g_qsa_v_pre", "past_key_values.layers[layer_idx].values（调用前）",
            ["B", "Nkv", "kvcachelen", "Dh"], ["B", "Nkv", "kvcachelen", "Dh"], "cache",
            "DynamicIndexedLayer.values；QSA 的 V 缓存 [B,2,L,256]")
        self.tensor(
            "g_qsa_k_post", "self.keys（调用后）", ["B", "Nkv", "Lkv", "Dh"],
            ["B", "Nkv", "Lkv", "Dh"], "cache",
            "update 线性追加当前 K；Lkv=kvcachelen+seqlen")
        self.tensor(
            "g_qsa_v_post", "self.values（调用后）", ["B", "Nkv", "Lkv", "Dh"],
            ["B", "Nkv", "Lkv", "Dh"], "cache",
            "update 线性追加当前 V；Lkv=kvcachelen+seqlen")
        self.tensor(
            "g_qsa_indexer_keys_pre", "past_key_values.layers[layer_idx].indexer_keys（调用前）",
            ["B", "kvcachelen", "Id"], ["B", "kvcachelen", "Id"], "cache",
            "DynamicIndexedLayer.indexer_keys；单头 [B,L,128]，不是 pooled block key")
        self.tensor(
            "g_qsa_indexer_keys_post", "self.indexer_keys（调用后）", ["B", "Lidx", "Id"],
            ["B", "Lidx", "Id"], "cache",
            "update_indexer 按序列拼接；Lidx=kvcachelen+seqlen")

    def _build_flow(self) -> None:
        v = self.values
        src, cfg = self.src, self.cfg
        streams_in = "g_hidden_streams_in"

        # Gated Residual（attn 站点）
        self._gated_residual_site("attn", streams_in)
        mixed = "g_hc_attn_mixed"

        # 1) Q 投影 + gate
        self.weight("g_qsa_q_proj_weight", "self.self_attn.q_proj.weight",
                    ["2×Nh×Dh", "H"], ["Nh", "Dh", "H"],
                    "nn.Linear(2560, 2×24×256=12288, bias=False)（:822-824）")
        self.tensor("g_qsa_q_flat", "self.q_proj(mixed)",
                    ["B", "seqlen", "2×Nh×Dh"], ["B", "seqlen", "Nh", "Dh"], "activation",
                    "q 与 gate 打包在 12288 维里")
        self.op("g_qsa_q_proj", "Qwen4ExpTextAttention.q_proj", "QSA Q 投影（含 gate）",
                "linear", [mixed], ["g_qsa_q_flat"], ["g_qsa_q_proj_weight"],
                ["H", "Nh", "Dh", "B", "seqlen"],
                [src("822-824", "self.q_proj = nn.Linear(hidden_size, "
                                "num_attention_heads * head_dim * 2, bias=False)"),
                 src("858-860", "query_states, gate = torch.chunk(self.q_proj(...).view(...), 2, -1)")],
                "一次投影出 q 与输出门；不是 q 后再做 gate 投影")
        self.tensor("g_qsa_q_gate_view", "q_proj(...).view(B,S,Nh,2*Dh)",
                    ["B", "seqlen", "Nh", "2×Dh"], ["B", "seqlen", "Nh", "Dh"],
                    "activation", "按头把 12288 拆成每头 512 维")
        self.op("g_qsa_q_gate_view", "Qwen4ExpTextAttention.forward",
                "QSA Q/gate 分头布局", "view",
                ["g_qsa_q_flat"], ["g_qsa_q_gate_view"], [], ["Nh", "Dh", "B", "seqlen"],
                [src("858-859", "self.q_proj(hidden_states).view(*input_shape, -1, head_dim*2)")],
                "每头 512 = q 256 + gate 256")
        self.tensor("g_qsa_q_raw", "query_states（chunk 后）", ["B", "seqlen", "Nh", "Dh"],
                    ["B", "seqlen", "Nh", "Dh"], "activation", "q 的 256 维")
        self.tensor("g_qsa_gate_raw", "gate（chunk 后）", ["B", "seqlen", "Nh", "Dh"],
                    ["B", "seqlen", "Nh", "Dh"], "activation", "输出门的 256 维")
        self.op("g_qsa_q_gate_chunk", "Qwen4ExpTextAttention.forward",
                "QSA Q/gate 拆分", "chunk",
                ["g_qsa_q_gate_view"], ["g_qsa_q_raw", "g_qsa_gate_raw"], [],
                ["Nh", "Dh"],
                [src("858-860", "torch.chunk(q_proj(...).view(...), 2, dim=-1)")],
                "沿最后一维均分；两者都是 24×256")
        self.tensor("g_qsa_gate", "gate.reshape(B,S,Nh*Dh)", ["B", "seqlen", "Dqw"],
                    ["B", "seqlen", "Nh", "Dh"], "activation",
                    "输出门拼成 6144 维，与 attn 输出逐元素相乘")
        self.op("g_qsa_gate_view", "Qwen4ExpTextAttention.forward",
                "QSA gate 拼宽", "reshape", ["g_qsa_gate_raw"], ["g_qsa_gate"],
                [], ["Nh", "Dh"], [src("861", "gate = gate.reshape(*input_shape, -1)")],
                "gate 不做 softmax；逐元素 sigmoid 后乘 attention 输出")
        self.weight("g_qsa_q_norm_weight", "self.self_attn.q_norm.weight", ["Dh"], ["Dh"],
                    "QSA q RMSNorm 的 weight（256 维，eps=1e-6）（:834）")
        self.tensor("g_qsa_q_normed", "self.q_norm(query_states)",
                    ["B", "seqlen", "Nh", "Dh"], ["B", "seqlen", "Nh", "Dh"], "activation",
                    "每头 256 维 RMSNorm")
        self.op("g_qsa_q_norm", "Qwen4ExpTextAttention.q_norm", "QSA Q RMSNorm",
                "rmsnorm", ["g_qsa_q_raw"], ["g_qsa_q_normed"], ["g_qsa_q_norm_weight"],
                ["Dh", "EpsN", "B", "seqlen", "Nh"],
                [src("834", "self.q_norm = Qwen4ExpTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)"),
                 src("863", "query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1,2)")],
                "Q 用 RMSNorm 而不是 LayerNorm；V 不归一")
        self.tensor("g_qsa_q", "query_states.transpose(1,2)", ["B", "Nh", "seqlen", "Dh"],
                    ["B", "Nh", "seqlen", "Dh"], "activation",
                    "转成 [B,24,S,256] 供 attention")
        self.op("g_qsa_q_transpose", "Qwen4ExpTextAttention.forward",
                "QSA Q 转置", "transpose", ["g_qsa_q_normed"], ["g_qsa_q"], [],
                ["Nh", "Dh", "B", "seqlen"],
                [src("863", ".transpose(1, 2)")],
                "头维在前；这是布局变换，不改变数据")

        # 2) K/V 投影 + K norm
        self.weight("g_qsa_k_proj_weight", "self.self_attn.k_proj.weight",
                    ["Dkv", "H"], ["Dkv", "H"],
                    "nn.Linear(2560, 2×256=512, bias=False)（:825-827）")
        self.weight("g_qsa_v_proj_weight", "self.self_attn.v_proj.weight",
                    ["Dkv", "H"], ["Dkv", "H"],
                    "nn.Linear(2560, 2×256=512, bias=False)（:828-830）")
        self.weight("g_qsa_k_norm_weight", "self.self_attn.k_norm.weight", ["Dh"], ["Dh"],
                    "QSA k RMSNorm 的 weight（256 维，eps=1e-6）（:835）")
        self.tensor("g_qsa_k_flat", "self.k_proj(mixed)", ["B", "seqlen", "Dkv"],
                    ["B", "seqlen", "Dkv"], "activation", "K 投影输出 [B,S,512]")
        self.op("g_qsa_k_proj", "Qwen4ExpTextAttention.k_proj", "QSA K 投影", "linear",
                [mixed], ["g_qsa_k_flat"], ["g_qsa_k_proj_weight"],
                ["H", "Dkv", "B", "seqlen"],
                [src("825-827", "self.k_proj = nn.Linear(hidden_size, "
                                "num_key_value_heads * head_dim, bias=False)"),
                 src("864", "key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))")],
                "2 个 KV 头；GQA 由 downstream repeat_kv 扩展")
        self.tensor("g_qsa_k_view", "k_proj(...).view(B,S,Nkv,Dh)",
                    ["B", "seqlen", "Nkv", "Dh"], ["B", "seqlen", "Nkv", "Dh"],
                    "activation", "把 512 拆成 2×256")
        self.op("g_qsa_k_view", "Qwen4ExpTextAttention.forward", "QSA K 分头",
                "view", ["g_qsa_k_flat"], ["g_qsa_k_view"], [], ["Nkv", "Dh"],
                [src("864", ".view(hidden_shape)")],
                "hidden_shape 用 -1 推断出 Nkv=2")
        self.tensor("g_qsa_k_normed", "self.k_norm(k)", ["B", "seqlen", "Nkv", "Dh"],
                    ["B", "seqlen", "Nkv", "Dh"], "activation", "每头 256 维 RMSNorm")
        self.op("g_qsa_k_norm", "Qwen4ExpTextAttention.k_norm", "QSA K RMSNorm",
                "rmsnorm", ["g_qsa_k_view"], ["g_qsa_k_normed"], ["g_qsa_k_norm_weight"],
                ["Dh", "EpsN", "B", "seqlen", "Nkv"],
                [src("835", "self.k_norm = Qwen4ExpTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)")],
                "K 用 RMSNorm；V 不归一")
        self.tensor("g_qsa_k", "key_states.transpose(1,2)", ["B", "Nkv", "seqlen", "Dh"],
                    ["B", "Nkv", "seqlen", "Dh"], "activation",
                    "转成 [B,2,S,256] 供 RoPE 与缓存")
        self.op("g_qsa_k_transpose", "Qwen4ExpTextAttention.forward",
                "QSA K 转置", "transpose", ["g_qsa_k_normed"], ["g_qsa_k"], [],
                ["Nkv", "Dh", "B", "seqlen"], [src("864", ".transpose(1, 2)")],
                "头维在前；RoPE 在 [B,Nkv,S,Dh] 上做")
        self.tensor("g_qsa_v_flat", "self.v_proj(mixed)", ["B", "seqlen", "Dkv"],
                    ["B", "seqlen", "Dkv"], "activation", "V 投影输出 [B,S,512]")
        self.op("g_qsa_v_proj", "Qwen4ExpTextAttention.v_proj", "QSA V 投影", "linear",
                [mixed], ["g_qsa_v_flat"], ["g_qsa_v_proj_weight"],
                ["H", "Dkv", "B", "seqlen"],
                [src("828-830", "self.v_proj = nn.Linear(hidden_size, "
                                "num_key_value_heads * head_dim, bias=False)")],
                "V 与 K 同宽；没有 norm")
        self.tensor("g_qsa_v", "value_states.view(...).transpose(1,2)",
                    ["B", "Nkv", "seqlen", "Dh"], ["B", "Nkv", "seqlen", "Dh"],
                    "activation", "转成 [B,2,S,256] 供缓存与 repeat_kv")
        self.op("g_qsa_v_view", "Qwen4ExpTextAttention.forward", "QSA V 分头/转置",
                "view+transpose", ["g_qsa_v_flat"], ["g_qsa_v"], [],
                ["Nkv", "Dh", "B", "seqlen"],
                [src("865", "value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)")],
                "V 没有 RMSNorm；与 K 的 q_norm/k_norm 不同")

        # 3) partial RoPE
        self.tensor("g_qsa_q_rope", "apply_rotary_pos_emb(q, cos, sin)",
                    ["B", "Nh", "seqlen", "Dh"], ["B", "Nh", "seqlen", "Dh"], "activation",
                    "只转前 64 维；后 192 维 nope 原样 cat 回（:626-661）")
        self.op("g_qsa_q_rope", "apply_rotary_pos_emb", "QSA Q partial RoPE",
                "rope-partial", ["g_qsa_q", "g_position_cos", "g_position_sin"],
                ["g_qsa_q_rope"], [], ["Nh", "Dh", "Drot", "B", "seqlen"],
                [src("626-653", "apply_rotary_pos_emb：q_rope 前 rotary_dim 维旋转，"
                                "q_nope 原样 cat 回"),
                 src("867-868", "query_states, key_states = apply_rotary_pos_emb(...)")],
                "partial_rotary_factor=0.25 → 只转前 64/256 维，其余 192 维是 nope 段；不是全维 RoPE")
        self.tensor("g_qsa_k_rope", "apply_rotary_pos_emb(k, cos, sin)",
                    ["B", "Nkv", "seqlen", "Dh"], ["B", "Nkv", "seqlen", "Dh"], "activation",
                    "K 同样只转前 64 维；与 Q 共用 cos/sin")
        self.op("g_qsa_k_rope", "apply_rotary_pos_emb", "QSA K partial RoPE",
                "rope-partial", ["g_qsa_k", "g_position_cos", "g_position_sin"],
                ["g_qsa_k_rope"], [], ["Nkv", "Dh", "Drot", "B", "seqlen"],
                [src("655-659", "k_rope 前 rotary_dim 维旋转，k_nope 原样 cat 回")],
                "K 只转前 64 维；cos/sin 与 Q 相同（text-only 时三行 grid 相同）")

        # 4) 主 K/V 缓存
        self.op("g_qsa_cache_update", "DynamicIndexedLayer.update",
                "QSA K/V 缓存写入", "cache_update",
                ["g_qsa_k_pre", "g_qsa_v_pre", "g_qsa_k_rope", "g_qsa_v"],
                ["g_qsa_k_post", "g_qsa_v_post"], [],
                ["B", "Nkv", "Lkv", "Dh", "seqlen"],
                [src("870-871", "key_states, value_states = past_key_values.update("
                                "key_states, value_states, self.layer_idx)"),
                 self.srcf("cache_utils.py", "113-153", "DynamicLayer.update：concat + 返回全部 Lkv")],
                "K/V 逐层独立；QSA 的缓存随历史线性增长（与 GDN 固定状态不同）")

        # 5) indexer
        self.weight("g_qsa_ix_qk_weight", "self.self_attn.indexer.index_qk_proj.weight",
                    ["(Iqh+Ikh)×Id", "H"], ["Iqh", "Ikh", "Id", "H"],
                    "nn.Linear(2560, (4+1)×128=640, bias=False)（:676-680）")
        self.weight("g_qsa_ix_q_norm_weight", "self.self_attn.indexer.q_layernorm.weight",
                    ["Id"], ["Id"], "indexer q RMSNorm 的 weight（:681）")
        self.weight("g_qsa_ix_k_norm_weight", "self.self_attn.indexer.k_layernorm.weight",
                    ["Id"], ["Id"], "indexer k RMSNorm 的 weight（:682）")
        self.tensor("g_qsa_ix_qk", "self.indexer.index_qk_proj(mixed)",
                    ["B", "seqlen", "(Iqh+Ikh)×Id"], ["B", "seqlen", "Iqh", "Ikh", "Id"],
                    "activation", "640 维：4 个 query 头 + 1 个共享 key 头")
        self.op("g_qsa_ix_qk_proj", "Qwen4ExpTextQSAIndexer.index_qk_proj",
                "QSA indexer QK 投影", "linear",
                [mixed], ["g_qsa_ix_qk"], ["g_qsa_ix_qk_weight"],
                ["H", "Iqh", "Ikh", "Id", "B", "seqlen"],
                [src("676-680", "self.index_qk_proj = nn.Linear(hidden_size, "
                                "(index_n_heads + index_kv_heads) * index_head_dim, bias=False)"),
                 src("697-703", "q, token_k = torch.split(qk, [Iqh*Id, Ikh*Id], -1)")],
                "indexer 输入是层 hidden（不是 q_residual）；与主 Q/K 投影独立")
        self.tensor("g_qsa_ix_q_raw", "q.view(B,S,Iqh,Id)", ["B", "seqlen", "Iqh", "Id"],
                    ["B", "seqlen", "Iqh", "Id"], "activation", "4 个 indexer query 头")
        self.tensor("g_qsa_ix_token_k", "token_k.view(B,S,Ikh,Id).squeeze(2)",
                    ["B", "seqlen", "Id"], ["B", "seqlen", "Id"], "activation",
                    "单头共享 key；写入 indexer 缓存的原始逐 token key")
        self.op("g_qsa_ix_split", "Qwen4ExpTextQSAIndexer.forward",
                "QSA indexer Q/K 拆分", "split+squeeze",
                ["g_qsa_ix_qk"], ["g_qsa_ix_q_raw", "g_qsa_ix_token_k"], [],
                ["Iqh", "Ikh", "Id"],
                [src("698-703", "q, token_k = torch.split(...); q.reshape(*hidden_shape); "
                                "token_k.reshape(*hidden_shape).squeeze(2)")],
                "4 个 query 头共享 1 个 key 头（MQA 式 indexer）")
        self.tensor("g_qsa_ix_q_normed", "self.q_layernorm(q)", ["B", "seqlen", "Iqh", "Id"],
                    ["B", "seqlen", "Iqh", "Id"], "activation",
                    "每头 128 维 RMSNorm")
        self.op("g_qsa_ix_q_norm", "Qwen4ExpTextQSAIndexer.q_layernorm",
                "QSA indexer Q RMSNorm", "rmsnorm",
                ["g_qsa_ix_q_raw"], ["g_qsa_ix_q_normed"],
                ["g_qsa_ix_q_norm_weight"], ["Id", "EpsN", "Iqh"],
                [src("681", "self.q_layernorm = Qwen4ExpTextRMSNorm(index_head_dim, eps=...)"),
                 src("704", "q = self.q_layernorm(q)")],
                "indexer q 也是 RMSNorm(128)，不是 LayerNorm")
        self.tensor("g_qsa_ix_q_rope", "apply_rotary_pos_emb(q, current_cos, current_sin)",
                    ["B", "seqlen", "Iqh", "Id"], ["B", "seqlen", "Iqh", "Id"],
                    "activation", "只转前 64 维（Drot=64 < Id=128）")
        self.op("g_qsa_ix_q_rope", "Qwen4ExpTextQSAIndexer.forward",
                "QSA indexer Q partial RoPE", "rope-partial",
                ["g_qsa_ix_q_normed", "g_position_cos", "g_position_sin"],
                ["g_qsa_ix_q_rope"], [], ["Iqh", "Id", "Drot"],
                [src("705", "q = apply_rotary_pos_emb(q, cos=current_cos, sin=current_sin, unsqueeze_dim=2)"),
                 src("693-695", "current_cos/current_sin = full_cos[:, -seqlen:, :]")],
                "indexer q 只取当前 seqlen 的 cos/sin；partial RoPE 64/128")
        self.op("g_qsa_ix_cache_update", "DynamicIndexedLayer.update_indexer",
                "QSA indexer key 缓存写入", "cache_update",
                ["g_qsa_indexer_keys_pre", "g_qsa_ix_token_k"],
                ["g_qsa_indexer_keys_post"], [], ["B", "Lidx", "Id"],
                [src("707-708", "raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)"),
                 self.srcf("cache_utils.py", "350-363", "update_indexer：concat 单头 key")],
                "缓存的是未归一化、未池化的逐 token raw key；block key 在每次调用内由均值+norm+RoPE 得到")
        self.tensor("g_qsa_ix_visible", "visible_token_indices = (attention_mask==0)",
                    ["B", "seqlen", "Lidx"], ["B", "seqlen", "Lidx"], "mask",
                    "每个 query 的因果可见 token（无 padding 时第 j 个 query 有 V_q 个）；"
                    "来自 4D causal mask，不是重新构图")
        self.op("g_qsa_ix_visible", "Qwen4ExpTextQSAIndexer.forward",
                "QSA indexer 可见 token", "mask->indices",
                ["g_causal_mask"], ["g_qsa_ix_visible"], [], ["B", "seqlen", "Lidx"],
                [src("710-712", "visible_token_indices = attention_mask if bool else "
                                "attention_mask == 0"),
                 src("722-724", "local_visible_indices = nonzero(visible_token_indices"
                                "[batch_idx,0,query_idx])")],
                "QSA 的因果性由主 causal mask 提供；indexer 只在该可见集合内选块")
        self.tensor("g_qsa_ix_has_blocks", "N_q > 0（本 query 是否有完整块）",
                    ["B", "seqlen"], ["B", "seqlen"], "mask",
                    "N_q=floor(V_q/Kc)；为 0 时 pool/norm/RoPE/score/topk 分支不执行，"
                    "selected blocks 为空，tail 仍按源码追加")
        self.op("g_qsa_ix_query_guard", "Qwen4ExpTextQSAIndexer.forward",
                "QSA 逐 query 零块守卫", "guard(V_q >= Kc)",
                ["g_qsa_ix_visible"], ["g_qsa_ix_has_blocks"], [],
                ["B", "seqlen", "Kc"],
                [src("720-733", "for query_idx: local_visible_indices = nonzero(...); "
                                "num_complete_blocks = visible_len // compress_ratio")],
                "每个 batch/query 独立计算；N_q=0 时后续 pool/norm/RoPE/score/topk 跳过，"
                "不是用全长 Npool/Bsel 冒充")
        self.tensor("g_qsa_ix_block_tokens", "local_visible_indices.view(num_blocks, Kc)",
                    ["B", "seqlen", "Nq", "Kc"], ["B", "seqlen", "Nq", "Kc"],
                    "activation",
                    "只取前 N_q*Kc 个可见 token；每 Kc=4 个一块；N_q 是逐 query 局部维，"
                    "N_q=0 时为空；不是全长 N_pool 上界")
        self.op("g_qsa_ix_block_view", "Qwen4ExpTextQSAIndexer.forward",
                "QSA micro-block 分组（逐 query）", "view",
                ["g_qsa_ix_visible", "g_qsa_ix_has_blocks"],
                ["g_qsa_ix_block_tokens"], [],
                ["B", "seqlen", "Nq", "Kc"],
                [src("725-733", "num_complete_blocks = visible_len // compress_ratio; "
                                "block_token_indices = visible[:num_complete*Kc].view(...)")],
                "micro-block 是连续 4 个可见 token；不是滑动窗口，也不是单 token 选择")
        self.tensor("g_qsa_ix_pooled_keys", "key_groups.mean(dim=1)",
                    ["B", "seqlen", "Nq", "Id"], ["B", "seqlen", "Nq", "Id"],
                    "activation", "块内 4 个 raw key 的均值（不是 softmax 加权）")
        self.op("g_qsa_ix_pool", "Qwen4ExpTextQSAIndexer.forward",
                "QSA micro-block 均值池化（逐 query）", "mean+index_select",
                ["g_qsa_indexer_keys_post", "g_qsa_ix_block_tokens",
                 "g_qsa_ix_has_blocks"],
                ["g_qsa_ix_pooled_keys"], [], ["Nq", "Kc", "Id"],
                [src("732-734", "key_groups = raw_keys.index_select(0, block_token_indices.flatten()); "
                                "pooled_keys = key_groups.float().mean(dim=1)")],
                "均值池化，不是 softmax/加权；块内 token 不能单独被选中")
        self.tensor("g_qsa_ix_pooled_normed", "self.k_layernorm(pooled_keys)",
                    ["B", "seqlen", "Nq", "Id"], ["B", "seqlen", "Nq", "Id"],
                    "activation", "块 key 的 RMSNorm(128)；只对 pooled key，不是逐 token")
        self.op("g_qsa_ix_pool_norm", "Qwen4ExpTextQSAIndexer.k_layernorm",
                "QSA block key RMSNorm（逐 query）", "rmsnorm",
                ["g_qsa_ix_pooled_keys", "g_qsa_ix_has_blocks"],
                ["g_qsa_ix_pooled_normed"],
                ["g_qsa_ix_k_norm_weight"], ["Id", "EpsN", "Nq"],
                [src("735", "pooled_keys = self.k_layernorm(pooled_keys)")],
                "只对 pooled key 做 RMSNorm(128)；indexer 不做逐 token key 归一化，"
                "因此不存在与之共权重；缓存里存的是未归一化 raw key")
        self.tensor("g_qsa_ix_block_rope", "block_key_states = apply_rotary_pos_emb("
                    "pooled_keys, group_starts)",
                    ["B", "seqlen", "Nq", "Id"], ["B", "seqlen", "Nq", "Id"],
                    "activation", "用块首 token 的位置做前 64 维 RoPE")
        self.op("g_qsa_ix_block_rope", "Qwen4ExpTextQSAIndexer.forward",
                "QSA block key partial RoPE（逐 query）", "rope-partial",
                ["g_qsa_ix_pooled_normed", "g_position_cos", "g_position_sin",
                 "g_qsa_ix_block_tokens", "g_qsa_ix_has_blocks"],
                ["g_qsa_ix_block_rope"], [], ["Nq", "Id", "Drot"],
                [src("736-741", "group_starts = block_token_indices[:,0]; "
                                "block_key_states = apply_rotary_pos_emb(pooled_keys.unsqueeze(1), "
                                "cos=full_cos.index_select(0, group_starts), ...)")],
                "块 key 只转前 64 维；RoPE 位置是块首 token，不是块内平均位置")
        self.tensor("g_qsa_ix_scores", "relu(q·k).sum(-1)/sqrt(Id)",
                    ["B", "seqlen", "Nq"], ["B", "seqlen", "Nq"], "activation",
                    "每个 query 对该 query 的 N_q 个完整块打分；4 个 indexer 头求和")
        self.op("g_qsa_ix_scores", "Qwen4ExpTextQSAIndexer.forward",
                "QSA indexer 块打分（逐 query）", "matmul+relu+sum+scale",
                ["g_qsa_ix_q_rope", "g_qsa_ix_block_rope", "g_qsa_ix_has_blocks"],
                ["g_qsa_ix_scores"], [], ["Iqh", "Id", "Nq"],
                [src("743-746", "scores = matmul(q[batch,query].float(), "
                                "block_key_states.float().T).transpose(-1,-2); "
                                "scores = relu(scores).sum(-1) / sqrt(index_head_dim)")],
                "relu 后按 indexer 头 sum(-1)，再除 sqrt(128)；不是 softmax 概率归一/加权 gate")
        self.tensor("g_qsa_ix_block_idx", "topk(min(Btop, N_q))",
                    ["B", "seqlen", "Bq"], ["B", "seqlen", "Bq"], "activation",
                    "选中的块索引；按分数降序；N_q=0 时为空")
        self.op("g_qsa_ix_block_topk", "Qwen4ExpTextQSAIndexer.forward",
                "QSA indexer 块 top-k（逐 query）", "topk",
                ["g_qsa_ix_scores", "g_qsa_ix_has_blocks"],
                ["g_qsa_ix_block_idx"], [], ["Bq", "Nq"],
                [src("748", "selected_block_indices = scores.topk(min(block_topk, "
                            "num_complete_blocks), dim=0).indices")],
                "每 query 最多 512 个块 = 2048 token；宽度是 B_q=min(512,N_q)，"
                "不是全长 B_sel")
        self.tensor("g_qsa_ix_selected_tokens", "block_token_indices[selected].flatten()",
                    ["B", "seqlen", "Bq×Kc"], ["B", "seqlen", "Bq", "Kc"],
                    "activation", "选中块展开成原始 token 索引（每块 4 个）；B_q=0 时为空")
        self.op("g_qsa_ix_select_blocks", "Qwen4ExpTextQSAIndexer.forward",
                "QSA 选中块展开（逐 query）", "index_select+flatten",
                ["g_qsa_ix_block_tokens", "g_qsa_ix_block_idx",
                 "g_qsa_ix_has_blocks"],
                ["g_qsa_ix_selected_tokens"], [], ["Bq", "Kc"],
                [src("750", "selected_tokens = block_token_indices.index_select(0, "
                            "selected_block_indices).flatten()")],
                "块选择等价于选 4×B_q 个 token；B_q 逐 query，不是全长 B_sel")
        self.tensor("g_qsa_ix_tail", "local_visible_indices[N_q*Kc:]",
                    ["B", "seqlen", "Rtail"], ["B", "seqlen", "Rtail"], "activation",
                    "当前不完整块的剩余 token（R_tail=V_q%Kc，0..3）；N_q=0 时全部可见 token")
        self.op("g_qsa_ix_tail", "Qwen4ExpTextQSAIndexer.forward",
                "QSA 不完整 tail 追加", "slice",
                ["g_qsa_ix_visible"], ["g_qsa_ix_tail"], [], ["Kc", "Rtail"],
                [src("753-754", "tail = local_visible_indices[num_complete_blocks*compress_ratio:]; "
                                "selected_tokens = cat([selected_tokens, tail])")],
                "tail 是当前 query 可见但不足一块的 token；N_q=0 时 tail=V_q，"
                "即使没有完整块 mask 仍按 tail 选择")
        self.tensor("g_qsa_ix_selected", "selected_tokens（块展开 + tail）",
                    ["B", "seqlen", "Ksel"], ["B", "seqlen", "Ksel"], "activation",
                    "每 query 的实际选择数 Ksel；随 query 位置变化")
        self.op("g_qsa_ix_concat", "Qwen4ExpTextQSAIndexer.forward",
                "QSA 选择结果拼接", "concat",
                ["g_qsa_ix_selected_tokens", "g_qsa_ix_tail"], ["g_qsa_ix_selected"],
                [], ["Ksel", "Bq", "Rtail", "Kc"],
                [src("754", "selected_tokens = torch.cat([selected_tokens, tail]).to(int32)")],
                "块展开在前、tail 在后；块展开宽度 B_q*Kc，tail 宽度 R_tail；"
                "N_q=0 时前半为空、tail 全部保留")
        self.tensor("g_qsa_ix_padded", "selected_token_indices（pad -1 到 Kout）",
                    ["B", "seqlen", "Kout"], ["B", "seqlen", "Kout"], "activation",
                    "固定宽度 2051，空位 -1")
        self.op("g_qsa_ix_pad", "Qwen4ExpTextQSAIndexer.forward",
                "QSA 选择索引 pad", "pad",
                ["g_qsa_ix_selected"], ["g_qsa_ix_padded"], [], ["Kout", "Ksel"],
                [src("714-719", "selected_token_indices = torch.full((B,S,Kout), -1)"),
                 src("755", "selected_token_indices[batch,query,:numel] = selected_tokens")],
                "不是把 Ksel 当成固定 2051；有效项才是选中 token")
        self.tensor("g_qsa_ix_mask_bool", "selected_token_mask（bool，True=可见）",
                    ["B", "1", "seqlen", "Lkv"], ["B", "seqlen", "Lkv"], "mask",
                    "zeros [B,S,Lkv+1] scatter True 后丢弃最后一列，再 unsqueeze(1)")
        self.op("g_qsa_ix_mask_bool", "Qwen4ExpTextQSAIndexer.forward",
                "QSA selected mask scatter（bool）", "scatter+unsqueeze",
                ["g_qsa_ix_padded"], ["g_qsa_ix_mask_bool"], [],
                ["B", "seqlen", "Lkv", "Kout"],
                [src("757-764", "selected_token_mask = zeros(..., kv_length+1); "
                                "scatter(-1, where(idx>=0, idx, kv_length), True)[..., :kv_length]"
                                ".unsqueeze(1)")],
                "True=可见；-1 落到最后一列后被丢弃，不产生假可见位置")
        self.tensor("g_qsa_ix_mask_float", "torch.where(selected_token_mask, 0.0, min_dtype)",
                    ["B", "1", "seqlen", "Lkv"], ["B", "seqlen", "Lkv"], "mask",
                    "eager float 路径：True→0.0（可见），False→finfo(dtype).min（屏蔽）")
        self.op("g_qsa_ix_mask_float", "Qwen4ExpTextAttention.forward",
                "QSA selected mask bool→eager additive", "where+cast",
                ["g_qsa_ix_mask_bool"], ["g_qsa_ix_mask_float"], [],
                ["B", "seqlen", "Lkv"],
                [src("766-769", "if attention_mask.is_floating_point(): "
                                "min_dtype = torch.finfo(attention_mask.dtype).min; "
                                "selected_token_mask = torch.where(selected_token_mask, "
                                "attention_mask.new_zeros(()), min_dtype)")],
                "本图分析 eager float 分支：先把 bool selected mask 转成 0.0/min，"
                "再与 causal mask 相加；bool/SDPA 分支才是相与，不能跳过转换直接相加")
        self.tensor("g_qsa_mask_combined", "causal_mask + selected_token_mask(float)",
                    ["B", "1", "seqlen", "Lkv"], ["B", "seqlen", "Lkv"], "mask",
                    "0 可见 / min 屏蔽；两个 additive mask 相加")
        self.op("g_qsa_mask_combine", "Qwen4ExpTextAttention.forward",
                "QSA 稀疏 mask 与 causal mask 合并", "add",
                ["g_causal_mask", "g_qsa_ix_mask_float"], ["g_qsa_mask_combined"], [],
                ["B", "seqlen", "Lkv"],
                [src("848-851", "if attention_mask.is_floating_point(): "
                                "attention_mask = attention_mask + selected_token_mask; "
                                "else: attention_mask & selected_token_mask")],
                "QSA 的稀疏选择只收窄主 causal mask，不会让被因果屏蔽的位置变可见")

        # 6) eager attention core
        self.tensor("g_qsa_k_rep", "repeat_kv(key_states, Nrep)",
                    ["B", "Nh", "Lkv", "Dh"], ["B", "Nh", "Lkv", "Dh"], "activation",
                    "2→24 头（GQA）；K 的每一份被 12 个 query 头共享")
        self.tensor("g_qsa_v_rep", "repeat_kv(value_states, Nrep)",
                    ["B", "Nh", "Lkv", "Dh"], ["B", "Nh", "Lkv", "Dh"], "activation",
                    "2→24 头；与 K 同样的复制")
        self.op("g_qsa_repeat_kv", "repeat_kv", "QSA K/V 头扩展", "repeat_kv",
                ["g_qsa_k_post", "g_qsa_v_post"], ["g_qsa_k_rep", "g_qsa_v_rep"],
                [], ["Nkv", "Nh", "Nrep", "Lkv", "Dh"],
                [src("773-782", "repeat_kv(...): expand + reshape"),
                 src("795-796", "key_states = repeat_kv(key, num_key_value_groups)")],
                "GQA 复制不是权重共享；KV 缓存仍只存 2 头")
        self.tensor("g_qsa_qk", "matmul(q, k^T) * scaling",
                    ["B", "Nh", "seqlen", "Lkv"], ["B", "Nh", "seqlen", "Lkv"],
                    "activation", "缩放 1/√256=1/16")
        self.op("g_qsa_qk", "eager_attention_forward", "QSA QK 打分", "matmul+scale",
                ["g_qsa_q_rope", "g_qsa_k_rep"], ["g_qsa_qk"], [], ["Nh", "Dh", "Lkv"],
                [src("798", "attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling"),
                 src("819", "self.scaling = self.head_dim ** -0.5")],
                "24 个 query 头对全部 Lkv 个 key 打分；缩放是 256^-0.5")
        self.tensor("g_qsa_scores_masked", "attn_weights + attention_mask",
                    ["B", "Nh", "seqlen", "Lkv"], ["B", "Nh", "seqlen", "Lkv"],
                    "activation", "causal + selected 合并 mask 加到分数上")
        self.op("g_qsa_mask_add", "eager_attention_forward", "QSA mask 加到分数",
                "add", ["g_qsa_qk", "g_qsa_mask_combined"], ["g_qsa_scores_masked"],
                [], ["B", "Nh", "seqlen", "Lkv"],
                [src("799-800", "if attention_mask is not None: attn_weights = attn_weights + attention_mask")],
                "0 可见、min 屏蔽；QSA 的 selected mask 也在这里生效")
        self.tensor("g_qsa_probs", "softmax(attn_weights, -1)",
                    ["B", "Nh", "seqlen", "Lkv"], ["B", "Nh", "seqlen", "Lkv"],
                    "activation", "fp32 softmax 后转回 query dtype")
        self.op("g_qsa_softmax", "eager_attention_forward", "QSA softmax", "softmax",
                ["g_qsa_scores_masked"], ["g_qsa_probs"], [], ["B", "Nh", "seqlen", "Lkv"],
                [src("802", "attn_weights = softmax(attn_weights, dim=-1, dtype=float32)"
                            ".to(query.dtype)")],
                "softmax 在全部 Lkv 上，但非选中位置是 min，概率为 0")
        self.tensor("g_qsa_av", "matmul(attn_weights, value_states)",
                    ["B", "Nh", "seqlen", "Dh"], ["B", "Nh", "seqlen", "Dh"],
                    "activation", "value 聚合")
        self.op("g_qsa_av", "eager_attention_forward", "QSA AV 聚合", "matmul",
                ["g_qsa_probs", "g_qsa_v_rep"], ["g_qsa_av"], [], ["Nh", "Dh", "Lkv"],
                [src("804", "attn_output = torch.matmul(attn_weights, value_states)")],
                "每个 query 头独立聚合；V 也是 2→24 复制")
        self.tensor("g_qsa_attn_t", "attn_output.transpose(1,2)",
                    ["B", "seqlen", "Nh", "Dh"], ["B", "seqlen", "Nh", "Dh"],
                    "activation", "头维放回序列后")
        self.op("g_qsa_attn_transpose", "eager_attention_forward",
                "QSA 输出转置", "transpose", ["g_qsa_av"], ["g_qsa_attn_t"], [],
                ["B", "seqlen", "Nh", "Dh"],
                [src("805", "attn_output = attn_output.transpose(1, 2).contiguous()")],
                "布局变换；不改变数值")
        self.tensor("g_qsa_attn_flat", "attn_output.reshape(B,S,Nh*Dh)",
                    ["B", "seqlen", "Dqw"], ["B", "seqlen", "Dqw"], "activation",
                    "拼头成 6144")
        self.op("g_qsa_attn_reshape", "Qwen4ExpTextAttention.forward",
                "QSA 拼头", "reshape", ["g_qsa_attn_t"], ["g_qsa_attn_flat"], [],
                ["Nh", "Dh"], [src("888", "attn_output = attn_output.reshape(*input_shape, -1).contiguous()")],
                "24×256→6144")
        self.tensor("g_qsa_gate_sigmoid", "sigmoid(gate)", ["B", "seqlen", "Dqw"],
                    ["B", "seqlen", "Dqw"], "activation",
                    "输出门；源码没有额外缩放")
        self.op("g_qsa_gate_sigmoid", "Qwen4ExpTextAttention.forward",
                "QSA 输出门 sigmoid", "sigmoid",
                ["g_qsa_gate"], ["g_qsa_gate_sigmoid"], [], ["Dqw"],
                [src("889", "attn_output = attn_output * torch.sigmoid(gate)")],
                "逐元素门；不是 softmax 后的 attention 权重")
        self.tensor("g_qsa_gated", "attn_output * sigmoid(gate)",
                    ["B", "seqlen", "Dqw"], ["B", "seqlen", "Dqw"], "activation",
                    "门控后的 attention 输出")
        self.op("g_qsa_gate_mul", "Qwen4ExpTextAttention.forward",
                "QSA 输出门相乘", "broadcast-mul",
                ["g_qsa_attn_flat", "g_qsa_gate_sigmoid"], ["g_qsa_gated"], [],
                ["B", "seqlen", "Dqw"],
                [src("889", "attn_output = attn_output * torch.sigmoid(gate)")],
                "gate 与 attn 输出同宽 6144")
        self.weight("g_qsa_o_proj_weight", "self.self_attn.o_proj.weight", ["H", "Dqw"],
                    ["H", "Dqw"], "nn.Linear(6144, 2560, bias=False)（:831-833）")
        self.tensor("g_attn_out", "self.o_proj(attn_output)", ["B", "seqlen", "H"],
                    ["B", "seqlen", "H"], "activation", "Attention 输出（:891）")
        self.op("g_qsa_o_proj", "Qwen4ExpTextAttention.o_proj", "QSA o_proj",
                "linear", ["g_qsa_gated"], ["g_attn_out"], ["g_qsa_o_proj_weight"],
                ["Dqw", "H", "B", "seqlen"],
                [src("831-833", "self.o_proj = nn.Linear(num_attention_heads*head_dim, "
                                "hidden_size, bias=False)"),
                 src("891", "attn_output = self.o_proj(attn_output)")],
                "bias=False；输出回到 hidden_size=2560")

        # Gated Residual（attn 段）
        self._gated_residual_apply("attn", "g_attn_out", streams_in,
                                   "g_hidden_streams_mid")
        # Gated Residual（ffn 站点）
        self._gated_residual_site("ffn", "g_hidden_streams_mid")
        self._moe_ops("g_hc_ffn_mixed")
        self._gated_residual_apply("ffn", "g_mlp_out", "g_hidden_streams_mid",
                                   "g_hidden_streams_out")

    def _build_cache_notes(self) -> None:
        self.cache_notes = [
            "层缓存类：config.layer_types 在该层为 indexed_attention（QSA），dispatch 到 "
            "DynamicIndexedLayer（cache_utils:1285-1294）。",
            "主 K/V 缓存：[B,Nkv,Lkv,Dh]=[B,2,L,256]，update 线性追加；"
            "attention 在全部 Lkv 上做 masked softmax，稀疏 mask 只收窄可见范围"
            "（cache_utils:113-153）。",
            "indexer key 缓存：[B,Lidx,128] 单头、逐 token raw key；"
            "update_indexer 按序列拼接（modeling:707-708；cache_utils:350-363）。",
            "P = past_key_values.get_seq_length()（modeling:1423）：QSA 用第一个 "
            "indexed_attention 层（layer 3）的 keys 长度；空缓存 Prefill 时 kvcachelen=0。",
            "QSA 的 K/V 与 indexer key 都随历史线性增长；与 GDN 的固定 conv/recurrent "
            "状态不同，不能用一个 P+S 或滑动窗口公式套全部层。",
            "indexer 的 block 选择不改变 K/V 缓存：它只生成 selected token mask，"
            "attention 仍读取全部 Lkv 个 K/V。",
        ]

    def _build_coverage_notes(self) -> None:
        v = self.values
        gdn = [i for i in range(v.num_hidden_layers) if v.layer_type(i) == "linear_attention"]
        qsa = [i for i in range(v.num_hidden_layers) if v.layer_type(i) == "indexed_attention"]
        self.coverage_notes = [
            "已表达：Gated Residual 两站点（attn/ffn）的分组 RMSNorm、10240→320→10240 "
            "逐元素读门、四路加权 mean、4 路写门展开相加；QSA 的 q/gate 合并投影、"
            "q/k RMSNorm、partial RoPE（前 64 维）、2 头 KV 缓存与 repeat_kv、"
            "indexer 投影/逐 token key 缓存/micro-block 均值池化/block RoPE/块打分/"
            "块 top-k/tail/pad/selected mask、eager QK/softmax/AV、输出 sigmoid 门、"
            "o_proj；TopK MoE + 并行共享专家。",
            f"真实层归属：QSA 层 {len(qsa)} 个（{qsa}）；GDN 层 {len(gdn)} 个"
            f"（含 PLE 层 {v.ple_layer_ids} one-indexed）。本 Block 只代表 QSA+MoE 组模板。",
            "根配置没有 quantization_config；本图只表达 HF 逻辑 eager 路径，"
            "不分析设备 kernel、bfloat16 数值或权重内容。",
            "未覆盖：MTP 1 层（配置声明 full_attention，但本地 HF 实现没有 MTP 模块，"
            "加载时忽略 mtp.*）、视觉 27 层、generate/采样循环、loss/aux_loss、"
            "output_router_logits 辅助输出、torch.compile/梯度检查点、数值等价。",
            "数据相关量（N_hit、N_e、实际选中的块/ token、路由分布）保留未知；"
            "不根据权重推断 MoE 实际路由，不默认 0。专家只有一份模板，不按 512 个专家复制图。",
            "QSA 的 micro-block 选择按语义操作表达，不铺开 512 个块或全部 token；"
            "具体块/token 由打分数据决定，图上不猜。",
        ]
