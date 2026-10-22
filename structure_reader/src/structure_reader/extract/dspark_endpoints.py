"""DeepSeek-V4-Flash-0731 DSpark 首入口语义图与端口绑定（工作包 B · 步骤 B3a）。

数据来源：models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/{model.py,
generate.py} 与 inference/config.json（固定 SHA256 基线，
:mod:`structure_reader.extract.evidence` 逐条核对原文）。静态取证，
不导入/不执行模型代码。

B3a 范围（B3 整体仍 in_progress，本模块不是完整 B3 交付）：
1. 首入口 SemanticGraph（主干 readout 边界 917-925 + forward_embed
   851-858 + forward_spec 929-936 首入口部分）：三个主干层 40/41/42 的
   外部 h 输入 [B,T,N_hc,H]（graph_input，只展开这三处边界读出、不复制
   主干）→ 各自 mean(dim=2)（921）[B,T,H] → 按主干层遍历顺序 concat
   （925；固定 config [40,41,42] 升序与遍历序一致，不泛称任意 target
   配置都按列表顺序输出）→ main_hidden [B,T,H×N_tgt] → stage0 唯一
   main_proj（fp8 权重，Transformer default_dtype 机制）→ main_norm
   （fp32 weight，输出回转 bf16）→ **唯一 main_x** [B,T,H]；
   noise 填充 [B,K]（dspark_noise_token_id=128799，run config:9）+ 首列
   写 seed input_ids（[B] rank-1：主干 hc_head 折合 HC 为 [B,T,H] 后
   ParallelHead x[:,-1]（731-734）→ logits [B,V] → sample argmax → [B]，
   855 首列 [B] 槽精确写入、无广播）→ 共享 embed（ParallelEmbedding
   单份 weight = Transformer.embed 同一对象，model.py:903 绑定非复制）
   查表 [B,K,H] → unsqueeze(2)→repeat(1,1,N_hc,1) → 草稿 h
   [B,K,N_hc,H]；同一 main_x 向三 stage 分发（entry.stage_feed_0/1/2
   边界节点，forward_spec 931-932 第三实参透传），草稿 h 只进 stage0。
   全部节点/边两相位无条件（forward_spec 930 先 forward_embed 后 stage
   loop、933-934 prefill 在 loop 后 return）——不挂 g_decode/g_prefill，
   prefill start_pos=0 仍执行首入口。
2. 首入口端口/角色绑定规范 JSON（canonical 往返）：main_x 唯一生产
   （mtp.0.main_proj/main_norm）+ 三 stage 共享分发 + readout 顺序（固定
   config 40/41/42，图中以 entry.readout_mean_40/41/42 →
   entry.readout_concat 端口 s0/s1/s2 可联查）+ 三 stage 端口绑定
   （main_x 全部、draft h 仅 stage0，stage1/2 的 h_in 来源是前一 stage
   的 h_out——B2 链路已建）+ 共享 embed 引用（mtp.{i}.embed 同一对象
   非复制）+ noise_token_id/temperature 来源记录 + T_main/K 相容调用
   前提与 unknown 边界（不声称 generate.py 实际调用 forward_spec——
   generate.py:41-45 只调 model.forward）。读回以固定 ModelValues/run
   config 原文证据为可信期望核对，篡改具名拒绝。
3. 篡改拒绝（读回/图级 verify）：必要生产/消费链断链（readout→concat→
   main_proj→main_norm→main_x 三 stage、noise/seed→首列写→embed→HC
   展开→stage0 的必要边缺失/一致重接/共享参数消费边丢失）、readout
   concat 端口交换、seed input_ids rank2 伪造、多份独立 main_x（删分发
   或注入第二投影路径——引用合法但语义错）、共享 embed 复制身份、
   stage0 角色错绑（main_proj 绑非 stage0）、prefill 入口停用（入口边
   挂相位 guard）、noise/temperature 来源伪造——全部具名拒绝。

不做（B3 后续步骤续接）：末出口（hc_head→norm→head full_logits→Markov
采样循环→Confidence）、Markov 递推展开、Confidence 内部图、全量集成
（B4）。本模块不声称 B3/DSpark/V2 完成；不声称 sample 产生真实 token/
置信度数值。

可调用入口：
- build_dspark_entry_graph() → DSparkEntryBundle（首入口图 + 证据注册表
  （B1/B2/reference 同源全量 + 本包新增 ev_ep_*） + ModelValues + 绑定）
- verify_entry_graph(graph, values)（构建期已调用，测试/读回复用）
- build_entry_bindings(values) → EntryBindings
- serialize_entry_bindings / deserialize_entry_bindings（公开 JSON 往返，
  读回全量一致性核对）

失败异常接口（以实际接口为准）：证据核对失败抛
structure_reader.extract.evidence.EvidenceError；图构造错误、
validate_graph_references 未通过、绑定 JSON 结构不合法或语义篡改抛
structure_reader.extract.deepseek_v4_reference.ExtractionError
（ValueError 子类）。均具名拒绝、失败不给图/不给绑定。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from structure_reader.checks.graph_contracts import validate_graph_references
from structure_reader.extract.deepseek_v4_dspark import (
    _axes as _attn_axes,
    _evidence_defs as _attn_evidence_defs,
)
from structure_reader.extract.deepseek_v4_reference import (
    ExtractionError,
    _GraphBuilder,
    _evidence_defs as _reference_evidence_defs,
)
from structure_reader.extract.dspark_stage import (
    EXPR_MAIN_PROJ_IN,
    _stage_evidence_defs,
)
from structure_reader.extract.evidence import (
    EvidenceBase,
    EvidenceRegistry,
    SourceEvidence,
    load_evidence_base,
)
from structure_reader.graph_document import SemanticGraph
from structure_reader.graph_ir import NamedAxis
from structure_reader.model_registry import find_workspace_root
from structure_reader.rules.deepseek_v4_flash_0731 import (
    AXIS_DEFS,
    ModelValues,
    load_model_values,
)
from structure_reader.scenario_evaluator import check_expression

__all__ = [
    "ENTRY_TEMPLATE_ID",
    "ENTRY_SCHEMA",
    "NOISE_TOKEN_ID",
    "PREMISE_KEYS",
    "EntryMainXBinding",
    "EntryDraftTokens",
    "EntrySharedEmbed",
    "EntryStagePort",
    "EntryTemperature",
    "EntryBindings",
    "DSparkEntryBundle",
    "verify_entry_graph",
    "build_entry_bindings",
    "serialize_entry_bindings",
    "deserialize_entry_bindings",
    "build_dspark_entry_graph",
]

ENTRY_TEMPLATE_ID = "deepseek_v4_flash_0731:dspark_entry"
ENTRY_SCHEMA = "dspark-entry-bindings/v1"

# 草稿噪声 token（run config dspark_noise_token_id，config.json:9）。
# ModelValues 无此字段——常量 + ev_ep_noise_cfg 原文锚定表达，不冒充
# ModelValues 字段；输入基线 SHA 核对保证 run config 不漂移。
# 注意：ModelValues 确有 dspark_block_size 字段（rules 178/260，K 轴
# 已直接使用）；真正缺的字段只有 noise_token_id 与 temperature。
NOISE_TOKEN_ID = 128799

# call_premises 固定键集（读回核对：缺键/多键/改义具名拒绝）。
PREMISE_KEYS = (
    "t_main_vs_draft_k",
    "input_ids_width",
    "forward_spec_caller",
    "noise_token_id_source",
    "temperature_source",
)

# 主干 readout 源层（固定 config dspark_target_layer_ids）与图中 mean
# 节点/concat 端口的固定对应（917-925 主干遍历顺序 == 升序列表，仅对
# 本固定输入成立，不泛称任意 target 配置都按列表顺序输出）。
READOUT_PORT_ORDER = ("s0", "s1", "s2")

# 首入口必要节点闭世界（含所有 op 类型约束的核对对象）。
_EXPECTED_NODES: frozenset[str] = frozenset({
    # readout 边界
    "entry.input_h_40", "entry.input_h_41", "entry.input_h_42",
    "entry.readout_mean_40", "entry.readout_mean_41",
    "entry.readout_mean_42", "entry.readout_concat",
    # main 投影链
    "entry.main_proj_w", "entry.main_norm_w", "entry.main_proj",
    "entry.main_norm",
    # draft token 链
    "entry.seed_src", "entry.noise_fill", "entry.first_col_write",
    "entry.embed_w", "entry.embed_lookup", "entry.hc_expand",
    # stage 分发
    "entry.stage_feed_0", "entry.stage_feed_1", "entry.stage_feed_2",
})

# 首入口必要张量闭世界（多余张量 = 独立身份/复制）。
_EXPECTED_TENSORS: frozenset[str] = frozenset({
    "entry:src_h_40", "entry:src_h_41", "entry:src_h_42",
    "entry:mean_h_40", "entry:mean_h_41", "entry:mean_h_42",
    "entry:main_hidden", "entry:main_proj_w", "entry:main_proj_y",
    "entry:main_norm_w", "entry:main_x",
    "entry:seed_ids", "entry:draft_noise", "entry:draft_input_ids",
    "entry:embed_w", "entry:embed_y", "entry:entry_h",
})

# 首入口必要生产/消费链闭世界（边 ID → 源节点.端口 → 目的节点.端口 +
# 张量）。readout→concat→main_proj→main_norm→唯一 main_x 三 stage；
# noise/seed→首列写→embed→HC 展开→stage0。通用引用检查只做三方一致，
# 不查必要链——缺边、保 ID 一致重接、共享参数消费边丢失由此具名拒绝。
_EXPECTED_CHAIN: tuple[tuple[str, str, str, str, str, str], ...] = (
    # 主干 readout 边界（917-925）：外部 h → mean → 有序 concat
    ("e_ep_h40_mean", "entry.input_h_40", "y", "entry.readout_mean_40",
     "x", "entry:src_h_40"),
    ("e_ep_h41_mean", "entry.input_h_41", "y", "entry.readout_mean_41",
     "x", "entry:src_h_41"),
    ("e_ep_h42_mean", "entry.input_h_42", "y", "entry.readout_mean_42",
     "x", "entry:src_h_42"),
    ("e_ep_mean40_cat", "entry.readout_mean_40", "y",
     "entry.readout_concat", "s0", "entry:mean_h_40"),
    ("e_ep_mean41_cat", "entry.readout_mean_41", "y",
     "entry.readout_concat", "s1", "entry:mean_h_41"),
    ("e_ep_mean42_cat", "entry.readout_mean_42", "y",
     "entry.readout_concat", "s2", "entry:mean_h_42"),
    # readout → main_proj → main_norm → 唯一 main_x 三 stage
    ("e_ep_cat_proj", "entry.readout_concat", "y", "entry.main_proj",
     "x", "entry:main_hidden"),
    ("e_ep_projw_proj", "entry.main_proj_w", "y", "entry.main_proj",
     "w", "entry:main_proj_w"),
    ("e_ep_proj_norm", "entry.main_proj", "y", "entry.main_norm",
     "x", "entry:main_proj_y"),
    ("e_ep_normw_norm", "entry.main_norm_w", "y", "entry.main_norm",
     "w", "entry:main_norm_w"),
    ("e_ep_norm_feed0", "entry.main_norm", "y", "entry.stage_feed_0",
     "x", "entry:main_x"),
    ("e_ep_norm_feed1", "entry.main_norm", "y", "entry.stage_feed_1",
     "x", "entry:main_x"),
    ("e_ep_norm_feed2", "entry.main_norm", "y", "entry.stage_feed_2",
     "x", "entry:main_x"),
    # noise/seed → 首列写 → embed → HC 展开 → stage0
    ("e_ep_noise_firstcol", "entry.noise_fill", "y",
     "entry.first_col_write", "dst", "entry:draft_noise"),
    ("e_ep_seed_firstcol", "entry.seed_src", "y",
     "entry.first_col_write", "src", "entry:seed_ids"),
    ("e_ep_firstcol_embed", "entry.first_col_write", "y",
     "entry.embed_lookup", "x", "entry:draft_input_ids"),
    ("e_ep_embedw_embed", "entry.embed_w", "y", "entry.embed_lookup",
     "w", "entry:embed_w"),
    ("e_ep_embed_hcexp", "entry.embed_lookup", "y", "entry.hc_expand",
     "x", "entry:embed_y"),
    ("e_ep_hcexp_feed0", "entry.hc_expand", "y", "entry.stage_feed_0",
     "h", "entry:entry_h"),
)


# ---------------------------------------------------------------------------
# 证据目录：本包新增条目（ev_ep_*）。hc/MoE/Attention/stage/readout 等
# 既有条目复用 B1/B2/reference 同源注册（同 ID 同原文，重复注册即拒绝）。
# ---------------------------------------------------------------------------

_E = SourceEvidence


def _entry_evidence_defs() -> tuple[SourceEvidence, ...]:
    m = "model.py"
    rc = "run_config.json"
    gen = "generate.py"
    return (
        _E("ev_ep_noise_cfg", rc, "dspark_noise_token_id", 9, 9,
           '"dspark_noise_token_id": 128799,',
           "草稿噪声 token 来源：run config dspark_noise_token_id=128799"
           "（noise_fill 常量，NOISE_TOKEN_ID）。事实澄清：ModelValues "
           "确有 dspark_block_size 字段（rules 178/260，构建器 K=N_draft"
           " 轴已直接使用）——真正缺的字段只有 noise_token_id 与 "
           "temperature，不把已有字段误称为缺失", "source_checked"),
        _E("ev_ep_readout_bound", m, "Transformer.forward main readout", 917, 925,
           "main_hiddens = []\n"
           "for i, layer in enumerate(self.layers):\n"
           "    h = layer(h, start_pos, input_ids)\n"
           "    if i in self.target_layer_ids:\n"
           "        main_hiddens.append(h.mean(dim=2))",
           "主干 readout 边界证据：主干层 i∈{40,41,42}（固定 config "
           "dspark_target_layer_ids 升序）各自 h.mean(dim=2)（921）得 "
           "[B,T,H]，按主干层遍历顺序 append（917-921）；925 "
           "torch.cat(main_hiddens, dim=-1) → main_hidden [B,T,H×N_tgt]。"
           "本固定输入下遍历顺序与列表顺序相同（升序 [40,41,42]），但这是"
           "遍历序的性质——不泛称任意 target 配置都按列表顺序输出。图中只"
           "展开这三处边界读出（外部 h graph_input → mean → concat），"
           "主干其余部分不复制、不改 B2", "source_checked"),
        _E("ev_ep_temp_default", m, "ModelArgs.temperature default", 39, 39,
           "    temperature: float = 1",
           "temperature 来源：ModelArgs 字段默认值 1（model.py:39）；run "
           "config 无 temperature 字段（无覆盖）。temperature 属 last "
           "stage forward_head 的 sample（871 self.temperature）与主干 "
           "forward（924）——不属于首入口 forward_embed（851-858）。"
           "ModelValues 无此字段，绑定表只记录来源不冒充字段，也不声称"
           "采样产生真实 token 数值", "located"),
        _E("ev_ep_generate_calls", gen, "generate loop calls model.forward",
           41, 45,
           "for cur_pos in range(min(prompt_lens), total_len):\n"
           "    next_token = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)[0]",
           "generate.py 推理循环只调 model.forward（41-45）——不声称 "
           "generate.py 实际调用 forward_spec；forward_spec 的运行时调用"
           "方在本工程可见输入中未知（unknown，不编造）。参考入口 "
           "__main__（model.py:957-961）演示 prefill 一次 + 逐 token "
           "decode 的调用模式，属演示非运行时保证", "located"),
        _E("ev_ep_embed_param", m, "ParallelEmbedding init/weight", 89, 100,
           "class ParallelEmbedding(nn.Module):\n"
           "    def __init__(self, vocab_size: int, dim: int):\n"
           "    self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))",
           "共享 embed 参数身份：ParallelEmbedding.weight torch.empty 无"
           "显式 dtype → torch 默认 dtype；参考入口 model.py:950 "
           "set_default_dtype(bfloat16) → 存储 bf16（同 "
           "ev_st_entry_default_dtype 机制）。world_size=1 分析假设下形状"
           " [V,H]（world_size>1 时各 rank 持 part 行，本图 world_size "
           "固定 1）。首入口图单份 weight 张量 = Transformer.embed 同一"
           "对象（903 绑定到各 mtp 实例属性），非三份复制", "located"),
        _E("ev_ep_entry_premise", m, "forward_spec seed width premise",
           None, None, None,
           "forward_spec seed input_ids 宽度前提与 T_main/K 相容性",
           "derived",
           derivation="seed（forward_spec 的 input_ids 实参）rank-1 [B]："
                      "主干 forward 922 hc_head 把 [B,T,N_hc,H] 折合为 "
                      "[B,T,H]（715 sum(dim=2)）→ 923 ParallelHead "
                      "not full_logits 取 x[:, -1]（731-734）→ logits "
                      "[B,V] → 924 sample（939-946）argmax(dim=-1) → "
                      "[B]。958/961 参考入口把该 [B] output_ids 直接作为 "
                      "forward_spec 的 input_ids。forward_embed 855 "
                      "draft_input_ids[:, 0] = input_ids 以 [B] 槽精确"
                      "写入（new_full [B,K] 的首列赋值，无广播）。因此 "
                      "seed 宽度前提 = 1（T_in=1，rank-1 [B] 的唯一元素"
                      "数）；main_hidden 的 T_main 由 readout cat 承载"
                      "（925），与 seed 宽度解耦。generate.py 只调 "
                      "forward（41-45），forward_spec 实际运行时调用方"
                      "未知（unknown，不编造）——前提属静态可证范围的"
                      "调用约定，非全程序保证"),
    )


def _entry_axes(values: ModelValues) -> dict[str, NamedAxis]:
    """首入口图轴：B1 模块轴同源复制 + 本包四条（V/N_tgt rules 同源、
    N_tgtH 派生、T_in 调用前提轴）。"""
    axes = _attn_axes(values)
    by_symbol = {a.symbol: a for a in AXIS_DEFS}
    for symbol in ("V", "N_tgt"):
        a = by_symbol[symbol]
        axes[symbol] = NamedAxis(
            symbol=a.symbol, name=a.name, meaning=a.meaning,
            origin=a.origin, evidence_ref=a.evidence_ref,
            value=a.value, expression=a.expression,
            unknown_reason=a.unknown_reason,
        )
    # main_proj 输入宽（model.py:832 args.dim × len(dspark_target_layer_ids)）；
    # readout mean/concat 边界证据 ev_ep_readout_bound 与其一致（925 cat 宽）。
    axes["N_tgtH"] = NamedAxis(
        "N_tgtH", "main_proj_in_width",
        "main_proj 输入宽 H×N_tgt（三个 readout 有序 concat）",
        "推导 model.py:832", "ev_st_main_proj", expression=EXPR_MAIN_PROJ_IN)
    # forward_spec 的 seed input_ids 宽度前提（rank-1 [B]，855 首列 [B] 槽
    # 写入）；主干 token 轴 T 由 main_hidden 承载，两者解耦。
    axes["T_in"] = NamedAxis(
        "T_in", "spec_seed_width",
        "forward_spec seed input_ids 元素数（rank-1 [B] 调用前提，"
        "855 首列 [B] 槽写入）",
        "调用前提（derived，ev_ep_entry_premise）", "ev_ep_entry_premise",
        value=1)
    return axes


# ---------------------------------------------------------------------------
# 首入口图
# ---------------------------------------------------------------------------

def _add_entry_tensors(b: _GraphBuilder, axes: dict[str, NamedAxis]
                       ) -> dict[str, str]:
    """首入口张量角色 → stable_id（前缀 entry:）。

    dtype 依据：readout 外部 h / mean 结果 / main_hidden/main_proj_y/
    main_x/embed 链 bf16（主干 h bf16；mean 后仍 bf16；main_proj fp8
    权重 GEMM 输出 dtype 未在 model.py 展开（kernel.py），按 B1 惯例
    （main_kv_proj 等）记 bf16，main_norm 内部 .float() 求值后回转
    x.dtype）；main_proj_w fp8_e4m3fn（自定义 Linear 默认 dtype 机制，
    Transformer default_dtype——dtype="fp8" config）；norm weight fp32；
    token 侧 int64；embed_w bf16（torch.empty 无显式 dtype + 参考入口
    set_default_dtype(bf16)，ev_ep_embed_param）。"""
    t: dict[str, str] = {}

    def add(role: str, dtype: str, syms: tuple[str, ...],
            evidence: tuple[str, ...] = ()) -> str:
        tid = b.tensor(f"entry:{role}", dtype, syms, evidence)
        t[role] = tid
        return tid

    # 主干 readout 边界（917-925）：三处外部 h 输入 + 各自 mean 结果。
    # 只展开这三处边界读出，不复制主干其余部分。
    for lid in (40, 41, 42):
        add(f"src_h_{lid}", "bf16", ("B", "T", "N_hc", "H"),
            ("ev_ep_readout_bound",))
        add(f"mean_h_{lid}", "bf16", ("B", "T", "H"),
            ("ev_ep_readout_bound",))
    # main 链（主干 token 轴 T）
    add("main_hidden", "bf16", ("B", "T", "N_tgtH"), ("ev_ep_readout_bound",))
    add("main_proj_w", "fp8_e4m3fn", ("H", "N_tgtH"),
        ("ev_st_main_proj", "ev_st_linear_dtype_mech",
         "ev_st_module_default_dtype"))
    add("main_proj_y", "bf16", ("B", "T", "H"), ())
    add("main_norm_w", "fp32", ("H",), ("ev_st_norm_params",))
    add("main_x", "bf16", ("B", "T", "H"), ())
    # token/draft 链（草稿轴 K=N_draft；seed rank-1 [B]）
    add("seed_ids", "int64", ("B",), ("ev_ep_entry_premise",))
    add("draft_noise", "int64", ("B", "N_draft"), ("ev_ep_noise_cfg",))
    add("draft_input_ids", "int64", ("B", "N_draft"),
        ("ev_st_forward_embed",))
    add("embed_w", "bf16", ("V", "H"), ("ev_ep_embed_param",
                                         "ev_st_entry_default_dtype"))
    add("embed_y", "bf16", ("B", "N_draft", "H"), ())
    add("entry_h", "bf16", ("B", "N_draft", "N_hc", "H"),
        ("ev_st_forward_embed",))
    return t


def _add_entry_flow(b: _GraphBuilder, t: dict[str, str]) -> None:
    """首入口流（readout 边界 917-925 + forward_embed 851-858 +
    forward_spec 929-936 首入口）。

    相位规则：全部节点/边两相位无条件（不挂 guard）——forward_spec 930
    先 forward_embed、931-932 stage loop 无条件执行，933-934 prefill 在
    loop 后才 return；首入口不是 decode-only。不通过删边/挂 guard 实现
    相位屏蔽。"""
    def node(sid, op, gr, inputs=(), outputs=(), ev=()):
        for ev_id in ev:
            b.require_evidence(ev_id)
        return b.node(sid, op, gr, inputs, outputs, ev)

    def edge(sid, src, dst, role, kind="data", **kw):
        if kw.pop("guard_ref", None) is not None:
            raise ExtractionError(
                f"首入口边 {sid} 不允许挂相位 guard（两相位无条件）")
        ev = kw.pop("ev", ())
        for ev_id in ev:
            b.require_evidence(ev_id)
        return b.edge(sid, src, dst, t[role], kind, **kw)

    # ===== 主干 readout 边界（917-925）：三处外部 h → mean(dim=2) →
    # 有序 concat。端口 s0/s1/s2 对应主干层遍历顺序（本固定输入
    # [40,41,42] 升序遍历 == 列表顺序；不泛称任意配置都按列表序）。=====
    node("entry.input_h_40", "graph_input", "tensor_op",
         (), (("y", t["src_h_40"]),), ("ev_ep_readout_bound",))
    node("entry.input_h_41", "graph_input", "tensor_op",
         (), (("y", t["src_h_41"]),), ("ev_ep_readout_bound",))
    node("entry.input_h_42", "graph_input", "tensor_op",
         (), (("y", t["src_h_42"]),), ("ev_ep_readout_bound",))
    node("entry.readout_mean_40", "reduce_mean", "tensor_op",
         (("x", t["src_h_40"]),), (("y", t["mean_h_40"]),),
         ("ev_ep_readout_bound",))
    node("entry.readout_mean_41", "reduce_mean", "tensor_op",
         (("x", t["src_h_41"]),), (("y", t["mean_h_41"]),),
         ("ev_ep_readout_bound",))
    node("entry.readout_mean_42", "reduce_mean", "tensor_op",
         (("x", t["src_h_42"]),), (("y", t["mean_h_42"]),),
         ("ev_ep_readout_bound",))
    node("entry.readout_concat", "concat", "tensor_op",
         (("s0", t["mean_h_40"]), ("s1", t["mean_h_41"]),
          ("s2", t["mean_h_42"])),
         (("y", t["main_hidden"]),), ("ev_ep_readout_bound",))
    for lid, port in zip((40, 41, 42), READOUT_PORT_ORDER):
        edge(f"e_ep_h{lid}_mean", (f"entry.input_h_{lid}", "y"),
             (f"entry.readout_mean_{lid}", "x"), f"src_h_{lid}")
        edge(f"e_ep_mean{lid}_cat", (f"entry.readout_mean_{lid}", "y"),
             ("entry.readout_concat", port), f"mean_h_{lid}")

    # ===== main 链：main_proj → main_norm → 唯一 main_x =====
    node("entry.main_proj_w", "linear_weight", "parameter_op",
         (), (("y", t["main_proj_w"]),), ("ev_st_main_proj",))
    node("entry.main_norm_w", "rms_norm_weight", "parameter_op",
         (), (("y", t["main_norm_w"]),), ("ev_st_norm_params",))
    node("entry.main_proj", "linear", "parameter_op",
         (("x", t["main_hidden"]), ("w", t["main_proj_w"])),
         (("y", t["main_proj_y"]),), ("ev_st_main_proj",))
    node("entry.main_norm", "rms_norm", "parameter_op",
         (("x", t["main_proj_y"]), ("w", t["main_norm_w"])),
         (("y", t["main_x"]),), ("ev_st_norm_params",))
    edge("e_ep_cat_proj", ("entry.readout_concat", "y"),
         ("entry.main_proj", "x"), "main_hidden")
    edge("e_ep_projw_proj", ("entry.main_proj_w", "y"),
         ("entry.main_proj", "w"), "main_proj_w")
    edge("e_ep_proj_norm", ("entry.main_proj", "y"),
         ("entry.main_norm", "x"), "main_proj_y")
    edge("e_ep_normw_norm", ("entry.main_norm_w", "y"),
         ("entry.main_norm", "w"), "main_norm_w")

    # ===== draft token 链：noise 填充 → 首列写 rank-1 seed =====
    node("entry.seed_src", "graph_input", "tensor_op",
         (), (("y", t["seed_ids"]),), ("ev_ep_entry_premise",))
    node("entry.noise_fill", "fill_const", "tensor_op",
         (), (("y", t["draft_noise"]),), ("ev_ep_noise_cfg",))
    node("entry.first_col_write", "index_write", "index_op",
         (("dst", t["draft_noise"]), ("src", t["seed_ids"])),
         (("y", t["draft_input_ids"]),), ("ev_st_forward_embed",))
    edge("e_ep_noise_firstcol", ("entry.noise_fill", "y"),
         ("entry.first_col_write", "dst"), "draft_noise")
    edge("e_ep_seed_firstcol", ("entry.seed_src", "y"),
         ("entry.first_col_write", "src"), "seed_ids")

    # ===== 共享 embed 查表 → HC 展开 =====
    node("entry.embed_w", "embedding_weight", "parameter_op",
         (), (("y", t["embed_w"]),), ("ev_ep_embed_param",))
    node("entry.embed_lookup", "embedding", "parameter_op",
         (("x", t["draft_input_ids"]), ("w", t["embed_w"])),
         (("y", t["embed_y"]),), ("ev_st_forward_embed",))
    node("entry.hc_expand", "unsqueeze_repeat", "tensor_op",
         (("x", t["embed_y"]),), (("y", t["entry_h"]),),
         ("ev_st_forward_embed",))
    edge("e_ep_firstcol_embed", ("entry.first_col_write", "y"),
         ("entry.embed_lookup", "x"), "draft_input_ids")
    edge("e_ep_embedw_embed", ("entry.embed_w", "y"),
         ("entry.embed_lookup", "w"), "embed_w")
    edge("e_ep_embed_hcexp", ("entry.embed_lookup", "y"),
         ("entry.hc_expand", "x"), "embed_y")

    # ===== 三 stage 分发（forward_spec 930-932）：同一 main_x 第三实参
    # 透传全部三 stage（三条边共享同一 main_x 张量身份）；草稿 h 只进
    # stage0（stage1/2 的 h_in 来自前一 stage h_out——B2 链路已建）。=====
    node("entry.stage_feed_0", "stage_dispatch", "boundary_op",
         (("x", t["main_x"]), ("h", t["entry_h"])),
         (("main_x_out", t["main_x"]), ("h_out", t["entry_h"])),
         ("ev_st_forward_spec", "ev_st_layer_ids"))
    node("entry.stage_feed_1", "stage_dispatch", "boundary_op",
         (("x", t["main_x"]),), (("main_x_out", t["main_x"]),),
         ("ev_st_forward_spec",))
    node("entry.stage_feed_2", "stage_dispatch", "boundary_op",
         (("x", t["main_x"]),), (("main_x_out", t["main_x"]),),
         ("ev_st_forward_spec",))
    edge("e_ep_norm_feed0", ("entry.main_norm", "y"),
         ("entry.stage_feed_0", "x"), "main_x")
    edge("e_ep_hcexp_feed0", ("entry.hc_expand", "y"),
         ("entry.stage_feed_0", "h"), "entry_h")
    edge("e_ep_norm_feed1", ("entry.main_norm", "y"),
         ("entry.stage_feed_1", "x"), "main_x")
    edge("e_ep_norm_feed2", ("entry.main_norm", "y"),
         ("entry.stage_feed_2", "x"), "main_x")


def verify_entry_graph(graph: SemanticGraph, values: ModelValues) -> None:
    """首入口图语义核对（构建期调用；测试/读回复用）。

    结构/身份/必要链/相位逐项核对，语义篡改（引用合法但语义错）具名
    拒绝；篡改负例应先证 validate_graph_references 为 0 再断言本函数
    拒绝（引用合法前提下的语义拒绝，非悬空引用兜底）：
    - 两相位无条件：任何边挂 guard 即拒（prefill 入口停用/decode-only）；
    - 必要生产/消费链闭世界（readout→concat→main_proj→main_norm→唯一
      main_x 三 stage；noise/seed→首列写→embed→HC 展开→stage0）：必要
      边缺失、保 ID 一致重接（源/目的端口/张量任一不符）、必要链之外
      多余节点/张量/边（第二投影路径、独立身份、多余通路）、共享参数
      消费边丢失——全部具名拒绝；
    - 唯一 main_x：main_x 张量恰一、linear/rms_norm 投影节点恰一、
      main_x 出边全部来自 main_norm.y、三 feed 的 x 端口三方一致消费
      同一 main_x；
    - readout 边界：三 reduce_mean + 唯一 concat，concat 端口 s0/s1/s2
      依次消费 mean_h_40/41/42（主干遍历顺序，本固定输入 == 列表序），
      端口/边三方一致交换具名拒绝；
    - seed rank-1：seed input_ids 必须是 (B,) 一维（主干 hc_head 折合 +
      ParallelHead x[:,-1] → sample [B]，855 首列 [B] 槽写入）——
      rank-2 [B,T_in] 伪造具名拒绝；
    - draft 链：noise_fill/index_write/embedding 节点恰一、
      draft_input_ids 唯一消费 embed_lookup.x、首列写入端口
      (dst=noise, src=seed)、entry_h 唯一消费 feed_0.h；
    - T/K 不混：main 链轴含 T 不含 N_draft，draft 链轴含 N_draft 不含
      T；
    - 无 cache 状态（首入口不碰 win_cache）。"""
    if graph.template_id != ENTRY_TEMPLATE_ID:
        raise ExtractionError(
            f"首入口图 template_id 不符: {graph.template_id!r}")
    for e in graph.edges:
        if e.guard_ref is not None:
            raise ExtractionError(
                f"首入口边 {e.stable_id} 挂相位 guard {e.guard_ref!r}——"
                "forward_spec 930 先 forward_embed 后 stage loop、933-934 "
                "prefill 在 loop 后 return，首入口两相位无条件；prefill "
                "入口停用/decode-only 语义篡改具名拒绝")
    if graph.state_versions or graph.state_transitions:
        raise ExtractionError(
            "首入口图不应含 cache 状态（首入口不碰 win_cache）")

    tensors = {t_.name: t_ for t_ in graph.tensors}
    nodes = {n.stable_id: n for n in graph.nodes}
    edges_by_id = {e.stable_id: e for e in graph.edges}

    # 必要节点/张量/边闭世界（先于具体端口核对，缺/多/重接都点名）
    missing_nodes = sorted(_EXPECTED_NODES - set(nodes))
    if missing_nodes:
        raise ExtractionError(
            f"首入口图缺少必要节点: {missing_nodes}——必要生产/消费链"
            "断链具名拒绝")
    extra_nodes = sorted(set(nodes) - _EXPECTED_NODES)
    if extra_nodes:
        raise ExtractionError(
            f"首入口图出现必要链之外的节点: {extra_nodes}——第二投影"
            "路径/独立身份/多余通路具名拒绝")
    missing_tensors = sorted(_EXPECTED_TENSORS - set(tensors))
    if missing_tensors:
        raise ExtractionError(
            f"首入口图缺少必要张量: {missing_tensors}——断链具名拒绝")
    extra_tensors = sorted(set(tensors) - _EXPECTED_TENSORS)
    if extra_tensors:
        raise ExtractionError(
            f"首入口图出现必要链之外的张量: {extra_tensors}——独立身份"
            "（多份 main_x/embed 复制等）具名拒绝")
    for eid, src_n, src_p, dst_n, dst_p, tname in _EXPECTED_CHAIN:
        e = edges_by_id.get(eid)
        if e is None:
            raise ExtractionError(
                f"必要生产/消费链边 {eid} 缺失（期望 {src_n}.{src_p} → "
                f"{dst_n}.{dst_p}，张量 {tname}）——readout→concat→"
                "main_proj→main_norm→main_x 或 noise/seed→首列写→embed"
                "→HC 展开→stage0 断链具名拒绝")
        got = (e.src_node_id, e.src_port, e.dst_node_id, e.dst_port,
               e.tensor_id)
        want = (src_n, src_p, dst_n, dst_p, tensors[tname].stable_id)
        if got != want:
            raise ExtractionError(
                f"必要链边 {eid} 被一致重接：期望 {src_n}.{src_p} → "
                f"{dst_n}.{dst_p}（张量 {tname}），得到 "
                f"{got[0]}.{got[1]} → {got[2]}.{got[3]}（张量 {got[4]}）"
                "——源/目的端口被一致重接具名拒绝")
    extra_edges = sorted(set(edges_by_id) - {c[0] for c in _EXPECTED_CHAIN})
    if extra_edges:
        e = edges_by_id[extra_edges[0]]
        raise ExtractionError(
            f"首入口图出现必要链之外的边: {extra_edges}（{e.src_node_id}."
            f"{e.src_port} → {e.dst_node_id}.{e.dst_port}，张量 "
            f"{e.tensor_id}）——多余数据通路具名拒绝")

    def _only(items, what):
        if len(items) != 1:
            raise ExtractionError(
                f"首入口图 {what} 不唯一（{sorted(items)}）——多个独立"
                "main_x/embed 身份语义篡改具名拒绝")
        return items[0]

    # 唯一 main_x 与唯一投影路径
    main_x_names = [n for n in tensors if "main_x" in n]
    _only(main_x_names, "main_x 张量")
    mx = tensors["entry:main_x"]
    if mx.logical_layout != ("B", "T", "H"):
        raise ExtractionError(
            f"main_x 轴 {mx.logical_layout} 应为 (B,T,H)")
    proj = sorted(n for n, nd in nodes.items() if nd.op_type == "linear")
    if proj != ["entry.main_proj"]:
        raise ExtractionError(
            f"首入口 linear 节点应恰为 entry.main_proj，得到 {proj}——"
            "第二投影路径（多份独立 main_x）具名拒绝")
    norms = sorted(n for n, nd in nodes.items() if nd.op_type == "rms_norm")
    if norms != ["entry.main_norm"]:
        raise ExtractionError(
            f"首入口 rms_norm 节点应恰为 entry.main_norm，得到 {norms}——"
            "第二 norm 生产者（独立 main_x）具名拒绝")
    mx_out = [e for e in graph.edges if e.tensor_id == mx.stable_id]
    for e in mx_out:
        if e.src_node_id != "entry.main_norm" or e.src_port != "y":
            raise ExtractionError(
                f"main_x 出边 {e.stable_id} 源 {e.src_node_id}.{e.src_port}"
                " 非 entry.main_norm.y——main_x 存在第二生产者")

    # readout 边界：三 mean + 唯一 concat + 端口顺序（主干遍历顺序）
    means = sorted(n for n, nd in nodes.items() if nd.op_type == "reduce_mean")
    expect_means = [f"entry.readout_mean_{lid}" for lid in (40, 41, 42)]
    if means != expect_means:
        raise ExtractionError(
            f"首入口 reduce_mean 节点应恰为 {expect_means}，得到 {means}"
            "——readout 边界缺失/多余具名拒绝")
    cats = sorted(n for n, nd in nodes.items() if nd.op_type == "concat")
    if cats != ["entry.readout_concat"]:
        raise ExtractionError(
            f"首入口 concat 节点应恰为 entry.readout_concat，得到 {cats}"
            "——第二 concat 通路具名拒绝")
    cat_ports = {p.name: p.tensor_id for p in
                 nodes["entry.readout_concat"].input_ports}
    expect_cat_ports = {port: tensors[f"entry:mean_h_{lid}"].stable_id
                        for lid, port in zip((40, 41, 42),
                                             READOUT_PORT_ORDER)}
    if cat_ports != expect_cat_ports:
        raise ExtractionError(
            f"readout concat 端口 {cat_ports} 应依次为 {expect_cat_ports}"
            "（s0/s1/s2 = 主干层 40/41/42 遍历顺序 mean 结果）——交换两个"
            " mean 结果的 concat 端口/来源具名拒绝")

    # 三 stage 分发：feed 节点 x 端口全部消费同一 main_x
    feeds = sorted(n for n, nd in nodes.items()
                   if nd.op_type == "stage_dispatch")
    expect_feeds = [f"entry.stage_feed_{i}"
                    for i in range(values.n_mtp_layers)]
    if feeds != expect_feeds:
        raise ExtractionError(
            f"stage 分发节点 {feeds} 应恰为 {expect_feeds}——main_x 必须"
            "分发全部三 stage（forward_spec 931-932 透传），缺失即"
            "多份独立 main_x / 单 stage 私有化具名拒绝")
    for fid in feeds:
        f = nodes[fid]
        xport = [p for p in f.input_ports if p.name == "x"]
        if len(xport) != 1 or xport[0].tensor_id != mx.stable_id:
            raise ExtractionError(
                f"{fid}.x 端口不消费唯一 main_x 张量——分发边被替换为"
                "独立投影产物即具名拒绝")
        inbound = [e for e in graph.edges
                   if e.dst_node_id == fid and e.dst_port == "x"]
        if len(inbound) != 1 or inbound[0].tensor_id != mx.stable_id:
            raise ExtractionError(
                f"{fid}.x 的入边不消费唯一 main_x 张量")
    # 草稿 h 只进 stage0
    eh = tensors["entry:entry_h"]
    if eh.logical_layout != ("B", "N_draft", "N_hc", "H"):
        raise ExtractionError(
            f"entry_h 轴 {eh.logical_layout} 应为 (B,N_draft,N_hc,H)")
    h_in_feeds = [fid for fid in feeds
                  if any(p.name == "h" for p in nodes[fid].input_ports)]
    if h_in_feeds != ["entry.stage_feed_0"]:
        raise ExtractionError(
            f"草稿 h 的 stage 消费 {h_in_feeds} 应恰为 "
            "['entry.stage_feed_0']——h 分发到 stage1/2 与源码不符"
            "（stage1/2 h_in 来自前一 stage h_out）")
    h_edges = [e for e in graph.edges if e.tensor_id == eh.stable_id
               and e.dst_node_id.startswith("entry.stage_feed")]
    if [(e.dst_node_id, e.dst_port) for e in h_edges] != \
            [("entry.stage_feed_0", "h")]:
        raise ExtractionError("entry_h 到 stage 的边应恰为 feed_0.h")

    # T/K 不混
    for name in ("entry:main_hidden", "entry:main_proj_y", "entry:main_x"):
        syms = tensors[name].logical_layout
        if "T" not in syms or "N_draft" in syms:
            raise ExtractionError(
                f"{name} 轴 {syms} 应含主干 T、不含草稿 N_draft——"
                "T_main 与 draft K 混用具名拒绝")
    for name in ("entry:draft_noise", "entry:draft_input_ids",
                 "entry:embed_y", "entry:entry_h"):
        syms = tensors[name].logical_layout
        if "N_draft" not in syms or "T" in syms:
            raise ExtractionError(
                f"{name} 轴 {syms} 应含草稿 N_draft、不含主干 T")

    # seed rank-1（rank-2 [B,T_in] 伪造具名拒绝）
    seed = tensors["entry:seed_ids"]
    if seed.logical_layout != ("B",):
        raise ExtractionError(
            f"seed input_ids 轴 {seed.logical_layout} 应为 rank-1 (B,)——"
            "主干 hc_head 折合 HC 为 [B,T,H] 后 ParallelHead 取 x[:,-1]"
            "（731-734）→ logits [B,V] → sample argmax → [B]（922-924、"
            "939-946），855 首列 [B] 槽精确写入无广播；rank-2 [B,T_in] "
            "seed 伪造具名拒绝")

    # draft 链结构：noise → 首列写 → embed → HC 展开
    fill = sorted(n for n, nd in nodes.items() if nd.op_type == "fill_const")
    _only(fill, "noise_fill 节点")
    write = sorted(n for n, nd in nodes.items() if nd.op_type == "index_write")
    _only(write, "first_col_write 节点")
    emb = sorted(n for n, nd in nodes.items() if nd.op_type == "embedding")
    if emb != ["entry.embed_lookup"]:
        raise ExtractionError(
            f"首入口 embedding 节点应恰为 entry.embed_lookup，得到 {emb}——"
            "多份 embed 查表（复制身份）具名拒绝")
    embw = sorted(n for n, nd in nodes.items()
                  if nd.op_type == "embedding_weight")
    if embw != ["entry.embed_w"]:
        raise ExtractionError(
            f"首入口 embedding_weight 节点应恰为 entry.embed_w，得到 "
            f"{embw}——共享 embed 复制为独立参数具名拒绝")
    ew_names = [n for n in tensors if "embed_w" in n]
    _only(ew_names, "embed_w 张量")
    diid = tensors["entry:draft_input_ids"]
    di_out = [e for e in graph.edges if e.tensor_id == diid.stable_id]
    if [(e.src_node_id, e.dst_node_id, e.dst_port) for e in di_out] != \
            [("entry.first_col_write", "entry.embed_lookup", "x")]:
        raise ExtractionError(
            "draft_input_ids 应由 first_col_write 唯一生产且只被 "
            "embed_lookup.x 消费")
    fcw_ports = {p.name: p.tensor_id
                 for p in nodes["entry.first_col_write"].input_ports}
    if fcw_ports != {"dst": tensors["entry:draft_noise"].stable_id,
                     "src": tensors["entry:seed_ids"].stable_id}:
        raise ExtractionError(
            "first_col_write 输入端口应为 dst=noise / src=seed_ids"
            "（854-855 noise 填充 + 首列写 rank-1 seed input_ids）")
    axmap = {a.symbol: a for a in graph.named_axes}
    if axmap["N_tgtH"].expression != EXPR_MAIN_PROJ_IN:
        raise ExtractionError(
            f"N_tgtH 表达式 {axmap['N_tgtH'].expression!r} 应为 "
            f"{EXPR_MAIN_PROJ_IN!r}")
    if axmap["N_tgt"].value != len(values.dspark_target_layer_ids):
        raise ExtractionError("N_tgt 轴值与固定 config 不一致")
    if axmap["T_in"].value != 1:
        raise ExtractionError("T_in 前提轴应为 1（rank-1 [B] seed 的"
                              "每批元素数）")
    return None


# ---------------------------------------------------------------------------
# 首入口端口/角色绑定
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EntryMainXBinding:
    """main_x 角色/参数绑定：唯一生产 + 三 stage 共享分发。"""

    producer: str
    param_path: str
    modules: tuple[str, ...]
    in_width_expr: str
    readout_order: tuple[int, ...]
    consumers: tuple[str, ...]
    shared_identity: bool
    out_shape: str
    note: str

    def to_dict(self) -> dict:
        return {
            "producer": self.producer,
            "param_path": self.param_path,
            "modules": list(self.modules),
            "in_width_expr": self.in_width_expr,
            "readout_order": list(self.readout_order),
            "consumers": list(self.consumers),
            "shared_identity": self.shared_identity,
            "out_shape": self.out_shape,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, doc: dict) -> "EntryMainXBinding":
        if not isinstance(doc, dict):
            raise ExtractionError("EntryMainXBinding 必须是 JSON 对象")
        for k in ("producer", "param_path", "modules", "in_width_expr",
                  "readout_order", "consumers", "shared_identity",
                  "out_shape", "note"):
            if k not in doc:
                raise ExtractionError(f"EntryMainXBinding 缺字段: {k}")
        for k in ("producer", "param_path", "in_width_expr", "out_shape",
                  "note"):
            if not isinstance(doc[k], str) or not doc[k].strip():
                raise ExtractionError(
                    f"EntryMainXBinding.{k} 必须是非空字符串")
        for k in ("modules", "consumers"):
            if not isinstance(doc[k], list) or not all(
                    isinstance(s, str) and s.strip() for s in doc[k]):
                raise ExtractionError(
                    f"EntryMainXBinding.{k} 必须是非空字符串列表")
        ro = doc["readout_order"]
        if not isinstance(ro, list) or not all(
                isinstance(v, int) and not isinstance(v, bool) for v in ro):
            raise ExtractionError(
                "EntryMainXBinding.readout_order 必须是 int 列表")
        if not isinstance(doc["shared_identity"], bool):
            raise ExtractionError(
                "EntryMainXBinding.shared_identity 必须是布尔")
        return cls(producer=doc["producer"], param_path=doc["param_path"],
                   modules=tuple(doc["modules"]),
                   in_width_expr=doc["in_width_expr"],
                   readout_order=tuple(ro),
                   consumers=tuple(doc["consumers"]),
                   shared_identity=doc["shared_identity"],
                   out_shape=doc["out_shape"], note=doc["note"])


@dataclass(frozen=True)
class EntryDraftTokens:
    """草稿 token 构造绑定：noise 来源/首列/形状/消费。"""

    noise_token_id: int
    noise_source: str
    block_size_source: str
    draft_k: int
    first_col_src: str
    shape: str
    consumer: str

    def to_dict(self) -> dict:
        return {
            "noise_token_id": self.noise_token_id,
            "noise_source": self.noise_source,
            "block_size_source": self.block_size_source,
            "draft_k": self.draft_k,
            "first_col_src": self.first_col_src,
            "shape": self.shape,
            "consumer": self.consumer,
        }

    @classmethod
    def from_dict(cls, doc: dict) -> "EntryDraftTokens":
        if not isinstance(doc, dict):
            raise ExtractionError("EntryDraftTokens 必须是 JSON 对象")
        for k in ("noise_token_id", "noise_source", "block_size_source",
                  "draft_k", "first_col_src", "shape", "consumer"):
            if k not in doc:
                raise ExtractionError(f"EntryDraftTokens 缺字段: {k}")
        for k in ("noise_token_id", "draft_k"):
            v = doc[k]
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ExtractionError(
                    f"EntryDraftTokens.{k} 必须是非负 int")
        for k in ("noise_source", "block_size_source", "first_col_src",
                  "shape", "consumer"):
            if not isinstance(doc[k], str) or not doc[k].strip():
                raise ExtractionError(
                    f"EntryDraftTokens.{k} 必须是非空字符串")
        return cls(noise_token_id=doc["noise_token_id"],
                   noise_source=doc["noise_source"],
                   block_size_source=doc["block_size_source"],
                   draft_k=doc["draft_k"], first_col_src=doc["first_col_src"],
                   shape=doc["shape"], consumer=doc["consumer"])


@dataclass(frozen=True)
class EntrySharedEmbed:
    """共享 embed 引用绑定：Transformer 同一对象，非三份复制。"""

    ref: str
    transformer_object: str
    instance_paths: tuple[str, ...]
    copied: bool
    weight_dtype: str
    weight_shape: str

    def to_dict(self) -> dict:
        return {
            "ref": self.ref,
            "transformer_object": self.transformer_object,
            "instance_paths": list(self.instance_paths),
            "copied": self.copied,
            "weight_dtype": self.weight_dtype,
            "weight_shape": self.weight_shape,
        }

    @classmethod
    def from_dict(cls, doc: dict) -> "EntrySharedEmbed":
        if not isinstance(doc, dict):
            raise ExtractionError("EntrySharedEmbed 必须是 JSON 对象")
        for k in ("ref", "transformer_object", "instance_paths", "copied",
                  "weight_dtype", "weight_shape"):
            if k not in doc:
                raise ExtractionError(f"EntrySharedEmbed 缺字段: {k}")
        ip = doc["instance_paths"]
        if not isinstance(ip, list) or not all(
                isinstance(s, str) and s.strip() for s in ip):
            raise ExtractionError(
                "EntrySharedEmbed.instance_paths 必须是非空字符串列表")
        if not isinstance(doc["copied"], bool):
            raise ExtractionError("EntrySharedEmbed.copied 必须是布尔")
        for k in ("ref", "transformer_object", "weight_dtype",
                  "weight_shape"):
            if not isinstance(doc[k], str) or not doc[k].strip():
                raise ExtractionError(
                    f"EntrySharedEmbed.{k} 必须是非空字符串")
        return cls(ref=doc["ref"], transformer_object=doc["transformer_object"],
                   instance_paths=tuple(ip), copied=doc["copied"],
                   weight_dtype=doc["weight_dtype"],
                   weight_shape=doc["weight_shape"])


@dataclass(frozen=True)
class EntryStagePort:
    """一个 stage 的首入口端口绑定（main_x 全部、draft h 仅 stage0）。"""

    stage_id: int
    param_path: str
    main_x_in: bool
    draft_h_in: bool
    draft_h_src: str
    feed_node: str

    def to_dict(self) -> dict:
        return {
            "stage_id": self.stage_id,
            "param_path": self.param_path,
            "main_x_in": self.main_x_in,
            "draft_h_in": self.draft_h_in,
            "draft_h_src": self.draft_h_src,
            "feed_node": self.feed_node,
        }

    @classmethod
    def from_dict(cls, doc: dict) -> "EntryStagePort":
        if not isinstance(doc, dict):
            raise ExtractionError("EntryStagePort 必须是 JSON 对象")
        for k in ("stage_id", "param_path", "main_x_in", "draft_h_in",
                  "draft_h_src", "feed_node"):
            if k not in doc:
                raise ExtractionError(f"EntryStagePort 缺字段: {k}")
        sid = doc["stage_id"]
        if not isinstance(sid, int) or isinstance(sid, bool) or sid < 0:
            raise ExtractionError(
                f"EntryStagePort.stage_id 必须是非负 int，得到 {sid!r}")
        for k in ("main_x_in", "draft_h_in"):
            if not isinstance(doc[k], bool):
                raise ExtractionError(f"EntryStagePort.{k} 必须是布尔")
        for k in ("param_path", "draft_h_src", "feed_node"):
            if not isinstance(doc[k], str) or not doc[k].strip():
                raise ExtractionError(
                    f"EntryStagePort.{k} 必须是非空字符串")
        return cls(stage_id=sid, param_path=doc["param_path"],
                   main_x_in=doc["main_x_in"], draft_h_in=doc["draft_h_in"],
                   draft_h_src=doc["draft_h_src"], feed_node=doc["feed_node"])


@dataclass(frozen=True)
class EntryTemperature:
    """temperature 来源记录（不属于首入口；ModelValues 无此字段）。"""

    value_source: str
    run_config_override: None
    consumers: tuple[str, ...]
    in_entry: bool

    def to_dict(self) -> dict:
        return {
            "value_source": self.value_source,
            "run_config_override": self.run_config_override,
            "consumers": list(self.consumers),
            "in_entry": self.in_entry,
        }

    @classmethod
    def from_dict(cls, doc: dict) -> "EntryTemperature":
        if not isinstance(doc, dict):
            raise ExtractionError("EntryTemperature 必须是 JSON 对象")
        for k in ("value_source", "run_config_override", "consumers",
                  "in_entry"):
            if k not in doc:
                raise ExtractionError(f"EntryTemperature 缺字段: {k}")
        if not isinstance(doc["value_source"], str) or \
                not doc["value_source"].strip():
            raise ExtractionError(
                "EntryTemperature.value_source 必须是非空字符串")
        if "run_config_override" not in doc or \
                doc["run_config_override"] is not None:
            raise ExtractionError(
                "EntryTemperature.run_config_override 必须为 null——run "
                "config 无 temperature 字段，声称覆盖即伪造来源")
        cons = doc["consumers"]
        if not isinstance(cons, list) or not all(
                isinstance(s, str) and s.strip() for s in cons):
            raise ExtractionError(
                "EntryTemperature.consumers 必须是非空字符串列表")
        if not isinstance(doc["in_entry"], bool):
            raise ExtractionError("EntryTemperature.in_entry 必须是布尔")
        return cls(value_source=doc["value_source"],
                   run_config_override=None,
                   consumers=tuple(cons), in_entry=doc["in_entry"])


@dataclass(frozen=True)
class EntryBindings:
    """B3a 首入口角色/参数/三 stage 端口绑定规范（canonical JSON 往返）。"""

    schema: str
    template_id: str
    n_stages: int
    n_main_layers: int
    main_x: EntryMainXBinding
    draft_tokens: EntryDraftTokens
    shared_embed: EntrySharedEmbed
    stage_ports: tuple[EntryStagePort, ...]
    temperature: EntryTemperature
    call_premises: dict[str, str]
    evidence: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "template_id": self.template_id,
            "n_stages": self.n_stages,
            "n_main_layers": self.n_main_layers,
            "main_x": self.main_x.to_dict(),
            "draft_tokens": self.draft_tokens.to_dict(),
            "shared_embed": self.shared_embed.to_dict(),
            "stage_ports": [p.to_dict() for p in self.stage_ports],
            "temperature": self.temperature.to_dict(),
            "call_premises": dict(self.call_premises),
            "evidence": list(self.evidence),
        }


def build_entry_bindings(values: ModelValues) -> EntryBindings:
    """首入口绑定（纯派生，数值来自 ModelValues/固定常量，不改固定输入）。"""
    n = values.n_mtp_layers
    ports = []
    for i in range(n):
        src = "mtp.0.forward_embed" if i == 0 else f"mtp.{i - 1}.h_out"
        ports.append(EntryStagePort(
            stage_id=i, param_path=f"mtp.{i}", main_x_in=True,
            draft_h_in=(i == 0), draft_h_src=src,
            feed_node=f"entry.stage_feed_{i}"))
    readout_order = tuple(values.dspark_target_layer_ids)
    premises = {
        "t_main_vs_draft_k":
            "main_hidden/main_x 沿主干 token 轴 T（prefill/decode 场景轴，"
            f"prefill=T_main 个 token、decode=1）；草稿侧沿 K=N_draft="
            f"{values.dspark_block_size}（ModelValues.dspark_block_size，"
            "run config dspark_block_size——ModelValues 确有该字段，构建"
            "器直接使用）。forward_embed 同时消费两轴：main_hidden "
            "[B,T,H*N_tgt] 进 main_proj（T 不变），draft_input_ids [B,K] "
            "进 shared.embed（K 独立）——两轴不混用，main_proj 不改 T、"
            "embed 不见 T。forward_head 的 output_ids 是 [B,K+1]（B3b "
            "末出口范围，本单不建）",
        "input_ids_width":
            "forward_spec 的 seed input_ids 是 rank-1 [B]（主干 forward "
            "922 hc_head 折合 HC 为 [B,T,H] → 923 ParallelHead not "
            "full_logits 取 x[:,-1]（731-734）→ logits [B,V] → 924 "
            "sample（939-946）argmax(dim=-1) → [B]；958/961 参考入口把"
            "该 [B] output_ids 直接作为 forward_spec 的 input_ids）；"
            "model.py:855 draft_input_ids[:,0]=input_ids 以 [B] 槽精确"
            "写入（new_full [B,K] 首列赋值，无广播）。T_main 由 "
            "main_hidden 承载，与 seed 宽度解耦（证据 ev_ep_entry_premise "
            "derived）",
        "forward_spec_caller":
            "generate.py:41-45 推理循环只调 model.forward——不声称 "
            "generate.py 实际调用 forward_spec；forward_spec 运行时调用方"
            " unknown（不编造）。参考入口 __main__（model.py:957-961）"
            "演示 prefill 一次 + 逐 token decode 的调用模式，属演示非"
            "运行时保证",
        "noise_token_id_source":
            f"run config dspark_noise_token_id={NOISE_TOKEN_ID}"
            "（config.json:9，证据 ev_ep_noise_cfg）。事实澄清：ModelValues "
            f"确有 dspark_block_size={values.dspark_block_size} 字段"
            "（rules 178/260，K 轴已直接使用）；真正缺的字段只有 "
            "noise_token_id 与 temperature——以常量+证据原文锚定表达，"
            "不冒充也不否认 ModelValues 字段（输入基线 SHA 核对防 config "
            "漂移）",
        "temperature_source":
            "ModelArgs 默认 temperature: float = 1（model.py:39，证据 "
            "ev_ep_temp_default）；run config 无 temperature 字段（无"
            "覆盖）。temperature 属 last stage forward_head 的 sample"
            "（871）与主干 forward（924），不属于首入口 forward_embed"
            "（851-858）——in_entry=false",
    }
    return EntryBindings(
        schema=ENTRY_SCHEMA, template_id=ENTRY_TEMPLATE_ID,
        n_stages=n, n_main_layers=values.n_layers,
        main_x=EntryMainXBinding(
            producer="mtp.0.forward_embed", param_path="mtp.0",
            modules=("main_proj", "main_norm"),
            in_width_expr=EXPR_MAIN_PROJ_IN,
            readout_order=readout_order,
            consumers=tuple(f"mtp.{i}" for i in range(n)),
            shared_identity=True, out_shape="[B, T, H]",
            note="唯一 main_x：三 readout（40/41/42 主干遍历顺序 mean 后"
                 " concat，图中 entry.readout_mean_40/41/42 → "
                 "entry.readout_concat 端口 s0/s1/s2）全部汇入 stage0 "
                 "唯一 main_proj/main_norm（无 readout→stage1/2 直接"
                 "投影）；同一 main_x 第三实参透传全部 stage（forward_spec "
                 "931-932）"),
        draft_tokens=EntryDraftTokens(
            noise_token_id=NOISE_TOKEN_ID,
            noise_source="run config dspark_noise_token_id（config.json:9，"
                         "证据 ev_ep_noise_cfg）",
            block_size_source="ModelValues.dspark_block_size（rules 178/"
                              "260）= run config dspark_block_size"
                              "（config.json:8）",
            draft_k=values.dspark_block_size, first_col_src="seed_ids",
            shape="[B, K]", consumer="shared.embed"),
        shared_embed=EntrySharedEmbed(
            ref="shared.embed", transformer_object="transformer.embed",
            instance_paths=tuple(f"mtp.{i}.embed" for i in range(n)),
            copied=False, weight_dtype="bf16",
            weight_shape="[V, H]（world_size=1 分析假设）"),
        stage_ports=tuple(ports),
        temperature=EntryTemperature(
            value_source="ModelArgs 默认 temperature: float = 1"
                         "（model.py:39，证据 ev_ep_temp_default）",
            run_config_override=None,
            consumers=("mtp.last.forward_head.sample",
                       "transformer.forward.sample"),
            in_entry=False),
        call_premises=premises,
        evidence=("ev_ep_noise_cfg", "ev_ep_readout_bound",
                  "ev_ep_temp_default", "ev_ep_generate_calls",
                  "ev_ep_embed_param", "ev_ep_entry_premise",
                  "ev_st_forward_embed", "ev_st_forward_spec",
                  "ev_st_layer_ids", "ev_st_main_proj",
                  "ev_st_norm_params"),
    )


def _verify_entry_bindings(b: EntryBindings, values: ModelValues) -> None:
    """绑定一致性核对（读回必检）：期望来自固定 ModelValues/run config
    原文证据/固定常量——不以被检文档自身拼“正确”。角色错绑/身份复制/
    readout 交换/来源伪造具名拒绝。"""
    if b.schema != ENTRY_SCHEMA:
        raise ExtractionError(f"entry bindings schema 不符: {b.schema!r}")
    if b.template_id != ENTRY_TEMPLATE_ID:
        raise ExtractionError(
            f"entry bindings template_id 不符: {b.template_id!r}")
    if b.n_stages != values.n_mtp_layers or \
            b.n_main_layers != values.n_layers:
        raise ExtractionError(
            f"entry bindings n_stages/n_main_layers（{b.n_stages}/"
            f"{b.n_main_layers}）与固定 config（{values.n_mtp_layers}/"
            f"{values.n_layers}）不一致")

    mx = b.main_x
    # stage0 角色错绑：生产者/参数路径必须 stage0
    if mx.producer != "mtp.0.forward_embed" or mx.param_path != "mtp.0":
        raise ExtractionError(
            f"main_x 生产者/参数路径（{mx.producer!r}/{mx.param_path!r}）"
            "必须为 mtp.0.forward_embed/mtp.0——stage0 角色错绑具名拒绝")
    if mx.modules != ("main_proj", "main_norm"):
        raise ExtractionError(
            f"main_x modules {mx.modules} 应为 ('main_proj','main_norm')")
    if mx.in_width_expr != EXPR_MAIN_PROJ_IN:
        raise ExtractionError(
            f"main_x in_width_expr {mx.in_width_expr!r} 应为 "
            f"{EXPR_MAIN_PROJ_IN!r}")
    # readout 顺序：期望来自固定 config（交换 40/41/42 即拒）；图中对应
    # entry.readout_mean_40/41/42 → entry.readout_concat 端口 s0/s1/s2。
    if mx.readout_order != tuple(values.dspark_target_layer_ids):
        raise ExtractionError(
            f"main_x readout_order {mx.readout_order} 与固定 config "
            f"dspark_target_layer_ids {tuple(values.dspark_target_layer_ids)}"
            " 不一致（图端口 s0/s1/s2 = entry.readout_mean_40/41/42.y）"
            "——readout 交换具名拒绝")
    expect_consumers = tuple(f"mtp.{i}" for i in range(values.n_mtp_layers))
    if mx.consumers != expect_consumers or mx.shared_identity is not True:
        raise ExtractionError(
            f"main_x 消费者 {mx.consumers}/shared_identity="
            f"{mx.shared_identity} 应为全部三 stage 共享同一身份——删除"
            "分发或声称独立 main_x 具名拒绝")

    dt = b.draft_tokens
    if dt.noise_token_id != NOISE_TOKEN_ID:
        raise ExtractionError(
            f"draft_tokens.noise_token_id {dt.noise_token_id} 与固定 run "
            f"config {NOISE_TOKEN_ID}（ev_ep_noise_cfg 原文锚定）不一致")
    if dt.draft_k != values.dspark_block_size:
        raise ExtractionError(
            f"draft_tokens.draft_k {dt.draft_k} 与固定 config "
            f"dspark_block_size {values.dspark_block_size} 不一致")
    # 事实边界：ModelValues 有 dspark_block_size（rules 178/260），真正
    # 缺的字段只有 noise_token_id/temperature——来源声明不得声称其缺失。
    if "ModelValues.dspark_block_size" not in dt.block_size_source:
        raise ExtractionError(
            f"draft_tokens.block_size_source {dt.block_size_source!r} 必须"
            "锚定 ModelValues.dspark_block_size（rules 178/260，该字段"
            "实际存在）——把已有字段误称为缺失/来源漂移具名拒绝")
    if dt.first_col_src != "seed_ids" or dt.consumer != "shared.embed":
        raise ExtractionError(
            "draft_tokens 首列来源/consumer 与 forward_embed 851-858 不符"
            "（首列写入 rank-1 seed_ids [B]）")

    se = b.shared_embed
    if se.ref != "shared.embed" or se.transformer_object != \
            "transformer.embed" or se.copied is not False:
        raise ExtractionError(
            "shared_embed 必须引用 transformer.embed 同一对象且 copied="
            f"false——得到 {se.ref!r}/{se.transformer_object!r}/"
            f"{se.copied}；共享 embed 复制身份具名拒绝")
    expect_paths = tuple(f"mtp.{i}.embed"
                         for i in range(values.n_mtp_layers))
    if se.instance_paths != expect_paths:
        raise ExtractionError(
            f"shared_embed.instance_paths {se.instance_paths} 应恰为 "
            f"{expect_paths}（model.py:903 逐实例绑定同一对象）——复制/"
            "改名路径具名拒绝")

    ports = b.stage_ports
    if len(ports) != values.n_mtp_layers:
        raise ExtractionError(
            f"stage_ports 数量 {len(ports)} 与 n_stages 不一致")
    for i, p in enumerate(ports):
        if p.stage_id != i or p.param_path != f"mtp.{i}":
            raise ExtractionError(
                f"stage_ports[{i}] 序号/参数路径（{p.stage_id}/"
                f"{p.param_path}）不对应——角色错绑具名拒绝")
        if p.main_x_in is not True:
            raise ExtractionError(
                f"stage_ports[{i}].main_x_in 应为 True（同一 main_x 分发"
                "全部三 stage）")
        if p.draft_h_in is not (i == 0):
            raise ExtractionError(
                f"stage_ports[{i}].draft_h_in {p.draft_h_in} 应为 "
                f"{i == 0}——草稿 h 只进 stage0（stage1/2 h_in 来自前一 "
                "stage h_out，非 forward_embed）")
        expect_src = "mtp.0.forward_embed" if i == 0 else f"mtp.{i-1}.h_out"
        if p.draft_h_src != expect_src:
            raise ExtractionError(
                f"stage_ports[{i}].draft_h_src {p.draft_h_src!r} 应为 "
                f"{expect_src!r}（草稿 h 链条）")
        if p.feed_node != f"entry.stage_feed_{i}":
            raise ExtractionError(
                f"stage_ports[{i}].feed_node {p.feed_node!r} 与首入口图"
                "边界节点不对应")

    temp = b.temperature
    if "model.py:39" not in temp.value_source:
        raise ExtractionError(
            "temperature.value_source 必须锚定 ModelArgs 默认值"
            "（model.py:39）——其它来源声称具名拒绝")
    if temp.run_config_override is not None or temp.in_entry is not False:
        raise ExtractionError(
            "temperature.run_config_override 必须为 null 且 in_entry="
            "false——run config 无该字段、temperature 不属于首入口，"
            "伪造覆盖/入口归属具名拒绝")
    if not temp.consumers:
        raise ExtractionError("temperature.consumers 不应为空")

    if set(b.call_premises.keys()) != set(PREMISE_KEYS):
        raise ExtractionError(
            f"call_premises 键集 {sorted(b.call_premises)} 应恰为 "
            f"{sorted(PREMISE_KEYS)}")
    for k, v in b.call_premises.items():
        if not isinstance(v, str) or not v.strip():
            raise ExtractionError(f"call_premises[{k}] 必须是非空字符串")
    if "generate.py" not in b.call_premises["forward_spec_caller"] or \
            "不声称" not in b.call_premises["forward_spec_caller"]:
        raise ExtractionError(
            "call_premises.forward_spec_caller 必须声明 generate.py 只调 "
            "model.forward、不声称其实际调用 forward_spec")
    if "model.py:39" not in b.call_premises["temperature_source"]:
        raise ExtractionError(
            "call_premises.temperature_source 必须锚定 model.py:39")
    if str(NOISE_TOKEN_ID) not in b.call_premises["noise_token_id_source"]:
        raise ExtractionError(
            "call_premises.noise_token_id_source 必须锚定固定噪声 token")
    width = b.call_premises["input_ids_width"]
    if "rank-1 [B]" not in width or "x[:,-1]" not in width or \
            "无广播" not in width:
        raise ExtractionError(
            "call_premises.input_ids_width 必须声明 seed 为 rank-1 [B]"
            "（ParallelHead x[:,-1] → sample [B]，855 首列 [B] 槽写入"
            "无广播）——rank-2 [B,1] 广播声称具名拒绝")
    if "ModelValues.dspark_block_size" not in \
            b.call_premises["noise_token_id_source"] and \
            "ModelValues.dspark_block_size" not in b.draft_tokens.block_size_source:
        raise ExtractionError(
            "noise_token_id_source 或 block_size_source 必须澄清 "
            "ModelValues 有 dspark_block_size 字段（rules 178/260）——"
            "把已有字段误称为缺失具名拒绝")
    return None


def serialize_entry_bindings(b: EntryBindings) -> str:
    """绑定公开规范序列化（规范化 JSON，重复构建逐字节一致）。"""
    return json.dumps(b.to_dict(), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def deserialize_entry_bindings(text: str, values: ModelValues | None = None,
                               workspace_root: Path | None = None
                               ) -> EntryBindings:
    """绑定读回：schema/template_id 显式核对 + 重建后全量一致性核对
    （_verify_entry_bindings——期望来自固定 ModelValues/run config 证据/
    固定常量，不以被检 JSON 充当可信基线）。values 缺省从固定工作区加载
    （load_model_values，固定输入只读）。"""
    root = Path(workspace_root) if workspace_root is not None \
        else find_workspace_root()
    values = values if values is not None else load_model_values(root)
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"entry bindings JSON 不合法: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema") != ENTRY_SCHEMA:
        raise ExtractionError(
            f"entry bindings schema 不符，期望 {ENTRY_SCHEMA}")
    for key in ("template_id", "n_stages", "n_main_layers", "main_x",
                "draft_tokens", "shared_embed", "stage_ports", "temperature",
                "call_premises", "evidence"):
        if key not in doc:
            raise ExtractionError(f"entry bindings 缺字段: {key}")
    premises = doc["call_premises"]
    if not isinstance(premises, dict):
        raise ExtractionError("call_premises 必须是 JSON 对象")
    b = EntryBindings(
        schema=doc["schema"], template_id=doc["template_id"],
        n_stages=doc["n_stages"], n_main_layers=doc["n_main_layers"],
        main_x=EntryMainXBinding.from_dict(doc["main_x"]),
        draft_tokens=EntryDraftTokens.from_dict(doc["draft_tokens"]),
        shared_embed=EntrySharedEmbed.from_dict(doc["shared_embed"]),
        stage_ports=tuple(EntryStagePort.from_dict(p)
                          for p in doc["stage_ports"]),
        temperature=EntryTemperature.from_dict(doc["temperature"]),
        call_premises={k: v for k, v in premises.items()},
        evidence=tuple(doc["evidence"]),
    )
    _verify_entry_bindings(b, values)
    return b


# ---------------------------------------------------------------------------
# bundle 与入口
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DSparkEntryBundle:
    """B3a 交付包：首入口图 + 证据注册表（同源全量）+ 模型数值 + 绑定。"""

    graph: SemanticGraph
    registry: EvidenceRegistry
    values: ModelValues
    bindings: EntryBindings


def build_dspark_entry_graph(
    workspace_root: Path | None = None,
    *,
    values: ModelValues | None = None,
    evidence_base: EvidenceBase | None = None,
) -> DSparkEntryBundle:
    """构建首入口语义图与绑定（静态取证，不导入/不执行模型代码）。

    证据核对 → 同源注册（reference + B1 Attention + B2 stage + 本包
    ev_ep_*）→ 图构建 → 通用引用检查 → verify_entry_graph 语义核对 →
    绑定构建与核对。任何失败具名拒绝、不给图（EvidenceError /
    ExtractionError）。"""
    root = Path(workspace_root) if workspace_root is not None \
        else find_workspace_root()
    values = values or load_model_values(root)
    evidence_base = evidence_base or load_evidence_base(root)
    registry = EvidenceRegistry(evidence_base, root)
    for ev in _reference_evidence_defs() + _attn_evidence_defs() \
            + _stage_evidence_defs() + _entry_evidence_defs():
        registry.register(ev)

    axes = _entry_axes(values)
    b = _GraphBuilder(axes)
    t = _add_entry_tensors(b, axes)
    _add_entry_flow(b, t)

    for ev_id in b.evidence_ids:
        if ev_id not in registry:
            raise ExtractionError(f"图引用了未注册证据: {ev_id!r}")
    for axis in axes.values():
        if axis.evidence_ref.startswith("ev_") \
                and axis.evidence_ref not in registry:
            raise ExtractionError(
                f"轴 {axis.symbol} 引用了未注册证据: {axis.evidence_ref!r}")

    graph = b.build((), ENTRY_TEMPLATE_ID)
    verify_entry_graph(graph, values)

    bindings = build_entry_bindings(values)
    _verify_entry_bindings(bindings, values)

    return DSparkEntryBundle(graph=graph, registry=registry, values=values,
                             bindings=bindings)
