"""HiDream-O1-Image 单次 forward 的整网总览（L0）：输入 → 嵌入 → 36 层 → x_pred。

同一份结构事实的另一层视图：``path`` 是网络级算子流（文本嵌入、时间步嵌入、
像素 patch 嵌入、条件视觉塔、DeepStack、36 层循环、final_layer2 与生成 token
选择），``layer_order`` 是 36 层 decoder 的真实顺序，``groups`` 是两份真实重复
层组（decoder 36 层 / ViT 27 blocks）。层内结构在各层组的 Block/算子层，这里不
重复铺开。

与既有文本模型整网的关键差别（都按源码表达，不套模板）：

- 终点是 ``x_pred``（每生成 token 32×32×3），不是 logits；``lm_head`` 在生成路径
  根本不调用（modeling:1906-1912）。
- 时间步不是 adaLN：``t_embedder1`` 的输出被写进 tms token 的位置
  （modeling:1514-1523）。
- 文本 token 与像素 patch 在同一个序列里：``[文本嵌入; x_embedder(vinputs)]``
  （modeling:1525-1532）。
- 视觉塔只在 Kref>0（参考图/编辑/IP）执行；t2i 路径没有 pixel_values。
- 没有 KV cache（use_cache=False），没有 scheduler/采样循环（外部）。
"""
from __future__ import annotations

from pathlib import Path

from structure_reader.extract.hidream_o1_hf import MODELING, PIPELINE, UTILS, _op, _param, _rule, _src, _tensor
from structure_reader.rules.hidream_o1_hf import (
    HIDREAM_BIND_KEYS,
    HIDREAM_SCENARIOS,
    layer_group_assignments,
    load_hf_model_values,
)

__all__ = ["NETWORK_SCHEMA", "build_network_overview"]

NETWORK_SCHEMA = "hf_network_overview.v1"


def _parameters(values) -> list[dict]:
    return [
        _param("B", "B", "batch 批大小", note="可填写；结构保留符号",
               domain={"kind": "integer", "min": 1}),
        _param("Himg", "Himg", "目标图像高（生效网络尺寸）",
               note="必须能被 patch 边长整除（Himg % Ppix == 0）；原始请求尺寸的吸附在外部",
               domain={"kind": "integer", "min": 32}, consistency="Himg % Ppix == 0"),
        _param("Wimg", "Wimg", "目标图像宽（生效网络尺寸）",
               note="必须能被 patch 边长整除（Wimg % Ppix == 0）",
               domain={"kind": "integer", "min": 32}, consistency="Wimg % Ppix == 0"),
        _param("Ppix", "Ppix", "像素 patch 边长", value=32,
               source_refs=[_src("1039", "self.patch_size = 32", PIPELINE),
                            _src("18", "PATCH_SIZE = 32", PIPELINE)]),
        _param("Cin", "Cin", "像素通道数", value=3,
               source_refs=[_src("1040", "self.in_channels = 3")]),
        _param("Ptxt", "Ptxt", "文本 prompt token 数", note="chat template + boi + 1×tms；至少 1 个 token",
               domain={"kind": "integer", "min": 1}),
        _param("Kref", "Kref", "参考图数量 Kref", note="0=t2i；>0 时视觉塔与参考图 patch 分支执行"),
        _param("Sref", "Sref", "参考图像素 patch 总数（数据相关）", value=None,
               unknown_reason="参考图尺寸未给定",
               domain={"kind": "integer", "min": 0},
               consistency=("((Kref > 0) or (Sref == 0)) and "
                            "((Kref == 0) or (Sref > 0))")),
        _param("Svis", "Svis", "VLM 条件图 patch 总数（数据相关）", value=None,
               unknown_reason="条件图尺寸未给定",
               domain={"kind": "integer", "min": 0},
               consistency=("((Kref > 0) or (Svis == 0)) and "
                            "((Kref == 0) or (Svis > 0)) and "
                            "(Svis % (Mv * Mv) == 0)")),
        _param("Spix", "S_pix", "目标图像 patch 数", expression="S_pix = (Himg // Ppix) * (Wimg // Ppix)",
               source_refs=[_src("313-317", "noise → [B,(H/32)*(W/32),3072]", PIPELINE)]),
        _param("Sgen", "S_gen", "生成 token 数", expression="S_gen = S_pix + Sref",
               note="vinputs 列数（目标 patch + 参考图 patch）"),
        _param("Stot", "S_tot", "主干序列长度", expression="S_tot = Ptxt + S_gen"),
        _param("Nar", "N_ar", "AR 文本行数", expression="N_ar = Ptxt - 1"),
        _param("Ngt", "N_gt", "生成侧行数（含 tms）", expression="N_gt = S_gen + 1"),
        _param("Svism", "S_vis_m", "merge 后条件 token 数", expression="S_vis_m = Svis // (Mv * Mv)"),
        _param("H", "H", "文本主干 hidden_size", value=4096),
        _param("Nh", "Nh", "Q 头数", value=32),
        _param("Nkv", "Nkv", "KV 头数", value=8),
        _param("Dh", "Dh", "head_dim", value=128),
        _param("Hq", "H_q", "Q 拼接宽度", value=4096, expression="H_q = Nh * Dh"),
        _param("Hkv", "H_kv", "K/V 拼接宽度", value=1024, expression="H_kv = Nkv * Dh"),
        _param("Fi", "Fi", "SwiGLU 中间维", value=12288),
        _param("EpsN", "EpsN", "文本 RMSNorm eps", value=1e-6),
        _param("Nlay", "Nlay", "decoder 层数", value=36),
        _param("Vocab", "Vocab", "词表大小", value=151936),
        _param("Tms", "Tms", "timestep token id（tms_token_id）", value=151673,
               source_refs=[_src("1050", "self.tms_token_id = 151673")]),
        _param("Fq", "Fq", "时间步正弦频率维", value=256,
               source_refs=[_src("987/996", "frequency_embedding_size = 256")]),
        _param("Hv", "Hv", "视觉塔 hidden_size", value=1152),
        _param("Nhv", "Nhv", "视觉塔头数", value=16),
        _param("Dhv", "D_hv", "视觉塔 head_dim", value=72, expression="D_hv = Hv // Nhv"),
        _param("Hv3", "H_v3", "视觉 qkv 宽度", value=3456, expression="H_v3 = 3 * Hv"),
        _param("Hv4", "H_v4", "merge 后单元宽度（merge²×Hv）", value=4608,
               expression="H_v4 = Hv * (Mv * Mv)",
               note="PatchMerger 的 hidden_size = hidden * merge²（modeling:111）"),
        _param("Fv", "Fv", "视觉塔 MLP 中间维", value=4304),
        _param("EpsV", "EpsV", "视觉塔 LayerNorm eps", value=1e-6),
        _param("Pv", "Pv", "视觉 patch 边长", value=16),
        _param("Tp", "Tp", "视觉 temporal patch", value=2),
        _param("Mv", "Mv", "spatial_merge_size", value=2),
        _param("Npm", "Npm", "可学习位置嵌入数", value=2304),
        _param("Nv", "Nv", "视觉塔 block 数", value=27),
        _param("Ndh", "Ndh", "DeepStack 特征数", value=3),
        _param("Vpin", "V_pin", "条件图像素输入宽度", value=1536,
               expression="V_pin = Cin * Tp * Pv * Pv",
               note="Qwen2VLImageProcessor 的 pixel_values 每行 = C×temporal×patch×patch"),
        _param("Pin", "P_in", "像素 patch 输入宽度", value=3072,
               expression="P_in = Ppix * Ppix * Cin",
               note="BottleneckPatchEmbed 的输入维（32×32×3）"),
        _param("Pca", "P_ca", "像素 patch 低秩维（hidden//4）", value=1024,
               expression="P_ca = H // 4",
               source_refs=[_src("1042", "bottleneck_dim = hidden_size // 4")]),
        _param("Ncnd", "Ncnd", "条件图数量（=Kref，占位）", expression="Ncnd = Kref",
               note="仅用于 image_grid_thw 的第一维；不改变 token 规模"),
    ]


def _derivations() -> list[dict]:
    return [
        {"id": "S_pix", "expression": "(Himg // Ppix) * (Wimg // Ppix)", "label": "目标图像 patch 数",
         "note": "32×32 像素一个 patch token"},
        {"id": "S_gen", "expression": "S_pix + Sref", "label": "生成 token 数",
         "note": "目标 patch + 参考图 patch（同一 x_embedder）"},
        {"id": "S_tot", "expression": "Ptxt + S_gen", "label": "主干序列长度",
         "note": "文本 token + 生成 token"},
        {"id": "N_ar", "expression": "Ptxt - 1", "label": "AR 文本行数",
         "note": "tms token 归生成侧"},
        {"id": "N_gt", "expression": "S_gen + 1", "label": "生成侧行数（含 tms）",
         "note": "双向注意力行"},
        {"id": "S_vis_m", "expression": "Svis // (Mv * Mv)", "label": "merge 后条件 token 数",
         "note": "2×2 merge"},
        {"id": "H_q", "expression": "Nh * Dh", "label": "Q 拼接宽度", "note": "4096"},
        {"id": "H_kv", "expression": "Nkv * Dh", "label": "K/V 拼接宽度", "note": "1024"},
        {"id": "D_hv", "expression": "Hv // Nhv", "label": "视觉塔 head_dim", "note": "72"},
        {"id": "H_v3", "expression": "3 * Hv", "label": "视觉 qkv 宽度", "note": "3456"},
        {"id": "H_v4", "expression": "Hv * (Mv * Mv)", "label": "merge 单元宽度", "note": "4608"},
        {"id": "V_pin", "expression": "Cin * Tp * Pv * Pv", "label": "条件图像素输入宽度",
         "note": "1536"},
        {"id": "P_in", "expression": "Ppix * Ppix * Cin", "label": "像素 patch 输入宽度",
         "note": "3072"},
        {"id": "P_ca", "expression": "H // 4", "label": "像素 patch 低秩维", "note": "1024"},
    ]


def _tensors() -> list[dict]:
    return [
        _tensor("n_input_ids", "input_ids（chat template + boi + tms）", ["B", "Ptxt"], "boundary",
                "pipeline 用 apply_chat_template + boi + tms 组文本 ids（pipeline:41-46）；"
                "图像 token 不写进 input_ids，而是走 vinputs", ["B", "Ptxt"]),
        _tensor("n_position_ids", "position_ids（MRoPE 三轴）", [3, "B", "Stot"], "boundary",
                "utils:246-349 按 image_grid_thw 构造；目标图像 patch 位置钉在 4096 起",
                ["B", "Stot"]),
        _tensor("n_token_types", "token_types / vinput_mask", ["B", "Stot"], "boundary",
                "0=AR 文本；>0=生成侧（tms=3，生成 token=1，参考图 patch=2）"
                "（pipeline:62-77/294-302）", ["B", "Stot"]),
        _tensor("n_vinputs", "vinputs（像素 patch 序列）", ["B", "Sgen", "Pin"], "boundary",
                "噪声 z 与参考图 patch 的像素块（[-1,1]，32×32×3）；本层不区分来源，"
                "由 token_types 区分（pipeline:313-317/388）", ["B", "Sgen", "Pin"]),
        _tensor("n_timestep", "timestep（t = 1 - step_t/1000）", ["B"], "boundary",
                "外部调度给出的标量（pipeline:373）；本网络只做嵌入，不推进调度",
                ["B"]),
        _tensor("n_pixel_values", "pixel_values（条件图）", ["Svis", "Vpin"], "boundary",
                "Qwen3VLProcessor 产出的条件图像素块；t2i 时不存在（pipeline:264-311）",
                ["Svis", "Vpin"]),
        _tensor("n_image_grid_thw", "image_grid_thw（条件图/目标图/参考图网格）", ["Ncnd", 3],
                "boundary", "pipeline:245-272：条件图网格先除以 merge，再拼目标图与参考图",
                ["Ncnd"]),
        _tensor("n_text_embeds", "embed_tokens(input_ids)", ["B", "Ptxt", "H"], "activation",
                "词表查表；tie_word_embeddings=false（与 lm_head 不共享）", ["B", "Ptxt", "H"]),
        _tensor("n_t_freq", "正弦时间步嵌入", ["B", "Fq"], "activation",
                "timestep_embedding(t*1000, 256)", ["B", "Fq"]),
        _tensor("n_t_h1", "t_mlp[0] 输出", ["B", "H"], "activation", "Linear(256→4096)", ["B", "H"]),
        _tensor("n_t_act", "SiLU(t_h1)", ["B", "H"], "activation", "MLP 中间激活", ["B", "H"]),
        _tensor("n_t_emb", "t_emb", ["B", "H"], "activation", "Linear(4096→4096)", ["B", "H"]),
        _tensor("n_vpatches", "视觉 patch 嵌入", ["Svis", "Hv"], "activation",
                "Conv3d(Cin→Hv, kernel [Tp,Pv,Pv], stride 同)", ["Svis", "Hv"]),
        _tensor("n_vpatches_pos", "视觉 patch + 位置嵌入", ["Svis", "Hv"], "activation",
                "可学习位置插值后相加（modeling:741-742）", ["Svis", "Hv"]),
        _tensor("n_vcos", "视觉 2D rotary cos/sin", ["Svis", "Dhv"], "activation",
                "按 grid 行列坐标查频率（modeling:628-665）", ["Svis", "Dhv"]),
        _tensor("n_vtower", "视觉塔最终状态", ["Svis", "Hv"], "activation",
                "27 个 block 顺序执行后的 hidden_states（modeling:762-776）", ["Svis", "Hv"]),
        _tensor("n_vtaps", "DeepStack tap 集合", ["Ndh", "Svis", "Hv"], "activation",
                "block 8/16/24 之后的 hidden_states（deepstack_visual_indexes）",
                ["Ndh", "Svis", "Hv"]),
        _tensor("n_vmerge_normed", "主线 merger LayerNorm 输出（view 前）", ["Svis", "Hv"],
                "activation", "use_postshuffle_norm=False：先对每 token 的 1152 维做 LayerNorm，"
                "再 view(-1, 4608)（modeling:119）", ["Svis", "Hv"]),
        _tensor("n_vmerge_units", "主线 merge 单元（view 后）", ["Svism", "Hv4"], "activation",
                "view(-1, Hv*merge²)：每 4 个相邻 token 视为一个 4608 维单元", ["Svism", "Hv4"]),
        _tensor("n_vmerge_h", "主线 fc1 输出", ["Svism", "Hv4"], "activation",
                "Linear(4608→4608)，bias=True", ["Svism", "Hv4"]),
        _tensor("n_vmerge_act", "主线 GELU 输出", ["Svism", "Hv4"], "activation",
                "nn.GELU()（merger 固定用 GELU，不是 config 的 gelu-tanh 文本）", ["Svism", "Hv4"]),
        _tensor("n_vimg", "image_embeds（写回文本序列）", ["Svism", "H"], "activation",
                "Linear(4608→4096)，bias=True（modeling:108-121）", ["Svism", "H"]),
        _tensor("n_ds_units", "DeepStack merge 单元（view 后）", ["Ndh", "Svism", "Hv4"],
                "activation",
                "3 个 tap 各自 view(-1, Hv*merge²)；postshuffle 的 norm 在 view 之后",
                ["Ndh", "Svism", "Hv4"]),
        _tensor("n_ds_normed", "DeepStack merger LayerNorm 输出（view 后）",
                ["Ndh", "Svism", "Hv4"], "activation",
                "use_postshuffle_norm=True：LayerNorm(4608) 作用在 view 之后", ["Ndh", "Svism", "Hv4"]),
        _tensor("n_ds_h", "DeepStack fc1 输出", ["Ndh", "Svism", "Hv4"], "activation",
                "Linear(4608→4608)，bias=True（一份模板 ×3）", ["Ndh", "Svism", "Hv4"]),
        _tensor("n_ds_act", "DeepStack GELU 输出", ["Ndh", "Svism", "Hv4"], "activation",
                "nn.GELU()（每份 merger 一次）", ["Ndh", "Svism", "Hv4"]),
        _tensor("n_vds", "DeepStack 特征集合", ["Ndh", "Svism", "H"], "activation",
                "Linear(4608→4096)，bias=True；分别来自 block 8/16/24 的 tap",
                ["Ndh", "Svism", "H"]),
        _tensor("n_text_cond", "条件图替换后的文本嵌入", ["B", "Ptxt", "H"], "activation",
                "masked_scatter 把 image placeholder 行替换成 image_embeds（modeling:1447-1449）；"
                "t2i 时原样传递", ["B", "Ptxt", "H"]),
        _tensor("n_visual_mask", "visual_pos_masks", ["B", "Stot"], "mask",
                "条件图 placeholder 位置（含 vinputs 段的 False 补齐，modeling:1534-1544）；"
                "DeepStack 注入用", ["B", "Stot"]),
        _tensor("n_text_t", "tms token 位置替换后的文本嵌入", ["B", "Ptxt", "H"], "activation",
                "torch.where(input_ids==tms, t_emb, text_embeds)（modeling:1520-1523）",
                ["B", "Ptxt", "H"]),
        _tensor("n_patch_h1", "x_embedder.proj1 输出", ["B", "Sgen", "Pca"], "activation",
                "Linear(3072→1024)，bias=False", ["B", "Sgen", "Pca"]),
        _tensor("n_gen_embeds", "x_embedder 输出", ["B", "Sgen", "H"], "activation",
                "Linear(1024→4096)，bias=True", ["B", "Sgen", "H"]),
        _tensor("n_x", "主干输入 x", ["B", "Stot", "H"], "activation",
                "cat([文本嵌入, 生成 patch 嵌入], dim=1)（modeling:1530）", ["B", "Stot", "H"]),
        _tensor("n_cos", "MRoPE cos", ["B", "Stot", "Dh"], "activation",
                "interleaved [24,20,20]（modeling:334-354）", ["B", "Stot", "Dh"]),
        _tensor("n_sin", "MRoPE sin", ["B", "Stot", "Dh"], "activation", "同上", ["B", "Stot", "Dh"]),
        _tensor("n_ar_index", "idx_ar", ["Nar"], "index",
                "AR 行索引（flash 第 1 趟用；eager 路径改成 4D mask）", ["Nar"]),
        _tensor("n_hidden", "36 层输出", ["B", "Stot", "H"], "activation",
                "层循环体输出（每层 = decoder Block 模板的一次实例）", ["B", "Stot", "H"]),
        _tensor("n_hidden_normed", "Final RMSNorm 输出", ["B", "Stot", "H"], "activation",
                "Qwen3VLTextModel.norm（modeling:929）", ["B", "Stot", "H"]),
        _tensor("n_x_pred_all", "final_layer2 输出", ["B", "Stot", "Pin"], "activation",
                "Linear(4096→3072) 作用于全部 token（modeling:1613）", ["B", "Stot", "Pin"]),
        _tensor("n_x_pred", "x_pred（生成本次图像 patch 预测）", ["B", "Spix", "Pin"], "activation",
                "按 vinput_mask 取生成 token，再取前 tgt_image_len 个（pipeline:350-357）",
                ["B", "Spix", "Pin"]),
        # 权重
        _tensor("w_embed", "embed_tokens.weight", ["Vocab", "H"], "weight",
                "词表嵌入；tie_word_embeddings=false", ["Vocab", "H"]),
        _tensor("w_t1", "t_embedder1.mlp[0].weight", ["H", "Fq"], "weight",
                "Linear(256→4096)，bias=True", ["H", "Fq"]),
        _tensor("w_t2", "t_embedder1.mlp[2].weight", ["H", "H"], "weight",
                "Linear(4096→4096)，bias=True", ["H", "H"]),
        _tensor("w_p1", "x_embedder.proj1.weight", ["Pca", "Pin"], "weight",
                "Linear(3072→1024)，bias=False", ["Pca", "Pin"]),
        _tensor("w_p2", "x_embedder.proj2.weight", ["H", "Pca"], "weight",
                "Linear(1024→4096)，bias=True", ["H", "Pca"]),
        _tensor("w_final", "final_layer2.linear.weight", ["Pin", "H"], "weight",
                "Linear(4096→3072)，bias=True；没有 adaLN", ["Pin", "H"]),
        _tensor("w_norm_final", "language_model.norm.weight", ["H"], "weight",
                "Final RMSNorm 权重", ["H"]),
        _tensor("w_vpatch", "visual.patch_embed.proj.weight", ["Hv", "Cin", "Tp", "Pv", "Pv"],
                "weight", "Conv3d(Cin→Hv)，bias=True", ["Hv", "Cin", "Tp", "Pv", "Pv"]),
        _tensor("w_vpos", "visual.pos_embed.weight", ["Npm", "Hv"], "weight",
                "可学习位置嵌入（插值后使用）", ["Npm", "Hv"]),
        _tensor("w_vmerge_norm", "visual.merger.norm.weight", ["Hv"], "weight",
                "LayerNorm(1152) 权重（bias=True，另有 b_vmerge_norm [Hv]）", ["Hv"]),
        _tensor("b_vmerge_norm", "visual.merger.norm.bias", ["Hv"], "weight",
                "LayerNorm(1152) bias", ["Hv"]),
        _tensor("w_vmerge1", "visual.merger.linear_fc1.weight", ["Hv4", "Hv4"], "weight",
                "Linear(4608→4608)", ["Hv4", "Hv4"]),
        _tensor("b_vmerge1", "visual.merger.linear_fc1.bias", ["Hv4"], "weight",
                "bias=True", ["Hv4"]),
        _tensor("w_vmerge2", "visual.merger.linear_fc2.weight", ["H", "Hv4"], "weight",
                "Linear(4608→4096)", ["H", "Hv4"]),
        _tensor("b_vmerge2", "visual.merger.linear_fc2.bias", ["H"], "weight",
                "bias=True", ["H"]),
        _tensor("w_ds_norm", "deepstack_merger_list[i].norm.weight", ["Hv4"], "weight",
                "postshuffle 的 LayerNorm(4608) 权重（bias=True）", ["Hv4"]),
        _tensor("w_ds1", "deepstack_merger_list[i].linear_fc1.weight", ["Hv4", "Hv4"], "weight",
                "3 个 DeepStack merger 的同构 fc1（一份模板 ×3）", ["Hv4", "Hv4"]),
        _tensor("b_ds1", "deepstack_merger_list[i].linear_fc1.bias", ["Hv4"], "weight",
                "bias=True", ["Hv4"]),
        _tensor("w_ds2", "deepstack_merger_list[i].linear_fc2.weight", ["H", "Hv4"], "weight",
                "同构 fc2（4608→4096）", ["H", "Hv4"]),
        _tensor("b_ds2", "deepstack_merger_list[i].linear_fc2.bias", ["H"], "weight",
                "bias=True", ["H"]),
    ]


def _operators() -> list[dict]:
    return [
        _op("net_text_embed", "Qwen3VLTextModel.embed_tokens", "self.embed_tokens(input_ids)",
            "embedding", ["n_input_ids"], ["n_text_embeds"], weights=["w_embed"],
            parameter_refs=["B", "Ptxt", "H", "Vocab"],
            source_refs=[_src("1429", "inputs_embeds = self.get_input_embeddings()(input_ids)")],
            note="文本 token 查表；不含位置嵌入（位置在 attention 内以 RoPE 施加）"),
        _op("net_t_freq", "TimestepEmbedder.timestep_embedding",
            "timestep_embedding(t*1000, 256)", "sinusoidal-embedding",
            ["n_timestep"], ["n_t_freq"], parameter_refs=["B", "Fq"],
            source_refs=[_src("998-1020", "torch.cos/sin(t*freqs)，dim=256")],
            note="t 由外部调度给出（pipeline:373）；这里只做正弦编码"),
        _op("net_t_lin1", "TimestepEmbedder.mlp[0]", "self.mlp[0](t_freq)", "linear",
            ["n_t_freq"], ["n_t_h1"], weights=["w_t1"], parameter_refs=["B", "H", "Fq"],
            source_refs=[_src("989-993", "Linear(256, hidden) → SiLU → Linear(hidden, hidden)")],
            note="Linear(256→4096)，bias=True"),
        _op("net_t_silu", "TimestepEmbedder.mlp[1]", "nn.SiLU()", "silu",
            ["n_t_h1"], ["n_t_act"], parameter_refs=["B", "H"],
            source_refs=[_src("991", "nn.SiLU()")], note="单层激活"),
        _op("net_t_lin2", "TimestepEmbedder.mlp[2]", "self.mlp[2](act)", "linear",
            ["n_t_act"], ["n_t_emb"], weights=["w_t2"], parameter_refs=["B", "H"],
            source_refs=[_src("992", "Linear(hidden, hidden)")], note="Linear(4096→4096)，bias=True"),
        _op("net_vision_patch", "Qwen3VLVisionPatchEmbed.proj", "self.proj(hidden_states)",
            "conv3d", ["n_pixel_values"], ["n_vpatches"], weights=["w_vpatch"],
            parameter_refs=["Svis", "Hv", "Cin", "Tp", "Pv"],
            source_refs=[_src("83/85-91", "Conv3d(Cin, embed_dim, kernel=[Tp,Pv,Pv], stride 同)")],
            note="条件：Kref>0。一次 3D 卷积把每个 patch 的 Cin×Tp×Pv×Pv 像素映射到 1152 维"),
        _op("net_vision_pos", "Qwen3VLVisionModel.fast_pos_embed_interpolate",
            "hidden_states + fast_pos_embed_interpolate(grid_thw)", "pos-embed-add",
            ["n_vpatches", "n_image_grid_thw"], ["n_vpatches_pos"], weights=["w_vpos"],
            parameter_refs=["Svis", "Hv", "Npm"],
            source_refs=[_src("667-742", "双线性 4 点插值 + merge 单元重排 + 相加")],
            note="条件：Kref>0。位置嵌入按图网格插值；不是把 48×48 表直接截断"),
        _op("net_vision_rope", "Qwen3VLVisionModel.rot_pos_emb",
            "rot_pos_emb(grid_thw) → (cos, sin)", "rope-2d",
            ["n_image_grid_thw"], ["n_vcos"], parameter_refs=["Svis", "Dhv"],
            source_refs=[_src("628-665/744-750", "行/列频率拼接后 cos/sin")],
            note="条件：Kref>0。2D rotary：head_dim 前一半行坐标、后一半列坐标"),
        _op("net_vision_loop", "Qwen3VLVisionModel.forward",
            "for layer_num, blk in enumerate(self.blocks)", "layer-loop",
            ["n_vpatches_pos", "n_vcos"], ["n_vtower", "n_vtaps"],
            parameter_refs=["Svis", "Hv", "Nv", "Ndh"],
            source_refs=[_src("762-776", "27 个 block 顺序执行；8/16/24 处收集 deepstack 特征")],
            note=("条件：Kref>0。循环体 = vit-tower 层组的 Block 模板（27 次同构实例）；"
                  "同一次循环顺带产出 3 个 DeepStack tap，不重复跑塔")),
        _op("net_vmerge_norm", "Qwen3VLVisionPatchMerger.norm（主线）",
            "self.norm(x)", "layernorm",
            ["n_vtower"], ["n_vmerge_normed"],
            weights=["w_vmerge_norm", "b_vmerge_norm"],
            parameter_refs=["Svis", "Hv", "EpsV"],
            source_refs=[_src("113/119", "self.norm = nn.LayerNorm(hidden if postshuffle else config.hidden_size)")],
            note=("条件：Kref>0。主线 merger 的 norm 在 view 之前，作用在每 token 的 1152 维上"
                  "（use_postshuffle_norm=False）")),
        _op("net_vmerge_view", "Qwen3VLVisionPatchMerger.norm（主线）",
            "x.view(-1, hidden_size * merge²)", "view",
            ["n_vmerge_normed"], ["n_vmerge_units"], parameter_refs=["Svis", "Svism", "Hv", "Hv4"],
            source_refs=[_src("119", "self.norm(x).view(-1, self.hidden_size)")],
            note="条件：Kref>0。把每 4 个相邻 token 拼成一个 4608 维 merge 单元（2×2 merge）"),
        _op("net_vmerge_fc1", "Qwen3VLVisionPatchMerger.linear_fc1",
            "self.linear_fc1(units)", "linear",
            ["n_vmerge_units"], ["n_vmerge_h"], weights=["w_vmerge1", "b_vmerge1"],
            parameter_refs=["Svism", "Hv4"],
            source_refs=[_src("114/120", "linear_fc1 = Linear(hidden_size, hidden_size)")],
            note="条件：Kref>0。Linear(4608→4608)，bias=True"),
        _op("net_vmerge_gelu", "Qwen3VLVisionPatchMerger.act_fn",
            "self.act_fn(fc1)", "gelu",
            ["n_vmerge_h"], ["n_vmerge_act"], parameter_refs=["Svism", "Hv4"],
            source_refs=[_src("115/120", "self.act_fn = nn.GELU()")],
            note="条件：Kref>0。merger 固定用 GELU（nn.GELU），与视觉塔 config 的 gelu_pytorch_tanh 不是同一处"),
        _op("net_vmerge_fc2", "Qwen3VLVisionPatchMerger.linear_fc2",
            "self.linear_fc2(gelu)", "linear",
            ["n_vmerge_act"], ["n_vimg"], weights=["w_vmerge2", "b_vmerge2"],
            parameter_refs=["Svism", "H"],
            source_refs=[_src("116/120", "linear_fc2 = Linear(hidden_size, out_hidden_size)")],
            note="条件：Kref>0。Linear(4608→4096)，bias=True；输出 [Svism,H] 写回文本序列"),
        _op("net_ds_view", "Qwen3VLVisionPatchMerger.norm（deepstack）",
            "x.view(-1, hidden_size * merge²)", "view",
            ["n_vtaps"], ["n_ds_units"], parameter_refs=["Ndh", "Svis", "Svism", "Hv4"],
            source_refs=[_src("615-624", "deepstack_merger_list 的 3 个 postshuffle merger")],
            note=("条件：Kref>0。DeepStack 的 3 个同构 merger（一份模板 ×3）：先 view 成 4608 维"
                  "单元；tap 来自 block 8/16/24")),
        _op("net_ds_norm", "Qwen3VLVisionPatchMerger.norm（deepstack）",
            "self.norm(units)", "layernorm",
            ["n_ds_units"], ["n_ds_normed"], weights=["w_ds_norm"],
            parameter_refs=["Ndh", "Svism", "Hv4", "EpsV"],
            source_refs=[_src("113/119", "use_postshuffle_norm=True → LayerNorm(merge²*hidden) 在 view 之后")],
            note="条件：Kref>0。与主线相反：postshuffle 的 LayerNorm 作用在 4608 维（view 之后），×3"),
        _op("net_ds_fc1", "Qwen3VLVisionPatchMerger.linear_fc1（deepstack）",
            "self.linear_fc1(normed)", "linear",
            ["n_ds_normed"], ["n_ds_h"], weights=["w_ds1", "b_ds1"],
            parameter_refs=["Ndh", "Svism", "Hv4"],
            source_refs=[_src("114/120", "linear_fc1 = Linear(hidden_size, hidden_size)")],
            note="条件：Kref>0。Linear(4608→4608)，bias=True；3 个 merger 各执行一次"),
        _op("net_ds_gelu", "Qwen3VLVisionPatchMerger.act_fn（deepstack）",
            "self.act_fn(fc1)", "gelu",
            ["n_ds_h"], ["n_ds_act"], parameter_refs=["Ndh", "Svism", "Hv4"],
            source_refs=[_src("115/120", "self.act_fn = nn.GELU()")],
            note="条件：Kref>0。每个 DeepStack merger 一次 GELU"),
        _op("net_ds_fc2", "Qwen3VLVisionPatchMerger.linear_fc2（deepstack）",
            "self.linear_fc2(gelu)", "linear",
            ["n_ds_act"], ["n_vds"], weights=["w_ds2", "b_ds2"],
            parameter_refs=["Ndh", "Svism", "H"],
            source_refs=[_src("116/120", "linear_fc2 = Linear(hidden_size, out_hidden_size)")],
            note=("条件：Kref>0。Linear(4608→4096)，bias=True；3 个特征分别对应 block 8/16/24 的 tap，"
                  "注入 decoder 前 3 层")),
        _op("net_cond_scatter", "Qwen3VLModel.get_placeholder_mask / masked_scatter",
            "inputs_embeds.masked_scatter(image_mask, image_embeds)", "masked-scatter",
            ["n_text_embeds", "n_vimg", "n_input_ids"], ["n_text_cond", "n_visual_mask"],
            parameter_refs=["B", "Ptxt", "H", "Svism"],
            source_refs=[_src("1447-1449", "image_mask = (input_ids == image_token_id)"),
                         _src("1505-1508", "visual_pos_masks = image_mask")],
            note=("条件：Kref>0 时替换 image placeholder 行；t2i 时 pixel_values=None，"
                  "本算子原样传递（等价恒等）")),
        _op("net_tms_substitute", "Qwen3VLModel._forward_generation",
            "torch.where(input_ids == tms, t_emb, inputs_embeds)", "where",
            ["n_text_cond", "n_t_emb", "n_input_ids"], ["n_text_t"],
            parameter_refs=["B", "Ptxt", "H", "Tms"],
            source_refs=[_src("1514-1523", "tms_mask_3d = (input_ids == tms_token_id)")],
            note=("时间步就是这样进入网络的：替换 1 个 tms token 的嵌入，"
                  "不是 adaLN、也不加到每个 token")),
        _op("net_patch_lin1", "BottleneckPatchEmbed.proj1", "self.proj1(x)", "linear",
            ["n_vinputs"], ["n_patch_h1"], weights=["w_p1"],
            parameter_refs=["B", "Sgen", "Pin", "Pca"],
            source_refs=[_src("950/961-963", "proj1 = Linear(patch_size²*in_chans, pca_dim, bias=False)")],
            note="Linear(3072→1024)，bias=False（低秩瓶颈）"),
        _op("net_patch_lin2", "BottleneckPatchEmbed.proj2", "self.proj2(h1)", "linear",
            ["n_patch_h1"], ["n_gen_embeds"], weights=["w_p2"],
            parameter_refs=["B", "Sgen", "Pca", "H"],
            source_refs=[_src("951/961-963", "proj2 = Linear(pca_dim, embed_dim, bias=True)")],
            note="Linear(1024→4096)，bias=True；目标噪声 patch 与参考图 patch 共用本嵌入"),
        _op("net_concat", "Qwen3VLModel._forward_generation",
            "torch.cat([inputs_embeds, vinputs_embedded], dim=1)", "concat",
            ["n_text_t", "n_gen_embeds"], ["n_x"], parameter_refs=["B", "Ptxt", "Sgen", "Stot", "H"],
            source_refs=[_src("1525-1532", "inputs_embeds = torch.cat([...], dim=1)")],
            note="文本在前、生成 token 在后；这个顺序是两趟注意力等价性的前提"),
        _op("net_rope", "Qwen3VLTextRotaryEmbedding.forward",
            "self.rotary_emb(hidden_states, position_ids)", "rope",
            ["n_position_ids"], ["n_cos", "n_sin"], parameter_refs=["B", "Stot", "Dh", "Nh"],
            source_refs=[_src("334-354", "apply_interleaved_mrope + cat(freqs,freqs) + cos/sin"),
                         _src("1284-1289", "flash 路径在层循环外算一次")],
            note="MRoPE：三轴位置按 [24,20,20] 交织；目标图像 patch 的位置从 4096 起（utils）"),
        _op("net_attn_partition", "Qwen3VLModel._run_decoder_flash",
            "is_gen = token_types[0].bool(); idx_ar = nonzero(~is_gen)", "index-select",
            ["n_token_types"], ["n_ar_index"], parameter_refs=["B", "Stot", "Nar", "Ngt"],
            source_refs=[_src("1291-1293", "idx_ar = torch.nonzero(~is_gen)"),
                         _src("1577-1589", "eager：生成行 mask 置 0，其余上三角 -inf")],
            note=("AR/生成 行的划分只算一次，层内共用；flash 与 eager 两种实现只是同一语义的"
                  "不同执行形式")),
        _op("net_layer_loop", "Qwen3VLTextModel.layers（Qwen3VLModel._run_decoder_flash）",
            "for layer_idx, decoder_layer in enumerate(text_model.layers)", "layer-loop",
            ["n_x", "n_cos", "n_sin", "n_ar_index", "n_visual_mask", "n_vds"], ["n_hidden"],
            parameter_refs=["Nlay", "Ndh", "B", "Stot", "H"],
            source_refs=[_src("1368-1378", "27 行内层循环 + DeepStack 注入"),
                         _src("1603", "use_cache=False（没有 KV cache）")],
            note=("循环体 = uit-decoder 层组的 Block 模板（36 次同构实例），层顺序见 layer_order；"
                  "前 3 层额外消费 DeepStack 特征，其余层原样传递")),
        _op("net_final_norm", "Qwen3VLTextModel.norm", "self.norm(hidden_states)", "rmsnorm",
            ["n_hidden"], ["n_hidden_normed"], weights=["w_norm_final"],
            parameter_refs=["B", "Stot", "H", "EpsN"],
            source_refs=[_src("929", "hidden_states = self.norm(hidden_states)")],
            note="Final RMSNorm（本模型末端有 norm，norm 后再进 final_layer2）"),
        _op("net_final_lin", "FinalLayer.linear", "self.linear(x)", "linear",
            ["n_hidden_normed"], ["n_x_pred_all"], weights=["w_final"],
            parameter_refs=["B", "Stot", "Pin", "H"],
            source_refs=[_src("968/977-979", "linear = Linear(hidden, patch²*out_ch)；adaln_input 未使用")],
            note="Linear(4096→3072)，作用于全部 token；没有 adaLN 缩放/平移"),
        _op("net_select_gen", "pipeline.forward_once / generate_image",
            "x_pred[0, vinput_mask][:tgt_image_len]", "gather-index",
            ["n_x_pred_all", "n_token_types"], ["n_x_pred"], parameter_refs=["Spix", "Sgen"],
            source_refs=[_src("1612-1613", "x_pred = self.final_layer2(hidden_states)"),
                         _src("350-357", "x_pred[0, sample['vinput_mask'][0]][:tgt_image_len]", PIPELINE)],
            note=("只取生成 token（参考图任务取前 tgt_image_len 个）。之后 v=(x_pred-z)/sigma 与 "
                  "scheduler.step 在外部，不进入本网络")),
    ]


def build_network_overview(workspace_root: str | Path, model_key: str = "hidream_o1_image_hf") -> dict:
    values = load_hf_model_values(workspace_root, model_key=model_key)
    groups = layer_group_assignments(values)
    parameters = _parameters(values)
    tensors = _tensors()
    operators = _operators()

    group_rows = []
    for group in groups:
        row = {
            "id": group["id"],
            "label": group["label"],
            "attention_type": group["attention_type"],
            "moe_type": group["moe_type"],
            "count": len(group["layer_indices"]),
            "layer_indices": list(group["layer_indices"]),
            "representative_layer_index": group["representative_layer_index"],
            "state_type": "decoder" if group["id"] == "uit-decoder" else "vision",
            "execution": ("always" if group["id"] == "uit-decoder"
                          else "conditional: Kref>0（参考图/编辑/IP）"),
        }
        group_rows.append(row)

    layer_order = [
        {"index": index, "group_id": "uit-decoder",
         "attention_type": "hybrid_causal_full", "moe_type": "dense_swiglu"}
        for index in range(values.num_hidden_layers)
    ]

    detail_groups = []
    for group in group_rows:
        detail_groups.append({
            "id": group["id"], "label": group["label"], "count": group["count"],
            "layer_indices": list(group["layer_indices"]),
            "representative_layer_index": group["representative_layer_index"],
            "anchors": {
                "group": f"/group/{group['id']}",
                "block": f"/group/{group['id']}/block",
                "operators": f"/group/{group['id']}/ops",
                "tensors": f"/group/{group['id']}/ops",
            },
        })

    path = {
        "id": "generation",
        "title": "HiDream-O1-Image 单次 forward：文本 + 像素 patch + 时间步 → x_pred",
        "parameters": parameters,
        "tensors": tensors,
        "operators": operators,
        "derivations": _derivations(),
        "cache_notes": [
            "单次 forward 没有 KV cache：_forward_generation 用 use_cache=False 调 language_model，"
            "_run_decoder_flash 直接循环 text_model.layers（modeling:1603/1368-1378）。",
            "不存在 past_key_values、cache_position 或逐层写回；跨采样步的 z 与 cond_image_embeds "
            "是外部循环的中间量，不是网络缓存。",
            "参考图任务在同一次 forward 内把条件图嵌入算一次（cond_image_embeds 可在后续步复用，"
            "属于外部循环的优化，不改变单次 forward 结构）。",
        ],
        "coverage_notes": [
            "已表达：文本 embedding → 时间步正弦 + 2 层 MLP → tms 位置替换；可选视觉塔"
            "（patch Conv3d → 位置插值 → 2D rotary → 27 blocks → 主线 merger + 3 个 DeepStack "
            "merger → masked_scatter）；像素 patch 两段低秩投影；文本+生成 token 拼接；MRoPE；"
            "AR/生成行划分；36 层循环；Final RMSNorm；final_layer2；按生成 token 取 x_pred。",
            "终点是 x_pred，不是 logits：生成路径不调用 lm_head（modeling:1906-1912）。"
            "lm_head 权重仍存在于模型里（自回归分支使用），不属于本单次生成 forward。",
            "没有 VAE：像素 patch 直接进网络、直接由 final_layer2 预测像素块；"
            "没有独立文本编码器：文本 token 与像素 token 在同一个 decoder 里。",
            "32 个 Q 头 / 8 个 KV 头 / head_dim 128；MRoPE interleaved [24,20,20]；"
            "SwiGLU 12288；每层两个 RMSNorm 站点 + 两个标量残差。",
            "未覆盖（明确列出）：50/28 步采样循环、CFG 的 cond/uncond 两次调用与 v 组合、"
            "scheduler（FlowUniPC/FlowMatch/Flash）更新、初始噪声与逐步噪声注入、"
            "最终像素反拼与 PIL 输出；loss/反向/FSDP/梯度检查点；prompt agent（Gemma-4-31B）；"
            "layout/skeleton 条件图预处理；bfloat16 数值等价与设备 kernel。",
            "数据相关量（参考图 patch 数、条件图 token 数、哪些行是条件图 placeholder）保留符号；"
            "不从权重或 prompt 内容推断。",
        ],
    }

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
            "analysis_path": "eager",
            "single_forward": True,
            "entry_class": "Qwen3VLForConditionalGeneration",
            "main_module_class": "Qwen3VLModel",
            "decoder_layer_class": "Qwen3VLTextDecoderLayer",
            "vision_block_class": "Qwen3VLVisionBlock",
            "hidden_size": values.hidden_size,
            "vocab_size": values.vocab_size,
            "layer_count": values.num_hidden_layers,
            "vision_block_count": values.vision_depth,
            "pixel_patch_size": values.pixel_patch_size,
            "pixel_pca_dim": values.pixel_pca_dim,
            "timestep_token_num": values.timestep_token_num,
            "tms_token_id": values.tms_token_id,
            "deepstack_visual_indexes": list(values.deepstack_visual_indexes),
            "main_path": [op["id"] for op in operators],
            "external_boundary": {
                "sampling_steps_full": values.external_steps_full,
                "sampling_steps_dev": values.external_steps_dev,
                "guidance_full": values.external_guidance_full,
                "scheduler": ["FlowUniPCMultistepScheduler", "FlowMatchEulerDiscreteScheduler",
                              "FlashFlowMatchEulerDiscreteScheduler"],
                "note": "采样循环/scheduler/CFG/噪声注入/像素反拼在外部，不分析、不乘步数",
            },
            "notes": [
                "整网只覆盖一次 forward；不把 50/28 步采样乘进结构或成本。",
                "视觉塔（27 blocks）是条件执行的分支：t2i 不跑，Kref>0 才跑。",
                "36 层 decoder 是唯一总是执行的重复层组。",
            ],
        },
        "path": path,
        "groups": group_rows,
        "layer_order": layer_order,
        "detail": {"href": "structure.html", "groups": detail_groups},
    }
