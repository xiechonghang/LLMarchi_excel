"""HiDream-O1-Image 官方 HF 实现的有效配置解释（静态只读，不实例化/不运行模型）。

身份（2026-10-29 锁定）：
- 官方模型仓库 ``HiDream-ai/HiDream-O1-Image``（Hugging Face，
  ``pipeline_tag=image-text-to-image``），revision
  ``0b0901d99f200389e138c61946af1185f5f49a13``（lastModified 2026-06-22）；
  根 ``config.json`` 的 ``model_type="qwen3_vl"``、
  ``architectures=["Qwen3VLForConditionalGeneration"]``。
- 官方实现仓库 ``HiDream-ai/HiDream-O1-Image``（GitHub），固定 commit
  ``2c2d29ff729e48f33e41f49edfdbd81d5ac103b4``（2026-06-22），本票分析的文件：
  ``models/qwen3_vl_transformers.py``（入口类与单次 forward）、
  ``models/pipeline.py``（输入构造/边界）、``models/utils.py``（MRoPE 位置构造）、
  ``inference.py``（外部采样边界）。
- 本文件只读根 ``config.json`` 与上述官方源码，静态解释实例化后实际生效的值；
  不导入 torch/transformers、不实例化模型、不运行网络、不下载任何权重分片。

**这是 HiDream-O1-Image（8B，Pixel-level Unified Transformer / UiT），不是
HiDream-I1 / HiDream-O1-Image-Dev / 任何 DiT+VAE 或双流 MMDiT。** 模型卡明确
“no external VAEs or disjoint text encoders”：像素 patch、文本 token 与任务条件
进入同一个共享 token 空间，由同一个 decoder 主干在一次 forward 内处理。

关键解释事实（每条以源码行区间为据；行号按固定 commit 2c2d29ff…）：

- 入口是 ``Qwen3VLForConditionalGeneration``（transformers Qwen3-VL 的仓库内
  改写副本），但生成路径不是自回归 next-token：``Qwen3VLModel.forward`` 检测到
  ``vinputs`` 就转到 ``_forward_generation``（modeling:1650-1660），返回
  ``x_pred``（每 token 的 32×32×3 像素预测），**不经过 lm_head**（modeling:1906-1912）。
- 文本侧：``Qwen3VLTextModel`` 36 层 decoder，hidden 4096、32 Q 头 / 8 KV 头、
  head_dim 128、SwiGLU 12288、RMSNorm eps 1e-6、q_norm/k_norm 逐头 RMSNorm、
  MRoPE interleaved ``mrope_section=[24,20,20]``、theta 5e6、vocab 151936
  （config.json text_config；modeling:293-354/405-477/480-539/787-936）。
- 视觉侧：``Qwen3VLVisionModel`` 27 个 ViT block（hidden 1152、16 头、
  head_dim 72、MLP 4304、LayerNorm、2D rotary、非因果注意力、patch 16×16、
  temporal 2、spatial merge 2），``deepstack_visual_indexes=[8,16,24]`` 在第 8/16/24
  个 block 后各接一个 DeepStack merger，主线再有 1 个 merger（modeling:589-778）。
  该塔**只在参考图/编辑/IP 任务（Kref>0）执行**：t2i 路径 ``pixel_values is None``
  且 inference 在 ``torch.no_grad`` 下，不跑 dummy 视觉（modeling:1440-1479）。
- 像素 patch 嵌入：``BottleneckPatchEmbed`` = Linear(32*32*3=3072 → 1024) →
  Linear(1024 → 4096)，pca_dim = hidden//4（modeling:947-963/1039-1045）。
- 时间步嵌入：``TimestepEmbedder`` = 正弦 256 维 → Linear(256→4096) → SiLU →
  Linear(4096→4096)；t 由外部调度给出，网络内只做 ``t*1000`` 的嵌入
  （modeling:983-1022/1514-1523）。``t_embedder2 = None``、``FinalLayer`` 的
  ``adaln_input`` 参数未使用 —— **没有 adaLN / 时间步调制 MLP**（modeling:977-979/1047-1049）。
- 时间步注入方式：把它**写进 tms token 的位置**（``tms_token_id=151673`` 硬编码，
  ``torch.where(input_ids == tms, t_emb, inputs_embeds)``；modeling:1050/1520-1523），
  不是作为一个额外向量加到每个 token 上。
- 序列：文本 token（chat template + boi + 1×tms）在后端接上 ``vinputs`` 的 patch
  嵌入（目标图像 patch + 可选参考图 patch），得到 ``[B, S_tot, H]``；
  ``token_types`` 0=AR 文本、>0=生成 token（tms 也是 >0），``vinput_mask``
  只覆盖生成 token（pipeline:62-77/294-302；modeling:1525-1556）。
- 注意力语义（同一层内，不是两种层型）：AR 文本行只对 AR 文本做因果注意力；
  生成 token 行对全部 token 做双向注意力。推理走 flash 两趟：第 1 趟只在 AR
  子集上 causal，第 2 趟在全序列上 non-causal，再用 ``index_copy`` 把 AR 行的
  结果换回（modeling:1331-1345）；``use_flash_attn=False`` 时用等价的 4D mask
  （生成行置 0、其余上三角 -inf；modeling:1577-1589）。文本在前、生成 token 在后的
  布局使两种实现语义一致。
- 输出：``final_layer2 = Linear(4096 → 32*32*3=3072)`` 作用于全部 token，再按
  ``vinput_mask`` 取生成 token（参考图任务取前 ``tgt_image_len`` 个）
  （modeling:1612-1613；pipeline:350-357）。
- 缓存：生成 forward **不使用 past_key_values**（``use_cache=False``，
  modeling:1603；``_run_decoder_flash`` 直接循环 ``text_model.layers``，
  modeling:1368-1378）。单次 forward 没有 KV cache 读写，不能虚构。
- 外部边界（不分析）：50 步（full）/28 步（dev）的采样循环、scheduler
  （FlowUniPCMultistep / FlowMatchEuler / FlashFlowMatch）、CFG 的 cond/uncond
  两次调用与 v 组合、noise 注入与 ``v=(x_pred-z)/sigma``、最终像素反拼
  （pipeline:313-435；inference.py:79-119）。这些不进入单次网络结构。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "HfConfigError",
    "HfModelValues",
    "SOURCE_FILES",
    "MODEL_REPOSITORY",
    "ANALYZED_SOURCE_FINGERPRINTS",
    "HIDREAM_BIND_KEYS",
    "HIDREAM_SCENARIOS",
    "DEFAULT_SCENARIO_ID",
    "verify_source_fingerprints",
    "load_hf_model_values",
    "layer_group_assignments",
    "decoder_layer_indices",
    "vision_block_indices",
    "effective_parameter_records",
]

# 场景只绑定「实际可填的输入规模」，不是长度组合枚举。图像尺寸在
# pipeline 里会吸附到预定义分辨率（utils:190-200），本表按 32 的倍数举例。
HIDREAM_SCENARIOS = (
    {
        "id": "t2i-1024",
        "label": "文生图 1024×1024（示例 prompt 128 token，无参考图）",
        "Himg": 1024,
        "Wimg": 1024,
        "Ptxt": 128,
        "Kref": 0,
        "Sref": 0,
        "Svis": 0,
        "S_i": 0,
        "note": ("示例取值：图像 1024×1024 → 1024 个 32×32 patch token；"
                 "Kref=0 时视觉塔不执行、参考图 patch 为 0。"),
    },
    {
        "id": "t2i-2048",
        "label": "文生图 2048×2048（示例 prompt 128 token，无参考图）",
        "Himg": 2048,
        "Wimg": 2048,
        "Ptxt": 128,
        "Kref": 0,
        "Sref": 0,
        "Svis": 0,
        "S_i": 0,
        "note": ("图像 2048×2048 → 4096 个 patch token；注意力矩阵 [B,Nh,S_tot,S_tot] "
                 "随 S_tot 平方增长，本表只列规模，不算耗时。"),
    },
    {
        "id": "edit-1ref",
        "label": "单参考图编辑（Kref=1，参考图 patch 数与 VLM 条件 token 数待填）",
        "Himg": 1024,
        "Wimg": 1024,
        "Ptxt": 128,
        "Kref": 1,
        "Sref": None,
        "Svis": None,
        "S_i": None,
        "note": ("Kref=1 时视觉塔（27 个 ViT block + DeepStack merger）执行，参考图同时以 "
                 "32×32 patch 进入共享 token 空间；参考图尺寸/长宽比决定 patch 数，"
                 "未给定就如实未知。"),
    },
    {
        "id": "edit-2ref-equal",
        "label": "两张等长参考图（Kref=2，各 576 patch token）",
        "Himg": 1024,
        "Wimg": 1024,
        "Ptxt": 128,
        "Kref": 2,
        "Sref": 2048,
        "Svis": 1152,
        "S_i": 576,
        "note": ("每图 576 个条件 token，两图等长：逐段 attention 总元素 = Nhv·Σ_i S_i² "
                 "= 16×(576²+576²)=10,616,832；不是 16×1152²。"),
    },
    {
        "id": "edit-2ref-unequal",
        "label": "两张不等长参考图（Kref=2，576+624 token）",
        "Himg": 1024,
        "Wimg": 1024,
        "Ptxt": 128,
        "Kref": 2,
        "Sref": 2048,
        "Svis": 1200,
        "S_i": 576,
        "note": ("两图 576+624，Σ S_i=1200 但不等长：Kref×S_i=1152≠Svis，单值 S_i 不能表示；"
                 "逐段/总注意力规模如实未知（需要逐图长度），不伪造总数。"),
    },
    {
        "id": "edit-4ref-nondivisible",
        "label": "四张参考图（Kref=4，每图 577 非 merge 单元）",
        "Himg": 1024,
        "Wimg": 1024,
        "Ptxt": 128,
        "Kref": 4,
        "Sref": 4096,
        "Svis": 2308,
        "S_i": 577,
        "note": ("总长 2308 能被 4 整除、4×577=2308 也满足乘法一致，但每图 577 个 patch "
                 "无法组成 2×2 merge 单元（577 % 4 ≠ 0）：逐段 shape 与 S_attn 未知；"
                 "恢复每图 576（总 2304）后重算。"),
    },
    {
        "id": "t2i-symbolic",
        "label": "符号场景（图像尺寸/prompt 均未给定）",
        "Himg": None,
        "Wimg": None,
        "Ptxt": None,
        "Kref": 0,
        "Sref": 0,
        "Svis": 0,
        "S_i": 0,
        "note": ("图像尺寸与 prompt 长度保持符号；与之相关的 patch 数、序列长度、"
                 "注意力规模如实标未知，不默认 0。"),
    },
)

# 场景可绑定的参数 id（结构 shape 仍保留符号；场景只代入取值）。
HIDREAM_BIND_KEYS = ("Himg", "Wimg", "Ptxt", "Kref", "Sref", "Svis", "S_i")

DEFAULT_SCENARIO_ID = "t2i-1024"

# 官方模型仓库身份（HF API /api/models/HiDream-ai/HiDream-O1-Image 于 2026-10-29 返回）。
MODEL_REPOSITORY = {
    "repo_id": "HiDream-ai/HiDream-O1-Image",
    "revision": "0b0901d99f200389e138c61946af1185f5f49a13",
    "last_modified": "2026-06-22T07:53:39.000Z",
    "pipeline_tag": "image-text-to-image",
    "library_name": "transformers",
    "model_type": "qwen3_vl",
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "text_model_type": "qwen3_vl_text",
    "checkpoint_transformers_version": "4.57.0.dev0",
    "code_repo": "HiDream-ai/HiDream-O1-Image（GitHub）",
    "code_revision": "2c2d29ff729e48f33e41f49edfdbd81d5ac103b4",
    "note": (
        "配置与模型卡按上述 HF revision 取得；官方实现按上述 GitHub commit 取得。"
        "只取 config.json（1,474 B）、README.md（23,917 B）与 4 个小型 .py 源码，"
        "不下载 8 个权重分片（合计约 33.2 GB）中的任何一个。"
    ),
}

# 所选生成 forward 实际读取的定义闭包 + 外部边界说明（相对工作区根）。
SOURCE_FILES = {
    "config.json": "models/HiDream/HiDream-O1-Image/config.json",
    "README.md": "models/HiDream/HiDream-O1-Image/README.md",
    "qwen3_vl_transformers.py": "models/HiDream/HiDream-O1-Image/qwen3_vl_transformers.py",
    "pipeline.py": "models/HiDream/HiDream-O1-Image/pipeline.py",
    "utils.py": "models/HiDream/HiDream-O1-Image/utils.py",
    "inference.py": "models/HiDream/HiDream-O1-Image/inference.py",
}

# 已分析来源指纹（SHA256）。来源漂移即具名拒绝，要求重新分析后更新指纹。
ANALYZED_SOURCE_FINGERPRINTS = {
    "config.json": "5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661",
    "README.md": "901f12f248a2e43a7ac6092d2a6de9f68f444c83e8ba4a53d55eb3069240e7b7",
    "qwen3_vl_transformers.py": "5614a339654a77e1af7c455a07e3665bfcf60da476b37097aa123b3c051ab17b",
    "pipeline.py": "68156223cfa6cf86e52776cf8480cd5446d311dd3313a8104adc0355bf0460a6",
    "utils.py": "32faef1e2b2fe4c5e75fc12dbefb89d708dfccb1cd390a2a59d2f9011c319a7e",
    "inference.py": "df1280dc648193fa007780b4618a89cda3004758ba99d0e81dfbe28f575b4031",
}


class HfConfigError(ValueError):
    """根配置/来源与锁定事实不符（具名拒绝，不用近似模型继续）。"""


@dataclass(frozen=True)
class HfModelValues:
    """HiDream-O1-Image 单次生成 forward 的有效配置值（静态解释，非运行时探值）。"""

    # 身份
    model_key: str
    display_name: str
    implementation: str
    source_revision: str
    config_path: str
    model_revision: str
    code_repository: str
    checkpoint_transformers_version: str
    # 文本主干（config.text_config）
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    hidden_act: str
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float
    mrope_section: tuple
    mrope_interleaved: bool
    attention_bias: bool
    attention_dropout: float
    tie_word_embeddings: bool
    dtype: str
    use_cache_config: bool
    # 视觉塔（config.vision_config）；只在参考图任务执行
    vision_depth: int
    vision_hidden_size: int
    vision_heads: int
    vision_head_dim: int
    vision_intermediate_size: int
    vision_patch_size: int
    vision_temporal_patch_size: int
    spatial_merge_size: int
    vision_out_hidden_size: int
    num_position_embeddings: int
    deepstack_visual_indexes: tuple
    vision_hidden_act: str
    in_channels: int
    # 像素 patch 主干（models/qwen3_vl_transformers.py 的硬编码常量）
    pixel_patch_size: int
    pixel_in_channels: int
    pixel_pca_dim: int
    pixel_in_dim: int
    pixel_out_dim: int
    # 时间步嵌入 / 特殊 token
    timestep_frequency_embedding_size: int
    timestep_token_num: int
    tms_token_id: int
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    # 外部采样边界（仅记录，不展开）
    external_steps_full: int
    external_steps_dev: int
    external_guidance_full: float
    external_shift_full: float


def _load_root_config(workspace_root: Path) -> dict:
    path = workspace_root / SOURCE_FILES["config.json"]
    if not path.is_file():
        raise HfConfigError(f"根配置缺失: {SOURCE_FILES['config.json']}（不下载、不伪造）")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HfConfigError(f"根配置无法读取/解析: {exc}") from exc
    if not isinstance(raw, dict):
        raise HfConfigError("根配置必须是 JSON 对象")
    return raw


def verify_source_fingerprints(workspace_root: str | Path) -> dict:
    """复算 6 项来源 SHA256；缺失或漂移时具名拒绝。"""
    import hashlib

    workspace_root = Path(workspace_root)
    issues: list[str] = []
    actual: dict[str, str] = {}
    for key, relpath in SOURCE_FILES.items():
        path = workspace_root / relpath
        if not path.is_file():
            issues.append(f"缺失 {key} ({relpath})")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        actual[key] = digest
        expected = ANALYZED_SOURCE_FINGERPRINTS[key]
        if digest != expected:
            issues.append(f"{key} 漂移: 期望 {expected[:16]}… 实得 {digest[:16]}…")
    if issues:
        raise HfConfigError(
            "HiDream-O1-Image 来源与已分析版本不符（需重新分析后更新指纹，"
            "不沿用旧结论）: " + "; ".join(issues))
    return actual


def load_hf_model_values(
    workspace_root: str | Path,
    model_key: str = "hidream_o1_image_hf",
) -> HfModelValues:
    """读取根 config.json 并静态解释为 HiDream-O1-Image 有效配置；失配具名拒绝。"""
    workspace_root = Path(workspace_root)
    raw = _load_root_config(workspace_root)
    if raw.get("model_type") != "qwen3_vl":
        raise HfConfigError(
            f"根配置 model_type 必须是 qwen3_vl（HiDream-O1-Image 的已锁定事实），"
            f"实得 {raw.get('model_type')!r}")
    architectures = raw.get("architectures")
    if architectures != ["Qwen3VLForConditionalGeneration"]:
        raise HfConfigError(
            "根配置 architectures 必须是 ['Qwen3VLForConditionalGeneration']，"
            f"实得 {architectures!r}；不以 HiDream-I1/Dev 或其它架构代替")
    text = raw.get("text_config")
    vision = raw.get("vision_config")
    if not isinstance(text, dict) or not isinstance(vision, dict):
        raise HfConfigError("根配置缺少 text_config / vision_config 对象")

    def need(obj: dict, name: str, where: str):
        value = obj.get(name)
        if value is None:
            raise HfConfigError(f"{where} 缺少必要字段 {name!r}")
        return value

    if need(text, "model_type", "text_config") != "qwen3_vl_text":
        raise HfConfigError("text_config.model_type 必须是 qwen3_vl_text")

    hidden_size = int(need(text, "hidden_size", "text_config"))
    num_layers = int(need(text, "num_hidden_layers", "text_config"))
    num_heads = int(need(text, "num_attention_heads", "text_config"))
    num_kv = int(need(text, "num_key_value_heads", "text_config"))
    head_dim = int(need(text, "head_dim", "text_config"))
    intermediate = int(need(text, "intermediate_size", "text_config"))
    eps = float(need(text, "rms_norm_eps", "text_config"))
    vocab = int(need(text, "vocab_size", "text_config"))
    max_pos = int(need(text, "max_position_embeddings", "text_config"))
    rope_theta = float(need(text, "rope_theta", "text_config"))
    rope_scaling = need(text, "rope_scaling", "text_config")
    if not isinstance(rope_scaling, dict):
        raise HfConfigError("text_config.rope_scaling 必须是对象（MRoPE 语义来源）")
    mrope_section = tuple(int(x) for x in rope_scaling.get("mrope_section", []))
    mrope_interleaved = bool(rope_scaling.get("mrope_interleaved", False))
    if (hidden_size, num_layers, num_heads, num_kv, head_dim, intermediate) != (
            4096, 36, 32, 8, 128, 12288):
        raise HfConfigError(
            "锁定的文本主干事实是 H=4096 / 36 层 / 32 Q 头 / 8 KV 头 / head_dim 128 / "
            f"FFN 12288（实得 {hidden_size}/{num_layers}/{num_heads}/{num_kv}/"
            f"{head_dim}/{intermediate}）")
    if mrope_section != (24, 20, 20) or not mrope_interleaved:
        raise HfConfigError(
            f"锁定的 MRoPE 事实是 interleaved section (24,20,20)（实得 "
            f"{mrope_section}/{mrope_interleaved}）")

    vision_depth = int(need(vision, "depth", "vision_config"))
    v_hidden = int(need(vision, "hidden_size", "vision_config"))
    v_heads = int(need(vision, "num_heads", "vision_config"))
    v_inter = int(need(vision, "intermediate_size", "vision_config"))
    v_patch = int(need(vision, "patch_size", "vision_config"))
    v_temporal = int(need(vision, "temporal_patch_size", "vision_config"))
    merge = int(need(vision, "spatial_merge_size", "vision_config"))
    v_out = int(need(vision, "out_hidden_size", "vision_config"))
    num_pos = int(need(vision, "num_position_embeddings", "vision_config"))
    deepstack = tuple(int(x) for x in need(vision, "deepstack_visual_indexes", "vision_config"))
    in_ch = int(need(vision, "in_channels", "vision_config"))
    if (vision_depth, v_hidden, v_heads, v_inter, v_patch, v_temporal, merge, v_out) != (
            27, 1152, 16, 4304, 16, 2, 2, 4096):
        raise HfConfigError(
            "锁定的视觉塔事实是 27 blocks / Hv=1152 / 16 头 / FFN 4304 / patch 16 / "
            f"temporal 2 / merge 2 / out 4096（实得 {vision_depth}/{v_hidden}/{v_heads}/"
            f"{v_inter}/{v_patch}/{v_temporal}/{merge}/{v_out}）")
    if deepstack != (8, 16, 24):
        raise HfConfigError(f"锁定的 deepstack_visual_indexes 是 (8,16,24)（实得 {deepstack}）")
    if v_heads * (v_hidden // v_heads) != v_hidden:
        raise HfConfigError("视觉塔 head_dim 必须整除 hidden_size")

    # 像素 patch / 时间步常量来自 models/qwen3_vl_transformers.py 的硬编码
    # （modeling:1039-1050）与 pipeline.py:18；不是 config 字段。
    pixel_patch = 32
    pca_dim = hidden_size // 4
    if pca_dim != 1024:
        raise HfConfigError(f"BottleneckPatchEmbed 的 pca_dim 必须是 1024（实得 {pca_dim}）")

    image_token_id = int(need(raw, "image_token_id", "config"))
    video_token_id = int(need(raw, "video_token_id", "config"))
    vision_start = int(need(raw, "vision_start_token_id", "config"))
    if (image_token_id, video_token_id, vision_start) != (151655, 151656, 151652):
        raise HfConfigError(
            "锁定的特殊 token 事实是 image=151655 / video=151656 / vision_start=151652"
            f"（实得 {image_token_id}/{video_token_id}/{vision_start}）")

    return HfModelValues(
        model_key=model_key,
        display_name=(
            "HiDream-O1-Image（HiDream-ai/HiDream-O1-Image, Pixel-level Unified "
            "Transformer 单次生成网络）"),
        implementation="transformers",
        source_revision=MODEL_REPOSITORY["code_revision"],
        config_path=SOURCE_FILES["config.json"],
        model_revision=MODEL_REPOSITORY["revision"],
        code_repository=MODEL_REPOSITORY["code_repo"],
        checkpoint_transformers_version=str(raw.get("transformers_version", "")),
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv,
        head_dim=head_dim,
        intermediate_size=intermediate,
        hidden_act=str(need(text, "hidden_act", "text_config")),
        rms_norm_eps=eps,
        vocab_size=vocab,
        max_position_embeddings=max_pos,
        rope_theta=rope_theta,
        mrope_section=mrope_section,
        mrope_interleaved=mrope_interleaved,
        attention_bias=bool(text.get("attention_bias", False)),
        attention_dropout=float(text.get("attention_dropout", 0.0)),
        tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
        dtype=str(text.get("dtype", "")),
        use_cache_config=bool(text.get("use_cache", False)),
        vision_depth=vision_depth,
        vision_hidden_size=v_hidden,
        vision_heads=v_heads,
        vision_head_dim=v_hidden // v_heads,
        vision_intermediate_size=v_inter,
        vision_patch_size=v_patch,
        vision_temporal_patch_size=v_temporal,
        spatial_merge_size=merge,
        vision_out_hidden_size=v_out,
        num_position_embeddings=num_pos,
        deepstack_visual_indexes=deepstack,
        vision_hidden_act=str(need(vision, "hidden_act", "vision_config")),
        in_channels=in_ch,
        pixel_patch_size=pixel_patch,
        pixel_in_channels=3,
        pixel_pca_dim=pca_dim,
        pixel_in_dim=pixel_patch * pixel_patch * 3,
        pixel_out_dim=pixel_patch * pixel_patch * 3,
        timestep_frequency_embedding_size=256,
        timestep_token_num=1,
        tms_token_id=151673,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=vision_start,
        external_steps_full=50,
        external_steps_dev=28,
        external_guidance_full=5.0,
        external_shift_full=3.0,
    )


def decoder_layer_indices(values: HfModelValues) -> list[int]:
    return list(range(values.num_hidden_layers))


def vision_block_indices(values: HfModelValues) -> list[int]:
    return list(range(values.vision_depth))


def layer_group_assignments(values: HfModelValues) -> list[dict]:
    """两个真实重复层组：统一 decoder 主干 36 层、视觉塔 27 个 block（条件执行）。

    两者是**不同的重复结构**，索引空间各自从 0 开始（decoder layer 0..35 /
    ViT block 0..26），不能把 ViT block 当成 decoder 的第 36..62 层。
    """
    return [
        {
            "id": "uit-decoder",
            "label": "统一 Transformer 主干（Qwen3VLTextDecoderLayer ×36）",
            "attention_type": "hybrid_causal_full",
            "moe_type": "dense_swiglu",
            "layer_indices": decoder_layer_indices(values),
            "representative_layer_index": 0,
        },
        {
            "id": "vit-tower",
            "label": "视觉塔（Qwen3VLVisionBlock ×27，仅参考图/编辑任务执行）",
            "attention_type": "bidirectional_vision",
            "moe_type": "dense_gelu_mlp",
            "layer_indices": vision_block_indices(values),
            "representative_layer_index": 0,
        },
    ]


def effective_parameter_records(values: HfModelValues) -> list[dict]:
    """逐参数 raw 字段/静态推导规则与来源（供参数符号区联查）。"""
    modeling = SOURCE_FILES["qwen3_vl_transformers.py"]
    pipeline = SOURCE_FILES["pipeline.py"]
    utils = SOURCE_FILES["utils.py"]
    config_json = SOURCE_FILES["config.json"]

    def cfg(locator: str, note: str) -> dict:
        return {"path": config_json, "locator": locator, "kind": "config", "note": note}

    def src(path: str, locator: str, note: str) -> dict:
        return {"path": path, "locator": locator, "kind": "source", "note": note}

    def derived(locator: str, note: str) -> dict:
        return {"path": modeling, "locator": locator, "kind": "derived", "note": note}

    return [
        {
            "field": "model_identity",
            "value": {
                "repo_id": MODEL_REPOSITORY["repo_id"],
                "revision": MODEL_REPOSITORY["revision"],
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "text_config.model_type": "qwen3_vl_text",
                "code_revision": MODEL_REPOSITORY["code_revision"],
                "pipeline_tag": MODEL_REPOSITORY["pipeline_tag"],
            },
            "rule": ("HF 仓库 revision 与根配置 model_type/architectures 双条件锁定；"
                     "实现按官方 GitHub commit 固定。不以 HiDream-I1、HiDream-O1-Image-Dev "
                     "或任意 DiT/VAE 双流实现代替"),
            "source_refs": [
                cfg("model_type", "根配置字段"),
                cfg("architectures", "根配置字段"),
                cfg("text_config.model_type", "文本子配置字段"),
            ],
        },
        {
            "field": "single_forward_boundary",
            "value": {
                "entry_class": "Qwen3VLForConditionalGeneration",
                "generation_dispatch": "Qwen3VLModel.forward → _forward_generation",
                "output": "x_pred（每生成 token 32×32×3），不经 lm_head",
                "external_sampling_steps_full": values.external_steps_full,
                "external_sampling_steps_dev": values.external_steps_dev,
                "external_scheduler": "FlowUniPCMultistep / FlowMatchEuler / FlashFlowMatch",
            },
            "rule": ("只分析一次网络 forward：像素 patch + 文本 token + 时间步 → 主干 → "
                     "final_layer2 的 x_pred。采样步数/CFG 两次调用/scheduler/噪声注入/"
                     "像素反拼在外部，不乘步数、不算完整出图耗时"),
            "source_refs": [
                derived("1650-1660", "vinputs → _forward_generation 分派"),
                derived("1886-1912", "generation 路径只返回 x_pred，不调用 lm_head"),
                src(pipeline, "313-435", "外部采样循环、scheduler.step 与像素反拼"),
                src(SOURCE_FILES["inference.py"], "79-119", "50/28 步与 guidance/shift 选择"),
            ],
        },
        {
            "field": "text_backbone",
            "value": {
                "hidden_size": values.hidden_size,
                "num_hidden_layers": values.num_hidden_layers,
                "num_attention_heads": values.num_attention_heads,
                "num_key_value_heads": values.num_key_value_heads,
                "head_dim": values.head_dim,
                "intermediate_size": values.intermediate_size,
                "mrope_section": list(values.mrope_section),
                "mrope_interleaved": values.mrope_interleaved,
                "rope_theta": values.rope_theta,
                "rms_norm_eps": values.rms_norm_eps,
                "vocab_size": values.vocab_size,
            },
            "rule": ("Qwen3-VL 文本 decoder：q_norm/k_norm 逐头 RMSNorm、MRoPE 交织、"
                     "SwiGLU 12288、无 attention bias。层内没有时间步调制模块"),
            "source_refs": [
                cfg("text_config.hidden_size", "隐层维度"),
                cfg("text_config.num_hidden_layers", "层数"),
                cfg("text_config.rope_scaling", "MRoPE section + interleaved"),
                src(modeling, "405-477", "TextAttention：q/k/v/o 投影与 q_norm/k_norm"),
                src(modeling, "480-539", "TextMLP（SwiGLU）与 DecoderLayer 残差"),
            ],
        },
        {
            "field": "pixel_patch_io",
            "value": {
                "pixel_patch_size": values.pixel_patch_size,
                "pixel_in_dim": values.pixel_in_dim,
                "pixel_pca_dim": values.pixel_pca_dim,
                "pixel_out_dim": values.pixel_out_dim,
                "hidden_size": values.hidden_size,
            },
            "rule": ("BottleneckPatchEmbed：Linear(3072→1024) → Linear(1024→4096)；"
                     "final_layer2：Linear(4096→3072)。均无 adaLN"),
            "source_refs": [
                derived("1039-1045", "patch_size=32、pca_dim=hidden//4、x_embedder 构造"),
                derived("947-963", "BottleneckPatchEmbed.forward"),
                derived("965-979", "FinalLayer.forward（adaln_input 未使用）"),
            ],
        },
        {
            "field": "timestep_embedding",
            "value": {
                "frequency_embedding_size": values.timestep_frequency_embedding_size,
                "hidden_size": values.hidden_size,
                "timestep_token_num": values.timestep_token_num,
                "tms_token_id": values.tms_token_id,
            },
            "rule": ("正弦 256 维 → Linear(256→4096) → SiLU → Linear(4096→4096)；"
                     "结果写进 tms token 位置（torch.where），不是加到全部 token；"
                     "没有 adaLN，t_embedder2=None"),
            "source_refs": [
                derived("983-1022", "TimestepEmbedder（正弦 + 2 层 MLP）"),
                derived("1044-1050", "t_embedder1 / t_embedder2=None / tms_token_id"),
                src(pipeline, "26-29/41-45", "1 个 tms token 拼在 chat template 之后"),
            ],
        },
        {
            "field": "vision_tower_condition",
            "value": {
                "vision_depth": values.vision_depth,
                "vision_hidden_size": values.vision_hidden_size,
                "vision_heads": values.vision_heads,
                "vision_head_dim": values.vision_head_dim,
                "vision_intermediate_size": values.vision_intermediate_size,
                "patch_size": values.vision_patch_size,
                "spatial_merge_size": values.spatial_merge_size,
                "deepstack_visual_indexes": list(values.deepstack_visual_indexes),
                "conditional": "仅参考图/编辑/IP 任务（Kref>0）",
            },
            "rule": ("t2i 路径没有 pixel_values，视觉塔不执行；Kref>0 时视觉塔编码 384 级条件图，"
                     "输出 image_embeds 与 3 个 DeepStack 特征，后者注入 decoder 前 3 层"),
            "source_refs": [
                derived("1440-1451", "pixel_values → get_image_features → masked_scatter"),
                derived("913-919/1382-1391", "前 3 层的 DeepStack 注入条件"),
                src(pipeline, "236-243/264-271", "条件图尺寸与 image_grid_thw"),
            ],
        },
        {
            "field": "attention_semantics",
            "value": {
                "ar_rows": "只对 AR 文本 token 做因果注意力",
                "gen_rows": "对全部 token 做双向注意力",
                "flash_two_pass": "pass1 AR 子集 causal + pass2 全序列 non-causal + index_copy",
                "eager_fallback": "4D mask（生成行置 0，其余上三角 -inf）",
                "kv_cache": "不使用（use_cache=False）",
            },
            "rule": ("token_types 0=AR、>0=生成（tms 也是 >0）；文本在前、生成 token 在后的"
                     "布局使 flash 两趟与 4D mask 语义一致。单次 forward 无 KV cache"),
            "source_refs": [
                derived("1331-1345", "两趟 flash + index_copy"),
                derived("1577-1589", "eager 4D mask 构造"),
                src(pipeline, "62-77", "token_types / vinput_mask 构造"),
            ],
        },
        {
            "field": "position_ids",
            "value": {
                "mrope_section": list(values.mrope_section),
                "fix_point": 4096,
                "text_positions": "0..S_txt-1",
                "target_image_positions": "4096 + (h, w) 网格（与文本长度无关）",
            },
            "rule": ("get_rope_index_fix_point 把目标图像 patch 的位置钉在 4096 起的网格；"
                     "vision_start token 不计入位置；参考图 patch 的位置继续累加"),
            "source_refs": [
                src(utils, "246-349", "get_rope_index_fix_point（fix_point=4096）"),
                src(pipeline, "48-60", "t2i 的 image_grid_thw 与 skip_vision_start_token"),
            ],
        },
        {
            "field": "not_analyzed",
            "value": {
                "external_sampling": "50/28 步循环、CFG 两次调用、scheduler 更新、噪声注入",
                "output_decode": "z → 像素反拼与 PIL 输出",
                "training_paths": "loss/反向、梯度检查点、FSDP、dummy 视觉前向",
                "other_tasks": "prompt agent（Gemma-4-31B）、Web Demo、layout/skeleton 预处理",
                "numerics": "bfloat16/FP32 数值等价、设备 kernel、flash-attn 版本差异",
            },
            "rule": "明确不分析的内容单列，不写成已覆盖",
            "source_refs": [
                src(pipeline, "80-97/372-435", "scheduler 选择与 step/反拼"),
                derived("1452-1479", "训练期 dummy 视觉前向（inference 不执行）"),
            ],
        },
    ]
