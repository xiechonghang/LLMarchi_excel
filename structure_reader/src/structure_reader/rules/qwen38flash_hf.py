"""Qwen3.8-Flash-Next 官方 HF 实现的有效配置解释（静态只读，不实例化/不运行模型）。

身份（2026-10-29 锁定）：
- 官方模型仓库 ``Qwen/Qwen3.8-Flash-Next``（Hugging Face，
  ``pipeline_tag=image-text-to-text``），revision
  ``de4b8e4d43b917e7706784d8bb445c9af86a3540``（lastModified 2026-08-27）；
  根 ``config.json`` 的 ``model_type="qwen4_exp"``、
  ``architectures=["Qwen4ExpForConditionalGeneration"]``，
  ``text_config.model_type="qwen4_exp_text"``。
- 实现源码：工作区本地官方 transformers 检出 ``hflib/transformers``（HEAD 见
  ``source_revision``，本检出为 5.19.0.dev0），模块
  ``src/transformers/models/qwen4_exp/``（configuration/modeling）。
- 本文件只读根 ``config.json`` 与本地源码，按 ``Qwen4ExpTextConfig.__post_init__``
  与继承的 ``PreTrainedConfig`` 标准化规则推导实例化后实际生效的值；不导入
  torch/transformers、不实例化配置类、不运行模型、不下载权重。

**这是 Qwen3.8-Flash-Next（qwen4_exp），不是 Qwen3-Next/Qwen3.5/Qwen3-MoE。**
模型卡写明这是 Qwen4 架构的实验预览：48 层 = 12×(3×Gated DeltaNet → MoE →
1×Qwen Sparse Attention → MoE)；Gated DeltaNet 线性注意力、Qwen Sparse Attention
（QSA，micro-block 稀疏选择）、4 路 Gated Residual（hyper-connections）、
N-gram Embedding（PLE，bigram/trigram at layer 2）、512 专家选 10 + 1 共享专家。
不能用名称相近的 Qwen 版本的 GQA/线性注意力/MoE 结论替代。

关键解释事实（每条以源码行区间为据；行号按本地检出 536ecc00…）：

- ``layer_types``：显式 48 项，每 4 层一个 ``full_attention``（3,7,11,…,47，共
  12 层），其余 36 层 ``linear_attention``。``Qwen4ExpTextConfig.__post_init__``
  把 ``full_attention`` 规范成 ``indexed_attention``（configuration:179-183）；
  本票按规范后的 ``indexed_attention`` 表达 QSA 层。
- Gated DeltaNet（modeling:456-616）：q/k/v 合并投影（``in_proj_qkv`` 2560→10240）
  → depthwise 因果短卷积（kernel=4，conv state [B,10240,4]）→ split 成
  q/k [B,S,16,128]、v [B,S,48,128] → q/k ``repeat_interleave(3)`` 到 48 头 →
  beta=sigmoid、g=-exp(A_log)*softplus(a+dt_bias) → L2Norm(q/k)+q/√128 →
  gated delta-rule（chunk 64 / recurrent 单 token，状态 [B,48,128,128] 固定）→
  Gated RMSNorm（输出门 activation=sigmoid）→ out_proj（6144→2560）。
- Qwen Sparse Attention（modeling:810-892）：q_proj 输出 2×24×256=12288，按 head
  chunk 成 q 与 gate（各 24×256）；q/k 各做 RMSNorm(256)；RoPE 只作用于前
  ``head_dim*partial_rotary_factor=64`` 维（partial RoPE，后 192 维 nope）；
  KV 头 2 个（GQA，repeat 12）；o_proj 6144→2560；输出乘 sigmoid(gate)。
- QSA indexer（modeling:664-770）：``index_qk_proj`` 2560→(4+1)×128=640；q 4 头
  + 共享 key 1 头；q/k 各 RMSNorm(128)；q 与 pooled block key 都只对前 64 维做
  RoPE；per-token raw key 写入 indexer 缓存 [B,L,128]；每 4 个连续**可见** token
  取均值压成一个 micro-block，block key 在组首位置做 RoPE；分数 =
  relu(q·k).sum(dim=-1)/√128（4 个 indexer 头求和，不是 softmax/加权）；
  每 query 选 min(512, 完整块数) 个块 → 展开成 token → 追加不完整 tail
  （≤3 个）→ pad 到 2051（-1）→ scatter 成 True=可见的 additional mask。
- 注意力掩码：网络入口同时构建 ``create_causal_mask``（4D float additive，
  indexed_attention 用）与 ``create_recurrent_attention_mask``（2D padding，
  linear_attention 用）（modeling:1446-1460）。QSA 把 indexer 的 selected mask
  与 causal mask 相加（float）或相与（bool）（modeling:848-851）。
- Gated Residual / hyper-connections（modeling:994-1022）：每层 attn/ffn 两个站点。
  4 路流先做分组 RMSNorm（group_size=hidden_size，无乘 weight 以外的特殊归一）；
  ``input_mix_weight_down`` 10240→320、SiLU、``input_mix_weight_up`` 320→10240、
  sigmoid 得**逐元素读门** [B,S,4,H]；四路加权后取 mean 得单流进入子层；
  ``block_inject_weight`` 10240→4 得**每支路标量写门** 2×sigmoid(·/4)；
  子层输出按写门展开加到**原 4 路流**（不是标量残差）。
  网络末端 ``hyper_connection_mixer`` 用同一读门结构把 4 路加权 mean 成单流；
  **没有 Final RMSNorm**，直接进 lm_head。
- N-gram / PLE（modeling:1071-1246）：``ple_layer_ids=[2]``（one-indexed）→
  layer_idx=1（第 2 层，linear_attention）。ngram_size=3、heads_per_ngram=8 →
  16 个哈希头（bigram 8 + trigram 8），每头 160 维 → ple_embed_dim=2560。
  哈希：token×每层乘子（3 个）后 XOR，按每头素数码表大小取模，再加头偏移；
  EOS 会切断 n-gram 上下文（shift_right_ignore_eos）。PLE 用上一 token 上下文
  缓存 [B,2]（state_idx=2）+ 4-tap dilation-3 depthwise 短卷积（state 9 步，
  state_idx=1）；key_proj 2560→10240、value_proj 2560→2560，query 是 4 路流；
  gate=(key_normed*query_normed).sum(-1)/√H、abs().clamp_min(1e-6).sqrt()*sign、
  sigmoid(gate)*value；输出加回 4 路流。
- MoE（modeling:951-991）：router softmax 512 选 10，``norm_topk_prob=True``
  把 top-10 归一化；没有 correction bias、没有 routed scaling factor；专家
  gate_up [512,2×640,2560]、down [512,2560,640]；共享专家 2560→640→2560，
  另有 ``shared_expert_gate`` 2560→1 sigmoid 标量门；SwiGLU 无 clamp。
- 缓存：linear_attention 层 → LinearAttentionLayer（conv/recurrent 状态，固定）；
  indexed_attention 层 → DynamicIndexedLayer（K/V [B,2,L,256] + indexer keys
  [B,L,128]）。``number_of_conv_states=3``（PLE 开启），Gated DeltaNet 用 0、
  PLE 短卷积用 1、PLE ngram 上下文用 2；只有 layer 1 真正使用 1/2。
- MTP：``mtp_num_hidden_layers=1``、``mtp.layer_types=["full_attention"]``，但
  本地 HF 实现没有 MTP 模块，``_keys_to_ignore_on_load_unexpected=[r"^mtp.*"]``
  （modeling:1313/1589），forward 只循环 0..47 —— MTP 不在本主干。
- 视觉：``vision_config`` 27 层 ViT，本票只分析 ``text_config`` 的 token→logits
  语言主干；图像/视频 placeholder 替换、vision tower、M-RoPE 视觉网格不在范围。

版本差异（明确，不混用）：根配置记录 ``transformers_version="5.8.0.dev0"``，但
``qwen4_exp`` 在 v5.8.0 tag 中**不存在**：该模块首次提交 ``fc5c5bde8e``
（2026-08-26，版本 5.16.0.dev0），首次随 v5.17.0 发布。本票以本地官方检出
5.19.0.dev0 的实际代码为**唯一实现版本**，并记录该模块在首次提交后仍有语义
提交（例如 layer type 的 ``full_attention→indexed_attention`` 重映射、
partial mask、MRoPE 调整）。不把任何“名称相近的 Qwen”实现当作本模型实现。

``SOURCE_FILES`` 是所选 eager forward 路径实际读取的定义闭包（相对工作区根）；
``ANALYZED_SOURCE_FINGERPRINTS`` 固定已分析版本，来源漂移即具名拒绝。
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
    "LAYER_GROUP_SPECS",
    "QWEN38FLASH_SCENARIOS",
    "verify_source_fingerprints",
    "load_hf_model_values",
    "layer_group_assignments",
    "effective_parameter_records",
    "effective_layer_types",
    "qsa_layer_indices",
    "gdn_layer_indices",
    "ple_layer_indices",
]

# 少量有区别场景（不是长度组合枚举）。本模型同时有：
# - 固定大小 Gated DeltaNet 状态（不随 kvcachelen 增长）；
# - 随历史线性增长的 QSA K/V + indexer key 缓存；
# - QSA 预算 2048 token / 512 micro-block（4 token 一块）+ 不完整 tail。
QWEN38FLASH_SCENARIOS = (
    {
        "id": "prefill-empty",
        "label": "空缓存 Prefill（示例 seqlen=8）",
        "seqlen": 8,
        "kvcachelen": 0,
        "note": "kvcachelen=0 即空缓存起步；seqlen=8 只是示例取值，结构 shape 仍保留符号 seqlen。"
                "Lkv=8 时只有 2 个完整 micro-block，QSA 全选、tail 为空。",
    },
    {
        "id": "prefill-symbolic",
        "label": "空缓存 Prefill（符号 seqlen，未给数值）",
        "seqlen": None,
        "kvcachelen": 0,
        "note": "seqlen 保持符号：与 seqlen 有关的派生量如实标为未知，不默认成 0。",
    },
    {
        "id": "decode-nonzero",
        "label": "非零历史单 token Decode（kvcachelen=4096）",
        "seqlen": 1,
        "kvcachelen": 4096,
        "note": "kvcachelen=4096 是示例逻辑历史长度；QSA K/V 与 indexer key 缓存都线性保留，"
                "Lkv=Lidx=4097；单 token 第 0 个 query 的可见长度 V_q=4097。",
    },
    {
        "id": "decode-long",
        "label": "长历史单 token Decode（kvcachelen=131072）",
        "seqlen": 1,
        "kvcachelen": 131072,
        "note": "历史远超 QSA 预算 2048 token：每 query 只选 512 个完整 micro-block"
                "（2048 token）+ 不完整 tail；K/V 缓存本身仍保留全部 Lkv。",
    },
    {
        "id": "decode-multi",
        "label": "非零历史多 token Decode（seqlen=4，kvcachelen=4092）",
        "seqlen": 4,
        "kvcachelen": 4092,
        "note": "多 token Decode：检查统一结构按 seqlen 表达，而不是固定 +1/T=1。"
                "Lkv=4096 恰为 compress_ratio 的整数倍，最后一个 query 的 tail 为空。",
    },
)

# 官方模型仓库身份（HF API /api/models/Qwen/Qwen3.8-Flash-Next 于 2026-10-29 返回）。
MODEL_REPOSITORY = {
    "repo_id": "Qwen/Qwen3.8-Flash-Next",
    "revision": "de4b8e4d43b917e7706784d8bb445c9af86a3540",
    "last_modified": "2026-08-27T05:03:36.000Z",
    "pipeline_tag": "image-text-to-text",
    "library_name": "transformers",
    "model_type": "qwen4_exp",
    "architectures": ["Qwen4ExpForConditionalGeneration"],
    "text_model_type": "qwen4_exp_text",
    "checkpoint_transformers_version": "5.8.0.dev0",
    "note": (
        "配置与模型卡按上述 revision 取得；只取 config.json（4,745 B）与 README.md（65,155 B）"
        "两个小文件，不下载 131 个权重分片中的任何 shard。视觉编码器字段存在但不在本票范围。"
    ),
}

# 所选 eager forward 路径实际读取的定义闭包（相对工作区根）。
SOURCE_FILES = {
    "config.json": "models/Qwen/Qwen3.8-Flash-Next/config.json",
    "README.md": "models/Qwen/Qwen3.8-Flash-Next/README.md",
    "modeling_qwen4_exp.py": (
        "hflib/transformers/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py"
    ),
    "configuration_qwen4_exp.py": (
        "hflib/transformers/src/transformers/models/qwen4_exp/configuration_qwen4_exp.py"
    ),
    "cache_utils.py": "hflib/transformers/src/transformers/cache_utils.py",
    "masking_utils.py": "hflib/transformers/src/transformers/masking_utils.py",
    "activations.py": "hflib/transformers/src/transformers/activations.py",
    "configuration_utils.py": "hflib/transformers/src/transformers/configuration_utils.py",
}

# 已分析来源指纹（SHA256，本地检出 536ecc00… / 官方 config revision 见上）。
# 来源漂移（任一文件变化）时构建具名拒绝、要求重新分析后更新指纹，而不是拿旧
# 结论继续产出。
ANALYZED_SOURCE_FINGERPRINTS = {
    "config.json": "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b",
    "README.md": "35ca37ccc366f1ba478dab33841a2c0c18ce53fd62f291ca05341f7728b225b2",
    "modeling_qwen4_exp.py": "f94401c10c1bbafd814cc174ba6f40dde1f40b35715ce1ae827ccf8ee3cd81fa",
    "configuration_qwen4_exp.py": "ac0950ab34516d26fe4f6b28cffe0649d763449effe44e42b5913463806f7e0b",
    "cache_utils.py": "2bef552bc39e8a4dc7ddfcf25740e6a4a7c4b836d92f5e56629a8ebfcd9cb246",
    "masking_utils.py": "c1d0e1e61754db819382bb03acdd2ab38c1bed3e9e9cf010918d2bf2771a8415",
    "activations.py": "5b20c0a3625edc0001a98f09ce3c6b5baa1100e1d7ad8dee649e4d45c8468665",
    "configuration_utils.py": "6cde90e0992d0dcf0216ab773254c5dd243e04f3c896d6e3d88a2b8be0d46567",
}


class HfConfigError(ValueError):
    """HF 根配置缺失/非法，或与已解释的身份不一致（具名拒绝）。"""


def verify_source_fingerprints(workspace_root: Path) -> None:
    """复算 SOURCE_FILES 各输入 SHA256 并与已分析指纹核对；漂移具名拒绝。"""
    import hashlib

    for key, relpath in SOURCE_FILES.items():
        path = Path(workspace_root) / relpath
        if not path.is_file():
            raise HfConfigError(f"已分析输入缺失: {relpath}（不下载、不伪造）")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = ANALYZED_SOURCE_FINGERPRINTS.get(key)
        if expected is None:
            raise HfConfigError(f"输入 {key} 未登记已分析指纹（拒绝未分析来源）")
        if actual != expected:
            raise HfConfigError(
                f"来源漂移: {relpath} 实际 SHA256 {actual} ≠ 已分析指纹 {expected}；"
                "来源已变化，既有分析结论作废，需重新分析并更新指纹后才可构建"
            )


def _load_root_config(workspace_root: Path) -> dict:
    path = Path(workspace_root) / SOURCE_FILES["config.json"]
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise HfConfigError(f"HF 根配置不可读: {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise HfConfigError(f"HF 根配置不是合法 JSON: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise HfConfigError(f"HF 根配置顶层必须是对象: {path}")
    if raw.get("model_type") != "qwen4_exp":
        raise HfConfigError(
            f"HF 根配置 model_type 必须是 qwen4_exp（Qwen3.8-Flash-Next 官方实现）: {path}"
        )
    if raw.get("architectures") != ["Qwen4ExpForConditionalGeneration"]:
        raise HfConfigError(
            f"HF 根配置 architectures 必须是 ['Qwen4ExpForConditionalGeneration']: {path}"
        )
    text = raw.get("text_config")
    if not isinstance(text, dict):
        raise HfConfigError("根配置缺少 text_config 对象（本票只分析语言主干）")
    if text.get("model_type") != "qwen4_exp_text":
        raise HfConfigError(
            f"text_config.model_type 必须是 qwen4_exp_text: {text.get('model_type')!r}"
        )
    vision = raw.get("vision_config")
    if not isinstance(vision, dict) or vision.get("model_type") not in (
        "qwen4_exp", "qwen4_exp_vision"
    ):
        raise HfConfigError("根配置缺少合法的 vision_config（多模态复合配置的结构事实）")
    return raw


def effective_layer_types(text: dict) -> list[str]:
    """注意力层型：显式列表，按源码 ``__post_init__`` 把 full_attention 规范成
    ``indexed_attention``（configuration:179-183）；再有 ``remap_legacy_layer_types``
    兜底（configuration_utils.py:87-134）。不能按配置字面值判断层型。
    """
    n = int(text["num_hidden_layers"])
    explicit = text.get("layer_types")
    if not isinstance(explicit, list) or len(explicit) != n:
        raise HfConfigError(f"layer_types 必须是长度 {n} 的数组（实得 {explicit!r}）")
    remap = {
        "full_attention": "indexed_attention",
        "deepseek_sparse_attention": "indexed_attention",
    }
    types: list[str] = []
    for index, value in enumerate(explicit):
        mapped = remap.get(value, value)
        if mapped not in ("linear_attention", "indexed_attention"):
            raise HfConfigError(f"layer_types[{index}]={value!r} 不在已分析层型内")
        types.append(mapped)
    if "linear_attention" not in types or "indexed_attention" not in types:
        raise HfConfigError("本模型必须是 Gated DeltaNet 与 QSA 的混合层型（缺一种即身份不符）")
    # 模型卡与源码都给出每 4 层一个 QSA：12 层，索引 3,7,11,...,47。
    expected_qsa = [i for i in range(n) if (i + 1) % 4 == 0]
    actual_qsa = [i for i, value in enumerate(types) if value == "indexed_attention"]
    if actual_qsa != expected_qsa:
        raise HfConfigError(
            f"QSA 层 {actual_qsa} 与 12×(3 GDN + 1 QSA) 的每 4 层模式 {expected_qsa} 不一致"
        )
    return types


def qsa_layer_indices(values: "HfModelValues") -> list[int]:
    return [i for i in range(values.num_hidden_layers)
            if values.layer_types[i] == "indexed_attention"]


def gdn_layer_indices(values: "HfModelValues") -> list[int]:
    return [i for i in range(values.num_hidden_layers)
            if values.layer_types[i] == "linear_attention"]


def ple_layer_indices(values: "HfModelValues") -> list[int]:
    """配置的 one-indexed PLE 层 → 0-based decoder layer 索引。"""
    return [int(layer_id) - 1 for layer_id in values.ple_layer_ids]


@dataclass(frozen=True)
class HfModelValues:
    """layer_idx 视角的 Qwen3.8-Flash-Next 文本 HF 有效配置值（静态解释，非运行时探值）。"""

    # 身份
    model_key: str
    display_name: str
    implementation: str
    source_revision: str
    config_path: str
    model_revision: str
    checkpoint_transformers_version: str
    # 全局
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rotary_dim: int
    partial_rotary_factor: float
    rope_theta: float
    mrope_section: tuple
    mrope_interleaved: bool
    vocab_size: int
    rms_norm_eps: float
    hidden_act: str
    use_cache: bool
    max_position_embeddings: int
    tie_word_embeddings: bool
    pad_token_id: int | None
    dtype: str
    attention_bias: bool
    attention_dropout: float
    attention_scaling: float
    # Gated DeltaNet
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    output_gate_type: str
    # QSA
    indexer_n_heads: int
    indexer_kv_heads: int
    indexer_head_dim: int
    indexer_budget: int
    indexer_compress_ratio: int
    indexer_block_topk: int
    # Gated Residual / hyper-connections
    hc_count: int
    hc_lowrank: int
    # N-gram / PLE
    ple_layer_ids: list[int]
    ple_embed_dim: int
    ple_conv_kernel_size: int
    ple_conv_dilation: int
    ple_short_conv_state_len: int
    ngram_size: int
    heads_per_ngram: int
    ngram_heads: int
    ngram_head_dim: int
    ngram_vocab_size_base: int
    ngram_vocab_divisor: int
    split_ngram_parts: int
    ngram_seed: int
    ngram_context_len: int
    eos_token_id: int
    # MoE
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    norm_topk_prob: bool
    # 层规划
    layer_types: list[str]
    # 声明但明确不在本票范围
    mtp_num_hidden_layers: int
    mtp_layer_types: tuple
    vision_depth: int
    vision_hidden_size: int
    image_token_id: int
    video_token_id: int
    number_of_conv_states: int

    def layer_type(self, layer_idx: int) -> str:
        return self.layer_types[layer_idx]

    def is_ple_layer(self, layer_idx: int) -> bool:
        return (layer_idx + 1) in self.ple_layer_ids

    def head_dim_rope(self, layer_idx: int | None = None) -> int:
        return self.rotary_dim


def _git_head(checkout_root: Path) -> str:
    """本地 Transformers 检出的 HEAD 身份；不可得时如实标注 unknown。"""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=checkout_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def load_hf_model_values(
    workspace_root: str | Path,
    model_key: str = "qwen_3_8_flash_next_hf",
) -> HfModelValues:
    """读取根 config.json 并静态解释为 Qwen3.8-Flash-Next 文本有效配置；失配/缺字段具名拒绝。"""
    workspace_root = Path(workspace_root)
    raw = _load_root_config(workspace_root)
    text = raw["text_config"]
    vision = raw["vision_config"]

    def _require(name: str):
        value = text.get(name)
        if value is None:
            raise HfConfigError(f"text_config 缺少必要字段 {name!r}")
        return value

    layer_types = effective_layer_types(text)
    n = int(_require("num_hidden_layers"))

    hidden_size = int(_require("hidden_size"))
    num_heads = int(_require("num_attention_heads"))
    num_kv = int(_require("num_key_value_heads"))
    head_dim = int(_require("head_dim"))
    if head_dim != 256:
        raise HfConfigError(f"本票锁定的实现事实是 head_dim=256（实得 {head_dim}）")
    if num_heads != 24 or num_kv != 2:
        raise HfConfigError(
            f"QSA 头数必须是 24 Q / 2 KV（实得 {num_heads}/{num_kv}）；"
            "GQA repeat=12 与 o_proj 6144 依赖该事实"
        )
    rope_params = text.get("rope_parameters")
    if not isinstance(rope_params, dict):
        raise HfConfigError("text_config 缺少 rope_parameters 对象（RoPE 语义来源）")
    partial = float(rope_params.get("partial_rotary_factor", 1.0))
    rotary_dim = int(head_dim * partial)
    if partial != 0.25 or rotary_dim != 64:
        raise HfConfigError(
            f"本票锁定的实现事实是 partial_rotary_factor=0.25 → rotary_dim=64"
            f"（实得 {partial}/{rotary_dim}）"
        )
    rope_theta = float(rope_params.get("rope_theta"))
    mrope_section = tuple(int(x) for x in rope_params.get("mrope_section", []))
    if mrope_section != (11, 11, 10):
        raise HfConfigError(f"本票锁定的 mrope_section=(11,11,10)（实得 {mrope_section}）")
    if not bool(rope_params.get("mrope_interleaved", False)):
        raise HfConfigError("本票锁定的实现事实是 mrope_interleaved=true")

    # Gated DeltaNet
    key_heads = int(_require("linear_num_key_heads"))
    value_heads = int(_require("linear_num_value_heads"))
    key_dim = int(_require("linear_key_head_dim"))
    value_dim = int(_require("linear_value_head_dim"))
    conv_kernel = int(_require("linear_conv_kernel_dim"))
    if (key_heads, value_heads, key_dim, value_dim, conv_kernel) != (16, 48, 128, 128, 4):
        raise HfConfigError(
            "Gated DeltaNet 必须是 16 QK 头 / 48 V 头 / 128 维 / conv kernel=4"
            f"（实得 {key_heads}/{value_heads}/{key_dim}/{value_dim}/{conv_kernel}）"
        )
    if value_heads % key_heads != 0:
        raise HfConfigError("linear_num_value_heads 必须能被 linear_num_key_heads 整除")
    output_gate_type = str(text.get("output_gate_type") or text.get("hidden_act", "silu"))
    if output_gate_type != "sigmoid":
        raise HfConfigError(
            f"本票锁定的实现事实是 Gated RMSNorm 输出门=sigmoid（实得 {output_gate_type!r}）"
        )

    # QSA
    idx_heads = int(_require("indexer_n_heads"))
    idx_kv_heads = int(_require("indexer_kv_heads"))
    idx_head_dim = int(_require("indexer_head_dim"))
    idx_budget = int(_require("indexer_budget"))
    idx_ratio = int(_require("indexer_compress_ratio"))
    if (idx_heads, idx_kv_heads, idx_head_dim, idx_budget, idx_ratio) != (4, 1, 128, 2048, 4):
        raise HfConfigError(
            "QSA indexer 必须是 4 Q 头 / 1 KV 头 / 128 维 / budget 2048 / compress 4"
            f"（实得 {idx_heads}/{idx_kv_heads}/{idx_head_dim}/{idx_budget}/{idx_ratio}）"
        )
    if idx_budget % idx_ratio != 0:
        raise HfConfigError("indexer_budget 必须能被 indexer_compress_ratio 整除")
    if rotary_dim > idx_head_dim:
        raise HfConfigError("attention rotary_dim 必须不超过 indexer_head_dim")

    # Gated Residual
    hc_count = int(_require("hc_count"))
    hc_lowrank = int(_require("hc_lowrank"))
    if hc_count != 4 or hc_lowrank != 320:
        raise HfConfigError(
            f"Gated Residual 必须是 4 路流 / lowrank=320（实得 {hc_count}/{hc_lowrank}）"
        )

    # PLE / n-gram
    ple_ids = [int(x) for x in _require("ple_layer_ids")]
    if ple_ids != [2]:
        raise HfConfigError(f"本票锁定的 PLE 层是 one-indexed [2]（实得 {ple_ids}）")
    if layer_types[ple_ids[0] - 1] != "linear_attention":
        raise HfConfigError("PLE 层必须是 linear_attention 层（配置类 validate_architecture）")
    ple_embed_dim = int(_require("ple_embed_dim"))
    ple_conv = int(_require("ple_conv_kernel_size"))
    ngram_size = int(_require("ngram_size"))
    heads_per_ngram = int(_require("heads_per_ngram"))
    ngram_heads = (ngram_size - 1) * heads_per_ngram
    if ngram_heads <= 0 or ple_embed_dim % ngram_heads != 0:
        raise HfConfigError("ple_embed_dim 必须能被 (ngram_size-1)*heads_per_ngram 整除")
    ngram_head_dim = ple_embed_dim // ngram_heads
    if (ple_embed_dim, ple_conv, ngram_size, heads_per_ngram) != (2560, 4, 3, 8):
        raise HfConfigError(
            "PLE 必须是 ple_embed_dim=2560 / conv kernel=4 / ngram=3 / 8 heads"
            f"（实得 {ple_embed_dim}/{ple_conv}/{ngram_size}/{heads_per_ngram}）"
        )
    ngram_vocab_base = int(_require("ngram_vocab_size_base"))
    ngram_vocab_divisor = int(_require("make_ngram_vocab_size_divisible_by"))
    split_parts = int(_require("split_ngram_parts"))
    seed = int(text.get("seed", 1234))
    eos = text.get("eos_token_id")
    if isinstance(eos, list):
        eos_id = int(eos[0])
    elif eos is not None:
        eos_id = int(eos)
    else:
        raise HfConfigError("PLE 开启时 eos_token_id 必须存在（ngram 上下文切断依据）")
    conv_dilation = ngram_size
    short_conv_state_len = (ple_conv - 1) * conv_dilation

    # MoE
    num_experts = int(_require("num_experts"))
    top_k = int(_require("num_experts_per_tok"))
    moe_inter = int(_require("moe_intermediate_size"))
    shared_inter = int(_require("shared_expert_intermediate_size"))
    if (num_experts, top_k, moe_inter, shared_inter) != (512, 10, 640, 640):
        raise HfConfigError(
            "MoE 必须是 512 专家 / top-10 / 专家中间维 640 / 共享专家 640"
            f"（实得 {num_experts}/{top_k}/{moe_inter}/{shared_inter}）"
        )
    if not bool(text.get("norm_topk_prob", True)):
        raise HfConfigError("本票锁定的实现事实是 norm_topk_prob=true")

    rms_norm_eps = float(_require("rms_norm_eps"))
    if rms_norm_eps != 1e-6:
        raise HfConfigError(f"本票锁定的 RMSNorm eps=1e-6（实得 {rms_norm_eps}）")
    hidden_act = str(text.get("hidden_act", "silu"))
    if hidden_act != "silu":
        raise HfConfigError(f"本票只分析 hidden_act=silu 的 SwiGLU（实得 {hidden_act!r}）")

    mtp = text.get("mtp")
    if not isinstance(mtp, dict) or int(mtp.get("num_hidden_layers", 0)) != 1:
        raise HfConfigError("本票锁定的 MTP 声明是 1 层（mtp.num_hidden_layers=1）")
    mtp_types = tuple(mtp.get("layer_types") or [])
    if mtp_types != ("full_attention",):
        raise HfConfigError(f"MTP 层型必须是 ['full_attention']（实得 {mtp_types}）")

    number_of_conv_states = 3 if ple_ids else 1
    if number_of_conv_states != 3:
        raise HfConfigError("PLE 开启时 number_of_conv_states 必须是 3")

    return HfModelValues(
        model_key=model_key,
        display_name=(
            "Qwen3.8-Flash-Next (Qwen/Qwen3.8-Flash-Next, HF transformers "
            "qwen4_exp 文本主干)"
        ),
        implementation="transformers",
        source_revision=_git_head(workspace_root / "hflib" / "transformers"),
        config_path=SOURCE_FILES["config.json"],
        model_revision=MODEL_REPOSITORY["revision"],
        checkpoint_transformers_version=MODEL_REPOSITORY["checkpoint_transformers_version"],
        hidden_size=hidden_size,
        num_hidden_layers=n,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        partial_rotary_factor=partial,
        rope_theta=rope_theta,
        mrope_section=mrope_section,
        mrope_interleaved=True,
        vocab_size=int(_require("vocab_size")),
        rms_norm_eps=rms_norm_eps,
        hidden_act=hidden_act,
        use_cache=bool(text.get("use_cache", True)),
        max_position_embeddings=int(_require("max_position_embeddings")),
        tie_word_embeddings=bool(raw.get("tie_word_embeddings",
                                         text.get("tie_word_embeddings", False))),
        pad_token_id=(
            int(text["pad_token_id"]) if text.get("pad_token_id") is not None else None
        ),
        dtype=str(text.get("dtype", "bfloat16")),
        attention_bias=bool(text.get("attention_bias", False)),
        attention_dropout=float(text.get("attention_dropout", 0.0)),
        attention_scaling=head_dim ** -0.5,
        linear_num_key_heads=key_heads,
        linear_num_value_heads=value_heads,
        linear_key_head_dim=key_dim,
        linear_value_head_dim=value_dim,
        linear_conv_kernel_dim=conv_kernel,
        output_gate_type=output_gate_type,
        indexer_n_heads=idx_heads,
        indexer_kv_heads=idx_kv_heads,
        indexer_head_dim=idx_head_dim,
        indexer_budget=idx_budget,
        indexer_compress_ratio=idx_ratio,
        indexer_block_topk=idx_budget // idx_ratio,
        hc_count=hc_count,
        hc_lowrank=hc_lowrank,
        ple_layer_ids=ple_ids,
        ple_embed_dim=ple_embed_dim,
        ple_conv_kernel_size=ple_conv,
        ple_conv_dilation=conv_dilation,
        ple_short_conv_state_len=short_conv_state_len,
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        ngram_heads=ngram_heads,
        ngram_head_dim=ngram_head_dim,
        ngram_vocab_size_base=ngram_vocab_base,
        ngram_vocab_divisor=ngram_vocab_divisor,
        split_ngram_parts=split_parts,
        ngram_seed=seed,
        ngram_context_len=ngram_size - 1,
        eos_token_id=eos_id,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=moe_inter,
        shared_expert_intermediate_size=shared_inter,
        norm_topk_prob=bool(text.get("norm_topk_prob", True)),
        layer_types=layer_types,
        mtp_num_hidden_layers=int(mtp.get("num_hidden_layers", 0)),
        mtp_layer_types=mtp_types,
        vision_depth=int(vision.get("depth", 0)),
        vision_hidden_size=int(vision.get("hidden_size", 0)),
        image_token_id=int(raw.get("image_token_id", 0)),
        video_token_id=int(raw.get("video_token_id", 0)),
        number_of_conv_states=number_of_conv_states,
    )


def effective_parameter_records(values: HfModelValues) -> list[dict]:
    """逐参数 raw 字段/默认/派生规则与来源（供参数符号区联查）。"""
    modeling = SOURCE_FILES["modeling_qwen4_exp.py"]
    configuration = SOURCE_FILES["configuration_qwen4_exp.py"]
    config_json = SOURCE_FILES["config.json"]
    cache_utils = SOURCE_FILES["cache_utils.py"]
    masking = SOURCE_FILES["masking_utils.py"]
    config_utils = SOURCE_FILES["configuration_utils.py"]

    def cfg(locator: str, note: str) -> dict:
        return {"path": config_json, "locator": locator, "kind": "config", "note": note}

    def rule(locator: str, note: str) -> dict:
        return {"path": configuration, "locator": locator, "kind": "derived", "note": note}

    def src(path: str, locator: str, note: str) -> dict:
        return {"path": path, "locator": locator, "kind": "source", "note": note}

    return [
        {
            "field": "model_identity",
            "value": {
                "repo_id": MODEL_REPOSITORY["repo_id"],
                "revision": MODEL_REPOSITORY["revision"],
                "model_type": "qwen4_exp",
                "architectures": ["Qwen4ExpForConditionalGeneration"],
                "text_config.model_type": "qwen4_exp_text",
                "pipeline_tag": MODEL_REPOSITORY["pipeline_tag"],
            },
            "rule": "官方模型仓库 revision 与根配置 model_type/architectures 双条件锁定；"
            "不以目录名或相似模型（Qwen3-Next/Qwen3.5/Qwen3-MoE）代替",
            "source_refs": [
                cfg("model_type", "根配置字段"),
                cfg("architectures", "根配置字段"),
                cfg("text_config.model_type", "文本子配置字段"),
            ],
        },
        {
            "field": "text_vision_boundary",
            "value": {
                "text_config": "qwen4_exp_text",
                "vision_config": "qwen4_exp（配置类 __post_init__ 规范为 qwen4_exp_vision）",
                "vision_depth": values.vision_depth,
                "vision_hidden_size": values.vision_hidden_size,
                "image_token_id": values.image_token_id,
                "video_token_id": values.video_token_id,
            },
            "rule": "多模态复合配置：本票只分析 text_config（语言主干），vision_config 与"
            "图像/视频 placeholder 替换路径不在范围",
            "source_refs": [
                cfg("text_config", "文本子配置"),
                cfg("vision_config", "视觉子配置（仅记录边界，不分析）"),
                cfg("image_token_id", "图像占位 token（不在本主干）"),
                cfg("video_token_id", "视频占位 token（不在本主干）"),
            ],
        },
        {
            "field": "hidden_size",
            "value": values.hidden_size,
            "rule": "text_config 直接字段；Gated DeltaNet/Attention/MoE 的公共宽度",
            "source_refs": [cfg("text_config.hidden_size", "文本子配置字段"),
                            rule("108", "配置类字段声明")],
        },
        {
            "field": "num_hidden_layers",
            "value": values.num_hidden_layers,
            "rule": "text_config 直接字段；主干层数 48（0..47）。模型卡给出 12×"
            "(3×Gated DeltaNet + 1×QSA)。checkpoint 的 mtp.* 不被本地实现加载"
            "（modeling:1313/1589），forward 只循环 0..47",
            "source_refs": [
                cfg("text_config.num_hidden_layers", "文本子配置字段"),
                rule("110", "配置类字段声明"),
                src(modeling, "1313", "_keys_to_ignore_on_load_unexpected = [r\"^mtp.*\"]"),
                src(modeling, "1473-1482", "for layer_idx, decoder_layer in enumerate(self.layers[:num_hidden_layers])"),
            ],
        },
        {
            "field": "layer_types",
            "value": values.layer_types,
            "rule": "显式 48 项；每 4 层一个 full_attention（3,7,...,47），其余 "
            "linear_attention。Qwen4ExpTextConfig.__post_init__ 把 full_attention 规范为 "
            "indexed_attention（configuration:179-183）；不能按配置字面值判断层型",
            "source_refs": [
                cfg("text_config.layer_types", "文本子配置显式数组（full_attention）"),
                rule("173-183", "__post_init__：full_attention → indexed_attention"),
                src(config_utils, "87-134", "remap_legacy_layer_types 兜底"),
                src(modeling, "1446-1460", "causal_mask_mapping 只含 indexed_attention/linear_attention 两个键"),
            ],
        },
        {
            "field": "linear_attention_gated_deltanet",
            "value": {
                "num_key_heads": values.linear_num_key_heads,
                "num_value_heads": values.linear_num_value_heads,
                "key_head_dim": values.linear_key_head_dim,
                "value_head_dim": values.linear_value_head_dim,
                "conv_kernel": values.linear_conv_kernel_dim,
                "output_gate_type": values.output_gate_type,
            },
            "rule": "Gated DeltaNet：q/k/v 合并投影 2560→10240；q/k 16 头、v 48 头，"
            "q/k repeat_interleave(3) 到 48 头；短卷积 kernel=4；状态 "
            "[B,48,128,128] 固定；输出 Gated RMSNorm 的 activation=sigmoid",
            "source_refs": [
                cfg("text_config.linear_num_key_heads", "文本子配置字段"),
                cfg("text_config.linear_num_value_heads", "文本子配置字段"),
                cfg("text_config.linear_key_head_dim", "文本子配置字段"),
                cfg("text_config.linear_value_head_dim", "文本子配置字段"),
                cfg("text_config.linear_conv_kernel_dim", "文本子配置字段"),
                cfg("text_config.output_gate_type", "文本子配置字段（sigmoid）"),
                src(modeling, "456-616", "Qwen4ExpTextGatedDeltaNet"),
            ],
        },
        {
            "field": "qsa_attention",
            "value": {
                "num_attention_heads": values.num_attention_heads,
                "num_key_value_heads": values.num_key_value_heads,
                "head_dim": values.head_dim,
                "rotary_dim": values.rotary_dim,
                "partial_rotary_factor": values.partial_rotary_factor,
                "rope_theta": values.rope_theta,
                "mrope_section": list(values.mrope_section),
                "attention_scaling": values.attention_scaling,
            },
            "rule": "QSA：q_proj 输出 2×24×256=12288，chunk 成 q 与 gate；q/k RMSNorm "
            "256；partial RoPE 只转前 64 维（192 维 nope）；KV 2 头，repeat 12；"
            "o_proj 6144→2560；输出乘 sigmoid(gate)",
            "source_refs": [
                cfg("text_config.num_attention_heads", "文本子配置字段"),
                cfg("text_config.num_key_value_heads", "文本子配置字段"),
                cfg("text_config.head_dim", "文本子配置字段"),
                cfg("text_config.rope_parameters", "partial_rotary_factor/rope_theta/mrope_section"),
                src(modeling, "810-892", "Qwen4ExpTextAttention"),
            ],
        },
        {
            "field": "qsa_indexer",
            "value": {
                "indexer_n_heads": values.indexer_n_heads,
                "indexer_kv_heads": values.indexer_kv_heads,
                "indexer_head_dim": values.indexer_head_dim,
                "indexer_budget": values.indexer_budget,
                "indexer_compress_ratio": values.indexer_compress_ratio,
                "indexer_block_topk": values.indexer_block_topk,
            },
            "rule": "QSA indexer：index_qk_proj 2560→(4+1)×128；4 query 头 + 1 共享 "
            "key 头；q/k RMSNorm 128；q 与 pooled block key 前 64 维 RoPE；每 "
            "compress_ratio=4 个可见 token 均值压成 micro-block；分数 = "
            "relu(q·k).sum(-1)/√128；每 query 选 min(512, 完整块数) 块 + tail",
            "source_refs": [
                cfg("text_config.indexer_n_heads", "文本子配置字段"),
                cfg("text_config.indexer_kv_heads", "文本子配置字段（必须为 1）"),
                cfg("text_config.indexer_head_dim", "文本子配置字段"),
                cfg("text_config.indexer_budget", "文本子配置字段（2048 token）"),
                cfg("text_config.indexer_compress_ratio", "文本子配置字段（4 token/块）"),
                rule("206-230", "validate_architecture：QSA 字段/整除/rotary_dim≤indexer_head_dim"),
                src(modeling, "664-770", "Qwen4ExpTextQSAIndexer"),
            ],
        },
        {
            "field": "gated_residual",
            "value": {
                "hc_count": values.hc_count,
                "hc_lowrank": values.hc_lowrank,
            },
            "rule": "Gated Residual：4 路流；每站点 hc_norm + down 10240→320 + SiLU + "
            "up 320→10240 + sigmoid 得逐元素读门；block_inject 10240→4 得每支路标量"
            "写门 2×sigmoid(·/4)；子层输出按写门展开加到原 4 路流；末端 mixer 用"
            "同一读门加权 mean，没有 Final RMSNorm",
            "source_refs": [
                cfg("text_config.hc_count", "文本子配置字段（4）"),
                cfg("text_config.hc_lowrank", "文本子配置字段（320）"),
                src(modeling, "994-1022", "Qwen4ExpTextGatedResidual"),
                src(modeling, "1261-1262", "attn_hyper_connection / mlp_hyper_connection"),
                src(modeling, "1471", "hidden_states.repeat(1, 1, hc_count)"),
                src(modeling, "1484", "hidden_states = self.hyper_connection_mixer(hidden_states)"),
            ],
        },
        {
            "field": "ngram_ple",
            "value": {
                "ple_layer_ids": values.ple_layer_ids,
                "ple_embed_dim": values.ple_embed_dim,
                "ple_conv_kernel_size": values.ple_conv_kernel_size,
                "ple_conv_dilation": values.ple_conv_dilation,
                "ple_short_conv_state_len": values.ple_short_conv_state_len,
                "ngram_size": values.ngram_size,
                "heads_per_ngram": values.heads_per_ngram,
                "ngram_heads": values.ngram_heads,
                "ngram_head_dim": values.ngram_head_dim,
                "ngram_vocab_size_base": values.ngram_vocab_size_base,
                "ngram_vocab_divisor": values.ngram_vocab_divisor,
                "split_ngram_parts": values.split_ngram_parts,
                "ngram_seed": values.ngram_seed,
                "ngram_context_len": values.ngram_context_len,
                "eos_token_id": values.eos_token_id,
            },
            "rule": "PLE at one-indexed layer 2（0-based 1）：16 个哈希头（bigram 8 + "
            "trigram 8）×160 维；token×每层乘子 XOR 后按每头素数码表取模；EOS 切断"
            "上下文；上一 token 上下文 [B,2]（state_idx=2）+ dilation-3 depthwise "
            "短卷积（state 9 步，state_idx=1）；key 2560→10240、value 2560→2560、"
            "query 为当前 4 路流",
            "source_refs": [
                cfg("text_config.ple_layer_ids", "文本子配置字段（one-indexed [2]）"),
                cfg("text_config.ple_embed_dim", "文本子配置字段"),
                cfg("text_config.ple_conv_kernel_size", "文本子配置字段"),
                cfg("text_config.ngram_size", "文本子配置字段"),
                cfg("text_config.heads_per_ngram", "文本子配置字段"),
                cfg("text_config.ngram_vocab_size_base", "文本子配置字段"),
                cfg("text_config.make_ngram_vocab_size_divisible_by", "文本子配置字段"),
                cfg("text_config.split_ngram_parts", "文本子配置字段"),
                cfg("text_config.seed", "文本子配置字段"),
                rule("165-185", "__post_init__：ple_layer_ids 排序、ple_embed_dim、number_of_conv_states=3"),
                rule("232-256", "validate_architecture：ngram_heads/ple_embed_dim/层型约束"),
                src(modeling, "1071-1246", "Qwen4ExpTextNGramEmbedding / Qwen4ExpTextPLELayer"),
            ],
        },
        {
            "field": "moe",
            "value": {
                "num_experts": values.num_experts,
                "num_experts_per_tok": values.num_experts_per_tok,
                "moe_intermediate_size": values.moe_intermediate_size,
                "shared_expert_intermediate_size": values.shared_expert_intermediate_size,
                "norm_topk_prob": values.norm_topk_prob,
            },
            "rule": "MoE：router softmax 512 选 10，top-10 归一化；无 correction bias、"
            "无 routed scaling factor；专家 gate_up [512,2×640,2560]、down "
            "[512,2560,640]；共享专家 2560→640→2560 另有 sigmoid 标量门；SwiGLU 无 clamp",
            "source_refs": [
                cfg("text_config.num_experts", "文本子配置字段（512）"),
                cfg("text_config.num_experts_per_tok", "文本子配置字段（10）"),
                cfg("text_config.moe_intermediate_size", "文本子配置字段（640）"),
                cfg("text_config.shared_expert_intermediate_size", "文本子配置字段（640）"),
                cfg("text_config.norm_topk_prob", "文本子配置字段（true）"),
                src(modeling, "951-991", "Qwen4ExpTextTopKRouter / SparseMoeBlock"),
                src(modeling, "911-948", "Qwen4ExpTextExperts"),
            ],
        },
        {
            "field": "cache_layout",
            "value": {
                "number_of_conv_states": values.number_of_conv_states,
                "gated_deltanet_conv_state": "[B,10240,4]",
                "gated_deltanet_recurrent_state": "[B,48,128,128]",
                "ple_conv_state": "[B,10240,9]（仅 layer 1）",
                "ple_ngram_context": "[B,2]（仅 layer 1）",
                "qsa_kv_cache": "[B,2,L,256]",
                "qsa_indexer_keys": "[B,L,128]",
            },
            "rule": "linear_attention → LinearAttentionLayer；indexed_attention → "
            "DynamicIndexedLayer；number_of_conv_states=3 因 PLE 开启，但只有 layer 1 "
            "使用 state_idx=1/2；QSA 的 K/V 与 indexer key 随历史线性增长",
            "source_refs": [
                cfg("text_config.num_hidden_layers", "层数"),
                rule("169-171", "number_of_conv_states = 3 if ple_layer_ids else 1"),
                src(cache_utils, "1285-1294", "DYNAMIC_LAYER_TYPE_MAPPING：linear_attention/indexed_attention"),
                src(cache_utils, "331-407", "DynamicIndexedLayer：主 K/V + indexer keys"),
                src(cache_utils, "1067-1149", "LinearAttentionLayer：conv/recurrent states"),
            ],
        },
        {
            "field": "position_ids_mrope",
            "value": {
                "position_rows": 4,
                "text_row": 0,
                "mrope_rows": "1..3（text-only 时三行相同）",
                "mrope_section": list(values.mrope_section),
                "mrope_interleaved": values.mrope_interleaved,
            },
            "rule": "网络默认 position_ids 是 arange(seqlen)+past_seen，扩展成 4×B×S；"
            "row0 用于 causal mask，rows1..3 进入 mRoPE 重组。text-only 输入没有视觉"
            "网格，三行相同，重组退化为同一组 cos/sin；本票保留该真实代码路径并注明",
            "source_refs": [
                cfg("text_config.rope_parameters.mrope_section", "文本子配置字段"),
                cfg("text_config.rope_parameters.mrope_interleaved", "文本子配置字段"),
                src(modeling, "1421-1436", "默认 position_ids 4 行；text_position_ids=row0"),
                src(modeling, "76-140", "Qwen4ExpTextRotaryEmbedding：mRoPE 重组"),
                src(modeling, "2255-2284", "text-only 时 compute_3d_position_ids 返回 None，"
                                           "由 TextModel 走默认 arange"),
            ],
        },
        {
            "field": "mtp",
            "value": {
                "mtp_num_hidden_layers": values.mtp_num_hidden_layers,
                "mtp_layer_types": list(values.mtp_layer_types),
            },
            "rule": "配置声明 1 层 MTP（full_attention），但本地 HF 实现没有 MTP 模块，"
            "加载时忽略 mtp.*，forward 只循环 0..47 —— 不在本主干",
            "source_refs": [
                cfg("text_config.mtp", "文本子配置字段"),
                cfg("text_config.mtp_num_hidden_layers", "文本子配置字段"),
                src(modeling, "1313", "_keys_to_ignore_on_load_unexpected = [r\"^mtp.*\"]"),
                src(modeling, "1589", "CausalLM 也忽略 ^mtp.* / ^model.visual.*"),
                src(modeling, "1473", "layers[:num_hidden_layers] 循环"),
            ],
        },
        {
            "field": "tie_word_embeddings",
            "value": values.tie_word_embeddings,
            "rule": "根配置 false；embed_tokens 与 lm_head 是两份独立权重",
            "source_refs": [cfg("tie_word_embeddings", "根配置字段"),
                            src(modeling, "2404", "lm_head 独立 Linear")],
        },
        {
            "field": "rms_norm_eps",
            "value": values.rms_norm_eps,
            "rule": "text_config 直接字段 1e-6；所有 Qwen4ExpTextRMSNorm（含 QSA q/k "
            "norm、indexer q/k norm、Gated Residual hc_norm、PLE norm_key/query/conv）"
            "都用它；Gated RMSNorm 也用同一 eps",
            "source_refs": [
                cfg("text_config.rms_norm_eps", "文本子配置字段"),
                src(modeling, "143-167", "Qwen4ExpTextRMSNorm（weight 初始 0，使用 1+weight）"),
                src(modeling, "171-187", "Qwen4ExpTextRMSNormGated"),
            ],
        },
    ]


# ---------------------------------------------------------------------------
# 三组真实层组（注意力层型 × MLP 类型）：归属、代表层与覆盖校验的单一事实来源
# ---------------------------------------------------------------------------

# 本配置只有三种真实组合：Gated DeltaNet + MoE（35 层）、Gated DeltaNet + PLE +
# MoE（1 层，layer 1）、QSA + MoE（12 层）。不存在 dense MLP 层。
LAYER_GROUP_SPECS = (
    {
        "id": "gdn-moe",
        "label": "Gated DeltaNet 线性注意力 + TopK MoE",
        "attention_type": "linear_attention",
        "moe_type": "sparse",
        "state_type": "gdn",
        "ple": False,
        "representative_layer_index": 0,
    },
    {
        "id": "gdn-ple-moe",
        "label": "Gated DeltaNet + N-gram PLE + TopK MoE（第 2 层）",
        "attention_type": "linear_attention",
        "moe_type": "sparse",
        "state_type": "gdn",
        "ple": True,
        "representative_layer_index": 1,
    },
    {
        "id": "qsa-moe",
        "label": "Qwen Sparse Attention（QSA） + TopK MoE",
        "attention_type": "indexed_attention",
        "moe_type": "sparse",
        "state_type": "qsa",
        "ple": False,
        "representative_layer_index": 3,
    },
)


def layer_group_assignments(values: HfModelValues) -> list[dict]:
    """按真实 layer_types/PLE 声明推导三组层归属（保持 SPECS 顺序）。

    返回 [{id,label,attention_type,moe_type,state_type,ple,
    representative_layer_index,layer_indices,count}]。任何未归属组合、空组、
    代表层不在组内、或三组未恰好覆盖全部主干层，均具名拒绝（ValueError）。
    """
    by_key: dict[tuple[str, bool], list[int]] = {}
    for index in range(values.num_hidden_layers):
        by_key.setdefault((values.layer_type(index), values.is_ple_layer(index)), []).append(index)
    groups: list[dict] = []
    covered: list[int] = []
    for spec in LAYER_GROUP_SPECS:
        key = (spec["attention_type"], bool(spec["ple"]))
        layer_indices = by_key.pop(key, [])
        if not layer_indices:
            raise ValueError(
                f"层组 {spec['id']} 在配置中没有真实层"
                f"（{spec['attention_type']} + ple={spec['ple']}）"
            )
        representative = spec["representative_layer_index"]
        if representative not in layer_indices:
            raise ValueError(
                f"层组 {spec['id']} 代表层 {representative} 不属于真实层集合 {layer_indices}"
            )
        groups.append({**spec, "layer_indices": layer_indices, "count": len(layer_indices)})
        covered.extend(layer_indices)
    if by_key:
        leftover = {f"{attn}+ple={ple}": idx for (attn, ple), idx in by_key.items()}
        raise ValueError(f"存在未归属到任何已知层组的层型组合: {leftover}")
    expected = list(range(values.num_hidden_layers))
    if sorted(covered) != expected or len(covered) != len(set(covered)):
        raise ValueError(
            f"三组层集合未恰好覆盖 {values.num_hidden_layers} 层: 实得 {len(covered)} 项"
            f"（去重 {len(set(covered))}）"
        )
    return groups
