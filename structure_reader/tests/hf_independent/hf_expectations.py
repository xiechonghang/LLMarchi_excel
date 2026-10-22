"""Independent expectations for the DeepSeek-V4-Flash-0731 HF decoder sample.

This module is written by the verification role (opencode) and is deliberately
self-contained: it reads only the read-only HF inputs under the workspace
(``models/.../config.json`` and ``hflib/transformers``) plus its own frozen
constants. It never imports ``torch``/``transformers``, never executes the HF
model, and never reads product output, ``.work`` coordination state or any
generated figure.

The facts encoded here were derived *independently* by reading the pinned HF
sources (identity frozen in :data:`FROZEN_SHA256`) and re-deriving the effective
configuration from the raw ``config.json`` following the logic in
``configuration_deepseek_v4.py``. They are NOT copied from any generated product
artifact. See ``INDEPENDENT_CHECKLIST.md`` in this directory for the human-readable
checklist and the mapping to source locations.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Locations (resolved relative to this file, never hard-coded absolute paths)
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
PKG_DIR = _HERE.parent              # structure_reader/tests/hf_independent
TESTS_DIR = PKG_DIR.parent          # structure_reader/tests
STRUCTURE_READER = TESTS_DIR.parent  # structure_reader
WORKSPACE_ROOT = STRUCTURE_READER.parent  # repo root

MODEL_DIR = WORKSPACE_ROOT / "models" / "deepseek-ai" / "DeepSeek-V4-Flash-0731"
CONFIG_PATH = MODEL_DIR / "config.json"

_TRANSFORMERS = WORKSPACE_ROOT / "hflib" / "transformers" / "src" / "transformers"
DEEPSEEK_DIR = _TRANSFORMERS / "models" / "deepseek_v4"

SOURCE_PATHS = {
    "config": CONFIG_PATH,
    "modeling": DEEPSEEK_DIR / "modeling_deepseek_v4.py",
    "configuration": DEEPSEEK_DIR / "configuration_deepseek_v4.py",
    "cache_utils": _TRANSFORMERS / "cache_utils.py",
    "masking_utils": _TRANSFORMERS / "masking_utils.py",
    "modeling_rope_utils": _TRANSFORMERS / "modeling_rope_utils.py",
    "activations": _TRANSFORMERS / "activations.py",
    "deepseek_init": DEEPSEEK_DIR / "__init__.py",
}

TRANSFORMERS_HEAD = "536ecc007387a50e77603bb5d92100e9b07514cc"
MODEL_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
MODEL_REPO_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"

# Frozen identities of the inputs used to build these expectations. A mismatch
# means the independent expectation must be re-derived, not patched silently.
FROZEN_SHA256 = {
    "config": "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023",
    "modeling": "d6eb834f9d10eca879ca5390f219259eb452d19e1cf72d827f4334cbe4682266",
    "configuration": "49c818404caff1b4e76dff0ae7f468512932a5c40cc41b6cbd6585da4dec3aa2",
    "cache_utils": "2bef552bc39e8a4dc7ddfcf25740e6a4a7c4b836d92f5e56629a8ebfcd9cb246",
    "masking_utils": "c1d0e1e61754db819382bb03acdd2ab38c1bed3e9e9cf010918d2bf2771a8415",
    "modeling_rope_utils": "6a0727073e080e163d0287e15fc1acb90035e9a9df9cb19a13b632a40abfac24",
    "activations": "5b20c0a3625edc0001a98f09ce3c6b5baa1100e1d7ad8dee649e4d45c8468665",
    "deepseek_init": "542434a8efcd630378a559a0819a24fb0703b8dca13f4e120d9593dcd67abe2c",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_raw_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def read_source(name: str) -> str:
    return SOURCE_PATHS[name].read_text(encoding="utf-8")


_LINE_COMMENT = re.compile(r"#.*")
_WS = re.compile(r"\s+")


def strip_line_comments(text: str) -> str:
    return "\n".join(_LINE_COMMENT.sub("", line) for line in text.splitlines())


def normalize(text: str) -> str:
    """Collapse all whitespace so multi-line code can be searched as one string."""
    return _WS.sub(" ", text).strip()


def code(name: str) -> str:
    """Comment-free, whitespace-normalized source for substring/ordering checks."""
    return normalize(strip_line_comments(read_source(name)))


# ---------------------------------------------------------------------------
# Effective configuration derivation (mirrors configuration_deepseek_v4.py)
# ---------------------------------------------------------------------------

COMPRESS_RATIO_TO_LAYER_TYPE = {
    0: "sliding_attention",
    4: "compressed_sparse_attention",
    128: "heavily_compressed_attention",
}
DEFAULT_COMPRESS_RATES = {
    "compressed_sparse_attention": 4,
    "heavily_compressed_attention": 128,
}
LAYER_TYPE_TO_ROPE_LABEL = {
    "sliding_attention": "main",
    "compressed_sparse_attention": "compress",
    "heavily_compressed_attention": "compress",
}


def derive_layer_types(compress_ratios, n):
    """legacy compress_ratios ints -> layer_types, truncated to n layers."""
    return [COMPRESS_RATIO_TO_LAYER_TYPE[r] for r in compress_ratios][:n]


def v4pro_default_layer_types(n):
    """The non-legacy default branch (no compress_ratios): 2x HCA bootstrap + interleave.

    Kept so a negative test can prove that ignoring the legacy field changes the
    attention type of layer 4.
    """
    interleave = [
        "compressed_sparse_attention" if i % 2 else "heavily_compressed_attention"
        for i in range(max(n - 2, 0))
    ]
    return (["heavily_compressed_attention"] * min(n, 2) + interleave)[:n]


def derive_mlp_layer_types(n, num_hash_layers):
    return ["hash_moe"] * min(n, num_hash_layers) + ["moe"] * max(0, n - num_hash_layers)


def derive_effective(layer_idx: int = 4) -> dict:
    raw = load_raw_config()
    n = raw["num_hidden_layers"]
    layer_types = derive_layer_types(raw["compress_ratios"], n)
    mlp_layer_types = derive_mlp_layer_types(n, raw["num_hash_layers"])

    head_dim = raw["head_dim"]
    qk_rope_head_dim = raw["qk_rope_head_dim"]
    partial_rotary_factor = qk_rope_head_dim / head_dim

    compress_rates = dict(DEFAULT_COMPRESS_RATES)

    tree = {
        "model_type": raw["model_type"],
        "num_hidden_layers": n,
        "hidden_size": raw["hidden_size"],
        "num_attention_heads": raw["num_attention_heads"],
        "num_key_value_heads": raw["num_key_value_heads"],
        "num_key_value_groups": raw["num_attention_heads"] // raw["num_key_value_heads"],
        "head_dim": head_dim,
        "q_lora_rank": raw["q_lora_rank"],
        "qk_rope_head_dim": qk_rope_head_dim,
        "partial_rotary_factor": partial_rotary_factor,
        "rope_dim": qk_rope_head_dim,
        "nope_dim": head_dim - qk_rope_head_dim,
        "o_groups": raw["o_groups"],
        "o_lora_rank": raw["o_lora_rank"],
        "moe_intermediate_size": raw["moe_intermediate_size"],
        "n_routed_experts": raw["n_routed_experts"],
        "n_shared_experts": raw["n_shared_experts"],
        "num_experts_per_tok": raw["num_experts_per_tok"],
        "routed_scaling_factor": raw["routed_scaling_factor"],
        "scoring_func": raw["scoring_func"],
        "norm_topk_prob": raw["norm_topk_prob"],
        "swiglu_limit": raw["swiglu_limit"],
        "hidden_act": raw["hidden_act"],
        "sliding_window": raw["sliding_window"],
        "hc_mult": raw["hc_mult"],
        "hc_sinkhorn_iters": raw["hc_sinkhorn_iters"],
        "hc_eps": raw["hc_eps"],
        "index_n_heads": raw["index_n_heads"],
        "index_head_dim": raw["index_head_dim"],
        "index_topk": raw["index_topk"],
        "compress_rope_theta": raw["compress_rope_theta"],
        "rope_theta": raw["rope_theta"],
        "rope_scaling": raw["rope_scaling"],
        "num_hash_layers": raw["num_hash_layers"],
        "compress_rates": compress_rates,
        "layer_types": layer_types,
        "mlp_layer_types": mlp_layer_types,
        "layer_idx": layer_idx,
        "attention_type": layer_types[layer_idx],
        "moe_type": mlp_layer_types[layer_idx],
        "rope_label": LAYER_TYPE_TO_ROPE_LABEL[layer_types[layer_idx]],
        # attribute_map in configuration_deepseek_v4.py
        "num_local_experts": raw["n_routed_experts"],
        "intermediate_size": raw["moe_intermediate_size"],
        "persistent_sliding_keys": raw["sliding_window"] - 1,
        "csa_compress_rate": compress_rates["compressed_sparse_attention"],
        "hca_compress_rate": compress_rates["heavily_compressed_attention"],
    }
    return tree


def layer_index_groups(layer_types, mlp_layer_types):
    groups = {}
    for i, (attn, moe) in enumerate(zip(layer_types, mlp_layer_types)):
        groups.setdefault((attn, moe), []).append(i)
    return groups


# ---------------------------------------------------------------------------
# Expected weight / activation shapes for layer_idx=4 (CSA + TopK MoE + mHC)
# ---------------------------------------------------------------------------

def expected_attention_shapes(e: dict) -> dict:
    h, heads, hd = e["hidden_size"], e["num_attention_heads"], e["head_dim"]
    q_rank, og, olr = e["q_lora_rank"], e["o_groups"], e["o_lora_rank"]
    in_per_group = heads * hd // og
    return {
        "q_a_proj": [q_rank, h],
        "q_a_norm": [q_rank],
        "q_b_proj": [heads * hd, q_rank],
        "q_b_norm": [],  # unweighted RMSNorm
        "kv_proj": [hd, h],
        "kv_norm": [hd],
        "o_a_proj": [og * olr, in_per_group],
        "o_b_proj": [h, og * olr],
        "sinks": [heads],
        "q_heads_axis": ["B", "S", heads, hd],
        "kv_heads_axis": ["B", "S", e["num_key_value_heads"], hd],
    }


def expected_csa_compressor_shapes(e: dict) -> dict:
    h, hd = e["hidden_size"], e["head_dim"]
    r = e["csa_compress_rate"]
    ih, inH, iht = e["index_head_dim"], e["index_n_heads"], e["index_topk"]
    return {
        "kv_proj": [2 * hd, h],
        "gate_proj": [2 * hd, h],
        "position_bias": [r, 2 * hd],
        "kv_norm": [hd],
        "indexer.kv_proj": [2 * ih, h],
        "indexer.gate_proj": [2 * ih, h],
        "indexer.position_bias": [r, 2 * ih],
        "indexer.kv_norm": [ih],
        "indexer.q_b_proj": [inH * ih, e["q_lora_rank"]],
        "indexer.scorer.weights_proj": [inH, h],
        "overlap_ca_slice_width": hd,
        "new_series_width": 2 * r,
        "index_topk": iht,
    }


def expected_moe_shapes(e: dict) -> dict:
    h, i = e["hidden_size"], e["moe_intermediate_size"]
    ne = e["n_routed_experts"]
    return {
        "router.weight": [ne, h],
        "router.e_score_correction_bias": [ne],
        "experts.gate_up_proj": [ne, 2 * i, h],
        "experts.down_proj": [ne, h, i],
        "shared_experts.gate_proj": [i, h],
        "shared_experts.up_proj": [i, h],
        "shared_experts.down_proj": [h, i],
    }


def expected_mhc_shapes(e: dict) -> dict:
    n, h = e["hc_mult"], e["hidden_size"]
    cws = (2 + n) * n
    return {
        "fn": [cws, n * h],
        "base": [cws],
        "scale": [3],
        "streams": ["B", "S", n, h],
        "collapsed": ["B", "S", h],
        "pre": ["B", "S", n],
        "post": ["B", "S", n],
        "comb": ["B", "S", n, n],
        "cws": cws,
    }


__all__ = [
    "SOURCE_PATHS",
    "FROZEN_SHA256",
    "TRANSFORMERS_HEAD",
    "MODEL_REVISION",
    "sha256_file",
    "load_raw_config",
    "read_source",
    "strip_line_comments",
    "normalize",
    "code",
    "derive_layer_types",
    "v4pro_default_layer_types",
    "derive_mlp_layer_types",
    "derive_effective",
    "layer_index_groups",
    "expected_attention_shapes",
    "expected_csa_compressor_shapes",
    "expected_moe_shapes",
    "expected_mhc_shapes",
]
