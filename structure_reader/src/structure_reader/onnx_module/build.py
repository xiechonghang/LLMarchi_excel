"""模块级 ONNX 构建编排：GB → ModelProto → checker / shape inference / onnxsim → 文件与状态。

不执行 HF、不加载真实权重；权重是 graph input；sim 用 check_n=0（不做数值自检）。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from structure_reader.onnx_module import builders
from structure_reader.onnx_module.modeling import (
    DTYPE, IR_VERSION, OPSET, derive_compress_rope, derive_main_rope)
from structure_reader.onnx_module.view import proto_display_graph, proto_details

FLOAT32 = TensorProto.FLOAT


def _dims(shape):
    out = []
    for d in shape:
        if isinstance(d, (int, np.integer)):
            out.append(int(d))
        else:
            out.append(str(d))
    return out


def _vi(name, shape, dtype=FLOAT32):
    return helper.make_tensor_value_info(name, dtype, _dims(shape))


def _type_of(dtype: str):
    return {"float32": FLOAT32, "int64": TensorProto.INT64,
            "tensor(float)": FLOAT32, "tensor(int64)": TensorProto.INT64}[dtype]


def gb_to_model(gb, module_id: str, phase: str) -> onnx.ModelProto:
    nodes = []
    for n in gb.nodes:
        nodes.append(helper.make_node(n["op_type"], n["inputs"], n["outputs"],
                                      name=n["name"], **n["attrs"]))
    inputs = [_vi(i["name"], i["shape"], _type_of(i["dtype"])) for i in gb.inputs]
    outputs = [_vi(o["name"], o["shape"], _type_of(o["dtype"])) for o in gb.outputs]
    inits = []
    total = 0
    for c in gb.inits:
        arr = np.asarray(c["array"])
        total += int(arr.size)
        inits.append(numpy_helper.from_array(arr, name=c["name"]))
    if total > 100_000:
        raise ValueError(f"{module_id}: initializer 元素数 {total} 超过常量折叠上限")
    graph = helper.make_graph(nodes, f"hf_decoder_{phase}_{module_id}", inputs, outputs,
                              initializer=inits)
    graph.doc_string = (f"{module_id}/{phase}: structural FP32 ONNX derived from HF source; "
                        "see mapping.json for group->JSON op/source locator")
    model = helper.make_model(graph, producer_name="structure_reader.onnx_module",
                              opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    model.doc_string = (
        "structural FP32 graph (HF source-derived); weights are graph inputs with real config "
        "shapes but no values; cache pre/post are separate tensors; HF numerical equivalence "
        "is NOT claimed; onnxsim check_n=0 (no numeric check)")
    for k, v in (("dtype", DTYPE), ("hf_numeric_equivalence", "not_claimed"),
                 ("weights", "graph_inputs"), ("cache", "pre_input_post_output"),
                 ("opset", str(OPSET)), ("ir_version", str(IR_VERSION)),
                 ("phase", phase), ("module", module_id)):
        e = model.metadata_props.add()
        e.key, e.value = k, v
    return model


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build_module(ctx, module_id: str, builder) -> dict:
    gb = builder(ctx)
    gb.ensure_unique_names()
    model = gb_to_model(gb, module_id, ctx.pid)
    st = {"module": module_id, "phase": ctx.pid, "dtype": DTYPE, "opset": OPSET,
          "ir_version": IR_VERSION, "node_count_original": len(model.graph.node),
          "weight_inputs": [i["name"] for i in gb.inputs if i["role"] == "weight"],
          "cache_pre_inputs": [i["name"] for i in gb.inputs if i["role"] == "cache_pre"],
          "outputs": [{"name": o["name"], "json_id": o.get("json_id", o["name"]),
                       "role": o["role"], "shape": o["shape"]}
                      for o in gb.outputs],
          "checks": {}}
    # V1 checker
    onnx.checker.check_model(model, full_check=True)
    st["checks"]["checker"] = "ok"
    # V2 shape inference
    try:
        inf = onnx.shape_inference.infer_shapes(model, strict_mode=False)
        st["checks"]["shape_inference"] = "ok"
        unknown = 0
        for vi in list(inf.graph.value_info):
            for d in vi.type.tensor_type.shape.dim:
                if not d.HasField("dim_value") and not d.HasField("dim_param"):
                    unknown += 1
        st["checks"]["shape_inference_unknown_dims"] = unknown
    except Exception as exc:  # noqa: BLE001 - 如实记录，不伪造成功
        st["checks"]["shape_inference"] = f"failed: {type(exc).__name__}: {exc}"
    # V3 onnxsim（check_n=0，不做数值自检）
    try:
        from onnxsim import simplify
        sim, ok = simplify(model, check_n=0, perform_optimization=True)
        onnx.checker.check_model(sim, full_check=True)
        orig_names = [n.name for n in model.graph.node]
        sim_names = [n.name for n in sim.graph.node]
        st["checks"]["simplify"] = {
            "check_n": 0, "returned": bool(ok),
            "node_count_simplified": len(sim.graph.node),
            "changed": len(sim.graph.node) != len(model.graph.node),
            "removed_nodes": sorted(set(orig_names) - set(sim_names)),
            "added_nodes": sorted(set(sim_names) - set(orig_names))}
        model_sim = sim
    except Exception as exc:  # noqa: BLE001
        st["checks"]["simplify"] = f"failed: {type(exc).__name__}: {exc}"
        model_sim = model
    st["node_count_simplified"] = len(model_sim.graph.node)
    ob = model.SerializeToString()
    sb = model_sim.SerializeToString()
    st["sha256_original"] = _sha(ob)
    st["sha256_simplified"] = _sha(sb)
    st["bytes_original"] = len(ob)
    st["bytes_simplified"] = len(sb)
    st["identical_after_simplify"] = ob == sb

    # 映射：组 → JSON op id / 源码行
    from structure_reader.render.diagram_source_labels import op_source_locator
    mapping = {}
    for gid in gb.groups:
        op = ctx.ops.get(gid)
        loc = op_source_locator(op) if op else None
        mapping[gid] = {"json_op": gid if op else None,
                        "source_locator": loc,
                        "label": gb.groups[gid]["label"],
                        "node_count": len(gb.groups[gid]["nodes"]),
                        "nodes": [{"name": n["name"], "op_type": n["op_type"],
                                   "outputs": n["outputs"]}
                                  for n in gb.groups[gid]["nodes"]]}
    model_nodes = {n.name for n in model.graph.node}
    mapped_nodes = {n["name"] for g in mapping.values() for n in g["nodes"]}
    unmapped = sorted(model_nodes - mapped_nodes)
    if unmapped:
        raise ValueError(f"{module_id}/{ctx.pid}: {len(unmapped)} 个实际节点无 JSON/source 映射: "
                         f"{unmapped[:5]}")
    # onnxsim 插入节点：按实际 producer/consumer 邻接唯一关联到源组（非源码原生节点）
    sim_nodes = list(model_sim.graph.node)
    producer = {t: n for n in sim_nodes for t in n.output}
    node_group = {n["name"]: gid for gid, gd in mapping.items() for n in gd["nodes"]}
    derived_groups: dict[str, str] = {}
    inserted = []
    for n in sim_nodes:
        if n.name in node_group:
            continue
        prod_groups = sorted({node_group.get(producer[t].name) for t in n.input
                              if t in producer and node_group.get(producer[t].name)})
        cons_groups = sorted({node_group.get(c.name) for c in sim_nodes
                              if set(c.input) & set(n.output) and node_group.get(c.name)})
        cand = sorted(set(prod_groups) | set(cons_groups))
        evidence = {"inputs": [{"tensor": t,
                                "producer": producer[t].name if t in producer else None}
                               for t in n.input],
                    "producer_groups": prod_groups, "consumer_groups": cons_groups}
        rec = {"name": n.name, "op_type": n.op_type, "outputs": list(n.output),
               "derived": True, "kind": "onnxsim_inserted",
               "evidence": evidence,
               "note": "onnxsim 插入；依据实际 producer/consumer 邻接关联，非源码原生节点"}
        if len(cand) == 1:
            gid = cand[0]
            rec["source_groups"] = [gid]
            rec["locator"] = mapping[gid].get("source_locator")
            mapping[gid]["nodes"].append(rec)
            mapping[gid]["derived_count"] = mapping[gid].get("derived_count", 0) + 1
            derived_groups[n.name] = gid
        else:
            rec["source_groups"] = cand
            rec["unknown"] = True
            mapping.setdefault("sim_inserted_unresolved", {"label": "未唯一关联",
                                                          "nodes": []})["nodes"].append(rec)
        inserted.append(rec)
    st["unmapped_nodes"] = 0
    st["sim_extra_nodes"] = sorted({n.name for n in sim_nodes} - mapped_nodes)
    st["sim_inserted_bound"] = len(derived_groups)
    st["sim_inserted_unresolved"] = [r["name"] for r in inserted if r.get("unknown")]
    return {"model": model, "sim": model_sim, "state": st, "mapping": mapping,
            "display": proto_display_graph(model_sim, module_id, ctx.pid,
                                           {k: v["label"] for k, v in mapping.items()},
                                           derived_groups=derived_groups),
            "details": proto_details(model_sim, derived_groups=derived_groups),
            "bytes_original": ob, "bytes_simplified": sb}


REGISTRY = {
    "norm_attn": lambda c: builders.build_norm(c, "attn_norm"),
    "norm_ffn": lambda c: builders.build_norm(c, "ffn_norm"),
    "attn_csa": builders.build_attn_csa,
    "attn_indexer": builders.build_attn_indexer,
    "attn_mask": builders.build_attn_mask,
    "attn_q": builders.build_attn_q,
    "attn_kv": builders.build_attn_kv,
    "attn_cache": builders.build_attn_cache,
    "attn_core": builders.build_attn_core,
    "attn_out": builders.build_attn_out,
    "mhc_attn": lambda c: builders.build_mhc(c, "mhc_attn"),
    "mhc_ffn": lambda c: builders.build_mhc(c, "mhc_ffn"),
    "res_attn": builders.build_res_attn,
    "res_ffn": builders.build_res_ffn,
    "moe_shared": builders.build_moe_shared,
    "moe_merge": builders.build_moe_merge,
    "moe_route": builders.build_moe_route,
    "moe_expert": builders.build_moe_expert,
}


def _module_name_map():
    """图（17）→ ONNX 模块：CSA 复合图含 csa+indexer 两个模块。"""
    return {"attn_csa": ["attn_csa", "attn_indexer"]}


def build_all(workspace_root: Path, project_root: Path, doc: dict, config: dict,
              modules=None) -> dict:
    rope_main = derive_main_rope(config)
    rope_compress = derive_compress_rope(config)
    files: dict[str, bytes] = {}
    bundle: dict[str, dict] = {}
    report = {"schema": "hf_onnx_modules_build_v1",
              "dtype": DTYPE, "opset": OPSET, "ir_version": IR_VERSION,
              "weights": "graph_inputs_shapes_from_config_no_values",
              "simplify": {"check_n": 0, "numeric_check": "none",
                           "weights_folded": "no (weights are graph inputs)"},
              "modules": {}}
    for phase in doc["phases"]:
        ctx = builders.Ctx(doc, phase, config, rope_main, rope_compress)
        bundle[phase["id"]] = {}
        report["modules"][phase["id"]] = {}
        for mid, fn in REGISTRY.items():
            if modules and mid not in modules:
                continue
            try:
                res = build_module(ctx, mid, fn)
            except Exception as exc:  # noqa: BLE001 - 逐模块如实记录失败，不伪造
                report["modules"][phase["id"]][mid] = {
                    "module": mid, "phase": phase["id"], "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}"}
                continue
            base = f"onnx/{phase['id']}/{mid}"
            files[f"{base}/model.onnx"] = res["bytes_original"]
            files[f"{base}/model.sim.onnx"] = res["bytes_simplified"]
            files[f"{base}/mapping.json"] = json.dumps(
                {"schema": "hf_onnx_module_mapping_v1", "module": mid,
                 "phase": phase["id"], "groups": res["mapping"]},
                ensure_ascii=False, indent=2).encode()
            files[f"{base}/state.json"] = json.dumps(
                res["state"], ensure_ascii=False, indent=2).encode()
            bundle[phase["id"]][mid] = {"display": res["display"],
                                        "details": res["details"],
                                        "state": res["state"],
                                        "base": base}
            report["modules"][phase["id"]][mid] = res["state"]
    files["reports/onnx_build.json"] = json.dumps(
        report, ensure_ascii=False, indent=2).encode()
    return {"files": files, "bundle": bundle, "report": report}
