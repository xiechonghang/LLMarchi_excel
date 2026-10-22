"""把真实 ModelProto 转成右侧阅读图（语义分组；不是复制原 SVG）。

- 读简化后的 ModelProto：节点按 node name 前缀 `group/` 归入 JSON 算子组；
- 组 → 一个框；组间边由真实 tensor 连接折叠；边界 tensor → 端口节点；
- 组内真实节点/张量名与 shape 放入可展开明细（由 build 层生成表格数据）。
"""
from __future__ import annotations

import onnx


def _short(t: str) -> str:
    return t.split(".")[-1] if "." in t else t


def proto_display_graph(model: onnx.ModelProto, module_id: str, phase: str,
                        group_labels: dict[str, str], dtype: str = "float32",
                        derived_groups: dict[str, str] | None = None) -> dict:
    g = model.graph
    # 节点 → 组
    derived_groups = derived_groups or {}

    def group_of(name: str) -> str:
        if name in derived_groups:
            return derived_groups[name]
        return name.split("/", 1)[0] if "/" in name else "misc"

    group_order: list[str] = []
    group_nodes: dict[str, list] = {}
    for n in g.node:
        gid = group_of(n.name)
        if gid not in group_nodes:
            group_nodes[gid] = []
            group_order.append(gid)
        group_nodes[gid].append(n)

    # tensor 生产者（组粒度）
    producer: dict[str, str] = {}
    for gid, nodes in group_nodes.items():
        for n in nodes:
            for o in n.output:
                producer[o] = gid

    nodes = []
    for gid in group_order:
        ops = sorted({n.op_type for n in group_nodes[gid]})
        nodes.append({
            "id": f"g:{gid}", "kind": "op",
            "label": group_labels.get(gid, gid),
            "sub": f"ONNX · {len(group_nodes[gid])} 节点：{', '.join(ops)}",
            "href": "", "title": f"{gid} · nodes={len(group_nodes[gid])} · ops={','.join(ops)}",
            "w": 208, "wrap": 20,
        })
    graph_inputs = {i.name for i in g.input}
    graph_outputs = {o.name for o in g.output}
    in_edges: list = []
    input_names: list = []
    edges = []
    seen = set()
    for gid, glist in group_nodes.items():
        for n in glist:
            for t in n.input:
                src = producer.get(t)
                if src and src != gid:
                    key = (f"g:{src}", f"g:{gid}", t)
                    if key in seen:
                        continue
                    seen.add(key)
                    edges.append({"src": f"g:{src}", "dst": f"g:{gid}", "kind": "flow",
                                  "label": "", "cond": "", "href": "",
                                  "title": f"{_short(t)} · {t}"})
                elif not src and t in graph_inputs:
                    key = (f"i:{t}", f"g:{gid}", t)
                    if key in seen:
                        continue
                    seen.add(key)
                    in_edges.append({"src": "i:inputs", "dst": f"g:{gid}", "kind": "flow",
                                     "label": "", "cond": "", "href": "", "title": t})
                    input_names.append(t)
    # 输出合并成一个节点，避免每个输出各占一列把右图撑得过宽（完整输出名/形状在明细表）
    out_names = [o.name for o in g.output]
    if out_names:
        nodes.append({"id": "o:outputs", "kind": "port",
                      "label": f"输出 × {len(out_names)}",
                      "sub": "、".join(_short(n) for n in out_names)[:48],
                      "href": "", "title": "模块输出：" + ", ".join(out_names),
                      "w": 200, "wrap": 18})
        seen_out = set()
        for out in g.output:
            src = producer.get(out.name)
            if src and (src, out.name) not in seen_out:
                seen_out.add((src, out.name))
                edges.append({"src": f"g:{src}", "dst": "o:outputs", "kind": "flow",
                              "label": "", "cond": "", "href": "", "title": out.name})
    if input_names:
        nodes.append({"id": "i:inputs", "kind": "boundary",
                      "label": f"输入 × {len(input_names)}",
                      "sub": "、".join(_short(n) for n in input_names)[:48],
                      "href": "", "title": "图输入：" + ", ".join(input_names),
                      "w": 200, "wrap": 18})
        edges.extend(in_edges)
    # 去重端口节点
    uniq = {}
    for n in nodes:
        uniq[n["id"]] = n
    return {"id": f"onnx-{phase}-{module_id}", "phase": phase,
            "title": f"{module_id} · ONNX", "nodes": list(uniq.values()), "edges": edges}


def proto_details(model: onnx.ModelProto,
                  derived_groups: dict[str, str] | None = None) -> list[dict]:
    """右侧可展开明细：组 → 真实节点(op_type/name) + 输入输出 tensor(name/shape)。"""
    g = model.graph
    shape_by = {}
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        dims = []
        tt = vi.type.tensor_type
        for d in tt.shape.dim:
            if d.HasField("dim_value"):
                dims.append(d.dim_value)
            elif d.HasField("dim_param"):
                dims.append(d.dim_param)
            else:
                dims.append("?")
        shape_by[vi.name] = {"dtype": onnx.TensorProto.DataType.Name(tt.elem_type), "shape": dims}
    derived_groups = derived_groups or {}
    groups: dict[str, list] = {}
    for n in g.node:
        gid = (derived_groups.get(n.name) or
               (n.name.split("/", 1)[0] if "/" in n.name else "misc"))
        groups.setdefault(gid, []).append({
            "op_type": n.op_type, "name": n.name,
            "inputs": [{"name": t, **shape_by.get(t, {})} for t in n.input if t],
            "outputs": [{"name": t, **shape_by.get(t, {})} for t in n.output if t],
            "attrs": {k: (v if not isinstance(v, onnx.AttributeProto) else "attr") for k, v in
                      ((a.name, onnx.helper.get_attribute_value(a)) for a in n.attribute)},
        })
    return [{"group": k, "nodes": v} for k, v in groups.items()]
