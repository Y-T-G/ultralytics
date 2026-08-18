"""Text dump of a model's data flow, written for an LLM reader rather than a human eye.

No diagrams, no indentation games: every fact an ASCII graph would encode visually is written out as a
token instead, so nothing depends on spatial reasoning. Each layer gets its real output shape from a
forward pass, its consumers, and its role in the graph (skip source, fusion point, detect input).

Usage:
    python model_flow.py yolo26s-rep-fpn-rep-sres-strip-dualsa.yaml        # a cfg name or path
    python model_flow.py runs/x/weights/best.pt --imgsz 640 --detail 24    # + full repr of layer 24
    python model_flow.py a.yaml --diff b.yaml                              # what changed between two models
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from ultralytics import YOLO


def _norm_from(f) -> list[int]:
    """Normalize a layer's `.f` (int or list, -1 = previous layer) to absolute indices."""
    return [f] if isinstance(f, int) else list(f)


def _scan(model, imgsz: int = 640) -> dict:
    """Run one forward pass and return per-layer shapes, sources and consumers."""
    layers = list(model.model)
    shapes: dict[int, tuple] = {}
    hooks = [
        m.register_forward_hook(lambda mod, i, o, k=idx: shapes.__setitem__(k, tuple(o.shape) if torch.is_tensor(o) else None))
        for idx, m in enumerate(layers)
    ]
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        model(torch.zeros(1, 3, imgsz, imgsz, device=next(model.parameters()).device))
    for h in hooks:
        h.remove()
    model.train(was_training)

    consumers: dict[int, list[int]] = {i: [] for i in range(len(layers))}
    srcs: dict[int, list[int]] = {}
    for i, m in enumerate(layers):
        srcs[i] = [i - 1 if f == -1 else f for f in _norm_from(m.f)]  # -1 at layer 0 becomes -1 = the image
        for src in srcs[i]:
            if src in consumers:
                consumers[src].append(i)
    return dict(layers=layers, shapes=shapes, srcs=srcs, consumers=consumers)


def _sig(sc: dict, i: int) -> tuple:
    """Alignment key for a layer: type, output shape and parameter count."""
    sh = sc["shapes"].get(i)
    return (type(sc["layers"][i]).__name__, sh[1:] if sh else None, sc["layers"][i].np)


def diff(model_a, model_b, imgsz: int = 640, show_all: bool = False) -> str:
    """Return a text diff of two models' layer graphs, aligned so inserted layers do not misalign the rest.

    Args:
        model_a (nn.Module): Reference model.
        model_b (nn.Module): Model to compare against the reference.
        imgsz (int): Square input size used for the shape-probing forward passes.
        show_all (bool): Print matched layers too, not only differing ones.

    Returns:
        (str): The diff report.
    """
    import difflib

    A, B = _scan(model_a, imgsz), _scan(model_b, imgsz)
    sa = [_sig(A, i) for i in range(len(A["layers"]))]
    sb = [_sig(B, i) for i in range(len(B["layers"]))]
    ops = difflib.SequenceMatcher(a=sa, b=sb, autojunk=False).get_opcodes()

    b2a: dict[int, int] = {}  # B index -> A index, for layers that survived unchanged
    rows, changed = [], 0
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            for k in range(i2 - i1):
                b2a[j1 + k] = i1 + k
                if show_all:
                    rows.append(f"  =    A{i1 + k:>3} B{j1 + k:>3}  {sa[i1 + k][0]}")
        elif tag == "replace":
            changed += max(i2 - i1, j2 - j1)
            for k in range(max(i2 - i1, j2 - j1)):
                ia, jb = i1 + k, j1 + k
                la = f"A{ia:>3} {sa[ia][0]} {sa[ia][1]} {sa[ia][2]:,}p" if ia < i2 else "A  -"
                lb = f"B{jb:>3} {sb[jb][0]} {sb[jb][1]} {sb[jb][2]:,}p" if jb < j2 else "B  -"
                rows.append(f"  CHG  {la}   ->   {lb}")
        elif tag == "delete":
            changed += i2 - i1
            rows.extend(f"  DEL  A{i:>3} {sa[i][0]} {sa[i][1]} {sa[i][2]:,}p" for i in range(i1, i2))
        elif tag == "insert":
            changed += j2 - j1
            rows.extend(f"  ADD  B{j:>3} {sb[j][0]} {sb[j][1]} {sb[j][2]:,}p" for j in range(j1, j2))

    na = sum(p.numel() for p in model_a.parameters())
    nb = sum(p.numel() for p in model_b.parameters())
    out = [
        f"A = {getattr(model_a, 'yaml_file', None) or model_a.yaml.get('yaml_file', '?')}  "
        f"layers={len(A['layers'])} params={na:,}",
        f"B = {getattr(model_b, 'yaml_file', None) or model_b.yaml.get('yaml_file', '?')}  "
        f"layers={len(B['layers'])} params={nb:,}  ({nb - na:+,} params, {len(B['layers']) - len(A['layers']):+d} layers)",
        "",
        f"LAYERS  ({changed} differ, {len(b2a)} identical; identical rows hidden unless --all)",
    ]
    out += rows or ["  (no layer differs in type, shape or parameter count)"]

    # Wiring can differ while every layer signature matches (e.g. a residual re-pointed to another source).
    out += ["", "WIRING  (sources in A's index space; a plain number is a layer both models share, `Bn` is a B-only layer)"]
    wire = []
    for jb, ia in sorted(b2a.items()):
        mapped = ["IMG" if x < 0 else b2a.get(x, f"B{x}") for x in B["srcs"][jb]]
        ref = ["IMG" if x < 0 else x for x in A["srcs"][ia]]
        if mapped != ref:
            name = type(B["layers"][jb]).__name__
            wire.append(f"  {name} A{ia}/B{jb}: reads {ref} -> {mapped}")
    out += wire or ["  (identical for every matched layer)"]

    out += ["", "FAN-OUT  (consumer count per matched layer, A -> B)"]
    fan = []
    for jb, ia in sorted(b2a.items()):
        ca, cb = A["consumers"][ia], [b2a.get(x, f"B{x}") for x in B["consumers"][jb]]
        if ca != cb:
            fan.append(f"  A{ia}/B{jb} {type(B['layers'][jb]).__name__}: {ca} -> {cb}")
    out += fan or ["  (identical for every matched layer)"]

    da = [f"{x}" for x in A["srcs"][len(A["layers"]) - 1]]
    db = [f"{b2a.get(x, f'B{x}')}" for x in B["srcs"][len(B["layers"]) - 1]]
    out += ["", f"DETECT INPUTS  A {da}  ->  B {db}"]
    return "\n".join(out)


def flow(model, imgsz: int = 640, detail: tuple[int, ...] = ()) -> str:
    """Return a text description of `model`'s layer graph, shapes and data flow.

    Args:
        model (nn.Module): A built detection model (`YOLO(...).model`).
        imgsz (int): Square input size used for the shape-probing forward pass.
        detail (tuple[int, ...]): Layer indices to additionally print in full `repr` form.

    Returns:
        (str): The report.
    """
    sc = _scan(model, imgsz)
    layers, shapes, srcs, consumers = sc["layers"], sc["shapes"], sc["srcs"], sc["consumers"]

    detect_i = len(layers) - 1
    detect_inputs = srcs[detect_i]
    out = []
    n_p = sum(p.numel() for p in model.parameters())
    name = getattr(model, "yaml_file", None) or model.yaml.get("yaml_file", "?")
    out.append(f"MODEL {name}  layers={len(layers)}  params={n_p:,}  input=1x3x{imgsz}x{imgsz}")
    out.append("")
    out.append("LAYERS  (stride = imgsz / H; 'reads' are absolute layer indices; 'feeds' lists every consumer)")
    out.append(f"{'idx':>3}  {'module':22s} {'reads':14s} {'out (C,H,W)':18s} {'str':>4} {'params':>10}  feeds")
    for i, m in enumerate(layers):
        sh = shapes.get(i)
        cshw = f"{sh[1]},{sh[2]},{sh[3]}" if sh else "-"
        stride = f"/{imgsz // sh[2]}" if sh and sh[2] else "-"
        fed = consumers[i] or (["DETECT"] if i == detect_i else ["(unused)"])
        reads = "[IMG]" if srcs[i] == [-1] else str(srcs[i])
        out.append(f"{i:>3}  {type(m).__name__:22s} {reads:14s} {cshw:18s} {stride:>4} {m.np:>10,}  {fed}")

    out.append("")
    out.append("FAN-OUT  (layers read by more than one consumer: these are the skip/fusion sources)")
    for i, c in consumers.items():
        if len(c) > 1:
            out.append(f"  {i} ({type(layers[i]).__name__}) -> {c}")

    out.append("")
    out.append("MERGE POINTS  (layers with more than one input)")
    for i, m in enumerate(layers):
        if len(srcs[i]) > 1:
            name = type(m).__name__
            kind = (
                "concat (channels add)"
                if name == "Concat"
                else "multi-level head input (one tower per level)"
                if "Detect" in name or name in {"Segment", "Pose", "OBB"}
                else "elementwise (shapes must match)"
            )
            ins = ", ".join(f"{s}:{shapes[s][1]}ch" if shapes.get(s) else str(s) for s in srcs[i])
            out.append(f"  {i} {type(m).__name__:12s} <- [{ins}]  {kind}")

    out.append("")
    out.append("DETECT INPUTS  (each traced back to the backbone, following the first listed source)")
    for j, d in enumerate(detect_inputs):
        chain, cur, seen = [], d, set()
        while cur is not None and cur >= 0 and cur not in seen:
            seen.add(cur)
            chain.append(f"{cur}:{type(layers[cur]).__name__}")
            nxt = srcs.get(cur) or [None]
            cur = nxt[0]
        chain.append("IMG")
        sh = shapes.get(d)
        lvl = f"P{imgsz // sh[2]}".replace("P8", "P3/8").replace("P16", "P4/16").replace("P32", "P5/32") if sh else "?"
        out.append(f"  input {j} = layer {d} ({lvl}, {sh[1]}ch): " + " <- ".join(chain))

    scalars = [
        (n, tuple(p.detach().flatten()[:4].tolist()))
        for n, p in model.named_parameters()
        if p.numel() <= 4 and p.ndim <= 1
    ]
    if scalars:
        out.append("")
        out.append("SCALAR PARAMS  (learnable gates/mixers; read these on a trained checkpoint to see what training chose)")
        for n, v in scalars:
            out.append(f"  {n} = {v}")

    if detail:
        out.append("")
        out.append("DETAIL")
        for i in detail:
            out.append(f"--- layer {i} ---")
            out.append(repr(layers[i]))
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--detail", type=int, nargs="*", default=())
    ap.add_argument("--diff", help="second model: report what changed from `model` to it")
    ap.add_argument("--all", action="store_true", help="diff mode: also list identical layers")
    a = ap.parse_args()
    if a.diff:
        print(diff(YOLO(a.model).model, YOLO(a.diff).model, a.imgsz, a.all))
    else:
        print(flow(YOLO(a.model).model, a.imgsz, tuple(a.detail)))
