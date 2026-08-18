"""Text dump of a model's data flow, written for an LLM reader rather than a human eye.

No diagrams, no indentation games: every fact an ASCII graph would encode visually is written out as a
token instead, so nothing depends on spatial reasoning. Each layer gets its real output shape from a
forward pass, its consumers, and its role in the graph (skip source, fusion point, detect input).

Usage:
    python model_flow.py yolo26s-rep-fpn-rep-sres-strip-dualsa.yaml        # a cfg name or path
    python model_flow.py runs/x/weights/best.pt --imgsz 640 --detail 24    # + full repr of layer 24
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from ultralytics import YOLO


def _norm_from(f) -> list[int]:
    """Normalize a layer's `.f` (int or list, -1 = previous layer) to absolute indices."""
    return [f] if isinstance(f, int) else list(f)


def flow(model, imgsz: int = 640, detail: tuple[int, ...] = ()) -> str:
    """Return a text description of `model`'s layer graph, shapes and data flow.

    Args:
        model (nn.Module): A built detection model (`YOLO(...).model`).
        imgsz (int): Square input size used for the shape-probing forward pass.
        detail (tuple[int, ...]): Layer indices to additionally print in full `repr` form.

    Returns:
        (str): The report.
    """
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

    # Reverse edges: who reads each layer
    consumers: dict[int, list[int]] = {i: [] for i in range(len(layers))}
    srcs: dict[int, list[int]] = {}
    for i, m in enumerate(layers):
        srcs[i] = [i - 1 if f == -1 else f for f in _norm_from(m.f)]  # -1 at layer 0 becomes -1 = the image
        for s in srcs[i]:
            if s in consumers:
                consumers[s].append(i)

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
    a = ap.parse_args()
    print(flow(YOLO(a.model).model, a.imgsz, tuple(a.detail)))
