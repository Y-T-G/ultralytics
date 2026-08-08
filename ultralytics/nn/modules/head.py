# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model head modules."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils import NOT_MACOS14
from ultralytics.utils.tal import dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import TORCH_1_11, fuse_conv_and_bn, smart_inference_mode

from .block import (
    DFL,
    SAVPE,
    BNContrastiveHead,
    ContrastiveHead,
    Proto,
    Proto26,
    RealNVP,
    Residual,
    SwiGLUFFN,
    SpatialSuppressionGate,
)
from .conv import Conv, DilatedReparamDW, DWConv, GCAttn, RepConv, RepDWConv, SoftRFMix, SoftRFMixFast
from .dfine_transformer import DeimTransformerDecoder, DeimTransformerDecoderLayer, Integral
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init

__all__ = (
    "OBB",
    "Classify",
    "Detect",
    "DetectNorm",
    "Pose",
    "RTDETRDecoder",
    "Segment",
    "YOLOEDetect",
    "YOLOESegment",
    "v10Detect",
)


class Detect(nn.Module):
    """YOLO Detect head for object detection models.

    This class implements the detection head used in YOLO models for predicting bounding boxes and class probabilities.
    It supports both training and inference modes, with optional end-to-end detection capabilities.

    Attributes:
        dynamic (bool): Force grid reconstruction.
        export (bool): Export mode flag.
        format (str): Export format.
        end2end (bool): End-to-end detection mode.
        max_det (int): Maximum detections per image.
        shape (tuple): Input shape.
        anchors (torch.Tensor): Anchor points.
        strides (torch.Tensor): Feature map strides.
        legacy (bool): Backward compatibility for v3/v5/v8/v9 models.
        xyxy (bool): Output format, xyxy or xywh.
        nc (int): Number of classes.
        nl (int): Number of detection layers.
        reg_max (int): DFL channels.
        no (int): Number of outputs per anchor.
        stride (torch.Tensor): Strides computed during build.
        cv2 (nn.ModuleList): Convolution layers for box regression.
        cv3 (nn.ModuleList): Convolution layers for classification.
        dfl (nn.Module): Distribution Focal Loss layer.
        one2one_cv2 (nn.ModuleList): One-to-one convolution layers for box regression.
        one2one_cv3 (nn.ModuleList): One-to-one convolution layers for classification.

    Methods:
        forward: Perform forward pass and return predictions.
        forward_end2end: Perform forward pass for end-to-end detection.
        bias_init: Initialize detection head biases.
        decode_bboxes: Decode bounding boxes from predictions.
        postprocess: Post-process model predictions.

    Examples:
        Create a detection head for 80 classes
        >>> detect = Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = detect(x)
    """

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models
    xyxy = False  # xyxy or xywh output
    suppress = False
    rep_head = False  # use RepConv in head (fuses at inference, zero overhead)
    no_detach = False  # allow one2one gradients to flow back to backbone
    o2o_grad_scale = 0.0  # gradient scale for one2one features (0=detach, 1=full gradient)
    o2o_residual_head = False  # lightweight residual cls head for one2one (share boxes with o2m)
    o2o_dilated = False  # multi-dilation RepDWConv in one2one cls branch (fuses to DW 5x5, wider RF for duplicate suppression)
    head_dilated = False  # multi-dilation RepDWConv in BOTH o2m and o2o cls branches (swapped before deepcopy)
    head_repdw = False  # non-dilated RepDWConv (pure rep DW 3x3, unchanged inference graph) in both cls branches
    peak_pool_k = 0  # end2end inference-only local-max suppression kernel (0=off, odd int like 3/5)
    o2o_maxfilter = 0  # 3D max filtering (DeFCN, CVPR21) on o2o scores, trained end-to-end (0=off, odd int)
    o2o_pss = False  # PSS (TMM 2023): deploy the o2m branch gated by a class-agnostic one-positive selector
    o2o_res_feat = False  # feed the residual head the penultimate cls-tower feature instead of raw neck features
    keep_one2many = False  # keep one2many head through fuse() to allow dual-head validation (set by the validator)
    fixed_c3 = 0  # fix cls branch hidden width regardless of nc (0 = auto: min(nc, 100)); keeps head shape stable across datasets for finetuning
    head_depth = 2  # conv stages per tower before the predictor (2 = stock); depth instead of width
    head_c2 = 0  # override box branch hidden width (0 = auto: max(16, ch[0] // 4, reg_max * 4))
    o2o_share_towers = False  # o2o shares the o2m tower convs, only the final 1x1 predictors are separate
    dfl_bias_prior = False  # init box predictor bias as a peaked DFL distribution around bin 2 (decoder anchor prior)
    box_dilated = 0  # UniRepLKNet dilated-reparam DW kernel in the box tower (0=off, e.g. 9); fuses to 1x1 + DW KxK
    box_rf_mix = ()  # per-location soft receptive-field mixing in the MAIN box tower (no FDR refine tower needed)
    box_dilated_levels = (1, 2)  # levels the dilated-reparam box tower applies to (P4/P5 by default)

    def __init__(self, nc: int = 80, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize the YOLO detection layer with specified number of classes and channels.

        Args:
            nc (int): Number of classes.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = reg_max  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2 = self.head_c2 or max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], (self.fixed_c3 or min(self.nc, 100)))  # channels
        if self.rep_head:
            self.cv2 = nn.ModuleList(
                nn.Sequential(RepConv(x, c2, 3, bn=(x == c2)), RepConv(c2, c2, 3, bn=True), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
            )
        else:
            self.cv2 = nn.ModuleList(
                nn.Sequential(
                    Conv(x, c2, 3),
                    *(Conv(c2, c2, 3) for _ in range(self.head_depth - 1)),
                    nn.Conv2d(c2, 4 * self.reg_max, 1),
                )
                for x in ch
            )
        if self.rep_head:
            self.cv3 = (
                nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
                if self.legacy
                else nn.ModuleList(
                    nn.Sequential(
                        nn.Sequential(RepConv(x, x, 3, g=x, bn=True), Conv(x, c3, 1)),
                        nn.Sequential(RepConv(c3, c3, 3, g=c3, bn=True), Conv(c3, c3, 1)),
                        nn.Conv2d(c3, self.nc, 1),
                    )
                    for x in ch
                )
            )
        else:
            self.cv3 = (
                nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
                if self.legacy
                else nn.ModuleList(
                    nn.Sequential(
                        nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                        *(nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)) for _ in range(self.head_depth - 1)),
                        nn.Conv2d(c3, self.nc, 1),
                    )
                    for x in ch
                )
            )
        if self.box_dilated:
            for i in self.box_dilated_levels:
                self.cv2[i][0] = nn.Sequential(Conv(ch[i], c2, 1), DilatedReparamDW(c2, self.box_dilated))
        if self.box_rf_mix:
            for m in self.cv2:
                m[0] = nn.Sequential(m[0], SoftRFMixFast(c2, len(self.box_rf_mix), cascade=False))
        if (self.head_dilated or self.head_repdw) and not self.legacy and not self.rep_head:
            dilated = bool(self.head_dilated)
            for m, x in zip(self.cv3, ch):
                m[0][0] = RepDWConv(x, dilated=dilated)
                m[1][0] = RepDWConv(c3, dilated=dilated)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if end2end:
            if self.o2o_residual_head or self.o2o_pss:
                # Share the o2m box head; the o2o path is either a per-class residual or a PSS gate
                self.one2one_cv2 = None  # reuse o2m boxes
                self.one2one_cv3 = None  # handled via residual / selector
                if self.o2o_pss:
                    self.o2o_sel = nn.ModuleList()
                    for x in ch:
                        conv = nn.Conv2d(c2, 1, 1)
                        nn.init.zeros_(conv.weight)
                        nn.init.constant_(conv.bias, 0.0)  # sigmoid(0)=0.5: keeps the selector gradient alive (bias 4 is ~28x weaker)
                        self.o2o_sel.append(nn.Sequential(Conv(x, c2, 1), DWConv(c2, c2, 3), conv))
                else:
                    self.o2o_cls_res = nn.ModuleList()
                    for x in ch:
                        # o2o_res_feat: read the o2m cls tower's penultimate feature (c3 wide), so
                        # z_o2o = z_o2m + W_r f is an independent final classifier over a strong feature
                        cin = c3 if self.o2o_res_feat else x
                        conv = nn.Conv2d(cin, self.nc, 1, bias=True)
                        nn.init.zeros_(conv.weight)
                        nn.init.zeros_(conv.bias)
                        self.o2o_cls_res.append(conv if self.o2o_res_feat else nn.Sequential(DWConv(x, x, 3), conv))
            elif self.o2o_share_towers:
                # Shared feature towers, split predictors: both losses train the same tower convs
                # (the only channel for rich o2m supervision to reach the deployed branch when the
                # trunk is frozen); only the final 1x1 predictions are branch-specific.
                self.one2one_cv2 = nn.ModuleList(nn.Sequential(*m[:-1], copy.deepcopy(m[-1])) for m in self.cv2)
                self.one2one_cv3 = nn.ModuleList(nn.Sequential(*m[:-1], copy.deepcopy(m[-1])) for m in self.cv3)
            else:
                self.one2one_cv2 = copy.deepcopy(self.cv2)
                self.one2one_cv3 = copy.deepcopy(self.cv3)
                if self.o2o_dilated and not self.legacy:
                    for m, x in zip(self.one2one_cv3, ch):
                        m[0][0] = RepDWConv(x)
                        m[1][0] = RepDWConv(c3)
            if self.suppress:
                self.o2o_suppress = nn.ModuleList(SpatialSuppressionGate(nc, k=5) for _ in range(self.nl))
            if self.o2o_maxfilter:
                self.mf_beta = nn.Parameter(torch.full((self.nl,), -3.0))

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v5/v5/v8/v9/11 backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        if hasattr(self, "o2o_suppress"):
            return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, suppress=self.o2o_suppress)
        else:
            return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3)

    @property
    def end2end(self):
        """Checks if the model has one2one for v5/v5/v8/v9/11 backward compatibility."""
        return getattr(self, "_end2end", True) and hasattr(self, "one2one")

    @end2end.setter
    def end2end(self, value):
        """Override the end-to-end detection mode."""
        self._end2end = value

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module = None,
        cls_head: torch.nn.Module = None,
        suppress: torch.nn.Module = None,
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if box_head is None or cls_head is None:  # for fused inference
            return dict()
        bs = x[0].shape[0]  # batch size
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        cls_feats = []
        for i in range(self.nl):
            c = cls_head[i](x[i])  # (B, nc, H, W)
            if suppress is not None:
                c = suppress[i](c)  # spatial gating
            cls_feats.append(c.view(bs, self.nc, -1))
        scores = torch.cat(cls_feats, dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)

    def forward(
        self, x: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            if self.o2o_pss:
                # Deploy the o2m branch, gated by a class-agnostic selector: p = sigmoid(cls) * sigmoid(sel).
                # Base logits/boxes are detached, so the selector loss only trains the selector.
                bs = x[0].shape[0]
                sel = torch.cat([self.o2o_sel[i](x[i].detach()).view(bs, 1, -1) for i in range(self.nl)], dim=-1)
                one2one = dict(boxes=preds["boxes"].detach(), scores=self._product_logit(preds["scores"].detach(), sel), feats=x)
            elif self.o2o_residual_head:
                # Shared boxes from o2m (detached), cls = o2m_cls + lightweight residual
                bs = x[0].shape[0]
                o2o_scores = preds["scores"].detach()  # (B, nc, A) from o2m
                cls_res = []
                for i in range(self.nl):
                    f = self.cv3[i][:-1](x[i]).detach() if self.o2o_res_feat else x[i].detach()
                    cls_res.append(self.o2o_cls_res[i](f).view(bs, self.nc, -1))
                o2o_scores = o2o_scores + torch.cat(cls_res, dim=-1)
                one2one = dict(boxes=preds["boxes"].detach(), scores=o2o_scores, feats=x)
            else:
                if self.no_detach:
                    x_o2o = x
                elif self.o2o_grad_scale > 0:
                    s = self.o2o_grad_scale
                    x_o2o = [xi * s + xi.detach() * (1 - s) for xi in x]
                else:
                    x_o2o = [xi.detach() for xi in x]
                one2one = self.forward_head(x_o2o, **self.one2one)
            if self.o2o_maxfilter:
                one2one = {**one2one, "scores": self._max_filter(one2one["scores"], one2one["feats"])}
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.

        Args:
            x (dict[str, torch.Tensor]): List of feature maps from different detection layers.

        Returns:
            (torch.Tensor): Concatenated tensor of decoded bounding boxes and class probabilities.
        """
        # Inference path
        if self.end2end and self.peak_pool_k > 0:
            x = {**x, "scores": self._peak_suppress(x["scores"], x["feats"])}
        dbox = self._get_decode_boxes(x)
        return torch.cat((dbox, x["scores"].sigmoid()), 1)

    @staticmethod
    def _product_logit(z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Logit whose sigmoid equals sigmoid(z) * sigmoid(s), so loss/assigner/inference need no changes."""
        z, s = z.float().clamp(-15, 15), s.float().clamp(-15, 15)
        return -torch.log((1 + torch.exp(-z)) * (1 + torch.exp(-s)) - 1 + 1e-9)

    def _max_filter(self, scores: torch.Tensor, feats: list[torch.Tensor]) -> torch.Tensor:
        """3D max filtering (DeFCN, CVPR 2021): suppress scores beaten by a neighbour in space or scale.

        Unlike NMS this is a fixed local+cross-level max, so it is one max-pool plus fixed-scale resizes at
        export. Trained end-to-end, so the head learns to cooperate with it. beta is zero-init: identity at
        step 0, and a per-level learnable strength thereafter.
        """
        ks = self.o2o_maxfilter if isinstance(self.o2o_maxfilter, (list, tuple)) else [self.o2o_maxfilter] * self.nl
        bs, nc = scores.shape[:2]
        maps, offset = [], 0
        for f in feats:
            h, w = f.shape[2:]
            maps.append(scores[..., offset : offset + h * w].view(bs, nc, h, w))
            offset += h * w
        beta = self.mf_beta.sigmoid()  # keep the suppression strength in (0, 1); raw init -3 -> ~0.05
        out = []
        for i, m in enumerate(maps):
            tube = m  # align raw neighbours first, then pool once at the target resolution
            if i > 0:  # finer level, pooled down
                tube = torch.maximum(tube, F.max_pool2d(maps[i - 1], 2, 2))
            if i < self.nl - 1:  # coarser level, upsampled
                tube = torch.maximum(tube, F.interpolate(maps[i + 1], scale_factor=2, mode="nearest"))
            best = F.max_pool2d(tube, ks[i], 1, ks[i] // 2)
            out.append((m - beta[i].view(1, 1, 1, 1) * (best - m).clamp(min=0)).view(bs, nc, -1))
        return torch.cat(out, dim=-1)

    def _peak_suppress(self, scores: torch.Tensor, feats: list[torch.Tensor]) -> torch.Tensor:
        """Zero non-local-max cls logits per level (NMS-free inference dedup)."""
        k = self.peak_pool_k
        bs, nc = scores.shape[:2]
        out, offset = [], 0
        for f in feats:
            h, w = f.shape[2:]
            n = h * w
            s = scores[..., offset : offset + n].view(bs, nc, h, w)
            peak = F.max_pool2d(s, k, stride=1, padding=k // 2)
            s = torch.where(s == peak, s, torch.full_like(s, -10.0))
            out.append(s.view(bs, nc, n))
            offset += n
        return torch.cat(out, dim=-1)

    def _get_decode_boxes(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get decoded boxes based on anchors and strides."""
        shape = x["feats"][0].shape  # BCHW
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape

        dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
        return dbox

    def inference_one2many(self, preds: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode the one2many branch in standard NMS-ready (xywh) format for dual-head validation.

        Reuses the one2many predictions already produced by the shared backbone/neck forward pass (no recompute),
        so the regular (non NMS-free) head can be evaluated alongside the end2end one2one head.

        Args:
            preds (dict[str, torch.Tensor]): The dict returned by forward() at inference, containing the "one2many" key.

        Returns:
            (torch.Tensor): Decoded predictions with shape (B, 4 + nc, num_anchors), boxes in xywh for NMS.
        """
        x = preds["one2many"]
        shape = x["feats"][0].shape  # BCHW
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape
        dbox = dist2bbox(self.dfl(x["boxes"]), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        return torch.cat((dbox, x["scores"].sigmoid()), 1)

    def _box_bias(self, bias):
        """Fill a box predictor bias: constant 2.0, or a peaked DFL prior around bin 2."""
        if self.dfl_bias_prior:
            j = torch.arange(self.reg_max, dtype=bias.dtype, device=bias.device)
            bias.data[:] = (-(j - 2.0).pow(2) / 2.0).repeat(4)
        else:
            bias.data[:] = 2.0

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):  # from
            self._box_bias(a[-1].bias)  # box
            b[-1].bias.data[: self.nc] = math.log(
                5 / self.nc / (640 / self.stride[i]) ** 2
            )  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end and not self.o2o_residual_head and not self.o2o_pss:
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):  # from
                self._box_bias(a[-1].bias)  # box
                b[-1].bias.data[: self.nc] = math.log(
                    5 / self.nc / (640 / self.stride[i]) ** 2
                )  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        """Decode bounding boxes from predictions."""
        return dist2bbox(
            bboxes,
            anchors,
            xywh=xywh and not self.end2end and not self.xyxy,
            dim=1,
        )

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        boxes, scores = preds.split([4, self.nc], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        return torch.cat([boxes, scores, conf], dim=-1)

    def get_topk_index(self, scores: torch.Tensor, max_det: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get top-k indices from scores.

        Args:
            scores (torch.Tensor): Scores tensor with shape (batch_size, num_anchors, num_classes).
            max_det (int): Maximum detections per image.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): Top scores, class indices, and filtered indices.
        """
        batch_size, anchors, nc = scores.shape  # i.e. shape(16,8400,84)
        # Use max_det directly during export for TensorRT compatibility (requires k to be constant),
        # otherwise use min(max_det, anchors) for safety with small inputs during Python inference
        k = max_det if self.export else min(max_det, anchors)
        ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
        scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(k)
        idx = ori_index[torch.arange(batch_size)[..., None], index // nc]  # original index
        return scores[..., None], (index % nc)[..., None].float(), idx

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        if not self.o2o_residual_head and not self.o2o_pss and not self.keep_one2many:
            self.cv2 = self.cv3 = None
        # residual head keeps cv2+cv3 since o2o inference needs o2m's box+cls as base
        # keep_one2many keeps cv2+cv3 so dual-head validation can score both heads (still conv-bn fused)


class DetectNorm(Detect):
    """YOLO detection head with anchor-relative normalized box predictions.

    Predicts ltrb offsets from each anchor point using a shifted sigmoid, giving outputs in [-0.2, 1.3] space.
    Formula: sigmoid(x) * 1.5 - 0.2, which allows slight negative offsets and offsets beyond 1.0.
    Decoding: xyxy_norm = dist2bbox(shifted_sigmoid(raw), anchor_norm), then × imgsz for pixel coords.
    """

    def __init__(self, nc: int = 80, reg_max: int = 1, end2end: bool = False, ch: tuple = ()):
        """Initialize DetectNorm, forcing reg_max=1 (4-channel ltrb output, no DFL)."""
        super().__init__(nc=nc, reg_max=1, end2end=end2end, ch=ch)

    def _get_decode_boxes(self, x: dict) -> torch.Tensor:
        """Decode shifted-sigmoid ltrb offsets relative to normalized anchor points → pixel coords."""
        shape = x["feats"][0].shape  # BCHW
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape
        h = shape[2] * self.stride[0]
        w = shape[3] * self.stride[0]
        # Use a single isotropic scale so training (square 640×640) and rect-val (non-square) are consistent.
        # Normalizing x and y by different W/H would cause a train/val mismatlch when H ≠ W.
        scale = max(h, w)
        anchor_norm = self.anchors * self.strides / scale  # (2, A) in [0, 1]
        # Decode ltrb: x["boxes"] is (B, 4, A) raw logits → sigmoid * 1.5 - 0.2 → clamped to [0, 1.3]
        # Clamped to min=0 to avoid invalid boxes; gradient still flows since 0 ≈ sigmoid(-1.9), far from saturation.
        pred_ltrb = (x["boxes"].sigmoid() * 1.5 - 0.2).clamp(min=0)  # (B, 4, A)
        x1y1 = anchor_norm.unsqueeze(0) - pred_ltrb[:, :2]  # (B, 2, A)
        x2y2 = anchor_norm.unsqueeze(0) + pred_ltrb[:, 2:]  # (B, 2, A)
        boxes_norm = torch.cat([x1y1, x2y2], dim=1)  # (B, 4, A) normalized by scale
        return boxes_norm * scale  # back to pixel coords (same space as actual image pixels)

    def bias_init(self) -> None:
        """Initialize box biases so initial ltrb offsets match the baseline's 2-grid-unit initial size."""
        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):
            # sigmoid(bias) * 1.5 - 0.2 = ltrb0  →  sigmoid(bias) = (ltrb0 + 0.2) / 1.5  →  bias = logit(...)
            ltrb0 = 2.0 * self.stride[i].item() / 640.0
            s = (ltrb0 + 0.2) / 1.5
            a[-1].bias.data[:] = math.log(s / (1.0 - s))
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):
                ltrb0 = 2.0 * self.stride[i].item() / 640.0
                s = (ltrb0 + 0.2) / 1.5
                a[-1].bias.data[:] = math.log(s / (1.0 - s))
                b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)


class DetectBoxContext(Detect):
    """Detection head where one2one cls branch uses shared regression features as context.

    Instead of duplicating regression branch for one2one, this head:
    1. Shares regression branch between one2many and one2one (no weight duplication)
    2. Extracts intermediate features from regression stem as spatial context
    3. One2one cls branch receives [FPN_features, box_features] for better duplicate suppression
    """

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        """Initialize with shared regression and context-aware o2o classification."""
        # Init parent without end2end to skip duplicate head creation
        super().__init__(nc, reg_max, end2end=False, ch=ch)

        if end2end:
            c2 = max((16, ch[0] // 4, self.reg_max * 4))
            c3 = max(ch[0], (self.fixed_c3 or min(self.nc, 100)))

            # Shared regression reference (no duplication)
            self.one2one_cv2 = self.cv2

            # O2O cls branch: input channels = FPN (x) + box intermediate (c2)
            if self.rep_head:
                self.one2one_cv3 = nn.ModuleList(
                    nn.Sequential(
                        nn.Sequential(RepConv(x + c2, x + c2, 3, g=x + c2, bn=True), Conv(x + c2, c3, 1)),
                        nn.Sequential(RepConv(c3, c3, 3, g=c3, bn=True), Conv(c3, c3, 1)),
                        nn.Conv2d(c3, self.nc, 1),
                    )
                    for x in ch
                )
            else:
                self.one2one_cv3 = nn.ModuleList(
                    nn.Sequential(
                        nn.Sequential(DWConv(x + c2, x + c2, 3), Conv(x + c2, c3, 1)),
                        nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                        nn.Conv2d(c3, self.nc, 1),
                    )
                    for x in ch
                )

    def forward(self, x):
        """Forward with shared regression and context-aware o2o classification."""
        bs = x[0].shape[0]

        # Run regression branch once, capture intermediate features
        box_feats, box_preds = [], []
        for i in range(self.nl):
            feat = self.cv2[i][:2](x[i])  # box stem (2 conv layers)
            pred = self.cv2[i][2](feat)  # box final (1x1 conv)
            box_feats.append(feat)
            box_preds.append(pred.view(bs, 4 * self.reg_max, -1))
        boxes = torch.cat(box_preds, dim=-1)

        # O2M classification
        if self.cv3 is not None:
            cls_preds = [self.cv3[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)]
            scores = torch.cat(cls_preds, dim=-1)
            preds = dict(boxes=boxes, scores=scores, feats=x)

        if self.end2end:
            # O2O classification with box context (all detached)
            o2o_cls = []
            for i in range(self.nl):
                ctx = torch.cat([x[i].detach(), box_feats[i].detach()], dim=1)
                c = self.one2one_cv3[i](ctx)
                o2o_cls.append(c.view(bs, self.nc, -1))
            o2o_scores = torch.cat(o2o_cls, dim=-1)
            one2one = dict(boxes=boxes.detach(), scores=o2o_scores, feats=x)

            if self.cv3 is not None:
                preds = {"one2many": preds, "one2one": one2one}
            else:
                preds = {"one2many": dict(), "one2one": one2one}

        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def bias_init(self):
        """Initialize biases for regression and both cls branches."""
        for i, (a, b) in enumerate(zip(self.cv2, self.cv3)):
            a[-1].bias.data[:] = 2.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, b in enumerate(self.one2one_cv3):
                b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)

    def fuse(self):
        """Remove o2m cls head for inference; keep shared cv2 for o2o box predictions."""
        self.cv3 = None


class DetectSharedReg(Detect):
    """Shared regression branch between o2m and o2o; o2o cls is a standard (non-context) head.

    O2o reuses o2m's boxes via detach, so the o2o loss never updates the shared regression branch.
    O2o cls branch is a duplicated normal cls head (same input shape as o2m cv3).
    """

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        """Initialize with shared regression and standard o2o classification."""
        super().__init__(nc, reg_max, end2end=False, ch=ch)
        if end2end:
            self.one2one_cv2 = self.cv2  # shared reference, no duplication
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """Forward with shared regression; o2o boxes detached to block reg-branch grads from o2o."""
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            bs = x[0].shape[0]
            x_o2o = [xi.detach() for xi in x]
            o2o_scores = torch.cat(
                [self.one2one_cv3[i](x_o2o[i]).view(bs, self.nc, -1) for i in range(self.nl)],
                dim=-1,
            )
            one2one = dict(boxes=preds["boxes"].detach(), scores=o2o_scores, feats=x)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def fuse(self):
        """Remove o2m cls head for inference; keep shared cv2 for o2o box predictions."""
        self.cv3 = None


class DetectSharedFPN(Detect):
    """FPN-shared detection towers (RetinaNet/FCOS style): every level runs the same head weights.

    Only the first conv of each tower stays per-level, since it adapts the level's channel count to the
    tower width; all later convs and the final predictor are the same modules for P3/P4/P5. Sharing is by
    module identity, so parameters, BN stats and fused weights exist once and gradients from all levels
    accumulate into them.
    """

    per_level_bn = False  # keep each level's own BN (RetinaNet/FCOS style) and tie only the conv weights

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        """Initialize a standard head, then replace each level's post-stem convs with level 0's modules."""
        super().__init__(nc, reg_max, end2end=end2end, ch=ch)
        for tower in (self.cv2, self.cv3, getattr(self, "one2one_cv2", None), getattr(self, "one2one_cv3", None)):
            if tower is None:
                continue
            if self.per_level_bn:
                for i in range(1, self.nl):
                    for a, b in zip(tower[i][1:].modules(), tower[0][1:].modules()):
                        if isinstance(a, nn.Conv2d):
                            a.weight = b.weight
                            if a.bias is not None:
                                a.bias = b.bias
            else:
                tail = list(tower[0])[1:]
                for i in range(1, self.nl):
                    tower[i] = nn.Sequential(tower[i][0], *tail)

    def bias_init(self):
        """Initialize the shared predictors once; the cls prior uses the middle level's stride."""
        s = self.stride[self.nl // 2]
        heads = [self.one2many]
        if self.end2end and not self.o2o_residual_head and not self.o2o_pss:
            heads.append(self.one2one)
        for h in heads:
            self._box_bias(h["box_head"][0][-1].bias)
            h["cls_head"][0][-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)


class DetectSharedFPNBN(DetectSharedFPN):
    """FPN-shared head that keeps per-level BatchNorm: only the conv weights are tied across P3/P4/P5.

    RetinaNet and FCOS share head weights but not the normalization, because stride 8/16/32 features have
    different statistics. Convs alias one weight tensor here, so the optimizer and EMA must skip the
    duplicate aliases (handled by id-dedup in `build_optimizer` and `ModelEMA.update`).
    """

    per_level_bn = True


class DetectBoxContextFull(DetectBoxContext):
    """Same as DetectBoxContext but BOTH o2m and o2o cls branches receive box context.

    O2m cls uses non-detached box features (gradients flow back through box stem).
    O2o cls uses detached box features (no gradient interference with o2m regression).
    """

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        """Initialize with box-context cls branches for both o2m and o2o."""
        super().__init__(nc, reg_max, end2end=end2end, ch=ch)

        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], (self.fixed_c3 or min(self.nc, 100)))

        # Rebuild cv3 with box context input: x + c2 channels
        if self.rep_head:
            self.cv3 = (
                nn.ModuleList(
                    nn.Sequential(Conv(x + c2, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
                )
                if self.legacy
                else nn.ModuleList(
                    nn.Sequential(
                        nn.Sequential(RepConv(x + c2, x + c2, 3, g=x + c2, bn=True), Conv(x + c2, c3, 1)),
                        nn.Sequential(RepConv(c3, c3, 3, g=c3, bn=True), Conv(c3, c3, 1)),
                        nn.Conv2d(c3, self.nc, 1),
                    )
                    for x in ch
                )
            )
        else:
            self.cv3 = (
                nn.ModuleList(
                    nn.Sequential(Conv(x + c2, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
                )
                if self.legacy
                else nn.ModuleList(
                    nn.Sequential(
                        nn.Sequential(DWConv(x + c2, x + c2, 3), Conv(x + c2, c3, 1)),
                        nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                        nn.Conv2d(c3, self.nc, 1),
                    )
                    for x in ch
                )
            )

    def forward(self, x):
        """Forward with shared regression; both o2m and o2o cls use box context."""
        bs = x[0].shape[0]

        # Shared regression, capture intermediates
        box_feats, box_preds = [], []
        for i in range(self.nl):
            feat = self.cv2[i][:2](x[i])
            pred = self.cv2[i][2](feat)
            box_feats.append(feat)
            box_preds.append(pred.view(bs, 4 * self.reg_max, -1))
        boxes = torch.cat(box_preds, dim=-1)

        # O2M cls with box context (non-detached: grads flow through box stem)
        if self.cv3 is not None:
            cls_preds = []
            for i in range(self.nl):
                ctx = torch.cat([x[i], box_feats[i]], dim=1)
                cls_preds.append(self.cv3[i](ctx).view(bs, self.nc, -1))
            scores = torch.cat(cls_preds, dim=-1)
            preds = dict(boxes=boxes, scores=scores, feats=x)

        if self.end2end:
            # O2O cls with box context (detached)
            o2o_cls = []
            for i in range(self.nl):
                ctx = torch.cat([x[i].detach(), box_feats[i].detach()], dim=1)
                o2o_cls.append(self.one2one_cv3[i](ctx).view(bs, self.nc, -1))
            o2o_scores = torch.cat(o2o_cls, dim=-1)
            one2one = dict(boxes=boxes.detach(), scores=o2o_scores, feats=x)

            if self.cv3 is not None:
                preds = {"one2many": preds, "one2one": one2one}
            else:
                preds = {"one2many": dict(), "one2one": one2one}

        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)


class DetectBoxContextSep(Detect):
    """Like DetectBoxContext but with SEPARATE (duplicated) regression branch for o2o.

    O2m: standard cls (no context).
    O2o: own regression branch + cls uses o2o box features as context (detached).
    """

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        """Initialize with duplicated regression and context-aware o2o classification."""
        super().__init__(nc, reg_max, end2end=False, ch=ch)

        if end2end:
            c2 = max((16, ch[0] // 4, self.reg_max * 4))
            c3 = max(ch[0], (self.fixed_c3 or min(self.nc, 100)))

            # Separate o2o regression (deepcopy of cv2 structure)
            self.one2one_cv2 = copy.deepcopy(self.cv2)

            # O2O cls: input = FPN (x) + o2o box intermediate (c2)
            if self.rep_head:
                self.one2one_cv3 = nn.ModuleList(
                    nn.Sequential(
                        nn.Sequential(RepConv(x + c2, x + c2, 3, g=x + c2, bn=True), Conv(x + c2, c3, 1)),
                        nn.Sequential(RepConv(c3, c3, 3, g=c3, bn=True), Conv(c3, c3, 1)),
                        nn.Conv2d(c3, self.nc, 1),
                    )
                    for x in ch
                )
            else:
                self.one2one_cv3 = nn.ModuleList(
                    nn.Sequential(
                        nn.Sequential(DWConv(x + c2, x + c2, 3), Conv(x + c2, c3, 1)),
                        nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                        nn.Conv2d(c3, self.nc, 1),
                    )
                    for x in ch
                )

    def forward(self, x):
        """Forward with separate o2m/o2o regression; o2o cls uses o2o box context."""
        bs = x[0].shape[0]

        # O2M regression + cls (standard)
        if self.cv3 is not None:
            cls_preds = [self.cv3[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)]
            scores = torch.cat(cls_preds, dim=-1)
        if self.cv2 is not None:
            box_preds = [self.cv2[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)]
            boxes = torch.cat(box_preds, dim=-1)
        if self.cv3 is not None and self.cv2 is not None:
            preds = dict(boxes=boxes, scores=scores, feats=x)

        if self.end2end:
            # O2O regression on detached features (standard pattern)
            x_o2o = [xi.detach() for xi in x]
            o2o_box_feats, o2o_box_preds = [], []
            for i in range(self.nl):
                feat = self.one2one_cv2[i][:2](x_o2o[i])
                pred = self.one2one_cv2[i][2](feat)
                o2o_box_feats.append(feat)
                o2o_box_preds.append(pred.view(bs, 4 * self.reg_max, -1))
            o2o_boxes = torch.cat(o2o_box_preds, dim=-1)

            # O2O cls with o2o box context (detached so cls grads don't update box stem)
            o2o_cls = []
            for i in range(self.nl):
                ctx = torch.cat([x_o2o[i], o2o_box_feats[i].detach()], dim=1)
                o2o_cls.append(self.one2one_cv3[i](ctx).view(bs, self.nc, -1))
            o2o_scores = torch.cat(o2o_cls, dim=-1)
            one2one = dict(boxes=o2o_boxes, scores=o2o_scores, feats=x)

            if self.cv3 is not None and self.cv2 is not None:
                preds = {"one2many": preds, "one2one": one2one}
            else:
                preds = {"one2many": dict(), "one2one": one2one}

        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def bias_init(self):
        """Initialize biases for o2m and o2o branches."""
        for i, (a, b) in enumerate(zip(self.cv2, self.cv3)):
            a[-1].bias.data[:] = 2.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, (a, b) in enumerate(zip(self.one2one_cv2, self.one2one_cv3)):
                a[-1].bias.data[:] = 2.0
                b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)

    def fuse(self):
        """Remove o2m heads for inference (o2o uses its own cv2)."""
        self.cv2 = self.cv3 = None


class DetectBoxContextFullSep(DetectBoxContextSep):
    """Like DetectBoxContextFull but with SEPARATE (duplicated) regression branches.

    O2m: own regression + cls uses o2m box features as context (non-detached).
    O2o: own regression + cls uses o2o box features as context (detached).
    """

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        """Initialize with duplicated regression and box-context cls for both o2m and o2o."""
        super().__init__(nc, reg_max, end2end=end2end, ch=ch)

        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], (self.fixed_c3 or min(self.nc, 100)))

        # Rebuild cv3 with box context input: x + c2 channels
        if self.rep_head:
            self.cv3 = (
                nn.ModuleList(
                    nn.Sequential(Conv(x + c2, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
                )
                if self.legacy
                else nn.ModuleList(
                    nn.Sequential(
                        nn.Sequential(RepConv(x + c2, x + c2, 3, g=x + c2, bn=True), Conv(x + c2, c3, 1)),
                        nn.Sequential(RepConv(c3, c3, 3, g=c3, bn=True), Conv(c3, c3, 1)),
                        nn.Conv2d(c3, self.nc, 1),
                    )
                    for x in ch
                )
            )
        else:
            self.cv3 = (
                nn.ModuleList(
                    nn.Sequential(Conv(x + c2, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
                )
                if self.legacy
                else nn.ModuleList(
                    nn.Sequential(
                        nn.Sequential(DWConv(x + c2, x + c2, 3), Conv(x + c2, c3, 1)),
                        nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                        nn.Conv2d(c3, self.nc, 1),
                    )
                    for x in ch
                )
            )

    def forward(self, x):
        """Forward with separate o2m/o2o regression; both cls use respective box context."""
        bs = x[0].shape[0]

        # O2M regression with intermediate features captured
        if self.cv2 is not None:
            o2m_box_feats, o2m_box_preds = [], []
            for i in range(self.nl):
                feat = self.cv2[i][:2](x[i])
                pred = self.cv2[i][2](feat)
                o2m_box_feats.append(feat)
                o2m_box_preds.append(pred.view(bs, 4 * self.reg_max, -1))
            boxes = torch.cat(o2m_box_preds, dim=-1)

        # O2M cls with o2m box context (non-detached)
        if self.cv3 is not None:
            cls_preds = []
            for i in range(self.nl):
                ctx = torch.cat([x[i], o2m_box_feats[i]], dim=1)
                cls_preds.append(self.cv3[i](ctx).view(bs, self.nc, -1))
            scores = torch.cat(cls_preds, dim=-1)
            preds = dict(boxes=boxes, scores=scores, feats=x)

        if self.end2end:
            # O2O regression separate, on detached features
            x_o2o = [xi.detach() for xi in x]
            o2o_box_feats, o2o_box_preds = [], []
            for i in range(self.nl):
                feat = self.one2one_cv2[i][:2](x_o2o[i])
                pred = self.one2one_cv2[i][2](feat)
                o2o_box_feats.append(feat)
                o2o_box_preds.append(pred.view(bs, 4 * self.reg_max, -1))
            o2o_boxes = torch.cat(o2o_box_preds, dim=-1)

            # O2O cls with o2o box context (detached)
            o2o_cls = []
            for i in range(self.nl):
                ctx = torch.cat([x_o2o[i], o2o_box_feats[i].detach()], dim=1)
                o2o_cls.append(self.one2one_cv3[i](ctx).view(bs, self.nc, -1))
            o2o_scores = torch.cat(o2o_cls, dim=-1)
            one2one = dict(boxes=o2o_boxes, scores=o2o_scores, feats=x)

            if self.cv3 is not None and self.cv2 is not None:
                preds = {"one2many": preds, "one2one": one2one}
            else:
                preds = {"one2many": dict(), "one2one": one2one}

        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)


class DetectROI(Detect):
    """Two-stage detector: YOLO Detect (stage 1) + RoIAlign MLP head (stage 2).

    Stage 1 (inherited): o2m training + o2o end2end head produce top-N proposals.
    Stage 2: lateral 1x1 on each FPN level to common channels, RoIAlign at canonical
    FPN level, shared MLP, class-agnostic box-delta + cls logits. Proposals detached;
    FPN feats kept live so stage-2 grads reach neck/backbone.
    Inference: refined box + fused score sqrt(s1 * s2_c1).
    """

    roi_out = 7
    roi_ch = 128
    roi_proposals = 300

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        """Initialize stage-1 (forced end2end) + stage-2 RoI head (conv + GAP)."""
        assert end2end, "DetectROI requires end2end=True (uses o2o proposals)"
        super().__init__(nc, reg_max, end2end=True, ch=ch)
        c = self.roi_ch
        self.roi_lateral = nn.ModuleList(nn.Conv2d(ci, c, 1) for ci in ch)
        self.roi_head = nn.Sequential(Conv(c, c, 3), Conv(c, c, 3))
        self.roi_cls = nn.Linear(c, self.nc)
        self.roi_reg = nn.Linear(c, 4)
        nn.init.zeros_(self.roi_reg.weight)
        nn.init.zeros_(self.roi_reg.bias)
        nn.init.constant_(self.roi_cls.bias, math.log(0.01 / 0.99))

    def _roi_levels(self, wh):
        """Canonical FPN level per RoI (0..nl-1) from sqrt(area)."""
        k = 4.0 + torch.log2(torch.sqrt(wh.prod(-1).clamp(min=1.0)) / 224.0)
        return k.floor().clamp(3, 3 + self.nl - 1).long() - 3

    def _roi_pool_grid(self, proj, rois, lvls):
        """grid_sample pool (training path): compile-friendly, no torchvision CUDA op."""
        P = self.roi_out
        C = self.roi_ch
        B = proj[0].shape[0]
        device, dtype = proj[0].device, proj[0].dtype
        rois = rois.to(dtype)
        # Per-cell fractional sample positions in [0, 1] (cell centers)
        t = torch.linspace(0.5 / P, 1.0 - 0.5 / P, P, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(t, t, indexing="ij")  # (P, P)
        pooled = proj[0].new_zeros(rois.shape[0], C, P, P)
        roi_b = rois[:, 0].long()
        for i in range(self.nl):
            feat = proj[i]
            _, _, H, W = feat.shape
            stride = float(self.stride[i])
            for b in range(B):
                mask = (lvls == i) & (roi_b == b)
                if not mask.any():
                    continue
                boxes = rois[mask, 1:5] / stride  # feature-map coords
                M = boxes.shape[0]
                x1 = boxes[:, 0:1, None]
                y1 = boxes[:, 1:2, None]
                x2 = boxes[:, 2:3, None]
                y2 = boxes[:, 3:4, None]
                px = x1 + (x2 - x1) * gx.unsqueeze(0)  # (M, P, P)
                py = y1 + (y2 - y1) * gy.unsqueeze(0)
                # Normalize to [-1, 1] (align_corners=False convention)
                nx = (2.0 * px + 1.0) / W - 1.0
                ny = (2.0 * py + 1.0) / H - 1.0
                grid = torch.stack([nx, ny], dim=-1).view(1, M * P, P, 2)
                sampled = F.grid_sample(
                    feat[b : b + 1], grid, mode="bilinear", padding_mode="zeros", align_corners=False,
                )  # (1, C, M*P, P)
                sampled = sampled.view(1, C, M, P, P).permute(0, 2, 1, 3, 4).reshape(M, C, P, P)
                pooled[mask] = sampled.to(pooled.dtype)
        return pooled

    def _roi_pool_align(self, proj, rois, lvls):
        """torchvision roi_align pool (inference path)."""
        from torchvision.ops import roi_align
        P = self.roi_out
        pooled = proj[0].new_zeros(rois.shape[0], self.roi_ch, P, P)
        for i in range(self.nl):
            m = lvls == i
            if m.any():
                pooled[m] = roi_align(
                    proj[i], rois[m], output_size=P,
                    spatial_scale=1.0 / float(self.stride[i]), sampling_ratio=2, aligned=True,
                )
        return pooled

    def _roi_forward(self, feats, rois):
        """rois (N,5) [b,x1,y1,x2,y2] pixel → refined (N,4), cls (N,nc), deltas (N,4)."""
        # Guard against stride-discovery build pass (strides not yet set)
        if float(self.stride.min()) <= 0:
            n = rois.shape[0]
            z = rois.new_zeros(n, self.nc)
            return rois[:, 1:5].clone(), z, rois.new_zeros(n, 4)
        proj = [self.roi_lateral[i](feats[i]) for i in range(self.nl)]
        if rois.numel() == 0:
            z = proj[0].new_zeros(0, self.nc)
            return proj[0].new_zeros(0, 4), z, proj[0].new_zeros(0, 4)
        wh = rois[:, 3:5] - rois[:, 1:3]
        lvls = self._roi_levels(wh)
        pooled = self._roi_pool_grid(proj, rois, lvls) if self.training else self._roi_pool_align(proj, rois, lvls)
        h = self.roi_head(pooled).mean(dim=[-2, -1])  # GAP → (N, C)
        cls_logits = self.roi_cls(h)
        deltas = self.roi_reg(h)
        px = (rois[:, 1] + rois[:, 3]) * 0.5
        py = (rois[:, 2] + rois[:, 4]) * 0.5
        pw = (rois[:, 3] - rois[:, 1]).clamp(min=1.0)
        ph = (rois[:, 4] - rois[:, 2]).clamp(min=1.0)
        dx, dy, dw, dh = deltas.unbind(-1)
        cx = px + dx * pw
        cy = py + dy * ph
        w = pw * dw.clamp(max=4.0).exp()
        h_ = ph * dh.clamp(max=4.0).exp()
        refined = torch.stack([cx - w * 0.5, cy - h_ * 0.5, cx + w * 0.5, cy + h_ * 0.5], dim=-1)
        return refined, cls_logits, deltas

    def _select_proposals(self, preds_o2o):
        """Decode o2o preds, top-N per image. Returns (B, N, 6) [xyxy, score, cls_idx]."""
        y = self._inference(preds_o2o)  # (B, 4+nc, A)
        n = min(self.roi_proposals, y.shape[-1])
        y = y.permute(0, 2, 1)
        boxes, scores = y.split([4, self.nc], dim=-1)
        s, conf, idx = self.get_topk_index(scores, n)
        boxes = boxes.gather(1, idx.repeat(1, 1, 4))
        return torch.cat([boxes, s, conf], dim=-1)

    def forward(self, x):
        """Stage 1 forward + stage-2 RoI refinement on top-N proposals."""
        out = super().forward(x)
        # Skip stage-2 during stride-discovery build pass or tiny profile input
        # (FLOPs scale badly when ultralytics profiles at (stride×stride) then multiplies by (imgsz/stride)²)
        if float(self.stride.min()) <= 0 or x[0].shape[-1] < 16:
            return out
        if self.training:
            props = self._select_proposals(out["one2one"])  # (B, N, 6)
            B, N = props.shape[:2]
            b_idx = torch.arange(B, device=props.device).view(B, 1, 1).expand(-1, N, 1).reshape(-1, 1).float()
            rois = torch.cat([b_idx, props[..., :4].reshape(-1, 4).detach()], dim=1)
            refined, cls_logits, deltas = self._roi_forward(x, rois)
            out["stage2"] = dict(
                rois=rois, refined=refined, cls_logits=cls_logits, deltas=deltas,
                prop_score=props[..., 4].reshape(-1), prop_cls=props[..., 5].reshape(-1),
            )
            return out
        y, raw = (out, None) if self.export else out
        B, N, _ = y.shape
        b_idx = torch.arange(B, device=y.device).view(B, 1, 1).expand(-1, N, 1).reshape(-1, 1).float()
        rois = torch.cat([b_idx, y[..., :4].reshape(-1, 4).detach()], dim=1)
        refined, cls_logits, _ = self._roi_forward(x, rois)
        refined = refined.view(B, N, 4)
        s2 = cls_logits.view(B, N, self.nc).sigmoid()
        cls_idx = y[..., 5].long()
        s1 = y[..., 4]
        s2_c = s2.gather(-1, cls_idx.unsqueeze(-1)).squeeze(-1)
        fused = (s1 * s2_c).clamp(min=0).sqrt()
        y_out = torch.cat([refined, fused.unsqueeze(-1), cls_idx.float().unsqueeze(-1)], dim=-1)
        return y_out if self.export else (y_out, raw)


class DetectFDR(Detect):
    """Dense conv port of D-FINE FDR (ICLR 2025): residual refinement of DFL logits.

    Stage 1 (inherited cv2) predicts coarse DFL logits. Stage 2 conditions on the feature map
    concatenated with the stage-1 distribution (softmax over reg_max per side) and predicts a
    residual added to the stage-1 logits. Last refine conv is zero-init so training starts
    exactly at the Detect baseline. No queries, no attention, per-anchor conv only.
    fdr_steps > 1 applies the (weight-shared) refinement recurrently, mimicking D-FINE's
    multi-layer refinement at zero extra params. During training the stage-1 logits are kept
    in the output dict ("boxes_s1") so E2ELoss can apply GO-LSD self-distillation.
    """

    fdr_steps = 1  # number of (weight-shared) refinement iterations
    fdr_rf_mix = ()  # dilations of the per-location soft receptive-field mixer in the refine tower (()=off)
    fdr_ms_read = False  # cross-level soft rereading of box evidence (ScaleRead)
    fdr_ms_levels = None  # target levels for ScaleRead (None = all; [1, 2] = P4/P5 only)
    fdr_rf_fast = False  # use the cheaper cascaded/bottlenecked SoftRFMixFast instead of SoftRFMix
    fdr_box_pool = False  # box-conditioned pooling-pyramid rereading (BoxPoolRead)

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        """Initialize Detect then add per-level refinement branches (o2m + o2o copies)."""
        super().__init__(nc, reg_max, end2end, ch)
        c2 = self.head_c2 or max(16, ch[0] // 4, self.reg_max * 4)
        mix = [(SoftRFMixFast(c2, len(self.fdr_rf_mix), cascade=self.fdr_rf_fast != 2) if self.fdr_rf_fast else SoftRFMix(c2, self.fdr_rf_mix))] if self.fdr_rf_mix else []
        self.cv2_ref = nn.ModuleList(
            nn.Sequential(Conv(x + 4 * self.reg_max, c2, 3), *copy.deepcopy(mix), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1))
            for x in ch
        )
        for m in self.cv2_ref:
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)
        if self.fdr_ms_read:
            self.cv2_ms = ScaleRead(ch, c2, 4 * self.reg_max, self.fdr_ms_levels)
        if self.fdr_box_pool:
            self.cv2_bp = nn.ModuleList(BoxPoolRead(x, self.reg_max) for x in ch)
        if end2end and self.one2one_cv2 is not None:
            self.one2one_cv2_ref = copy.deepcopy(self.cv2_ref)
            if self.fdr_ms_read:
                self.one2one_cv2_ms = copy.deepcopy(self.cv2_ms)
            if self.fdr_box_pool:
                self.one2one_cv2_bp = copy.deepcopy(self.cv2_bp)

    @property
    def one2many(self):
        """One-to-many heads plus refinement branch."""
        d = dict(box_head=self.cv2, cls_head=self.cv3, box_ref=self.cv2_ref)
        if self.fdr_ms_read:
            d["box_ms"] = self.cv2_ms
        if self.fdr_box_pool:
            d["box_pool"] = self.cv2_bp
        return d

    @property
    def one2one(self):
        """One-to-one heads plus refinement branch."""
        d = dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, box_ref=getattr(self, "one2one_cv2_ref", None))
        if hasattr(self, "o2o_suppress"):
            d["suppress"] = self.o2o_suppress
        if self.fdr_ms_read:
            d["box_ms"] = getattr(self, "one2one_cv2_ms", None)
        if self.fdr_box_pool:
            d["box_pool"] = getattr(self, "one2one_cv2_bp", None)
        return d

    def _extra_reads(self, i, x, b, box_ms_out, box_pool):
        """Add the cross-level and box-conditioned residual reads to the refined box logits."""
        if box_ms_out is not None and box_ms_out[i] is not None:
            b = b + box_ms_out[i]
        if box_pool is not None:
            b = b + box_pool[i](x[i], b.view(b.shape[0], 4, self.reg_max, *b.shape[2:]).softmax(2))
        return b

    def forward_head(self, x, box_head=None, cls_head=None, suppress=None, box_ref=None, box_ms=None, box_pool=None):
        """Stage-1 boxes + residual distribution refinement, then concat as in Detect."""
        if box_head is None or cls_head is None:  # for fused inference
            return dict()
        bs = x[0].shape[0]
        boxes, boxes_s1 = [], []
        ms = box_ms(x) if box_ms is not None else None
        for i in range(self.nl):
            b = box_head[i](x[i])  # (B, 4*reg_max, H, W)
            boxes_s1.append(b.view(bs, 4 * self.reg_max, -1))
            for _ in range(self.fdr_steps):
                prob = b.view(bs, 4, self.reg_max, *b.shape[2:]).softmax(2).flatten(1, 2)
                b = b + box_ref[i](torch.cat([x[i], prob], 1))
            b = self._extra_reads(i, x, b, ms, box_pool)
            boxes.append(b.view(bs, 4 * self.reg_max, -1))
        boxes = torch.cat(boxes, dim=-1)
        cls_feats = []
        for i in range(self.nl):
            c = cls_head[i](x[i])
            if suppress is not None:
                c = suppress[i](c)
            cls_feats.append(c.view(bs, self.nc, -1))
        out = dict(boxes=boxes, scores=torch.cat(cls_feats, dim=-1), feats=x)
        if self.training:
            out["boxes_s1"] = torch.cat(boxes_s1, dim=-1)
        return out

    def fuse(self):
        """Drop o2m heads (incl. refinement) for inference."""
        super().fuse()
        if self.cv2 is None:
            self.cv2_ref = None


class DetectDGQP(Detect):
    """GFLv2-style Distribution-Guided Quality Prediction for the e2e Detect head.

    Statistics of the predicted DFL distribution (max, mean, sum of squares per side) feed a tiny
    shared subnet whose output is added to the cls logits as a localization-quality logit.
    Additive-logit variant of GFLv2's multiplicative J = C * I, with the paper's top-k stats
    replaced by reduce-based stats for export friendliness. Zero-init so training starts
    exactly at the Detect baseline. ~1k params, conv-only.
    """

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        """Initialize Detect then add the shared quality subnet (o2m + o2o copies)."""
        super().__init__(nc, reg_max, end2end, ch)
        cq, ci = 64, 4 * 3
        self.reg_conf = nn.Sequential(nn.Conv2d(ci, cq, 1), nn.ReLU(inplace=True), nn.Conv2d(cq, 1, 1))
        nn.init.zeros_(self.reg_conf[-1].weight)
        nn.init.zeros_(self.reg_conf[-1].bias)
        if end2end and self.one2one_cv2 is not None:
            self.one2one_reg_conf = copy.deepcopy(self.reg_conf)

    @property
    def one2many(self):
        """One-to-many heads plus quality subnet."""
        return dict(box_head=self.cv2, cls_head=self.cv3, quality=self.reg_conf)

    @property
    def one2one(self):
        """One-to-one heads plus quality subnet."""
        d = dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, quality=getattr(self, "one2one_reg_conf", None))
        if hasattr(self, "o2o_suppress"):
            d["suppress"] = self.o2o_suppress
        return d

    def forward_head(self, x, box_head=None, cls_head=None, suppress=None, quality=None):
        """Detect forward with DFL-statistics quality logit added to cls scores."""
        if box_head is None or cls_head is None:  # for fused inference
            return dict()
        bs = x[0].shape[0]
        boxes, cls_feats = [], []
        for i in range(self.nl):
            b = box_head[i](x[i])  # (B, 4*reg_max, H, W)
            prob = b.view(bs, 4, self.reg_max, *b.shape[2:]).softmax(2)
            stat = torch.cat([prob.max(2, keepdim=True)[0], prob.mean(2, keepdim=True), prob.pow(2).sum(2, keepdim=True)], 2).flatten(1, 2)
            c = cls_head[i](x[i]) + quality(stat)
            if suppress is not None:
                c = suppress[i](c)
            boxes.append(b.view(bs, 4 * self.reg_max, -1))
            cls_feats.append(c.view(bs, self.nc, -1))
        return dict(boxes=torch.cat(boxes, dim=-1), scores=torch.cat(cls_feats, dim=-1), feats=x)

    def fuse(self):
        """Drop o2m heads (incl. quality subnet) for inference."""
        super().fuse()
        if self.cv2 is None:
            self.reg_conf = None


class DetectFDRQ(DetectFDR):
    """DetectFDR + DGQP: quality logit computed from the REFINED distribution.

    Combines D-FINE-style residual distribution refinement with a GFLv2-style quality logit
    (max, mean, sum of squares of the refined DFL distribution) added to the cls score.
    Both extras zero-init, so training starts exactly at the Detect baseline.
    """

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        """Initialize DetectFDR then add the shared quality subnet (o2m + o2o copies)."""
        super().__init__(nc, reg_max, end2end, ch)
        cq, ci = 64, 4 * 3
        self.reg_conf = nn.Sequential(nn.Conv2d(ci, cq, 1), nn.ReLU(inplace=True), nn.Conv2d(cq, 1, 1))
        nn.init.zeros_(self.reg_conf[-1].weight)
        nn.init.zeros_(self.reg_conf[-1].bias)
        if end2end and self.one2one_cv2 is not None:
            self.one2one_reg_conf = copy.deepcopy(self.reg_conf)

    @property
    def one2many(self):
        """One-to-many heads plus refinement branch and quality subnet."""
        return dict(**super().one2many, quality=self.reg_conf)

    @property
    def one2one(self):
        """One-to-one heads plus refinement branch and quality subnet."""
        return dict(**super().one2one, quality=getattr(self, "one2one_reg_conf", None))

    def forward_head(self, x, box_head=None, cls_head=None, suppress=None, box_ref=None, quality=None, box_ms=None, box_pool=None):
        """FDR refinement, then quality logit from the refined distribution added to cls."""
        if box_head is None or cls_head is None:  # for fused inference
            return dict()
        bs = x[0].shape[0]
        boxes, boxes_s1, cls_feats = [], [], []
        ms = box_ms(x) if box_ms is not None else None
        for i in range(self.nl):
            b = box_head[i](x[i])  # (B, 4*reg_max, H, W)
            boxes_s1.append(b.view(bs, 4 * self.reg_max, -1))
            for _ in range(self.fdr_steps):
                prob = b.view(bs, 4, self.reg_max, *b.shape[2:]).softmax(2).flatten(1, 2)
                b = b + box_ref[i](torch.cat([x[i], prob], 1))
            b = self._extra_reads(i, x, b, ms, box_pool)
            prob = b.view(bs, 4, self.reg_max, *b.shape[2:]).softmax(2)
            stat = torch.cat([prob.max(2, keepdim=True)[0], prob.mean(2, keepdim=True), prob.pow(2).sum(2, keepdim=True)], 2).flatten(1, 2)
            c = cls_head[i](x[i]) + quality(stat)
            if suppress is not None:
                c = suppress[i](c)
            boxes.append(b.view(bs, 4 * self.reg_max, -1))
            cls_feats.append(c.view(bs, self.nc, -1))
        out = dict(boxes=torch.cat(boxes, dim=-1), scores=torch.cat(cls_feats, dim=-1), feats=x)
        if self.training:
            out["boxes_s1"] = torch.cat(boxes_s1, dim=-1)
        return out

    def fuse(self):
        """Drop o2m heads (incl. refinement and quality subnet) for inference."""
        super().fuse()
        if self.cv2 is None:
            self.reg_conf = None


class ScaleRead(nn.Module):
    """Cross-level soft rereading of the box evidence (ASFF / DynamicHead scale attention direction).

    Deformable attention reads every feature level per query; this reads all levels at the anchor's own
    location after fixed-ratio alignment (strided avg-pool down, nearest upsample up) and mixes them with
    a per-location softmax over the three levels. Output convs are zero-init, so it starts as a no-op.
    Export: 1x1 conv, fixed-stride AvgPool, fixed-scale Resize, concat, softmax, mul, add.
    """

    def __init__(self, ch: tuple, c: int = 64, nb: int = 64, levels=None):
        """Initialize ScaleRead.

        Args:
            ch (tuple): Input channels per level.
            c (int): Shared projection width.
            nb (int): Output channels (4 * reg_max).
            levels (tuple | None): Target levels that get the cross-level read (None = all). Skipping P3
                keeps the large-object gain while leaving high-resolution small-object evidence unmixed,
                and drops the block's cost since the P3 branch does all the upsampling at 80x80.
        """
        super().__init__()
        self.nl = len(ch)
        self.levels = tuple(range(self.nl)) if levels is None else tuple(levels)
        self.proj = nn.ModuleList(Conv(x, c, 1) for x in ch)
        self.gate = nn.ModuleList(nn.Conv2d(c * self.nl, self.nl, 1) if i in self.levels else nn.Identity() for i in range(self.nl))
        self.out = nn.ModuleList(nn.Conv2d(c, nb, 1) if i in self.levels else nn.Identity() for i in range(self.nl))
        for m in self.out:
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor | None]:
        """Return one box-logit residual per target level, mixed from all aligned levels."""
        p = [self.proj[i](x[i]) for i in range(self.nl)]
        out = []
        for i in range(self.nl):
            if i not in self.levels:
                out.append(None)
                continue
            aligned = []
            for j in range(self.nl):
                if j < i:
                    aligned.append(F.avg_pool2d(p[j], 2 ** (i - j), 2 ** (i - j)))
                elif j > i:
                    aligned.append(F.interpolate(p[j], scale_factor=2 ** (j - i), mode="nearest"))
                else:
                    aligned.append(p[j])
            a = self.gate[i](torch.cat(aligned, 1)).softmax(1)
            y = aligned[0] * a.narrow(1, 0, 1)
            for j in range(1, self.nl):
                y = y + aligned[j] * a.narrow(1, j, 1)
            out.append(self.out[i](y))
        return out


class BoxPoolRead(nn.Module):
    """Box-conditioned pooling pyramid: predicted-box-shaped rereading without grid_sample.

    RoIAlign and box attention need to sample at predicted coordinates. This keeps the read centered on
    the anchor but lets the predicted (l, t, r, b) distances pick the pooling footprint (point, 3x3, 7x7,
    1x9, 9x1) through a softmax gate, recovering box-conditioned receptive-field scale and aspect.
    Output conv is zero-init. Export: 1x1 conv, fixed AvgPool, softmax, mul, add.
    """

    def __init__(self, c1: int, reg_max: int = 16, cq: int = 32):
        """Initialize BoxPoolRead.

        Args:
            c1 (int): Input channels of the level feature.
            reg_max (int): DFL bins per side.
            cq (int): Pooling-branch width.
        """
        super().__init__()
        self.reg_max = reg_max
        self.q = Conv(c1, cq, 1)
        self.gate = nn.Sequential(Conv(4, 16, 1), nn.Conv2d(16, 5, 1))
        self.out = nn.Conv2d(cq, 4 * reg_max, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("proj", torch.arange(reg_max, dtype=torch.float), persistent=False)

    def forward(self, x: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
        """Mix pooled reads by a gate conditioned on the predicted per-side distances."""
        d = prob.mul(self.proj.view(1, 1, -1, 1, 1).to(prob.dtype)).sum(2)  # (B, 4, H, W)
        a = self.gate(d).softmax(1)
        q = self.q(x)
        pools = (
            q,
            F.avg_pool2d(q, 3, 1, 1),
            F.avg_pool2d(q, 7, 1, 3),
            F.avg_pool2d(q, (1, 9), 1, (0, 4)),
            F.avg_pool2d(q, (9, 1), 1, (4, 0)),
        )
        y = pools[0] * a.narrow(1, 0, 1)
        for i in range(1, len(pools)):
            y = y + pools[i] * a.narrow(1, i, 1)
        return self.out(y)


class DetectFGL(Detect):
    """Relative fine-grained refinement (D-FINE, ICLR 2025) with a fixed non-uniform residual codebook.

    Stage 1 (inherited cv2) predicts absolute DFL logits, decoded to distances d0. Stage 2 predicts a
    distribution over a fixed, zero-centered, non-uniformly spaced codebook W in [-1, 1] and applies a
    scale-relative correction d1 = d0 + rho * d0 * (softmax(r) @ W), so refinement solves a small
    centered correction instead of relearning absolute edge distances. Zero-init residual predictor
    makes the codebook expectation 0 at step 0, i.e. training starts exactly at the Detect baseline.
    Export ops: conv, softmax, matmul with a constant, mul, add.
    """

    fgl_rho = 0.5  # residual scale relative to the stage-1 distance
    fgl_curve = 3.0  # codebook curvature (larger = finer spacing near zero)

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        """Initialize Detect then add the residual codebook branches (o2m + o2o copies)."""
        super().__init__(nc, reg_max, end2end, ch)
        c2 = self.head_c2 or max(16, ch[0] // 4, self.reg_max * 4)
        self.cv2_ref = nn.ModuleList(
            nn.Sequential(Conv(x + 4 * self.reg_max, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1))
            for x in ch
        )
        for m in self.cv2_ref:
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)
        u = torch.linspace(-1, 1, self.reg_max)
        w = u.sign() * (self.fgl_curve * u.abs()).expm1() / math.expm1(self.fgl_curve)
        self.register_buffer("fgl_W", w, persistent=False)
        self.register_buffer("fgl_proj", torch.arange(self.reg_max, dtype=torch.float), persistent=False)
        if end2end and self.one2one_cv2 is not None:
            self.one2one_cv2_ref = copy.deepcopy(self.cv2_ref)

    @property
    def one2many(self):
        """One-to-many heads plus refinement branch."""
        return dict(box_head=self.cv2, cls_head=self.cv3, box_ref=self.cv2_ref)

    @property
    def one2one(self):
        """One-to-one heads plus refinement branch."""
        d = dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, box_ref=getattr(self, "one2one_cv2_ref", None))
        if hasattr(self, "o2o_suppress"):
            d["suppress"] = self.o2o_suppress
        return d

    def forward_head(self, x, box_head=None, cls_head=None, suppress=None, box_ref=None):
        """Stage-1 logits + codebook residual, returning both plus the refined distances."""
        if box_head is None or cls_head is None:  # for fused inference
            return dict()
        bs = x[0].shape[0]
        boxes, refs, dists = [], [], []
        for i in range(self.nl):
            b = box_head[i](x[i])  # (B, 4*reg_max, H, W)
            prob = b.view(bs, 4, self.reg_max, -1).softmax(2)
            d0 = prob.mul(self.fgl_proj.view(1, 1, -1, 1).to(prob.dtype)).sum(2)  # (B, 4, A_l)
            r = box_ref[i](torch.cat([x[i], prob.flatten(1, 2).view_as(b)], 1))
            delta = r.view(bs, 4, self.reg_max, -1).softmax(2).mul(self.fgl_W.view(1, 1, -1, 1).to(b.dtype)).sum(2)
            dists.append(d0 + self.fgl_rho * d0 * delta)
            boxes.append(b.view(bs, 4 * self.reg_max, -1))
            refs.append(r.view(bs, 4 * self.reg_max, -1))
        cls_feats = []
        for i in range(self.nl):
            c = cls_head[i](x[i])
            if suppress is not None:
                c = suppress[i](c)
            cls_feats.append(c.view(bs, self.nc, -1))
        out = dict(
            boxes=torch.cat(boxes, dim=-1),
            scores=torch.cat(cls_feats, dim=-1),
            feats=x,
            distances=torch.cat(dists, dim=-1),
        )
        if self.training:
            out["boxes_ref"] = torch.cat(refs, dim=-1)
        return out

    def _get_decode_boxes(self, x):
        """Decode from the refined distances instead of the stage-1 DFL logits."""
        shape = x["feats"][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape
        return self.decode_bboxes(x["distances"], self.anchors.unsqueeze(0)) * self.strides

    def fuse(self):
        """Drop o2m heads (incl. refinement) for inference."""
        super().fuse()
        if self.cv2 is None:
            self.cv2_ref = None


class DetectFDRC(DetectFDR):
    """DetectFDR with target-duplicated residual predictors (Deep Regression Tightness, ICLR 2025).

    The final 1x1 of each refinement branch predicts `box_copies` independent residual maps trained on
    the same target; the deployed residual is their mean, which folds back into a single conv at fuse()
    because convolution is linear. Copies are initialized to +/- eps so their mean stays exactly the
    zero-init FDR residual while symmetry is broken.
    """

    box_copies = 2  # number of duplicated residual predictors
    box_copy_eps = 1e-3  # symmetry-breaking init scale

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        """Initialize DetectFDR then widen the residual predictors into `box_copies` copies."""
        super().__init__(nc, reg_max, end2end, ch)
        nb = 4 * self.reg_max
        for m in self.cv2_ref:
            old = m[-1]
            conv = nn.Conv2d(old.in_channels, nb * self.box_copies, 1)
            nn.init.normal_(conv.weight.data, std=self.box_copy_eps)
            nn.init.zeros_(conv.bias.data)
            conv.weight.data[nb:] = -conv.weight.data[: nb * (self.box_copies - 1)]  # copies cancel exactly
            m[-1] = conv
        if end2end and self.one2one_cv2 is not None:
            self.one2one_cv2_ref = copy.deepcopy(self.cv2_ref)

    def forward_head(self, x, box_head=None, cls_head=None, suppress=None, box_ref=None):
        """FDR refinement using the mean of the duplicated residual predictors."""
        if box_head is None or cls_head is None:  # for fused inference
            return dict()
        bs = x[0].shape[0]
        nb = 4 * self.reg_max
        boxes, boxes_s1, copies = [], [], []
        for i in range(self.nl):
            b = box_head[i](x[i])
            boxes_s1.append(b.view(bs, nb, -1))
            for _ in range(self.fdr_steps):
                prob = b.view(bs, 4, self.reg_max, *b.shape[2:]).softmax(2).flatten(1, 2)
                r = box_ref[i](torch.cat([x[i], prob], 1))
                nk = r.shape[1] // nb  # 1 once the copies are folded at fuse()
                if self.training and nk > 1:
                    copies.append(b.detach().view(bs, nb, -1).unsqueeze(0) + r.view(bs, nk, nb, -1).transpose(0, 1))
                b = b + r.view(bs, nk, nb, *r.shape[2:]).mean(1)
            boxes.append(b.view(bs, nb, -1))
        cls_feats = []
        for i in range(self.nl):
            c = cls_head[i](x[i])
            if suppress is not None:
                c = suppress[i](c)
            cls_feats.append(c.view(bs, self.nc, -1))
        out = dict(boxes=torch.cat(boxes, dim=-1), scores=torch.cat(cls_feats, dim=-1), feats=x)
        if self.training:
            out["boxes_s1"] = torch.cat(boxes_s1, dim=-1)
            out["boxes_copies"] = torch.cat(copies, dim=-1)  # (copies, B, 4*reg_max, A)
        return out

    def fuse(self):
        """Fold the duplicated residual predictors into their mean, then drop the o2m heads."""
        nb = 4 * self.reg_max
        for refs in (getattr(self, "one2one_cv2_ref", None), self.cv2_ref):
            if refs is None:
                continue
            for m in refs:
                old = m[-1]
                if old.out_channels == nb:
                    continue
                conv = nn.Conv2d(old.in_channels, nb, 1).requires_grad_(False)
                conv.weight.data = old.weight.data.view(self.box_copies, nb, *old.weight.shape[1:]).mean(0)
                conv.bias.data = old.bias.data.view(self.box_copies, nb).mean(0)
                m[-1] = conv
        super().fuse()


class TATower(nn.Module):
    """Per-level TOOD-lite tower: shared inter-task convs + per-task layer attention + preds."""

    def __init__(self, c1, ct, nc, reg_max):
        """Build shared 2-conv stack, box/cls layer-attention gates, and prediction convs."""
        super().__init__()
        self.inter = nn.ModuleList([Conv(c1, ct, 3), Conv(ct, ct, 3)])
        self.la_box = nn.Conv2d(2 * ct, 2, 1)
        self.la_cls = nn.Conv2d(2 * ct, 2, 1)
        self.box = nn.Conv2d(ct, 4 * reg_max, 3, padding=1)
        self.cls = nn.Conv2d(ct, nc, 3, padding=1)

    def forward(self, x):
        """Return (box_out, cls_out) from task-attended blends of the shared stack."""
        f1 = self.inter[0](x)
        f2 = self.inter[1](f1)
        g = torch.cat([f1, f2], 1).mean((2, 3), keepdim=True)
        wb = self.la_box(g).sigmoid()
        wc = self.la_cls(g).sigmoid()
        fb = f1 * wb[:, 0:1] + f2 * wb[:, 1:2]
        fc = f1 * wc[:, 0:1] + f2 * wc[:, 1:2]
        return self.box(fb), self.cls(fc)


class DetectTA(Detect):
    """TOOD-lite task-aligned head (TOOD, ICCV 2021 oral).

    Replaces the decoupled parallel box/cls branches with a shared inter-task conv stack per
    level; each task blends the stack's features via a learned layer-attention gate before its
    prediction conv, so cls and box supervision interact through shared features. The paper's
    deformable spatial alignment is intentionally omitted (export constraint). Conv, pool,
    sigmoid, mul only.
    """

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        """Build parent, then swap the decoupled branches for shared TA towers."""
        super().__init__(nc, reg_max, end2end, ch)
        ct = self.fixed_c3 or max(ch[0], 128)
        del self.cv2, self.cv3
        self.ta = nn.ModuleList(TATower(x, ct, nc, self.reg_max) for x in ch)
        if end2end:
            del self.one2one_cv2, self.one2one_cv3
            self.one2one_ta = copy.deepcopy(self.ta)

    @property
    def one2many(self):
        """One-to-many TA towers."""
        return dict(ta=self.ta)

    @property
    def one2one(self):
        """One-to-one TA towers."""
        d = dict(ta=self.one2one_ta)
        if hasattr(self, "o2o_suppress"):
            d["suppress"] = self.o2o_suppress
        return d

    def forward_head(self, x, ta=None, suppress=None):
        """Concatenate box and cls predictions from the TA towers."""
        if ta is None:  # for fused inference
            return dict()
        bs = x[0].shape[0]
        boxes, cls_feats = [], []
        for i in range(self.nl):
            b, c = ta[i](x[i])
            if suppress is not None:
                c = suppress[i](c)
            boxes.append(b.view(bs, 4 * self.reg_max, -1))
            cls_feats.append(c.view(bs, self.nc, -1))
        return dict(boxes=torch.cat(boxes, dim=-1), scores=torch.cat(cls_feats, dim=-1), feats=x)

    def bias_init(self):
        """Initialize TA tower prediction biases (requires stride availability)."""
        heads = [self.ta] + ([self.one2one_ta] if self.end2end else [])
        for towers in heads:
            for i, t in enumerate(towers):
                t.box.bias.data[:] = 2.0
                t.cls.bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)

    def fuse(self):
        """Drop o2m TA towers for inference."""
        if not self.o2o_residual_head and not self.keep_one2many:
            self.ta = None


class DetectGC(Detect):
    """Detect with per-level global context (GCNet) on the head inputs.

    A GCAttn block per level adds a softmax-pooled scene-level context vector to every
    position before the box/cls branches: a cheap stand-in for the decoder's global
    attention. Gradients reach the GC blocks through the o2m branch only (o2o input is
    detached downstream, matching stock Detect behavior).
    """

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        """Initialize Detect then add one GCAttn per level."""
        super().__init__(nc, reg_max, end2end, ch)
        self.gc = nn.ModuleList(GCAttn(x) for x in ch)

    def forward(self, x):
        """Apply global context per level, then standard Detect forward."""
        return super().forward([g(xi) for g, xi in zip(self.gc, x)])


class SeedClsBranch(nn.Module):
    """Cls branch shaped like the decoder's encoder classifier: proj -> residual tower -> 1x1 cls.

    proj mirrors the decoder input_proj (1x1 Conv+BN to hd, Identity when channels already match);
    the tower is zero-init so at step zero the branch computes exactly proj -> classifier.
    """

    def __init__(self, c1, hd, nc):
        """Build projection, zero-init residual tower, and 1x1 classifier."""
        super().__init__()
        self.proj = nn.Identity() if c1 == hd else Conv(c1, hd, 1, act=False)
        self.tower = nn.Sequential(DWConv(hd, hd, 3), Conv(hd, hd, 1), DWConv(hd, hd, 3), nn.Conv2d(hd, hd, 1))
        nn.init.zeros_(self.tower[-1].weight)
        nn.init.zeros_(self.tower[-1].bias)
        self.cls = nn.Conv2d(hd, nc, 1)

    def forward(self, x):
        """Project, residual-refine, classify."""
        z = self.proj(x)
        return self.cls(z + self.tower(z))


class DetectSeed(Detect):
    """Detect with the cls branch transplanted from the decoder teacher's encoder classifier.

    Copies input_proj (P3/P5 1x1 Conv+BN, P4 Identity) and enc_score_head (Linear reshaped to 1x1
    conv) from the checkpoint named by the `seed_teacher` yaml key, so the dense cls logits equal
    the teacher's pre-top-k encoder logits at step zero; a zero-init residual tower then learns
    the correction. Box branch is stock Detect. Conv and add only.
    """

    seed_teacher = ""  # ckpt path, set via yaml `seed_teacher`

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        """Build parent, replace cls branches, and transplant teacher weights if configured."""
        super().__init__(nc, reg_max, end2end, ch)
        hd = 256
        del self.cv3
        self.cv3 = nn.ModuleList(SeedClsBranch(x, hd, nc) for x in ch)
        if self.seed_teacher:
            ck = torch.load(self.seed_teacher, map_location="cpu", weights_only=False)
            t = (ck.get("ema") or ck["model"]).float().model[-1]
            for i, m in enumerate(self.cv3):
                src = t.input_proj[i]
                if not isinstance(m.proj, nn.Identity):
                    m.proj.conv.weight.data.copy_(src[0].weight.data)
                    m.proj.bn.load_state_dict(src[1].state_dict())
                m.cls.weight.data.copy_(t.enc_score_head.weight.data.view(self.nc, hd, 1, 1))
                m.cls.bias.data.copy_(t.enc_score_head.bias.data)
        if end2end and self.one2one_cv3 is not None:
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def bias_init(self):
        """Box biases as usual; seeded cls biases are kept (else standard prior)."""
        heads = [(self.cv2, self.cv3)] + (
            [(self.one2one_cv2, self.one2one_cv3)] if self.end2end and not self.o2o_residual_head else []
        )
        for box_l, cls_l in heads:
            for i, (a, b) in enumerate(zip(box_l, cls_l)):
                self._box_bias(a[-1].bias)
                if not self.seed_teacher:
                    b.cls.bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)


class Segment(Detect):
    """YOLO Segment head for segmentation models.

    This class extends the Detect head to include mask prediction capabilities for instance segmentation tasks.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto): Prototype generation module.
        cv4 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a segmentation head
        >>> segment = Segment(nc=80, nm=32, npr=256, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = segment(x)
    """

    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, mask_head=self.one2one_cv4)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x[0])  # mask protos
        if isinstance(preds, dict):  # training and validating during training
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with mask coefficients."""
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, mask_head: torch.nn.Module
    ) -> torch.Tensor:
        """Concatenates and returns predicted bounding boxes, class probabilities, and mask coefficients."""
        preds = super().forward_head(x, box_head, cls_head)
        if mask_head is not None:
            bs = x[0].shape[0]  # batch size
            preds["mask_coefficient"] = torch.cat([mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + nm) with last dimension
                format [x, y, w, h, class_probs, mask_coefficient].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + nm) and last
                dimension format [x, y, w, h, max_class_prob, class_index, mask_coefficient].
        """
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = self.cv4 = None


class Segment26(Segment):
    """YOLO26 Segment head for segmentation models.

    This class extends the Detect head to include mask prediction capabilities for instance segmentation tasks.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto): Prototype generation module.
        cv4 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a segmentation head
        >>> segment = Segment26(nc=80, nm=32, npr=256, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = segment(x)
    """

    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, nm, npr, reg_max, end2end, ch)
        self.proto = Proto26(ch, self.npr, self.nm, nc)  # protos

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = Detect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x)  # mask protos
        if isinstance(preds, dict):  # training and validating during training
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = (
                    tuple(p.detach() for p in proto) if isinstance(proto, tuple) else proto.detach()
                )
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def fuse(self) -> None:
        """Remove the one2many head and extra part of proto module for inference optimization."""
        super().fuse()
        if hasattr(self.proto, "fuse"):
            self.proto.fuse()


class OBB(Detect):
    """YOLO OBB detection head for detection with rotation models.

    This class extends the Detect head to include oriented bounding box prediction with rotation angles.

    Attributes:
        ne (int): Number of extra parameters.
        cv4 (nn.ModuleList): Convolution layers for angle prediction.
        angle (torch.Tensor): Predicted rotation angles.

    Methods:
        forward: Concatenate and return predicted bounding boxes and class probabilities.
        decode_bboxes: Decode rotated bounding boxes.

    Examples:
        Create an OBB detection head
        >>> obb = OBB(nc=80, ne=1, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = obb(x)
    """

    def __init__(self, nc: int = 80, ne: int = 1, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`.

        Args:
            nc (int): Number of classes.
            ne (int): Number of extra parameters.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.ne = ne  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, angle_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, angle_head=self.one2one_cv4)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with rotation angles."""
        # For decode_bboxes convenience
        self.angle = x["angle"]  # TODO: need to test obb
        preds = super()._inference(x)
        return torch.cat([preds, x["angle"]], dim=1)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, angle_head: torch.nn.Module
    ) -> torch.Tensor:
        """Concatenates and returns predicted bounding boxes, class probabilities, and angles."""
        preds = super().forward_head(x, box_head, cls_head)
        if angle_head is not None:
            bs = x[0].shape[0]  # batch size
            angle = torch.cat(
                [angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
            )  # OBB theta logits
            angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
            preds["angle"] = angle
        return preds

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + ne) with last dimension
                format [x, y, w, h, class_probs, angle].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 7) and last
                dimension format [x, y, w, h, max_class_prob, class_index, angle].
        """
        boxes, scores, angle = preds.split([4, self.nc, self.ne], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        angle = angle.gather(dim=1, index=idx.repeat(1, 1, self.ne))
        return torch.cat([boxes, scores, conf, angle], dim=-1)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = self.cv4 = None


class OBB26(OBB):
    """YOLO26 OBB detection head for detection with rotation models. This class extends the OBB head with modified angle
    processing that outputs raw angle predictions without sigmoid transformation, compared to the original
    OBB class.

    Attributes:
        ne (int): Number of extra parameters.
        cv4 (nn.ModuleList): Convolution layers for angle prediction.
        angle (torch.Tensor): Predicted rotation angles.

    Methods:
        forward_head: Concatenate and return predicted bounding boxes, class probabilities, and raw angles.

    Examples:
        Create an OBB26 detection head
        >>> obb26 = OBB26(nc=80, ne=1, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = obb26(x).
    """

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, angle_head: torch.nn.Module
    ) -> torch.Tensor:
        """Concatenates and returns predicted bounding boxes, class probabilities, and raw angles."""
        preds = Detect.forward_head(self, x, box_head, cls_head)
        if angle_head is not None:
            bs = x[0].shape[0]  # batch size
            angle = torch.cat(
                [angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
            )  # OBB theta logits (raw output without sigmoid transformation)
            preds["angle"] = angle
        return preds


class Pose(Detect):
    """YOLO Pose head for keypoints models.

    This class extends the Detect head to include keypoint prediction capabilities for pose estimation tasks.

    Attributes:
        kpt_shape (tuple): Number of keypoints and dimensions (2 for x,y or 3 for x,y,visible).
        nk (int): Total number of keypoint values.
        cv4 (nn.ModuleList): Convolution layers for keypoint prediction.

    Methods:
        forward: Perform forward pass through YOLO model and return predictions.
        kpts_decode: Decode keypoints from predictions.

    Examples:
        Create a pose detection head
        >>> pose = Pose(nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = pose(x)
    """

    def __init__(self, nc: int = 80, kpt_shape: tuple = (17, 3), reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize YOLO network with default parameters and Convolutional Layers.

        Args:
            nc (int): Number of classes.
            kpt_shape (tuple): Number of keypoints, number of dims (2 for x,y or 3 for x,y,visible).
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, pose_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, pose_head=self.one2one_cv4)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with keypoints."""
        preds = super()._inference(x)
        return torch.cat([preds, self.kpts_decode(x["kpts"])], dim=1)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, pose_head: torch.nn.Module
    ) -> torch.Tensor:
        """Concatenates and returns predicted bounding boxes, class probabilities, and keypoints."""
        preds = super().forward_head(x, box_head, cls_head)
        if pose_head is not None:
            bs = x[0].shape[0]  # batch size
            preds["kpts"] = torch.cat([pose_head[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + nk) with last dimension
                format [x, y, w, h, class_probs, keypoints].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + self.nk) and
                last dimension format [x, y, w, h, max_class_prob, class_index, keypoints].
        """
        boxes, scores, kpts = preds.split([4, self.nc, self.nk], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        kpts = kpts.gather(dim=1, index=idx.repeat(1, 1, self.nk))
        return torch.cat([boxes, scores, conf, kpts], dim=-1)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = self.cv4 = None

    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        """Decode keypoints from predictions."""
        ndim = self.kpt_shape[1]
        bs = kpts.shape[0]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                if NOT_MACOS14:
                    y[:, 2::ndim].sigmoid_()
                else:  # Apple macOS14 MPS bug https://github.com/ultralytics/ultralytics/pull/21878
                    y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Pose26(Pose):
    """YOLO26 Pose head for keypoints models.

    This class extends the Detect head to include keypoint prediction capabilities for pose estimation tasks.

    Attributes:
        kpt_shape (tuple): Number of keypoints and dimensions (2 for x,y or 3 for x,y,visible).
        nk (int): Total number of keypoint values.
        cv4 (nn.ModuleList): Convolution layers for keypoint prediction.

    Methods:
        forward: Perform forward pass through YOLO model and return predictions.
        kpts_decode: Decode keypoints from predictions.

    Examples:
        Create a pose detection head
        >>> pose = Pose(nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = pose(x)
    """

    def __init__(self, nc: int = 80, kpt_shape: tuple = (17, 3), reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize YOLO network with default parameters and Convolutional Layers.

        Args:
            nc (int): Number of classes.
            kpt_shape (tuple): Number of keypoints, number of dims (2 for x,y or 3 for x,y,visible).
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, kpt_shape, reg_max, end2end, ch)
        self.flow_model = RealNVP()

        c4 = max(ch[0] // 4, kpt_shape[0] * (kpt_shape[1] + 2))
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3)) for x in ch)

        self.cv4_kpts = nn.ModuleList(nn.Conv2d(c4, self.nk, 1) for _ in ch)
        self.nk_sigma = kpt_shape[0] * 2  # sigma_x, sigma_y for each keypoint
        self.cv4_sigma = nn.ModuleList(nn.Conv2d(c4, self.nk_sigma, 1) for _ in ch)

        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)
            self.one2one_cv4_kpts = copy.deepcopy(self.cv4_kpts)
            self.one2one_cv4_sigma = copy.deepcopy(self.cv4_sigma)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(
            box_head=self.cv2,
            cls_head=self.cv3,
            pose_head=self.cv4,
            kpts_head=self.cv4_kpts,
            kpts_sigma_head=self.cv4_sigma,
        )

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            pose_head=self.one2one_cv4,
            kpts_head=self.one2one_cv4_kpts,
            kpts_sigma_head=self.one2one_cv4_sigma,
        )

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        pose_head: torch.nn.Module,
        kpts_head: torch.nn.Module,
        kpts_sigma_head: torch.nn.Module,
    ) -> torch.Tensor:
        """Concatenates and returns predicted bounding boxes, class probabilities, and keypoints."""
        preds = Detect.forward_head(self, x, box_head, cls_head)
        if pose_head is not None:
            bs = x[0].shape[0]  # batch size
            features = [pose_head[i](x[i]) for i in range(self.nl)]
            preds["kpts"] = torch.cat([kpts_head[i](features[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)
            if self.training:
                preds["kpts_sigma"] = torch.cat(
                    [kpts_sigma_head[i](features[i]).view(bs, self.nk_sigma, -1) for i in range(self.nl)], 2
                )
        return preds

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        super().fuse()
        self.cv4_kpts = self.cv4_sigma = self.flow_model = self.one2one_cv4_sigma = None

    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        """Decode keypoints from predictions."""
        ndim = self.kpt_shape[1]
        bs = kpts.shape[0]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            # NCNN fix
            a = (y[:, :, :2] + self.anchors) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                if NOT_MACOS14:
                    y[:, 2::ndim].sigmoid_()
                else:  # Apple macOS14 MPS bug https://github.com/ultralytics/ultralytics/pull/21878
                    y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] + self.anchors[0]) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] + self.anchors[1]) * self.strides
            return y


class Classify(nn.Module):
    """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2).

    This class implements a classification head that transforms feature maps into class predictions.

    Attributes:
        export (bool): Export mode flag.
        conv (Conv): Convolutional layer for feature transformation.
        pool (nn.AdaptiveAvgPool2d): Global average pooling layer.
        drop (nn.Dropout): Dropout layer for regularization.
        linear (nn.Linear): Linear layer for final classification.

    Methods:
        forward: Perform forward pass of the YOLO model on input image data.

    Examples:
        Create a classification head
        >>> classify = Classify(c1=1024, c2=1000)
        >>> x = torch.randn(1, 1024, 20, 20)
        >>> output = classify(x)
    """

    export = False  # export mode

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize YOLO classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output classes.
            k (int, optional): Kernel size.
            s (int, optional): Stride.
            p (int, optional): Padding.
            g (int, optional): Groups.
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor | tuple:
        """Perform forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)  # get final output
        return y if self.export else (y, x)


class WorldDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings.

    This class extends the standard Detect head to incorporate text embeddings for enhanced semantic understanding in
    object detection tasks.

    Attributes:
        cv3 (nn.ModuleList): Convolution layers for embedding features.
        cv4 (nn.ModuleList): Contrastive head layers for text-vision alignment.

    Methods:
        forward: Concatenate and return predicted bounding boxes and class probabilities.
        bias_init: Initialize detection head biases.

    Examples:
        Create a WorldDetect head
        >>> world_detect = WorldDetect(nc=80, embed=512, with_bn=False, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> text = torch.randn(1, 80, 512)
        >>> outputs = world_detect(x, text)
    """

    def __init__(
        self,
        nc: int = 80,
        embed: int = 512,
        with_bn: bool = False,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        """Initialize YOLO detection layer with nc classes and layer channels ch.

        Args:
            nc (int): Number of classes.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max=reg_max, end2end=end2end, ch=ch)
        c3 = max(ch[0], (self.fixed_c3 or min(self.nc, 100)))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> dict[str, torch.Tensor] | tuple:
        """Concatenate and return predicted bounding boxes and class probabilities."""
        feats = [xi.clone() for xi in x]  # save original features for anchor generation
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        bs = x[0].shape[0]
        x_cat = torch.cat([xi.view(bs, self.no, -1) for xi in x], 2)
        boxes, scores = x_cat.split((self.reg_max * 4, self.nc), 1)
        preds = dict(boxes=boxes, scores=scores, feats=feats)
        if self.training:
            return preds
        y = self._inference(preds)
        return y if self.export else (y, preds)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class LRPCHead(nn.Module):
    """Lightweight Region Proposal and Classification Head for efficient object detection.

    This head combines region proposal filtering with classification to enable efficient detection with dynamic
    vocabulary support.

    Attributes:
        vocab (nn.Module): Vocabulary/classification layer.
        pf (nn.Module): Proposal filter module.
        loc (nn.Module): Localization module.
        enabled (bool): Whether the head is enabled.

    Methods:
        conv2linear: Convert a 1x1 convolutional layer to a linear layer.
        forward: Process classification and localization features to generate detection proposals.

    Examples:
        Create an LRPC head
        >>> vocab = nn.Conv2d(256, 80, 1)
        >>> pf = nn.Conv2d(256, 1, 1)
        >>> loc = nn.Conv2d(256, 4, 1)
        >>> head = LRPCHead(vocab, pf, loc, enabled=True)
    """

    def __init__(self, vocab: nn.Module, pf: nn.Module, loc: nn.Module, enabled: bool = True):
        """Initialize LRPCHead with vocabulary, proposal filter, and localization components.

        Args:
            vocab (nn.Module): Vocabulary/classification module.
            pf (nn.Module): Proposal filter module.
            loc (nn.Module): Localization module.
            enabled (bool): Whether to enable the head functionality.
        """
        super().__init__()
        self.vocab = self.conv2linear(vocab) if enabled else vocab
        self.pf = pf
        self.loc = loc
        self.enabled = enabled

    @staticmethod
    def conv2linear(conv: nn.Conv2d) -> nn.Linear:
        """Convert a 1x1 convolutional layer to a linear layer."""
        assert isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1)
        linear = nn.Linear(conv.in_channels, conv.out_channels)
        linear.weight.data = conv.weight.view(conv.out_channels, -1).data
        linear.bias.data = conv.bias.data
        return linear

    def forward(self, cls_feat: torch.Tensor, loc_feat: torch.Tensor, conf: float) -> tuple[tuple, torch.Tensor]:
        """Process classification and localization features to generate detection proposals."""
        if self.enabled:
            pf_score = self.pf(cls_feat)[0, 0].flatten(0)
            mask = pf_score.sigmoid() > conf
            cls_feat = cls_feat.flatten(2).transpose(-1, -2)
            cls_feat = self.vocab(cls_feat[:, mask] if conf else cls_feat * mask.unsqueeze(-1).int())
            return self.loc(loc_feat), cls_feat.transpose(-1, -2), mask
        else:
            cls_feat = self.vocab(cls_feat)
            loc_feat = self.loc(loc_feat)
            return (
                loc_feat,
                cls_feat.flatten(2),
                torch.ones(cls_feat.shape[2] * cls_feat.shape[3], device=cls_feat.device, dtype=torch.bool),
            )


class YOLOEDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings.

    This class extends the standard Detect head to support text-guided detection with enhanced semantic understanding
    through text embeddings and visual prompt embeddings.

    Attributes:
        is_fused (bool): Whether the model is fused for inference.
        cv3 (nn.ModuleList): Convolution layers for embedding features.
        cv4 (nn.ModuleList): Contrastive head layers for text-vision alignment.
        reprta (Residual): Residual block for text prompt embeddings.
        savpe (SAVPE): Spatial-aware visual prompt embeddings module.
        embed (int): Embedding dimension.

    Methods:
        fuse: Fuse text features with model weights for efficient inference.
        get_tpe: Get text prompt embeddings with normalization.
        get_vpe: Get visual prompt embeddings with spatial awareness.
        forward_lrpc: Process features with fused text embeddings for prompt-free model.
        forward: Process features with class prompt embeddings to generate detections.
        bias_init: Initialize biases for detection heads.

    Examples:
        Create a YOLOEDetect head
        >>> yoloe_detect = YOLOEDetect(nc=80, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> cls_pe = torch.randn(1, 80, 512)
        >>> outputs = yoloe_detect(x, cls_pe)
    """

    is_fused = False

    def __init__(
        self, nc: int = 80, embed: int = 512, with_bn: bool = False, reg_max=16, end2end=False, ch: tuple = ()
    ):
        """Initialize YOLO detection layer with nc classes and layer channels ch.

        Args:
            nc (int): Number of classes.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        c3 = max(ch[0], (self.fixed_c3 or min(self.nc, 100)))
        assert c3 <= embed
        assert with_bn
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, embed, 1),
                )
                for x in ch
            )
        )
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)
        if end2end:
            self.one2one_cv3 = copy.deepcopy(self.cv3)  # overwrite with new cv3
            self.one2one_cv4 = copy.deepcopy(self.cv4)

        self.reprta = Residual(SwiGLUFFN(embed, embed))
        self.savpe = SAVPE(ch, c3, embed)
        self.embed = embed

    @smart_inference_mode()
    def fuse(self, txt_feats: torch.Tensor = None):
        """Fuse text features with model weights for efficient inference."""
        if txt_feats is None:  # means eliminate one2many branch
            self.cv2 = self.cv3 = self.cv4 = None
            return
        if self.is_fused:
            return

        assert not self.training
        txt_feats = txt_feats.to(torch.float32).squeeze(0)
        self._fuse_tp(txt_feats, self.cv3, self.cv4)
        if self.end2end:
            self._fuse_tp(txt_feats, self.one2one_cv3, self.one2one_cv4)
        del self.reprta
        self.reprta = nn.Identity()
        self.is_fused = True

    def _fuse_tp(self, txt_feats: torch.Tensor, cls_head: torch.nn.Module, bn_head: torch.nn.Module) -> None:
        """Fuse text prompt embeddings with model weights for efficient inference."""
        for cls_h, bn_h in zip(cls_head, bn_head):
            assert isinstance(cls_h, nn.Sequential)
            assert isinstance(bn_h, BNContrastiveHead)
            conv = cls_h[-1]
            assert isinstance(conv, nn.Conv2d)
            logit_scale = bn_h.logit_scale
            bias = bn_h.bias
            norm = bn_h.norm

            t = txt_feats * logit_scale.exp()
            conv: nn.Conv2d = fuse_conv_and_bn(conv, norm)

            w = conv.weight.data.squeeze(-1).squeeze(-1)
            b = conv.bias.data

            w = t @ w
            b1 = (t @ b.reshape(-1).unsqueeze(-1)).squeeze(-1)
            b2 = torch.ones_like(b1) * bias

            conv = (
                nn.Conv2d(
                    conv.in_channels,
                    w.shape[0],
                    kernel_size=1,
                )
                .requires_grad_(False)
                .to(conv.weight.device)
            )

            conv.weight.data.copy_(w.unsqueeze(-1).unsqueeze(-1))
            conv.bias.data.copy_(b1 + b2)
            cls_h[-1] = conv

            bn_h.fuse()

    def get_tpe(self, tpe: torch.Tensor | None) -> torch.Tensor | None:
        """Get text prompt embeddings with normalization."""
        return None if tpe is None else F.normalize(self.reprta(tpe), dim=-1, p=2)

    def get_vpe(self, x: list[torch.Tensor], vpe: torch.Tensor) -> torch.Tensor:
        """Get visual prompt embeddings with spatial awareness."""
        if vpe.shape[1] == 0:  # no visual prompt embeddings
            return torch.zeros(x[0].shape[0], 0, self.embed, device=x[0].device)
        if vpe.ndim == 4:  # (B, N, H, W)
            vpe = self.savpe(x, vpe)
        assert vpe.ndim == 3  # (B, N, D)
        return vpe

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Process features with class prompt embeddings to generate detections."""
        if hasattr(self, "lrpc"):  # for prompt-free inference
            return self.forward_lrpc(x[:3])
        return super().forward(x)

    def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Process features with fused text embeddings to generate detections for prompt-free model."""
        boxes, scores, index = [], [], []
        bs = x[0].shape[0]
        cv2 = self.cv2 if not self.end2end else self.one2one_cv2
        cv3 = self.cv3 if not self.end2end else self.one2one_cv2
        for i in range(self.nl):
            cls_feat = cv3[i](x[i])
            loc_feat = cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            box, score, idx = self.lrpc[i](
                cls_feat,
                loc_feat,
                0 if self.export and not self.dynamic else getattr(self, "conf", 0.001),
            )
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)
        preds = dict(boxes=torch.cat(boxes, 2), scores=torch.cat(scores, 2), feats=x, index=torch.cat(index))
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _get_decode_boxes(self, x):
        """Decode predicted bounding boxes for inference."""
        dbox = super()._get_decode_boxes(x)
        if hasattr(self, "lrpc"):
            dbox = dbox if self.export and not self.dynamic else dbox[..., x["index"]]
        return dbox

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v5/v5/v8/v9/11 backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, contrastive_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, contrastive_head=self.one2one_cv4)

    def forward_head(self, x, box_head, cls_head, contrastive_head):
        """Concatenates and returns predicted bounding boxes, class probabilities, and text embeddings."""
        assert len(x) == 4, f"Expected 4 features including 3 feature maps and 1 text embeddings, but got {len(x)}."
        if box_head is None or cls_head is None:  # for fused inference
            return dict()
        bs = x[0].shape[0]  # batch size
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        self.nc = x[-1].shape[1]
        scores = torch.cat(
            [contrastive_head[i](cls_head[i](x[i]), x[-1]).reshape(bs, self.nc, -1) for i in range(self.nl)], dim=-1
        )
        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        return dict(boxes=boxes, scores=scores, feats=x[:3])

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        for i, (a, b, c) in enumerate(
            zip(self.one2many["box_head"], self.one2many["cls_head"], self.one2many["contrastive_head"])
        ):
            a[-1].bias.data[:] = 2.0  # box
            b[-1].bias.data[:] = 0.0
            c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, (a, b, c) in enumerate(
                zip(self.one2one["box_head"], self.one2one["cls_head"], self.one2one["contrastive_head"])
            ):
                a[-1].bias.data[:] = 2.0  # box
                b[-1].bias.data[:] = 0.0
                c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)


class YOLOESegment(YOLOEDetect):
    """YOLO segmentation head with text embedding capabilities.

    This class extends YOLOEDetect to include mask prediction capabilities for instance segmentation tasks with
    text-guided semantic understanding.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto): Prototype generation module.
        cv5 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a YOLOESegment head
        >>> yoloe_segment = YOLOESegment(nc=80, nm=32, npr=256, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> text = torch.randn(1, 80, 512)
        >>> outputs = yoloe_segment(x, text)
    """

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = False,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        """Initialize YOLOESegment with class count, mask parameters, and embedding dimensions.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, embed, with_bn, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)

        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v5/v5/v8/v9/11 backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv5, contrastive_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            mask_head=self.one2one_cv5,
            contrastive_head=self.one2one_cv4,
        )

    def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Process features with fused text embeddings to generate detections for prompt-free model."""
        boxes, scores, index = [], [], []
        bs = x[0].shape[0]
        cv2 = self.cv2 if not self.end2end else self.one2one_cv2
        cv3 = self.cv3 if not self.end2end else self.one2one_cv3
        cv5 = self.cv5 if not self.end2end else self.one2one_cv5
        for i in range(self.nl):
            cls_feat = cv3[i](x[i])
            loc_feat = cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            box, score, idx = self.lrpc[i](
                cls_feat,
                loc_feat,
                0 if self.export and not self.dynamic else getattr(self, "conf", 0.001),
            )
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)
        mc = torch.cat([cv5[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        index = torch.cat(index)
        preds = dict(
            boxes=torch.cat(boxes, 2),
            scores=torch.cat(scores, 2),
            feats=x,
            index=index,
            mask_coefficient=mc * index.int() if self.export and not self.dynamic else mc[..., index],
        )
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x[0])  # mask protos
        if isinstance(preds, dict):  # training and validating during training
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with mask coefficients."""
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        mask_head: torch.nn.Module,
        contrastive_head: torch.nn.Module,
    ) -> torch.Tensor:
        """Concatenates and returns predicted bounding boxes, class probabilities, and mask coefficients."""
        preds = super().forward_head(x, box_head, cls_head, contrastive_head)
        if mask_head is not None:
            bs = x[0].shape[0]  # batch size
            preds["mask_coefficient"] = torch.cat([mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + nm) with last dimension
                format [x, y, w, h, class_probs, mask_coefficient].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + nm) and last
                dimension format [x, y, w, h, max_class_prob, class_index, mask_coefficient].
        """
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def fuse(self, txt_feats: torch.Tensor = None):
        """Fuse text features with model weights for efficient inference."""
        super().fuse(txt_feats)
        if txt_feats is None:  # means eliminate one2many branch
            self.cv5 = None
            if hasattr(self.proto, "fuse"):
                self.proto.fuse()
            return


class YOLOESegment26(YOLOESegment):
    """YOLOE-style segmentation head module using Proto26 for mask generation.

    This class extends the YOLOEDetect functionality to include segmentation capabilities by integrating a prototype
    generation module and convolutional layers to predict mask coefficients.

    Args:
        nc (int): Number of classes. Defaults to 80.
        nm (int): Number of masks. Defaults to 32.
        npr (int): Number of prototype channels. Defaults to 256.
        embed (int): Embedding dimensionality. Defaults to 512.
        with_bn (bool): Whether to use Batch Normalization. Defaults to False.
        reg_max (int): Maximum regression value for bounding boxes. Defaults to 16.
        end2end (bool): Whether to use end-to-end detection mode. Defaults to False.
        ch (tuple[int, ...]): Input channels for each scale.

    Attributes:
        nm (int): Number of segmentation masks.
        npr (int): Number of prototype channels.
        proto (Proto26): Prototype generation module for segmentation.
        cv5 (nn.ModuleList): Convolutional layers for generating mask coefficients from features.
        one2one_cv5 (nn.ModuleList, optional): Deep copy of cv5 for end-to-end detection branches.
    """

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = False,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        """Initialize YOLOESegment26 with class count, mask parameters, and embedding dimensions."""
        YOLOEDetect.__init__(self, nc, embed, with_bn, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto26(ch, self.npr, self.nm, nc)  # protos

        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = YOLOEDetect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto([xi.detach() for xi in x], return_semseg=False)  # mask protos

        if isinstance(preds, dict):  # training and validating during training
            if self.end2end and not hasattr(self, "lrpc"):  # not prompt-free
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)


class RTDETRDecoder(nn.Module):
    """Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.

    Attributes:
        export (bool): Export mode flag.
        hidden_dim (int): Dimension of hidden layers.
        nhead (int): Number of heads in multi-head attention.
        nl (int): Number of feature levels.
        nc (int): Number of classes.
        num_queries (int): Number of query points.
        num_decoder_layers (int): Number of decoder layers.
        input_proj (nn.ModuleList): Input projection layers for backbone features.
        decoder (DeformableTransformerDecoder): Transformer decoder module.
        denoising_class_embed (nn.Embedding): Class embeddings for denoising.
        num_denoising (int): Number of denoising queries.
        label_noise_ratio (float): Label noise ratio for training.
        box_noise_scale (float): Box noise scale for training.
        learnt_init_query (bool): Whether to learn initial query embeddings.
        tgt_embed (nn.Embedding): Target embeddings for queries.
        query_pos_head (MLP): Query position head.
        enc_output (nn.Sequential): Encoder output layers.
        enc_score_head (nn.Linear): Encoder score prediction head.
        enc_bbox_head (MLP): Encoder bbox prediction head.
        dec_score_head (nn.ModuleList): Decoder score prediction heads.
        dec_bbox_head (nn.ModuleList): Decoder bbox prediction heads.

    Methods:
        forward: Run forward pass and return bounding box and classification scores.

    Examples:
        Create an RTDETRDecoder
        >>> decoder = RTDETRDecoder(nc=80, ch=(512, 1024, 2048), hd=256, nq=300)
        >>> x = [torch.randn(1, 512, 64, 64), torch.randn(1, 1024, 32, 32), torch.randn(1, 2048, 16, 16)]
        >>> outputs = decoder(x)
    """

    export = False  # export mode
    max_det = 300  # max detections per image
    shapes = []
    anchors = torch.empty(0)
    valid_mask = torch.empty(0)
    dynamic = False

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,  # hidden dim
        nq: int = 300,  # num queries
        ndp: int = 4,  # num decoder points
        nh: int = 8,  # num head
        ndl: int = 6,  # num decoder layers
        d_ffn: int = 1024,  # dim of feedforward
        dropout: float = 0.0,
        act: nn.Module | None = None,
        eval_idx: int = -1,
        # Training args
        nd: int = 100,  # num denoising
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
    ):
        """Initialize the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes.
            ch (tuple): Channels in the backbone feature maps.
            hd (int): Dimension of hidden layers.
            nq (int): Number of query points.
            ndp (int): Number of decoder points.
            nh (int): Number of heads in multi-head attention.
            ndl (int): Number of decoder layers.
            d_ffn (int): Dimension of the feed-forward networks.
            dropout (float): Dropout rate.
            act (nn.Module): Activation function.
            eval_idx (int): Evaluation index.
            nd (int): Number of denoising.
            label_noise_ratio (float): Label noise ratio.
            box_noise_scale (float): Box noise scale.
            learnt_init_query (bool): Whether to learn initial query embeddings.
        """
        super().__init__()
        act = nn.ReLU() if act is None else act
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = self._build_input_proj(ch, hd)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = self._build_query_pos_head(hd)

        # Encoder head
        self.enc_output = self._build_enc_output(hd)
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = self._build_bbox_head(hd)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList(self._build_bbox_head(hd) for _ in range(ndl))

        self._reset_parameters()

    def forward(self, x: list[torch.Tensor], batch: dict | None = None) -> tuple | torch.Tensor:
        """Run the forward pass of the module, returning bounding box and classification scores for the input.

        Args:
            x (list[torch.Tensor]): List of feature maps from the backbone.
            batch (dict, optional): Batch information for training.

        Returns:
            outputs (tuple | torch.Tensor): During training, returns a tuple of bounding boxes, scores, and other
                metadata. During inference, returns a tensor of shape (bs, num_queries, 6) containing bounding boxes,
                confidence scores, and class labels.
        """
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        if self.training and dn_meta is None:
            # Touch denoising_class_embed so DDP sees it as used when batch has zero GTs.
            dec_bboxes = dec_bboxes + 0 * self.denoising_class_embed.weight.sum()
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, num_queries, 4), (bs, num_queries, nc)
        y = self.postprocess(dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid())
        return y if self.export else (y, x)

    def postprocess(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Post-process predictions to select top-k detections.

        Args:
            boxes (torch.Tensor): Predicted bounding boxes with shape (batch_size, num_queries, 4) in xywh format.
            scores (torch.Tensor): Class scores with shape (batch_size, num_queries, nc).

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, num_queries, 6), limited to max_det during
                export, and last dimension format [cx, cy, w, h, max_class_prob, class_index].
        """
        k = min(self.num_queries, self.max_det) if self.export else self.num_queries
        scores, index = scores.flatten(1).topk(k)
        # CoreML MIL lacks integer floor-div and mod lowering: use torch.div(rounding_mode="floor") and (index - q*nc).
        query_idx = torch.div(index, self.nc, rounding_mode="floor")
        boxes = boxes.gather(dim=1, index=query_idx.unsqueeze(-1).expand(-1, -1, 4).long())
        return torch.cat([boxes, scores[..., None], (index - query_idx * self.nc)[..., None].float()], dim=-1)

    @staticmethod
    def _generate_anchors(
        shapes: list[list[int]],
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        eps: float = 1e-2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate anchor bounding boxes for given shapes with specific grid size and validate them.

        Args:
            shapes (list): List of feature map shapes.
            grid_size (float, optional): Base size of grid cells.
            dtype (torch.dtype, optional): Data type for tensors.
            device (str, optional): Device to create tensors on.
            eps (float, optional): Small value for numerical stability.

        Returns:
            anchors (torch.Tensor): Generated anchor boxes.
            valid_mask (torch.Tensor): Valid mask for anchors.
        """
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, list[list[int]]]:
        """Process and return encoder inputs by getting projection features from input and concatenating them.

        Args:
            x (list[torch.Tensor]): List of feature maps from the backbone.

        Returns:
            feats (torch.Tensor): Processed features.
            shapes (list): List of feature map shapes.
        """
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _project_encoder_features(self, feats: torch.Tensor) -> torch.Tensor:
        """Project masked encoder memory into the query-selection space; override to skip enc_output."""
        return self.enc_output(self.valid_mask * feats)

    def _get_decoder_input(
        self,
        feats: torch.Tensor,
        shapes: list[list[int]],
        dn_embed: torch.Tensor | None = None,
        dn_bbox: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate and prepare the input required for the decoder from the provided features and shapes.

        Args:
            feats (torch.Tensor): Processed features from encoder.
            shapes (list): List of feature map shapes.
            dn_embed (torch.Tensor, optional): Denoising embeddings.
            dn_bbox (torch.Tensor, optional): Denoising bounding boxes.

        Returns:
            embeddings (torch.Tensor): Query embeddings for decoder.
            refer_bbox (torch.Tensor): Reference bounding boxes.
            enc_bboxes (torch.Tensor): Encoded bounding boxes.
            enc_scores (torch.Tensor): Encoded scores.
        """
        bs = feats.shape[0]
        if self.dynamic or self.shapes != shapes:
            self.anchors, self.valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
            self.shapes = shapes

        features = self._project_encoder_features(feats)  # bs, h*w, 256
        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs*num_queries,)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs*num_queries,)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = self.anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    def _build_input_proj(self, ch: tuple, hd: int) -> nn.ModuleList:
        """Build the per-level backbone feature projection; override to change projection (e.g. skip when x == hd)."""
        # NOTE: the simplified `nn.ModuleList(Conv(x, hd, act=False) for x in ch)` is not consistent with .pt weights.
        return nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)

    def _build_query_pos_head(self, hd: int) -> "MLP":
        """Build the reference-box position MLP; override to change depth, width, or activation."""
        return MLP(4, 2 * hd, hd, num_layers=2)

    def _build_enc_output(self, hd: int) -> nn.Module:
        """Build the encoder-memory projection applied before query selection; override to skip (nn.Identity)."""
        return nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))

    def _build_bbox_head(self, hd: int) -> "MLP":
        """Build one 3-layer bbox-regression MLP (reused for enc head and each decoder layer); override for act."""
        return MLP(hd, hd, 4, num_layers=3)

    def _reset_parameters(self):
        """Initialize or reset the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class RTDETRDecoderEfficient(RTDETRDecoder):
    """RT-DETR decoder with v2-style efficiency tweaks on the base MSDeformAttn path.

    Signature is the base `RTDETRDecoder` params plus `efficient_ms`. Behavioral changes over `RTDETRDecoder` are
    expressed by overriding the parent's `_build_*` factory hooks so the correct submodules are constructed once (no
    build-then-replace):

    - `input_proj`: `nn.Identity()` when a backbone channel already matches `hd`, else `Conv2d(1x1, bias=False) + BN`.
    - `query_pos_head`: DEIM-style 3-layer `MLP(4, hd, hd)` (vs base's 2-layer `MLP(4, 2*hd, hd)`).
    - `enc_output`: `nn.Identity()`; encoder features feed `enc_score_head` directly via `_project_encoder_features`.
    - `enc_bbox_head`/`dec_bbox_head`: built with the ctor `act` (origin's `mlp_act`, silu on Efficient YAMLs).
    - `decoder.fixed_query_pos = True` hoists `pos_mlp(refer_bbox)` out of the per-layer loop.
    - `efficient_ms=True` rebuilds the decoder with `n_levels=1` cross-attention and round-robin per-layer scheduling.

    `learnt_init_query=True` is rejected: the enc_output skip leaves no matching-init path.

    Examples:
        >>> decoder = RTDETRDecoderEfficient(nc=80, ch=(512, 1024, 2048), hd=256, nq=300, efficient_ms=True)
    """

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,
        nq: int = 300,
        ndp: int = 4,
        nh: int = 8,
        ndl: int = 6,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        eval_idx: int = -1,
        nd: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
        efficient_ms: bool = False,
    ):
        """Initialize the RTDETRDecoderEfficient module.

        Args:
            nc (int): Number of classes.
            ch (tuple): Channels in the backbone feature maps.
            hd (int): Dimension of hidden layers.
            nq (int): Number of query points.
            ndp (int): Number of decoder points per attention head per level.
            nh (int): Number of heads in multi-head attention.
            ndl (int): Number of decoder layers.
            d_ffn (int): Dimension of the feed-forward networks.
            dropout (float): Dropout rate.
            act (nn.Module): Activation function.
            eval_idx (int): Evaluation index.
            nd (int): Number of denoising.
            label_noise_ratio (float): Label noise ratio.
            box_noise_scale (float): Box noise scale.
            learnt_init_query (bool): Unsupported (raises); enc_output is skipped so init pathway does not match.
            efficient_ms (bool): Enable round-robin single-level cross-attention per decoder layer.
        """
        if learnt_init_query:
            raise ValueError("RTDETRDecoderEfficient does not support learnt_init_query=True.")
        if isinstance(act, str):
            act = {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}[act]
        # Store the activation class for the _build_* hooks invoked during super().__init__().
        self._act_cls = type(act) if isinstance(act, nn.Module) else act
        super().__init__(
            nc,
            ch,
            hd,
            nq,
            ndp,
            nh,
            ndl,
            d_ffn,
            dropout,
            act,
            eval_idx,
            nd,
            label_noise_ratio,
            box_noise_scale,
            learnt_init_query,
        )
        self.efficient_ms = efficient_ms
        if efficient_ms:
            self.nl = 1
            decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, 1, ndp)
            self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx, efficient_ms=True)
        # Hoist query_pos out of the decoder layer loop (compute once from initial refer_bbox).
        self.decoder.fixed_query_pos = True

    def _build_input_proj(self, ch: tuple, hd: int) -> nn.ModuleList:
        """Skip the 1x1 conv projection when a backbone channel already matches hd."""
        return nn.ModuleList(
            nn.Identity() if x == hd else nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch
        )

    def _build_query_pos_head(self, hd: int) -> "MLP":
        """DEIM-style 3-layer query_pos MLP with the head's own activation."""
        return MLP(4, hd, hd, num_layers=3, act=self._act_cls)

    def _build_enc_output(self, hd: int) -> nn.Module:
        """Skip the encoder-memory projection; _project_encoder_features scores from masked memory directly."""
        return nn.Identity()

    def _build_bbox_head(self, hd: int) -> "MLP":
        """Build the bbox-regression MLP with the head's own activation (origin's `mlp_act`) instead of base ReLU."""
        return MLP(hd, hd, 4, num_layers=3, act=self._act_cls)

    def _reset_parameters(self):
        """Initialize parameters; skips enc_output (Identity) and input_proj (may contain nn.Identity)."""
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)
        xavier_uniform_(self.query_pos_head.layers[0].weight)

    def _project_encoder_features(self, feats: torch.Tensor) -> torch.Tensor:
        """Skip enc_output projection; score directly from masked encoder memory."""
        return self.valid_mask.to(feats.dtype) * feats


class v10Detect(Detect):
    """v10 Detection head from https://arxiv.org/pdf/2405.14458.

    This class implements the YOLOv10 detection head with dual-assignment training and consistent dual predictions for
    improved efficiency and performance.

    Attributes:
        end2end (bool): End-to-end detection mode.
        max_det (int): Maximum number of detections.
        cv3 (nn.ModuleList): Light classification head layers.
        one2one_cv3 (nn.ModuleList): One-to-one classification head layers.

    Methods:
        __init__: Initialize the v10Detect object with specified number of classes and input channels.
        forward: Perform forward pass of the v10Detect module.
        bias_init: Initialize biases of the Detect module.
        fuse: Remove the one2many head for inference optimization.

    Examples:
        Create a v10Detect head
        >>> v10_detect = v10Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = v10_detect(x)
    """

    end2end = True

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """Initialize the v10Detect object with the specified number of classes and input channels.

        Args:
            nc (int): Number of classes.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, end2end=True, ch=ch)
        c3 = max(ch[0], (self.fixed_c3 or min(self.nc, 100)))  # channels
        # Light cls head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)

    def fuse(self):
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = None


class DeimDecoder(RTDETRDecoder):
    """DEIMv2 decoder head with DEIM transformer layers and integral-based bbox refinement."""

    @staticmethod
    def _select_activation(act: str) -> nn.Module:
        """Map activation name to nn.Module."""
        if act == "relu":
            return nn.ReLU()
        if act == "gelu":
            return nn.GELU()
        if act == "silu":
            return nn.SiLU()
        raise ValueError(f"Unsupported activation function: {act}")

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,
        nq: int = 300,
        ndp: int = 4,
        nh: int = 8,
        ndl: int = 6,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: str = "relu",
        eval_idx: int = -1,
        nd: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
        query_select_method: str = "default",
        reg_max: int = 32,
        reg_scale: float = 4.0,
        layer_scale: float = 1.0,
        mlp_act: str = "relu",
        o2m_topk_mode: str = "unshared",
        use_gateway: bool = True,
        share_bbox_head: bool = False,
        share_score_head: bool = False,
        use_rmsnorm: bool = True,
    ):
        nn.Module.__init__(self)
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl
        self.reg_max = reg_max
        self.layer_scale = layer_scale
        self.query_select_method = query_select_method
        self.query_noise_scale = 0.0
        self.o2m_topk_mode = o2m_topk_mode
        self.learnt_init_query = learnt_init_query
        if self.learnt_init_query:
            raise ValueError("DeimDecoder does not support learnt_init_query=True.")

        if self.query_select_method not in {"default", "one2many"}:
            raise ValueError(f"Unsupported query_select_method: {self.query_select_method}")

        act_layer = self._select_activation(act)
        act_mlp = self._select_activation(mlp_act)
        scaled_dim = round(layer_scale * hd)

        self.input_proj = nn.ModuleList(
            nn.Identity() if x == hd else nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch
        )

        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        decoder_layer = DeimTransformerDecoderLayer(
            hd,
            nh,
            d_ffn,
            dropout,
            act_layer,
            self.nl,
            ndp,
            use_gateway=use_gateway,
            use_rmsnorm=use_rmsnorm,
        )
        decoder_layer_wide = DeimTransformerDecoderLayer(
            hd,
            nh,
            d_ffn,
            dropout,
            act_layer,
            self.nl,
            ndp,
            layer_scale=layer_scale if layer_scale > 1 else None,
            use_gateway=use_gateway,
            use_rmsnorm=use_rmsnorm,
        )
        self.decoder = DeimTransformerDecoder(
            hd,
            decoder_layer,
            decoder_layer_wide,
            ndl,
            nh,
            reg_max,
            self.reg_scale,
            self.up,
            eval_idx,
            layer_scale,
            act=act_layer,
        )

        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)

        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3, act=act_mlp)
        self.query_pos_head = MLP(4, hd, hd, num_layers=3, act=act_mlp)

        self.pre_bbox_head = MLP(hd, hd, 4, num_layers=3, act=act_mlp)
        self.integral = Integral(reg_max)

        self.eval_idx = eval_idx if eval_idx >= 0 else ndl + eval_idx
        score_head = nn.Linear(hd, nc)
        self.dec_score_head = nn.ModuleList(
            [score_head if share_score_head else copy.deepcopy(score_head) for _ in range(self.eval_idx + 1)]
            + [copy.deepcopy(score_head) for _ in range(ndl - self.eval_idx - 1)]
        )
        bbox_head = MLP(hd, hd, 4 * (reg_max + 1), num_layers=3, act=act_mlp)
        self.dec_bbox_head = nn.ModuleList(
            [bbox_head if share_bbox_head else copy.deepcopy(bbox_head) for _ in range(self.eval_idx + 1)]
            + [
                MLP(scaled_dim, scaled_dim, 4 * (reg_max + 1), num_layers=3, act=act_mlp)
                for _ in range(ndl - self.eval_idx - 1)
            ]
        )

        self._reset_parameters()

    def forward(self, x: list[torch.Tensor], batch: dict | None = None) -> tuple | torch.Tensor:
        """Run the forward pass of the module."""
        from ultralytics.models.utils.ops import get_cdn_group

        feats, shapes = self._get_encoder_input(x)

        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        dec_bboxes, dec_scores, dec_pred_corners, dec_refs, pre_bboxes, pre_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=attn_mask,
            memory_mask=None,
            dn_meta=dn_meta,
        )
        dfine_meta = {
            "pred_corners": dec_pred_corners,
            "ref_points": dec_refs,
            "pre_bboxes": pre_bboxes,
            "pre_logits": pre_scores,
            "up": self.up,
            "reg_scale": self.reg_scale,
        }
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta, dfine_meta
        if self.training:
            return x
        y = self.postprocess(dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid())
        return y if self.export else (y, x)

    def _select_topk(self, outputs_logits: torch.Tensor, topk: int) -> torch.Tensor:
        if self.query_select_method == "default":
            return torch.topk(outputs_logits.max(-1).values, topk, dim=1).indices
        if self.query_select_method == "one2many":
            return torch.topk(outputs_logits.flatten(1), topk, dim=1).indices // self.nc
        raise ValueError(f"Unsupported query_select_method: {self.query_select_method}")

    def _project_encoder_features(self, feats: torch.Tensor) -> torch.Tensor:
        """DEIM path: skip enc_output projection and score directly from masked encoder memory."""
        return self.valid_mask.to(feats.dtype) * feats

    def _get_decoder_input(
        self,
        feats: torch.Tensor,
        shapes: list[list[int]],
        dn_embed: torch.Tensor | None = None,
        dn_bbox: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run query selection using the swappable encoder-feature projection hook."""
        bs = feats.shape[0]
        if self.dynamic or self.shapes != shapes:
            self.anchors, self.valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
            self.shapes = shapes

        features = self._project_encoder_features(feats)
        enc_outputs_scores = self.enc_score_head(features)

        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        top_k_anchors = self.anchors[:, topk_ind].view(bs, self.num_queries, -1)

        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors
        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    def _reset_parameters(self):
        """Initialize parameters; bypasses RTDETRDecoder._reset_parameters since input_proj may contain nn.Identity."""
        bias_cls = bias_init_with_prob(0.01)
        if self.num_denoising > 0:
            nn.init.normal_(self.denoising_class_embed.weight)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        constant_(self.pre_bbox_head.layers[-1].weight, 0.0)
        constant_(self.pre_bbox_head.layers[-1].bias, 0.0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            constant_(cls_.bias, bias_cls)
            if hasattr(reg_, "layers"):
                constant_(reg_.layers[-1].weight, 0.0)
                constant_(reg_.layers[-1].bias, 0.0)

        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        xavier_uniform_(self.query_pos_head.layers[-1].weight)
        for layer in self.input_proj:
            if isinstance(layer, nn.Sequential) and len(layer) and hasattr(layer[0], "weight"):
                xavier_uniform_(layer[0].weight)
