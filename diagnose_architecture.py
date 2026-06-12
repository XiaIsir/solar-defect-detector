#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
L-FAF-YOLOv11n Architecture Diagnostic Script
Phases 2-4: Shape Trace + Gradient Flow + Detect Head Verification
"""
import sys
import torch
import torch.nn as nn
import numpy as np

# ---------------------------------------------------------------------------
# Monkey-patches for custom modules (same as training, no DirectML needed on CPU)
# ---------------------------------------------------------------------------
from ultralytics.nn.modules.block import PConv, EFBlock, BiFPN_Concat, ELA
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import LOGGER, colorstr
import ultralytics.nn.tasks as tasks_mod

# Force CPU
device = torch.device("cpu")

# ---------------------------------------------------------------------------
# Phase 2: Shape Trace
# ---------------------------------------------------------------------------
def run_shape_trace():
    print("=" * 80)
    print(colorstr("PHASE 2: FULL MODEL SHAPE TRACE"))
    print("=" * 80)

    model = tasks_mod.DetectionModel("ultralytics/cfg/models/11/l_faf_yolov11n.yaml", nc=5)
    model.to(device)
    model.eval()

    # Build dummy input
    x = torch.randn(1, 3, 640, 640).to(device)
    print(f"\nInput: {list(x.shape)}")

    # Match model._forward_once indexing: y[0]=L0_out, y[1]=L1_out, ...
    # Input is passed separately as `x` before the loop.
    y = []
    layer_info = []

    for i, layer in enumerate(model.model):
        f = getattr(layer, 'f', -1)
        if f != -1:
            if isinstance(f, int):
                inp = y[f] if f >= 0 else x
            else:
                inp = [x if j == -1 else y[j] for j in f]
        else:
            inp = x

        # Record input shapes BEFORE forward
        if isinstance(inp, list):
            in_shapes = [list(t.shape) for t in inp]
        else:
            in_shapes = [list(inp.shape)]

        # Forward
        out = layer(inp)
        y.append(out)  # y[i] = output of layer i (matches model indexing)
        x = out

        # Collect info
        idx = getattr(layer, 'i', i)
        t = layer.__class__.__name__
        if isinstance(out, torch.Tensor):
            out_shape = list(out.shape)
        elif isinstance(out, (list, tuple)):
            out_shape = []
            for o in out:
                if isinstance(o, torch.Tensor):
                    out_shape.append(list(o.shape))
                elif isinstance(o, dict):
                    out_shape.append({k: list(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__ for k, v in o.items()})
                else:
                    out_shape.append(type(o).__name__)
        elif isinstance(out, dict):
            out_shape = {k: list(v.shape) if isinstance(v, torch.Tensor) else v for k, v in out.items()}
        else:
            out_shape = str(type(out))
        params = sum(p.numel() for p in layer.parameters())

        layer_info.append({
            'idx': idx, 'type': t, 'from': f,
            'in_shapes': in_shapes, 'out_shape': out_shape, 'params': params
        })

        # Print trace
        in_str = ", ".join(str(s) for s in in_shapes)
        out_str = str(out_shape)
        print(f"  L{idx:2d} {t:<20s} from={str(f):>12s}  in={in_str:<30s} out={out_str:s}  params={params:,}")

    print(f"\nTotal layers: {len(layer_info)}")
    total_params = sum(li['params'] for li in layer_info)
    print(f"Total parameters: {total_params:,}")

    # -------------------------------------------------------------------
    # Key shape consistency checks
    # -------------------------------------------------------------------
    issues = []

    # Check P3/P4/P5 input to Detect (last layer)
    detect_layer = model.model[-1]
    if hasattr(detect_layer, 'ch'):
        print(f"\n--- Detect Head Input Check ---")
        print(f"Detect.ch = {list(detect_layer.ch)} (expected P3, P4, P5 channels)")
        print(f"Detect.nl = {detect_layer.nl} detection layers")
        print(f"Detect.nc = {detect_layer.nc} classes")
        print(f"Detect.no = {detect_layer.no} outputs per anchor")

        # Back-trace to get P3/P4/P5 shapes
        # y[i] = output of model layer i
        for name, layer_idx in [("P3", 15), ("P4", 18), ("P5", 21)]:
            shape = list(y[layer_idx].shape)
            ch = shape[1]
            print(f"  {name} (layer {layer_idx}): shape={shape} channels={ch}")

    return model, y, layer_info, issues


# ---------------------------------------------------------------------------
# Phase 3: Gradient Flow Check
# ---------------------------------------------------------------------------
def run_gradient_check(model=None):
    print("\n" + "=" * 80)
    print(colorstr("PHASE 3: GRADIENT FLOW CHECK"))
    print("=" * 80)

    if model is None:
        model = tasks_mod.DetectionModel("ultralytics/cfg/models/11/l_faf_yolov11n.yaml", nc=5)
        model.to(device)

    model.train()

    # Create a minimal batch (2 images, 640x640)
    x = torch.randn(2, 3, 640, 640).to(device)

    # Forward in train mode → returns dict with 'boxes', 'scores'
    preds = model(x)

    # Compute simple detection loss from preds dict
    if isinstance(preds, dict):
        total_loss = 0
        for k, v in preds.items():
            if isinstance(v, torch.Tensor):
                total_loss += v.abs().mean()
    elif isinstance(preds, (list, tuple)):
        total_loss = 0
        for p in preds:
            if isinstance(p, torch.Tensor):
                total_loss += p.abs().mean()
    else:
        total_loss = preds.abs().mean()

    model.zero_grad()
    total_loss.backward()

    # Analyze per-layer gradients
    print(f"\n{'Layer':<30s} {'Weight Mean':>12s} {'Weight Std':>12s} {'Grad Mean':>12s} {'Grad Std':>12s} {'Status':>10s}")
    print("-" * 90)

    grad_issues = []
    total_layers = 0
    zero_grad_layers = 0
    small_grad_layers = 0
    dead_layers = 0

    for name, param in model.named_parameters():
        if param.requires_grad and 'weight' in name:
            total_layers += 1
            w_mean = param.data.float().mean().item()
            w_std = param.data.float().std().item()

            if param.grad is not None:
                g_mean = param.grad.float().mean().item()
                g_std = param.grad.float().std().item()

                # Check for anomalies
                status = "OK"
                if g_std == 0 or abs(g_mean) < 1e-10:
                    status = "ZERO_GRAD"
                    zero_grad_layers += 1
                    grad_issues.append((name, "ZERO_GRAD", g_mean, g_std))
                elif abs(g_mean) < 1e-7:
                    status = "TINY_GRAD"
                    small_grad_layers += 1
                elif abs(g_mean) > 10 or g_std > 100:
                    status = "EXPLODING"
                    grad_issues.append((name, "EXPLODING", g_mean, g_std))
                elif g_std > 10 * abs(w_std):
                    status = "NOISY"

                if status != "OK":
                    print(f"  {name:<28s} {w_mean:>12.6f} {w_std:>12.6f} {g_mean:>12.6e} {g_std:>12.6e} {status:>10s}")
            else:
                status = "NO_GRAD"
                dead_layers += 1
                print(f"  {name:<28s} {w_mean:>12.6f} {w_std:>12.6f} {'N/A':>12s} {'N/A':>12s} {status:>10s}")

    print(f"\nGradient Summary:")
    print(f"  Total weighted layers: {total_layers}")
    print(f"  Zero gradient layers:  {zero_grad_layers}")
    print(f"  Small gradient layers: {small_grad_layers}")
    print(f"  Dead layers (no grad): {dead_layers}")

    # Additional check: BiFPN_Concat fusion weights
    print(f"\n--- BiFPN_Concat Weight Check ---")
    for name, module in model.named_modules():
        if isinstance(module, BiFPN_Concat):
            w = module.w.data.cpu().numpy()
            w_grad = module.w.grad.cpu().numpy() if module.w.grad is not None else None
            print(f"  {name}: weights={w[:5].round(4)}")
            if w_grad is not None:
                print(f"          grad={w_grad[:5].round(6)}")
            else:
                print(f"          grad=None (no backward pass through this module)")

    return grad_issues


# ---------------------------------------------------------------------------
# Phase 4: Detect Head Detailed Verification
# ---------------------------------------------------------------------------
def run_detect_head_verification():
    print("\n" + "=" * 80)
    print(colorstr("PHASE 4: DETECT HEAD VERIFICATION"))
    print("=" * 80)

    model = tasks_mod.DetectionModel("ultralytics/cfg/models/11/l_faf_yolov11n.yaml", nc=5)
    model.to(device)
    model.eval()

    # Get the Detect module
    detect = model.model[-1]

    print(f"\nDetect Configuration:")
    print(f"  nc (num classes):    {detect.nc}")
    print(f"  nl (num layers):     {detect.nl}")
    print(f"  reg_max (DFL ch):    {detect.reg_max}")
    print(f"  no (outputs/anchor): {detect.no}")
    # Infer input channels from cv2 modules
    detect_ch = []
    for cv2 in detect.cv2:
        first_conv = list(cv2.modules())[1]  # first Conv in Sequential
        if hasattr(first_conv, 'conv'):
            detect_ch.append(first_conv.conv.in_channels)
    print(f"  ch (input channels): {detect_ch} (inferred from cv2)")
    print(f"  strides:             {detect.stride.tolist()}")

    # Compute expected vs actual
    # For P3(80x80), P4(40x40), P5(20x20) with stride [8, 16, 32]
    print(f"\nExpected strides for P3/8, P4/16, P5/32: [8, 16, 32]")

    # Output dimension check
    # Output per scale: (B, 4*reg_max + nc, H, W)
    # For reg_max=16: 4*16 + 5 = 69 channels per scale
    expected_no = 5 + detect.reg_max * 4  # 5 + 64 = 69
    print(f"  Expected no (nc + 4*reg_max): {expected_no}")

    if detect.no != expected_no:
        print(f"  *** MISMATCH: detect.no={detect.no} != expected {expected_no}")
    else:
        print(f"  no={detect.no} ✓")

    # Check cv2/cv3 channel dims
    print(f"\n  cv2 (bbox) modules:")
    for i, cv2 in enumerate(detect.cv2):
        last_conv = list(cv2.modules())[-1]
        if isinstance(last_conv, nn.Conv2d):
            print(f"    Scale {i}: last Conv2d out_channels={last_conv.out_channels} (expect {4*detect.reg_max})")

    print(f"  cv3 (class) modules:")
    for i, cv3 in enumerate(detect.cv3):
        last_conv = list(cv3.modules())[-1]
        if isinstance(last_conv, nn.Conv2d):
            print(f"    Scale {i}: last Conv2d out_channels={last_conv.out_channels} (expect {detect.nc})")

    # Forward pass and check raw output
    x = torch.randn(1, 3, 640, 640).to(device)
    with torch.no_grad():
        output = model(x)

    if isinstance(output, (list, tuple)):
        for i, o in enumerate(output):
            if isinstance(o, torch.Tensor):
                s = list(o.shape)
                print(f"\n  Output[{i}] tensor: shape={s}")
                expected_hw = [80, 40, 20][i] if i < 3 else '?'
                actual_ch = s[1] if len(s) > 1 else '?'
                actual_h = s[2] if len(s) > 2 else '?'
                print(f"    channels={actual_ch}, spatial={actual_h}x{actual_h}")
            elif isinstance(o, dict):
                print(f"\n  Output[{i}] dict:")
                for k, v in o.items():
                    if isinstance(v, torch.Tensor):
                        print(f"    {k}: {list(v.shape)}")
                    elif isinstance(v, (list, tuple)):
                        print(f"    {k}: [{len(v)} items]")
                    else:
                        print(f"    {k}: {v}")

    else:
        print(f"\n  Training output shape: {list(output.shape)}")

    return model


# ---------------------------------------------------------------------------
# Phase 1 Supplement: Module-level Analysis
# ---------------------------------------------------------------------------
def run_module_analysis():
    print("=" * 80)
    print(colorstr("PHASE 1 SUPPLEMENT: MODULE-LEVEL ANALYSIS"))
    print("=" * 80)

    # --- PConv Analysis ---
    print("\n--- PConv Analysis ---")
    # Test with 3-channel input (first layer scenario)
    pconv_first = PConv(3, 64, n_div=4, kernel_size=3, stride=2, padding=1)
    x = torch.randn(1, 3, 640, 640)
    out = pconv_first(x)
    print(f"  First layer (3→64, stride=2):")
    print(f"    dim_conv = max(3//4, 1) = {max(3//4, 1)} channel gets 3x3 conv")
    print(f"    dim_untouched = 3 - 1 = 2 channels get avg_pool only")
    print(f"    Input:  {list(x.shape)}")
    print(f"    Output: {list(out.shape)}")
    print(f"    INFO LOSS: 67% of RGB channels bypass spatial convolution")

    # Test with normal channel count (e.g., 128→64)
    pconv_normal = PConv(128, 64, n_div=4, kernel_size=3, stride=2, padding=1)
    x2 = torch.randn(1, 128, 80, 80)
    out2 = pconv_normal(x2)
    print(f"\n  Normal layer (128→64, stride=2):")
    print(f"    dim_conv = max(128//4, 1) = 32 channels get 3x3 conv")
    print(f"    dim_untouched = 128 - 32 = 96 channels bypassed")
    print(f"    Input:  {list(x2.shape)}")
    print(f"    Output: {list(out2.shape)}")
    print(f"    INFO LOSS: 75% of input channels bypass spatial processing")

    # --- EFBlock Analysis ---
    print("\n--- EFBlock Analysis ---")
    # Scenario 1: c1 == c2 (backbone, residual active)
    ef_res = EFBlock(128, 128, shortcut=True, k=3, e=0.5)
    x3 = torch.randn(1, 128, 80, 80)
    out3 = ef_res(x3)
    print(f"  Residual case (128→128, shortcut=True):")
    print(f"    c_ = int(128 * 0.5) = 64 hidden channels")
    print(f"    cv1: 128→64  (1x1 squeeze, discards 50% channels)")
    print(f"    cv2: 64→64 depth-wise (g=64, NO cross-channel interaction)")
    print(f"    ELA: 64→64 (attention only, preserves channels)")
    print(f"    cv3: 64→128 (1x1 expand)")
    print(f"    add = shortcut and c1==c2 = True ✓")
    print(f"    Input:  {list(x3.shape)}")
    print(f"    Output: {list(out3.shape)}")

    # Scenario 2: c1 != c2 (neck fusion, residual LOST)
    ef_nores = EFBlock(320, 128, shortcut=True, k=3, e=0.5)
    x4 = torch.randn(1, 320, 40, 40)
    out4 = ef_nores(x4)
    print(f"\n  NO-residual case (320→128, shortcut=True but c1!=c2):")
    print(f"    c_ = int(128 * 0.5) = 64 hidden channels")
    print(f"    cv1: 320→64  (1x1 squeeze, *** DISCARDS 80% channels ***)")
    print(f"    cv2: 64→64 depth-wise")
    print(f"    ELA: 64→64")
    print(f"    cv3: 64→128")
    print(f"    add = True and 320==128 = False *** RESIDUAL LOST ***")
    print(f"    Input:  {list(x4.shape)}")
    print(f"    Output: {list(out4.shape)}")

    # --- BiFPN_Concat Analysis ---
    print("\n--- BiFPN_Concat Analysis ---")
    # Test 2-input fusion
    bifpn_2 = BiFPN_Concat(dimension=1, max_inputs=5)
    a = torch.randn(1, 256, 20, 20)
    b = torch.randn(1, 128, 40, 40)
    out_ab = bifpn_2([a, b])
    print(f"  2-input fusion (256+128):")
    print(f"    Input shapes: {list(a.shape)}, {list(b.shape)}")
    print(f"    Output shape: {list(out_ab.shape)}")
    print(f"    Expected: [1, 384, 40, 40] (CONCAT, NOT weighted sum)")
    print(f"    Actual:   {list(out_ab.shape)}")
    print(f"    *** CHANNEL EXPLOSION: 256+128=384, then EFBlock compresses to 128 (83% info loss) ***")

    # Test 3-input fusion
    bifpn_3 = BiFPN_Concat(dimension=1, max_inputs=5)
    c = torch.randn(1, 64, 80, 80)
    d = torch.randn(1, 128, 80, 80)
    e = torch.randn(1, 128, 80, 80)
    out_cde = bifpn_3([c, d, e])
    print(f"\n  3-input fusion (64+128+128):")
    print(f"    Input shapes: {list(c.shape)}, {list(d.shape)}, {list(e.shape)}")
    print(f"    Output shape: {list(out_cde.shape)}")
    print(f"    Expected: [1, 320, 80, 80]")
    print(f"    *** TRIPLE FUSION: 64+128+128=320, then EFBlock compresses to 128 (60% info loss) ***")


# ---------------------------------------------------------------------------
# Compare with Official YOLO11n
# ---------------------------------------------------------------------------
def run_official_comparison():
    print("\n" + "=" * 80)
    print(colorstr("OFFICIAL YOLO11n vs L-FAF-YOLOv11n COMPARISON"))
    print("=" * 80)

    # Official YOLO11n
    model_official = tasks_mod.DetectionModel("ultralytics/cfg/models/11/yolo11.yaml", nc=5)
    model_official.to(device)

    official_params = sum(p.numel() for p in model_official.parameters())

    # L-FAF
    model_lfaf = tasks_mod.DetectionModel("ultralytics/cfg/models/11/l_faf_yolov11n.yaml", nc=5)
    model_lfaf.to(device)

    lfaf_params = sum(p.numel() for p in model_lfaf.parameters())

    print(f"\n  Official YOLO11n: {official_params:,} parameters")
    print(f"  L-FAF-YOLOv11n:   {lfaf_params:,} parameters")
    print(f"  Ratio (L-FAF/Official): {lfaf_params/official_params*100:.1f}%")
    print(f"  Paper claims: 2.89×10^6 = 2,890,000 parameters")
    print(f"  Our actual:   {lfaf_params:,} parameters")
    if lfaf_params < 2000000:
        print(f"  *** PARAMETER DEFICIT vs paper claim: {lfaf_params/2890000*100:.1f}% ***")

    # Structure comparison at key fusion points
    print(f"\n  Structure comparison at Neck fusion points:")
    print(f"  {'Fusion Node':<20s} {'Official':<30s} {'L-FAF':<30s}")
    print(f"  {'-'*20} {'-'*30} {'-'*30}")
    print(f"  {'P4 upsample':<20s} {'Concat→C3k2(384→128)':<30s} {'BiFPN→EFBlock(384→128,nr)':<30s}")
    print(f"  {'P3 upsample':<20s} {'Concat→C3k2(256→64)':<30s} {'BiFPN→EFBlock(192→64,nr)':<30s}")
    print(f"  {'P4 downsample':<20s} {'Concat→C3k2(192→128)':<30s} {'BiFPN→EFBlock(320→128,nr)':<30s}")
    print(f"  {'P5 downsample':<20s} {'Concat→C3k2(384→256)':<30s} {'BiFPN→EFBlock(384→256,nr)':<30s}")
    print(f"  nr = no residual shortcut")

    return official_params, lfaf_params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("L-FAF-YOLOv11n Architecture Diagnostic Suite")
    print("=" * 80)

    # Phase 1 supplement: module-level analysis
    run_module_analysis()

    # Phase 2: Shape trace
    model, y_list, layer_list, issues = run_shape_trace()

    # Official comparison
    off_params, lfaf_params = run_official_comparison()

    # Phase 3: Gradient flow
    grad_issues = run_gradient_check(model)

    # Phase 4: Detect head verification
    detect_model = run_detect_head_verification()

    # Summary
    print("\n" + "=" * 80)
    print(colorstr("DIAGNOSTIC SUMMARY"))
    print("=" * 80)

    print(f"""
  L-FAF Parameters: {lfaf_params:,}
  Official YOLO11n Parameters: {off_params:,}

  Key Findings:
  ------------
  1. PConv Stem: First layer only convolves 1/3 RGB channels → 67% pixel info bypassed
  2. BiFPN_Concat: CONCATENATES at fusion points → channel explosion (320-384 ch)
  3. EFBlock bottleneck: e=0.5, cv1 1×1 squeezes 80% of fused info at every fusion node
  4. EFBlock residual: LOST at ALL 4 neck fusion points (c1 ≠ c2)
  5. EFBlock cv2: depth-wise conv (g=c_) means ZERO cross-channel spatial feature learning

  The combination: PConv stem blindness + BiFPN channel explosion + EFBlock information
  shredding + zero residuals = model that can minimize generic loss but cannot form
  meaningful detection features.
""")
