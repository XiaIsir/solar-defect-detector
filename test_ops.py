"""
Temporary test script for the 6 custom YOLOv11 operators.

Validates forward-pass shapes for: ELA, PConv, FEM, BiFPN_Concat, C3k2_AGA, EFBlock.
Run: python test_ops.py
"""
import torch
import sys

# Ensure we can import from ultralytics
sys.path.insert(0, ".")

from ultralytics.nn.modules import ELA, PConv, FEM, BiFPN_Concat, C3k2_AGA, EFBlock


def test_ela():
    print("=" * 60)
    print("Testing ELA ...")
    m = ELA(channels=64, kernel_size=7)
    x = torch.randn(1, 64, 40, 40)
    y = m(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    print(f"  Input:  {x.shape}  →  Output: {y.shape}  ✓")


def test_pconv():
    print("=" * 60)
    print("Testing PConv ...")
    # Case 1: dim == o_dim (Identity proj)
    m = PConv(dim=64, o_dim=64, n_div=4)
    x = torch.randn(1, 64, 40, 40)
    y = m(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    print(f"  Case 1 (equal io):  {x.shape}  →  {y.shape}  ✓")

    # Case 2: dim != o_dim (1x1 proj)
    m2 = PConv(dim=64, o_dim=128, n_div=4)
    x2 = torch.randn(1, 64, 40, 40)
    y2 = m2(x2)
    expected = (1, 128, 40, 40)
    assert y2.shape == expected, f"Shape mismatch: {y2.shape} vs {expected}"
    print(f"  Case 2 (proj):      {x2.shape}  →  {y2.shape}  ✓")


def test_fem():
    print("=" * 60)
    print("Testing FEM ...")
    m = FEM(c1=64, c2=128)
    x = torch.randn(1, 64, 40, 40)
    y = m(x)
    expected = (1, 128, 40, 40)
    assert y.shape == expected, f"Shape mismatch: {y.shape} vs {expected}"
    print(f"  Input:  {x.shape}  →  Output: {y.shape}  ✓")


def test_bifpn_concat():
    print("=" * 60)
    print("Testing BiFPN_Concat ...")
    # Case 1: two inputs, same spatial size
    m = BiFPN_Concat(dimension=1)
    x1 = torch.randn(1, 64, 40, 40)
    x2 = torch.randn(1, 128, 40, 40)
    y = m([x1, x2])
    expected = (1, 192, 40, 40)  # 64 + 128
    assert y.shape == expected, f"Shape mismatch: {y.shape} vs {expected}"
    print(f"  Case 1 (2 inputs, same size):  {x1.shape} + {x2.shape}  →  {y.shape}  ✓")

    # Case 2: three inputs, different spatial sizes (simulating FPN)
    m2 = BiFPN_Concat(dimension=1)
    xa = torch.randn(1, 128, 20, 20)
    xb = torch.randn(1, 128, 40, 40)  # larger — will be downscaled
    xc = torch.randn(1, 256, 40, 40)
    y2 = m2([xa, xb, xc])
    expected2 = (1, 512, 20, 20)  # 128+128+256
    assert y2.shape == expected2, f"Shape mismatch: {y2.shape} vs {expected2}"
    print(f"  Case 2 (3 inputs, mixed sizes):  →  {y2.shape}  ✓")


def test_c3k2_aga():
    print("=" * 60)
    print("Testing C3k2_AGA ...")
    # C3k2_AGA uses AgentBottleneck internally
    m = C3k2_AGA(c1=64, c2=128, n=2, shortcut=False, num_agent_tokens=16)
    x = torch.randn(1, 64, 40, 40)
    y = m(x)
    expected = (1, 128, 40, 40)
    assert y.shape == expected, f"Shape mismatch: {y.shape} vs {expected}"
    print(f"  Input:  {x.shape}  →  Output: {y.shape}  ✓")

    # Also test when c1 == c2 (shortcut path)
    m2 = C3k2_AGA(c1=128, c2=128, n=1, shortcut=True, num_agent_tokens=16)
    x2 = torch.randn(1, 128, 20, 20)
    y2 = m2(x2)
    expected2 = (1, 128, 20, 20)
    assert y2.shape == expected2, f"Shape mismatch: {y2.shape} vs {expected2}"
    print(f"  Case 2 (shortcut=True):  {x2.shape}  →  {y2.shape}  ✓")


def test_efblock():
    print("=" * 60)
    print("Testing EFBlock ...")
    # Case 1: shortcut=False
    m = EFBlock(c1=64, c2=128, shortcut=False)
    x = torch.randn(1, 64, 40, 40)
    y = m(x)
    expected = (1, 128, 40, 40)
    assert y.shape == expected, f"Shape mismatch: {y.shape} vs {expected}"
    print(f"  Case 1 (no shortcut):  {x.shape}  →  {y.shape}  ✓")

    # Case 2: shortcut=True, c1 == c2 (residual add)
    m2 = EFBlock(c1=128, c2=128, shortcut=True)
    x2 = torch.randn(1, 128, 40, 40)
    y2 = m2(x2)
    expected2 = (1, 128, 40, 40)
    assert y2.shape == expected2, f"Shape mismatch: {y2.shape} vs {expected2}"
    print(f"  Case 2 (shortcut=True):  {x2.shape}  →  {y2.shape}  ✓")


if __name__ == "__main__":
    all_passed = True
    try:
        test_ela()
    except Exception as e:
        all_passed = False
        print(f"  ❌ ELA FAILED: {e}")

    try:
        test_pconv()
    except Exception as e:
        all_passed = False
        print(f"  ❌ PConv FAILED: {e}")

    try:
        test_fem()
    except Exception as e:
        all_passed = False
        print(f"  ❌ FEM FAILED: {e}")

    try:
        test_bifpn_concat()
    except Exception as e:
        all_passed = False
        print(f"  ❌ BiFPN_Concat FAILED: {e}")

    try:
        test_c3k2_aga()
    except Exception as e:
        all_passed = False
        print(f"  ❌ C3k2_AGA FAILED: {e}")

    try:
        test_efblock()
    except Exception as e:
        all_passed = False
        print(f"  ❌ EFBlock FAILED: {e}")

    print("=" * 60)
    if all_passed:
        print("ALL 6 MODULES PASSED ✓")
    else:
        print("SOME MODULES FAILED — check errors above.")
        sys.exit(1)
