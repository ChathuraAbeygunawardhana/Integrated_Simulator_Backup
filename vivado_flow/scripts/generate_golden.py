#!/usr/bin/env python3
"""
generate_golden.py
==================
Generates the three hex memory files consumed by the testbench's $readmemh:

    tb_data/input_hex.mem   – randomised (or real) input activations
    tb_data/weight_hex.mem  – randomised (or real) weight tensor
    tb_data/golden_hex.mem  – software-computed expected output

The computation is a valid (no-padding) 2-D convolution:
    output[k, h_out, w_out] = sum_c sum_kh sum_kw
                               input[c, h_out+kh, w_out+kw] * weight[k, c, kh, kw]

All values are 32-bit integers (signed) to match DATA_WIDTH=32.
The golden output is also 32-bit (saturated if needed).

Usage
-----
    # From config JSON (written by generate_config.py)
    python generate_golden.py

    # Explicit dims
    python generate_golden.py --K 16 --C 8 --KH 3 --KW 3 --H 30 --W 30

    # Use real weights from a Keras layer
    python generate_golden.py --tf-model tiny_cnn --layer-idx 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# ── Cross-platform UTF-8 stdout/stderr ──────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).resolve().parent      # vivado_flow/scripts/
_FLOW_DIR = _HERE.parent                         # vivado_flow/
_TB_DATA  = _FLOW_DIR / "tb_data"
_CFG_JSON = _FLOW_DIR / "configs" / "generated" / "optimal_config.json"

_TB_DATA.mkdir(parents=True, exist_ok=True)


# ── Convolution reference ──────────────────────────────────────────────────

def conv2d_reference(
    inputs:  np.ndarray,   # (C, H_in, W_in)  int32
    weights: np.ndarray,   # (K, C, KH, KW)   int32
) -> np.ndarray:           # (K, H_out, W_out) int64 → clipped to int32
    C, HIN, WIN = inputs.shape
    K, C2, KH, KW = weights.shape
    assert C == C2

    HOUT = HIN - KH + 1
    WOUT = WIN - KW + 1

    output = np.zeros((K, HOUT, WOUT), dtype=np.int64)

    for k in range(K):
        for c in range(C):
            for kh in range(KH):
                for kw in range(KW):
                    # slice the input patch
                    patch = inputs[c, kh:kh+HOUT, kw:kw+WOUT]  # (HOUT, WOUT)
                    output[k] += patch.astype(np.int64) * int(weights[k, c, kh, kw])

    # Saturate to int32
    I32_MAX =  2**31 - 1
    I32_MIN = -2**31
    output = np.clip(output, I32_MIN, I32_MAX).astype(np.int32)
    return output


# ── Hex file writers ───────────────────────────────────────────────────────

def write_hex_mem(array: np.ndarray, path: Path):
    """Write a flat array of int32 values as 8-hex-digit lines."""
    flat = array.flatten().astype(np.int32)
    with path.open("w", encoding="ascii") as f:
        for v in flat:
            # Convert numpy int32 → Python int before bitwise AND to avoid overflow
            pv = int(v)
            f.write(f"{pv & 0xFFFF_FFFF:08X}\n")
    print(f"  Wrote {len(flat):>8,} elements -> {path}")


# ── Main ───────────────────────────────────────────────────────────────────

def generate(
    dims:         Dict,
    seed:         int  = 42,
    val_range:    int  = 16,       # random values in [-val_range, val_range)
    tf_model:     Optional[str] = None,
    layer_idx:    int  = 0,
    out_dir:      Path = _TB_DATA,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate input, weight, and golden arrays; write to out_dir.

    Returns (inputs, weights, golden).
    """
    K, C, KH, KW = dims['K'], dims['C'], dims['KH'], dims['KW']
    HIN, WIN      = dims['H'], dims['W']

    rng = np.random.default_rng(seed)

    if tf_model is not None:
        # Use real weights from Keras
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        try:
            import tensorflow as tf
        except ImportError:
            print("TensorFlow not available; using random weights.")
            tf_model = None

    if tf_model is None:
        # Random integer tensors (small range keeps intermediate products safe)
        inputs  = rng.integers(-val_range, val_range, size=(C, HIN, WIN),
                               dtype=np.int32)
        weights = rng.integers(-val_range, val_range, size=(K, C, KH, KW),
                               dtype=np.int32)
    else:
        # Keras model weights (quantised to int8 range for integer MAC)
        _APPS = {
            "mobilenet_v2":   lambda: tf.keras.applications.MobileNetV2(weights=None),
            "resnet50":       lambda: tf.keras.applications.ResNet50(weights=None),
            "vgg16":          lambda: tf.keras.applications.VGG16(weights=None),
            "tiny_cnn":       _build_tiny_cnn,
        }
        mn = tf_model.lower()
        if mn not in _APPS:
            raise ValueError(f"Unknown model '{tf_model}'")
        model = _APPS[mn]()
        conv_layers = [l for l in model.layers
                       if isinstance(l, tf.keras.layers.Conv2D)]
        if not conv_layers:
            raise ValueError("No Conv2D found")
        layer = conv_layers[min(layer_idx, len(conv_layers)-1)]
        w = layer.get_weights()
        if w:
            kern = w[0]           # (KH, KW, C_in, C_out) float32
            # Quantise to int8 so products fit in int32
            scale = 127.0 / (np.max(np.abs(kern)) + 1e-9)
            kern_int = np.round(kern * scale).astype(np.int32)
            # Transpose to (K, C, KH, KW)
            weights = kern_int.transpose(3, 2, 0, 1)
        else:
            weights = rng.integers(-8, 8, size=(K, C, KH, KW), dtype=np.int32)

        # Random input
        inputs = rng.integers(-16, 16, size=(C, HIN, WIN), dtype=np.int32)

    print(f"\nDimensions: K={K} C={C} KH={KH} KW={KW} H={HIN} W={WIN}")
    print(f"  input  shape : {inputs.shape}   min={inputs.min()} max={inputs.max()}")
    print(f"  weight shape : {weights.shape}  min={weights.min()} max={weights.max()}")

    print("  Computing golden reference …")
    golden = conv2d_reference(inputs, weights)
    print(f"  golden shape : {golden.shape}   min={golden.min()} max={golden.max()}")

    print("\nWriting hex memory files:")
    write_hex_mem(inputs,  out_dir / "input_hex.mem")
    write_hex_mem(weights, out_dir / "weight_hex.mem")
    write_hex_mem(golden,  out_dir / "golden_hex.mem")

    # Also write a dims JSON so other tools can pick up the config
    meta = {
        "K": K, "C": C, "KH": KH, "KW": KW, "H": HIN, "W": WIN,
        "H_OUT": HIN - KH + 1, "W_OUT": WIN - KW + 1,
        "seed": seed, "val_range": val_range,
    }
    (out_dir / "dims.json").write_text(json.dumps(meta, indent=2))
    print(f"  Wrote dims.json → {out_dir / 'dims.json'}")

    return inputs, weights, golden


def _build_tiny_cnn():
    import tensorflow as tf
    inp = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(16, 3, padding="same", name="conv1")(inp)
    x = tf.keras.layers.Conv2D(32, 3, name="conv2")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(10)(x)
    return tf.keras.Model(inp, x, name="tiny_cnn")


# ── CLI ────────────────────────────────────────────────────────────────────

def _build_parser():
    p = argparse.ArgumentParser(
        description="Generate tb_data/ hex memory files for the testbench."
    )
    p.add_argument("--K",  type=int, default=None)
    p.add_argument("--C",  type=int, default=None)
    p.add_argument("--KH", type=int, default=None)
    p.add_argument("--KW", type=int, default=None)
    p.add_argument("--H",  type=int, default=None)
    p.add_argument("--W",  type=int, default=None)
    p.add_argument("--from-json", type=Path, default=_CFG_JSON,
                   help=f"Load dims from JSON (default: {_CFG_JSON})")
    p.add_argument("--tf-model",  default=None)
    p.add_argument("--layer-idx", type=int, default=0)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--val-range", type=int, default=16,
                   help="Random values in [-val_range, val_range)")
    p.add_argument("--out-dir",   type=Path, default=_TB_DATA)
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    # Resolve dims: CLI > JSON > defaults
    dims = {"K":16,"C":8,"KH":3,"KW":3,"H":30,"W":30}

    if args.from_json and args.from_json.exists():
        try:
            raw = json.loads(args.from_json.read_text())
            # Accept both our own JSON and generate_config.py JSON
            if "layer_dims" in raw:
                ld = raw["layer_dims"]
                dims = {"K":ld["K"],"C":ld["C"],"KH":ld["KH"],
                        "KW":ld["KW"],"H":ld["H"],"W":ld["W"]}
            elif "K" in raw:
                dims = {k: raw[k] for k in ("K","C","KH","KW","H","W")}
            print(f"[generate_golden] Loaded dims from {args.from_json}")
        except Exception as e:
            print(f"[generate_golden] WARNING: could not read JSON: {e}")

    # CLI overrides
    for key in ("K","C","KH","KW"):
        v = getattr(args, key, None)
        if v is not None: dims[key] = v
    if args.H: dims["H"] = args.H
    if args.W: dims["W"] = args.W

    args.out_dir.mkdir(parents=True, exist_ok=True)
    generate(
        dims=dims,
        seed=args.seed,
        val_range=args.val_range,
        tf_model=args.tf_model,
        layer_idx=args.layer_idx,
        out_dir=args.out_dir,
    )
    print("\n✓ tb_data/ ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
