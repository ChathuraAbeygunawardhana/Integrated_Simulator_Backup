"""Dump a pretrained TF/Keras model into per-layer .npy + .hex artifacts
that the simulator test bench can consume as the golden reference.

Each Conv2D / Dense layer produces a subdirectory:
    layer_NN/config.json
    layer_NN/input.npy       (float32, NHWC)
    layer_NN/weights.npy
    layer_NN/expected.npy
    layer_NN/input.hex       (Qm.n)
    layer_NN/weights.hex
    layer_NN/expected.hex
    layer_NN/quant.json      (frac_bits per tensor used by the comparator)
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from .quantize import calibrate_per_tensor, float_to_fixed
from .sim_config import LayerConfig


def _write_hex(arr: np.ndarray, path: Path) -> None:
    """Write the array as one signed-hex word per line, two's-complement."""
    flat = arr.flatten()
    # Mask to the dtype's width (auto for int8/16/32)
    width = arr.dtype.itemsize * 8
    mask = (1 << width) - 1
    with open(path, "w") as fh:
        for v in flat:
            fh.write(f"{int(v) & mask:0{width // 4}x}\n")


def dump_layer(layer, layer_idx: int, out_dir: Path,
               fp_total_bits: int = 32, seed: int = 0) -> Optional[dict]:
    """Dump a single Conv2D or Dense layer; returns its LayerConfig dict."""
    import tensorflow as tf  # local import; only required for the dump flow

    layer_dir = out_dir / f"layer_{layer_idx:02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Keras 3 (TF 2.16+) renamed layer.input_shape -> layer.input.shape; fall back gracefully.
    def _in_shape(l):
        try:
            return tuple(l.input.shape)
        except Exception:
            return tuple(l.input_shape)

    if isinstance(layer, tf.keras.layers.Conv2D):
        kH, kW = layer.kernel_size
        ish    = _in_shape(layer)
        in_c   = int(ish[-1])
        out_c  = int(layer.filters)
        in_h   = int(ish[1] or 8)
        in_w   = int(ish[2] or 8)
        out_h  = in_h - kH + 1
        out_w  = in_w - kW + 1
        x = rng.standard_normal((1, in_h, in_w, in_c)).astype(np.float32)
        weights = layer.get_weights()
        w = weights[0]                       # (kH, kW, in_c, out_c)
        y = layer(x).numpy()                 # forward pass
    elif isinstance(layer, tf.keras.layers.Dense):
        ish = _in_shape(layer)
        in_c, out_c = int(ish[-1]), int(layer.units)
        in_h = in_w = 1; kH = kW = 1; out_h = out_w = 1
        x = rng.standard_normal((1, in_c)).astype(np.float32)
        weights = layer.get_weights()
        w = weights[0]                       # (in_c, out_c)
        # Conform to NHWC (1,1,1,in_c) for the testbench
        w = w.reshape(1, 1, in_c, out_c)
        x_n = x.reshape(1, 1, 1, in_c)
        y = layer(x).numpy().reshape(1, 1, 1, out_c)
        x = x_n
    else:
        return None

    # Calibrate frac_bits per tensor
    fb_x = calibrate_per_tensor(x, fp_total_bits)
    fb_w = calibrate_per_tensor(w, fp_total_bits)
    fb_y = calibrate_per_tensor(y, fp_total_bits)

    np.save(layer_dir / "input.npy",    x)
    np.save(layer_dir / "weights.npy",  w)
    np.save(layer_dir / "expected.npy", y)
    _write_hex(float_to_fixed(x, fp_total_bits, fb_x), layer_dir / "input.hex")
    _write_hex(float_to_fixed(w, fp_total_bits, fb_w), layer_dir / "weights.hex")
    _write_hex(float_to_fixed(y, fp_total_bits, fb_y), layer_dir / "expected.hex")

    cfg = LayerConfig(
        name=f"layer_{layer_idx:02d}",
        input_channels=int(in_c), input_height=int(in_h), input_width=int(in_w),
        weight_k=int(out_c), weight_c=int(in_c),
        weight_kh=int(kH), weight_kw=int(kW),
        output_channels=int(out_c),
        output_height=int(out_h), output_width=int(out_w),
        input_base_addr=0x00000000,
        weight_base_addr=0x00010000,
        output_base_addr=0x00020000,
    )
    (layer_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (layer_dir / "quant.json").write_text(json.dumps({
        "fp_total_bits": fp_total_bits,
        "input_frac_bits":   fb_x,
        "weight_frac_bits":  fb_w,
        "expected_frac_bits": fb_y,
    }, indent=2))
    return asdict(cfg)


def dump_model(model, out_dir: str | Path, fp_total_bits: int = 32) -> list[dict]:
    """Walk a Keras model and dump every Conv2D/Dense layer."""
    import tensorflow as tf  # noqa: F401 (require import to fail fast if missing)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dumped: list[dict] = []
    idx = 0
    for layer in model.layers:
        if hasattr(layer, "get_weights") and layer.get_weights():
            cfg = dump_layer(layer, idx, out_path, fp_total_bits=fp_total_bits)
            if cfg:
                dumped.append(cfg)
                idx += 1
    (out_path / "model_index.json").write_text(json.dumps({
        "n_layers": len(dumped),
        "layers": dumped,
    }, indent=2))
    return dumped


def build_tiny_cnn():
    """A 2-layer Keras CNN suitable for the smoke test. No external download."""
    import tensorflow as tf
    inputs = tf.keras.Input(shape=(8, 8, 3))
    x = tf.keras.layers.Conv2D(8, (3, 3), activation=None)(inputs)
    outputs = tf.keras.layers.Dense(4)(tf.keras.layers.Flatten()(x))
    model = tf.keras.Model(inputs, outputs, name="tiny_cnn")
    model.build((None, 8, 8, 3))
    return model
