#!/usr/bin/env python3
"""CLI to dump a TF model into per-layer golden reference vectors.

Examples:
    python models/dump_tf_model.py --model tiny_cnn  --out models/tiny_cnn
    python models/dump_tf_model.py --model mobilenet_v2 --out models/mobilenet_v2 --layers 0,1,2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRAMEWORK_ROOT))

from pysim.tf_dump import build_tiny_cnn, dump_model  # noqa: E402


def _load_model(name: str):
    import tensorflow as tf  # local import
    name = name.lower()
    if name == "tiny_cnn":
        return build_tiny_cnn()
    if name == "mobilenet_v2":
        return tf.keras.applications.MobileNetV2(weights=None, input_shape=(96, 96, 3))
    if name == "mnist_cnn":
        # Tiny MNIST classifier; matches what verification flows often want
        inputs = tf.keras.Input(shape=(28, 28, 1))
        x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(10)(x)
        return tf.keras.Model(inputs, outputs, name="mnist_cnn")
    raise ValueError(f"Unknown model {name!r} (try tiny_cnn, mnist_cnn, mobilenet_v2)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", default="", help="Comma-separated layer indices (default: all)")
    p.add_argument("--total-bits", type=int, default=32)
    args = p.parse_args(argv)

    model = _load_model(args.model)
    print(model.summary())
    dumped = dump_model(model, args.out, fp_total_bits=args.total_bits)
    print(f"\nDumped {len(dumped)} layers to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
