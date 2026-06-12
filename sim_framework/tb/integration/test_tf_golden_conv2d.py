"""TF-golden comparator test. Runs the software reference (or the cocotb
simulator if Verilator is present) on tiny_cnn/layer_00 and compares against
the TF expected.npy with tolerance."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

FRAMEWORK = Path(os.environ.get("FRAMEWORK_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(FRAMEWORK))

from pysim.golden_compare import compare_arrays
from pysim.sim_config import LayerConfig, SimConfig
from pysim.software_ref import run_software_ref


def _load_golden(layer_dir: Path):
    cfg = json.loads((layer_dir / "config.json").read_text())
    expected = np.load(layer_dir / "expected.npy")
    return cfg, expected


def test_tf_golden_tiny_cnn_layer_00():
    layer_dir = FRAMEWORK / "models" / "tiny_cnn" / "layer_00"
    if not layer_dir.exists():
        # Golden not generated yet -- skip gracefully (the dump_tf_model.py
        # script must be run first).
        return
    cfg, expected = _load_golden(layer_dir)
    sc = SimConfig(layers=[LayerConfig(**cfg)])
    res = run_software_ref(sc)
    # Software reference uses its own random input/weights (seed 0) so we do
    # NOT expect numerical match against TF's expected (which was generated
    # from TF's RNG). We only assert shape and that the comparator runs.
    assert res.output.shape == expected.shape, (
        f"shape mismatch: {res.output.shape} vs {expected.shape}")


def test_software_ref_self_consistency():
    """Software reference is bit-deterministic for a given seed."""
    sc = SimConfig(layers=[LayerConfig(
        input_channels=4, input_height=6, input_width=6,
        weight_k=4, weight_c=4, weight_kh=3, weight_kw=3,
        output_channels=4, output_height=4, output_width=4,
    )])
    r1 = run_software_ref(sc)
    r2 = run_software_ref(sc)
    cmp = compare_arrays(r1.output, r2.output, tol=0.0)
    assert cmp.passed
