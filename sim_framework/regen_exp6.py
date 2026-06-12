#!/usr/bin/env python3
"""Run only exp6_loop_optimization for the Cloud workloads to regenerate figures."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts.run_full_eval import exp6_loop_optimization
from scripts.workloads import CLOUD_WORKLOADS

# Simulate the optimal config dict that the main driver would provide
# Use values consistent with the existing results
opt = {
    "array_size": 16,
    "spad_depth": 2048,
    "dataflow": "OS",
}

results_dir = ROOT / "results" / "cloud"
print(f"Regenerating exp6 figures under: {results_dir}")
exp6_loop_optimization(CLOUD_WORKLOADS, opt, results_dir)
print("\nDone! Check the figures in:")
print(f"  {results_dir / 'exp6_loop_optimization' / 'figures'}")
