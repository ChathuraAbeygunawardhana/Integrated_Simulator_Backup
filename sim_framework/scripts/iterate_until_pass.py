#!/usr/bin/env python3
"""Iterate-until-pass runner. Expands ci_matrix.yaml into SimConfig instances,
runs each, archives failures, retries until 100% green (or the user-defined
max iteration count is hit).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRAMEWORK_ROOT))

from pysim import runner, reporter
from pysim.sim_config import LayerConfig, SimConfig


DEFAULT_LAYER = LayerConfig(
    input_channels=4, input_height=6, input_width=6,
    weight_k=4, weight_c=4, weight_kh=3, weight_kw=3,
    output_channels=4, output_height=4, output_width=4,
)


def _materialise(entry: dict) -> SimConfig:
    layers = [LayerConfig(**l) for l in entry.pop("layers", [])] or [DEFAULT_LAYER]
    return SimConfig(layers=layers, **entry)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default=str(FRAMEWORK_ROOT / "configs" / "ci_matrix.yaml"))
    p.add_argument("--max-iter", type=int, default=3,
                   help="Maximum re-run iterations on failures")
    p.add_argument("--csv", default=str(FRAMEWORK_ROOT / "build" / "matrix_results.csv"))
    p.add_argument("--md",  default=str(FRAMEWORK_ROOT / "docs"  / "verification_matrix.md"))
    args = p.parse_args(argv)

    with open(args.matrix) as fh:
        matrix = yaml.safe_load(fh)["matrix"]

    cfgs = [_materialise(dict(e)) for e in matrix]
    print(f"[iterate] {len(cfgs)} configs in matrix")

    results: list[tuple[SimConfig, runner.RunResult]] = []
    pending = list(cfgs)

    for it in range(1, args.max_iter + 1):
        print(f"\n[iterate] === iteration {it}/{args.max_iter} ===")
        failing = []
        for cfg in pending:
            res = runner.run(cfg)
            tag = f"{cfg.top}/{cfg.dataflow}/{cfg.memory}/{cfg.scheduler or '-'}"
            print(f"  {tag:<60s}  {'PASS' if res.passed else 'FAIL'}  cycles={res.cycles}")
            results.append((cfg, res))
            if not res.passed:
                failing.append(cfg)
        if not failing:
            print(f"[iterate] all configs pass on iteration {it}")
            break
        pending = failing
    else:
        print(f"[iterate] {len(pending)} configs still failing after {args.max_iter} iterations")

    # De-duplicate (keep last result per config sha)
    by_sha: dict[str, tuple[SimConfig, runner.RunResult]] = {}
    for cfg, res in results:
        by_sha[res.sha] = (cfg, res)
    final = list(by_sha.values())

    reporter.write_csv(final, Path(args.csv))
    reporter.write_markdown_matrix(final, Path(args.md))
    n_pass = sum(1 for _, r in final if r.passed)
    print(f"[iterate] final: {n_pass}/{len(final)} pass")
    print(f"[iterate] csv -> {args.csv}")
    print(f"[iterate] md  -> {args.md}")
    return 0 if n_pass == len(final) else 1


if __name__ == "__main__":
    sys.exit(main())
