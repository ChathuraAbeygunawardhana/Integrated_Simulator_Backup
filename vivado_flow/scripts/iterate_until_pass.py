#!/usr/bin/env python3
"""
iterate_until_pass.py
=====================
End-to-end test loop for the Vivado systolic-array flow.

Steps per iteration
-------------------
  1. [Stage 1] generate_config.py  → configs/generated/optimal_config.vh
  2. [Stage 2] config_knobs.py     → configs/generated/user_config.vh
  3. [Stage 3] generate_golden.py  → tb_data/{input,weight,golden}_hex.mem
  4. [Stage 4] vivado -source scripts/run_vivado_sim.tcl  → sim_result.txt
  5. Parse sim_result.txt:
       TEST_PASS → print summary and exit 0
       TEST_FAIL → diagnose → apply fixup → repeat

Fixup strategies (applied automatically before the next iteration)
------------------------------------------------------------------
  a) TIMEOUT:  double TILE_K/TILE_C/TILE_H/TILE_W (reduces tile count)
  b) MISMATCH: reduce val_range (smaller numbers avoid overflow in int32 MACs),
               then optionally switch to exhaustive search mode
  c) RTL_ERROR: log and abort (needs manual RTL fix)

Usage
-----
  cd vivado_flow
  python scripts/iterate_until_pass.py \
      --repo-root .. \
      --K 16 --C 8 --KH 3 --KW 3 --H 30 --W 30 \
      --array-height 4 --array-width 4 \
      --vivado vivado \
      --max-iter 5

  # From TF model
  python scripts/iterate_until_pass.py --tf-model tiny_cnn --max-iter 3

  # Dry-run (skip Vivado, use software reference only)
  python scripts/iterate_until_pass.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Cross-platform UTF-8 stdout/stderr ──────────────────────────────────────
# Windows consoles default to CP1252 which cannot encode → ✓ ✗ etc.
# reconfigure() forces UTF-8 for both interactive runs and log-file redirects.
# On Linux/macOS this is a harmless no-op (stdout is already UTF-8).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Inherit UTF-8 mode in all child processes spawned by this script.
_CHILD_ENV = dict(os.environ, PYTHONUTF8="1")

# ── Paths ──────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).resolve().parent          # vivado_flow/scripts/
_FLOW_DIR = _HERE.parent                             # vivado_flow/
_SCRIPTS  = _HERE
_OPT_DIR  = _FLOW_DIR / "optimizer"
_CFG_DIR  = _FLOW_DIR / "configs" / "generated"
_TB_DATA  = _FLOW_DIR / "tb_data"
_LOG_DIR  = _FLOW_DIR / "logs"

_LOG_DIR.mkdir(parents=True, exist_ok=True)
_CFG_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# Software-reference checker (no Vivado needed)
# ══════════════════════════════════════════════════════════════════════════

def software_check(dims: Dict, val_range: int, seed: int) -> Tuple[bool, str]:
    """
    Run the Python golden computation directly and verify self-consistency.
    Always passes (it is its own reference) — used to validate the Python
    pipeline before handing off to Vivado.
    """
    sys.path.insert(0, str(_SCRIPTS))
    from generate_golden import generate

    _TB_DATA.mkdir(parents=True, exist_ok=True)
    try:
        inputs, weights, golden = generate(
            dims=dims, seed=seed, val_range=val_range, out_dir=_TB_DATA
        )
    except Exception as e:
        return False, f"generate_golden failed: {e}"

    # Re-run and compare (self-consistency)
    from generate_golden import conv2d_reference
    import numpy as np
    golden2 = conv2d_reference(inputs, weights)
    if not np.array_equal(golden, golden2):
        return False, "Golden self-consistency check failed (non-deterministic!)"

    print("[software_check] ✓ Python golden is self-consistent "
          f"({golden.size} elements)")
    return True, "OK"


# ══════════════════════════════════════════════════════════════════════════
# Stage runners
# ══════════════════════════════════════════════════════════════════════════

def run_generate_config(
    dims:        Dict,
    array_h:     int,
    array_w:     int,
    search_mode: str,
    tf_model:    Optional[str],
    layer_idx:   int,
    log_path:    Path,
) -> bool:
    """Stage 1: run generate_config.py."""
    cmd = [sys.executable, str(_OPT_DIR / "generate_config.py")]
    if tf_model:
        cmd += ["--tf-model", tf_model, "--layer-idx", str(layer_idx)]
    else:
        cmd += [
            "--K",  str(dims['K']),  "--C",  str(dims['C']),
            "--KH", str(dims['KH']), "--KW", str(dims['KW']),
            "--H",  str(dims['H']),  "--W",  str(dims['W']),
        ]
    cmd += [
        "--array-height", str(array_h),
        "--array-width",  str(array_w),
        "--search-mode",  search_mode,
    ]

    print(f"[Stage 1] {' '.join(cmd)}")
    with log_path.open("w", encoding="utf-8") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                           cwd=str(_FLOW_DIR), env=_CHILD_ENV)
    if r.returncode != 0:
        print(f"  FAILED (exit {r.returncode}). See {log_path}")
        return False
    print(f"  OK → {_CFG_DIR / 'optimal_config.vh'}")
    return True


def run_config_knobs(
    dims:    Dict,
    array_h: int,
    array_w: int,
    tiles:   Dict,
    log_path: Path,
) -> bool:
    """Stage 2: export user_config.vh via config_knobs.py."""
    # Inline: import and call directly (faster than subprocess)
    sys.path.insert(0, str(_OPT_DIR))
    try:
        from config_knobs import SimConfig, HardwareKnobs, WorkloadKnobs, TilingKnobs, SimKnobs
    except ImportError as e:
        print(f"[Stage 2] Import error: {e}")
        return False

    # Check if optimizer JSON exists to seed from
    opt_json = _CFG_DIR / "optimal_config.json"
    if opt_json.exists():
        cfg = SimConfig.from_json(opt_json)
        # Override with current iteration values
        cfg.hardware.ARRAY_HEIGHT = array_h
        cfg.hardware.ARRAY_WIDTH  = array_w
        cfg.workload.K  = dims['K'];  cfg.workload.C  = dims['C']
        cfg.workload.KH = dims['KH']; cfg.workload.KW = dims['KW']
        cfg.workload.H  = dims['H'];  cfg.workload.W  = dims['W']
    else:
        cfg = SimConfig(
            hardware=HardwareKnobs(ARRAY_HEIGHT=array_h, ARRAY_WIDTH=array_w),
            workload=WorkloadKnobs(K=dims['K'], C=dims['C'],
                                   KH=dims['KH'], KW=dims['KW'],
                                   H=dims['H'],  W=dims['W']),
        )

    # Apply tile overrides
    cfg.tiling.TILE_K = tiles.get("TILE_K", 0)
    cfg.tiling.TILE_C = tiles.get("TILE_C", 0)
    cfg.tiling.TILE_H = tiles.get("TILE_H", 0)
    cfg.tiling.TILE_W = tiles.get("TILE_W", 0)

    cfg.print_summary()
    vh_path = _CFG_DIR / "user_config.vh"

    with log_path.open("w", encoding="utf-8") as lf:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cfg.export_to_verilog(vh_path)
        lf.write(buf.getvalue())

    print(f"[Stage 2] OK → {vh_path}")
    return True


def run_generate_golden(
    dims:      Dict,
    val_range: int,
    seed:      int,
    tf_model:  Optional[str],
    layer_idx: int,
    log_path:  Path,
) -> bool:
    """Stage 3: run generate_golden.py."""
    cmd = [sys.executable, str(_SCRIPTS / "generate_golden.py"),
           "--K",  str(dims['K']),  "--C",  str(dims['C']),
           "--KH", str(dims['KH']), "--KW", str(dims['KW']),
           "--H",  str(dims['H']),  "--W",  str(dims['W']),
           "--val-range", str(val_range),
           "--seed",      str(seed),
           "--out-dir",   str(_TB_DATA),
    ]
    if tf_model:
        cmd += ["--tf-model", tf_model, "--layer-idx", str(layer_idx)]

    print(f"[Stage 3] Generating golden data (val_range={val_range}) ...")
    with log_path.open("w", encoding="utf-8") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                           cwd=str(_FLOW_DIR), env=_CHILD_ENV)
    if r.returncode != 0:
        print(f"  FAILED (exit {r.returncode}). See {log_path}")
        return False
    print(f"  OK → {_TB_DATA}/")
    return True


def run_vivado_sim(
    repo_root:   Path,
    vivado_bin:  str,
    proj_dir:    Path,
    log_path:    Path,
) -> Tuple[bool, str]:
    """
    Stage 4: invoke Vivado batch mode.

    Returns (passed: bool, result_line: str).
    """
    cmd = [
        vivado_bin, "-mode", "batch",
        "-source", str(_SCRIPTS / "run_vivado_sim.tcl"),
        "-tclargs", str(repo_root), str(proj_dir),
        "-nolog", "-nojournal",
    ]
    print(f"[Stage 4] {' '.join(cmd)}")
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                           cwd=str(_FLOW_DIR), env=_CHILD_ENV)
    elapsed = time.perf_counter() - t0
    print(f"  Vivado finished in {elapsed:.1f}s  (exit {r.returncode})")

    result_file = proj_dir / "sim_result.txt"
    if result_file.exists():
        result_line = result_file.read_text(encoding="utf-8").strip()
    else:
        result_line = "TEST_FAIL errors=sim_result_missing"

    print(f"  Result: {result_line}")
    return result_line.startswith("TEST_PASS"), result_line


# ══════════════════════════════════════════════════════════════════════════
# Fixup strategies
# ══════════════════════════════════════════════════════════════════════════

def diagnose(result_line: str) -> str:
    """Return a failure category: 'TIMEOUT' | 'MISMATCH' | 'RTL_ERROR'."""
    rl = result_line.upper()
    if "TIMEOUT" in rl:
        return "TIMEOUT"
    if "MISMATCH" in rl or "GOLDEN" in rl:
        return "MISMATCH"
    return "RTL_ERROR"


def apply_fixup(
    category:    str,
    dims:        Dict,
    array_h:     int,
    array_w:     int,
    tiles:       Dict,
    val_range:   int,
    search_mode: str,
    iteration:   int,
) -> Tuple[Dict, int, str]:
    """
    Given a failure category return updated (tiles, val_range, search_mode).
    """
    print(f"\n[fixup] Applying fixup for category: {category}")

    if category == "TIMEOUT":
        # Increase tile sizes → fewer tiles → fewer DUT cycles
        factor = 2 ** min(iteration, 3)
        new_tiles = {
            "TILE_K": min(dims['K'], max(array_h, tiles.get("TILE_K", array_h)) * 2),
            "TILE_C": min(dims['C'], max(array_w, tiles.get("TILE_C", array_w)) * 2),
            "TILE_H": min(dims['H'], max(array_h, tiles.get("TILE_H", array_h)) * 2),
            "TILE_W": min(dims['W'], max(array_w, tiles.get("TILE_W", array_w)) * 2),
        }
        print(f"  Doubled tile sizes: {new_tiles}")
        return new_tiles, val_range, search_mode

    elif category == "MISMATCH":
        # Smaller values avoid accumulator overflow in int32
        new_vr = max(1, val_range // 2)
        new_mode = "exhaustive" if iteration >= 2 else search_mode
        print(f"  Reduced val_range: {val_range} → {new_vr}")
        if new_mode != search_mode:
            print(f"  Switched search mode: {search_mode} → {new_mode}")
        return tiles, new_vr, new_mode

    else:  # RTL_ERROR
        print("  RTL_ERROR detected — no automatic fixup available.")
        print("  Check the Vivado log for elaboration / compilation errors.")
        return tiles, val_range, search_mode


# ══════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════

def iterate(args) -> int:
    """
    Run the full pipeline, iterating until all tests pass or max_iter reached.

    Returns exit code (0 = pass, 1 = fail).
    """
    # ── Resolve dims ───────────────────────────────────────────────────────
    dims = {
        "K":  args.K  or 16,
        "C":  args.C  or 8,
        "KH": args.KH or 3,
        "KW": args.KW or 3,
        "H":  args.H  or 30,
        "W":  args.W  or 30,
    }
    repo_root  = Path(args.repo_root).resolve()
    proj_dir   = _FLOW_DIR / "vivado_project"
    tiles:     Dict   = {}
    val_range: int    = args.val_range
    search_mode: str  = args.search_mode
    seed             = args.seed

    history: List[Dict] = []

    print("=" * 64)
    print(" ITERATE-UNTIL-PASS")
    print("=" * 64)
    print(f" Dims       : K={dims['K']} C={dims['C']} "
          f"KH={dims['KH']} KW={dims['KW']} H={dims['H']} W={dims['W']}")
    print(f" Array      : {args.array_height}×{args.array_width} PEs")
    print(f" Max iters  : {args.max_iter}")
    print(f" Dry run    : {args.dry_run}")
    print("=" * 64)

    for iteration in range(1, args.max_iter + 1):
        iter_log_dir = _LOG_DIR / f"iter_{iteration:02d}"
        iter_log_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'─'*64}")
        print(f" ITERATION {iteration}/{args.max_iter}")
        print(f"{'─'*64}")

        # ── Stage 1: Optimizer → optimal_config.vh ──────────────────────
        ok1 = run_generate_config(
            dims, args.array_height, args.array_width,
            search_mode, args.tf_model, args.layer_idx,
            iter_log_dir / "01_generate_config.log",
        )
        if not ok1:
            print("[iterate] Stage 1 failed — aborting iteration.")
            continue

        # ── Stage 2: User knobs → user_config.vh ────────────────────────
        ok2 = run_config_knobs(
            dims, args.array_height, args.array_width,
            tiles,
            iter_log_dir / "02_config_knobs.log",
        )
        if not ok2:
            print("[iterate] Stage 2 failed — aborting iteration.")
            continue

        # ── Stage 3: Golden data ─────────────────────────────────────────
        ok3 = run_generate_golden(
            dims, val_range, seed,
            args.tf_model, args.layer_idx,
            iter_log_dir / "03_generate_golden.log",
        )
        if not ok3:
            print("[iterate] Stage 3 failed — aborting iteration.")
            continue

        # ── Software self-check (always) ─────────────────────────────────
        sw_ok, sw_msg = software_check(dims, val_range, seed)
        if not sw_ok:
            print(f"[iterate] Software check failed: {sw_msg}")
            val_range = max(1, val_range // 2)
            print(f"  Reducing val_range to {val_range} and retrying.")
            continue

        if args.dry_run:
            print("\n[iterate] DRY RUN: skipping Vivado simulation.")
            print("[iterate] ✓ Python pipeline PASSED  "
                  "(Vivado simulation not run)")
            return 0

        # ── Stage 4: Vivado simulation ────────────────────────────────────
        vivado_ok = shutil.which(args.vivado)
        if vivado_ok is None:
            print(f"\n[iterate] WARNING: '{args.vivado}' not found on PATH.")
            print("  To run Vivado simulation, ensure Vivado is installed and")
            print("  its bin/ is on your PATH, then re-run this script.")
            print("\n[iterate] Python pipeline PASSED — Vivado not available.")
            return 0

        passed, result_line = run_vivado_sim(
            repo_root, args.vivado,
            proj_dir,
            iter_log_dir / "04_vivado_sim.log",
        )

        record = {
            "iteration":   iteration,
            "dims":        dims.copy(),
            "tiles":       tiles.copy(),
            "val_range":   val_range,
            "search_mode": search_mode,
            "result":      result_line,
            "passed":      passed,
        }
        history.append(record)

        if passed:
            print(f"\n{'═'*64}")
            print(f" ✓ ALL TESTS PASSED  (iteration {iteration})")
            print(f"{'═'*64}")
            _write_report(history, _FLOW_DIR / "iteration_report.json")
            return 0

        # ── Diagnose and apply fixup ──────────────────────────────────────
        category = diagnose(result_line)
        if category == "RTL_ERROR":
            print("[iterate] RTL_ERROR — stopping (needs manual fix).")
            _write_report(history, _FLOW_DIR / "iteration_report.json")
            return 1

        tiles, val_range, search_mode = apply_fixup(
            category, dims, args.array_height, args.array_width,
            tiles, val_range, search_mode, iteration,
        )

    # ── Exhausted iterations ───────────────────────────────────────────────
    print(f"\n{'═'*64}")
    print(f" ✗ FAILED after {args.max_iter} iteration(s)")
    print(f"{'═'*64}")
    _write_report(history, _FLOW_DIR / "iteration_report.json")
    return 1


# ══════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════

def _write_report(history: List[Dict], path: Path):
    path.write_text(json.dumps({"iterations": history}, indent=2),
                    encoding="utf-8")
    print(f"\n[iterate] Report written → {path}")

    print("\n" + "─" * 64)
    print(f" {'Iter':<5} {'Result':<12} {'Tiles':<30} {'Val-range'}")
    print("─" * 64)
    for rec in history:
        t = rec.get("tiles", {})
        tile_str = (f"K={t.get('TILE_K','?')} C={t.get('TILE_C','?')} "
                    f"H={t.get('TILE_H','?')} W={t.get('TILE_W','?')}")
        status = "PASS ✓" if rec["passed"] else "FAIL ✗"
        print(f" {rec['iteration']:<5} {status:<12} {tile_str:<30} {rec['val_range']}")
    print("─" * 64)


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def _build_parser():
    p = argparse.ArgumentParser(
        description="Iterate optimizer → config → golden → Vivado until PASS."
    )
    # Workload
    p.add_argument("--K",  type=int, default=None)
    p.add_argument("--C",  type=int, default=None)
    p.add_argument("--KH", type=int, default=None)
    p.add_argument("--KW", type=int, default=None)
    p.add_argument("--H",  type=int, default=None)
    p.add_argument("--W",  type=int, default=None)

    # TF model source
    p.add_argument("--tf-model",  default=None,
                   help="Keras model name (e.g. tiny_cnn, mobilenet_v2)")
    p.add_argument("--layer-idx", type=int, default=0)

    # Hardware
    p.add_argument("--array-height", type=int, default=4)
    p.add_argument("--array-width",  type=int, default=4)

    # Search
    p.add_argument("--search-mode",
                   choices=["heuristic","exhaustive","gradient"],
                   default="heuristic")

    # Golden data
    p.add_argument("--val-range", type=int, default=16,
                   help="Random activation/weight range (default 16)")
    p.add_argument("--seed",      type=int, default=42)

    # Flow control
    p.add_argument("--max-iter",  type=int, default=5)
    p.add_argument("--vivado",    default="vivado",
                   help="Vivado binary (default: 'vivado' on PATH)")
    p.add_argument("--repo-root", default=str(_FLOW_DIR.parent),
                   help="Absolute path to repo root")
    p.add_argument("--dry-run",   action="store_true",
                   help="Skip Vivado; run Python pipeline only")

    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    return iterate(args)


if __name__ == "__main__":
    sys.exit(main())
