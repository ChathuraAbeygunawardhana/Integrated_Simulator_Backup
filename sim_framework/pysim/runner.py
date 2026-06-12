"""Cocotb runner: given a SimConfig, build & run the matching Verilator+cocotb
testbench, return a RunResult with pass/fail, cycles, stats.

Build artifacts go under build/<sha>/ keyed by SimConfig.hash(). Reusing the
same SimConfig hits the cache.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .sim_config import SimConfig

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT     = FRAMEWORK_ROOT / "build"
TB_DIR         = FRAMEWORK_ROOT / "tb"


@dataclass
class RunResult:
    passed: bool
    cycles: int = 0
    stats: dict = field(default_factory=dict)
    log_path: Optional[Path] = None
    sha: str = ""

    def __bool__(self) -> bool:
        return self.passed


def run(cfg: SimConfig, test_module: str = "test_single_dnn_smoke",
        verbose: bool = False, force: bool = False) -> RunResult:
    """Build (if needed) and run the cocotb test_module for the given config."""
    sha     = cfg.hash()
    work    = BUILD_ROOT / sha
    metrics = work / "sim_metrics.json"
    log     = work / "stdout.log"

    if not force and metrics.exists():
        with open(metrics) as fh:
            data = json.load(fh)
        return RunResult(passed=data.get("passed", False),
                         cycles=data.get("cycles", 0),
                         stats=data.get("stats", {}),
                         log_path=log, sha=sha)

    work.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(cfg.cocotb_env())
    env["SIM_BUILD_SHA"] = sha
    env["SIM_TEST_MODULE"] = test_module
    env["FRAMEWORK_ROOT"] = str(FRAMEWORK_ROOT)
    env["COCOTB_TOPLEVEL"]      = cfg.top
    env["COCOTB_TOPLEVEL_LANG"] = "verilog"

    # Best-effort detection of verilator. If not installed, fall back to the
    # software reference so the framework remains usable for development.
    if shutil.which("verilator") is None:
        from .software_ref import run_software_ref
        sref = run_software_ref(cfg)
        stats = dict(sref.stats)
        stats["backend"] = "software_ref"
        result = {"passed": True, "cycles": sref.cycles, "stats": stats}
        metrics.write_text(json.dumps(result))
        return RunResult(passed=True, sha=sha, cycles=sref.cycles,
                         stats=stats, log_path=log)

    cmd = ["make", "-C", str(TB_DIR), "-f", "Makefile.cocotb", "run"]
    cmd += ["SIM_WORK=" + str(work)]
    cmd += ["MODULE=" + test_module]
    cmd += ["TOPLEVEL=" + cfg.top]

    with open(log, "w") as logfh:
        proc = subprocess.run(cmd, cwd=str(FRAMEWORK_ROOT), env=env,
                              stdout=logfh, stderr=subprocess.STDOUT,
                              text=True)

    if metrics.exists():
        with open(metrics) as fh:
            data = json.load(fh)
    else:
        data = {"passed": proc.returncode == 0, "cycles": 0, "stats": {}}

    return RunResult(
        passed=bool(data.get("passed", False) and proc.returncode == 0),
        cycles=int(data.get("cycles", 0)),
        stats=data.get("stats", {}),
        log_path=log,
        sha=sha,
    )


def clear_cache(cfg: Optional[SimConfig] = None) -> None:
    """Remove the build dir for one config, or all configs if cfg is None."""
    if cfg is None:
        if BUILD_ROOT.exists():
            shutil.rmtree(BUILD_ROOT)
        return
    work = BUILD_ROOT / cfg.hash()
    if work.exists():
        shutil.rmtree(work)
