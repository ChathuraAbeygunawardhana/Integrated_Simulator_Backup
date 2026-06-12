"""Aggregate RunResult records into a matrix table (CSV + Markdown)."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .runner import RunResult
from .sim_config import SimConfig


def write_csv(results: Iterable[tuple[SimConfig, RunResult]], out_csv: Path) -> None:
    rows = []
    for cfg, res in results:
        rows.append({
            "sha":         res.sha,
            "top":         cfg.top,
            "dataflow":    cfg.dataflow,
            "memory":      cfg.memory,
            "array":       f"{cfg.array_height}x{cfg.array_width}",
            "scheduler":   cfg.scheduler or "-",
            "passed":      res.passed,
            "cycles":      res.cycles,
            "skip_reason": res.stats.get("skip_reason", ""),
        })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        if not rows:
            return
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_markdown_matrix(results: Iterable[tuple[SimConfig, RunResult]], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Verification matrix\n",
             "| sha | top | dataflow | memory | array | scheduler | passed | cycles | note |",
             "|-----|-----|----------|--------|-------|-----------|--------|--------|------|"]
    passed = 0
    total = 0
    for cfg, res in results:
        total += 1
        if res.passed: passed += 1
        lines.append(
            f"| `{res.sha}` | {cfg.top} | {cfg.dataflow} | {cfg.memory} | "
            f"{cfg.array_height}x{cfg.array_width} | {cfg.scheduler or '-'} | "
            f"{'PASS' if res.passed else 'FAIL'} | {res.cycles} | "
            f"{res.stats.get('skip_reason', '')} |"
        )
    summary = f"\nPassed: **{passed}/{total}**\n"
    out_md.write_text("\n".join(lines) + summary)
