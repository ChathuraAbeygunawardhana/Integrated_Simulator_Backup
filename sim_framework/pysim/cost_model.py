"""SimulatorCostModel: drop-in replacement for the optimizer's
MemoryAccessAnalyzer that scores configurations by running the unified
simulator (single_dnn_top) under Verilator+cocotb.

Two-stage usage (recommended): the optimizer pre-filters with the analytical
MemoryAccessAnalyzer, then re-scores the top-N survivors with this model.
Pure simulator-only search is supported but slow.

Caching: results are keyed on SimConfig.hash() AND a manual RTL_REVISION env
var so that RTL changes invalidate the cache.
"""
from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

from . import runner
from .sim_config import LayerConfig, SimConfig

CACHE_DIR = Path(__file__).resolve().parents[1] / "pysim" / "cache"


class SimulatorCostModel:
    """Plug-in analyzer that runs the cycle-accurate simulator.

    Constructor mirrors MemoryAccessAnalyzer's. base_cfg supplies the rest
    of the knobs (dataflow, memory, array size, etc.) that the simulator
    needs but the analytical analyzer does not care about.
    """

    def __init__(
        self,
        K: int, C: int, H: int, W: int, KH: int, KW: int,
        base_cfg: Optional[SimConfig] = None,
        scratchpad_bytes: int = 16_384,
        test_module: str = "test_single_dnn_smoke",
    ) -> None:
        self.K, self.C, self.H, self.W = K, C, H, W
        self.KH, self.KW = KH, KW
        self.scratchpad_bytes = scratchpad_bytes
        self.base_cfg = base_cfg or SimConfig()
        self.test_module = test_module
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # MemoryAccessAnalyzer-compatible API
    # ------------------------------------------------------------------

    def analyze(self, config) -> Dict:
        """Score a (blocking, tile-loop-order) configuration via simulator."""
        cfg = self._sim_config_for(config)
        cached = self._cache_get(cfg)
        if cached is not None:
            return cached

        res = runner.run(cfg, test_module=self.test_module)
        stats = res.stats

        # Map simulator stats into the analytical analyzer's key set so the
        # optimizer's _score function can consume either.
        dram_bytes = float(stats.get("bytes_loaded", 0))
        cycles     = float(res.cycles or stats.get("cycles", 1))
        spad_need  = int(stats.get("scratchpad_need", 0))

        out = dict(
            dram_traffic=dram_bytes if dram_bytes > 0 else cycles,
            weight_reuse=float(stats.get("weight_reuse",  1.0)),
            input_reuse= float(stats.get("input_reuse",   1.0)),
            weight_traffic=float(stats.get("weight_traffic", 0.0)),
            input_traffic= float(stats.get("input_traffic",  0.0)),
            output_traffic=float(stats.get("output_traffic", 0.0)),
            score=cycles if res.passed else cycles * 10.0,
            scratchpad_fits=(spad_need == 0 or spad_need <= self.scratchpad_bytes),
            scratchpad_need=spad_need,
            sim_cycles=cycles,
            sim_passed=bool(res.passed),
        )
        self._cache_put(cfg, out)
        return out

    def traffic_continuous(self, tk: float, tc: float, th: float, tw: float, order) -> float:
        """Not supported on the simulator backend (each eval is seconds and
        the function is not differentiable)."""
        raise NotImplementedError(
            "SimulatorCostModel does not support gradient-descent search. "
            "Use SearchMode.HEURISTIC or SearchMode.EXHAUSTIVE."
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sim_config_for(self, lnc) -> SimConfig:
        """Build a SimConfig from a LoopNestConfig (tile sizes + order)."""
        b = lnc.blocking
        cfg = deepcopy(self.base_cfg)
        layer = LayerConfig(
            input_channels=self.C,
            input_height=self.H,
            input_width=self.W,
            weight_k=self.K,
            weight_c=self.C,
            weight_kh=self.KH,
            weight_kw=self.KW,
            output_channels=self.K,
            output_height=max(1, self.H - self.KH + 1),
            output_width=max(1, self.W - self.KW + 1),
        )
        cfg.layers = [layer]
        return cfg

    def _cache_key(self, cfg: SimConfig) -> Path:
        rtl_rev = os.environ.get("RTL_REVISION", "default")
        return CACHE_DIR / f"{rtl_rev}_{cfg.hash()}.json"

    def _cache_get(self, cfg: SimConfig) -> Optional[Dict]:
        p = self._cache_key(cfg)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def _cache_put(self, cfg: SimConfig, data: Dict) -> None:
        self._cache_key(cfg).write_text(json.dumps(data))
