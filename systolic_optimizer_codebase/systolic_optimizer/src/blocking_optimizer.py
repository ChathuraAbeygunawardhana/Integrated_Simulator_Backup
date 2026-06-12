#!/usr/bin/env python3
"""
Blocking-First Loop Optimization
=================================

Three search modes
------------------
  HEURISTIC  — powers-of-2 tile sizes (fast, ~hundreds of configs)
  EXHAUSTIVE — every divisor of K, C, H, W x all 6 loop orders (exact)
  GRADIENT   — gradient descent in continuous tile-size space,
               snapping to integer divisors each step. Warm-started
               from the heuristic best; supports multi-restart.
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------

class SearchMode(Enum):
    HEURISTIC  = "heuristic"
    EXHAUSTIVE = "exhaustive"
    GRADIENT   = "gradient"


class TileLoopOrder(Enum):
    K_C_SPATIAL = "k_tile, c_tile, kh, kw"
    K_SPATIAL_C = "k_tile, kh, kw, c_tile"
    C_K_SPATIAL = "c_tile, k_tile, kh, kw"
    C_SPATIAL_K = "c_tile, kh, kw, k_tile"
    SPATIAL_K_C = "kh, kw, k_tile, c_tile"
    SPATIAL_C_K = "kh, kw, c_tile, k_tile"


@dataclass
class BlockingConfig:
    tile_K:  int = 16
    tile_C:  int = 16
    tile_H:  int = 14
    tile_W:  int = 14
    tile_KH: int = 0
    tile_KW: int = 0
    inner_h: int = 4
    inner_w: int = 4

    def scratchpad_requirement(
        self, K: int, C: int, H: int, W: int, KH: int, KW: int
    ) -> int:
        input_tile  = (self.tile_H + KH - 1) * (self.tile_W + KW - 1) * self.tile_C * 4
        weight_tile = KH * KW * self.tile_C * self.tile_K * 4
        output_tile = self.tile_H * self.tile_W * self.tile_K * 4
        return input_tile + weight_tile + output_tile

    def to_dict(self) -> Dict:
        return dict(tile_K=self.tile_K, tile_C=self.tile_C,
                    tile_H=self.tile_H, tile_W=self.tile_W,
                    inner_h=self.inner_h, inner_w=self.inner_w)


@dataclass
class LoopNestConfig:
    blocking:        BlockingConfig
    tile_loop_order: TileLoopOrder

    def __repr__(self) -> str:
        b = self.blocking
        return (f"LoopNest(blocking={b.tile_K}K×{b.tile_C}C"
                f"×{b.tile_H}H×{b.tile_W}W, order={self.tile_loop_order.name})")


# ---------------------------------------------------------------------------
# Analytical DRAM-traffic cost model
# ---------------------------------------------------------------------------

class MemoryAccessAnalyzer:
    """
    Closed-form DRAM-traffic model.  O(1) per evaluation, differentiable
    w.r.t. tile sizes (used by gradient-descent mode).

    Weight reuse model:
      For K/C-outer orders, a weight tile (TK×TC×KH×KW) is reused over
      n_h * n_w spatial tile iterations before being evicted.
      For spatial-outer orders the weight tile is streamed once per
      spatial tile, so reuse is lower by a factor of KH*KW.

    Input reuse model:
      An input tile is reused across all output-channel (K) iterations
      that share the same (C, H, W) position.  Spatial-outer orders keep
      the spatial tile resident longest, maximising input reuse.
    """

    def __init__(
        self,
        K: int, C: int, H: int, W: int, KH: int, KW: int,
        scratchpad_bytes: int = 16_384,
    ) -> None:
        self.K  = K;  self.C  = C
        self.H  = H;  self.W  = W
        self.KH = KH; self.KW = KW
        self.scratchpad_bytes = scratchpad_bytes

    # -- public ---------------------------------------------------------

    def analyze(self, config: LoopNestConfig) -> Dict:
        res = self._eval(
            float(config.blocking.tile_K), float(config.blocking.tile_C),
            float(config.blocking.tile_H), float(config.blocking.tile_W),
            config.tile_loop_order,
        )
        spad = config.blocking.scratchpad_requirement(
            self.K, self.C, self.H, self.W, self.KH, self.KW
        )
        fits = spad <= self.scratchpad_bytes
        res["scratchpad_fits"] = fits
        res["scratchpad_need"] = spad
        if not fits:
            res["dram_traffic"] *= 1.5
            res["score"]        *= 1.5
        return res

    def traffic_continuous(
        self, tk: float, tc: float, th: float, tw: float,
        order: TileLoopOrder,
    ) -> float:
        """Differentiable surrogate — accepts real-valued tile sizes."""
        return self._eval(tk, tc, th, tw, order)["dram_traffic"]

    # -- internals ------------------------------------------------------

    def _eval(
        self,
        tk: float, tc: float, th: float, tw: float,
        order: TileLoopOrder,
    ) -> Dict:
        K, C, H, W, KH, KW = self.K, self.C, self.H, self.W, self.KH, self.KW

        tk = max(tk, 1.0); tc = max(tc, 1.0)
        th = max(th, 1.0); tw = max(tw, 1.0)

        n_k = K / tk; n_c = C / tc
        n_h = H / th; n_w = W / tw

        # weight reuse
        if order in (TileLoopOrder.K_C_SPATIAL, TileLoopOrder.K_SPATIAL_C,
                     TileLoopOrder.C_K_SPATIAL, TileLoopOrder.C_SPATIAL_K):
            weight_reuse = max(1.0, n_h * n_w)
        else:  # spatial outermost — lower weight reuse
            weight_reuse = max(1.0, n_h * n_w / max(1.0, float(KH * KW)))

        # input reuse
        if order in (TileLoopOrder.SPATIAL_K_C, TileLoopOrder.SPATIAL_C_K):
            input_reuse = max(1.0, n_k * n_c)
        else:
            input_reuse = max(1.0, n_k)

        weight_bytes = KH * KW * C * K * 4
        input_bytes  = (H + KH - 1) * (W + KW - 1) * C * 4
        output_bytes = H * W * K * 4

        wt = weight_bytes / weight_reuse
        it = input_bytes  / input_reuse
        ot = output_bytes

        total = wt + it + ot
        return dict(
            dram_traffic=total,
            weight_reuse=weight_reuse, input_reuse=input_reuse,
            weight_traffic=wt, input_traffic=it, output_traffic=ot,
            score=total,
            scratchpad_fits=True, scratchpad_need=0,
        )


# ---------------------------------------------------------------------------
# Helper: integer divisors & heuristic candidates
# ---------------------------------------------------------------------------

def _divisors(n: int) -> List[int]:
    divs = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)


def _pow2_candidates(n: int, limit: int = 6) -> List[int]:
    cands = {2**i for i in range(1, 8) if 2**i <= n}
    cands.add(n)
    return sorted(cands)[-limit:]


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

class BlockingOptimizer:
    """
    Joint (blocking × loop-order) optimizer.

    Quick start
    -----------
        opt = BlockingOptimizer(K=128, C=64, H=28, W=28, KH=3, KW=3)

        best, stats = opt.search(mode=SearchMode.HEURISTIC)
        best, stats = opt.search(mode=SearchMode.EXHAUSTIVE)
        best, stats = opt.search(mode=SearchMode.GRADIENT, gd_iterations=300)
    """

    def __init__(
        self,
        K: int, C: int, H: int, W: int, KH: int, KW: int,
        array_height: int = 4, array_width: int = 4,
        scratchpad_kb: int = 16,
    ) -> None:
        self.K = K; self.C = C; self.H = H; self.W = W
        self.KH = KH; self.KW = KW
        self.array_height = array_height
        self.array_width  = array_width
        self.scratchpad_bytes = scratchpad_kb * 1024
        self._analyzer = MemoryAccessAnalyzer(
            K, C, H, W, KH, KW, self.scratchpad_bytes
        )

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def search(
        self,
        mode: SearchMode = SearchMode.HEURISTIC,
        objective: str = "min_traffic",
        verbose: bool = True,
        # gradient-descent knobs
        gd_iterations: int   = 300,
        gd_lr:         float = 0.05,
        gd_momentum:   float = 0.9,
        gd_restarts:   int   = 3,
    ) -> Tuple[LoopNestConfig, Dict[str, Dict]]:
        """
        Search for the optimal LoopNestConfig.

        Parameters
        ----------
        mode          : HEURISTIC | EXHAUSTIVE | GRADIENT
        objective     : 'min_traffic' | 'max_reuse' | 'balanced'
        verbose       : print progress and best result
        gd_iterations : GD steps per (order, restart) pair
        gd_lr         : log-space learning rate
        gd_momentum   : momentum coefficient
        gd_restarts   : random restarts per loop order (on top of warm-start)
        """
        if verbose:
            self._print_header(mode, objective)

        if mode == SearchMode.HEURISTIC:
            return self._grid_search(
                self._heuristic_configs(), objective, verbose
            )
        elif mode == SearchMode.EXHAUSTIVE:
            return self._grid_search(
                self._exhaustive_configs(), objective, verbose
            )
        elif mode == SearchMode.GRADIENT:
            return self._gradient_search(
                objective, verbose,
                gd_iterations, gd_lr, gd_momentum, gd_restarts,
            )
        else:
            raise ValueError(f"Unknown SearchMode: {mode}")

    # backward-compat alias
    def search_optimal_config(
        self, optimization_objective: str = "min_traffic"
    ) -> Tuple[LoopNestConfig, Dict]:
        return self.search(mode=SearchMode.HEURISTIC,
                           objective=optimization_objective)

    # ------------------------------------------------------------------
    # Grid search
    # ------------------------------------------------------------------

    def _grid_search(
        self,
        blocking_cfgs: List[BlockingConfig],
        objective: str,
        verbose: bool,
    ) -> Tuple[LoopNestConfig, Dict]:
        orders = list(TileLoopOrder)
        total  = len(blocking_cfgs) * len(orders)
        if verbose:
            print(f"  Blocking configs : {len(blocking_cfgs)}")
            print(f"  Loop orders      : {len(orders)}")
            print(f"  Total evaluations: {total}")

        all_results: Dict[str, Dict] = {}
        best_cfg:   Optional[LoopNestConfig] = None
        best_score = float("inf")

        for bc in blocking_cfgs:
            for order in orders:
                cfg     = LoopNestConfig(bc, order)
                metrics = self._analyzer.analyze(cfg)
                score   = self._score(metrics, objective)
                metrics["score"] = score
                all_results[str(cfg)] = metrics
                if score < best_score:
                    best_score = score
                    best_cfg   = cfg

        if verbose:
            self._print_result(best_cfg, all_results[str(best_cfg)])
        return best_cfg, all_results

    # ------------------------------------------------------------------
    # Gradient-descent search
    # ------------------------------------------------------------------

    def _gradient_search(
        self,
        objective: str,
        verbose: bool,
        iterations: int,
        lr: float,
        momentum: float,
        restarts: int,
    ) -> Tuple[LoopNestConfig, Dict]:
        """
        Gradient descent over log(tile_size) for each of the 6 loop orders,
        with multiple random restarts + a heuristic warm-start.

        Steps
        -----
        1. Run a heuristic sweep to get a warm-start point.
        2. For each TileLoopOrder × (warm-start + N random restarts):
             a. Represent tiles as x = log(tile_size).
             b. Each iteration:
                  - Compute ∂traffic/∂x_i via central finite differences.
                  - Update x with momentum SGD.
                  - Clip x so tile stays in [1, dim].
                  - Snap tile to nearest integer divisor of that dimension.
                  - Evaluate exact integer cost; track best seen.
        3. Return global best across all orders and restarts.
        """
        if verbose:
            print(f"  GD iterations per run : {iterations}")
            print(f"  Learning rate         : {lr}  (log-space)")
            print(f"  Momentum              : {momentum}")
            print(f"  Random restarts/order : {restarts}")
            print(f"  Total GD runs         : {len(TileLoopOrder) * (restarts + 1)}")

        # warm-start from heuristic best
        warm_cfg, _ = self._grid_search(
            self._heuristic_configs(), objective, verbose=False
        )
        warm_start = (
            warm_cfg.blocking.tile_K, warm_cfg.blocking.tile_C,
            warm_cfg.blocking.tile_H, warm_cfg.blocking.tile_W,
        )

        divs = {
            "K": _divisors(self.K), "C": _divisors(self.C),
            "H": _divisors(self.H), "W": _divisors(self.W),
        }

        all_results: Dict[str, Dict] = {}
        global_best_cfg:     Optional[LoopNestConfig] = None
        global_best_score  = float("inf")

        for order in TileLoopOrder:
            starts = [warm_start] + [
                (random.choice(divs["K"]), random.choice(divs["C"]),
                 random.choice(divs["H"]), random.choice(divs["W"]))
                for _ in range(restarts)
            ]

            for start in starts:
                cfg, metrics = self._gd_run(
                    order, start, divs, objective, iterations, lr, momentum
                )
                key = str(cfg)
                # keep the best metrics seen for this exact config string
                if key not in all_results or metrics["score"] < all_results[key]["score"]:
                    all_results[key] = metrics
                if metrics["score"] < global_best_score:
                    global_best_score = metrics["score"]
                    global_best_cfg   = cfg

        if verbose:
            self._print_result(global_best_cfg, all_results[str(global_best_cfg)])
        return global_best_cfg, all_results

    def _gd_run(
        self,
        order:      TileLoopOrder,
        start:      Tuple[int, int, int, int],
        divs:       Dict[str, List[int]],
        objective:  str,
        iterations: int,
        lr:         float,
        momentum:   float,
    ) -> Tuple[LoopNestConfig, Dict]:
        """Single GD trajectory for a fixed loop order."""
        x = np.log(np.array(start, dtype=float).clip(1.0))
        v = np.zeros(4)   # velocity (momentum)
        h = 0.01          # finite-difference step

        dim_max = np.log([self.K, self.C, self.H, self.W])
        best_score   = float("inf")
        best_cfg     = None
        best_metrics = None

        for _ in range(iterations):
            # numerical gradient: ∂L/∂x_i via central differences
            grad = np.zeros(4)
            for i in range(4):
                xf = x.copy(); xf[i] += h
                xb = x.copy(); xb[i] -= h
                tf = np.exp(xf); tb = np.exp(xb)
                Lf = self._analyzer.traffic_continuous(tf[0],tf[1],tf[2],tf[3],order)
                Lb = self._analyzer.traffic_continuous(tb[0],tb[1],tb[2],tb[3],order)
                grad[i] = (Lf - Lb) / (2 * h)

            # momentum SGD step (minimising, so subtract)
            v = momentum * v + lr * grad
            x = x - v

            # clip to valid log-range
            x = np.clip(x, 0.0, dim_max)

            # snap to nearest integer divisor
            t = np.exp(x)
            tk = self._snap(t[0], divs["K"])
            tc = self._snap(t[1], divs["C"])
            th = self._snap(t[2], divs["H"])
            tw = self._snap(t[3], divs["W"])

            bc  = BlockingConfig(
                tile_K=tk, tile_C=tc, tile_H=th, tile_W=tw,
                inner_h=self.array_height, inner_w=self.array_width,
            )
            cfg     = LoopNestConfig(bc, order)
            metrics = self._analyzer.analyze(cfg)
            score   = self._score(metrics, objective)
            metrics["score"] = score

            if score < best_score:
                best_score   = score
                best_cfg     = cfg
                best_metrics = metrics

        return best_cfg, best_metrics

    # ------------------------------------------------------------------
    # Candidate generators
    # ------------------------------------------------------------------

    def _heuristic_configs(self) -> List[BlockingConfig]:
        return self._build_configs(
            _pow2_candidates(self.K), _pow2_candidates(self.C),
            _pow2_candidates(self.H), _pow2_candidates(self.W),
        )

    def _exhaustive_configs(self) -> List[BlockingConfig]:
        return self._build_configs(
            _divisors(self.K), _divisors(self.C),
            _divisors(self.H), _divisors(self.W),
        )

    def _build_configs(
        self,
        k_cands: List[int], c_cands: List[int],
        h_cands: List[int], w_cands: List[int],
    ) -> List[BlockingConfig]:
        configs = []
        for tk, tc, th, tw in itertools.product(k_cands, c_cands, h_cands, w_cands):
            bc = BlockingConfig(
                tile_K=tk, tile_C=tc, tile_H=th, tile_W=tw,
                inner_h=self.array_height, inner_w=self.array_width,
            )
            spad = bc.scratchpad_requirement(
                self.K, self.C, self.H, self.W, self.KH, self.KW
            )
            if spad <= self.scratchpad_bytes * 1.2:
                configs.append(bc)
        return configs

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _snap(value: float, divisors: List[int]) -> int:
        return min(divisors, key=lambda d: abs(d - value))

    def _score(self, metrics: Dict, objective: str) -> float:
        if objective == "min_traffic":
            return metrics["dram_traffic"]
        elif objective == "max_reuse":
            return -(metrics["weight_reuse"] + metrics["input_reuse"])
        else:  # balanced
            return (metrics["dram_traffic"] / 1e6
                    - (metrics["weight_reuse"] + metrics["input_reuse"]) / 10.0)


    def search_space_report(
        self,
        gd_results:  Optional[Dict[str, Dict]] = None,
        exh_results: Optional[Dict[str, Dict]] = None,
    ) -> None:
        """
        Print a breakdown of the search space and, optionally, how much
        of it gradient descent covered and what optimality gap it achieved.

        Call after running .search() with the returned all_results dicts:

            best_h, exh = opt.search(mode=SearchMode.EXHAUSTIVE)
            best_g, gd  = opt.search(mode=SearchMode.GRADIENT)
            opt.search_space_report(gd_results=gd, exh_results=exh)
        """
        dk = _divisors(self.K); dc = _divisors(self.C)
        dh = _divisors(self.H); dw = _divisors(self.W)
        n_orders   = len(list(TileLoopOrder))
        full_space = len(dk) * len(dc) * len(dh) * len(dw) * n_orders

        print()
        print("Search-space breakdown")
        print("─" * 60)
        print(f"  Dimension divisors  K={len(dk)} C={len(dc)} H={len(dh)} W={len(dw)}")
        print(f"  Loop orders         {n_orders}")
        print(f"  Full grid           {full_space:>10,}  (divisors × orders)")

        if exh_results is not None:
            exh_n = len(exh_results)
            pct_filtered = (1 - exh_n / full_space) * 100
            print(f"  After spad filter   {exh_n:>10,}  "
                  f"({pct_filtered:.1f}% removed by scratchpad)")

            if gd_results is not None:
                gd_n = len(gd_results)
                reduction  = (1 - gd_n / exh_n) * 100
                exh_best   = min(v["dram_traffic"] for v in exh_results.values())
                gd_best    = min(v["dram_traffic"] for v in gd_results.values())
                gap_pct    = (gd_best - exh_best) / exh_best * 100
                speedup    = exh_n / max(gd_n, 1)
                print(f"  GD visited          {gd_n:>10,}  "
                      f"({reduction:.1f}% reduction vs exhaustive, "
                      f"{speedup:.0f}× fewer evals)")
                print(f"  Optimality gap      {gap_pct:>+10.4f}%  "
                      f"(GD best vs exhaustive best)")
        elif gd_results is not None:
            gd_n = len(gd_results)
            reduction = (1 - gd_n / full_space) * 100
            print(f"  GD visited          {gd_n:>10,}  "
                  f"({reduction:.1f}% reduction vs full grid)")
        print("─" * 60)

    def _print_header(self, mode: SearchMode, objective: str) -> None:
        print(f"\nBlockingOptimizer  K={self.K} C={self.C} "
              f"H={self.H}×{self.W}  kernel={self.KH}×{self.KW}")
        print(f"  Scratchpad : {self.scratchpad_bytes / 1024:.1f} KB")
        print(f"  Mode       : {mode.value}")
        print(f"  Objective  : {objective}")

    def _print_result(self, cfg: LoopNestConfig, metrics: Dict) -> None:
        print(f"\n  ── Best config ──────────────────────────────────────────")
        print(f"  {cfg}")
        print(f"  DRAM traffic  : {metrics['dram_traffic'] / 1e6:.4f} MB")
        print(f"  Weight reuse  : {metrics['weight_reuse']:.2f}×")
        print(f"  Input reuse   : {metrics['input_reuse']:.2f}×")
        spad = metrics["scratchpad_need"] / 1024
        fits = "✓ fits" if metrics["scratchpad_fits"] else "✗ overflow"
        print(f"  Scratchpad    : {spad:.1f} KB ({fits})")
        print(f"  ─────────────────────────────────────────────────────────")

    def print_top(self, all_results: Dict[str, Dict], top_n: int = 5) -> None:
        ranked = sorted(all_results.items(), key=lambda x: x[1]["score"])
        print(f"\n{'Rank':<5} {'Config':<58} {'Traffic MB':<12} "
              f"{'W-reuse':<9} {'I-reuse':<9} {'Spad KB'}")
        print("─" * 108)
        for rank, (cfg_str, m) in enumerate(ranked[:top_n], 1):
            spad = m["scratchpad_need"] / 1024
            tick = "✓" if m["scratchpad_fits"] else "✗"
            print(f"{rank:<5} {cfg_str[:57]:<58} "
                  f"{m['dram_traffic']/1e6:<12.4f}"
                  f"{m['weight_reuse']:<9.1f}"
                  f"{m['input_reuse']:<9.1f}"
                  f"{spad:.1f}{tick}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo() -> None:
    import time

    print("\n" + "=" * 70)
    print("BLOCKING OPTIMIZER  —  HEURISTIC vs EXHAUSTIVE vs GRADIENT")
    print("=" * 70)

    layer = dict(K=64, C=32, H=28, W=28, KH=3, KW=3,
                 array_height=8, array_width=8, scratchpad_kb=64)

    results_by_mode = {}
    for mode in SearchMode:
        opt = BlockingOptimizer(**layer)
        t0  = time.perf_counter()
        best, results = opt.search(mode=mode, verbose=True)
        elapsed = time.perf_counter() - t0
        opt.print_top(results, top_n=3)
        print(f"  Wall time : {elapsed*1000:.1f} ms   "
              f"Unique configs evaluated: {len(results)}\n")
        results_by_mode[mode] = (best, elapsed)

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Mode':<12} {'Best config':<65} {'Time ms'}")
    print("─" * 90)
    for mode, (best, t) in results_by_mode.items():
        print(f"{mode.value:<12} {str(best)[:64]:<65} {t*1000:.1f}")


if __name__ == "__main__":
    demo()
