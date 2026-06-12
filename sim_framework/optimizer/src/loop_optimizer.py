#!/usr/bin/env python3
"""
Loop Optimization for Output-Stationary Systolic Arrays
=======================================================

Dimension glossary
------------------
K    : Output channels  — number of filters (how many feature maps we produce)
C    : Input channels   — depth of the input feature map (e.g. 3 for RGB)
KH   : Kernel Height    — number of rows in each filter (e.g. 3 for a 3×3 kernel)
KW   : Kernel Width     — number of columns in each filter
H_in : Input Height     — spatial row count of the input feature map
W_in : Input Width      — spatial column count of the input feature map

Derived (not looped over — computed from above):
  H_out = H_in - KH + 1   (valid convolution, no padding)
  W_out = W_in - KW + 1

For output-stationary dataflow the 6 outer loops can be permuted freely.
That gives 6! = 720 possible orders; we expose the 6 most practically
useful ones as a LoopOrder6D enum and also provide ALL_LOOP_ORDERS for
exhaustive search.
"""

from enum import Enum
from itertools import permutations
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# All 720 permutations (use for exhaustive search)
# ---------------------------------------------------------------------------
LOOP_DIMS = ("K", "C", "KH", "KW", "H_in", "W_in")
ALL_LOOP_ORDERS: List[Tuple[str, ...]] = list(permutations(LOOP_DIMS))


class _AnonOrder:
    """
    Lightweight wrapper for a raw permutation tuple that is not one of the
    6 named LoopOrder6D presets.  Used by the exhaustive search so all 720
    permutations can flow through the same analysis path without defining
    720 enum members.
    """
    def __init__(self, perm: "Tuple[str, ...]") -> None:
        self.value = perm

    def __repr__(self) -> str:
        return "\u2192".join(self.value)



# ---------------------------------------------------------------------------
# Named enum for the most commonly analysed orders
# ---------------------------------------------------------------------------
class LoopOrder6D(Enum):
    """
    Each value is a tuple of dimension names ordered outermost → innermost.

    K    = output channels
    C    = input channels
    KH   = kernel height
    KW   = kernel width
    H_in = input feature map height
    W_in = input feature map width
    """

    # weight-stationary friendly: weights change slowest
    K_C_KH_KW_HIN_WIN = ("K", "C", "KH", "KW", "H_in", "W_in")
    # input-channel outermost: maximises C reuse per pass
    C_K_KH_KW_HIN_WIN = ("C", "K", "KH", "KW", "H_in", "W_in")
    # spatial-input outermost: streams input tile once per filter position
    HIN_WIN_K_C_KH_KW = ("H_in", "W_in", "K", "C", "KH", "KW")
    # kernel-spatial outermost: reuses the same kernel window across all channels
    KH_KW_C_K_HIN_WIN = ("KH", "KW", "C", "K", "H_in", "W_in")
    # input-reuse optimised: C and kernel dims together before switching K
    C_KH_KW_K_HIN_WIN = ("C", "KH", "KW", "K", "H_in", "W_in")
    # output-channel + output-spatial grouped
    K_HIN_WIN_C_KH_KW = ("K", "H_in", "W_in", "C", "KH", "KW")


# ---------------------------------------------------------------------------
# Tiling configuration
# ---------------------------------------------------------------------------
class LoopTileConfig:
    """Tile sizes for each dimension."""

    def __init__(
        self,
        tile_k: int = 4,
        tile_c: int = 4,
        tile_hin: int = 4,  # maps to systolic array rows
        tile_win: int = 4,  # maps to systolic array columns
    ):
        self.tile_k = tile_k
        self.tile_c = tile_c
        self.tile_hin = tile_hin
        self.tile_win = tile_win


# ---------------------------------------------------------------------------
# Core optimiser
# ---------------------------------------------------------------------------
class LoopOptimizer:
    """Analyse and optimise loop orders for a systolic array."""

    def __init__(self, array_height: int = 4, array_width: int = 4):
        self.array_height = array_height
        self.array_width = array_width

    # ------------------------------------------------------------------
    # Iteration-sequence generation
    # ------------------------------------------------------------------
    def generate_iteration_sequence(
        self,
        loop_order: LoopOrder6D,
        k: int,  # total output channels
        c: int,  # total input channels
        kh: int,  # kernel height
        kw: int,  # kernel width
        h_in: int,  # input feature map height
        w_in: int,  # input feature map width
        tile_config: Optional[LoopTileConfig] = None,
    ) -> List[Dict]:
        """
        Generate every tile-iteration descriptor for the given loop order.

        Each descriptor records the tile ranges for all 6 dimensions plus
        the derived output-spatial start coordinates.
        """
        if tile_config is None:
            tile_config = LoopTileConfig(
                tile_hin=self.array_height,
                tile_win=self.array_width,
            )

        # Derived output spatial size
        h_out = h_in - kh + 1
        w_out = w_in - kw + 1

        # Build range helpers keyed by dimension name
        def ranges(dim: str):
            if dim == "K":
                return range(0, k, tile_config.tile_k)
            if dim == "C":
                return range(0, c, tile_config.tile_c)
            if dim == "KH":
                return range(kh)
            if dim == "KW":
                return range(kw)
            if dim == "H_in":
                return range(0, h_in, tile_config.tile_hin)
            if dim == "W_in":
                return range(0, w_in, tile_config.tile_win)
            raise ValueError(f"Unknown dimension: {dim}")

        order = loop_order.value  # tuple of 6 dim names

        iterations: List[Dict] = []

        # 6 nested loops driven by the chosen order
        for i0 in ranges(order[0]):
            for i1 in ranges(order[1]):
                for i2 in ranges(order[2]):
                    for i3 in ranges(order[3]):
                        for i4 in ranges(order[4]):
                            for i5 in ranges(order[5]):
                                coords = {
                                    order[0]: i0,
                                    order[1]: i1,
                                    order[2]: i2,
                                    order[3]: i3,
                                    order[4]: i4,
                                    order[5]: i5,
                                }
                                k_s = coords["K"]
                                c_s = coords["C"]
                                kh_i = coords["KH"]
                                kw_i = coords["KW"]
                                h_s = coords["H_in"]
                                w_s = coords["W_in"]

                                iterations.append(
                                    {
                                        # ---- input-channel tile ----
                                        "c_start": c_s,
                                        "c_end": min(c_s + tile_config.tile_c, c),
                                        # ---- output-channel tile ----
                                        "k_start": k_s,
                                        "k_end": min(k_s + tile_config.tile_k, k),
                                        # ---- kernel position ----
                                        "kh": kh_i,
                                        "kw": kw_i,
                                        # ---- input spatial tile ----
                                        "h_in_start": h_s,
                                        "h_in_end": min(
                                            h_s + tile_config.tile_hin, h_in
                                        ),
                                        "w_in_start": w_s,
                                        "w_in_end": min(
                                            w_s + tile_config.tile_win, w_in
                                        ),
                                        # ---- derived output spatial tile ----
                                        # output pixel (h_out, w_out) = input pixel
                                        # (h_in + kh_i, w_in + kw_i) for valid conv
                                        "h_out_start": max(0, h_s - kh_i),
                                        "h_out_end": min(
                                            max(0, h_s - kh_i) + tile_config.tile_hin,
                                            h_out,
                                        ),
                                        "w_out_start": max(0, w_s - kw_i),
                                        "w_out_end": min(
                                            max(0, w_s - kw_i) + tile_config.tile_win,
                                            w_out,
                                        ),
                                    }
                                )

        return iterations

    # ------------------------------------------------------------------
    # Memory-access analysis (analytical DRAM traffic model)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_dram_traffic(
        order: Tuple[str, ...],
        k: int, c: int, kh: int, kw: int, h_in: int, w_in: int,
        tile_config: "LoopTileConfig",
    ) -> Dict:
        """
        Analytical DRAM-traffic model that accounts for loop ordering.

        For a 6-deep tiled loop nest, each data tensor (weights, inputs,
        outputs) has a set of dimensions it depends on.  When a loop
        iterating over a dimension the tensor does NOT depend on is placed
        between (interleaved with) the tensor's own dimensions, every
        iteration of that foreign loop evicts the tensor from scratchpad
        and forces a full reload.

        The number of DRAM loads for a tensor is:

            loads = product_of_tile_counts_for_own_dims
                    × product_of_tile_counts_for_foreign_dims_that_are_interleaved

        A foreign dimension is "interleaved" if it appears between the
        outermost and innermost of the tensor's own dimensions in the
        loop order.

        Example: order K→H_in→C→KH→KW→W_in
          Weight dims = {K, C, KH, KW}
            outermost weight dim: K (pos 0), innermost: KW (pos 4)
            H_in (pos 1) is between them → interleaved
            W_in (pos 5) is below → not interleaved
            weight_reload = n_tiles["H_in"] = ceil(14/4) = 4

          Input dims = {C, KH, KW, H_in, W_in}
            outermost: H_in (pos 1), innermost: W_in (pos 5)
            K (pos 0) is above → not interleaved (loaded before inputs)
            input_reload = 1

          Output dims = {K, H_in, W_in}
            outermost: K (pos 0), innermost: W_in (pos 5)
            C(2),KH(3),KW(4) are between → interleaved
            output_reload = n_tiles["C"] * n_tiles["KH"] * n_tiles["KW"]
        """
        import math as _math

        h_out = max(1, h_in - kh + 1)
        w_out = max(1, w_in - kw + 1)

        # Number of tile-iterations along each dimension
        n_tiles = {
            "K":    max(1, _math.ceil(k / tile_config.tile_k)),
            "C":    max(1, _math.ceil(c / tile_config.tile_c)),
            "KH":   max(1, kh),
            "KW":   max(1, kw),
            "H_in": max(1, _math.ceil(h_in / tile_config.tile_hin)),
            "W_in": max(1, _math.ceil(w_in / tile_config.tile_win)),
        }

        # Tensor dimension dependencies
        weight_dims = {"K", "C", "KH", "KW"}
        input_dims  = {"C", "KH", "KW", "H_in", "W_in"}
        output_dims = {"K", "H_in", "W_in"}

        def _reload_factor(own_dims: set) -> float:
            """
            Compute how many extra times a tensor is reloaded due to
            foreign dimensions interleaved between its own dimensions.
            """
            # Find positions of own dims in the loop order
            own_positions = [i for i, d in enumerate(order) if d in own_dims]
            if not own_positions:
                return 1.0

            outermost = min(own_positions)
            innermost = max(own_positions)

            # Foreign dims that fall strictly between outermost and innermost
            reload = 1.0
            for i in range(outermost + 1, innermost):
                if order[i] not in own_dims:
                    reload *= n_tiles[order[i]]
            return reload

        weight_reload = _reload_factor(weight_dims)
        input_reload  = _reload_factor(input_dims)
        output_reload = _reload_factor(output_dims)

        # Base tensor sizes (total elements, not per-tile)
        weight_size = float(k * c * kh * kw)
        input_size  = float(c * h_in * w_in)
        output_size = float(k * h_out * w_out)

        # DRAM traffic = tensor_size × reload_factor
        # (each reload is a full read of the tensor from DRAM)
        weight_traffic = weight_size * weight_reload
        input_traffic  = input_size  * input_reload
        # Output: partial sums must be read back and written again on
        # each reload (read-modify-write), so each extra reload costs 2×
        output_traffic = output_size * (1.0 + (output_reload - 1.0) * 2.0)

        total_traffic = weight_traffic + input_traffic + output_traffic

        # Reuse factors (higher = better; 1.0 = no reuse)
        weight_reuse = 1.0 / weight_reload if weight_reload > 0 else 1.0
        input_reuse  = 1.0 / input_reload  if input_reload  > 0 else 1.0

        return {
            "loop_order":      order,
            "weight_reuse":    weight_reuse,
            "input_reuse":     input_reuse,
            "output_reuse":    1.0 / output_reload if output_reload > 0 else 1.0,
            "weight_accesses": int(weight_traffic),
            "input_accesses":  int(input_traffic),
            "output_accesses": int(output_traffic),
        }

    def analyze_memory_access_pattern(
        self,
        loop_order: LoopOrder6D,
        k: int,
        c: int,
        kh: int,
        kw: int,
        h_in: int,
        w_in: int,
        tile_config: Optional[LoopTileConfig] = None,
    ) -> Dict:
        """
        Compute DRAM traffic for a given loop order using an analytical
        model that properly accounts for data reuse and eviction.

        Loop ordering controls which data stays resident in the on-chip
        scratchpad (inner loops) vs what gets evicted and must be
        re-fetched from DRAM (outer loops). Different orderings produce
        significantly different off-chip memory traffic.
        """
        if tile_config is None:
            tile_config = LoopTileConfig(
                tile_hin=self.array_height,
                tile_win=self.array_width,
            )

        return self._compute_dram_traffic(
            loop_order.value, k, c, kh, kw, h_in, w_in, tile_config
        )

    # ------------------------------------------------------------------
    # Find optimal order
    # ------------------------------------------------------------------
    def find_optimal_loop_order(
        self,
        k: int,
        c: int,
        kh: int,
        kw: int,
        h_in: int,
        w_in: int,
        tile_config: Optional[LoopTileConfig] = None,
        target: str = "balanced",
        exhaustive: bool = False,
    ) -> Tuple[LoopOrder6D, Dict]:
        """
        Return the best loop order and the full results dict.

        Parameters
        ----------
        target      : 'weight_reuse' | 'input_reuse' | 'output_reuse' | 'balanced'
        exhaustive  : False → search only the 6 named LoopOrder6D presets (fast).
                      True  → search all 720 permutations in ALL_LOOP_ORDERS and
                              return the best as a LoopOrder6D (exact match if one
                              exists, else the nearest named preset).

        Notes
        -----
        The exhaustive search calls analyze_memory_access_pattern() for every
        permutation, which iterates over the full tile sequence each time.
        For large layers with many tiles this can be slow; use a coarse
        tile_config (large tile sizes) to keep it tractable, or run with
        small k/c/h_in/w_in values.
        """
        best_order: Optional[LoopOrder6D] = None
        best_score = -1.0
        all_results: Dict = {}

        def _score(m: Dict) -> float:
            if target == "weight_reuse":
                return m["weight_reuse"]
            elif target == "input_reuse":
                return m["input_reuse"]
            elif target == "output_reuse":
                return m["output_reuse"]
            else:
                return (m["weight_reuse"] + m["input_reuse"] + m["output_reuse"]) / 3.0

        if not exhaustive:
            # ── Heuristic: 6 named presets ──────────────────────────────────
            for order in LoopOrder6D:
                m = self.analyze_memory_access_pattern(
                    order, k, c, kh, kw, h_in, w_in, tile_config
                )
                m["score"] = _score(m)
                all_results[order] = m
                if m["score"] > best_score:
                    best_score = m["score"]
                    best_order = order
        else:
            # ── Exhaustive: all 720 permutations ────────────────────────────
            # Build a lookup from value-tuple → LoopOrder6D for exact matches
            named_lookup: Dict[tuple, LoopOrder6D] = {
                o.value: o for o in LoopOrder6D
            }
            best_perm:   Optional[Tuple] = None

            for perm in ALL_LOOP_ORDERS:
                # Construct a temporary LoopOrder6D-compatible object
                # by patching a sentinel enum value to carry the perm tuple.
                # We re-use the existing named enum when the permutation
                # matches one, otherwise we wrap it in a lightweight object.
                if perm in named_lookup:
                    order_obj = named_lookup[perm]
                else:
                    order_obj = _AnonOrder(perm)

                m = self._analyze_perm(
                    perm, k, c, kh, kw, h_in, w_in, tile_config
                )
                m["score"] = _score(m)
                all_results[perm] = m

                if m["score"] > best_score:
                    best_score = m["score"]
                    best_perm  = perm

            # Map winner back to a LoopOrder6D (exact match or nearest named)
            if best_perm in named_lookup:
                best_order = named_lookup[best_perm]
            else:
                # Pick the named preset with the highest score among named ones
                best_order = max(
                    LoopOrder6D,
                    key=lambda o: all_results.get(o.value, {}).get("score", -1)
                )

        return best_order, all_results

    def _analyze_perm(
        self,
        perm: Tuple[str, ...],
        k: int, c: int, kh: int, kw: int, h_in: int, w_in: int,
        tile_config: Optional["LoopTileConfig"],
    ) -> Dict:
        """
        Like analyze_memory_access_pattern() but accepts a raw permutation
        tuple instead of a LoopOrder6D enum value.  Used internally by the
        exhaustive loop-order search.
        """
        if tile_config is None:
            tile_config = LoopTileConfig(
                tile_hin=self.array_height,
                tile_win=self.array_width,
            )

        return self._compute_dram_traffic(
            perm, k, c, kh, kw, h_in, w_in, tile_config
        )

    # ------------------------------------------------------------------
    # Pretty-print analysis
    # ------------------------------------------------------------------
    def print_optimization_analysis(
        self,
        k: int,
        c: int,
        kh: int,
        kw: int,
        h_in: int,
        w_in: int,
        tile_config: Optional[LoopTileConfig] = None,
    ) -> Dict:
        h_out = h_in - kh + 1
        w_out = w_in - kw + 1

        print("=" * 90)
        print("LOOP ORDER OPTIMISATION ANALYSIS")
        print("=" * 90)
        print(f"\nDimensions:")
        print(f"  K    (output channels)      : {k}")
        print(f"  C    (input channels)       : {c}")
        print(f"  KH   (kernel height)        : {kh}")
        print(f"  KW   (kernel width)         : {kw}")
        print(f"  H_in (input feature height) : {h_in}")
        print(f"  W_in (input feature width)  : {w_in}")
        print(f"  H_out (derived)             : {h_out}")
        print(f"  W_out (derived)             : {w_out}")

        if tile_config:
            print(f"\nTile sizes:")
            print(
                f"  tile_k={tile_config.tile_k}  tile_c={tile_config.tile_c}"
                f"  tile_hin={tile_config.tile_hin}  tile_win={tile_config.tile_win}"
            )

        print("\n" + "-" * 90)
        print(
            f"{'Loop Order':<35} {'Weight':>10} {'Input':>10} {'Output':>10} {'Balanced':>10}"
        )
        print(f"{'':35} {'Reuse':>10} {'Reuse':>10} {'Reuse':>10} {'Score':>10}")
        print("-" * 90)

        results: Dict[LoopOrder6D, Dict] = {}
        for order in LoopOrder6D:
            m = self.analyze_memory_access_pattern(
                order, k, c, kh, kw, h_in, w_in, tile_config
            )
            balanced = (m["weight_reuse"] + m["input_reuse"] + m["output_reuse"]) / 3.0
            results[order] = m
            label = "→".join(order.value)
            print(
                f"{label:<35} {m['weight_reuse']:>10.2f} {m['input_reuse']:>10.2f}"
                f" {m['output_reuse']:>10.2f} {balanced:>10.2f}"
            )

        print("\n" + "=" * 90)
        print("RECOMMENDATIONS")
        print("=" * 90)

        def best(key):
            return max(results.items(), key=lambda x: x[1][key])

        for label, key in [
            ("Best weight reuse", "weight_reuse"),
            ("Best input reuse", "input_reuse"),
            ("Best output reuse", "output_reuse"),
        ]:
            order, m = best(key)
            print(f"\n{label}: {'→'.join(order.value)}")
            print(f"  {key} = {m[key]:.2f}x")

        order, m = max(
            results.items(),
            key=lambda x: (
                x[1]["weight_reuse"] + x[1]["input_reuse"] + x[1]["output_reuse"]
            )
            / 3,
        )
        balanced = (m["weight_reuse"] + m["input_reuse"] + m["output_reuse"]) / 3.0
        print(f"\nBest balanced: {'→'.join(order.value)}")
        print(f"  balanced score = {balanced:.2f}")

        return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def demo():
    print("\n" + "=" * 90)
    print("DEMO: Loop Order Optimisation — Output-Stationary Systolic Array")
    print("=" * 90)

    # A typical early convolutional layer
    k = 16  # output channels
    c = 8  # input channels
    kh = 3  # kernel height
    kw = 3  # kernel width
    h_in = 30  # input height  →  h_out = 30 - 3 + 1 = 28
    w_in = 30  # input width   →  w_out = 28

    optimizer = LoopOptimizer(array_height=4, array_width=4)
    tile_cfg = LoopTileConfig(tile_k=4, tile_c=4, tile_hin=4, tile_win=4)

    optimizer.print_optimization_analysis(k, c, kh, kw, h_in, w_in, tile_cfg)

    # Show first few iterations for one order
    print("\n" + "=" * 90)
    print("EXAMPLE: First 5 tile-iterations for K→C→KH→KW→H_in→W_in")
    print("=" * 90)

    iters = optimizer.generate_iteration_sequence(
        LoopOrder6D.K_C_KH_KW_HIN_WIN,
        k=4,
        c=2,
        kh=2,
        kw=2,
        h_in=6,
        w_in=6,
        tile_config=LoopTileConfig(tile_k=2, tile_c=2, tile_hin=3, tile_win=3),
    )

    print(f"\nTotal iterations: {len(iters)}\n")
    print(
        f"{'#':<4} {'K':>6} {'C':>6} {'KH':>4} {'KW':>4} "
        f"{'H_in':>10} {'W_in':>10} {'H_out':>10} {'W_out':>10}"
    )
    print("-" * 70)
    for i, it in enumerate(iters[:5]):
        print(
            f"{i:<4} "
            f"{it['k_start']}-{it['k_end']-1:>3}  "
            f"{it['c_start']}-{it['c_end']-1:>3}  "
            f"{it['kh']:>3}  {it['kw']:>3}  "
            f"{it['h_in_start']}-{it['h_in_end']-1:>6}  "
            f"{it['w_in_start']}-{it['w_in_end']-1:>6}  "
            f"{it['h_out_start']}-{it['h_out_end']-1:>6}  "
            f"{it['w_out_start']}-{it['w_out_end']-1:>6}"
        )
    if len(iters) > 5:
        print(f"  ... ({len(iters) - 5} more)")


if __name__ == "__main__":
    demo()
