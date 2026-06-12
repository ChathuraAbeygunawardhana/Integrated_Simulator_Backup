#!/usr/bin/env python3
"""
Loop Order Optimization Examples
=================================
Demonstrates how to use loop order optimization for different scenarios.
"""

import numpy as np
from loop_optimizer import LoopOrder6D, LoopOptimizer, LoopTileConfig


def example1_basic_analysis():
    """Basic loop order analysis for a single layer."""

    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Loop Order Analysis")
    print("=" * 80)

    # Small conv layer
    optimizer = LoopOptimizer(array_height=4, array_width=4)

    optimizer.print_optimization_analysis(
        k_out=16,
        c_in=8,
        kh=3,
        kw=3,
        h_out=28,
        w_out=28,
        tile_config=LoopTileConfig(tile_k=4, tile_c=4),
    )


def example2_compare_layers():
    """Compare optimal loop orders for different layer types."""

    print("\n" + "=" * 80)
    print("EXAMPLE 2: Loop Orders for Different Layer Types")
    print("=" * 80)

    optimizer = LoopOptimizer(array_height=4, array_width=4)

    layers = [
        {
            "name": "Early Conv (3x3, 1→8 channels)",
            "k_out": 8,
            "c_in": 1,
            "kh": 3,
            "kw": 3,
            "h_out": 28,
            "w_out": 28,
        },
        {
            "name": "Middle Conv (3x3, 16→32 channels)",
            "k_out": 32,
            "c_in": 16,
            "kh": 3,
            "kw": 3,
            "h_out": 14,
            "w_out": 14,
        },
        {
            "name": "Deep Conv (3x3, 128→128 channels)",
            "k_out": 128,
            "c_in": 128,
            "kh": 3,
            "kw": 3,
            "h_out": 7,
            "w_out": 7,
        },
        {
            "name": "1x1 Conv (64→128 channels)",
            "k_out": 128,
            "c_in": 64,
            "kh": 1,
            "kw": 1,
            "h_out": 14,
            "w_out": 14,
        },
    ]

    print(f"\n{'Layer Type':<40} {'Best Order':<20} {'Score':<10}")
    print("-" * 80)

    for layer in layers:
        best_order, all_results = optimizer.find_optimal_loop_order(
            k=layer["k_out"],
            c=layer["c_in"],
            kh=layer["kh"],
            kw=layer["kw"],
            h_in=layer["h_out"],
            w_in=layer["w_out"],
            tile_config=LoopTileConfig(tile_k=4, tile_c=4),
            target="balanced",
        )

        score = all_results[best_order]["score"]
        print(f"{layer['name']:<40} {best_order.value:<20} {score:<10.2f}")


def example3_optimization_targets():
    """Compare different optimization targets for the same layer."""

    print("\n" + "=" * 80)
    print("EXAMPLE 3: Different Optimization Targets")
    print("=" * 80)

    optimizer = LoopOptimizer(array_height=4, array_width=4)

    # Example layer
    k_out, c_in = 32, 16
    kh, kw = 3, 3
    h_out, w_out = 14, 14

    print(f"\nLayer: {kh}x{kw} Conv, {c_in}→{k_out} channels, {h_out}x{w_out} output")

    targets = ["weight_reuse", "input_reuse", "output_reuse", "balanced"]

    print(f"\n{'Optimization Target':<25} {'Best Loop Order':<20} {'Score':<10}")
    print("-" * 80)

    for target in targets:
        best_order, all_results = optimizer.find_optimal_loop_order(
            k=k_out,
            c=c_in,
            kh=kh,
            kw=kw,
            h_in=h_out,
            w_in=w_out,
            tile_config=LoopTileConfig(tile_k=4, tile_c=4),
            target=target,
        )

        score = all_results[best_order]["score"]
        print(f"{target:<25} {best_order.value:<20} {score:<10.2f}")


def example4_tiling_impact():
    """Show impact of different tiling strategies."""

    print("\n" + "=" * 80)
    print("EXAMPLE 4: Impact of Tiling on Loop Order Choice")
    print("=" * 80)

    optimizer = LoopOptimizer(array_height=4, array_width=4)

    # Large layer that requires tiling
    k_out, c_in = 64, 64
    kh, kw = 3, 3
    h_out, w_out = 28, 28

    print(f"\nLayer: {kh}x{kw} Conv, {c_in}→{k_out} channels, {h_out}x{w_out} output")

    tiling_configs = [
        {"name": "Small tiles (2x2)", "tile_k": 2, "tile_c": 2},
        {"name": "Medium tiles (4x4)", "tile_k": 4, "tile_c": 4},
        {"name": "Large tiles (8x8)", "tile_k": 8, "tile_c": 8},
        {"name": "Asymmetric (16x4)", "tile_k": 16, "tile_c": 4},
    ]

    print(f"\n{'Tiling Strategy':<25} {'Best Loop Order':<20} {'Score':<10}")
    print("-" * 80)

    for config in tiling_configs:
        tile_config = LoopTileConfig(tile_k=config["tile_k"], tile_c=config["tile_c"])

        best_order, all_results = optimizer.find_optimal_loop_order(
            k=k_out,
            c=c_in,
            kh=kh,
            kw=kw,
            h_in=h_out,
            w_in=w_out,
            tile_config=tile_config,
            target="balanced",
        )

        score = all_results[best_order]["score"]
        print(f"{config['name']:<25} {best_order.value:<20} {score:<10.2f}")


def example5_iteration_sequences():
    """Show actual iteration sequences for different loop orders."""

    print("\n" + "=" * 80)
    print("EXAMPLE 5: Iteration Sequences for Different Loop Orders")
    print("=" * 80)

    optimizer = LoopOptimizer(array_height=4, array_width=4)

    # Small example for clarity
    k_out, c_in = 4, 2
    kh, kw = 2, 2
    h_out, w_out = 4, 4

    print(
        f"\nSmall layer: {kh}x{kw} Conv, {c_in}→{k_out} channels, {h_out}x{w_out} output"
    )
    print(f"Tile size: 2x2 output")

    tile_config = LoopTileConfig(tile_hin=2, tile_win=2)

    # Show iteration count for each order
    orders_to_show = [
        LoopOrder6D.K_C_KH_KW_HIN_WIN,
        LoopOrder6D.C_K_KH_KW_HIN_WIN,
        LoopOrder6D.KH_KW_K_C_HIN_KW,
    ]

    for loop_order in orders_to_show:
        iterations = optimizer.generate_iteration_sequence(
            loop_order, k_out, c_in, kh, kw, h_out, w_out, tile_config
        )

        print(f"\n{'-'*80}")
        print(f"Loop Order: {loop_order.name}")
        print(f"Total iterations: {len(iterations)}")
        print(f"\nFirst 5 iterations:")
        print(f"{'#':<4} {'K':<8} {'C':<8} {'KH,KW':<10} {'H_out':<12} {'W_out':<12}")
        print("-" * 80)

        for i, it in enumerate(iterations[:5]):
            k_range = f"{it['k_start']}-{it['k_end']-1}"
            c_range = f"{it['c_start']}-{it['c_end']-1}"
            kh_kw = f"{it['kh']},{it['kw']}"
            h_range = f"{it['h_start']}-{it['h_end']-1}"
            w_range = f"{it['w_start']}-{it['w_end']-1}"

            print(
                f"{i:<4} {k_range:<8} {c_range:<8} {kh_kw:<10} "
                f"{h_range:<12} {w_range:<12}"
            )


def example6_real_world_models():
    """Analyze loop orders for real-world model architectures."""

    print("\n" + "=" * 80)
    print("EXAMPLE 6: Loop Orders for Real-World Model Architectures")
    print("=" * 80)

    optimizer = LoopOptimizer(array_height=4, array_width=4)

    # Representative layers from popular models
    models = {
        "VGG-like": [
            {
                "name": "Block1_Conv1",
                "k": 64,
                "c": 3,
                "kh": 3,
                "kw": 3,
                "h": 224,
                "w": 224,
            },
            {
                "name": "Block2_Conv1",
                "k": 128,
                "c": 64,
                "kh": 3,
                "kw": 3,
                "h": 112,
                "w": 112,
            },
            {
                "name": "Block3_Conv1",
                "k": 256,
                "c": 128,
                "kh": 3,
                "kw": 3,
                "h": 56,
                "w": 56,
            },
        ],
        "ResNet-like": [
            {"name": "Conv1", "k": 64, "c": 3, "kh": 7, "kw": 7, "h": 112, "w": 112},
            {
                "name": "Bottleneck_1x1",
                "k": 64,
                "c": 256,
                "kh": 1,
                "kw": 1,
                "h": 56,
                "w": 56,
            },
            {
                "name": "Bottleneck_3x3",
                "k": 64,
                "c": 64,
                "kh": 3,
                "kw": 3,
                "h": 56,
                "w": 56,
            },
        ],
        "MobileNet-like": [
            {
                "name": "Depthwise_3x3",
                "k": 32,
                "c": 1,
                "kh": 3,
                "kw": 3,
                "h": 112,
                "w": 112,
            },
            {
                "name": "Pointwise_1x1",
                "k": 64,
                "c": 32,
                "kh": 1,
                "kw": 1,
                "h": 112,
                "w": 112,
            },
        ],
    }

    for model_name, layers in models.items():
        print(f"\n{model_name} Architecture:")
        print(f"{'Layer':<25} {'Config':<35} {'Best Order':<20}")
        print("-" * 80)

        for layer in layers:
            config_str = f"{layer['kh']}x{layer['kw']}, {layer['c']}→{layer['k']}, {layer['h']}x{layer['w']}"

            best_order, _ = optimizer.find_optimal_loop_order(
                k=layer["k"],
                c=layer["c"],
                kh=layer["kh"],
                kw=layer["kw"],
                h_in=layer["h"],
                w_in=layer["w"],
                tile_config=LoopTileConfig(tile_k=4, tile_c=4),
                target="balanced",
            )

            print(f"{layer['name']:<25} {config_str:<35} {best_order.value:<20}")


def main():
    """Run all examples."""

    print("\n" + "=" * 80)
    print("LOOP ORDER OPTIMIZATION EXAMPLES")
    print("=" * 80)

    example1_basic_analysis()
    example2_compare_layers()
    example3_optimization_targets()
    example4_tiling_impact()
    example5_iteration_sequences()
    example6_real_world_models()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Different layer types benefit from different loop orders")
    print("2. Optimization target affects the best choice")
    print("3. Tiling strategy impacts memory reuse patterns")
    print(
        "4. Early layers (few channels) vs deep layers (many channels) need different orders"
    )
    print("5. 1x1 convolutions have different characteristics than 3x3")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
