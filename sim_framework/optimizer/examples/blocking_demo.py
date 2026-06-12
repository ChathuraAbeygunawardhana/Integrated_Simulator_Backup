#!/usr/bin/env python3
"""
Updated Complete System Demo - Blocking-First Approach
======================================================
Demonstrates the CORRECT optimization approach:
1. Block loops FIRST (determine tile sizes)
2. Reorder INTER-TILE loops
3. Search for optimal (blocking, loop_order) pair
"""

from metadata_table import *
from inter_layer_reuse import *
from blocking_optimizer import BlockingOptimizer, TileLoopOrder
import json
import os


def create_example_network():
    """Create example CNN metadata."""
    
    metadata = MetadataTable(scratchpad_size_kb=64)
    
    # 3-layer CNN
    layers = [
        {
            'id': 0, 'name': 'conv1',
            'input': (28, 28, 1), 'output': (28, 28, 16), 'kernel': (3, 3, 1, 16)
        },
        {
            'id': 1, 'name': 'conv2',
            'input': (28, 28, 16), 'output': (14, 14, 32), 'kernel': (3, 3, 16, 32)
        },
        {
            'id': 2, 'name': 'conv3',
            'input': (14, 14, 32), 'output': (7, 7, 64), 'kernel': (3, 3, 32, 64)
        },
    ]
    
    # Add input tensor
    metadata.add_tensor(TensorMetadata(
        name='network_input',
        shape=(28, 28, 1),
        is_network_input=True
    ))
    
    # Add layers
    for i, config in enumerate(layers):
        input_tensor = 'network_input' if i == 0 else f"tensor_{i-1}"
        output_tensor = f"tensor_{i}"
        
        h_in, w_in, c_in = config['input']
        h_out, w_out, k_out = config['output']
        kh, kw, c_k, k_k = config['kernel']
        
        layer = LayerMetadata(
            layer_id=config['id'],
            layer_name=config['name'],
            layer_type=LayerType.CONV2D,
            input_tensors=[input_tensor],
            output_tensor=output_tensor,
            weight_tensor=f"weights_{i}",
            input_shape=config['input'],
            output_shape=config['output'],
            kernel_shape=config['kernel'],
            depends_on=[] if i == 0 else [i-1],
            input_size_bytes=h_in * w_in * c_in * 4,
            output_size_bytes=h_out * w_out * k_out * 4,
            weight_size_bytes=kh * kw * c_k * k_k * 4
        )
        
        metadata.add_layer(layer)
        
        # Add output tensor
        metadata.add_tensor(TensorMetadata(
            name=output_tensor,
            shape=config['output'],
            producer_layer=config['name'],
            consumer_layers=[layers[i+1]['name']] if i < len(layers)-1 else [],
            is_network_output=(i == len(layers)-1)
        ))
    
    return metadata


def main():
    """Run complete demonstration."""
    
    print("\n" + "="*90)
    print("COMPLETE SYSTEM WITH BLOCKING-FIRST OPTIMIZATION")
    print("="*90)
    
    # Step 1: Build metadata
    print("\n[STEP 1] Building Network Metadata")
    print("-" * 90)
    
    metadata = create_example_network()
    print(f"Created {len(metadata.layers)} layers")
    
    # Step 2: Compute execution order
    print("\n[STEP 2] Computing Execution Order")
    print("-" * 90)
    
    metadata.compute_execution_order()
    print(f"Execution order: {' → '.join(str(lid) for lid in metadata.execution_order)}")
    
    # Step 3: BLOCKING-FIRST OPTIMIZATION
    print("\n[STEP 3] Blocking-First Optimization (Search-Based)")
    print("-" * 90)
    print("\nThis is the CORRECT approach:")
    print("  1. Block loops first → determine tile sizes")
    print("  2. Apply loop reordering to INTER-TILE loops")
    print("  3. Search for optimal (blocking, loop_order) pair")
    
    metadata.optimize_blocking_factors(
        array_height=4,
        array_width=4,
        scratchpad_kb=64,
        optimization_objective='min_traffic'
    )
    
    # Step 4: Analyze inter-layer reuse
    print("\n[STEP 4] Analyzing Inter-Layer Reuse")
    print("-" * 90)
    
    metadata.analyze_inter_layer_reuse()
    
    # Step 5: Generate execution schedule
    print("\n[STEP 5] Generating Execution Schedule")
    print("-" * 90)
    
    scheduler = InterLayerReuseScheduler(
        metadata,
        scratchpad_size_kb=64,
        array_height=4,
        array_width=4
    )
    
    chains = scheduler.find_reuse_chains()
    blocks = scheduler.generate_execution_schedule()
    
    print(f"Generated {len(blocks)} execution blocks")
    
    # Step 6: Estimate memory traffic
    print("\n[STEP 6] Memory Traffic Analysis")
    print("-" * 90)
    
    traffic = scheduler.estimate_memory_traffic_with_reuse()
    
    print(f"\nDRAM Traffic Breakdown:")
    print(f"  Input reads:      {traffic['input_reads']/1e6:.3f} MB")
    print(f"  Weight reads:     {traffic['weight_reads']/1e6:.3f} MB")
    print(f"  Output writes:    {traffic['output_writes']/1e6:.3f} MB")
    print(f"  ────────────────────────────────────")
    print(f"  Total:            {traffic['total']/1e6:.3f} MB")
    print(f"  Saved (on-chip):  {traffic['intermediate_saved']/1e6:.3f} MB")
    
    if traffic['intermediate_saved'] > 0:
        total_without = traffic['total'] + traffic['intermediate_saved']
        reduction = (traffic['intermediate_saved'] / total_without) * 100
        print(f"  Traffic reduction: {reduction:.1f}%")
    
    # Step 7: Export configurations
    print("\n[STEP 7] Exporting Configurations")
    print("-" * 90)
    
    os.makedirs('./demo_output', exist_ok=True)
    
    # Export full metadata
    metadata.to_json('./demo_output/blocking_first_metadata.json')
    print("  ✓ Saved: ./demo_output/blocking_first_metadata.json")
    
    # Export runtime config
    runtime_config = {
        'system': {
            'array_height': 4,
            'array_width': 4,
            'scratchpad_kb': 64,
            'optimization_approach': 'blocking_first'
        },
        'layers': []
    }
    
    for lid in metadata.execution_order:
        layer = metadata.layers[lid]
        
        if layer.loop_nest_config:
            bf = layer.loop_nest_config.blocking
            
            layer_cfg = {
                'id': lid,
                'name': layer.layer_name,
                'dimensions': {
                    'K': layer.output_shape[2],
                    'C': layer.input_shape[2],
                    'H': layer.output_shape[0],
                    'W': layer.output_shape[1],
                    'KH': layer.kernel_shape[0],
                    'KW': layer.kernel_shape[1]
                },
                'blocking': {
                    'tile_K': bf.tile_K,
                    'tile_C': bf.tile_C,
                    'tile_H': bf.tile_H,
                    'tile_W': bf.tile_W,
                    'inner_h': bf.inner_h,
                    'inner_w': bf.inner_w
                },
                'tile_loop_order': {
                    'name': layer.loop_nest_config.tile_loop_order.name,
                    'code': list(TileLoopOrder).index(layer.loop_nest_config.tile_loop_order),
                    'description': layer.loop_nest_config.tile_loop_order.value
                },
                'memory': {
                    'scratchpad_requirement_kb': layer.scratchpad_requirement / 1024,
                    'input_on_chip': layer.input_can_stay_on_chip,
                    'output_on_chip': layer.output_can_stay_on_chip
                }
            }
            
            runtime_config['layers'].append(layer_cfg)
    
    with open('./demo_output/runtime_config_blocking_first.json', 'w') as f:
        json.dump(runtime_config, f, indent=2)
    
    print("  ✓ Saved: ./demo_output/runtime_config_blocking_first.json")
    
    # Step 8: Print summary
    print("\n[STEP 8] Final Summary")
    print("=" * 90)
    
    metadata.print_summary()
    scheduler.print_schedule()
    
    # Step 9: Show loop nest structure
    print("\n[STEP 9] Loop Nest Structure for Each Layer")
    print("=" * 90)
    
    for lid in metadata.execution_order:
        layer = metadata.layers[lid]
        
        if not layer.loop_nest_config:
            continue
        
        bf = layer.loop_nest_config.blocking
        order = layer.loop_nest_config.tile_loop_order
        
        print(f"\nLayer {lid}: {layer.layer_name}")
        print(f"  Dimensions: K={layer.output_shape[2]}, C={layer.input_shape[2]}, "
              f"H={layer.output_shape[0]}, W={layer.output_shape[1]}")
        
        print(f"\n  Loop Nest Structure:")
        print(f"  ──────────────────────────────────────────────────────────")
        
        # Show the actual loop nest based on order
        order_str = order.value  # e.g., "k_tile, c_tile, kh, kw"
        loops = [x.strip() for x in order_str.split(',')]
        
        indent = "  "
        for i, loop_var in enumerate(loops):
            spaces = "  " * i
            
            if 'k_tile' in loop_var:
                print(f"{indent}{spaces}for k_tile in range(0, {layer.output_shape[2]}, {bf.tile_K}):  # INTER-TILE")
            elif 'c_tile' in loop_var:
                print(f"{indent}{spaces}for c_tile in range(0, {layer.input_shape[2]}, {bf.tile_C}):  # INTER-TILE")
            elif 'kh' in loop_var:
                print(f"{indent}{spaces}for kh in range({layer.kernel_shape[0]}):                     # SPATIAL")
            elif 'kw' in loop_var:
                print(f"{indent}{spaces}for kw in range({layer.kernel_shape[1]}):                     # SPATIAL")
        
        # Inner loops (fixed - mapped to array)
        inner_indent = "  " * len(loops)
        print(f"{indent}{inner_indent}for k in range(k_tile, k_tile+{bf.tile_K}):        # INTRA-TILE")
        print(f"{indent}{inner_indent}  for c in range(c_tile, c_tile+{bf.tile_C}):      # INTRA-TILE")
        print(f"{indent}{inner_indent}    for h in range(0, {layer.output_shape[0]}, {bf.inner_h}):  # ARRAY ROW")
        print(f"{indent}{inner_indent}      for w in range(0, {layer.output_shape[1]}, {bf.inner_w}):  # ARRAY COL")
        print(f"{indent}{inner_indent}        PE[h][w] accumulate")
    
    print("\n" + "=" * 90)
    print("DEMO COMPLETE!")
    print("=" * 90)
    
    print("\nKey Achievements:")
    print("  ✓ Blocking-first optimization (CORRECT approach)")
    print("  ✓ Loop reordering on INTER-TILE loops")
    print("  ✓ Search-based optimization for each layer")
    print("  ✓ Joint optimization of (blocking, loop_order)")
    print("  ✓ Inter-layer data reuse")
    print("  ✓ Complete runtime configuration")
    
    print("\nGenerated Files:")
    print("  • blocking_first_metadata.json          - Complete metadata")
    print("  • runtime_config_blocking_first.json    - Hardware config")
    
    print("\n" + "=" * 90 + "\n")


if __name__ == '__main__':
    main()
