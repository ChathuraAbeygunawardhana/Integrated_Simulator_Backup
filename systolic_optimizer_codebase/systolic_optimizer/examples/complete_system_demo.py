#!/usr/bin/env python3
"""
Complete System Demo - No TensorFlow Required
==============================================
Demonstrates the complete metadata-driven optimization system.
"""

from metadata_table import *
from inter_layer_reuse import *
from loop_optimizer import LoopOrder, LoopOptimizer
import json
import os


def create_example_cnn_metadata():
    """Create metadata for an example CNN."""
    
    metadata = MetadataTable(scratchpad_size_kb=64)
    
    # Define a simple 3-layer CNN
    layers_config = [
        {
            'id': 0, 'name': 'conv1', 'type': LayerType.CONV2D,
            'input': (28, 28, 1), 'output': (28, 28, 16), 'kernel': (3, 3, 1, 16),
            'depends_on': []
        },
        {
            'id': 1, 'name': 'conv2', 'type': LayerType.CONV2D,
            'input': (28, 28, 16), 'output': (28, 28, 32), 'kernel': (3, 3, 16, 32),
            'depends_on': [0]
        },
        {
            'id': 2, 'name': 'conv3', 'type': LayerType.CONV2D,
            'input': (28, 28, 32), 'output': (28, 28, 64), 'kernel': (3, 3, 32, 64),
            'depends_on': [1]
        },
    ]
    
    # Add input tensor
    metadata.add_tensor(TensorMetadata(
        name='network_input',
        shape=(28, 28, 1),
        is_network_input=True
    ))
    
    # Add layers and tensors
    for i, config in enumerate(layers_config):
        # Input tensor name
        if i == 0:
            input_tensor = 'network_input'
        else:
            input_tensor = f"tensor_{i-1}"
        
        output_tensor = f"tensor_{i}"
        weight_tensor = f"weights_{i}"
        
        h_in, w_in, c_in = config['input']
        h_out, w_out, k_out = config['output']
        kh, kw, c_in_k, k_out_k = config['kernel']
        
        # Create layer metadata
        layer = LayerMetadata(
            layer_id=config['id'],
            layer_name=config['name'],
            layer_type=config['type'],
            input_tensors=[input_tensor],
            output_tensor=output_tensor,
            weight_tensor=weight_tensor,
            input_shape=config['input'],
            output_shape=config['output'],
            kernel_shape=config['kernel'],
            depends_on=config['depends_on'],
            input_size_bytes=h_in * w_in * c_in * 4,
            output_size_bytes=h_out * w_out * k_out * 4,
            weight_size_bytes=kh * kw * c_in_k * k_out_k * 4
        )
        
        metadata.add_layer(layer)
        
        # Add output tensor
        is_last = (i == len(layers_config) - 1)
        metadata.add_tensor(TensorMetadata(
            name=output_tensor,
            shape=config['output'],
            producer_layer=config['name'],
            consumer_layers=[layers_config[i+1]['name']] if not is_last else [],
            is_network_output=is_last
        ))
    
    return metadata


def main():
    """Run complete system demonstration."""
    
    print("\n" + "="*90)
    print("COMPLETE METADATA-DRIVEN OPTIMIZATION SYSTEM")
    print("="*90)
    
    # Step 1: Create metadata
    print("\n" + "="*90)
    print("STEP 1: Building Network Metadata")
    print("="*90)
    
    metadata = create_example_cnn_metadata()
    print(f"\nCreated metadata for {len(metadata.layers)} layers")
    
    # Step 2: Compute execution order
    print("\n" + "="*90)
    print("STEP 2: Computing Execution Order")
    print("="*90)
    
    metadata.compute_execution_order()
    print(f"\nExecution order: {' → '.join(str(lid) for lid in metadata.execution_order)}")
    
    # Step 3: Optimize loop orders
    print("\n" + "="*90)
    print("STEP 3: Optimizing Loop Orders")
    print("="*90)
    
    optimizer = LoopOptimizer(array_height=4, array_width=4)
    
    for lid in metadata.execution_order:
        layer = metadata.layers[lid]
        
        # Find optimal loop order
        best_order, all_results = optimizer.find_optimal_loop_order(
            k_out=layer.output_shape[2],
            c_in=layer.input_shape[2],
            kh=layer.kernel_shape[0],
            kw=layer.kernel_shape[1],
            h_out=layer.output_shape[0],
            w_out=layer.output_shape[1],
            tile_config=LoopTileConfig(
                tile_k=min(4, layer.output_shape[2]),
                tile_c=min(4, layer.input_shape[2])
            ),
            optimization_target='balanced'
        )
        
        layer.loop_order = best_order.value
        layer.loop_order_code = list(LoopOrder).index(best_order)
        
        print(f"\n  {layer.layer_name}:")
        print(f"    Input:  {layer.input_shape}")
        print(f"    Output: {layer.output_shape}")
        print(f"    Kernel: {layer.kernel_shape[:2]}")
        print(f"    Loop order: {best_order.value}")
    
    # Step 4: Optimize blocking
    print("\n" + "="*90)
    print("STEP 4: Optimizing Loop Blocking")
    print("="*90)
    
    metadata.optimize_blocking_factors(array_height=4, array_width=4)
    
    for lid in metadata.execution_order:
        layer = metadata.layers[lid]
        bf = layer.blocking_factors
        
        print(f"\n  {layer.layer_name}:")
        print(f"    L1 tiles: {bf.tile_h_out}×{bf.tile_w_out} (output), "
              f"{bf.tile_k_out} (K), {bf.tile_c_in} (C)")
        print(f"    L2 tiles: {bf.l2_tile_h}×{bf.l2_tile_w} (spatial), "
              f"{bf.l2_tile_k} (K), {bf.l2_tile_c} (C)")
        print(f"    Scratchpad need: {layer.scratchpad_requirement/1024:.1f} KB")
    
    # Step 5: Find fusion opportunities
    print("\n" + "="*90)
    print("STEP 5: Finding Fusion Opportunities")
    print("="*90)
    
    metadata.find_fusion_opportunities()
    
    fusion_count = sum(1 for l in metadata.layers.values() if l.can_fuse_with_next)
    print(f"\nFound {fusion_count} fusion opportunities")
    
    # Step 6: Analyze inter-layer reuse
    print("\n" + "="*90)
    print("STEP 6: Analyzing Inter-Layer Data Reuse")
    print("="*90)
    
    metadata.analyze_inter_layer_reuse()
    
    # Step 7: Generate execution schedule
    print("\n" + "="*90)
    print("STEP 7: Generating Execution Schedule")
    print("="*90)
    
    scheduler = InterLayerReuseScheduler(
        metadata,
        scratchpad_size_kb=64,
        array_height=4,
        array_width=4
    )
    
    chains = scheduler.find_reuse_chains()
    blocks = scheduler.generate_execution_schedule()
    
    print(f"\nGenerated {len(blocks)} execution blocks")
    
    # Step 8: Estimate memory traffic
    print("\n" + "="*90)
    print("STEP 8: Estimating Memory Traffic")
    print("="*90)
    
    traffic = metadata.estimate_memory_traffic()
    print(f"\nTotal memory traffic: {traffic/(1024*1024):.2f} MB")
    
    traffic_detailed = scheduler.estimate_memory_traffic_with_reuse()
    print(f"\nDetailed breakdown:")
    print(f"  Input reads:     {traffic_detailed['input_reads']/(1024*1024):.2f} MB")
    print(f"  Weight reads:    {traffic_detailed['weight_reads']/(1024*1024):.2f} MB")
    print(f"  Output writes:   {traffic_detailed['output_writes']/(1024*1024):.2f} MB")
    print(f"  Saved (on-chip): {traffic_detailed['intermediate_saved']/(1024*1024):.2f} MB")
    
    if traffic_detailed['intermediate_saved'] > 0:
        total_without_opt = traffic_detailed['total'] + traffic_detailed['intermediate_saved']
        reduction = (traffic_detailed['intermediate_saved'] / total_without_opt) * 100
        print(f"  Traffic reduction: {reduction:.1f}%")
    
    # Step 9: Export metadata
    print("\n" + "="*90)
    print("STEP 9: Exporting Metadata and Configs")
    print("="*90)
    
    os.makedirs('./demo_output', exist_ok=True)
    
    # Export full metadata
    metadata.to_json('./demo_output/complete_metadata.json')
    print("\n  ✓ Saved: ./demo_output/complete_metadata.json")
    
    # Export execution schedule
    schedule_data = {
        'network': {
            'layers': len(metadata.layers),
            'execution_order': metadata.execution_order,
            'scratchpad_size_kb': 64
        },
        'reuse_chains': [
            {
                'chain_id': i,
                'layers': [metadata.layers[lid].layer_name for lid in chain],
                'layer_ids': chain
            }
            for i, chain in enumerate(chains)
        ],
        'execution_blocks': [
            {
                'block_id': block.block_id,
                'layers': [metadata.layers[lid].layer_name for lid in block.layer_ids],
                'layer_ids': block.layer_ids,
                'tile_h': [block.h_start, block.h_end],
                'tile_w': [block.w_start, block.w_end],
                'tile_k': [block.k_start, block.k_end]
            }
            for block in blocks[:20]  # First 20 blocks
        ]
    }
    
    with open('./demo_output/execution_schedule.json', 'w') as f:
        json.dump(schedule_data, f, indent=2)
    
    print("  ✓ Saved: ./demo_output/execution_schedule.json")
    
    # Export runtime config for hardware
    runtime_config = {
        'array': {
            'height': 4,
            'width': 4,
            'clock_mhz': 100,
            'scratchpad_kb': 64
        },
        'layers': []
    }
    
    for lid in metadata.execution_order:
        layer = metadata.layers[lid]
        bf = layer.blocking_factors
        
        layer_config = {
            'id': lid,
            'name': layer.layer_name,
            'type': layer.layer_type.value,
            'dimensions': {
                'input_h': layer.input_shape[0],
                'input_w': layer.input_shape[1],
                'input_c': layer.input_shape[2],
                'output_h': layer.output_shape[0],
                'output_w': layer.output_shape[1],
                'output_c': layer.output_shape[2],
                'kernel_h': layer.kernel_shape[0],
                'kernel_w': layer.kernel_shape[1]
            },
            'optimization': {
                'loop_order': layer.loop_order,
                'loop_order_code': layer.loop_order_code,
                'l1_blocking': {
                    'h_out': bf.tile_h_out,
                    'w_out': bf.tile_w_out,
                    'k_out': bf.tile_k_out,
                    'c_in': bf.tile_c_in
                },
                'l2_blocking': {
                    'h': bf.l2_tile_h,
                    'w': bf.l2_tile_w,
                    'k': bf.l2_tile_k,
                    'c': bf.l2_tile_c
                }
            },
            'inter_layer_reuse': {
                'input_on_chip': layer.input_can_stay_on_chip,
                'output_on_chip': layer.output_can_stay_on_chip
            }
        }
        
        runtime_config['layers'].append(layer_config)
    
    with open('./demo_output/runtime_config.json', 'w') as f:
        json.dump(runtime_config, f, indent=2)
    
    print("  ✓ Saved: ./demo_output/runtime_config.json")
    
    # Step 10: Print final summary
    print("\n" + "="*90)
    print("FINAL SUMMARY")
    print("="*90)
    
    metadata.print_summary()
    scheduler.print_schedule()
    
    print("\n" + "="*90)
    print("DEMO COMPLETE!")
    print("="*90)
    
    print("\nKey Achievements:")
    print("  ✓ Built complete metadata table with dependencies")
    print("  ✓ Optimized loop orders for each layer")
    print("  ✓ Computed optimal blocking factors (L1 + L2)")
    print("  ✓ Identified fusion opportunities")
    print("  ✓ Enabled inter-layer data reuse")
    print("  ✓ Generated execution schedule")
    print("  ✓ Exported runtime configurations")
    
    print("\nGenerated Files:")
    print("  • complete_metadata.json    - Full metadata table")
    print("  • execution_schedule.json   - Execution blocks")
    print("  • runtime_config.json       - Hardware config")
    
    print("\n" + "="*90 + "\n")


if __name__ == '__main__':
    main()
