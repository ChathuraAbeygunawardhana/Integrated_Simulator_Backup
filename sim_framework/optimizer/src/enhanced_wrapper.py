#!/usr/bin/env python3
"""
Enhanced TensorFlow Wrapper with Metadata Table and Inter-Layer Reuse
=====================================================================
Complete integration of:
- Loop order optimization
- Loop blocking/tiling
- Inter-layer data reuse
- Metadata-driven execution
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional
from metadata_table import *
from inter_layer_reuse import *
from loop_optimizer import LoopOrder6D as LoopOrder, LoopOptimizer, LoopTileConfig
import json
import os


class EnhancedSystolicWrapper:
    """
    Enhanced wrapper with full metadata management and inter-layer optimization.
    """
    
    def __init__(
        self,
        array_height: int = 4,
        array_width: int = 4,
        scratchpad_size_kb: int = 16,
        data_width: int = 32,
        frac_bits: int = 16
    ):
        self.array_height = array_height
        self.array_width = array_width
        self.scratchpad_size_kb = scratchpad_size_kb
        self.data_width = data_width
        self.frac_bits = frac_bits
        
        # Metadata management
        self.metadata = MetadataTable(scratchpad_size_kb=scratchpad_size_kb)
        self.scheduler = None
        
        # Optimizers
        self.loop_optimizer = LoopOptimizer(array_height, array_width)
        
        # Model tracking
        self.model = None
        self.layer_id_map = {}  # TF layer name -> metadata layer ID
    
    def analyze_model(self, model: tf.keras.Model, sample_input_shape: Tuple = None):
        """
        Build complete metadata table from TensorFlow model.
        """
        print("\n" + "="*80)
        print("ANALYZING MODEL AND BUILDING METADATA TABLE")
        print("="*80)
        
        self.model = model
        
        # Get sample input shape
        if sample_input_shape is None:
            if hasattr(model, 'input_shape'):
                sample_input_shape = model.input_shape
            else:
                sample_input_shape = (1, 28, 28, 1)  # Default
        
        # Extract layers
        conv_layers = []
        layer_connections = {}  # layer_name -> input_layer_names
        
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                conv_layers.append(layer)
                
                # Track connections
                try:
                    inbound = layer.inbound_nodes[0].inbound_layers
                    if not isinstance(inbound, list):
                        inbound = [inbound]
                    layer_connections[layer.name] = [l.name for l in inbound]
                except:
                    layer_connections[layer.name] = []
        
        print(f"\nFound {len(conv_layers)} Conv2D layers")
        
        # Build metadata for each layer
        current_shape = sample_input_shape
        tensor_id = 0
        
        # Add network input tensor
        input_tensor_name = f"input_tensor"
        self.metadata.add_tensor(TensorMetadata(
            name=input_tensor_name,
            shape=current_shape[1:],  # Remove batch dimension
            is_network_input=True
        ))
        
        for layer_id, layer in enumerate(conv_layers):
            print(f"\n  Processing layer {layer_id}: {layer.name}")
            
            # Get layer parameters
            weights, biases = layer.get_weights()
            kh, kw, c_in, k_out = weights.shape
            
            # Determine input shape
            if layer_id == 0:
                h_in, w_in, c_in = current_shape[1:]
            else:
                h_in, w_in, c_in = current_shape[1:]
            
            # Calculate output shape
            if layer.padding.lower() == 'same':
                h_out = h_in // layer.strides[0]
                w_out = w_in // layer.strides[1]
            else:
                h_out = (h_in - kh) // layer.strides[0] + 1
                w_out = (w_in - kw) // layer.strides[1] + 1
            
            # Create tensor names
            input_tensor = input_tensor_name if layer_id == 0 else f"tensor_{layer_id-1}"
            output_tensor = f"tensor_{layer_id}"
            weight_tensor = f"weights_{layer_id}"
            
            # Determine dependencies
            depends_on = []
            if layer_id > 0:
                # Simple sequential assumption
                depends_on = [layer_id - 1]
            
            # Create layer metadata
            layer_meta = LayerMetadata(
                layer_id=layer_id,
                layer_name=layer.name,
                layer_type=LayerType.CONV2D,
                input_tensors=[input_tensor],
                output_tensor=output_tensor,
                weight_tensor=weight_tensor,
                input_shape=(h_in, w_in, c_in),
                output_shape=(h_out, w_out, k_out),
                kernel_shape=(kh, kw, c_in, k_out),
                depends_on=depends_on,
                input_size_bytes=h_in * w_in * c_in * 4,
                output_size_bytes=h_out * w_out * k_out * 4,
                weight_size_bytes=kh * kw * c_in * k_out * 4
            )
            
            self.metadata.add_layer(layer_meta)
            self.layer_id_map[layer.name] = layer_id
            
            # Add output tensor
            is_last = (layer_id == len(conv_layers) - 1)
            self.metadata.add_tensor(TensorMetadata(
                name=output_tensor,
                shape=(h_out, w_out, k_out),
                producer_layer=layer.name,
                is_network_output=is_last
            ))
            
            # Update current shape
            current_shape = (1, h_out, w_out, k_out)
        
        print(f"\nMetadata table built: {len(self.metadata.layers)} layers, "
              f"{len(self.metadata.tensors)} tensors")
    
    def optimize_full_network(
        self,
        optimization_target: str = 'balanced',
        enable_inter_layer_reuse: bool = True
    ):
        """
        Perform full network optimization:
        1. Compute execution order
        2. Optimize loop orders
        3. Optimize blocking factors
        4. Find fusion opportunities
        5. Enable inter-layer reuse
        6. Generate execution schedule
        """
        print("\n" + "="*80)
        print("FULL NETWORK OPTIMIZATION")
        print("="*80)
        
        # Step 1: Compute execution order
        print("\n[1/6] Computing execution order...")
        self.metadata.compute_execution_order()
        print(f"  Execution order: {self.metadata.execution_order}")
        
        # Step 2: Optimize loop orders
        print("\n[2/6] Optimizing loop orders...")
        for lid in self.metadata.execution_order:
            layer = self.metadata.layers[lid]
            
            if layer.layer_type != LayerType.CONV2D:
                continue
            
            # Find optimal loop order
            best_order, all_results = self.loop_optimizer.find_optimal_loop_order(
                k_out=layer.output_shape[2],
                c_in=layer.input_shape[2],
                kh=layer.kernel_shape[0],
                kw=layer.kernel_shape[1],
                h_out=layer.output_shape[0],
                w_out=layer.output_shape[1],
                tile_config=LoopTileConfig(
                    tile_k=min(self.array_height, layer.output_shape[2]),
                    tile_c=min(self.array_width, layer.input_shape[2])
                ),
                optimization_target=optimization_target
            )
            
            layer.loop_order = best_order.value
            layer.loop_order_code = list(LoopOrder).index(best_order)
            
            print(f"  {layer.layer_name}: {best_order.value}")
        
        # Step 3: Optimize blocking factors
        print("\n[3/6] Optimizing blocking factors...")
        self.metadata.optimize_blocking_factors(self.array_height, self.array_width)
        
        # Step 4: Find fusion opportunities
        print("\n[4/6] Finding fusion opportunities...")
        self.metadata.find_fusion_opportunities()
        
        # Step 5: Analyze inter-layer reuse
        if enable_inter_layer_reuse:
            print("\n[5/6] Analyzing inter-layer reuse...")
            self.metadata.analyze_inter_layer_reuse()
            
            # Step 6: Generate execution schedule
            print("\n[6/6] Generating execution schedule...")
            self.scheduler = InterLayerReuseScheduler(
                self.metadata,
                self.scratchpad_size_kb,
                self.array_height,
                self.array_width
            )
            
            chains = self.scheduler.find_reuse_chains()
            blocks = self.scheduler.generate_execution_schedule()
            
            print(f"  Generated {len(blocks)} execution blocks")
            print(f"  Found {len(chains)} reuse chains")
        else:
            print("\n[5/6] Inter-layer reuse: DISABLED")
            print("\n[6/6] Skipping execution schedule")
        
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
    
    def print_optimization_summary(self):
        """Print comprehensive optimization summary."""
        
        self.metadata.print_summary()
        
        if self.scheduler:
            self.scheduler.print_schedule()
    
    def export_metadata(
        self,
        json_path: str,
        include_execution_blocks: bool = True
    ):
        """Export complete metadata to JSON."""
        
        # Base metadata
        self.metadata.to_json(json_path)
        
        # Add execution blocks if scheduler exists
        if include_execution_blocks and self.scheduler:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Add execution blocks
            data['execution_blocks'] = [
                {
                    'block_id': block.block_id,
                    'layer_ids': block.layer_ids,
                    'tile': {
                        'h': [block.h_start, block.h_end],
                        'w': [block.w_start, block.w_end],
                        'k': [block.k_start, block.k_end],
                        'c': [block.c_start, block.c_end]
                    },
                    'tensors': {
                        'inputs': block.input_tensors,
                        'outputs': block.output_tensors,
                        'weights': block.weight_tensors
                    },
                    'can_reuse_from_prev': block.can_reuse_from_prev,
                    'reused_tensors': block.reused_tensors
                }
                for block in self.scheduler.execution_blocks[:100]  # Limit for file size
            ]
            
            # Add reuse chains
            data['reuse_chains'] = [
                {
                    'chain_id': i,
                    'layer_ids': chain,
                    'layer_names': [self.metadata.layers[lid].layer_name for lid in chain]
                }
                for i, chain in enumerate(self.scheduler.reuse_chains)
            ]
            
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        print(f"\nMetadata exported to {json_path}")
    
    def generate_runtime_config(self, output_dir: str):
        """
        Generate runtime configuration files for hardware execution.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate per-layer configs
        for lid in self.metadata.execution_order:
            layer = self.metadata.layers[lid]
            
            if layer.layer_type != LayerType.CONV2D:
                continue
            
            config = {
                'layer_id': lid,
                'layer_name': layer.layer_name,
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
                'loop_optimization': {
                    'loop_order': layer.loop_order,
                    'loop_order_code': layer.loop_order_code
                },
                'blocking': layer.blocking_factors.to_dict(),
                'inter_layer_reuse': {
                    'input_on_chip': layer.input_can_stay_on_chip,
                    'output_on_chip': layer.output_can_stay_on_chip
                }
            }
            
            config_file = os.path.join(output_dir, f"layer_{lid}_{layer.layer_name}.json")
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
        
        print(f"\nRuntime configs generated in {output_dir}")


def demo_enhanced_wrapper():
    """Demonstrate enhanced wrapper with full optimization."""
    
    print("\n" + "="*90)
    print("ENHANCED SYSTOLIC WRAPPER DEMO")
    print("="*90)
    
    # Create a simple model
    print("\n[Creating Model]")
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            16, (3, 3), padding='same', activation='relu',
            input_shape=(28, 28, 1), name='conv1'
        ),
        tf.keras.layers.Conv2D(
            32, (3, 3), padding='same', activation='relu', name='conv2'
        ),
        tf.keras.layers.Conv2D(
            64, (3, 3), padding='same', activation='relu', name='conv3'
        ),
    ])
    
    model.summary()
    
    # Create enhanced wrapper
    print("\n[Creating Enhanced Wrapper]")
    wrapper = EnhancedSystolicWrapper(
        array_height=4,
        array_width=4,
        scratchpad_size_kb=64,  # 64 KB scratchpad
        data_width=32,
        frac_bits=16
    )
    
    # Analyze model
    wrapper.analyze_model(model, sample_input_shape=(1, 28, 28, 1))
    
    # Optimize full network
    wrapper.optimize_full_network(
        optimization_target='balanced',
        enable_inter_layer_reuse=True
    )
    
    # Print summary
    wrapper.print_optimization_summary()
    
    # Export metadata
    wrapper.export_metadata(
        './demo_output/full_metadata.json',
        include_execution_blocks=True
    )
    
    # Generate runtime configs
    wrapper.generate_runtime_config('./demo_output/runtime_configs')
    
    print("\n" + "="*90)
    print("DEMO COMPLETE")
    print("="*90)
    print("\nGenerated files:")
    print("  ./demo_output/full_metadata.json - Complete metadata table")
    print("  ./demo_output/runtime_configs/   - Per-layer runtime configs")
    print("="*90 + "\n")


if __name__ == '__main__':
    os.makedirs('./demo_output', exist_ok=True)
    demo_enhanced_wrapper()
