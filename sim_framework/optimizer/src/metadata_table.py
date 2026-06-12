#!/usr/bin/env python3
"""
Layer Metadata Table and Dependency Graph
==========================================
Manages layer dependencies, topology, and optimization metadata for
inter-layer reuse and loop blocking.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import json
import numpy as np
from .blocking_optimizer import BlockingOptimizer, BlockingConfig, TileLoopOrder, LoopNestConfig


class LayerType(Enum):
    """Supported layer types."""
    CONV2D = "Conv2D"
    DEPTHWISE_CONV2D = "DepthwiseConv2D"
    DENSE = "Dense"
    POOLING = "Pooling"
    ACTIVATION = "Activation"
    BATCHNORM = "BatchNorm"
    ADD = "Add"
    CONCATENATE = "Concatenate"


class MemoryLocation(Enum):
    """Memory hierarchy levels."""
    DRAM = "DRAM"
    SCRATCHPAD = "Scratchpad"
    REGISTER_FILE = "RegisterFile"
    PE_REGISTER = "PERegister"


@dataclass
class TensorMetadata:
    """Metadata for a tensor (activation or weight)."""
    name: str
    shape: Tuple[int, ...]  # (H, W, C) or (N, M) for weights
    dtype: str = "float32"
    size_bytes: int = 0
    
    # Memory allocation
    memory_location: MemoryLocation = MemoryLocation.DRAM
    base_address: int = 0
    
    # Lifetime tracking
    producer_layer: Optional[str] = None
    consumer_layers: List[str] = field(default_factory=list)
    is_network_input: bool = False
    is_network_output: bool = False
    
    # Reuse tracking
    reuse_count: int = 0
    keep_on_chip: bool = False
    
    def __post_init__(self):
        if self.size_bytes == 0:
            self.size_bytes = int(np.prod(self.shape) * 4)  # 4 bytes for float32


@dataclass
class BlockingFactors:
    """Loop blocking/tiling factors for a layer."""
    # Output tiling
    tile_h_out: int = 4
    tile_w_out: int = 4
    tile_k_out: int = 4
    
    # Input tiling
    tile_c_in: int = 4
    
    # Kernel tiling (for very large kernels)
    tile_kh: int = 0  # 0 means no tiling
    tile_kw: int = 0
    
    # Multi-level tiling
    l2_tile_h: int = 0  # Level 2 (scratchpad) tiling
    l2_tile_w: int = 0
    l2_tile_k: int = 0
    l2_tile_c: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'l1_tiles': {
                'h_out': self.tile_h_out,
                'w_out': self.tile_w_out,
                'k_out': self.tile_k_out,
                'c_in': self.tile_c_in,
                'kh': self.tile_kh if self.tile_kh > 0 else None,
                'kw': self.tile_kw if self.tile_kw > 0 else None,
            },
            'l2_tiles': {
                'h': self.l2_tile_h if self.l2_tile_h > 0 else None,
                'w': self.l2_tile_w if self.l2_tile_w > 0 else None,
                'k': self.l2_tile_k if self.l2_tile_k > 0 else None,
                'c': self.l2_tile_c if self.l2_tile_c > 0 else None,
            }
        }


@dataclass
class LayerMetadata:
    """Complete metadata for a layer."""
    layer_id: int
    layer_name: str
    layer_type: LayerType
    
    # Topology
    input_tensors: List[str]  # Names of input tensors
    output_tensor: str         # Name of output tensor
    weight_tensor: Optional[str] = None
    
    # Dimensions
    input_shape: Tuple[int, ...] = ()
    output_shape: Tuple[int, ...] = ()
    kernel_shape: Tuple[int, ...] = ()
    
    # Dependencies
    depends_on: List[int] = field(default_factory=list)  # Layer IDs
    feeds_to: List[int] = field(default_factory=list)
    
    # Optimization metadata
    loop_nest_config: Optional[LoopNestConfig] = None  # Complete blocking + loop order
    
    # Legacy support (deprecated)
    loop_order: str = "K_C_Spatial"
    loop_order_code: int = 0
    blocking_factors: Optional[BlockingConfig] = None
    
    # Resource requirements
    weight_size_bytes: int = 0
    input_size_bytes: int = 0
    output_size_bytes: int = 0
    scratchpad_requirement: int = 0
    
    # Fusion opportunities
    can_fuse_with_next: bool = False
    can_fuse_with_prev: bool = False
    fused_with: List[int] = field(default_factory=list)
    
    # Performance hints
    is_memory_bound: bool = False
    is_compute_bound: bool = False
    estimated_cycles: int = 0
    
    # Inter-layer reuse
    input_can_stay_on_chip: bool = False
    output_can_stay_on_chip: bool = False
    
    def to_dict(self) -> Dict:
        result = {
            'layer_id': self.layer_id,
            'layer_name': self.layer_name,
            'layer_type': self.layer_type.value,
            'input_tensors': self.input_tensors,
            'output_tensor': self.output_tensor,
            'weight_tensor': self.weight_tensor,
            'input_shape': list(self.input_shape),
            'output_shape': list(self.output_shape),
            'kernel_shape': list(self.kernel_shape),
            'depends_on': self.depends_on,
            'feeds_to': self.feeds_to,
            'loop_order': self.loop_order,
            'loop_order_code': self.loop_order_code,
            'resource_requirements': {
                'weight_bytes': self.weight_size_bytes,
                'input_bytes': self.input_size_bytes,
                'output_bytes': self.output_size_bytes,
                'scratchpad_bytes': self.scratchpad_requirement
            },
            'optimization_hints': {
                'can_fuse_with_next': self.can_fuse_with_next,
                'can_fuse_with_prev': self.can_fuse_with_prev,
                'fused_with': self.fused_with,
                'is_memory_bound': self.is_memory_bound,
                'is_compute_bound': self.is_compute_bound,
                'estimated_cycles': self.estimated_cycles
            },
            'inter_layer_reuse': {
                'input_can_stay_on_chip': self.input_can_stay_on_chip,
                'output_can_stay_on_chip': self.output_can_stay_on_chip
            }
        }
        
        # Add new loop_nest_config if present
        if self.loop_nest_config is not None:
            result['loop_nest_config'] = {
                'blocking': self.loop_nest_config.blocking.to_dict(),
                'tile_loop_order': self.loop_nest_config.tile_loop_order.name,
                'tile_loop_order_code': list(TileLoopOrder).index(self.loop_nest_config.tile_loop_order)
            }
        elif self.blocking_factors is not None:
            # Legacy support
            result['blocking_factors'] = self.blocking_factors.to_dict()
        
        return result


class MetadataTable:
    """
    Central metadata table for the entire network.
    Manages dependencies, topology, and optimization metadata.
    """
    
    def __init__(self, scratchpad_size_kb: int = 16):
        self.scratchpad_size_bytes = scratchpad_size_kb * 1024
        
        # Storage
        self.layers: Dict[int, LayerMetadata] = {}
        self.tensors: Dict[str, TensorMetadata] = {}
        
        # Topology
        self.execution_order: List[int] = []
        self.dependency_graph: Dict[int, Set[int]] = {}  # layer_id -> dependencies
        
        # Global optimization state
        self.total_memory_traffic: int = 0
        self.on_chip_memory_usage: int = 0
        
    def add_layer(self, metadata: LayerMetadata):
        """Add layer metadata."""
        self.layers[metadata.layer_id] = metadata
        self.dependency_graph[metadata.layer_id] = set(metadata.depends_on)
    
    def add_tensor(self, metadata: TensorMetadata):
        """Add tensor metadata."""
        self.tensors[metadata.name] = metadata
    
    def compute_execution_order(self):
        """Compute topological sort for execution order."""
        # Kahn's algorithm
        in_degree = {lid: len(deps) for lid, deps in self.dependency_graph.items()}
        queue = [lid for lid, deg in in_degree.items() if deg == 0]
        order = []
        
        while queue:
            current = queue.pop(0)
            order.append(current)
            
            # Update in-degrees
            for lid, deps in self.dependency_graph.items():
                if current in deps:
                    in_degree[lid] -= 1
                    if in_degree[lid] == 0:
                        queue.append(lid)
        
        if len(order) != len(self.layers):
            raise ValueError("Cyclic dependency detected in network!")
        
        self.execution_order = order
        
        # Update feeds_to relationships
        for lid in order:
            layer = self.layers[lid]
            for dep_id in layer.depends_on:
                if dep_id in self.layers:
                    self.layers[dep_id].feeds_to.append(lid)
    
    def analyze_inter_layer_reuse(self):
        """
        Analyze which tensors can stay on-chip for inter-layer reuse.
        """
        if not self.execution_order:
            self.compute_execution_order()
        
        for i, lid in enumerate(self.execution_order):
            layer = self.layers[lid]
            output_tensor = self.tensors.get(layer.output_tensor)
            
            if output_tensor is None:
                continue
            
            # Check if output is consumed by next layer
            if i + 1 < len(self.execution_order):
                next_lid = self.execution_order[i + 1]
                next_layer = self.layers[next_lid]
                
                # If next layer uses this output AND it fits in scratchpad
                if layer.output_tensor in next_layer.input_tensors:
                    output_size = output_tensor.size_bytes
                    
                    if output_size <= self.scratchpad_size_bytes:
                        layer.output_can_stay_on_chip = True
                        next_layer.input_can_stay_on_chip = True
                        output_tensor.keep_on_chip = True
                        
                        print(f"  Inter-layer reuse: {layer.layer_name} → {next_layer.layer_name}")
                        print(f"    Tensor {output_tensor.name} ({output_size/1024:.1f} KB) stays on-chip")
    
    def optimize_blocking_factors(
        self,
        array_height: int = 4,
        array_width: int = 4,
        scratchpad_kb: int = None,
        optimization_objective: str = 'min_traffic'
    ):
        """
        Optimize blocking factors AND loop order using search-based approach.
        
        This is the CORRECT approach:
        1. Block the loops first (determine tile sizes)
        2. Reorder the INTER-TILE loops
        3. Search for optimal (blocking, order) pair
        
        Args:
            optimization_objective: 'min_traffic', 'max_reuse', or 'balanced'
        """
        if scratchpad_kb is None:
            scratchpad_kb = self.scratchpad_size_bytes // 1024
        
        print(f"\nOptimizing blocking + loop order with search...")
        print(f"  Scratchpad: {scratchpad_kb} KB")
        print(f"  Objective: {optimization_objective}")
        
        for lid in self.execution_order:
            layer = self.layers[lid]
            
            if layer.layer_type not in [LayerType.CONV2D, LayerType.DEPTHWISE_CONV2D]:
                continue
            
            h_out, w_out, k_out = layer.output_shape
            h_in, w_in, c_in = layer.input_shape
            kh, kw = layer.kernel_shape[:2]
            
            print(f"\n  Layer {lid}: {layer.layer_name}")
            print(f"    Dims: K={k_out}, C={c_in}, H={h_out}×{w_out}, kernel={kh}×{kw}")
            
            # Create optimizer for this layer
            optimizer = BlockingOptimizer(
                K=k_out, C=c_in, H=h_out, W=w_out, KH=kh, KW=kw,
                array_height=array_height,
                array_width=array_width,
                scratchpad_kb=scratchpad_kb
            )
            
            # Search for optimal configuration
            best_config, all_results = optimizer.search_optimal_config(optimization_objective)
            
            # Store in layer metadata
            layer.loop_nest_config = best_config
            
            # Update legacy fields for compatibility
            layer.loop_order = best_config.tile_loop_order.name
            layer.loop_order_code = list(TileLoopOrder).index(best_config.tile_loop_order)
            layer.blocking_factors = best_config.blocking
            
            # Update scratchpad requirement
            layer.scratchpad_requirement = best_config.blocking.scratchpad_requirement(
                k_out, c_in, h_out, w_out, kh, kw
            )
            
            print(f"    → Optimal: K_tile={best_config.blocking.tile_K}, "
                  f"C_tile={best_config.blocking.tile_C}, "
                  f"spatial={best_config.blocking.tile_H}×{best_config.blocking.tile_W}")
            print(f"    → Loop order: {best_config.tile_loop_order.name}")
            
            # Get metrics
            metrics = all_results[str(best_config)]
            print(f"    → DRAM traffic: {metrics['dram_traffic']/1e6:.2f} MB")
            print(f"    → Weight reuse: {metrics['weight_reuse']:.1f}x, "
                  f"Input reuse: {metrics['input_reuse']:.1f}x")
    
    def _estimate_scratchpad_usage(self, layer: LayerMetadata) -> int:
        """Estimate scratchpad usage for a layer with given blocking."""
        bf = layer.blocking_factors
        
        if layer.layer_type == LayerType.CONV2D:
            h_in, w_in, c_in = layer.input_shape
            h_out, w_out, k_out = layer.output_shape
            kh, kw = layer.kernel_shape[:2]
            
            # Input tile
            input_tile = (bf.l2_tile_h + kh - 1) * (bf.l2_tile_w + kw - 1) * bf.l2_tile_c * 4
            
            # Weight tile
            weight_tile = kh * kw * bf.l2_tile_c * bf.l2_tile_k * 4
            
            # Output tile
            output_tile = bf.l2_tile_h * bf.l2_tile_w * bf.l2_tile_k * 4
            
            return input_tile + weight_tile + output_tile
        
        return 0
    
    def find_fusion_opportunities(self):
        """
        Identify layers that can be fused together.
        Common patterns:
        - Conv + BatchNorm
        - Conv + Activation
        - Conv + BatchNorm + Activation
        """
        for i, lid in enumerate(self.execution_order[:-1]):
            layer = self.layers[lid]
            next_lid = self.execution_order[i + 1]
            next_layer = self.layers[next_lid]
            
            # Check if layers can be fused
            can_fuse = False
            
            # Pattern 1: Conv → Activation
            if (layer.layer_type == LayerType.CONV2D and 
                next_layer.layer_type == LayerType.ACTIVATION):
                if layer.output_tensor in next_layer.input_tensors:
                    can_fuse = True
            
            # Pattern 2: Conv → BatchNorm
            if (layer.layer_type == LayerType.CONV2D and 
                next_layer.layer_type == LayerType.BATCHNORM):
                if layer.output_tensor in next_layer.input_tensors:
                    can_fuse = True
            
            if can_fuse and len(next_layer.feeds_to) == 1:  # Only fuse if single consumer
                layer.can_fuse_with_next = True
                next_layer.can_fuse_with_prev = True
                
                # Mark fusion
                layer.fused_with.append(next_lid)
                next_layer.fused_with.append(lid)
    
    def estimate_memory_traffic(self):
        """
        Estimate total memory traffic with inter-layer reuse optimization.
        """
        total_traffic = 0
        
        for lid in self.execution_order:
            layer = self.layers[lid]
            
            # Input traffic (if not staying on-chip from previous layer)
            if not layer.input_can_stay_on_chip:
                total_traffic += layer.input_size_bytes
            
            # Weight traffic (always loaded)
            total_traffic += layer.weight_size_bytes
            
            # Output traffic (if not staying on-chip for next layer)
            if not layer.output_can_stay_on_chip:
                total_traffic += layer.output_size_bytes
        
        self.total_memory_traffic = total_traffic
        return total_traffic
    
    def to_json(self, filename: str):
        """Save metadata table to JSON file."""
        data = {
            'config': {
                'scratchpad_size_kb': self.scratchpad_size_bytes // 1024,
                'total_layers': len(self.layers),
                'total_tensors': len(self.tensors),
                'total_memory_traffic_mb': self.total_memory_traffic / (1024 * 1024)
            },
            'execution_order': self.execution_order,
            'layers': {
                str(lid): layer.to_dict() 
                for lid, layer in self.layers.items()
            },
            'tensors': {
                name: {
                    'name': tensor.name,
                    'shape': list(tensor.shape),
                    'size_bytes': tensor.size_bytes,
                    'memory_location': tensor.memory_location.value,
                    'base_address': f"0x{tensor.base_address:08x}",
                    'producer': tensor.producer_layer,
                    'consumers': tensor.consumer_layers,
                    'reuse_count': tensor.reuse_count,
                    'keep_on_chip': tensor.keep_on_chip
                }
                for name, tensor in self.tensors.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def print_summary(self):
        """Print summary of metadata table."""
        print("\n" + "="*80)
        print("METADATA TABLE SUMMARY")
        print("="*80)
        
        print(f"\nNetwork Configuration:")
        print(f"  Total Layers: {len(self.layers)}")
        print(f"  Total Tensors: {len(self.tensors)}")
        print(f"  Scratchpad Size: {self.scratchpad_size_bytes/1024:.1f} KB")
        
        print(f"\nExecution Order: {' → '.join(str(lid) for lid in self.execution_order)}")
        
        print(f"\nMemory Traffic Analysis:")
        print(f"  Total Traffic: {self.total_memory_traffic/(1024*1024):.2f} MB")
        
        # Count inter-layer reuse opportunities
        reuse_layers = sum(1 for l in self.layers.values() if l.output_can_stay_on_chip)
        print(f"  Inter-layer Reuse: {reuse_layers} layers")
        
        # Count fusion opportunities
        fusion_layers = sum(1 for l in self.layers.values() if l.can_fuse_with_next)
        print(f"  Fusion Opportunities: {fusion_layers} pairs")
        
        print(f"\nPer-Layer Details:")
        for lid in self.execution_order:
            layer = self.layers[lid]
            print(f"\n  [{lid}] {layer.layer_name} ({layer.layer_type.value})")
            print(f"      Input:  {layer.input_shape}")
            print(f"      Output: {layer.output_shape}")
            
            if layer.loop_nest_config:
                bf = layer.loop_nest_config.blocking
                print(f"      Blocking: K_tile={bf.tile_K}, C_tile={bf.tile_C}, "
                      f"spatial={bf.tile_H}×{bf.tile_W}")
                print(f"      Loop order (inter-tile): {layer.loop_nest_config.tile_loop_order.name}")
            else:
                print(f"      Loop Order: {layer.loop_order}")
                if layer.blocking_factors:
                    bf = layer.blocking_factors
                    print(f"      Blocking: K={bf.tile_K}, C={bf.tile_C}, H={bf.tile_H}×{bf.tile_W}")
            
            print(f"      Scratchpad Need: {layer.scratchpad_requirement/1024:.1f} KB")
            
            if layer.output_can_stay_on_chip:
                print(f"      ✓ Output stays on-chip for next layer")
            
            if layer.can_fuse_with_next:
                print(f"      ✓ Can fuse with next layer")


def demo_metadata_table():
    """Demonstration of metadata table usage."""
    
    print("\n" + "="*80)
    print("METADATA TABLE AND DEPENDENCY TRACKING DEMO")
    print("="*80)
    
    # Create metadata table
    metadata = MetadataTable(scratchpad_size_kb=16)
    
    # Build a simple CNN
    # Layer 0: Conv2D
    metadata.add_layer(LayerMetadata(
        layer_id=0,
        layer_name='conv1',
        layer_type=LayerType.CONV2D,
        input_tensors=['input'],
        output_tensor='conv1_out',
        weight_tensor='conv1_weights',
        input_shape=(224, 224, 3),
        output_shape=(224, 224, 32),
        kernel_shape=(3, 3, 3, 32),
        depends_on=[],
        input_size_bytes=224*224*3*4,
        output_size_bytes=224*224*32*4,
        weight_size_bytes=3*3*3*32*4
    ))
    
    # Layer 1: Activation
    metadata.add_layer(LayerMetadata(
        layer_id=1,
        layer_name='relu1',
        layer_type=LayerType.ACTIVATION,
        input_tensors=['conv1_out'],
        output_tensor='relu1_out',
        input_shape=(224, 224, 32),
        output_shape=(224, 224, 32),
        depends_on=[0],
        input_size_bytes=224*224*32*4,
        output_size_bytes=224*224*32*4,
        weight_size_bytes=0
    ))
    
    # Layer 2: Conv2D
    metadata.add_layer(LayerMetadata(
        layer_id=2,
        layer_name='conv2',
        layer_type=LayerType.CONV2D,
        input_tensors=['relu1_out'],
        output_tensor='conv2_out',
        weight_tensor='conv2_weights',
        input_shape=(112, 112, 32),
        output_shape=(112, 112, 64),
        kernel_shape=(3, 3, 32, 64),
        depends_on=[1],
        input_size_bytes=112*112*32*4,
        output_size_bytes=112*112*64*4,
        weight_size_bytes=3*3*32*64*4
    ))
    
    # Add tensors
    metadata.add_tensor(TensorMetadata(
        name='input',
        shape=(224, 224, 3),
        is_network_input=True
    ))
    
    metadata.add_tensor(TensorMetadata(
        name='conv1_out',
        shape=(224, 224, 32),
        producer_layer='conv1',
        consumer_layers=['relu1']
    ))
    
    metadata.add_tensor(TensorMetadata(
        name='relu1_out',
        shape=(224, 224, 32),
        producer_layer='relu1',
        consumer_layers=['conv2']
    ))
    
    metadata.add_tensor(TensorMetadata(
        name='conv2_out',
        shape=(112, 112, 64),
        producer_layer='conv2',
        is_network_output=True
    ))
    
    # Compute execution order
    print("\n[Step 1] Computing execution order...")
    metadata.compute_execution_order()
    print(f"Execution order: {metadata.execution_order}")
    
    # Optimize blocking
    print("\n[Step 2] Optimizing blocking factors...")
    metadata.optimize_blocking_factors(array_height=4, array_width=4)
    
    # Analyze inter-layer reuse
    print("\n[Step 3] Analyzing inter-layer reuse...")
    metadata.analyze_inter_layer_reuse()
    
    # Find fusion opportunities
    print("\n[Step 4] Finding fusion opportunities...")
    metadata.find_fusion_opportunities()
    
    # Estimate memory traffic
    print("\n[Step 5] Estimating memory traffic...")
    traffic = metadata.estimate_memory_traffic()
    print(f"Total memory traffic: {traffic/(1024*1024):.2f} MB")
    
    # Print summary
    metadata.print_summary()
    
    # Save to JSON
    metadata.to_json('./demo_output/metadata_table.json')
    print(f"\nMetadata saved to ./demo_output/metadata_table.json")


if __name__ == '__main__':
    import os
    os.makedirs('./demo_output', exist_ok=True)
    demo_metadata_table()
