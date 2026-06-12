#!/usr/bin/env python3
"""
Inter-Layer Reuse Scheduler with Loop Blocking
===============================================
Optimizes execution schedule to maximize data reuse across layers
and determines optimal blocking factors.
"""

from metadata_table import *
from loop_optimizer import LoopOrder6D as LoopOrder, LoopOptimizer, LoopTileConfig
from typing import List, Tuple, Dict
import numpy as np


class ReuseStrategy(Enum):
    """Data reuse strategies."""
    NO_REUSE = "NoReuse"
    OUTPUT_REUSE = "OutputReuse"        # Keep output for next layer
    WEIGHT_REUSE = "WeightReuse"        # Keep weights across batches
    VERTICAL_FUSION = "VerticalFusion"  # Fuse consecutive layers
    HORIZONTAL_FUSION = "HorizontalFusion"  # Fuse parallel branches


@dataclass
class ExecutionBlock:
    """
    A block of computation that can be executed together.
    Represents one iteration of the tiled loops.
    """
    block_id: int
    layer_ids: List[int]  # Layers in this block
    
    # Tile indices
    h_start: int
    h_end: int
    w_start: int
    w_end: int
    k_start: int
    k_end: int
    c_start: int
    c_end: int
    
    # Data requirements
    input_tensors: List[str]
    output_tensors: List[str]
    weight_tensors: List[str]
    
    # Memory footprint
    working_set_size: int = 0
    
    # Reuse information
    can_reuse_from_prev: bool = False
    reused_tensors: List[str] = field(default_factory=list)


class InterLayerReuseScheduler:
    """
    Schedules layer execution to maximize on-chip data reuse.
    """
    
    def __init__(
        self,
        metadata_table: MetadataTable,
        scratchpad_size_kb: int = 16,
        array_height: int = 4,
        array_width: int = 4
    ):
        self.metadata = metadata_table
        self.scratchpad_bytes = scratchpad_size_kb * 1024
        self.array_height = array_height
        self.array_width = array_width
        
        # Scheduling results
        self.execution_blocks: List[ExecutionBlock] = []
        self.reuse_chains: List[List[int]] = []  # Chains of layers that reuse data
        
    def find_reuse_chains(self) -> List[List[int]]:
        """
        Find chains of layers where data can be kept on-chip.
        
        A reuse chain is a sequence of layers where:
        - Each layer's output is consumed by the next layer
        - The output data fits in scratchpad
        - No other layer needs this intermediate data
        """
        chains = []
        visited = set()
        
        for lid in self.metadata.execution_order:
            if lid in visited:
                continue
            
            layer = self.metadata.layers[lid]
            chain = [lid]
            visited.add(lid)
            
            # Try to extend chain forward
            current_lid = lid
            while True:
                current_layer = self.metadata.layers[current_lid]
                
                # Check if there's exactly one dependent layer
                if len(current_layer.feeds_to) != 1:
                    break
                
                next_lid = current_layer.feeds_to[0]
                next_layer = self.metadata.layers[next_lid]
                
                # Check if output fits in scratchpad
                output_size = current_layer.output_size_bytes
                if output_size > self.scratchpad_bytes * 0.5:  # Reserve space for weights
                    break
                
                # Check if next layer has only one input (from current)
                if len(next_layer.depends_on) != 1:
                    break
                
                # Can extend chain
                chain.append(next_lid)
                visited.add(next_lid)
                current_lid = next_lid
            
            if len(chain) > 1:
                chains.append(chain)
                print(f"  Found reuse chain: {' → '.join(self.metadata.layers[lid].layer_name for lid in chain)}")
        
        self.reuse_chains = chains
        return chains
    
    def compute_blocking_for_chain(
        self,
        chain: List[int]
    ) -> Tuple[BlockingFactors, bool]:
        """
        Compute unified blocking factors for a chain of layers.
        
        The blocking must work for all layers in the chain.
        Returns (blocking_factors, is_fusible)
        """
        # Get output shape of last layer in chain (final output)
        last_layer = self.metadata.layers[chain[-1]]
        h_out, w_out, k_out = last_layer.output_shape
        
        # Get input shape of first layer
        first_layer = self.metadata.layers[chain[0]]
        h_in, w_in, c_in = first_layer.input_shape
        
        # Start with L1 tiles based on array size
        tile_h = min(self.array_height, h_out)
        tile_w = min(self.array_width, w_out)
        
        # Compute total working set for the chain
        max_working_set = 0
        
        for lid in chain:
            layer = self.metadata.layers[lid]
            
            # For this tile, how much data do we need?
            # - Input tile
            # - Weight tile  
            # - Output tile
            working_set = (
                layer.input_size_bytes +
                layer.weight_size_bytes +
                layer.output_size_bytes
            )
            max_working_set = max(max_working_set, working_set)
        
        # Check if fusible (working set fits in scratchpad)
        is_fusible = max_working_set <= self.scratchpad_bytes
        
        # Compute L2 blocking to fit in scratchpad
        if is_fusible:
            # Use aggressive blocking - keep everything on chip
            l2_h = min(16, h_out)
            l2_w = min(16, w_out)
            l2_k = k_out
            l2_c = c_in
        else:
            # Conservative blocking - ensure it fits
            available = self.scratchpad_bytes // len(chain)  # Divide among layers
            
            # Heuristic: allocate 50% for weights, 25% input, 25% output
            weight_budget = available * 0.5
            io_budget = available * 0.5
            
            # Start with full dimensions and reduce if needed
            l2_k = k_out
            l2_c = c_in
            l2_h = h_out
            l2_w = w_out
            
            # Reduce K dimension if weight doesn't fit
            first_kh, first_kw = first_layer.kernel_shape[:2]
            weight_per_k = first_kh * first_kw * c_in * 4
            if weight_per_k * l2_k > weight_budget:
                l2_k = max(tile_h, int(weight_budget / weight_per_k))
            
            # Reduce spatial if output doesn't fit
            output_per_tile = l2_h * l2_w * l2_k * 4
            if output_per_tile > io_budget:
                # Reduce both H and W proportionally
                scale = np.sqrt(io_budget / output_per_tile)
                l2_h = max(tile_h, int(l2_h * scale))
                l2_w = max(tile_w, int(l2_w * scale))
        
        blocking = BlockingFactors(
            tile_h_out=tile_h,
            tile_w_out=tile_w,
            tile_k_out=min(self.array_height, k_out),
            tile_c_in=min(self.array_width, c_in),
            l2_tile_h=l2_h,
            l2_tile_w=l2_w,
            l2_tile_k=l2_k,
            l2_tile_c=l2_c
        )
        
        return blocking, is_fusible
    
    def generate_execution_schedule(self) -> List[ExecutionBlock]:
        """
        Generate execution schedule with inter-layer reuse.
        
        Creates execution blocks that:
        1. Process data in tiles
        2. Keep intermediate results on-chip when possible
        3. Minimize memory traffic
        """
        blocks = []
        block_id = 0
        
        # Find reuse chains
        chains = self.find_reuse_chains()
        
        # Process each chain
        for chain in chains:
            print(f"\n  Processing chain: {chain}")
            
            # Compute blocking for this chain
            blocking, is_fusible = self.compute_blocking_for_chain(chain)
            
            # Apply blocking to all layers in chain
            for lid in chain:
                self.metadata.layers[lid].blocking_factors = blocking
            
            # Get dimensions from first and last layer
            first_layer = self.metadata.layers[chain[0]]
            last_layer = self.metadata.layers[chain[-1]]
            
            h_out, w_out, k_out = last_layer.output_shape
            
            # Generate execution blocks
            for h in range(0, h_out, blocking.l2_tile_h):
                for w in range(0, w_out, blocking.l2_tile_w):
                    for k in range(0, k_out, blocking.l2_tile_k):
                        
                        h_end = min(h + blocking.l2_tile_h, h_out)
                        w_end = min(w + blocking.l2_tile_w, w_out)
                        k_end = min(k + blocking.l2_tile_k, k_out)
                        
                        # Create execution block for entire chain
                        block = ExecutionBlock(
                            block_id=block_id,
                            layer_ids=chain.copy(),
                            h_start=h,
                            h_end=h_end,
                            w_start=w,
                            w_end=w_end,
                            k_start=k,
                            k_end=k_end,
                            c_start=0,
                            c_end=k_out,
                            input_tensors=[first_layer.input_tensors[0]],
                            output_tensors=[last_layer.output_tensor],
                            weight_tensors=[
                                self.metadata.layers[lid].weight_tensor
                                for lid in chain
                                if self.metadata.layers[lid].weight_tensor
                            ]
                        )
                        
                        # Mark intermediate outputs as reused
                        for lid in chain[:-1]:
                            layer = self.metadata.layers[lid]
                            block.reused_tensors.append(layer.output_tensor)
                            layer.output_can_stay_on_chip = True
                        
                        blocks.append(block)
                        block_id += 1
            
            if is_fusible:
                print(f"    ✓ Chain is fusible - all data stays on-chip")
            else:
                print(f"    ⚠ Chain requires blocking - partial on-chip reuse")
        
        # Process standalone layers (not in any chain)
        visited_layers = set(lid for chain in chains for lid in chain)
        
        for lid in self.metadata.execution_order:
            if lid in visited_layers:
                continue
            
            # Process as standalone layer
            layer = self.metadata.layers[lid]
            
            if layer.layer_type not in [LayerType.CONV2D, LayerType.DEPTHWISE_CONV2D]:
                continue
            
            # Use standard blocking
            h_out, w_out, k_out = layer.output_shape
            blocking = layer.blocking_factors
            
            for h in range(0, h_out, blocking.l2_tile_h):
                for w in range(0, w_out, blocking.l2_tile_w):
                    for k in range(0, k_out, blocking.l2_tile_k):
                        
                        block = ExecutionBlock(
                            block_id=block_id,
                            layer_ids=[lid],
                            h_start=h,
                            h_end=min(h + blocking.l2_tile_h, h_out),
                            w_start=w,
                            w_end=min(w + blocking.l2_tile_w, w_out),
                            k_start=k,
                            k_end=min(k + blocking.l2_tile_k, k_out),
                            c_start=0,
                            c_end=k_out,
                            input_tensors=layer.input_tensors,
                            output_tensors=[layer.output_tensor],
                            weight_tensors=[layer.weight_tensor] if layer.weight_tensor else []
                        )
                        
                        blocks.append(block)
                        block_id += 1
        
        self.execution_blocks = blocks
        return blocks
    
    def estimate_memory_traffic_with_reuse(self) -> Dict:
        """
        Estimate memory traffic with inter-layer reuse optimization.
        
        Returns detailed breakdown of traffic.
        """
        traffic = {
            'input_reads': 0,
            'weight_reads': 0,
            'output_writes': 0,
            'intermediate_saved': 0,  # Intermediate outputs kept on-chip
            'total': 0
        }
        
        for lid in self.metadata.execution_order:
            layer = self.metadata.layers[lid]
            
            # Input traffic (skip if staying on-chip from previous)
            if not layer.input_can_stay_on_chip:
                traffic['input_reads'] += layer.input_size_bytes
            else:
                traffic['intermediate_saved'] += layer.input_size_bytes
            
            # Weight traffic
            traffic['weight_reads'] += layer.weight_size_bytes
            
            # Output traffic (skip if staying on-chip for next)
            if not layer.output_can_stay_on_chip:
                traffic['output_writes'] += layer.output_size_bytes
            else:
                traffic['intermediate_saved'] += layer.output_size_bytes
        
        traffic['total'] = (
            traffic['input_reads'] +
            traffic['weight_reads'] +
            traffic['output_writes']
        )
        
        return traffic
    
    def print_schedule(self):
        """Print execution schedule summary."""
        print("\n" + "="*80)
        print("EXECUTION SCHEDULE WITH INTER-LAYER REUSE")
        print("="*80)
        
        print(f"\nTotal Execution Blocks: {len(self.execution_blocks)}")
        print(f"Reuse Chains Found: {len(self.reuse_chains)}")
        
        if self.reuse_chains:
            print(f"\nReuse Chain Details:")
            for i, chain in enumerate(self.reuse_chains):
                layer_names = [self.metadata.layers[lid].layer_name for lid in chain]
                print(f"  Chain {i+1}: {' → '.join(layer_names)}")
                
                # Show data flow
                first_layer = self.metadata.layers[chain[0]]
                last_layer = self.metadata.layers[chain[-1]]
                
                print(f"    Input:  {first_layer.input_tensors[0]} "
                      f"{first_layer.input_shape}")
                print(f"    Output: {last_layer.output_tensor} "
                      f"{last_layer.output_shape}")
                
                # Show blocking
                bf = first_layer.blocking_factors
                print(f"    Blocking: L2=({bf.l2_tile_h}×{bf.l2_tile_w}×{bf.l2_tile_k})")
        
        # Memory traffic analysis
        traffic = self.estimate_memory_traffic_with_reuse()
        
        print(f"\nMemory Traffic Analysis:")
        print(f"  Input Reads:     {traffic['input_reads']/(1024*1024):.2f} MB")
        print(f"  Weight Reads:    {traffic['weight_reads']/(1024*1024):.2f} MB")
        print(f"  Output Writes:   {traffic['output_writes']/(1024*1024):.2f} MB")
        print(f"  -" * 40)
        print(f"  Total Traffic:   {traffic['total']/(1024*1024):.2f} MB")
        print(f"  Saved (On-chip): {traffic['intermediate_saved']/(1024*1024):.2f} MB")
        
        reduction = (traffic['intermediate_saved'] / 
                    (traffic['total'] + traffic['intermediate_saved']) * 100)
        print(f"  Traffic Reduction: {reduction:.1f}%")


def demo_inter_layer_reuse():
    """Demonstrate inter-layer reuse scheduling."""
    
    print("\n" + "="*80)
    print("INTER-LAYER REUSE SCHEDULING DEMO")
    print("="*80)
    
    # Create a network with fusion opportunities
    metadata = MetadataTable(scratchpad_size_kb=64)  # Larger scratchpad
    
    # Layer 0: Conv
    metadata.add_layer(LayerMetadata(
        layer_id=0,
        layer_name='conv1',
        layer_type=LayerType.CONV2D,
        input_tensors=['input'],
        output_tensor='conv1_out',
        weight_tensor='conv1_w',
        input_shape=(56, 56, 64),
        output_shape=(56, 56, 128),
        kernel_shape=(3, 3, 64, 128),
        depends_on=[],
        input_size_bytes=56*56*64*4,
        output_size_bytes=56*56*128*4,
        weight_size_bytes=3*3*64*128*4
    ))
    
    # Layer 1: BatchNorm (can fuse with conv)
    metadata.add_layer(LayerMetadata(
        layer_id=1,
        layer_name='bn1',
        layer_type=LayerType.BATCHNORM,
        input_tensors=['conv1_out'],
        output_tensor='bn1_out',
        weight_tensor='bn1_params',
        input_shape=(56, 56, 128),
        output_shape=(56, 56, 128),
        depends_on=[0],
        input_size_bytes=56*56*128*4,
        output_size_bytes=56*56*128*4,
        weight_size_bytes=128*4*4  # gamma, beta, mean, var
    ))
    
    # Layer 2: ReLU (can fuse with previous)
    metadata.add_layer(LayerMetadata(
        layer_id=2,
        layer_name='relu1',
        layer_type=LayerType.ACTIVATION,
        input_tensors=['bn1_out'],
        output_tensor='relu1_out',
        input_shape=(56, 56, 128),
        output_shape=(56, 56, 128),
        depends_on=[1],
        input_size_bytes=56*56*128*4,
        output_size_bytes=56*56*128*4,
        weight_size_bytes=0
    ))
    
    # Layer 3: Conv (consumes output from chain)
    metadata.add_layer(LayerMetadata(
        layer_id=3,
        layer_name='conv2',
        layer_type=LayerType.CONV2D,
        input_tensors=['relu1_out'],
        output_tensor='conv2_out',
        weight_tensor='conv2_w',
        input_shape=(56, 56, 128),
        output_shape=(56, 56, 128),
        kernel_shape=(3, 3, 128, 128),
        depends_on=[2],
        input_size_bytes=56*56*128*4,
        output_size_bytes=56*56*128*4,
        weight_size_bytes=3*3*128*128*4
    ))
    
    # Build metadata
    print("\n[Step 1] Building metadata table...")
    metadata.compute_execution_order()
    metadata.optimize_blocking_factors(array_height=4, array_width=4)
    metadata.find_fusion_opportunities()
    
    # Create scheduler
    print("\n[Step 2] Creating inter-layer reuse scheduler...")
    scheduler = InterLayerReuseScheduler(
        metadata,
        scratchpad_size_kb=64,
        array_height=4,
        array_width=4
    )
    
    # Find reuse chains
    print("\n[Step 3] Finding reuse chains...")
    chains = scheduler.find_reuse_chains()
    
    # Generate schedule
    print("\n[Step 4] Generating execution schedule...")
    blocks = scheduler.generate_execution_schedule()
    print(f"Generated {len(blocks)} execution blocks")
    
    # Print schedule
    scheduler.print_schedule()
    
    # Save metadata with schedule
    metadata.to_json('./demo_output/metadata_with_reuse.json')
    print(f"\nSaved to ./demo_output/metadata_with_reuse.json")


if __name__ == '__main__':
    import os
    os.makedirs('./demo_output', exist_ok=True)
    demo_inter_layer_reuse()
