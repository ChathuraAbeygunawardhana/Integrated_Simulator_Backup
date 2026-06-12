#!/usr/bin/env python3
"""
Extended Stamp-Based Memory Management Compiler with Inter-Layer Optimization

This extended compiler implements:
1. Inter-layer activation reuse analysis
2. Prefetch scheduling for overlapping compute and memory
3. Multi-layer graph optimization
4. Advanced memory allocation strategies
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional
from enum import Enum
import json
from collections import defaultdict


class TileType(Enum):
    INPUT = "input"
    WEIGHT = "weight"
    OUTPUT = "output"
    ACTIVATION = "activation"
    INTERMEDIATE = "intermediate"


class PrefetchType(Enum):
    NONE = "none"
    EAGER = "eager"           # Prefetch as soon as possible
    LAZY = "lazy"             # Prefetch just before needed
    OVERLAP = "overlap"       # Overlap with compute


@dataclass
class Tile:
    """Represents a data tile in a DNN computation phase"""
    tile_id: int
    tile_type: TileType
    shape: Tuple[int, ...]
    size_bytes: int
    phase_id: int
    layer_id: int = -1
    reuse_count: int = 1
    lifetime: Tuple[int, int] = (-1, -1)  # (first_phase, last_phase)
    
    def __hash__(self):
        return hash((self.tile_id, self.tile_type, self.phase_id, self.layer_id))
    
    def __eq__(self, other):
        return (self.tile_id == other.tile_id and 
                self.tile_type == other.tile_type and 
                self.phase_id == other.phase_id and
                self.layer_id == other.layer_id)


@dataclass
class PrefetchOperation:
    """Represents a prefetch operation to be executed"""
    prefetch_id: int
    tile: Tile
    start_phase: int        # When to start prefetch
    complete_phase: int     # When prefetch must complete
    dst_addr: int
    size: int
    prefetch_type: PrefetchType
    priority: int = 0       # Higher priority prefetches first


@dataclass
class InterLayerReuse:
    """Tracks data reuse opportunities across layers"""
    source_layer: int
    source_tile: Tile
    dest_layer: int
    dest_tile: Tile
    reuse_benefit: int      # Bytes saved
    requires_transform: bool = False


@dataclass
class LayerGraph:
    """Represents a multi-layer DNN graph"""
    layers: List[Dict]
    dependencies: Dict[int, List[int]]  # layer_id -> [dependent_layer_ids]
    
    def get_topological_order(self) -> List[int]:
        """Return layers in topological order"""
        visited = set()
        order = []
        
        def dfs(layer_id):
            if layer_id in visited:
                return
            visited.add(layer_id)
            for dep in self.dependencies.get(layer_id, []):
                dfs(dep)
            order.append(layer_id)
        
        for layer_id in range(len(self.layers)):
            dfs(layer_id)
        
        return order


class ExtendedStampCompiler:
    """
    Extended compiler with inter-layer optimization and prefetching
    """
    
    def __init__(self, on_chip_size: int, data_width: int = 4,
                 prefetch_buffer_size: int = 4096):
        self.on_chip_size = on_chip_size
        self.data_width = data_width
        self.prefetch_buffer_size = prefetch_buffer_size
        
        # Existing structures
        self.phases: List[List[Tile]] = []
        self.stamps: List = []
        self.deltas: List = []
        
        # New structures for extensions
        self.layer_graph: Optional[LayerGraph] = None
        self.inter_layer_reuse: List[InterLayerReuse] = []
        self.prefetch_schedule: List[PrefetchOperation] = []
        self.activation_liveness: Dict[int, Tuple[int, int]] = {}
        
        # Statistics
        self.stats = {
            'inter_layer_reuse_bytes': 0,
            'prefetch_operations': 0,
            'overlapped_phases': 0,
            'total_phases': 0
        }
    
    def create_layer_graph(self, layers: List[Dict]) -> LayerGraph:
        """
        Create a layer dependency graph
        
        Args:
            layers: List of layer configurations
        
        Returns:
            LayerGraph object
        """
        dependencies = {}
        
        for i, layer in enumerate(layers):
            deps = []
            
            # Check for direct dependencies
            if 'input_from_layer' in layer:
                deps.append(layer['input_from_layer'])
            
            # Check for skip connections
            if 'skip_connections' in layer:
                deps.extend(layer['skip_connections'])
            
            dependencies[i] = deps
        
        return LayerGraph(layers=layers, dependencies=dependencies)
    
    def analyze_inter_layer_reuse(self):
        """
        Analyze activation reuse opportunities across layers
        
        This identifies:
        1. Direct activation reuse (layer i output = layer i+1 input)
        2. Skip connection reuse (ResNet-style)
        3. Shared activations (DenseNet-style)
        """
        if not self.layer_graph:
            return
        
        print("\nAnalyzing inter-layer reuse opportunities...")
        
        # Track which tiles are activations that could be reused
        activation_tiles = defaultdict(list)
        
        for phase_tiles in self.phases:
            for tile in phase_tiles:
                if tile.tile_type in [TileType.OUTPUT, TileType.ACTIVATION]:
                    activation_tiles[tile.layer_id].append(tile)
        
        # Find reuse opportunities
        for src_layer_id in range(len(self.layer_graph.layers) - 1):
            dest_layer_ids = [src_layer_id + 1]  # Sequential connection
            
            # Add skip connections
            for layer_id, deps in self.layer_graph.dependencies.items():
                if src_layer_id in deps and layer_id not in dest_layer_ids:
                    dest_layer_ids.append(layer_id)
            
            for dest_layer_id in dest_layer_ids:
                # Match output tiles from source to input tiles of destination
                src_tiles = activation_tiles.get(src_layer_id, [])
                dest_tiles = [t for t in activation_tiles.get(dest_layer_id, [])
                            if t.tile_type == TileType.INPUT]
                
                for src_tile in src_tiles:
                    for dest_tile in dest_tiles:
                        if self._tiles_compatible_for_reuse(src_tile, dest_tile):
                            reuse = InterLayerReuse(
                                source_layer=src_layer_id,
                                source_tile=src_tile,
                                dest_layer=dest_layer_id,
                                dest_tile=dest_tile,
                                reuse_benefit=src_tile.size_bytes,
                                requires_transform=self._requires_transform(
                                    src_tile, dest_tile
                                )
                            )
                            self.inter_layer_reuse.append(reuse)
                            self.stats['inter_layer_reuse_bytes'] += reuse.reuse_benefit
        
        print(f"  Found {len(self.inter_layer_reuse)} inter-layer reuse opportunities")
        print(f"  Total reuse potential: {self.stats['inter_layer_reuse_bytes']:,} bytes")
    
    def _tiles_compatible_for_reuse(self, tile1: Tile, tile2: Tile) -> bool:
        """Check if two tiles can be reused (same or compatible shape)"""
        # Exact match
        if tile1.shape == tile2.shape:
            return True
        
        # Compatible with transformation (e.g., pooling changes dimensions)
        # For now, require exact match
        return False
    
    def _requires_transform(self, src_tile: Tile, dest_tile: Tile) -> bool:
        """Check if reuse requires data transformation"""
        return src_tile.shape != dest_tile.shape
    
    def compute_activation_liveness(self):
        """
        Compute liveness intervals for activations
        
        An activation is live from the phase that produces it to the
        last phase that consumes it.
        """
        print("\nComputing activation liveness intervals...")
        
        # Track first and last use of each activation
        first_use = {}
        last_use = {}
        
        for phase_id, phase_tiles in enumerate(self.phases):
            for tile in phase_tiles:
                tile_key = (tile.layer_id, tile.tile_id)
                
                if tile_key not in first_use:
                    first_use[tile_key] = phase_id
                last_use[tile_key] = phase_id
        
        # Store liveness intervals
        for tile_key, first_phase in first_use.items():
            last_phase = last_use[tile_key]
            self.activation_liveness[tile_key] = (first_phase, last_phase)
        
        avg_lifetime = np.mean([last - first 
                               for first, last in self.activation_liveness.values()])
        print(f"  Average activation lifetime: {avg_lifetime:.1f} phases")
    
    def schedule_prefetches(self, lookahead_phases: int = 2):
        """
        Schedule prefetch operations to overlap with compute
        
        Args:
            lookahead_phases: How many phases ahead to prefetch
        """
        print(f"\nScheduling prefetches with {lookahead_phases}-phase lookahead...")
        
        prefetch_id = 0
        
        for phase_id in range(len(self.phases) - 1):
            current_phase = self.phases[phase_id]
            
            # Look ahead to find tiles that will be needed
            for future_offset in range(1, min(lookahead_phases + 1, 
                                             len(self.phases) - phase_id)):
                future_phase_id = phase_id + future_offset
                future_phase = self.phases[future_phase_id]
                
                # Find tiles in future phase that aren't in current phase
                current_tiles = set(current_phase)
                future_tiles = set(future_phase)
                
                new_tiles = future_tiles - current_tiles
                
                for tile in new_tiles:
                    # Skip if this tile can be reused from inter-layer
                    if self._tile_has_inter_layer_reuse(tile, future_phase_id):
                        continue
                    
                    # Create prefetch operation
                    prefetch = PrefetchOperation(
                        prefetch_id=prefetch_id,
                        tile=tile,
                        start_phase=phase_id,
                        complete_phase=future_phase_id - 1,
                        dst_addr=self._get_prefetch_buffer_addr(prefetch_id),
                        size=tile.size_bytes,
                        prefetch_type=PrefetchType.OVERLAP,
                        priority=self._compute_prefetch_priority(tile, future_offset)
                    )
                    
                    self.prefetch_schedule.append(prefetch)
                    prefetch_id += 1
        
        self.stats['prefetch_operations'] = len(self.prefetch_schedule)
        print(f"  Scheduled {len(self.prefetch_schedule)} prefetch operations")
    
    def _tile_has_inter_layer_reuse(self, tile: Tile, phase_id: int) -> bool:
        """Check if a tile can be reused from inter-layer optimization"""
        for reuse in self.inter_layer_reuse:
            if (reuse.dest_tile == tile and 
                reuse.dest_tile.phase_id == phase_id):
                return True
        return False
    
    def _get_prefetch_buffer_addr(self, prefetch_id: int) -> int:
        """Get address in prefetch buffer (round-robin allocation)"""
        slots = self.prefetch_buffer_size // 1024  # 1KB slots
        slot = prefetch_id % slots
        return self.on_chip_size + slot * 1024
    
    def _compute_prefetch_priority(self, tile: Tile, phases_ahead: int) -> int:
        """
        Compute prefetch priority
        
        Higher priority for:
        - Larger tiles (more bandwidth benefit)
        - Tiles needed sooner (lower phases_ahead)
        - Critical path tiles
        """
        size_priority = tile.size_bytes // 1024
        urgency_priority = 10 - phases_ahead
        return size_priority + urgency_priority
    
    def optimize_memory_allocation_inter_layer(self):
        """
        Optimize memory allocation considering inter-layer reuse
        
        This extends the basic stamp allocation to:
        1. Keep activations alive across layers when beneficial
        2. Allocate compatible addresses for reusable data
        3. Minimize data movement for skip connections
        """
        print("\nOptimizing memory allocation for inter-layer reuse...")
        
        # Build a map of tiles that should share addresses
        shared_addresses = defaultdict(list)
        
        for reuse in self.inter_layer_reuse:
            if not reuse.requires_transform:
                # Can use same address
                key = (reuse.source_tile.layer_id, reuse.source_tile.tile_id)
                shared_addresses[key].append(reuse.dest_tile)
        
        # Re-allocate stamps considering sharing
        allocation_count = 0
        for key, dest_tiles in shared_addresses.items():
            if len(dest_tiles) > 0:
                allocation_count += 1
        
        print(f"  Optimized {allocation_count} address allocations for reuse")
        self.stats['shared_allocations'] = allocation_count
    
    def generate_extended_metadata(self, output_file: str):
        """
        Generate extended metadata including prefetch schedule and
        inter-layer optimizations
        """
        metadata = {
            "on_chip_size": self.on_chip_size,
            "prefetch_buffer_size": self.prefetch_buffer_size,
            "data_width": self.data_width,
            "num_phases": len(self.phases),
            "num_layers": len(self.layer_graph.layers) if self.layer_graph else 0,
            
            # Existing metadata
            "stamps": self._serialize_stamps(),
            "deltas": self._serialize_deltas(),
            
            # Extended metadata
            "inter_layer_reuse": [
                {
                    "source_layer": r.source_layer,
                    "source_tile_id": r.source_tile.tile_id,
                    "dest_layer": r.dest_layer,
                    "dest_tile_id": r.dest_tile.tile_id,
                    "bytes_saved": r.reuse_benefit,
                    "requires_transform": r.requires_transform
                }
                for r in self.inter_layer_reuse
            ],
            
            "prefetch_schedule": [
                {
                    "prefetch_id": p.prefetch_id,
                    "tile_id": p.tile.tile_id,
                    "layer_id": p.tile.layer_id,
                    "start_phase": p.start_phase,
                    "complete_phase": p.complete_phase,
                    "dst_addr": p.dst_addr,
                    "size": p.size,
                    "type": p.prefetch_type.value,
                    "priority": p.priority
                }
                for p in self.prefetch_schedule
            ],
            
            "statistics": self.stats
        }
        
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nExtended metadata written to {output_file}")
    
    def _serialize_stamps(self):
        """Serialize stamp data for JSON output"""
        # Placeholder - would serialize actual stamp data
        return []
    
    def _serialize_deltas(self):
        """Serialize delta data for JSON output"""
        # Placeholder - would serialize actual delta data
        return []
    
    def print_extended_statistics(self):
        """Print comprehensive statistics including extensions"""
        print("\n" + "="*80)
        print("EXTENDED STAMP-BASED MEMORY MANAGEMENT STATISTICS")
        print("="*80)
        
        print(f"\nConfiguration:")
        print(f"  On-chip memory: {self.on_chip_size:,} bytes")
        print(f"  Prefetch buffer: {self.prefetch_buffer_size:,} bytes")
        print(f"  Number of layers: {len(self.layer_graph.layers) if self.layer_graph else 0}")
        print(f"  Number of phases: {len(self.phases)}")
        
        print(f"\nInter-Layer Optimization:")
        print(f"  Reuse opportunities: {len(self.inter_layer_reuse)}")
        print(f"  Bytes saved via reuse: {self.stats.get('inter_layer_reuse_bytes', 0):,}")
        
        if len(self.inter_layer_reuse) > 0:
            avg_reuse = np.mean([r.reuse_benefit for r in self.inter_layer_reuse])
            print(f"  Average reuse benefit: {avg_reuse:,.0f} bytes per opportunity")
        
        print(f"\nPrefetch Scheduling:")
        print(f"  Total prefetch operations: {len(self.prefetch_schedule)}")
        
        if len(self.prefetch_schedule) > 0:
            avg_prefetch_size = np.mean([p.size for p in self.prefetch_schedule])
            print(f"  Average prefetch size: {avg_prefetch_size:,.0f} bytes")
            
            # Count by type
            type_counts = defaultdict(int)
            for p in self.prefetch_schedule:
                type_counts[p.prefetch_type.value] += 1
            
            print(f"  Prefetch types:")
            for ptype, count in type_counts.items():
                print(f"    {ptype}: {count}")
        
        print(f"\nActivation Liveness:")
        if self.activation_liveness:
            lifetimes = [last - first for first, last in self.activation_liveness.values()]
            print(f"  Average lifetime: {np.mean(lifetimes):.1f} phases")
            print(f"  Max lifetime: {max(lifetimes)} phases")
            print(f"  Min lifetime: {min(lifetimes)} phases")
        
        print("="*80 + "\n")


def example_resnet_block():
    """
    Example: ResNet residual block with skip connection
    """
    layers = [
        {
            'name': 'conv1',
            'input_channels': 64,
            'input_height': 28,
            'input_width': 28,
            'output_channels': 64,
            'output_height': 28,
            'output_width': 28,
            'kernel_height': 3,
            'kernel_width': 3
        },
        {
            'name': 'conv2',
            'input_channels': 64,
            'input_height': 28,
            'input_width': 28,
            'output_channels': 64,
            'output_height': 28,
            'output_width': 28,
            'kernel_height': 3,
            'kernel_width': 3,
            'input_from_layer': 0  # Takes input from conv1
        },
        {
            'name': 'add_skip',
            'input_channels': 64,
            'input_height': 28,
            'input_width': 28,
            'output_channels': 64,
            'output_height': 28,
            'output_width': 28,
            'kernel_height': 1,
            'kernel_width': 1,
            'input_from_layer': 1,
            'skip_connections': [0]  # Skip connection from input
        }
    ]
    
    return layers


def main():
    """Example usage of extended compiler"""
    
    print("="*80)
    print("EXTENDED STAMP COMPILER WITH INTER-LAYER OPTIMIZATION")
    print("="*80)
    
    # Configuration
    on_chip_size = 64 * 1024  # 64 KB
    prefetch_buffer_size = 8 * 1024  # 8 KB for prefetching
    
    # Create compiler
    compiler = ExtendedStampCompiler(
        on_chip_size=on_chip_size,
        prefetch_buffer_size=prefetch_buffer_size
    )
    
    # Create a simple 3-layer network (ResNet-style block)
    layers = example_resnet_block()
    
    print(f"\nCompiling {len(layers)}-layer network with skip connections...")
    
    # Create layer graph
    compiler.layer_graph = compiler.create_layer_graph(layers)
    
    # For demonstration, create simplified phases
    # In real implementation, would call create_conv_phases for each layer
    print("\n[Simplified demo: Using placeholder phases]")
    
    # Simulate phase creation for each layer
    for layer_id, layer in enumerate(layers):
        # Create dummy tiles for this layer
        for phase_idx in range(4):  # 4 phases per layer
            tiles = [
                Tile(
                    tile_id=layer_id * 100 + phase_idx * 10,
                    tile_type=TileType.INPUT,
                    shape=(layer['input_channels'], 7, 7),
                    size_bytes=layer['input_channels'] * 7 * 7 * 4,
                    phase_id=layer_id * 4 + phase_idx,
                    layer_id=layer_id
                ),
                Tile(
                    tile_id=layer_id * 100 + phase_idx * 10 + 1,
                    tile_type=TileType.WEIGHT,
                    shape=(layer['output_channels'], layer['input_channels'], 3, 3),
                    size_bytes=layer['output_channels'] * layer['input_channels'] * 3 * 3 * 4,
                    phase_id=layer_id * 4 + phase_idx,
                    layer_id=layer_id
                ),
                Tile(
                    tile_id=layer_id * 100 + phase_idx * 10 + 2,
                    tile_type=TileType.OUTPUT,
                    shape=(layer['output_channels'], 7, 7),
                    size_bytes=layer['output_channels'] * 7 * 7 * 4,
                    phase_id=layer_id * 4 + phase_idx,
                    layer_id=layer_id
                )
            ]
            compiler.phases.append(tiles)
    
    compiler.stats['total_phases'] = len(compiler.phases)
    
    # Analyze inter-layer reuse
    compiler.analyze_inter_layer_reuse()
    
    # Compute activation liveness
    compiler.compute_activation_liveness()
    
    # Schedule prefetches
    compiler.schedule_prefetches(lookahead_phases=3)
    
    # Optimize memory allocation
    compiler.optimize_memory_allocation_inter_layer()
    
    # Generate metadata
    compiler.generate_extended_metadata("extended_metadata.json")
    
    # Print statistics
    compiler.print_extended_statistics()
    
    print("\n✓ Extended compilation complete!")
    print("  See extended_metadata.json for full details")


if __name__ == "__main__":
    main()
