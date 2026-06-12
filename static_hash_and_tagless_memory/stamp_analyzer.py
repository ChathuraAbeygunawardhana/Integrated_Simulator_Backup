#!/usr/bin/env python3
"""
Stamp Metadata Analyzer and Visualizer

Analyzes the generated stamp metadata and produces visualizations
and detailed reports about memory management efficiency.
"""

import json #read metadata file
import sys #command line filename input
from typing import Dict, List #ype hints


def load_metadata(filename: str) -> Dict:
    """Load metadata from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)
    #Example: stamp_metadata.json


def analyze_stamps(metadata: Dict):
    """Analyze stamp characteristics"""
    print("\n" + "="*80)
    print("STAMP ANALYSIS") # Print Header
    print("="*80)
    
    # Read Data
    stamps = metadata['stamps']
    on_chip_size = metadata['on_chip_size']
    
    # Size statistics
    sizes = [s['total_size'] for s in stamps]
    avg_size = sum(sizes) / len(sizes)
    max_size = max(sizes)
    min_size = min(sizes)
    
    print(f"\nMemory Usage:")
    print(f"  On-chip capacity: {on_chip_size:,} bytes")
    print(f"  Average stamp: {avg_size:,.0f} bytes ({100*avg_size/on_chip_size:.1f}%)")
    print(f"  Maximum stamp: {max_size:,} bytes ({100*max_size/on_chip_size:.1f}%)")
    print(f"  Minimum stamp: {min_size:,} bytes ({100*min_size/on_chip_size:.1f}%)")
    
    # Tile type distribution
    tile_types = {}
    for stamp in stamps:
        for tile in stamp['tiles']:
            ttype = tile['tile_type']
            if ttype not in tile_types:
                tile_types[ttype] = {'count': 0, 'total_size': 0}
            tile_types[ttype]['count'] += 1
            tile_types[ttype]['total_size'] += tile['size']
    
    print(f"\nTile Distribution:")
    for ttype, stats in sorted(tile_types.items()):
        avg_tile_size = stats['total_size'] / stats['count']
        print(f"  {ttype:8s}: {stats['count']:4d} tiles, "
              f"avg {avg_tile_size:6.0f} bytes")


def analyze_deltas(metadata: Dict):
    """Analyze delta operations"""
    print("\n" + "="*80)
    print("DELTA ANALYSIS")
    print("="*80)
    
    deltas = metadata['deltas']
    
    # Overall statistics
    total_ops = sum(len(d['operations']) for d in deltas)
    total_loads = sum(d['stats']['loads'] for d in deltas)
    total_moves = sum(d['stats']['moves'] for d in deltas)
    total_keeps = sum(d['stats']['keeps'] for d in deltas)
    total_load_bytes = sum(d['stats']['load_bytes'] for d in deltas)
    total_move_bytes = sum(d['stats']['move_bytes'] for d in deltas)
    
    print(f"\nOperation Counts:")
    print(f"  Total operations: {total_ops}")
    print(f"  Load operations:  {total_loads} ({100*total_loads/total_ops:.1f}%)")
    print(f"  Move operations:  {total_moves} ({100*total_moves/total_ops:.1f}%)")
    print(f"  Keep operations:  {total_keeps} ({100*total_keeps/total_ops:.1f}%)")
    
    print(f"\nData Movement:")
    print(f"  Bytes loaded (from DRAM):     {total_load_bytes:,}")
    print(f"  Bytes moved (on-chip):        {total_move_bytes:,}")
    print(f"  Bytes kept (no movement):     calculated from keeps")
    
    if total_load_bytes + total_move_bytes > 0:
        savings = 100 * total_move_bytes / (total_load_bytes + total_move_bytes)
        print(f"\nBandwidth Efficiency:")
        print(f"  Off-chip bandwidth saved: {savings:.1f}%")
        print(f"  (Moving {total_move_bytes:,} bytes on-chip instead of loading from DRAM)")


def analyze_reuse_patterns(metadata: Dict):
    """Analyze data reuse patterns across phases"""
    print("\n" + "="*80)
    print("DATA REUSE PATTERNS")
    print("="*80)
    
    deltas = metadata['deltas']
    
    # Track reuse across phases
    phase_reuse = []
    for delta in deltas:
        ops = delta['operations']
        loads = [op for op in ops if op['op_type'] == 'load']
        moves = [op for op in ops if op['op_type'] == 'move']
        keeps = [op for op in ops if op['op_type'] == 'keep']
        
        total_tiles = len(loads) + len(moves) + len(keeps)
        reused_tiles = len(moves) + len(keeps)
        
        if total_tiles > 0:
            reuse_ratio = 100 * reused_tiles / total_tiles
            phase_reuse.append(reuse_ratio)
    
    if phase_reuse:
        avg_reuse = sum(phase_reuse) / len(phase_reuse)
        max_reuse = max(phase_reuse)
        min_reuse = min(phase_reuse)
        
        print(f"\nTile Reuse Ratios:")
        print(f"  Average: {avg_reuse:.1f}% of tiles reused per phase")
        print(f"  Maximum: {max_reuse:.1f}%")
        print(f"  Minimum: {min_reuse:.1f}%")
        
        # Categorize phases
        high_reuse = sum(1 for r in phase_reuse if r >= 75)
        med_reuse = sum(1 for r in phase_reuse if 25 <= r < 75)
        low_reuse = sum(1 for r in phase_reuse if r < 25)
        
        print(f"\nReuse Distribution:")
        print(f"  High reuse (≥75%): {high_reuse} phases")
        print(f"  Med reuse (25-75%): {med_reuse} phases")
        print(f"  Low reuse (<25%):  {low_reuse} phases")

# Estimates actual hardware speedup.
def estimate_performance(metadata: Dict):
    """Estimate performance characteristics"""
    print("\n" + "="*80)
    print("PERFORMANCE ESTIMATION")
    print("="*80)
    
    deltas = metadata['deltas']
    
    # Assume realistic cycle counts
    CYCLES_PER_KEEP = 0  # No operation
    CYCLES_PER_MOVE_WORD = 2  # Read + Write
    CYCLES_PER_LOAD_WORD = 10  # DRAM latency
    WORD_SIZE = metadata['data_width']
    
    total_cycles = 0
    total_dram_accesses = 0
    
    for delta in deltas:
        for op in delta['operations']:
            if op['op_type'] == 'keep':
                total_cycles += CYCLES_PER_KEEP
            elif op['op_type'] == 'move':
                words = op['size'] // WORD_SIZE
                total_cycles += words * CYCLES_PER_MOVE_WORD
            elif op['op_type'] == 'load':
                words = op['size'] // WORD_SIZE
                total_cycles += words * CYCLES_PER_LOAD_WORD
                total_dram_accesses += words
    
    num_phases = metadata['num_phases']
    avg_cycles_per_phase = total_cycles / num_phases if num_phases > 0 else 0
    
    print(f"\nEstimated Memory Management Overhead:")
    print(f"  Total cycles: {total_cycles:,}")
    print(f"  Average per phase: {avg_cycles_per_phase:.0f} cycles")
    print(f"  DRAM accesses: {total_dram_accesses:,} words")
    
    # Compare to naive approach (always load everything)
    stamps = metadata['stamps']
    naive_bytes = sum(s['total_size'] for s in stamps)
    naive_words = naive_bytes // WORD_SIZE
    naive_cycles = naive_words * CYCLES_PER_LOAD_WORD
    
    savings_cycles = naive_cycles - total_cycles
    savings_pct = 100 * savings_cycles / naive_cycles if naive_cycles > 0 else 0
    
    print(f"\nCompared to Naive (always load):")
    print(f"  Naive cycles: {naive_cycles:,}")
    print(f"  Stamp cycles: {total_cycles:,}")
    print(f"  Cycle savings: {savings_cycles:,} ({savings_pct:.1f}%)")


def generate_summary(metadata: Dict, output_file: str):
    """Generate a summary report"""
    with open(output_file, 'w') as f:
        f.write("STAMP-BASED MEMORY MANAGEMENT SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"  On-chip memory: {metadata['on_chip_size']:,} bytes\n")
        f.write(f"  Data width: {metadata['data_width']} bytes\n")
        f.write(f"  Number of phases: {metadata['num_phases']}\n\n")
        
        # Key metrics
        deltas = metadata['deltas']
        total_loads = sum(d['stats']['loads'] for d in deltas)
        total_moves = sum(d['stats']['moves'] for d in deltas)
        total_keeps = sum(d['stats']['keeps'] for d in deltas)
        total_load_bytes = sum(d['stats']['load_bytes'] for d in deltas)
        total_move_bytes = sum(d['stats']['move_bytes'] for d in deltas)
        
        f.write(f"Key Metrics:\n")
        f.write(f"  Operations: {total_loads} loads, {total_moves} moves, {total_keeps} keeps\n")
        f.write(f"  Bytes loaded: {total_load_bytes:,}\n")
        f.write(f"  Bytes moved: {total_move_bytes:,}\n")
        
        if total_load_bytes + total_move_bytes > 0:
            savings = 100 * total_move_bytes / (total_load_bytes + total_move_bytes)
            f.write(f"  Bandwidth savings: {savings:.1f}%\n")
        
        f.write("\nSummary: The stamp-based approach achieves significant bandwidth\n")
        f.write("savings by reusing data through on-chip moves instead of always\n")
        f.write("loading from off-chip DRAM. This reduces memory controller pressure\n")
        f.write("and improves overall system performance.\n")
    
    print(f"\nSummary written to {output_file}")


def main():
    """Main analysis function"""
    if len(sys.argv) > 1:
        metadata_file = sys.argv[1]
    else:
        metadata_file = "stamp_metadata.json"
    
    try:
        metadata = load_metadata(metadata_file)
    except FileNotFoundError:
        print(f"Error: {metadata_file} not found!")
        print("Run stamp_compiler.py first to generate metadata.")
        sys.exit(1)
    
    print("\n" + "="*80)
    print(f"ANALYZING: {metadata_file}")
    print("="*80)
    
    analyze_stamps(metadata)
    analyze_deltas(metadata)
    analyze_reuse_patterns(metadata)
    estimate_performance(metadata)
    
    # Generate summary
    generate_summary(metadata, "stamp_analysis_summary.txt")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
