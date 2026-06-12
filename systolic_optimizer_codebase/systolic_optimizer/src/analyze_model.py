#!/usr/bin/env python3
"""
Model Analysis and Performance Estimation Utility
=================================================
Analyzes TensorFlow models and estimates systolic array performance.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple
import argparse


class ModelAnalyzer:
    """Analyze TensorFlow models for systolic array deployment."""
    
    def __init__(self, array_height: int = 4, array_width: int = 4, clock_freq_mhz: int = 100):
        self.array_height = array_height
        self.array_width = array_width
        self.clock_freq_mhz = clock_freq_mhz
        self.clock_period_ns = 1000.0 / clock_freq_mhz
    
    def analyze_model(self, model: tf.keras.Model) -> Dict:
        """
        Analyze a TensorFlow model.
        
        Returns:
            Dictionary with analysis results
        """
        results = {
            'total_params': model.count_params(),
            'layers': [],
            'total_macs': 0,
            'total_memory_bytes': 0
        }
        
        for layer in model.layers:
            layer_info = self._analyze_layer(layer)
            if layer_info:
                results['layers'].append(layer_info)
                results['total_macs'] += layer_info['macs']
                results['total_memory_bytes'] += layer_info['memory_bytes']
        
        return results
    
    def _analyze_layer(self, layer) -> Dict:
        """Analyze a single layer."""
        
        if isinstance(layer, tf.keras.layers.Conv2D):
            return self._analyze_conv2d(layer)
        elif isinstance(layer, tf.keras.layers.Dense):
            return self._analyze_dense(layer)
        elif isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            return self._analyze_depthwise_conv2d(layer)
        else:
            return None
    
    def _analyze_conv2d(self, layer: tf.keras.layers.Conv2D) -> Dict:
        """Analyze Conv2D layer."""
        
        config = layer.get_config()
        weights = layer.get_weights()
        
        if len(weights) > 0:
            kernel = weights[0]
            kh, kw, c_in, c_out = kernel.shape
        else:
            # Layer not built yet
            kh, kw = config['kernel_size']
            c_in = config['filters']  # Will be wrong, but placeholder
            c_out = config['filters']
        
        # Estimate output size (assumes square input for simplicity)
        # In real case, would need actual input shape
        if hasattr(layer, 'input_shape') and layer.input_shape is not None:
            _, h_in, w_in, _ = layer.input_shape
        else:
            h_in, w_in = 28, 28  # Default assumption
        
        if config['padding'] == 'same':
            h_out = h_in // config['strides'][0]
            w_out = w_in // config['strides'][1]
        else:
            h_out = (h_in - kh) // config['strides'][0] + 1
            w_out = (w_in - kw) // config['strides'][1] + 1
        
        # Calculate MACs
        macs_per_output = kh * kw * c_in
        total_outputs = h_out * w_out * c_out
        total_macs = macs_per_output * total_outputs
        
        # Memory footprint
        weight_bytes = kh * kw * c_in * c_out * 4  # 32-bit
        input_bytes = h_in * w_in * c_in * 4
        output_bytes = h_out * w_out * c_out * 4
        total_memory = weight_bytes + input_bytes + output_bytes
        
        return {
            'name': layer.name,
            'type': 'Conv2D',
            'kernel_size': (kh, kw),
            'input_shape': (h_in, w_in, c_in),
            'output_shape': (h_out, w_out, c_out),
            'macs': total_macs,
            'memory_bytes': total_memory,
            'weight_bytes': weight_bytes,
            'params': kh * kw * c_in * c_out
        }
    
    def _analyze_dense(self, layer: tf.keras.layers.Dense) -> Dict:
        """Analyze Dense layer."""
        
        config = layer.get_config()
        weights = layer.get_weights()
        
        if len(weights) > 0:
            kernel = weights[0]
            n_in, n_out = kernel.shape
        else:
            n_out = config['units']
            n_in = 128  # Placeholder
        
        macs = n_in * n_out
        memory_bytes = (n_in * n_out + n_in + n_out) * 4
        
        return {
            'name': layer.name,
            'type': 'Dense',
            'input_size': n_in,
            'output_size': n_out,
            'macs': macs,
            'memory_bytes': memory_bytes,
            'weight_bytes': n_in * n_out * 4,
            'params': n_in * n_out
        }
    
    def _analyze_depthwise_conv2d(self, layer: tf.keras.layers.DepthwiseConv2D) -> Dict:
        """Analyze DepthwiseConv2D layer."""
        
        config = layer.get_config()
        weights = layer.get_weights()
        
        if len(weights) > 0:
            kernel = weights[0]
            kh, kw, c_in, multiplier = kernel.shape
        else:
            kh, kw = config['kernel_size']
            c_in = 32  # Placeholder
            multiplier = config.get('depth_multiplier', 1)
        
        # Simplified analysis
        macs = kh * kw * c_in * multiplier * 28 * 28  # Assuming 28x28 output
        memory_bytes = kh * kw * c_in * multiplier * 4
        
        return {
            'name': layer.name,
            'type': 'DepthwiseConv2D',
            'kernel_size': (kh, kw),
            'channels': c_in,
            'multiplier': multiplier,
            'macs': macs,
            'memory_bytes': memory_bytes,
            'params': kh * kw * c_in * multiplier
        }
    
    def estimate_performance(self, model: tf.keras.Model) -> Dict:
        """
        Estimate systolic array performance for a model.
        
        Returns:
            Performance estimates
        """
        analysis = self.analyze_model(model)
        
        # Estimate cycles (very simplified)
        # Real implementation would consider tiling, memory bandwidth, etc.
        total_macs = analysis['total_macs']
        
        # Systolic array can perform (H * W) MACs per cycle
        macs_per_cycle = self.array_height * self.array_width
        
        # Ideal computation cycles (no overhead)
        ideal_cycles = total_macs / macs_per_cycle
        
        # Account for overhead (data loading, output writing, etc.)
        # Rough estimate: 2x overhead
        estimated_cycles = ideal_cycles * 2.0
        
        # Time in various units
        time_ns = estimated_cycles * self.clock_period_ns
        time_us = time_ns / 1000
        time_ms = time_us / 1000
        
        # Throughput
        gflops = (total_macs * 2 / 1e9) / (time_ns / 1e9)  # MACs = 2 ops
        
        # Utilization
        utilization = ideal_cycles / estimated_cycles
        
        return {
            'total_macs': total_macs,
            'ideal_cycles': int(ideal_cycles),
            'estimated_cycles': int(estimated_cycles),
            'time_ns': time_ns,
            'time_us': time_us,
            'time_ms': time_ms,
            'gflops': gflops,
            'utilization': utilization,
            'array_size': f"{self.array_height}x{self.array_width}",
            'clock_freq_mhz': self.clock_freq_mhz
        }
    
    def print_analysis(self, model: tf.keras.Model):
        """Print detailed analysis of model."""
        
        analysis = self.analyze_model(model)
        performance = self.estimate_performance(model)
        
        print("=" * 80)
        print("MODEL ANALYSIS FOR SYSTOLIC ARRAY DEPLOYMENT")
        print("=" * 80)
        
        print(f"\nModel: {model.name}")
        print(f"Total Parameters: {analysis['total_params']:,}")
        print(f"Total Memory: {analysis['total_memory_bytes'] / 1024 / 1024:.2f} MB")
        print(f"Total MACs: {analysis['total_macs']:,}")
        
        print("\n" + "=" * 80)
        print("LAYER-BY-LAYER BREAKDOWN")
        print("=" * 80)
        
        print(f"\n{'Layer':<20} {'Type':<15} {'Output Shape':<20} {'MACs':<15} {'Memory (KB)':<12}")
        print("-" * 80)
        
        for layer_info in analysis['layers']:
            name = layer_info['name'][:19]
            layer_type = layer_info['type']
            
            if layer_type == 'Conv2D':
                shape_str = f"{layer_info['output_shape']}"
            elif layer_type == 'Dense':
                shape_str = f"({layer_info['output_size']},)"
            else:
                shape_str = "N/A"
            
            macs = layer_info['macs']
            mem_kb = layer_info['memory_bytes'] / 1024
            
            print(f"{name:<20} {layer_type:<15} {shape_str:<20} {macs:<15,} {mem_kb:<12.1f}")
        
        print("\n" + "=" * 80)
        print("PERFORMANCE ESTIMATES")
        print("=" * 80)
        
        print(f"\nArray Configuration: {performance['array_size']}")
        print(f"Clock Frequency: {performance['clock_freq_mhz']} MHz")
        print(f"\nTotal Operations: {performance['total_macs']:,} MACs")
        print(f"Ideal Cycles: {performance['ideal_cycles']:,}")
        print(f"Estimated Cycles (with overhead): {performance['estimated_cycles']:,}")
        print(f"\nExecution Time:")
        print(f"  {performance['time_ns']:.1f} ns")
        print(f"  {performance['time_us']:.1f} µs")
        print(f"  {performance['time_ms']:.3f} ms")
        print(f"\nThroughput: {performance['gflops']:.2f} GFLOPS")
        print(f"Array Utilization: {performance['utilization']*100:.1f}%")
        
        print("\n" + "=" * 80)
        print("DEPLOYMENT RECOMMENDATIONS")
        print("=" * 80)
        
        # Generate recommendations
        if analysis['total_memory_bytes'] > 16 * 1024 * 1024:
            print("\n⚠ Model requires >16MB memory - consider using scratchpad cache")
        
        if performance['utilization'] < 0.5:
            print("\n⚠ Low array utilization - consider:")
            print("  - Increasing batch size")
            print("  - Optimizing layer dimensions")
            print("  - Using larger systolic array")
        
        max_layer_macs = max([l['macs'] for l in analysis['layers']])
        if max_layer_macs > 1e9:
            print("\n⚠ Large layers detected - consider:")
            print("  - Layer-wise tiling")
            print("  - Output-stationary dataflow")
        
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze TensorFlow models for systolic array deployment'
    )
    parser.add_argument(
        '--model',
        choices=['mnist', 'cifar10', 'mobilenet', 'resnet'],
        default='mnist',
        help='Model to analyze'
    )
    parser.add_argument(
        '--array-height',
        type=int,
        default=4,
        help='Systolic array height'
    )
    parser.add_argument(
        '--array-width',
        type=int,
        default=4,
        help='Systolic array width'
    )
    parser.add_argument(
        '--clock-mhz',
        type=int,
        default=100,
        help='Clock frequency in MHz'
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ModelAnalyzer(
        array_height=args.array_height,
        array_width=args.array_width,
        clock_freq_mhz=args.clock_mhz
    )
    
    # Load model
    if args.model == 'mnist':
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same',
                                  input_shape=(28, 28, 1), name='conv1'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
    
    elif args.model == 'cifar10':
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same',
                                  input_shape=(32, 32, 3), name='conv1'),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3'),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv4'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
    
    elif args.model == 'mobilenet':
        model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=True,
            weights=None
        )
    
    elif args.model == 'resnet':
        model = tf.keras.applications.ResNet50(
            input_shape=(224, 224, 3),
            include_top=True,
            weights=None
        )
    
    # Analyze and print
    analyzer.print_analysis(model)


if __name__ == '__main__':
    main()
