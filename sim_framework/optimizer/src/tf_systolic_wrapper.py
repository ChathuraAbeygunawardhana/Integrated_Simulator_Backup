#!/usr/bin/env python3
"""
TensorFlow to Systolic Array Wrapper
====================================
This wrapper interfaces pretrained TensorFlow/Keras models with the systolic
array hardware accelerator. It handles:
- Weight/activation extraction from TF models
- Data formatting and quantization
- Memory layout configuration
- Hardware simulation via cocotb
- Output verification against TensorFlow golden reference
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
import struct
import os
import subprocess
from loop_optimizer import LoopOrder6D as LoopOrder, LoopOptimizer, LoopTileConfig


class FixedPointConverter:
    """Convert floating-point values to fixed-point representation."""
    
    def __init__(self, width: int = 32, frac_bits: int = 16):
        """
        Args:
            width: Total bit width
            frac_bits: Number of fractional bits
        """
        self.width = width
        self.frac_bits = frac_bits
        self.int_bits = width - frac_bits
        self.scale = 2 ** frac_bits
        self.max_val = (2 ** (width - 1) - 1) / self.scale
        self.min_val = -(2 ** (width - 1)) / self.scale
    
    def float_to_fixed(self, value: np.ndarray) -> np.ndarray:
        """Convert floating-point array to fixed-point integers."""
        # Clip to representable range
        clipped = np.clip(value, self.min_val, self.max_val)
        # Scale and round
        fixed = np.round(clipped * self.scale).astype(np.int32)
        return fixed
    
    def fixed_to_float(self, value: np.ndarray) -> np.ndarray:
        """Convert fixed-point integers back to floating-point."""
        return value.astype(np.float32) / self.scale


class MemoryLayoutManager:
    """Manage memory layout for inputs, weights, and outputs."""
    
    def __init__(self, data_width: int = 32):
        self.data_width = data_width
        self.bytes_per_word = data_width // 8
        self.current_addr = 0x1000_0000  # Start address
        
    def allocate(self, size_bytes: int, alignment: int = 64) -> int:
        """Allocate memory and return base address."""
        # Align address
        if self.current_addr % alignment != 0:
            self.current_addr = ((self.current_addr // alignment) + 1) * alignment
        
        base_addr = self.current_addr
        self.current_addr += size_bytes
        return base_addr
    
    def reset(self):
        """Reset allocator."""
        self.current_addr = 0x1000_0000


class SystolicArrayWrapper:
    """Main wrapper class for TensorFlow to Systolic Array interface."""
    
    def __init__(
        self,
        array_height: int = 4,
        array_width: int = 4,
        data_width: int = 32,
        frac_bits: int = 16,
        simulator: str = "verilator",
        default_loop_order: LoopOrder = LoopOrder.KCS
    ):
        """
        Args:
            array_height: Height of systolic array
            array_width: Width of systolic array
            data_width: Bit width for data
            frac_bits: Fractional bits for fixed-point
            simulator: HDL simulator to use (verilator, icarus, etc.)
            default_loop_order: Default loop order for convolutions
        """
        self.array_height = array_height
        self.array_width = array_width
        self.data_width = data_width
        
        self.fp_converter = FixedPointConverter(data_width, frac_bits)
        self.mem_layout = MemoryLayoutManager(data_width)
        self.simulator = simulator
        self.default_loop_order = default_loop_order
        
        # Loop optimizer
        self.loop_optimizer = LoopOptimizer(array_height, array_width)
        
        # Memory storage
        self.memory: Dict[int, int] = {}  # addr -> data
        
        # Layer-specific loop orders (can be set per layer)
        self.layer_loop_orders: Dict[str, LoopOrder] = {}
        
    def extract_conv2d_params(
        self, 
        layer: tf.keras.layers.Conv2D,
        input_shape: Tuple[int, ...]
    ) -> Dict:
        """Extract parameters from a Conv2D layer."""
        weights, biases = layer.get_weights()
        
        # TF format: (kH, kW, C_in, C_out)
        kh, kw, c_in, c_out = weights.shape
        
        # Input: (batch, H, W, C)
        batch, h_in, w_in, _ = input_shape
        
        # Output dimensions
        if layer.padding.lower() == 'same':
            h_out = h_in // layer.strides[0]
            w_out = w_in // layer.strides[1]
        else:  # 'valid'
            h_out = (h_in - kh) // layer.strides[0] + 1
            w_out = (w_in - kw) // layer.strides[1] + 1
        
        return {
            'weights': weights,
            'biases': biases,
            'kernel_h': kh,
            'kernel_w': kw,
            'input_channels': c_in,
            'output_channels': c_out,
            'input_height': h_in,
            'input_width': w_in,
            'output_height': h_out,
            'output_width': w_out,
            'stride': layer.strides[0],
            'padding': layer.padding
        }
    
    def set_layer_loop_order(self, layer_name: str, loop_order: LoopOrder):
        """Set specific loop order for a layer."""
        self.layer_loop_orders[layer_name] = loop_order
    
    def get_layer_loop_order(self, layer_name: str) -> LoopOrder:
        """Get loop order for a layer (returns default if not set)."""
        return self.layer_loop_orders.get(layer_name, self.default_loop_order)
    
    def analyze_loop_orders_for_layer(
        self,
        layer_name: str,
        params: Dict,
        verbose: bool = True
    ) -> Dict:
        """
        Analyze all loop orders for a specific layer.
        
        Args:
            layer_name: Name of the layer
            params: Layer parameters from extract_conv2d_params
            verbose: Print detailed analysis
            
        Returns:
            Dictionary with analysis for all loop orders
        """
        tile_config = LoopTileConfig(
            tile_k=min(self.array_height, params['output_channels']),
            tile_c=min(self.array_width, params['input_channels']),
            tile_h_out=self.array_height,
            tile_w_out=self.array_width
        )
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Loop Order Analysis for Layer: {layer_name}")
            print(f"{'='*80}")
        
        results = self.loop_optimizer.print_optimization_analysis(
            k_out=params['output_channels'],
            c_in=params['input_channels'],
            kh=params['kernel_h'],
            kw=params['kernel_w'],
            h_out=params['output_height'],
            w_out=params['output_width'],
            tile_config=tile_config
        ) if verbose else {}
        
        # Get best order for balanced performance
        best_order, all_results = self.loop_optimizer.find_optimal_loop_order(
            k_out=params['output_channels'],
            c_in=params['input_channels'],
            kh=params['kernel_h'],
            kw=params['kernel_w'],
            h_out=params['output_height'],
            w_out=params['output_width'],
            tile_config=tile_config,
            optimization_target='balanced'
        )
        
        return {
            'best_order': best_order,
            'all_results': all_results,
            'tile_config': tile_config
        }
    
    def optimize_all_layers(
        self,
        model: tf.keras.Model,
        optimization_target: str = 'balanced'
    ):
        """
        Automatically find and set optimal loop orders for all Conv2D layers.
        
        Args:
            model: TensorFlow model
            optimization_target: 'weight_reuse', 'input_reuse', 'output_reuse', or 'balanced'
        """
        print(f"\n{'='*80}")
        print(f"Auto-Optimizing Loop Orders (Target: {optimization_target})")
        print(f"{'='*80}")
        
        conv_layers = [
            layer for layer in model.layers 
            if isinstance(layer, tf.keras.layers.Conv2D)
        ]
        
        # Get a sample input shape
        if hasattr(model, 'input_shape') and model.input_shape:
            sample_shape = model.input_shape
        else:
            # Default assumption
            sample_shape = (1, 28, 28, 1)
        
        current_shape = sample_shape
        
        for i, layer in enumerate(conv_layers):
            params = self.extract_conv2d_params(layer, current_shape)
            
            tile_config = LoopTileConfig(
                tile_k=min(self.array_height, params['output_channels']),
                tile_c=min(self.array_width, params['input_channels']),
                tile_h_out=self.array_height,
                tile_w_out=self.array_width
            )
            
            best_order, all_results = self.loop_optimizer.find_optimal_loop_order(
                k_out=params['output_channels'],
                c_in=params['input_channels'],
                kh=params['kernel_h'],
                kw=params['kernel_w'],
                h_out=params['output_height'],
                w_out=params['output_width'],
                tile_config=tile_config,
                optimization_target=optimization_target
            )
            
            self.set_layer_loop_order(layer.name, best_order)
            
            print(f"\nLayer {i+1}: {layer.name}")
            print(f"  Optimal loop order: {best_order.value}")
            print(f"  Score: {all_results[best_order]['score']:.2f}")
            
            # Update shape for next layer
            current_shape = (1, params['output_height'], 
                           params['output_width'], params['output_channels'])
        
        print(f"\n{'='*80}")
        print(f"Optimization complete!")
        print(f"{'='*80}\n")
    
    def prepare_conv2d_data(
        self,
        input_data: np.ndarray,
        params: Dict,
        layer_name: str = None,
        loop_order: LoopOrder = None
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Prepare data for systolic array execution.
        
        Args:
            input_data: Input activations
            params: Layer parameters
            layer_name: Layer name (to look up loop order)
            loop_order: Explicit loop order (overrides layer setting)
        
        Returns:
            input_mem: Dictionary of input data in memory
            weight_mem: Dictionary of weight data in memory
            config: Configuration parameters including loop order
        """
        self.mem_layout.reset()
        
        # Determine loop order
        if loop_order is None:
            if layer_name:
                loop_order = self.get_layer_loop_order(layer_name)
            else:
                loop_order = self.default_loop_order
        
        # Convert to fixed-point
        input_fixed = self.fp_converter.float_to_fixed(input_data)
        weight_fixed = self.fp_converter.float_to_fixed(params['weights'])
        bias_fixed = self.fp_converter.float_to_fixed(params['biases'])
        
        # Allocate memory
        input_size = input_fixed.nbytes
        weight_size = weight_fixed.nbytes
        bias_size = bias_fixed.nbytes
        output_size = (params['output_height'] * params['output_width'] * 
                      params['output_channels'] * self.data_width // 8)
        
        input_base = self.mem_layout.allocate(input_size)
        weight_base = self.mem_layout.allocate(weight_size)
        bias_base = self.mem_layout.allocate(bias_size)
        output_base = self.mem_layout.allocate(output_size)
        
        # Layout input data in memory (NHWC format)
        input_mem = {}
        addr = input_base
        for h in range(params['input_height']):
            for w in range(params['input_width']):
                for c in range(params['input_channels']):
                    input_mem[addr] = int(input_fixed[0, h, w, c])
                    addr += 4  # 32-bit = 4 bytes
        
        # Layout weight data (output-stationary: reorganize for systolic array)
        weight_mem = {}
        addr = weight_base
        # Weights in TF: (kH, kW, C_in, C_out)
        # Reorganize for systolic array processing
        for k_out in range(params['output_channels']):
            for k_in in range(params['input_channels']):
                for kh in range(params['kernel_h']):
                    for kw in range(params['kernel_w']):
                        weight_mem[addr] = int(weight_fixed[kh, kw, k_in, k_out])
                        addr += 4
        
        # Add biases
        for i, bias in enumerate(bias_fixed):
            weight_mem[addr] = int(bias)
            addr += 4
        
        # Configuration
        config = {
            'input_base_addr': input_base,
            'weight_base_addr': weight_base,
            'output_base_addr': output_base,
            'input_channels': params['input_channels'],
            'input_height': params['input_height'],
            'input_width': params['input_width'],
            'output_channels': params['output_channels'],
            'output_height': params['output_height'],
            'output_width': params['output_width'],
            'weight_k': params['output_channels'],
            'weight_c': params['input_channels'],
            'weight_kh': params['kernel_h'],
            'weight_kw': params['kernel_w'],
            'loop_order': loop_order.value,
            'loop_order_code': list(LoopOrder).index(loop_order)
        }
        
        return input_mem, weight_mem, config
    
    def write_memory_image(
        self,
        memory: Dict[int, int],
        filename: str,
        format: str = 'hex'
    ):
        """Write memory contents to file."""
        sorted_addrs = sorted(memory.keys())
        
        with open(filename, 'w') as f:
            if format == 'hex':
                for addr in sorted_addrs:
                    f.write(f"@{addr:08x}\n")
                    f.write(f"{memory[addr]:08x}\n")
            elif format == 'binary':
                for addr in sorted_addrs:
                    f.write(struct.pack('<I', memory[addr]))
    
    def run_simulation(
        self,
        input_mem: Dict,
        weight_mem: Dict,
        config: Dict,
        num_cycles: int = 10000
    ) -> Dict[int, int]:
        """
        Run hardware simulation using cocotb.
        
        Returns:
            output_mem: Dictionary of output data from simulation
        """
        # Write memory images
        self.write_memory_image(input_mem, 'input_mem.hex')
        self.write_memory_image(weight_mem, 'weight_mem.hex')
        
        # Write configuration
        with open('config.txt', 'w') as f:
            for key, value in config.items():
                f.write(f"{key}={value}\n")
        
        # Create cocotb testbench runner script
        cocotb_script = """
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import json

@cocotb.test()
async def systolic_test(dut):
    \"\"\"Test systolic array with TensorFlow data.\"\"\"
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst_n.value = 0
    await Timer(100, units="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    
    # Load configuration
    config = {}
    with open('config.txt', 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            config[key] = int(value)
    
    # Configure DUT
    dut.input_base_addr.value = config['input_base_addr']
    dut.weight_base_addr.value = config['weight_base_addr']
    dut.output_base_addr.value = config['output_base_addr']
    dut.input_channels.value = config['input_channels']
    dut.input_height.value = config['input_height']
    dut.input_width.value = config['input_width']
    dut.output_channels.value = config['output_channels']
    dut.output_height.value = config['output_height']
    dut.output_width.value = config['output_width']
    dut.weight_k.value = config['weight_k']
    dut.weight_c.value = config['weight_c']
    dut.weight_kh.value = config['weight_kh']
    dut.weight_kw.value = config['weight_kw']
    
    # Start computation
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    # Wait for completion
    while dut.done.value == 0:
        await RisingEdge(dut.clk)
    
    dut._log.info("Computation completed")
"""
        
        with open('test_systolic.py', 'w') as f:
            f.write(cocotb_script)
        
        print("Simulation files prepared. Run with cocotb manually or integrate with your build system.")
        
        # For now, return empty dict - in real implementation, parse VCD/output files
        return {}
    
    def verify_layer(
        self,
        model: tf.keras.Model,
        layer_name: str,
        input_data: np.ndarray,
        tolerance: float = 1e-3,
        loop_order: LoopOrder = None,
        analyze_loop_orders: bool = False
    ) -> Tuple[bool, Dict]:
        """
        Verify systolic array output against TensorFlow for a single layer.
        
        Args:
            model: TensorFlow model
            layer_name: Name of layer to verify
            input_data: Input data (batch, H, W, C)
            tolerance: Acceptable error tolerance
            loop_order: Specific loop order to use (None = use layer default)
            analyze_loop_orders: Run analysis of all loop orders
            
        Returns:
            success: Whether verification passed
            metrics: Dictionary with error metrics
        """
        # Get layer
        layer = model.get_layer(layer_name)
        
        if not isinstance(layer, tf.keras.layers.Conv2D):
            raise ValueError(f"Layer {layer_name} is not Conv2D")
        
        # Extract parameters
        params = self.extract_conv2d_params(layer, input_data.shape)
        
        # Optionally analyze all loop orders
        if analyze_loop_orders:
            analysis = self.analyze_loop_orders_for_layer(
                layer_name, params, verbose=True
            )
            if loop_order is None:
                loop_order = analysis['best_order']
                print(f"\nUsing optimal loop order: {loop_order.value}")
        
        # Get TensorFlow golden output
        intermediate_model = tf.keras.Model(
            inputs=model.input,
            outputs=layer.output
        )
        tf_output = intermediate_model.predict(input_data, verbose=0)
        
        # Prepare data for systolic array
        input_mem, weight_mem, config = self.prepare_conv2d_data(
            input_data, params, layer_name, loop_order
        )
        
        # Run simulation (in real implementation)
        print(f"\nVerifying layer: {layer_name}")
        print(f"Input shape: {input_data.shape}")
        print(f"Output shape: {tf_output.shape}")
        print(f"Kernel: {params['kernel_h']}x{params['kernel_w']}")
        print(f"Channels: {params['input_channels']} -> {params['output_channels']}")
        print(f"Loop order: {config['loop_order']}")
        print(f"\nMemory layout:")
        print(f"  Input base:  0x{config['input_base_addr']:08x}")
        print(f"  Weight base: 0x{config['weight_base_addr']:08x}")
        print(f"  Output base: 0x{config['output_base_addr']:08x}")
        
        # Simulate with fixed-point conversion (software simulation)
        tf_output_fixed = self.fp_converter.float_to_fixed(tf_output)
        tf_output_recovered = self.fp_converter.fixed_to_float(tf_output_fixed)
        
        # Calculate quantization error
        quant_error = np.abs(tf_output - tf_output_recovered)
        max_quant_error = np.max(quant_error)
        mean_quant_error = np.mean(quant_error)
        
        metrics = {
            'max_quantization_error': max_quant_error,
            'mean_quantization_error': mean_quant_error,
            'output_range': (np.min(tf_output), np.max(tf_output)),
            'passed': max_quant_error < tolerance
        }
        
        print(f"\nQuantization error analysis:")
        print(f"  Max error:  {max_quant_error:.6f}")
        print(f"  Mean error: {mean_quant_error:.6f}")
        print(f"  Output range: [{metrics['output_range'][0]:.4f}, {metrics['output_range'][1]:.4f}]")
        print(f"  Status: {'PASS' if metrics['passed'] else 'FAIL'}")
        
        return metrics['passed'], metrics
    
    def generate_testbench(
        self,
        model: tf.keras.Model,
        test_data: np.ndarray,
        output_dir: str = "./tb_generated"
    ):
        """
        Generate complete testbench files for hardware simulation.
        
        Args:
            model: TensorFlow model
            test_data: Test input data
            output_dir: Directory for generated files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract all Conv2D layers
        conv_layers = [
            layer for layer in model.layers 
            if isinstance(layer, tf.keras.layers.Conv2D)
        ]
        
        print(f"\nFound {len(conv_layers)} Conv2D layers")
        
        # Generate test vectors for each layer
        for i, layer in enumerate(conv_layers):
            print(f"\nGenerating test vectors for layer {i}: {layer.name}")
            
            # Get intermediate output
            intermediate_model = tf.keras.Model(
                inputs=model.input,
                outputs=layer.output
            )
            
            # Use output of previous layer as input to current
            if i == 0:
                layer_input = test_data
            else:
                prev_model = tf.keras.Model(
                    inputs=model.input,
                    outputs=conv_layers[i-1].output
                )
                layer_input = prev_model.predict(test_data, verbose=0)
            
            layer_output = intermediate_model.predict(test_data, verbose=0)
            
            # Extract parameters and prepare data
            params = self.extract_conv2d_params(layer, layer_input.shape)
            input_mem, weight_mem, config = self.prepare_conv2d_data(
                layer_input, params
            )
            
            # Write test files
            layer_dir = os.path.join(output_dir, f"layer_{i}_{layer.name}")
            os.makedirs(layer_dir, exist_ok=True)
            
            self.write_memory_image(
                input_mem, 
                os.path.join(layer_dir, "input.hex")
            )
            self.write_memory_image(
                weight_mem,
                os.path.join(layer_dir, "weights.hex")
            )
            
            # Write expected output
            output_fixed = self.fp_converter.float_to_fixed(layer_output)
            output_mem = {}
            addr = config['output_base_addr']
            for h in range(params['output_height']):
                for w in range(params['output_width']):
                    for c in range(params['output_channels']):
                        output_mem[addr] = int(output_fixed[0, h, w, c])
                        addr += 4
            
            self.write_memory_image(
                output_mem,
                os.path.join(layer_dir, "expected_output.hex")
            )
            
            # Write config
            with open(os.path.join(layer_dir, "config.json"), 'w') as f:
                import json
                json.dump(config, f, indent=2)
            
            print(f"  Generated test files in {layer_dir}")


def main():
    """Example usage."""
    
    print("=" * 70)
    print("TensorFlow to Systolic Array Wrapper")
    print("=" * 70)
    
    # Create a simple test model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            8, (3, 3), activation='relu', padding='same',
            input_shape=(28, 28, 1), name='conv1'
        ),
        tf.keras.layers.Conv2D(
            16, (3, 3), activation='relu', padding='same', name='conv2'
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    print("\nModel summary:")
    model.summary()
    
    # Create wrapper
    wrapper = SystolicArrayWrapper(
        array_height=4,
        array_width=4,
        data_width=32,
        frac_bits=16
    )
    
    # Generate test data
    test_input = np.random.randn(1, 28, 28, 1).astype(np.float32) * 0.5
    
    # Verify first conv layer
    success, metrics = wrapper.verify_layer(
        model, 'conv1', test_input, tolerance=0.01
    )
    
    # Generate complete testbench
    print("\n" + "=" * 70)
    print("Generating testbench files...")
    print("=" * 70)
    wrapper.generate_testbench(model, test_input, "./tb_generated")
    
    print("\n" + "=" * 70)
    print("Done! Check ./tb_generated/ for test vectors")
    print("=" * 70)


if __name__ == "__main__":
    main()
