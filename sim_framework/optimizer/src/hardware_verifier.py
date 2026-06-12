#!/usr/bin/env python3
"""
Hardware Verification Against TensorFlow Golden Model
=====================================================
Complete verification flow:
1. Extract golden outputs from TensorFlow model
2. Generate test vectors for hardware
3. Parse hardware outputs
4. Compare with golden model
5. Generate verification report
"""

import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class VerificationStatus(Enum):
    """Verification status for a layer."""
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"
    ERROR = "ERROR"


@dataclass
class FixedPointConfig:
    """Fixed-point representation configuration."""
    total_bits: int = 32
    frac_bits: int = 16
    
    @property
    def int_bits(self):
        return self.total_bits - self.frac_bits
    
    @property
    def scale(self):
        return 2 ** self.frac_bits
    
    @property
    def max_value(self):
        return (2 ** (self.total_bits - 1) - 1) / self.scale
    
    @property
    def min_value(self):
        return -(2 ** (self.total_bits - 1)) / self.scale
    
    def float_to_fixed(self, value: np.ndarray) -> np.ndarray:
        """Convert float to fixed-point."""
        clipped = np.clip(value, self.min_value, self.max_value)
        fixed = np.round(clipped * self.scale).astype(np.int32)
        return fixed
    
    def fixed_to_float(self, value: np.ndarray) -> np.ndarray:
        """Convert fixed-point to float."""
        return value.astype(np.float32) / self.scale


@dataclass
class LayerVerificationResult:
    """Verification result for a single layer."""
    layer_id: int
    layer_name: str
    status: VerificationStatus = VerificationStatus.NOT_RUN
    
    # Shapes
    golden_shape: Tuple = ()
    hardware_shape: Tuple = ()
    
    # Error metrics
    max_abs_error: float = 0.0
    mean_abs_error: float = 0.0
    max_rel_error: float = 0.0
    mean_rel_error: float = 0.0
    
    # Percentiles
    error_p50: float = 0.0
    error_p95: float = 0.0
    error_p99: float = 0.0
    
    # Quantization analysis
    quantization_error: float = 0.0
    
    # Mismatch details
    num_mismatches: int = 0
    total_elements: int = 0
    mismatch_rate: float = 0.0
    
    # Worst mismatches
    worst_mismatches: List[Dict] = field(default_factory=list)
    
    # Timing
    golden_time_ms: float = 0.0
    hardware_time_ms: float = 0.0
    
    def passed(self, tolerance: float) -> bool:
        """Check if verification passed."""
        return self.status == VerificationStatus.PASS and self.max_abs_error < tolerance


class GoldenModelExtractor:
    """
    Extract golden outputs from TensorFlow model.
    """
    
    def __init__(self, fp_config: FixedPointConfig = None):
        self.fp_config = fp_config or FixedPointConfig()
        self.golden_outputs = {}
        self.layer_configs = {}
    
    def extract_from_tf_model(
        self,
        model,  # tf.keras.Model
        test_inputs: np.ndarray,
        save_dir: str = './golden_outputs'
    ):
        """
        Extract golden outputs from TensorFlow model.
        
        Args:
            model: TensorFlow/Keras model
            test_inputs: Test input data (batch, H, W, C)
            save_dir: Directory to save golden outputs
        """
        import tensorflow as tf
        
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("EXTRACTING GOLDEN OUTPUTS FROM TENSORFLOW MODEL")
        print("="*80)
        
        # Get all Conv2D layers
        conv_layers = [
            layer for layer in model.layers 
            if isinstance(layer, tf.keras.layers.Conv2D)
        ]
        
        print(f"\nFound {len(conv_layers)} Conv2D layers")
        print(f"Test input shape: {test_inputs.shape}")
        
        # Extract outputs for each layer
        for i, layer in enumerate(conv_layers):
            print(f"\n[{i}] Extracting from layer: {layer.name}")
            
            # Create intermediate model
            intermediate_model = tf.keras.Model(
                inputs=model.input,
                outputs=layer.output
            )
            
            # Get output
            golden_output = intermediate_model.predict(test_inputs, verbose=0)
            
            print(f"    Golden output shape: {golden_output.shape}")
            print(f"    Output range: [{np.min(golden_output):.6f}, {np.max(golden_output):.6f}]")
            
            # Convert to fixed-point and back (simulate quantization)
            golden_fixed = self.fp_config.float_to_fixed(golden_output)
            golden_quantized = self.fp_config.fixed_to_float(golden_fixed)
            
            # Calculate quantization error
            quant_error = np.abs(golden_output - golden_quantized)
            max_quant_error = np.max(quant_error)
            
            print(f"    Quantization error: {max_quant_error:.6f}")
            
            # Save golden output (quantized version - what hardware should produce)
            golden_file = os.path.join(save_dir, f"layer_{i}_{layer.name}_golden.npy")
            np.save(golden_file, golden_quantized)
            
            # Also save in fixed-point format (for direct hardware comparison)
            golden_fixed_file = os.path.join(save_dir, f"layer_{i}_{layer.name}_golden_fixed.npy")
            np.save(golden_fixed_file, golden_fixed)
            
            # Save as hex for hardware loading
            self._save_as_hex(
                golden_fixed,
                os.path.join(save_dir, f"layer_{i}_{layer.name}_golden.hex")
            )
            
            # Store in memory
            self.golden_outputs[layer.name] = {
                'float': golden_quantized,
                'fixed': golden_fixed,
                'shape': golden_output.shape,
                'quantization_error': max_quant_error
            }
            
            # Store layer config
            weights, biases = layer.get_weights()
            self.layer_configs[layer.name] = {
                'kernel_shape': weights.shape,
                'has_bias': biases is not None,
                'padding': layer.padding,
                'strides': layer.strides
            }
        
        # Save metadata
        metadata = {
            'num_layers': len(conv_layers),
            'layer_names': [l.name for l in conv_layers],
            'input_shape': list(test_inputs.shape),
            'fixed_point_config': {
                'total_bits': self.fp_config.total_bits,
                'frac_bits': self.fp_config.frac_bits
            }
        }
        
        with open(os.path.join(save_dir, 'golden_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Golden outputs saved to {save_dir}")
        return self.golden_outputs
    
    def _save_as_hex(self, data: np.ndarray, filename: str):
        """Save data in hex format for hardware."""
        flat_data = data.flatten()
        
        with open(filename, 'w') as f:
            for i, value in enumerate(flat_data):
                if i % 8 == 0:
                    f.write(f"@{i*4:08x}\n")
                f.write(f"{int(value) & 0xFFFFFFFF:08x}\n")


class HardwareOutputParser:
    """
    Parse hardware simulation outputs.
    """
    
    def __init__(self, fp_config: FixedPointConfig = None):
        self.fp_config = fp_config or FixedPointConfig()
    
    def parse_from_hex(
        self,
        hex_file: str,
        expected_shape: Tuple,
        format: str = 'NHWC'
    ) -> np.ndarray:
        """
        Parse hardware output from hex file.
        
        Args:
            hex_file: Path to hex output file
            expected_shape: Expected output shape
            format: Data format (NHWC or NCHW)
        
        Returns:
            Parsed output array (in float)
        """
        print(f"  Parsing {hex_file}")
        
        # Read hex file
        values = []
        with open(hex_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('@'):
                    continue
                if line:
                    # Parse as signed 32-bit integer
                    val = int(line, 16)
                    # Convert to signed
                    if val >= 2**31:
                        val = val - 2**32
                    values.append(val)
        
        # Convert to numpy array
        values_np = np.array(values, dtype=np.int32)
        
        # Reshape
        total_elements = np.prod(expected_shape)
        if len(values_np) != total_elements:
            print(f"  Warning: Expected {total_elements} elements, got {len(values_np)}")
            # Pad or truncate
            if len(values_np) < total_elements:
                values_np = np.pad(values_np, (0, total_elements - len(values_np)))
            else:
                values_np = values_np[:total_elements]
        
        values_np = values_np.reshape(expected_shape)
        
        # Convert to float
        output_float = self.fp_config.fixed_to_float(values_np)
        
        print(f"    Parsed shape: {output_float.shape}")
        print(f"    Value range: [{np.min(output_float):.6f}, {np.max(output_float):.6f}]")
        
        return output_float
    
    def parse_from_vcd(
        self,
        vcd_file: str,
        signal_name: str,
        expected_shape: Tuple
    ) -> np.ndarray:
        """Parse output from VCD waveform file."""
        # TODO: Implement VCD parsing
        raise NotImplementedError("VCD parsing not yet implemented")
    
    def parse_from_log(
        self,
        log_file: str,
        expected_shape: Tuple
    ) -> np.ndarray:
        """Parse output from simulation log."""
        # Read log file and extract output values
        values = []
        
        with open(log_file, 'r') as f:
            for line in f:
                # Look for output lines (format depends on testbench)
                if 'OUTPUT:' in line or 'output_data' in line:
                    # Extract value
                    parts = line.split()
                    for part in parts:
                        if part.startswith('0x'):
                            val = int(part, 16)
                            if val >= 2**31:
                                val = val - 2**32
                            values.append(val)
        
        values_np = np.array(values, dtype=np.int32)
        
        # Reshape and convert
        total_elements = np.prod(expected_shape)
        if len(values_np) >= total_elements:
            values_np = values_np[:total_elements].reshape(expected_shape)
            return self.fp_config.fixed_to_float(values_np)
        else:
            raise ValueError(f"Not enough values in log: got {len(values_np)}, need {total_elements}")


class HardwareVerifier:
    """
    Main verification class - compares hardware outputs with golden model.
    """
    
    def __init__(
        self,
        fp_config: FixedPointConfig = None,
        tolerance: float = 1e-3
    ):
        self.fp_config = fp_config or FixedPointConfig()
        self.tolerance = tolerance
        self.results = {}
    
    def verify_layer(
        self,
        layer_id: int,
        layer_name: str,
        golden_output: np.ndarray,
        hardware_output: np.ndarray
    ) -> LayerVerificationResult:
        """
        Verify a single layer output.
        
        Args:
            layer_id: Layer identifier
            layer_name: Layer name
            golden_output: Expected output from TensorFlow (float)
            hardware_output: Actual output from hardware (float)
        
        Returns:
            Verification result
        """
        result = LayerVerificationResult(
            layer_id=layer_id,
            layer_name=layer_name,
            golden_shape=golden_output.shape,
            hardware_shape=hardware_output.shape
        )
        
        # Check shapes match
        if golden_output.shape != hardware_output.shape:
            result.status = VerificationStatus.ERROR
            print(f"  ✗ Shape mismatch: golden {golden_output.shape} vs hardware {hardware_output.shape}")
            return result
        
        # Calculate errors
        abs_error = np.abs(golden_output - hardware_output)
        
        # Relative error (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_error = abs_error / (np.abs(golden_output) + 1e-10)
            rel_error = np.nan_to_num(rel_error, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Error statistics
        result.max_abs_error = float(np.max(abs_error))
        result.mean_abs_error = float(np.mean(abs_error))
        result.max_rel_error = float(np.max(rel_error))
        result.mean_rel_error = float(np.mean(rel_error))
        
        # Percentiles
        result.error_p50 = float(np.percentile(abs_error, 50))
        result.error_p95 = float(np.percentile(abs_error, 95))
        result.error_p99 = float(np.percentile(abs_error, 99))
        
        # Count mismatches
        result.total_elements = golden_output.size
        result.num_mismatches = int(np.sum(abs_error > self.tolerance))
        result.mismatch_rate = result.num_mismatches / result.total_elements
        
        # Find worst mismatches
        flat_abs_error = abs_error.flatten()
        flat_golden = golden_output.flatten()
        flat_hardware = hardware_output.flatten()
        
        worst_indices = np.argsort(flat_abs_error)[-10:][::-1]  # Top 10 worst
        
        for idx in worst_indices:
            # Convert flat index to multi-dimensional index
            multi_idx = np.unravel_index(idx, golden_output.shape)
            
            result.worst_mismatches.append({
                'index': [int(i) for i in multi_idx],
                'golden': float(flat_golden[idx]),
                'hardware': float(flat_hardware[idx]),
                'error': float(flat_abs_error[idx])
            })
        
        # Determine pass/fail
        if result.max_abs_error < self.tolerance:
            result.status = VerificationStatus.PASS
        else:
            result.status = VerificationStatus.FAIL
        
        return result
    
    def verify_full_model(
        self,
        golden_dir: str,
        hardware_dir: str,
        layer_names: List[str]
    ) -> Dict[str, LayerVerificationResult]:
        """
        Verify all layers of a model.
        
        Args:
            golden_dir: Directory with golden outputs
            hardware_dir: Directory with hardware outputs
            layer_names: List of layer names to verify
        
        Returns:
            Dictionary of verification results
        """
        print("\n" + "="*80)
        print("HARDWARE VERIFICATION")
        print("="*80)
        
        results = {}
        
        for i, layer_name in enumerate(layer_names):
            print(f"\n[{i}] Verifying layer: {layer_name}")
            
            # Load golden output
            golden_file = os.path.join(golden_dir, f"layer_{i}_{layer_name}_golden.npy")
            if not os.path.exists(golden_file):
                print(f"  ✗ Golden file not found: {golden_file}")
                continue
            
            golden_output = np.load(golden_file)
            print(f"  Golden shape: {golden_output.shape}")
            
            # Load hardware output
            hardware_file = os.path.join(hardware_dir, f"layer_{i}_{layer_name}_output.npy")
            if not os.path.exists(hardware_file):
                # Try hex format
                hardware_hex = os.path.join(hardware_dir, f"layer_{i}_{layer_name}_output.hex")
                if os.path.exists(hardware_hex):
                    parser = HardwareOutputParser(self.fp_config)
                    hardware_output = parser.parse_from_hex(hardware_hex, golden_output.shape)
                else:
                    print(f"  ✗ Hardware file not found: {hardware_file}")
                    continue
            else:
                hardware_output = np.load(hardware_file)
            
            print(f"  Hardware shape: {hardware_output.shape}")
            
            # Verify
            result = self.verify_layer(i, layer_name, golden_output, hardware_output)
            results[layer_name] = result
            
            # Print result
            status_symbol = "✓" if result.status == VerificationStatus.PASS else "✗"
            print(f"  {status_symbol} Status: {result.status.value}")
            print(f"    Max error: {result.max_abs_error:.6f}")
            print(f"    Mean error: {result.mean_abs_error:.6f}")
            print(f"    Mismatch rate: {result.mismatch_rate*100:.2f}%")
        
        self.results = results
        return results
    
    def generate_report(
        self,
        output_file: str = './verification_report.txt'
    ):
        """Generate detailed verification report."""
        
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HARDWARE VERIFICATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Summary
            total = len(self.results)
            passed = sum(1 for r in self.results.values() if r.status == VerificationStatus.PASS)
            failed = sum(1 for r in self.results.values() if r.status == VerificationStatus.FAIL)
            
            f.write(f"Total Layers: {total}\n")
            f.write(f"Passed: {passed}\n")
            f.write(f"Failed: {failed}\n")
            f.write(f"Pass Rate: {passed/total*100:.1f}%\n\n")
            
            # Per-layer details
            f.write("="*80 + "\n")
            f.write("PER-LAYER RESULTS\n")
            f.write("="*80 + "\n\n")
            
            for layer_name, result in self.results.items():
                f.write(f"\nLayer: {layer_name}\n")
                f.write(f"  Status: {result.status.value}\n")
                f.write(f"  Shape: {result.golden_shape}\n")
                f.write(f"\n  Error Metrics:\n")
                f.write(f"    Max Absolute Error:  {result.max_abs_error:.6f}\n")
                f.write(f"    Mean Absolute Error: {result.mean_abs_error:.6f}\n")
                f.write(f"    Max Relative Error:  {result.max_rel_error:.6f}\n")
                f.write(f"    Mean Relative Error: {result.mean_rel_error:.6f}\n")
                f.write(f"\n  Error Distribution:\n")
                f.write(f"    50th percentile: {result.error_p50:.6f}\n")
                f.write(f"    95th percentile: {result.error_p95:.6f}\n")
                f.write(f"    99th percentile: {result.error_p99:.6f}\n")
                f.write(f"\n  Mismatches:\n")
                f.write(f"    Total elements: {result.total_elements}\n")
                f.write(f"    Mismatches: {result.num_mismatches}\n")
                f.write(f"    Mismatch rate: {result.mismatch_rate*100:.2f}%\n")
                
                if result.worst_mismatches:
                    f.write(f"\n  Worst Mismatches:\n")
                    for mm in result.worst_mismatches[:5]:
                        f.write(f"    Index {mm['index']}: "
                               f"golden={mm['golden']:.6f}, "
                               f"hardware={mm['hardware']:.6f}, "
                               f"error={mm['error']:.6f}\n")
        
        print(f"\n✓ Verification report saved to {output_file}")
    
    def generate_json_report(self, output_file: str = './verification_report.json'):
        """Generate JSON verification report."""
        
        report = {
            'summary': {
                'total_layers': len(self.results),
                'passed': sum(1 for r in self.results.values() if r.status == VerificationStatus.PASS),
                'failed': sum(1 for r in self.results.values() if r.status == VerificationStatus.FAIL),
                'tolerance': self.tolerance
            },
            'layers': {}
        }
        
        for layer_name, result in self.results.items():
            report['layers'][layer_name] = {
                'layer_id': result.layer_id,
                'status': result.status.value,
                'shape': list(result.golden_shape),
                'errors': {
                    'max_absolute': result.max_abs_error,
                    'mean_absolute': result.mean_abs_error,
                    'max_relative': result.max_rel_error,
                    'mean_relative': result.mean_rel_error,
                    'p50': result.error_p50,
                    'p95': result.error_p95,
                    'p99': result.error_p99
                },
                'mismatches': {
                    'total_elements': result.total_elements,
                    'num_mismatches': result.num_mismatches,
                    'mismatch_rate': result.mismatch_rate
                },
                'worst_mismatches': result.worst_mismatches[:5]
            }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ JSON report saved to {output_file}")


def demo_verification():
    """Demonstrate verification without actual TensorFlow."""
    
    print("\n" + "="*80)
    print("HARDWARE VERIFICATION DEMO")
    print("="*80)
    
    # Create mock golden and hardware outputs
    os.makedirs('./demo_output/golden', exist_ok=True)
    os.makedirs('./demo_output/hardware', exist_ok=True)
    
    # Simulate 2 layers
    layers = [
        {'id': 0, 'name': 'conv1', 'shape': (1, 28, 28, 16)},
        {'id': 1, 'name': 'conv2', 'shape': (1, 14, 14, 32)}
    ]
    
    fp_config = FixedPointConfig(total_bits=32, frac_bits=16)
    
    print("\n[1] Generating mock golden outputs...")
    for layer in layers:
        # Generate random golden output
        golden = np.random.randn(*layer['shape']).astype(np.float32) * 2.0
        
        # Save
        np.save(f"./demo_output/golden/layer_{layer['id']}_{layer['name']}_golden.npy", golden)
        print(f"  ✓ {layer['name']}: shape {layer['shape']}")
    
    print("\n[2] Generating mock hardware outputs (with small errors)...")
    for layer in layers:
        # Load golden
        golden = np.load(f"./demo_output/golden/layer_{layer['id']}_{layer['name']}_golden.npy")
        
        # Add small random error to simulate hardware
        noise = np.random.randn(*golden.shape).astype(np.float32) * 0.001
        hardware = golden + noise
        
        # Quantize through fixed-point
        hardware_fixed = fp_config.float_to_fixed(hardware)
        hardware_quant = fp_config.fixed_to_float(hardware_fixed)
        
        # Save
        np.save(f"./demo_output/hardware/layer_{layer['id']}_{layer['name']}_output.npy", hardware_quant)
        print(f"  ✓ {layer['name']}: added noise")
    
    print("\n[3] Running verification...")
    verifier = HardwareVerifier(fp_config, tolerance=0.01)
    
    results = verifier.verify_full_model(
        golden_dir='./demo_output/golden',
        hardware_dir='./demo_output/hardware',
        layer_names=[l['name'] for l in layers]
    )
    
    print("\n[4] Generating reports...")
    verifier.generate_report('./demo_output/verification_report.txt')
    verifier.generate_json_report('./demo_output/verification_report.json')
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  ./demo_output/verification_report.txt")
    print("  ./demo_output/verification_report.json")
    print("="*80 + "\n")


if __name__ == '__main__':
    demo_verification()
