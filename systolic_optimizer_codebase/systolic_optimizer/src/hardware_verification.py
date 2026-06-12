#!/usr/bin/env python3
"""
Hardware Verification System with Golden Model Comparison
=========================================================
Compares systolic array hardware outputs against TensorFlow golden models.

Flow:
1. Extract golden outputs from TensorFlow model
2. Convert to fixed-point test vectors
3. Run hardware simulation (or read actual HW outputs)
4. Compare outputs with configurable tolerances
5. Generate detailed verification report
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import os
from pathlib import Path


@dataclass
class FixedPointConfig:
    """Fixed-point representation configuration."""
    total_bits: int = 32
    frac_bits: int = 16
    signed: bool = True
    
    @property
    def int_bits(self):
        return self.total_bits - self.frac_bits - (1 if self.signed else 0)
    
    @property
    def scale(self):
        return 2 ** self.frac_bits
    
    @property
    def max_value(self):
        if self.signed:
            return (2 ** (self.total_bits - 1) - 1) / self.scale
        else:
            return (2 ** self.total_bits - 1) / self.scale
    
    @property
    def min_value(self):
        if self.signed:
            return -(2 ** (self.total_bits - 1)) / self.scale
        else:
            return 0.0
    
    def float_to_fixed(self, value: np.ndarray) -> np.ndarray:
        """Convert floating-point to fixed-point."""
        clipped = np.clip(value, self.min_value, self.max_value)
        if self.signed:
            fixed = np.round(clipped * self.scale).astype(np.int32)
        else:
            fixed = np.round(clipped * self.scale).astype(np.uint32)
        return fixed
    
    def fixed_to_float(self, value: np.ndarray) -> np.ndarray:
        """Convert fixed-point to floating-point."""
        if self.signed:
            return value.astype(np.float32) / self.scale
        else:
            return value.astype(np.float32) / self.scale


@dataclass
class VerificationConfig:
    """Configuration for verification."""
    # Tolerance settings
    abs_tolerance: float = 1e-3
    rel_tolerance: float = 1e-2
    max_outliers_percent: float = 1.0  # Allow 1% outliers
    
    # Fixed-point config
    fp_config: FixedPointConfig = field(default_factory=FixedPointConfig)
    
    # Reporting
    verbose: bool = True
    save_mismatches: bool = True
    max_mismatches_to_show: int = 10


@dataclass
class LayerVerificationResult:
    """Verification result for a single layer."""
    layer_name: str
    layer_id: int
    
    # Golden model outputs
    golden_output: np.ndarray
    golden_output_fixed: np.ndarray
    
    # Hardware outputs
    hw_output_fixed: np.ndarray
    hw_output_float: np.ndarray
    
    # Comparison metrics
    passed: bool
    max_abs_error: float
    mean_abs_error: float
    max_rel_error: float
    mean_rel_error: float
    rmse: float
    num_mismatches: int
    mismatch_percentage: float
    
    # Error breakdown
    quantization_error: float
    computation_error: float
    
    # Mismatch locations
    mismatch_indices: List[Tuple[int, ...]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'layer_name': self.layer_name,
            'layer_id': self.layer_id,
            'passed': self.passed,
            'metrics': {
                'max_abs_error': float(self.max_abs_error),
                'mean_abs_error': float(self.mean_abs_error),
                'max_rel_error': float(self.max_rel_error),
                'mean_rel_error': float(self.mean_rel_error),
                'rmse': float(self.rmse),
                'num_mismatches': self.num_mismatches,
                'mismatch_percentage': float(self.mismatch_percentage)
            },
            'error_breakdown': {
                'quantization_error': float(self.quantization_error),
                'computation_error': float(self.computation_error)
            },
            'shape': list(self.golden_output.shape),
            'value_range': {
                'golden_min': float(np.min(self.golden_output)),
                'golden_max': float(np.max(self.golden_output)),
                'hw_min': float(np.min(self.hw_output_float)),
                'hw_max': float(np.max(self.hw_output_float))
            }
        }


class GoldenModelExtractor:
    """Extract golden outputs from TensorFlow model."""
    
    def __init__(self, fp_config: FixedPointConfig = None):
        self.fp_config = fp_config or FixedPointConfig()
    
    def extract_layer_outputs(
        self,
        model,  # tf.keras.Model
        input_data: np.ndarray,
        layer_names: List[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract outputs from specified layers.
        
        Args:
            model: TensorFlow/Keras model
            input_data: Input data (batch, H, W, C)
            layer_names: List of layer names to extract (None = all Conv2D)
        
        Returns:
            Dictionary mapping layer_name -> output_array
        """
        import tensorflow as tf
        
        if layer_names is None:
            # Extract all Conv2D layers
            layer_names = [
                layer.name for layer in model.layers
                if isinstance(layer, tf.keras.layers.Conv2D)
            ]
        
        golden_outputs = {}
        
        for layer_name in layer_names:
            try:
                # Create intermediate model
                layer = model.get_layer(layer_name)
                intermediate_model = tf.keras.Model(
                    inputs=model.input,
                    outputs=layer.output
                )
                
                # Get output
                output = intermediate_model.predict(input_data, verbose=0)
                golden_outputs[layer_name] = output
                
                print(f"  Extracted {layer_name}: shape {output.shape}, "
                      f"range [{np.min(output):.4f}, {np.max(output):.4f}]")
                
            except Exception as e:
                print(f"  Warning: Could not extract {layer_name}: {e}")
        
        return golden_outputs
    
    def save_golden_outputs(
        self,
        golden_outputs: Dict[str, np.ndarray],
        output_dir: str
    ):
        """Save golden outputs to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        for layer_name, output in golden_outputs.items():
            # Save as numpy
            np_path = os.path.join(output_dir, f"{layer_name}_golden.npy")
            np.save(np_path, output)
            
            # Save fixed-point version
            output_fixed = self.fp_config.float_to_fixed(output)
            np_fixed_path = os.path.join(output_dir, f"{layer_name}_golden_fixed.npy")
            np.save(np_fixed_path, output_fixed)
            
            # Save as hex (for hardware testbench)
            hex_path = os.path.join(output_dir, f"{layer_name}_golden.hex")
            self._save_as_hex(output_fixed, hex_path)
            
            print(f"  Saved {layer_name} golden outputs")
    
    def _save_as_hex(self, data: np.ndarray, filepath: str):
        """Save fixed-point data as hex file."""
        flat_data = data.flatten()
        
        with open(filepath, 'w') as f:
            addr = 0
            for value in flat_data:
                f.write(f"@{addr:08x}\n")
                # Handle signed integers - convert to unsigned for hex representation
                value_int = int(value)
                if value_int < 0:
                    value_uint = (1 << 32) + value_int  # Two's complement
                else:
                    value_uint = value_int
                f.write(f"{value_uint & 0xFFFFFFFF:08x}\n")
                addr += 4


class HardwareOutputLoader:
    """Load and parse hardware simulation outputs."""
    
    def __init__(self, fp_config: FixedPointConfig = None):
        self.fp_config = fp_config or FixedPointConfig()
    
    def load_from_hex(
        self,
        filepath: str,
        shape: Tuple[int, ...]
    ) -> np.ndarray:
        """Load hardware output from hex file."""
        data = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith('@'):
                    # Address line, skip
                    i += 1
                    if i < len(lines):
                        # Data line
                        value_hex = lines[i].strip()
                        # Convert from hex to signed int32
                        value = int(value_hex, 16)
                        if value >= 2**31:
                            value -= 2**32
                        data.append(value)
                        i += 1
                else:
                    i += 1
        
        # Reshape to expected shape
        data_array = np.array(data, dtype=np.int32)
        
        # Handle case where we read more data than needed
        total_elements = np.prod(shape)
        data_array = data_array[:total_elements]
        
        return data_array.reshape(shape)
    
    def load_from_npy(
        self,
        filepath: str
    ) -> np.ndarray:
        """Load hardware output from numpy file."""
        return np.load(filepath)
    
    def load_from_vcd(
        self,
        vcd_filepath: str,
        signal_name: str,
        shape: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Load hardware output from VCD file (from simulation).
        
        Note: This is a placeholder. Real implementation would use
        a VCD parser library like 'vcd' or 'pyDigitalWaveTools'.
        """
        # TODO: Implement VCD parsing
        raise NotImplementedError("VCD parsing not yet implemented")


class OutputComparator:
    """Compare golden and hardware outputs."""
    
    def __init__(self, config: VerificationConfig = None):
        self.config = config or VerificationConfig()
        self.fp_config = self.config.fp_config
    
    def compare_outputs(
        self,
        layer_name: str,
        layer_id: int,
        golden_output: np.ndarray,
        hw_output_fixed: np.ndarray
    ) -> LayerVerificationResult:
        """
        Compare golden and hardware outputs.
        
        Args:
            layer_name: Name of the layer
            layer_id: Layer ID
            golden_output: Golden output (float)
            hw_output_fixed: Hardware output (fixed-point int)
        
        Returns:
            LayerVerificationResult
        """
        # Convert golden to fixed-point
        golden_output_fixed = self.fp_config.float_to_fixed(golden_output)
        
        # Convert hardware output to float
        hw_output_float = self.fp_config.fixed_to_float(hw_output_fixed)
        
        # Also convert golden fixed back to float (for quantization error)
        golden_output_requant = self.fp_config.fixed_to_float(golden_output_fixed)
        
        # Ensure shapes match
        if golden_output.shape != hw_output_float.shape:
            raise ValueError(
                f"Shape mismatch: golden {golden_output.shape} vs "
                f"hw {hw_output_float.shape}"
            )
        
        # Calculate errors
        abs_error = np.abs(golden_output - hw_output_float)
        rel_error = np.abs((golden_output - hw_output_float) / (np.abs(golden_output) + 1e-10))
        
        # Quantization error (inherent to fixed-point)
        quant_error = np.abs(golden_output - golden_output_requant)
        quantization_error = np.mean(quant_error)
        
        # Computation error (hardware vs golden fixed-point)
        comp_error = np.abs(golden_output_fixed - hw_output_fixed)
        computation_error = np.mean(self.fp_config.fixed_to_float(comp_error))
        
        # Find mismatches
        abs_threshold = self.config.abs_tolerance
        rel_threshold = self.config.rel_tolerance
        
        mismatches = (abs_error > abs_threshold) & (rel_error > rel_threshold)
        mismatch_indices = np.argwhere(mismatches)
        
        num_mismatches = len(mismatch_indices)
        total_elements = golden_output.size
        mismatch_percentage = (num_mismatches / total_elements) * 100
        
        # Determine if passed
        passed = mismatch_percentage <= self.config.max_outliers_percent
        
        # Create result
        result = LayerVerificationResult(
            layer_name=layer_name,
            layer_id=layer_id,
            golden_output=golden_output,
            golden_output_fixed=golden_output_fixed,
            hw_output_fixed=hw_output_fixed,
            hw_output_float=hw_output_float,
            passed=passed,
            max_abs_error=float(np.max(abs_error)),
            mean_abs_error=float(np.mean(abs_error)),
            max_rel_error=float(np.max(rel_error)),
            mean_rel_error=float(np.mean(rel_error)),
            rmse=float(np.sqrt(np.mean(abs_error ** 2))),
            num_mismatches=num_mismatches,
            mismatch_percentage=mismatch_percentage,
            quantization_error=quantization_error,
            computation_error=computation_error,
            mismatch_indices=[tuple(idx) for idx in mismatch_indices[:self.config.max_mismatches_to_show]]
        )
        
        return result
    
    def print_result(self, result: LayerVerificationResult):
        """Print verification result."""
        status = "✓ PASS" if result.passed else "✗ FAIL"
        
        print(f"\n{'='*80}")
        print(f"Layer {result.layer_id}: {result.layer_name} - {status}")
        print(f"{'='*80}")
        
        print(f"\nShape: {result.golden_output.shape}")
        print(f"Total elements: {result.golden_output.size:,}")
        
        print(f"\nValue Ranges:")
        print(f"  Golden (float): [{np.min(result.golden_output):.6f}, {np.max(result.golden_output):.6f}]")
        print(f"  HW (float):     [{np.min(result.hw_output_float):.6f}, {np.max(result.hw_output_float):.6f}]")
        
        print(f"\nError Metrics:")
        print(f"  Max absolute error:  {result.max_abs_error:.6f}")
        print(f"  Mean absolute error: {result.mean_abs_error:.6f}")
        print(f"  Max relative error:  {result.max_rel_error:.4%}")
        print(f"  Mean relative error: {result.mean_rel_error:.4%}")
        print(f"  RMSE:                {result.rmse:.6f}")
        
        print(f"\nError Breakdown:")
        print(f"  Quantization error:  {result.quantization_error:.6f}")
        print(f"  Computation error:   {result.computation_error:.6f}")
        
        print(f"\nMismatches:")
        print(f"  Count: {result.num_mismatches:,} / {result.golden_output.size:,} "
              f"({result.mismatch_percentage:.2f}%)")
        print(f"  Threshold: {self.config.max_outliers_percent:.1f}%")
        
        if result.num_mismatches > 0 and self.config.verbose:
            print(f"\nFirst {min(len(result.mismatch_indices), 5)} mismatches:")
            for idx in result.mismatch_indices[:5]:
                golden_val = result.golden_output[idx]
                hw_val = result.hw_output_float[idx]
                abs_err = abs(golden_val - hw_val)
                rel_err = abs_err / (abs(golden_val) + 1e-10)
                print(f"  [{','.join(map(str, idx))}]: "
                      f"golden={golden_val:.6f}, hw={hw_val:.6f}, "
                      f"err={abs_err:.6f} ({rel_err:.2%})")


class VerificationReport:
    """Generate comprehensive verification report."""
    
    def __init__(self):
        self.results: List[LayerVerificationResult] = []
    
    def add_result(self, result: LayerVerificationResult):
        """Add a layer verification result."""
        self.results.append(result)
    
    def generate_summary(self) -> Dict:
        """Generate summary statistics."""
        total_layers = len(self.results)
        passed_layers = sum(1 for r in self.results if r.passed)
        
        if total_layers == 0:
            return {}
        
        return {
            'total_layers': total_layers,
            'passed_layers': passed_layers,
            'failed_layers': total_layers - passed_layers,
            'pass_rate': passed_layers / total_layers * 100,
            'avg_max_abs_error': np.mean([r.max_abs_error for r in self.results]),
            'avg_mean_abs_error': np.mean([r.mean_abs_error for r in self.results]),
            'avg_mismatch_percentage': np.mean([r.mismatch_percentage for r in self.results]),
            'worst_layer': max(self.results, key=lambda r: r.max_abs_error).layer_name,
            'best_layer': min(self.results, key=lambda r: r.max_abs_error).layer_name
        }
    
    def print_summary(self):
        """Print summary of all results."""
        summary = self.generate_summary()
        
        print("\n" + "="*80)
        print("VERIFICATION SUMMARY")
        print("="*80)
        
        print(f"\nOverall Results:")
        print(f"  Total layers:  {summary['total_layers']}")
        print(f"  Passed:        {summary['passed_layers']} ✓")
        print(f"  Failed:        {summary['failed_layers']} ✗")
        print(f"  Pass rate:     {summary['pass_rate']:.1f}%")
        
        print(f"\nAverage Errors:")
        print(f"  Max abs error:    {summary['avg_max_abs_error']:.6f}")
        print(f"  Mean abs error:   {summary['avg_mean_abs_error']:.6f}")
        print(f"  Mismatch rate:    {summary['avg_mismatch_percentage']:.2f}%")
        
        print(f"\nLayer Analysis:")
        print(f"  Best layer:  {summary['best_layer']}")
        print(f"  Worst layer: {summary['worst_layer']}")
        
        # Per-layer summary table
        print(f"\n{'Layer':<20} {'Status':<10} {'Max Error':<15} {'Mismatches':<15}")
        print("-" * 80)
        
        for result in self.results:
            status = "PASS ✓" if result.passed else "FAIL ✗"
            print(f"{result.layer_name:<20} {status:<10} "
                  f"{result.max_abs_error:<15.6f} "
                  f"{result.mismatch_percentage:<15.2f}%")
    
    def save_to_json(self, filepath: str):
        """Save verification report to JSON."""
        report_data = {
            'summary': self.generate_summary(),
            'layers': [r.to_dict() for r in self.results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\nVerification report saved to {filepath}")


def demo_verification_flow():
    """Demonstrate complete verification flow."""
    
    print("\n" + "="*90)
    print("HARDWARE VERIFICATION WITH GOLDEN MODEL DEMO")
    print("="*90)
    
    # Step 1: Create synthetic data (simulating TF model output)
    print("\n[STEP 1] Creating Synthetic Golden Model Outputs")
    print("-" * 90)
    
    # Simulate a 3-layer CNN output
    golden_outputs = {
        'conv1': np.random.randn(1, 28, 28, 16).astype(np.float32) * 0.5,
        'conv2': np.random.randn(1, 14, 14, 32).astype(np.float32) * 0.3,
        'conv3': np.random.randn(1, 7, 7, 64).astype(np.float32) * 0.2,
    }
    
    print(f"Generated golden outputs for {len(golden_outputs)} layers")
    
    # Step 2: Save golden outputs
    print("\n[STEP 2] Saving Golden Outputs")
    print("-" * 90)
    
    extractor = GoldenModelExtractor()
    extractor.save_golden_outputs(golden_outputs, './demo_output/golden')
    
    # Step 3: Simulate hardware outputs (with some error)
    print("\n[STEP 3] Simulating Hardware Outputs")
    print("-" * 90)
    
    fp_config = FixedPointConfig(total_bits=32, frac_bits=16)
    
    hw_outputs_fixed = {}
    for layer_name, golden in golden_outputs.items():
        # Convert to fixed-point
        golden_fixed = fp_config.float_to_fixed(golden)
        
        # Simulate hardware with small computation error
        # 99.9% of values are correct, 0.1% have small errors
        hw_fixed = golden_fixed.copy()
        
        # Add random bit flips to simulate hardware errors
        error_mask = np.random.random(hw_fixed.shape) < 0.001
        hw_fixed[error_mask] += np.random.randint(-10, 10, size=np.sum(error_mask))
        
        hw_outputs_fixed[layer_name] = hw_fixed
        
        print(f"  Simulated HW output for {layer_name}")
    
    # Step 4: Run verification
    print("\n[STEP 4] Running Verification")
    print("-" * 90)
    
    config = VerificationConfig(
        abs_tolerance=0.01,
        rel_tolerance=0.02,
        max_outliers_percent=1.0,
        verbose=True
    )
    
    comparator = OutputComparator(config)
    report = VerificationReport()
    
    for i, (layer_name, golden) in enumerate(golden_outputs.items()):
        hw_fixed = hw_outputs_fixed[layer_name]
        
        result = comparator.compare_outputs(
            layer_name=layer_name,
            layer_id=i,
            golden_output=golden,
            hw_output_fixed=hw_fixed
        )
        
        comparator.print_result(result)
        report.add_result(result)
    
    # Step 5: Generate report
    print("\n[STEP 5] Generating Verification Report")
    print("-" * 90)
    
    report.print_summary()
    
    os.makedirs('./demo_output', exist_ok=True)
    report.save_to_json('./demo_output/verification_report.json')
    
    print("\n" + "="*90)
    print("DEMO COMPLETE")
    print("="*90)


if __name__ == '__main__':
    demo_verification_flow()
